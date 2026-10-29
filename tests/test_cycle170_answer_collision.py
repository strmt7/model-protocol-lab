"""Cycle 170: answer-collision audit (wave-6 idea 4).

Pure-module, stage (scripted backends in two tokenizer regimes),
CLI-wrapper, reporting, and diff tests, plus tempdir mutation gates on
the exactness-critical lines. The construct (docs/RESEARCH_LOG.md
cycle 170): the lab audits its OWN token-ID boundary detector -- fixture
expected answers embedding marker-like text are emitted verbatim
through core.evaluate and classified by a total order (collision /
ends_at_marker / intact) per boundary CONFIGURATION, in a reserved arm
(the measurement) and a subword control arm.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

from model_protocol_lab import cli, core, diffreports, experiments, reporting
from model_protocol_lab import collision as col
from model_protocol_lab.control_tokens import HOSTILE_MARKER_STRINGS
from model_protocol_lab.core import Format

CANDIDATES = ("alpha", "beta")


def _format(name: str) -> Format:
    """Format."""
    return Format(
        name=name,
        bos="",
        user_prefix=f"[{name}]",
        assistant_prefix="A:",
        turn_end="\n",
    )


class OrdinalBackend:
    """Char-ordinals backend: no special-token parsing at all."""

    fingerprint = "cycle170-ordinal"

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(map(ord, text))

    @staticmethod
    def decode(ids):
        """Decode."""
        return "".join(map(chr, ids))

    @staticmethod
    def generate(input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        return core.Generation(tuple(input_ids[:max_new_tokens]), "limit")


class SpecialBackend:
    """Parses the literal text '<|im_end|>' to the reserved id 7."""

    fingerprint = "cycle170-special"
    MARKER = "<|im_end|>"

    @classmethod
    def encode(cls, text):
        """Encode."""
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
        """Decode."""
        return "".join(cls.MARKER if x == 7 else chr(x) for x in ids)

    @staticmethod
    def generate(input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        return core.Generation(tuple(input_ids[:max_new_tokens]), "limit")


class FixtureTests(unittest.TestCase):
    """Fixture tests."""

    def test_build_yields_frozen_set(self):
        """Test build yields frozen set."""
        specs = col.build_answer_fixtures("<|im_end|>")
        self.assertEqual(len(specs), 21)
        self.assertEqual(len(specs), col.n_fixtures_per_config())
        self.assertEqual(len({s["id"] for s in specs}), 21)
        positions = [s["position"] for s in specs]
        self.assertEqual(positions.count("hostile"), len(HOSTILE_MARKER_STRINGS))
        self.assertEqual(positions.count("clean"), 1)

    def test_position_semantics(self):
        """Test position semantics."""
        by_id = {s["id"]: s for s in col.build_answer_fixtures("<|im_end|>")}
        start = by_id["own/start"]["expected"]
        self.assertTrue(start.startswith("<|im_end|>"))
        self.assertEqual(start.count("<|im_end|>"), 1)
        interior = by_id["own/interior"]["expected"]
        self.assertEqual(interior.count("<|im_end|>"), 1)
        self.assertFalse(interior.startswith("<|im_end|>"))
        self.assertFalse(interior.endswith("<|im_end|>"))
        end = by_id["own/end"]["expected"]
        self.assertTrue(end.endswith("<|im_end|>"))
        self.assertEqual(end.count("<|im_end|>"), 1)
        repeat = by_id["own/repeat"]["expected"]
        self.assertEqual(repeat.count("<|im_end|>"), 2)
        clean = by_id["clean"]["expected"]
        self.assertNotIn("<|im_end|>", clean)
        for hostile in HOSTILE_MARKER_STRINGS:
            self.assertNotIn(hostile, clean)

    def test_hostile_fixtures_embed_each_hostile_string(self):
        """Test hostile fixtures embed each hostile string."""
        specs = col.build_answer_fixtures("<|im_end|>")
        hostile_markers = [s["marker"] for s in specs if s["position"] == "hostile"]
        self.assertEqual(hostile_markers, list(HOSTILE_MARKER_STRINGS))
        for spec in specs:
            if spec["position"] == "hostile":
                self.assertIn(spec["marker"], spec["expected"])

    def test_marker_inside_clean_text_rejected(self):
        """Test marker inside clean text rejected."""
        # The clean template mentions "marker": a config whose marker
        # text is the word "marker" makes the clean fixture unusable,
        # and the protocol refuses it fail-closed.
        with self.assertRaises(ValueError):
            col.build_answer_fixtures("marker")

    def test_malformed_specs_rejected(self):
        """Test malformed specs rejected."""
        good = col.build_answer_fixtures("<|im_end|>")
        for bad in (
            (),
            ("nope",),
            ({**good[0], "id": "  "},),
            ({**good[0], "expected": ""},),
            ({**good[0], "position": "sideways"},),
            ({**good[0], "expected": "no marker here at all"},),  # start, no marker
            (good[0], good[0]),  # duplicate ids
            ({**good[-1], "marker": "<|im_end|>"},),  # clean with a marker
        ):
            with self.assertRaises(ValueError, msg=repr(bad)[:60]):
                col.validate_answer_fixtures(bad, "<|im_end|>")
        for bad_marker in ("", 7, None):
            with self.assertRaises(ValueError, msg=repr(bad_marker)):
                col.build_answer_fixtures(bad_marker)

    def test_fingerprint_stable_and_protocol_shape(self):
        """Test fingerprint stable and protocol shape."""
        fp = col.protocol_fingerprint()
        self.assertEqual(fp, col.protocol_fingerprint())
        self.assertEqual(len(fp), 64)
        protocol = col.protocol_dict()
        self.assertEqual(protocol["fingerprint"], fp)
        self.assertEqual(protocol["n_fixtures"], 21)
        self.assertEqual(protocol["arms"], ["reserved", "subword"])
        self.assertEqual(
            protocol["classes"], ["collision", "ends_at_marker", "intact", "anomalous"]
        )


class ClassifyTests(unittest.TestCase):
    """Classify tests."""

    def test_total_order(self):
        """Test total order."""
        self.assertEqual(col.classify_observation(False, False), "intact")
        self.assertEqual(col.classify_observation(False, True), "intact")
        self.assertEqual(col.classify_observation(True, True), "collision")
        # Marker-at-end is its own cell, never a collision (red-team A R1).
        self.assertEqual(col.classify_observation(True, False), "ends_at_marker")


def _scored(cls, occurrences=1):
    """Scored."""
    return {
        "class": cls,
        "cut": 3,
        "exact_match": cls == "intact",
        "verbatim": True,
        "occurrences": occurrences,
    }


def _fixture(fid, reserved, subword, reserved_state="scored", subword_state="no_pair"):
    """Fixture."""
    return {
        "id": fid,
        "reserved": reserved,
        "subword": subword,
        "arm_state": {"reserved": reserved_state, "subword": subword_state},
    }


class CellTests(unittest.TestCase):
    """Cell tests."""

    def test_aggregation_and_rates(self):
        """Test aggregation and rates."""
        per_fixture = [
            _fixture("a", _scored("collision"), None),
            _fixture("b", _scored("ends_at_marker"), None),
            _fixture("c", _scored("intact"), None),
            _fixture("d", None, None, reserved_state="failed"),
            _fixture("e", None, None, reserved_state="roundtrip"),
        ]
        cell = col.collision_cell(per_fixture, n_fixtures=5, boundary_len=2)
        reserved = cell["reserved"]
        self.assertEqual(reserved["n_evaluable"], 3)
        self.assertEqual(reserved["n_scored"], 3)
        self.assertEqual(reserved["n_collision"], 1)
        self.assertEqual(reserved["n_ends_at_marker"], 1)
        self.assertEqual(reserved["n_intact"], 1)
        self.assertEqual(reserved["n_failed"], 1)
        self.assertEqual(reserved["n_roundtrip_excluded"], 1)
        self.assertAlmostEqual(reserved["collision_rate"], 1 / 3)
        self.assertAlmostEqual(reserved["ends_rate"], 1 / 3)
        self.assertTrue(cell["any_collision"])
        self.assertFalse(cell["uniform_intact"])
        self.assertTrue(cell["evaluable"])
        self.assertEqual(cell["n"], 5)
        self.assertEqual(cell["boundary_ids"], 2)

    def test_anomalous_excluded_from_rates(self):
        """Test anomalous excluded from rates."""
        per_fixture = [
            _fixture("a", _scored("collision"), None),
            _fixture("b", _scored("anomalous"), None),
        ]
        cell = col.collision_cell(per_fixture, n_fixtures=2, boundary_len=1)
        self.assertEqual(cell["reserved"]["n_evaluable"], 2)
        self.assertEqual(cell["reserved"]["n_scored"], 1)
        self.assertEqual(cell["reserved"]["collision_rate"], 1.0)

    def test_no_scored_observations_vacate_rates(self):
        """Test no scored observations vacate rates."""
        per_fixture = [_fixture("a", None, None, reserved_state="failed")]
        cell = col.collision_cell(per_fixture, n_fixtures=1, boundary_len=1)
        self.assertIsNone(cell["reserved"]["collision_rate"])
        self.assertIsNone(cell["reserved"]["ends_rate"])
        self.assertFalse(cell["evaluable"])
        self.assertFalse(cell["uniform_intact"])  # vacuous truth refused

    def test_uniform_intact(self):
        """Test uniform intact."""
        per_fixture = [
            _fixture("a", _scored("intact"), _scored("intact"), subword_state="scored"),
            _fixture("b", _scored("intact"), None),
        ]
        cell = col.collision_cell(per_fixture, n_fixtures=2, boundary_len=1)
        self.assertTrue(cell["uniform_intact"])
        self.assertFalse(cell["any_collision"])

    def test_bad_shape_arguments_rejected(self):
        """Test bad shape arguments rejected."""
        with self.assertRaises(ValueError):
            col.collision_cell([], n_fixtures=0, boundary_len=1)
        with self.assertRaises(ValueError):
            col.collision_cell([], n_fixtures=1, boundary_len=0)


class EmissionTests(unittest.TestCase):
    """Emission tests."""

    def test_reserved_emission_roundtrips(self):
        """Test reserved emission roundtrips."""
        ids = col.reserved_emission(OrdinalBackend.encode, OrdinalBackend.decode, "abc")
        self.assertEqual(ids, (97, 98, 99))

    def test_reserved_emission_fail_closed(self):
        """Test reserved emission fail closed."""

        def lossy_decode(ids):
            """Lossy decode."""
            return "".join(chr(x) for x in ids if x != 98)

        self.assertIsNone(col.reserved_emission(OrdinalBackend.encode, lossy_decode, "abc"))
        self.assertIsNone(col.reserved_emission(lambda t: (), OrdinalBackend.decode, "abc"))

        def boom(text):
            """Boom."""
            raise RuntimeError("encode boom")

        self.assertIsNone(col.reserved_emission(boom, OrdinalBackend.decode, "abc"))

    def test_subword_emission_control_arm(self):
        """Test subword emission control arm."""
        expected = "The marker is <|im_end|> in text."
        ids = col.subword_emission(
            SpecialBackend.encode, SpecialBackend.decode, expected, "<|im_end|>", (7,)
        )
        self.assertIsNotNone(ids)
        self.assertNotIn(7, ids)  # the control arm carries no reserved id
        self.assertEqual(SpecialBackend.decode(ids), expected)
        self.assertEqual(core.first_boundary(ids, (7,)), None)

    def test_subword_emission_no_pair_cases(self):
        """Test subword emission no pair cases."""
        # Char-ordinals: the marker's native encoding is not one token.
        self.assertIsNone(
            col.subword_emission(
                OrdinalBackend.encode,
                OrdinalBackend.decode,
                "x <|im_end|> y",
                "<|im_end|>",
                (60,),
            )
        )
        # Single-character marker: no split point exists.
        self.assertIsNone(
            col.subword_emission(SpecialBackend.encode, SpecialBackend.decode, "a x b", "x", (120,))
        )
        # Hostile marker argument.
        self.assertIsNone(
            col.subword_emission(SpecialBackend.encode, SpecialBackend.decode, "abc", "", (7,))
        )

    def test_subword_emission_replaces_every_occurrence(self):
        """Test subword emission replaces every occurrence."""
        expected = "<|im_end|> and <|im_end|> again"
        ids = col.subword_emission(
            SpecialBackend.encode, SpecialBackend.decode, expected, "<|im_end|>", (7,)
        )
        self.assertIsNotNone(ids)
        self.assertNotIn(7, ids)
        self.assertEqual(SpecialBackend.decode(ids), expected)


class ScoreTests(unittest.TestCase):
    """Score tests."""

    def test_unevaluable_returns_none(self):
        """Test unevaluable returns none."""
        self.assertIsNone(
            col.score_observation(
                status_ok=False,
                boundary_found=False,
                cut=None,
                suffix_nonempty=False,
                exact_match=False,
                verbatim=False,
                occurrences=0,
            )
        )

    def test_anomalous_paths(self):
        """Test anomalous paths."""
        not_verbatim = col.score_observation(
            status_ok=True,
            boundary_found=False,
            cut=None,
            suffix_nonempty=False,
            exact_match=True,
            verbatim=False,
            occurrences=0,
        )
        self.assertEqual(not_verbatim["class"], "anomalous")
        mismatch_no_cut = col.score_observation(
            status_ok=True,
            boundary_found=False,
            cut=None,
            suffix_nonempty=False,
            exact_match=False,
            verbatim=True,
            occurrences=0,
        )
        self.assertEqual(mismatch_no_cut["class"], "anomalous")

    def test_scored_record_fields(self):
        """Test scored record fields."""
        rec = col.score_observation(
            status_ok=True,
            boundary_found=True,
            cut=11,
            suffix_nonempty=True,
            exact_match=False,
            verbatim=True,
            occurrences=2,
        )
        self.assertEqual(rec["class"], "collision")
        self.assertEqual(rec["cut"], 11)
        self.assertEqual(rec["occurrences"], 2)


class PureProfileTests(unittest.TestCase):
    """Pure profile tests."""

    @staticmethod
    def _cell(cls_list):
        """Cell."""
        per_fixture = [_fixture(f"f{i}", _scored(c), None) for i, c in enumerate(cls_list)]
        return col.collision_cell(per_fixture, n_fixtures=len(cls_list), boundary_len=1)

    def test_unavailable_without_measurable_cells(self):
        """Test unavailable without measurable cells."""
        prof = col.answer_collision_profile(("a",), {"a": None}, col.protocol_dict())
        self.assertFalse(prof["available"])
        self.assertEqual(prof["detail"], "answer_collision_unavailable")
        self.assertIsNone(prof["max_collision_rate"])
        self.assertIsNone(prof["worst_candidate"])

    def test_colliding_profile(self):
        """Test colliding profile."""
        cells = {"a": self._cell(["collision", "intact"]), "b": self._cell(["intact"])}
        prof = col.answer_collision_profile(("a", "b"), cells, col.protocol_dict())
        self.assertTrue(prof["available"])
        self.assertTrue(prof["any_collision"])
        self.assertFalse(prof["uniform_intact"])
        self.assertEqual(prof["colliding_candidates"], ["a"])
        self.assertEqual(prof["max_collision_rate"], 0.5)
        self.assertEqual(prof["worst_candidate"], "a")
        self.assertEqual(prof["n_measurable"], 2)

    def test_uniform_intact_profile_is_the_publishable_negative(self):
        """Test uniform intact profile is the publishable negative."""
        cells = {"a": self._cell(["intact", "intact"])}
        prof = col.answer_collision_profile(("a",), cells, col.protocol_dict())
        self.assertTrue(prof["available"])
        self.assertTrue(prof["uniform_intact"])
        self.assertEqual(prof["max_collision_rate"], 0.0)


class StageTests(unittest.TestCase):
    """Stage tests."""

    def test_ordinal_boundary_pipe_counts(self):
        """Test ordinal boundary pipe counts."""
        # Boundary = decode((124,)) = "|": every marker-bearing hostile
        # string containing "|" collides interior; the six without it
        # and the clean fixture stay intact; own/end ends at the marker.
        fmt = _format("alpha")
        stage = experiments.answer_collision_stage(
            OrdinalBackend(), (fmt,), boundary=(124,), seed=1
        )
        profile = experiments.answer_collision_profile(("alpha",), stage)
        self.assertTrue(profile["available"])
        cell = profile["per_candidate"]["alpha"]
        reserved = cell["reserved"]
        self.assertEqual(reserved["n_evaluable"], 21)
        self.assertEqual(reserved["n_collision"], 13)
        self.assertEqual(reserved["n_ends_at_marker"], 1)
        self.assertEqual(reserved["n_intact"], 7)
        self.assertEqual(cell["subword"]["n_no_pair"], 20)
        self.assertEqual(cell["subword"]["n_not_applicable"], 1)
        self.assertEqual(profile["colliding_candidates"], ["alpha"])
        self.assertAlmostEqual(profile["max_collision_rate"], 13 / 21)

    def test_special_boundary_reserved_counts_and_subword_control(self):
        """Test special boundary reserved counts and subword control."""
        # Boundary = (7,) = "<|im_end|>": F1 -- the reserved arm yields
        # interior collisions; F2 -- the subword arm never collides.
        fmt = _format("alpha")
        stage = experiments.answer_collision_stage(SpecialBackend(), (fmt,), boundary=(7,), seed=1)
        profile = experiments.answer_collision_profile(("alpha",), stage)
        cell = profile["per_candidate"]["alpha"]
        reserved = cell["reserved"]
        # Own start/interior/repeat + hostile "<|im_end|>" itself + the
        # forged-turn hostile string that CONTAINS "<|im_end|>".
        self.assertEqual(reserved["n_collision"], 5)
        self.assertEqual(reserved["n_ends_at_marker"], 1)
        self.assertEqual(reserved["n_intact"], 15)
        subword = cell["subword"]
        # Subword arms exist for the four own fixtures and the hostile
        # "<|im_end|>" fixture; F2: none of them collides.
        self.assertEqual(subword["n_evaluable"], 5)
        self.assertEqual(subword["n_intact"], 5)
        self.assertEqual(subword["n_collision"], 0)
        self.assertEqual(subword["n_no_pair"], 15)

    def test_shared_boundary_identical_cells(self):
        """Test shared boundary identical cells."""
        # R5 evidence: candidates sharing a boundary produce identical
        # cells -- the configuration is the unit, not the template.
        fmts = (_format("alpha"), _format("beta"))
        stage = experiments.answer_collision_stage(SpecialBackend(), fmts, boundary=(7,), seed=1)
        self.assertEqual(stage["per_candidate"]["alpha"], stage["per_candidate"]["beta"])

    def test_boundary_map_gives_per_configuration_cells(self):
        """Test boundary map gives per configuration cells."""
        fmts = (_format("alpha"), _format("beta"))
        stage = experiments.answer_collision_stage(
            OrdinalBackend(),
            fmts,
            boundary=(33,),  # "!" -- its own fixtures collide, hostile do not
            boundary_map={"beta": (124,)},
            seed=1,
        )
        alpha = stage["per_candidate"]["alpha"]
        beta = stage["per_candidate"]["beta"]
        self.assertNotEqual(alpha, beta)
        # "!" config: own start/interior/repeat collide, own/end ends;
        # no hostile string and not the clean fixture contains "!".
        self.assertEqual(alpha["reserved"]["n_collision"], 3)
        self.assertEqual(alpha["reserved"]["n_ends_at_marker"], 1)
        self.assertEqual(alpha["reserved"]["n_intact"], 17)
        # "|" config: the 13-collision profile from the ordinal test.
        self.assertEqual(beta["reserved"]["n_collision"], 13)

    def test_encode_regime_blind_configuration_is_intact(self):
        """Test encode regime blind configuration is intact."""

        # R3's other regime: the boundary decodes to "<|im_end|>" but the
        # tokenizer NEVER emits the reserved id from text (spells it in
        # ordinals), so the detector is blind to the marker at text
        # level and every verbatim answer survives intact.
        class BlindBackend:
            """Blind backend."""

            fingerprint = "cycle170-blind"

            @staticmethod
            def encode(text):
                """Encode."""
                return tuple(map(ord, text))

            @staticmethod
            def decode(ids):
                """Decode."""
                return "".join("<|im_end|>" if x == 7 else chr(x) for x in ids)

        fmt = _format("alpha")
        stage = experiments.answer_collision_stage(BlindBackend(), (fmt,), boundary=(7,), seed=1)
        cell = stage["per_candidate"]["alpha"]
        self.assertEqual(cell["reserved"]["n_intact"], 21)
        self.assertTrue(cell["uniform_intact"])

    def test_stage_guards(self):
        """Test stage guards."""
        fmt = _format("alpha")
        with self.assertRaises(ValueError):
            experiments.answer_collision_stage(None, (fmt,), boundary=(7,))
        with self.assertRaises(ValueError):
            experiments.answer_collision_stage(OrdinalBackend(), (), boundary=(7,))
        with self.assertRaises(ValueError):
            experiments.answer_collision_stage(OrdinalBackend(), (fmt,), boundary=())
        with self.assertRaises(ValueError):
            experiments.answer_collision_stage(OrdinalBackend(), (fmt,), boundary=(7,), seed=-1)

    def test_undecodable_boundary_degrades_cell(self):
        """Test undecodable boundary degrades cell."""

        class BadDecodeBackend(OrdinalBackend):
            """Bad decode backend."""

            @staticmethod
            def decode(ids):
                """Decode."""
                if tuple(ids) == (7,):
                    raise RuntimeError("decode boom")
                return "".join(map(chr, ids))

        fmt = _format("alpha")
        stage = experiments.answer_collision_stage(
            BadDecodeBackend(), (fmt,), boundary=(7,), seed=1
        )
        self.assertIsNone(stage["per_candidate"]["alpha"])
        profile = experiments.answer_collision_profile(("alpha",), stage)
        self.assertFalse(profile["available"])

    def test_roundtrip_exclusion_axis(self):
        """Test roundtrip exclusion axis."""

        # R7: a lossy decode excludes fixtures from the reserved arm's
        # denominators instead of scoring them against the detector.
        class LossyBackend(OrdinalBackend):
            """Lossy backend."""

            @staticmethod
            def decode(ids):
                """Decode."""
                return "".join(chr(x) for x in ids if x != 109)  # drop "m"

        fmt = _format("alpha")
        stage = experiments.answer_collision_stage(LossyBackend(), (fmt,), boundary=(124,), seed=1)
        cell = stage["per_candidate"]["alpha"]
        # Only the two templates without an "m" (own/end, own/repeat)
        # survive the lossy decode; the other 19 leave the denominator.
        self.assertEqual(cell["reserved"]["n_roundtrip_excluded"], 19)
        self.assertEqual(cell["reserved"]["n_evaluable"], 2)


class ExperimentsProfileGuardTests(unittest.TestCase):
    """Experiments profile guard tests."""

    def test_invalid_candidates(self):
        """Test invalid candidates."""
        self.assertEqual(
            experiments.answer_collision_profile((), {})["detail"], "invalid_candidates"
        )
        self.assertEqual(
            experiments.answer_collision_profile(("a", "a"), {})["detail"],
            "invalid_candidates",
        )
        self.assertEqual(
            experiments.answer_collision_profile((" ",), {})["detail"], "invalid_candidates"
        )

    def test_invalid_stage(self):
        """Test invalid stage."""
        self.assertEqual(
            experiments.answer_collision_profile(("a",), "junk")["detail"], "stage_invalid"
        )
        self.assertEqual(
            experiments.answer_collision_profile(("a",), {"protocol": {}})["detail"],
            "stage_invalid",
        )

    def test_tampered_cell_rejected(self):
        """Test tampered cell rejected."""
        fmt = _format("alpha")
        stage = experiments.answer_collision_stage(
            OrdinalBackend(), (fmt,), boundary=(124,), seed=1
        )
        stage["per_candidate"]["alpha"]["reserved"]["collision_rate"] = 0.01
        self.assertEqual(
            experiments.answer_collision_profile(("alpha",), stage)["detail"],
            "stage_invalid",
        )


class CliTests(unittest.TestCase):
    """Cli tests."""

    def test_probe_records_payload(self):
        """Test probe records payload."""
        result = {}
        cli._run_answer_collision_probe(
            result,
            (_format("alpha"),),
            backend=OrdinalBackend(),
            boundary=(124,),
            boundary_map=None,
            seed=1,
        )
        payload = result["answer_collision"]
        self.assertTrue(payload["available"])
        self.assertEqual(payload["colliding_candidates"], ["alpha"])

    def test_probe_never_raises(self):
        """Test probe never raises."""
        result = {}
        cli._run_answer_collision_probe(
            result,
            ("not-a-format",),
            backend=None,
            boundary=None,
            boundary_map=None,
            seed=0,
        )
        self.assertEqual(
            result["answer_collision"],
            {"available": False, "detail": "answer_collision_unavailable"},
        )


class ReportingTests(unittest.TestCase):
    """Reporting tests."""

    @staticmethod
    def _payload(backend=None, boundary=(124,)):
        """Payload."""
        fmt = _format("alpha")
        stage = experiments.answer_collision_stage(
            backend or OrdinalBackend(), (fmt,), boundary=boundary, seed=1
        )
        return experiments.answer_collision_profile(("alpha",), stage)

    def test_section_renders_counts(self):
        """Test section renders counts."""
        section = reporting._answer_collision_section({"answer_collision": self._payload()})
        self.assertIn('id="answer-collision"', section)
        self.assertIn("alpha", section)
        self.assertIn("13", section)  # reserved collisions for boundary "|"
        self.assertIn("provably truncated", section)

    def test_section_absent_when_unavailable(self):
        """Test section absent when unavailable."""
        self.assertEqual(
            reporting._answer_collision_section({"answer_collision": {"available": False}}),
            "",
        )
        self.assertEqual(reporting._answer_collision_section({}), "")

    def test_marker_text_never_rendered(self):
        """Test marker text never rendered."""
        # R11: the marker text is hostile input to the report path; the
        # section shows counts and rates only.
        section = reporting._answer_collision_section(
            {"answer_collision": self._payload(SpecialBackend(), (7,))}
        )
        self.assertNotIn("<|im_end|>", section)

    def test_hostile_candidate_name_escaped(self):
        """Test hostile candidate name escaped."""
        fmt = _format("a<b>bold</b>")
        stage = experiments.answer_collision_stage(
            OrdinalBackend(), (fmt,), boundary=(124,), seed=1
        )
        payload = experiments.answer_collision_profile((fmt.name,), stage)
        section = reporting._answer_collision_section({"answer_collision": payload})
        self.assertIn("a&lt;b&gt;", section)
        self.assertNotIn("<b>bold</b>", section)


def _hand_payload(colliding: bool):
    """Hand payload."""
    cls_list = ["collision", "intact"] if colliding else ["intact", "intact"]
    per_fixture = [_fixture(f"f{i}", _scored(c), None) for i, c in enumerate(cls_list)]
    cell = col.collision_cell(per_fixture, n_fixtures=2, boundary_len=1)
    cell["protocol_fingerprint"] = col.protocol_fingerprint()
    profile = col.answer_collision_profile(("cand",), {"cand": cell}, col.protocol_dict())
    profile["n_fixtures"] = 2
    return profile


class DiffTests(unittest.TestCase):
    """Diff tests."""

    def test_identical_payloads_unchanged(self):
        """Test identical payloads unchanged."""
        result = diffreports.diff_answer_collision(_hand_payload(True), _hand_payload(True))
        self.assertEqual(result["available"], "both")
        self.assertFalse(result["changed"])
        self.assertEqual(result["rate_drift"], [])

    def test_rate_drift_and_colliding_added(self):
        """Test rate drift and colliding added."""
        result = diffreports.diff_answer_collision(_hand_payload(False), _hand_payload(True))
        self.assertTrue(result["changed"])
        self.assertEqual(len(result["rate_drift"]), 1)
        self.assertEqual(result["rate_drift"][0]["old"], 0.0)
        self.assertEqual(result["rate_drift"][0]["new"], 0.5)
        self.assertEqual(result["colliding_added"], ["cand"])
        self.assertTrue(result["uniform_changed"])

    def test_protocol_drift_detected(self):
        """Test protocol drift detected."""
        new = _hand_payload(False)
        proto = dict(col.protocol_dict())
        proto["fingerprint"] = "0" * 64
        new["protocol"] = proto
        result = diffreports.diff_answer_collision(_hand_payload(False), new)
        self.assertTrue(result["protocol_drift"])
        self.assertTrue(result["changed"])

    def test_stale_rate_rejected(self):
        """Test stale rate rejected."""
        bad = _hand_payload(True)
        bad["per_candidate"]["cand"]["reserved"]["collision_rate"] = 0.99
        self.assertFalse(diffreports._valid_answer_collision(bad))

    def test_bool_counts_rejected(self):
        """Test bool counts rejected."""
        bad = _hand_payload(True)
        bad["per_candidate"]["cand"]["reserved"]["n_collision"] = True
        self.assertFalse(diffreports._valid_answer_collision(bad))

    def test_inconsistent_profile_fields_rejected(self):
        """Test inconsistent profile fields rejected."""
        bad = _hand_payload(True)
        bad["colliding_candidates"] = []
        self.assertFalse(diffreports._valid_answer_collision(bad))
        bad2 = _hand_payload(False)
        bad2["uniform_intact"] = False
        self.assertFalse(diffreports._valid_answer_collision(bad2))

    def test_availability_sides(self):
        """Test availability sides."""
        self.assertEqual(
            diffreports.diff_answer_collision(_hand_payload(True), None)["available"], "old"
        )
        self.assertEqual(
            diffreports.diff_answer_collision(None, _hand_payload(True))["available"], "new"
        )
        self.assertEqual(diffreports.diff_answer_collision(None, None)["available"], "neither")

    def test_summary_lines(self):
        """Test summary lines."""
        lines = diffreports._render_answer_collision_summary(
            diffreports.diff_answer_collision(_hand_payload(False), _hand_payload(True))
        )
        self.assertTrue(any("rate drift" in line for line in lines))
        self.assertTrue(any("newly colliding" in line for line in lines))
        missing = diffreports._render_answer_collision_summary(
            diffreports.diff_answer_collision(_hand_payload(True), None)
        )
        self.assertIn("did not record", missing[0])
        self.assertEqual(
            diffreports._render_answer_collision_summary(
                diffreports.diff_answer_collision(None, None)
            ),
            [],
        )


class MutationGateTests(unittest.TestCase):
    """Tempdir mutation gates on the exactness-critical lines."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        """Run gate."""
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        import tempfile

        with tempfile.TemporaryDirectory(prefix="c170_mut_", dir="/tmp") as tmpdir:
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
    def _classify_scenario(mod):
        """Classify scenario."""
        assert mod.classify_observation(True, True) == "collision"
        assert mod.classify_observation(True, False) == "ends_at_marker"

    def test_mutant_classify_order_swapped_caught(self):
        """Test mutant classify order swapped caught."""
        self._run_gate(
            "collision",
            "classify_swap",
            '    if suffix_nonempty:\n        return "collision"',
            '    if suffix_nonempty:\n        return "ends_at_marker"  # MUTANT',
            self._classify_scenario,
        )

    @staticmethod
    def _rate_scenario(mod):
        """Rate scenario."""
        per_fixture = [
            _fixture("a", _scored("collision"), None),
            _fixture("b", _scored("intact"), None),
        ]
        cell = mod.collision_cell(per_fixture, n_fixtures=2, boundary_len=1)
        assert cell["reserved"]["collision_rate"] == 0.5

    def test_mutant_rate_denominator_caught(self):
        """Test mutant rate denominator caught."""
        self._run_gate(
            "collision",
            "rate_denominator",
            '"collision_rate": (counts["collision"] / n_scored) if n_scored else None,',
            '"collision_rate": (counts["collision"] / (n_scored + 1)) '
            "if n_scored else None,  # MUTANT",
            self._rate_scenario,
        )

    @staticmethod
    def _diff_scenario(mod):
        """Diff scenario."""

        def payload(n_coll):
            """Payload."""
            per_fixture = [
                _fixture(f"f{i}", _scored("collision" if i < n_coll else "intact"), None)
                for i in range(4)
            ]
            cell = col.collision_cell(per_fixture, n_fixtures=4, boundary_len=1)
            cell["protocol_fingerprint"] = col.protocol_fingerprint()
            profile = col.answer_collision_profile(("cand",), {"cand": cell}, col.protocol_dict())
            profile["n_fixtures"] = 4
            return profile

        result = mod.diff_answer_collision(payload(1), payload(2))
        assert result["changed"] is True
        assert len(result["rate_drift"]) == 1

    def test_mutant_diff_rate_disjunct_dropped_caught(self):
        """Test mutant diff rate disjunct dropped caught."""
        self._run_gate(
            "diffreports",
            "diff_rate_disjunct",
            "        or answer_collision_rate_drift\n",
            "        or []  # MUTANT\n",
            self._diff_scenario,
        )


if __name__ == "__main__":
    unittest.main()
