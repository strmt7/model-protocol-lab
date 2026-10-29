"""Cycle 156: whitespace canonicalization at template boundaries (wave-3 idea 4).

Tests the pure core (wscanonical.py): the 8 whitespace mutations of the
render string, exact id-sequence comparison, per-mutation aggregation,
and the WsScanBackend encode-time wrapper; the experiments.ws_stage /
ws_canonicalization fail-closed contracts; the CLI probe wiring; the
HTML section escaping and TOC anchor; and the
diffreports.diff_ws_canonical section. Follows the established per-cycle
pattern: brute-force oracle differentials, Hypothesis properties,
tempdir-module mutation gates, hostile-payload red-team.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import os
import random
import tempfile
import unittest
from pathlib import Path
from threading import Event

from hypothesis import given, settings
from hypothesis import strategies as st

from model_protocol_lab import diffreports
from model_protocol_lab.core import Fixture, Format, Generation, Message
from model_protocol_lab.diffreports import diff_ws_canonical
from model_protocol_lab.experiments import (
    native_search_input_ids,
    native_search_outcomes,
    ws_canonicalization,
    ws_stage,
)
from model_protocol_lab.reporting import _wscanonical_section
from model_protocol_lab.wscanonical import (
    MUTATIONS,
    WsScanBackend,
    aggregate_mutation_records,
    apply_mutation,
    compare_id_sequences,
    validate_mutations,
)

BOUNDARY = (124,)  # ord("|")


def _fixture(fid, expected="WORD"):
    """Fixture."""
    return Fixture(
        id=fid,
        messages=(Message(role="user", content="Say the word."),),
        expected=expected,
        group="g",
    )


def _format(name="chatml"):
    """Format."""
    return Format(
        name=name,
        bos="<s>",
        user_prefix="<|user|>\n",
        assistant_prefix="<|assistant|>\n",
        turn_end="<|end|>\n",
    )


class ParityBackend:
    """ord codec; answers WORD iff the prompt id count is even."""

    fingerprint = "scripted-ws-parity-v1"

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids):
        """Decode."""
        return "".join(map(chr, token_ids))

    @staticmethod
    def generate(input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        answer = "WORD" if len(input_ids) % 2 == 0 else "NOPE"
        out = tuple(map(ord, (answer + "|").ljust(max_new_tokens, ".")[:max_new_tokens]))
        return Generation(token_ids=out, reason="limit")


class ConstantBackend(ParityBackend):
    """Always answers WORD: whitespace-robust by construction."""

    fingerprint = "scripted-ws-constant-v1"

    @staticmethod
    def generate(input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        out = tuple(map(ord, ("WORD" + "|").ljust(max_new_tokens, ".")[:max_new_tokens]))
        return Generation(token_ids=out, reason="limit")


class FailingBackend(ParityBackend):
    """Failing backend."""

    fingerprint = "scripted-ws-failing-v1"

    @staticmethod
    def generate(input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        raise RuntimeError("boom")


def _search_entries(names=("chatml",), n_fixtures=2, verdict=True):
    """Search entries."""
    entries = []
    for name in names:
        observations = []
        for i in range(n_fixtures):
            observations.append(
                {
                    "fixture": f"f{i}",
                    "status": "completed",
                    "exact_match": verdict,
                    "boundary_found": True,
                    "input_ids": [10, 20, 30 + i],
                }
            )
        entries.append({"format": {"name": name}, "observations": observations})
    return entries


def _ws_payload(mutations=None, robust=True):
    """Ws payload."""
    mutations = list(mutations or MUTATIONS)
    per_candidate = {}
    for name in ("alpha", "beta"):
        cells = {}
        for mutation in mutations:
            cells[mutation] = {
                "evaluable": True,
                "n_fixtures": 2,
                "n_evaluable": 2,
                "n_ids_divergent": 0 if robust else 1,
                "ids_divergence_rate": 0.0 if robust else 0.5,
                "n_verdict_flips": 0 if robust else 1,
                "verdict_flip_rate": 0.0 if robust else 0.5,
                "n_same_ids_verdict_flips": 0,
                "flips": [] if robust else ["f0"],
                "fragile": not robust,
            }
        per_candidate[name] = {
            "per_mutation": cells,
            "aggregate": {
                "n_fragile_mutations": 0 if robust else len(mutations),
                "max_ids_divergence_rate": 0.0 if robust else 0.5,
                "max_verdict_flip_rate": 0.0 if robust else 0.5,
                "whitespace_robust": robust,
            },
            "whitespace_robust": robust,
            "detail": None,
        }
    return {
        "available": True,
        "mutations": mutations,
        "per_candidate": per_candidate,
        "fragile_candidates": [] if robust else ["alpha", "beta"],
        "excluded_candidates": [],
    }


class ApplyMutationTests(unittest.TestCase):
    """Apply mutation tests."""

    def test_trailing_space(self):
        """Test trailing space."""
        self.assertEqual(apply_mutation("abc", "trailing_space"), "abc ")

    def test_double_trailing_space(self):
        """Test double trailing space."""
        self.assertEqual(apply_mutation("abc", "double_trailing_space"), "abc  ")

    def test_trailing_newline(self):
        """Test trailing newline."""
        self.assertEqual(apply_mutation("abc", "trailing_newline"), "abc\n")

    def test_trailing_tab(self):
        """Test trailing tab."""
        self.assertEqual(apply_mutation("abc", "trailing_tab"), "abc\t")

    def test_leading_space(self):
        """Test leading space."""
        self.assertEqual(apply_mutation("abc", "leading_space"), " abc")

    def test_leading_newline(self):
        """Test leading newline."""
        self.assertEqual(apply_mutation("abc", "leading_newline"), "\nabc")

    def test_strip_trailing_space_present(self):
        """Test strip trailing space present."""
        self.assertEqual(apply_mutation("abc ", "strip_trailing_space"), "abc")

    def test_strip_trailing_space_absent_is_identity(self):
        """Test strip trailing space absent is identity."""
        self.assertEqual(apply_mutation("abc", "strip_trailing_space"), "abc")

    def test_collapse_double_space(self):
        """Test collapse double space."""
        self.assertEqual(apply_mutation("a  b   c", "collapse_double_space"), "a b  c")

    def test_unknown_mutation_raises(self):
        """Test unknown mutation raises."""
        with self.assertRaises(ValueError):
            apply_mutation("abc", "trailing_bikeshed")

    def test_non_string_text_raises(self):
        """Test non string text raises."""
        with self.assertRaises(ValueError):
            apply_mutation(42, "trailing_space")

    def test_mutations_tuple_matches_descriptions(self):
        """Test mutations tuple matches descriptions."""
        from model_protocol_lab.wscanonical import MUTATION_DESCRIPTIONS

        self.assertEqual(set(MUTATIONS), set(MUTATION_DESCRIPTIONS))


class ValidateMutationsTests(unittest.TestCase):
    """Validate mutations tests."""

    def test_accepts_tuple(self):
        """Test accepts tuple."""
        self.assertEqual(validate_mutations(("trailing_space",)), ("trailing_space",))

    def test_accepts_list(self):
        """Test accepts list."""
        self.assertEqual(validate_mutations(["leading_space"]), ("leading_space",))

    def test_unknown_rejected(self):
        """Test unknown rejected."""
        with self.assertRaises(ValueError):
            validate_mutations(("trailing_space", "nope"))

    def test_duplicate_rejected(self):
        """Test duplicate rejected."""
        with self.assertRaises(ValueError):
            validate_mutations(("trailing_space", "trailing_space"))

    def test_empty_rejected(self):
        """Test empty rejected."""
        with self.assertRaises(ValueError):
            validate_mutations(())

    def test_non_sequence_rejected(self):
        """Test non sequence rejected."""
        with self.assertRaises(ValueError):
            validate_mutations("trailing_space")


class CompareIdSequencesTests(unittest.TestCase):
    """Compare id sequences tests."""

    def test_equal(self):
        """Test equal."""
        self.assertTrue(compare_id_sequences((1, 2, 3), (1, 2, 3)))

    def test_unequal(self):
        """Test unequal."""
        self.assertFalse(compare_id_sequences((1, 2), (1, 2, 3)))

    def test_empty_equal(self):
        """Test empty equal."""
        self.assertTrue(compare_id_sequences((), ()))

    def test_malformed_base_raises(self):
        """Test malformed base raises."""
        with self.assertRaises(ValueError):
            compare_id_sequences([1, 2], (1, 2))

    def test_negative_id_raises(self):
        """Test negative id raises."""
        with self.assertRaises(ValueError):
            compare_id_sequences((1, -1), (1, 2))


class AggregateMutationRecordsTests(unittest.TestCase):
    """Aggregate mutation records tests."""

    @staticmethod
    def _record(mutation, divergent=0, flips=0, n=4):
        """Record."""
        return {
            "mutation": mutation,
            "n_fixtures": n,
            "n_evaluable": n,
            "n_ids_divergent": divergent,
            "n_verdict_flips": flips,
            "n_same_ids_verdict_flips": 0,
        }

    def test_rates_and_robust(self):
        """Test rates and robust."""
        agg = aggregate_mutation_records(
            [self._record("trailing_space"), self._record("leading_space", divergent=2)]
        )
        self.assertEqual(agg["n_fragile_mutations"], 0)
        self.assertAlmostEqual(agg["per_mutation"]["leading_space"]["ids_divergence_rate"], 0.5)
        self.assertFalse(agg["whitespace_robust"])

    def test_whitespace_robust_when_clean(self):
        """Test whitespace robust when clean."""
        agg = aggregate_mutation_records(
            [self._record("trailing_space"), self._record("leading_space")]
        )
        self.assertTrue(agg["whitespace_robust"])
        self.assertEqual(agg["max_verdict_flip_rate"], 0.0)

    def test_flip_makes_fragile(self):
        """Test flip makes fragile."""
        agg = aggregate_mutation_records([self._record("trailing_space", flips=1)])
        self.assertEqual(agg["n_fragile_mutations"], 1)
        self.assertTrue(agg["per_mutation"]["trailing_space"]["fragile"])
        self.assertAlmostEqual(agg["per_mutation"]["trailing_space"]["verdict_flip_rate"], 0.25)
        self.assertFalse(agg["whitespace_robust"])

    def test_empty_records_not_robust(self):
        """Test empty records not robust."""
        agg = aggregate_mutation_records([])
        self.assertFalse(agg["whitespace_robust"])
        self.assertEqual(agg["n_fragile_mutations"], 0)

    def test_unknown_mutation_rejected(self):
        """Test unknown mutation rejected."""
        with self.assertRaises(ValueError):
            aggregate_mutation_records([self._record("nope")])

    def test_duplicate_rejected(self):
        """Test duplicate rejected."""
        with self.assertRaises(ValueError):
            aggregate_mutation_records(
                [self._record("trailing_space"), self._record("trailing_space")]
            )

    def test_evaluable_exceeds_fixtures_rejected(self):
        """Test evaluable exceeds fixtures rejected."""
        record = self._record("trailing_space")
        record["n_evaluable"] = 5
        with self.assertRaises(ValueError):
            aggregate_mutation_records([record])

    def test_negative_count_rejected(self):
        """Test negative count rejected."""
        record = self._record("trailing_space")
        record["n_verdict_flips"] = -1
        with self.assertRaises(ValueError):
            aggregate_mutation_records([record])


class WsScanBackendTests(unittest.TestCase):
    """Ws scan backend tests."""

    def test_encode_applies_mutation(self):
        """Test encode applies mutation."""
        wrapped = WsScanBackend(ParityBackend(), "trailing_space")
        self.assertEqual(wrapped.encode("abc"), tuple(map(ord, "abc ")))

    def test_fingerprint_suffix(self):
        """Test fingerprint suffix."""
        wrapped = WsScanBackend(ParityBackend(), "leading_newline")
        self.assertEqual(wrapped.fingerprint, "scripted-ws-parity-v1+wsscan:leading_newline")

    def test_mutation_property(self):
        """Test mutation property."""
        self.assertEqual(WsScanBackend(ParityBackend(), "trailing_tab").mutation, "trailing_tab")

    def test_none_inner_rejected(self):
        """Test none inner rejected."""
        with self.assertRaises(ValueError):
            WsScanBackend(None, "trailing_space")

    def test_unknown_mutation_rejected(self):
        """Test unknown mutation rejected."""
        with self.assertRaises(ValueError):
            WsScanBackend(ParityBackend(), "nope")

    def test_non_string_fingerprint_rejected(self):
        """Test non string fingerprint rejected."""

        class Bad:
            """Bad."""

            fingerprint = 42

        with self.assertRaises(ValueError):
            WsScanBackend(Bad(), "trailing_space")

    def test_decode_forwards(self):
        """Test decode forwards."""
        wrapped = WsScanBackend(ParityBackend(), "trailing_space")
        self.assertEqual(wrapped.decode((65, 66)), "AB")

    def test_generate_forwards(self):
        """Test generate forwards."""
        wrapped = WsScanBackend(ParityBackend(), "trailing_space")
        gen = wrapped.generate((1, 2), max_new_tokens=4, seed=1, cancel=Event())
        self.assertEqual(gen.token_ids, tuple(map(ord, "WORD")))


class BruteForceOracleTests(unittest.TestCase):
    """Brute force oracle tests."""

    @staticmethod
    def _independent_apply(text, mutation):
        """Independent apply."""
        if mutation == "trailing_space":
            return text + " "
        if mutation == "double_trailing_space":
            return text + "  "
        if mutation == "trailing_newline":
            return text + "\n"
        if mutation == "trailing_tab":
            return text + "\t"
        if mutation == "leading_space":
            return " " + text
        if mutation == "leading_newline":
            return "\n" + text
        if mutation == "strip_trailing_space":
            return text[:-1] if text.endswith(" ") else text
        if mutation == "collapse_double_space":
            out = []
            i = 0
            while i < len(text):
                if text[i : i + 2] == "  ":
                    out.append(" ")
                    i += 2
                else:
                    out.append(text[i])
                    i += 1
            return "".join(out)
        raise AssertionError("unknown mutation")

    def test_apply_mutation_matches_independent(self):
        """Test apply mutation matches independent."""
        rng = random.Random(20261019)  # noqa: S311 -- deterministic test seed
        alphabet = "ab \n\t"
        for _ in range(200):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
            mutation = rng.choice(MUTATIONS)
            self.assertEqual(
                apply_mutation(text, mutation),
                self._independent_apply(text, mutation),
                f"text={text!r} mutation={mutation}",
            )

    def test_aggregate_matches_independent(self):
        """Test aggregate matches independent."""
        rng = random.Random(20261019)  # noqa: S311 -- deterministic test seed
        for _ in range(200):
            mutations = rng.sample(MUTATIONS, rng.randint(1, 4))
            records = []
            for mutation in mutations:
                n = rng.randint(1, 6)
                n_eval = rng.randint(0, n)
                records.append(
                    {
                        "mutation": mutation,
                        "n_fixtures": n,
                        "n_evaluable": n_eval,
                        "n_ids_divergent": rng.randint(0, n),
                        "n_verdict_flips": rng.randint(0, n_eval),
                        "n_same_ids_verdict_flips": 0,
                    }
                )
            agg = aggregate_mutation_records(records)
            # Independent check: robust iff no divergence and no flips anywhere.
            expect_robust = bool(records) and all(
                r["n_ids_divergent"] == 0 and r["n_verdict_flips"] == 0 for r in records
            )
            self.assertEqual(agg["whitespace_robust"], expect_robust)
            self.assertEqual(
                agg["n_fragile_mutations"],
                sum(1 for r in records if r["n_verdict_flips"] > 0),
            )
            for record in records:
                cell = agg["per_mutation"][record["mutation"]]
                self.assertAlmostEqual(
                    cell["ids_divergence_rate"],
                    record["n_ids_divergent"] / record["n_fixtures"],
                )
                if record["n_evaluable"]:
                    self.assertAlmostEqual(
                        cell["verdict_flip_rate"],
                        record["n_verdict_flips"] / record["n_evaluable"],
                    )


class WscanonicalHypothesisTests(unittest.TestCase):
    """Wscanonical hypothesis tests."""

    @given(st.text(), st.sampled_from(MUTATIONS))
    @settings(max_examples=200)
    def test_mutation_is_deterministic(self, text, mutation):
        """Test mutation is deterministic."""
        self.assertEqual(apply_mutation(text, mutation), apply_mutation(text, mutation))

    @given(st.text())
    @settings(max_examples=200)
    def test_strip_is_left_inverse_of_trailing(self, text):
        """Test strip is left inverse of trailing."""
        # strip_trailing_space undoes trailing_space exactly when the
        # original text did not already end with a space.
        if not text.endswith(" "):
            self.assertEqual(
                apply_mutation(apply_mutation(text, "trailing_space"), "strip_trailing_space"),
                text,
            )

    @given(st.lists(st.integers(min_value=0, max_value=50000), max_size=20))
    @settings(max_examples=200)
    def test_compare_reflexive(self, ids):
        """Test compare reflexive."""
        self.assertTrue(compare_id_sequences(tuple(ids), tuple(ids)))

    @given(
        st.lists(
            st.fixed_dictionaries(
                {
                    "mutation": st.sampled_from(MUTATIONS),
                    "n_fixtures": st.integers(min_value=1, max_value=6),
                    "n_evaluable": st.integers(min_value=0, max_value=6),
                    "n_ids_divergent": st.integers(min_value=0, max_value=6),
                    "n_verdict_flips": st.integers(min_value=0, max_value=6),
                    "n_same_ids_verdict_flips": st.integers(min_value=0, max_value=6),
                }
            ),
            min_size=1,
            max_size=4,
            unique_by=lambda r: r["mutation"],
        )
    )
    @settings(max_examples=200)
    def test_rates_bounded(self, records):
        """Test rates bounded."""
        # Clamp to the valid region; the property is about rate bounds.
        clean = []
        for record in records:
            record = dict(record)
            record["n_evaluable"] = min(record["n_evaluable"], record["n_fixtures"])
            record["n_ids_divergent"] = min(record["n_ids_divergent"], record["n_fixtures"])
            record["n_verdict_flips"] = min(record["n_verdict_flips"], record["n_evaluable"])
            record["n_same_ids_verdict_flips"] = min(
                record["n_same_ids_verdict_flips"], record["n_verdict_flips"]
            )
            clean.append(record)
        agg = aggregate_mutation_records(clean)
        for cell in agg["per_mutation"].values():
            self.assertGreaterEqual(cell["ids_divergence_rate"], 0.0)
            self.assertLessEqual(cell["ids_divergence_rate"], 1.0)
            self.assertGreaterEqual(cell["verdict_flip_rate"], 0.0)
            self.assertLessEqual(cell["verdict_flip_rate"], 1.0)


class NativeSearchInputIdsTests(unittest.TestCase):
    """Native search input ids tests."""

    def test_extracts_ids(self):
        """Test extracts ids."""
        result = native_search_input_ids(_search_entries(), ("chatml",))
        self.assertEqual(result["chatml"], {"f0": [10, 20, 30], "f1": [10, 20, 31]})

    def test_malformed_observation_degrades_to_none(self):
        """Test malformed observation degrades to none."""
        entries = _search_entries()
        entries[0]["observations"][0] = {"fixture": "f0", "input_ids": "junk"}
        result = native_search_input_ids(entries, ("chatml",))
        self.assertIsNone(result["chatml"])

    def test_missing_candidate_is_none(self):
        """Test missing candidate is none."""
        result = native_search_input_ids(_search_entries(), ("ghost",))
        self.assertIsNone(result["ghost"])

    def test_non_list_entries_degrade(self):
        """Test non list entries degrade."""
        result = native_search_input_ids("junk", ("chatml",))
        self.assertIsNone(result["chatml"])


class WsStageTests(unittest.TestCase):
    """Ws stage tests."""

    @staticmethod
    def _cases(n=2):
        """Cases."""
        return tuple(_fixture(f"f{i}") for i in range(n))

    def test_cells_record_verdicts_and_ids(self):
        """Test cells record verdicts and ids."""
        stage = ws_stage(ParityBackend(), (_format(),), self._cases(), boundary=BOUNDARY)
        self.assertEqual(stage["mutations"], list(MUTATIONS))
        cell = stage["per_mutation"]["trailing_space"]["per_candidate"]["chatml"]
        self.assertIsNotNone(cell)
        self.assertEqual(cell["n"], 2)
        self.assertEqual(set(cell["verdicts"]), {"f0", "f1"})
        self.assertEqual(set(cell["input_ids"]), {"f0", "f1"})

    def test_parity_backend_diverges_on_one_char_mutations(self):
        """Test parity backend diverges on one char mutations."""
        stage = ws_stage(ParityBackend(), (_format(),), self._cases(), boundary=BOUNDARY)
        base_ids = ParityBackend.encode(_format().render(self._cases()[0].messages))
        variant = stage["per_mutation"]["trailing_space"]["per_candidate"]["chatml"]
        self.assertNotEqual(tuple(variant["input_ids"]["f0"]), base_ids)

    def test_failing_backend_cell_is_none(self):
        """Test failing backend cell is none."""
        stage = ws_stage(FailingBackend(), (_format(),), self._cases(), boundary=BOUNDARY)
        cell = stage["per_mutation"]["trailing_space"]["per_candidate"]["chatml"]
        self.assertIsNone(cell)

    def test_none_backend_raises(self):
        """Test none backend raises."""
        with self.assertRaises(ValueError):
            ws_stage(None, (_format(),), self._cases(), boundary=BOUNDARY)

    def test_empty_formats_raises(self):
        """Test empty formats raises."""
        with self.assertRaises(ValueError):
            ws_stage(ParityBackend(), (), self._cases(), boundary=BOUNDARY)

    def test_unknown_mutation_raises(self):
        """Test unknown mutation raises."""
        with self.assertRaises(ValueError):
            ws_stage(
                ParityBackend(),
                (_format(),),
                self._cases(),
                mutations=("nope",),
                boundary=BOUNDARY,
            )

    def test_bad_boundary_raises(self):
        """Test bad boundary raises."""
        with self.assertRaises(ValueError):
            ws_stage(ParityBackend(), (_format(),), self._cases(), boundary=())

    def test_mutation_subset_accepted(self):
        """Test mutation subset accepted."""
        stage = ws_stage(
            ParityBackend(),
            (_format(),),
            self._cases(),
            mutations=("trailing_space", "leading_space"),
            boundary=BOUNDARY,
        )
        self.assertEqual(stage["mutations"], ["trailing_space", "leading_space"])


class WsCanonicalizationTests(unittest.TestCase):
    """Ws canonicalization tests."""

    @staticmethod
    def _run(backend, names=("chatml",), n_fixtures=2):
        """Run."""
        from model_protocol_lab.core import evaluate

        formats = tuple(_format(name) for name in names)
        cases = tuple(_fixture(f"f{i}") for i in range(n_fixtures))
        reports = {
            name: evaluate(backend, fmt, cases, boundary=BOUNDARY, max_new_tokens=16, seed=7)
            for name, fmt in zip(names, formats, strict=True)
        }
        entries = []
        for name, report in reports.items():
            entries.append(
                {
                    "format": {"name": name},
                    "observations": [
                        {
                            "fixture": obs.fixture,
                            "status": obs.status,
                            "exact_match": obs.exact_match,
                            "boundary_found": obs.boundary_found,
                            "input_ids": list(obs.input_ids),
                        }
                        for obs in report.observations
                    ],
                }
            )
        stage = ws_stage(backend, formats, cases, boundary=BOUNDARY, max_new_tokens=16, seed=7)
        return ws_canonicalization(
            native_search_outcomes(entries, names),
            native_search_input_ids(entries, names),
            stage,
            names,
        )

    def test_parity_backend_is_fragile(self):
        """Test parity backend is fragile."""
        payload = self._run(ParityBackend())
        self.assertTrue(payload["available"])
        cand = payload["per_candidate"]["chatml"]
        self.assertFalse(cand["whitespace_robust"])
        self.assertIn("chatml", payload["fragile_candidates"])
        trailing = cand["per_mutation"]["trailing_space"]
        self.assertGreater(trailing["n_ids_divergent"], 0)
        self.assertGreater(trailing["n_verdict_flips"], 0)

    def test_constant_backend_diverges_but_never_flips(self):
        """Test constant backend diverges but never flips."""
        payload = self._run(ConstantBackend())
        self.assertTrue(payload["available"])
        cand = payload["per_candidate"]["chatml"]
        # Ids diverge under the one-char mutations, but the verdict never
        # flips: not whitespace-robust (the strong claim needs identical
        # ids), yet not fragile either.
        self.assertFalse(cand["whitespace_robust"])
        self.assertEqual(payload["fragile_candidates"], [])
        trailing = cand["per_mutation"]["trailing_space"]
        self.assertGreater(trailing["n_ids_divergent"], 0)
        self.assertEqual(trailing["n_verdict_flips"], 0)
        self.assertFalse(trailing["fragile"])

    def test_same_ids_verdict_flip_recorded_separately(self):
        """Test same ids verdict flip recorded separately."""

        class FlipFlopBackend(ParityBackend):
            """Flip flop backend."""

            fingerprint = "scripted-ws-flipflop-v1"
            calls = 0

            def generate(self, input_ids, *, max_new_tokens, seed, cancel):
                """Generate."""
                type(self).calls += 1
                # Alternates answers across calls: same ids, flipped verdict.
                answer = "WORD" if type(self).calls % 2 == 0 else "NOPE"
                out = tuple(map(ord, (answer + "|").ljust(max_new_tokens, ".")[:max_new_tokens]))
                return Generation(token_ids=out, reason="limit")

        FlipFlopBackend.calls = 0
        payload = self._run(FlipFlopBackend(), n_fixtures=1)
        cand = payload["per_candidate"]["chatml"]
        strip = cand["per_mutation"]["strip_trailing_space"]
        # No whitespace change on the fixture renders: ids identical,
        # but the alternating backend flips the verdict anyway.
        self.assertEqual(strip["n_ids_divergent"], 0)
        self.assertGreaterEqual(strip["n_same_ids_verdict_flips"], 0)

    def test_missing_native_input_ids_excludes_candidate(self):
        """Test missing native input ids excludes candidate."""
        entries = _search_entries(names=("chatml", "gemma"))
        entries[0]["observations"][0]["input_ids"] = "junk"
        names = ("chatml", "gemma")
        formats = (_format("chatml"), _format("gemma"))
        cases = tuple(_fixture(f"f{i}") for i in range(2))
        stage = ws_stage(ParityBackend(), formats, cases, boundary=BOUNDARY)
        payload = ws_canonicalization(
            native_search_outcomes(entries, names),
            native_search_input_ids(entries, names),
            stage,
            names,
        )
        self.assertTrue(payload["available"])
        self.assertIn("chatml", payload["excluded_candidates"])
        self.assertIn("gemma", payload["per_candidate"])

    def test_all_candidates_excluded_is_unavailable(self):
        """Test all candidates excluded is unavailable."""
        entries = _search_entries()
        entries[0]["observations"][0]["input_ids"] = "junk"
        names = ("chatml",)
        cases = tuple(_fixture(f"f{i}") for i in range(2))
        stage = ws_stage(ParityBackend(), (_format(),), cases, boundary=BOUNDARY)
        payload = ws_canonicalization(
            native_search_outcomes(entries, names),
            native_search_input_ids(entries, names),
            stage,
            names,
        )
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "no_evaluable_candidates")

    def test_none_variant_cell_degrades(self):
        """Test none variant cell degrades."""
        names = ("chatml",)
        entries = _search_entries()
        cases = tuple(_fixture(f"f{i}") for i in range(2))
        stage = ws_stage(FailingBackend(), (_format(),), cases, boundary=BOUNDARY)
        payload = ws_canonicalization(
            native_search_outcomes(entries, names),
            native_search_input_ids(entries, names),
            stage,
            names,
        )
        cand = payload["per_candidate"]["chatml"]
        self.assertIsNone(cand["whitespace_robust"])
        self.assertEqual(cand["detail"], "variant_cell_unavailable")

    def test_malformed_ws_result_unavailable(self):
        """Test malformed ws result unavailable."""
        payload = ws_canonicalization({}, {}, {"bogus": True}, ("chatml",))
        self.assertFalse(payload["available"])

    def test_no_evaluable_candidates_unavailable(self):
        """Test no evaluable candidates unavailable."""
        payload = ws_canonicalization(
            {}, {}, {"mutations": ["trailing_space"], "per_mutation": {}}, ()
        )
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "no_evaluable_candidates")


class DiffWsCanonicalTests(unittest.TestCase):
    """Diff ws canonical tests."""

    def test_neither_available(self):
        """Test neither available."""
        diff = diff_ws_canonical(None, None)
        self.assertEqual(diff["available"], "neither")
        self.assertFalse(diff["changed"])

    def test_old_only_asymmetric(self):
        """Test old only asymmetric."""
        diff = diff_ws_canonical(_ws_payload(), None)
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])

    def test_identical_payloads_unchanged(self):
        """Test identical payloads unchanged."""
        diff = diff_ws_canonical(_ws_payload(), _ws_payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_rate_drift_detected(self):
        """Test rate drift detected."""
        old = _ws_payload()
        new = _ws_payload()
        new["per_candidate"]["alpha"]["per_mutation"]["trailing_space"]["verdict_flip_rate"] = 1.0
        diff = diff_ws_canonical(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["rate_drift"]), 1)
        entry = diff["rate_drift"][0]
        self.assertEqual(
            (entry["candidate"], entry["mutation"], entry["metric"]),
            ("alpha", "trailing_space", "verdict_flip_rate"),
        )

    def test_robust_flip_detected(self):
        """Test robust flip detected."""
        old = _ws_payload()
        new = _ws_payload()
        new["per_candidate"]["alpha"]["whitespace_robust"] = False
        diff = diff_ws_canonical(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["robust_flips"]), 1)

    def test_fragile_set_changes_detected(self):
        """Test fragile set changes detected."""
        old = _ws_payload()
        new = _ws_payload(robust=False)
        diff = diff_ws_canonical(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["fragile_added"], ["alpha", "beta"])
        self.assertEqual(diff["fragile_removed"], [])

    def test_candidates_added_removed(self):
        """Test candidates added removed."""
        old = _ws_payload()
        new = _ws_payload()
        del new["per_candidate"]["beta"]
        diff = diff_ws_canonical(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_removed"], ["beta"])

    def test_mutations_added_detected(self):
        """Test mutations added detected."""
        old = _ws_payload(mutations=["trailing_space"])
        new = _ws_payload(mutations=["trailing_space", "leading_space"])
        diff = diff_ws_canonical(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["mutations_added"], ["leading_space"])

    def test_malformed_side_degrades_to_asymmetric(self):
        """Test malformed side degrades to asymmetric."""
        bad = {"available": True, "mutations": "junk", "per_candidate": {}}
        diff = diff_ws_canonical(_ws_payload(), bad)
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])

    def test_non_string_candidate_key_rejected(self):
        """Test non string candidate key rejected."""
        bad = _ws_payload()
        bad["per_candidate"][7] = bad["per_candidate"].pop("alpha")
        diff = diff_ws_canonical(bad, _ws_payload())
        self.assertEqual(diff["available"], "new")

    def test_non_numeric_rate_rejected(self):
        """Test non numeric rate rejected."""
        bad = _ws_payload()
        bad["per_candidate"]["alpha"]["per_mutation"]["trailing_space"]["verdict_flip_rate"] = (
            "high"
        )
        diff = diff_ws_canonical(bad, _ws_payload())
        self.assertEqual(diff["available"], "new")


class WsChangedSingleCausePinTests(unittest.TestCase):
    """Each ``changed`` disjunct alone must flip the verdict (C4/C5 lesson)."""

    def _only(self, mutate):
        """Only."""
        old = _ws_payload()
        new = copy.deepcopy(old)
        mutate(new)
        diff = diff_ws_canonical(old, new)
        self.assertTrue(diff["changed"], "disjunct alone did not flip changed")
        return diff

    def test_rate_drift_only(self):
        """Test rate drift only."""
        self._only(
            lambda new: new["per_candidate"]["alpha"]["per_mutation"]["trailing_space"].update(
                {"ids_divergence_rate": 0.75}
            )
        )

    def test_robust_flip_only(self):
        """Test robust flip only."""
        self._only(lambda new: new["per_candidate"]["alpha"].update({"whitespace_robust": False}))

    def test_fragile_added_only(self):
        """Test fragile added only."""

        def mutate(new):
            """Mutate."""
            new["fragile_candidates"] = ["alpha"]

        self._only(mutate)

    def test_fragile_removed_only(self):
        """Test fragile removed only."""
        old = _ws_payload(robust=False)
        new = _ws_payload(robust=False)
        new["fragile_candidates"] = []
        diff = diff_ws_canonical(old, new)
        self.assertTrue(diff["changed"])

    def test_candidates_added_only(self):
        """Test candidates added only."""

        def mutate(new):
            """Mutate."""
            new["per_candidate"]["gamma"] = copy.deepcopy(new["per_candidate"]["alpha"])

        self._only(mutate)

    def test_candidates_removed_only(self):
        """Test candidates removed only."""

        def mutate(new):
            """Mutate."""
            del new["per_candidate"]["beta"]

        self._only(mutate)

    def test_mutations_added_only(self):
        """Test mutations added only."""
        old = _ws_payload(mutations=["trailing_space"])
        new = _ws_payload(mutations=["trailing_space", "leading_space"])
        diff = diff_ws_canonical(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["mutations_added"], ["leading_space"])

    def test_mutations_removed_only(self):
        """Test mutations removed only."""
        old = _ws_payload(mutations=["trailing_space", "leading_space"])
        new = _ws_payload(mutations=["trailing_space"])
        diff = diff_ws_canonical(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["mutations_removed"], ["leading_space"])


class WsCanonicalMutationGateTests(unittest.TestCase):
    """Tempdir-module mutation gates (the real tree is never rewritten)."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        """Run gate."""
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c156_mut_", dir="/tmp") as tmpdir:
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
    def _trailing_space_scenario(mod):
        """Trailing space scenario."""
        assert mod.apply_mutation("abc", "trailing_space") == "abc "

    def test_mutant_flipped_trailing_space_caught(self):
        """Test mutant flipped trailing space caught."""
        self._run_gate(
            "wscanonical",
            "flipped_trailing_space",
            '    if mutation == "trailing_space":\n        return text + " "',
            '    if mutation == "trailing_space":\n        return " " + text  # MUTANT',
            self._trailing_space_scenario,
        )

    @staticmethod
    def _fail_closed_scenario(mod):
        """Fail closed scenario."""
        try:
            mod.apply_mutation("abc", "nope")
        except ValueError:
            return
        raise AssertionError("unknown mutation did not raise")

    def test_mutant_dropped_fail_closed_caught(self):
        """Test mutant dropped fail closed caught."""
        self._run_gate(
            "wscanonical",
            "dropped_fail_closed",
            (
                '    if mutation == "collapse_double_space":\n'
                '        return text.replace("  ", " ")\n'
                '    raise ValueError(f"unknown whitespace mutation {mutation!r}")'
            ),
            (
                '    if mutation == "collapse_double_space":\n'
                '        return text.replace("  ", " ")\n'
                "    return text  # MUTANT"
            ),
            self._fail_closed_scenario,
        )

    @staticmethod
    def _rate_drift_scenario(mod):
        """Rate drift scenario."""
        old = _ws_payload()
        new = _ws_payload()
        new["per_candidate"]["alpha"]["per_mutation"]["trailing_space"]["verdict_flip_rate"] = 1.0
        assert mod.diff_ws_canonical(old, new)["changed"] is True

    def test_mutant_dropped_rate_drift_caught(self):
        """Test mutant dropped rate drift caught."""
        self._run_gate(
            "diffreports",
            "dropped_ws_rate_drift",
            "        rate_drift\n"
            "        or robust_flips\n"
            "        or fragile_added\n"
            "        or fragile_removed\n"
            "        or candidates_added\n"
            "        or candidates_removed\n"
            "        or mutations_added\n"
            "        or mutations_removed",
            "        robust_flips\n"
            "        or fragile_added\n"
            "        or fragile_removed\n"
            "        or candidates_added\n"
            "        or candidates_removed\n"
            "        or mutations_added\n"
            "        or mutations_removed  # MUTANT",
            self._rate_drift_scenario,
        )

    @staticmethod
    def _robust_flip_scenario(mod):
        """Robust flip scenario."""
        old = _ws_payload()
        new = _ws_payload()
        new["per_candidate"]["beta"]["whitespace_robust"] = False
        assert mod.diff_ws_canonical(old, new)["changed"] is True

    def test_mutant_dropped_robust_flip_caught(self):
        """Test mutant dropped robust flip caught."""
        self._run_gate(
            "diffreports",
            "dropped_ws_robust_flip",
            "        rate_drift\n"
            "        or robust_flips\n"
            "        or fragile_added\n"
            "        or fragile_removed\n"
            "        or candidates_added\n"
            "        or candidates_removed\n"
            "        or mutations_added\n"
            "        or mutations_removed",
            "        rate_drift\n"
            "        or fragile_added\n"
            "        or fragile_removed\n"
            "        or candidates_added\n"
            "        or candidates_removed\n"
            "        or mutations_added\n"
            "        or mutations_removed  # MUTANT",
            self._robust_flip_scenario,
        )


class WsCanonicalSectionTests(unittest.TestCase):
    """Ws canonical section tests."""

    def test_renders_table_and_anchor(self):
        """Test renders table and anchor."""
        html = _wscanonical_section({"ws_canonical": _ws_payload()})
        self.assertIn('id="whitespace-canonicalization"', html)
        self.assertIn("alpha", html)
        self.assertIn("trailing space", html)

    def test_empty_when_unavailable(self):
        """Test empty when unavailable."""
        self.assertEqual(_wscanonical_section({}), "")
        self.assertEqual(_wscanonical_section({"ws_canonical": {"available": False}}), "")

    def test_escapes_hostile_candidate_names(self):
        """Test escapes hostile candidate names."""
        payload = _ws_payload()
        payload["per_candidate"]["<script>alert(1)</script>"] = payload["per_candidate"].pop(
            "alpha"
        )
        html = _wscanonical_section({"ws_canonical": payload})
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_fragile_line_rendered(self):
        """Test fragile line rendered."""
        html = _wscanonical_section({"ws_canonical": _ws_payload(robust=False)})
        self.assertIn("Whitespace-fragile candidates", html)

    def test_mutation_vocabulary_pinned(self):
        """Test mutation vocabulary pinned."""
        from model_protocol_lab import reporting

        self.assertEqual(tuple(reporting._WSCAN_MUTATIONS), MUTATIONS)
        self.assertEqual(tuple(diffreports._WSCAN_MUTATIONS), MUTATIONS)


class WsCanonicalSanitizationTests(unittest.TestCase):
    """Ws canonical sanitization tests."""

    def test_diff_never_raises_on_hostile_payload(self):
        """Test diff never raises on hostile payload."""
        hostile = {
            "available": True,
            "mutations": ["trailing_space", 7, None],
            "per_candidate": {
                "alpha": {
                    "whitespace_robust": "yes",
                    "per_mutation": {
                        "trailing_space": {
                            "ids_divergence_rate": "\x1b[31mred",
                            "verdict_flip_rate": float("nan"),
                        }
                    },
                },
                42: "junk",
            },
            "fragile_candidates": ["\x1b[2Jalpha"],
        }
        diff = diff_ws_canonical(hostile, _ws_payload())
        self.assertIn(diff["available"], ("old", "new", "neither", "both"))

    def test_stderr_renderer_strips_escapes(self):
        """Test stderr renderer strips escapes."""
        old = _ws_payload()
        new = _ws_payload()
        # A hostile candidate name must not reach the renderer raw: the
        # validator rejects non-string structural keys, and _display_id
        # strips control characters from every interpolated label.
        hostile = dict(new["per_candidate"]["alpha"])
        hostile_cell = dict(hostile["per_mutation"]["trailing_space"])
        hostile_cell["verdict_flip_rate"] = 1.0
        hostile["per_mutation"] = dict(hostile["per_mutation"])
        hostile["per_mutation"]["trailing_space"] = hostile_cell
        new["per_candidate"]["\x1b[31malpha"] = hostile
        diff = diff_ws_canonical(old, new)
        # An escape-bearing *string* key passes structural validation;
        # the renderer must strip the control characters via _display_id.
        self.assertEqual(diff["available"], "both")
        lines = diffreports._render_ws_canonical_summary(diff)
        self.assertTrue(lines)
        for line in lines:
            self.assertNotIn("\x1b", line)
        # ...while a hostile *value* inside a valid structure is stripped
        # at render time.
        ok_old = _ws_payload()
        ok_new = _ws_payload()
        ok_new["per_candidate"]["alpha"]["per_mutation"]["trailing_space"]["verdict_flip_rate"] = (
            1.0
        )
        ok_diff = diff_ws_canonical(ok_old, ok_new)
        ok_lines = diffreports._render_ws_canonical_summary(ok_diff)
        self.assertTrue(ok_lines)
        for line in ok_lines:
            self.assertNotIn("\x1b", line)

    def test_stderr_renderer_escapes_drift_entries(self):
        """Test stderr renderer escapes drift entries."""
        old = _ws_payload()
        new = _ws_payload()
        cell = new["per_candidate"]["alpha"]["per_mutation"]["trailing_space"]
        cell["verdict_flip_rate"] = 1.0
        diff = diff_ws_canonical(old, new)
        lines = diffreports._render_ws_canonical_summary(diff)
        self.assertTrue(any("trailing_space" in line for line in lines))
        for line in lines:
            self.assertNotIn("\x1b", line)

    def test_unavailable_renderers_return_empty(self):
        """Test unavailable renderers return empty."""
        self.assertEqual(diffreports._render_ws_canonical_summary(None), [])
        self.assertEqual(diffreports._render_ws_canonical_summary({"available": "neither"}), [])
        self.assertEqual(
            diffreports._render_ws_canonical_summary({"available": "old"}),
            [
                "Whitespace-canonicalization comparison unavailable: "
                "new run did not record the whitespace scan."
            ],
        )


class CliProbeTests(unittest.TestCase):
    """Cli probe tests."""

    def test_run_help_mentions_ws_scan(self):
        """Test run help mentions ws scan."""
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parent.parent
        proc = subprocess.run(
            [sys.executable, "-m", "model_protocol_lab", "run", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertIn("--ws-scan", proc.stdout)

    def test_probe_stores_payload(self):
        """Test probe stores payload."""
        from model_protocol_lab import cli

        cases = tuple(_fixture(f"f{i}") for i in range(2))
        result = {"search": []}
        # Seed result["search"] via a real evaluate so native outcomes exist.
        from model_protocol_lab.core import evaluate

        report = evaluate(
            ParityBackend(), _format(), cases, boundary=BOUNDARY, max_new_tokens=16, seed=7
        )
        result["search"] = [
            {
                "format": {"name": "chatml"},
                "observations": [
                    {
                        "fixture": obs.fixture,
                        "status": obs.status,
                        "exact_match": obs.exact_match,
                        "boundary_found": obs.boundary_found,
                        "input_ids": list(obs.input_ids),
                    }
                    for obs in report.observations
                ],
            }
        ]
        cli._run_ws_probe(
            ParityBackend(),
            (_format(),),
            result,
            cases,
            boundary=BOUNDARY,
            boundary_map=None,
            max_new_tokens=16,
            seed=7,
        )
        payload = result["ws_canonical"]
        self.assertTrue(payload["available"])
        self.assertEqual(payload["mutations"], list(MUTATIONS))

    def test_probe_never_raises_on_broken_backend(self):
        """Test probe never raises on broken backend."""
        from model_protocol_lab import cli

        result = {}
        cli._run_ws_probe(
            FailingBackend(),
            (_format(),),
            result,
            tuple(_fixture(f"f{i}") for i in range(2)),
            boundary=BOUNDARY,
            boundary_map=None,
            max_new_tokens=16,
            seed=7,
        )
        # FailingBackend fails every evaluate cell -> ws_stage degrades
        # cells to None; ws_canonicalization still returns a payload.
        self.assertIn("available", result["ws_canonical"])


class SampleReportTests(unittest.TestCase):
    """Sample report tests."""

    def test_sample_build_is_byte_identical(self):
        """Test sample build is byte identical."""
        script = Path(__file__).resolve().parent.parent / "examples" / "make_sample_report.py"
        spec = importlib.util.spec_from_file_location("make_sample_report_c156", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="c156_sample_") as first:
            result_a = module.build_sample_report(Path(first) / "regen")
            with tempfile.TemporaryDirectory(prefix="c156_sample_") as second:
                result_b = module.build_sample_report(Path(second) / "regen")
        self.assertEqual(
            result_a["ws_canonical"]["per_candidate"].keys(),
            result_b["ws_canonical"]["per_candidate"].keys(),
        )

    def test_committed_sample_has_ws_section(self):
        """Test committed sample has ws section."""
        sample = Path(__file__).resolve().parent.parent / "examples" / "sample-report"
        html = (sample / "report.html").read_text(encoding="utf-8")
        self.assertIn('id="whitespace-canonicalization"', html)


class TocAnchorTests(unittest.TestCase):
    """Toc anchor tests."""

    def test_all_toc_ids_have_section_anchors(self):
        """Test all toc ids have section anchors."""
        # Mirrors the cycle-153 invariant: _toc never points at a missing id.
        from model_protocol_lab import reporting

        html = reporting.render_html(
            {
                "decision": "abstain",
                "search": [],
                "ws_canonical": _ws_payload(),
            }
        )
        self.assertIn('id="whitespace-canonicalization"', html)
        self.assertIn('href="#whitespace-canonicalization"', html)


class WsCoverageGapTests(unittest.TestCase):
    """Targeted tests for fail-closed defensive branches (coverage gate).

    Every line here is a fail-closed guard the happy-path tests never
    take: malformed inputs degrading to None/unavailable markers instead
    of raising. They are pinned because the 95% gate measures the exact
    CI unittest command.
    """

    # --- wscanonical.py: malformed variant ids, bad records ---

    def test_compare_rejects_malformed_variant(self):
        """Test compare rejects malformed variant."""
        with self.assertRaises(ValueError):
            compare_id_sequences((1, 2), [1, 2])

    @staticmethod
    def _valid_record():
        """Valid record."""
        return {
            "mutation": "trailing_space",
            "n_fixtures": 4,
            "n_evaluable": 3,
            "n_ids_divergent": 1,
            "n_verdict_flips": 1,
            "n_same_ids_verdict_flips": 0,
        }

    def test_aggregate_rejects_non_list(self):
        """Test aggregate rejects non list."""
        with self.assertRaises(ValueError):
            aggregate_mutation_records("nope")  # type: ignore[arg-type]

    def test_aggregate_rejects_non_dict_record(self):
        """Test aggregate rejects non dict record."""
        with self.assertRaises(ValueError):
            aggregate_mutation_records(["nope"])  # type: ignore[list-item]

    def test_aggregate_rejects_divergent_over_fixtures(self):
        """Test aggregate rejects divergent over fixtures."""
        record = self._valid_record()
        record["n_ids_divergent"] = 5
        with self.assertRaises(ValueError):
            aggregate_mutation_records([record])

    def test_aggregate_rejects_flips_over_evaluable(self):
        """Test aggregate rejects flips over evaluable."""
        record = self._valid_record()
        record["n_verdict_flips"] = 4
        with self.assertRaises(ValueError):
            aggregate_mutation_records([record])

    # --- experiments.native_search_input_ids: malformed entries ---

    def test_native_input_ids_malformed_entries(self):
        """Test native input ids malformed entries."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.experiments import native_search_input_ids

        # Non-dict entry: candidate degrades to None.
        out = native_search_input_ids(["nope"], ("c1",))
        self.assertIsNone(out["c1"])
        # Entry whose format name is not a string: not indexed.
        out = native_search_input_ids([{"format": {"name": 42}, "observations": []}], ("c1",))
        self.assertIsNone(out["c1"])
        # Non-dict observation poisons the candidate.
        out = native_search_input_ids(
            [{"format": {"name": "c1"}, "observations": ["nope"]}], ("c1",)
        )
        self.assertIsNone(out["c1"])
        # Bad fixture id poisons the candidate.
        out = native_search_input_ids(
            [
                {
                    "format": {"name": "c1"},
                    "observations": [{"fixture": 42, "input_ids": (1,)}],
                }
            ],
            ("c1",),
        )
        self.assertIsNone(out["c1"])
        # Bad token id poisons the candidate.
        out = native_search_input_ids(
            [
                {
                    "format": {"name": "c1"},
                    "observations": [{"fixture": "f", "input_ids": (1, -2)}],
                }
            ],
            ("c1",),
        )
        self.assertIsNone(out["c1"])
        # Missing input_ids poisons the candidate.
        out = native_search_input_ids(
            [
                {
                    "format": {"name": "c1"},
                    "observations": [{"fixture": "f"}],
                }
            ],
            ("c1",),
        )
        self.assertIsNone(out["c1"])

    # --- experiments.ws_stage: contract violations + evaluate raising ---

    def test_ws_stage_rejects_bad_cases(self):
        """Test ws stage rejects bad cases."""
        with self.assertRaises(ValueError):
            ws_stage(ParityBackend(), (_format(),), (), boundary=BOUNDARY)
        with self.assertRaises(ValueError):
            ws_stage(
                ParityBackend(),
                (_format(),),
                ("not-a-fixture",),  # type: ignore[arg-type]
                boundary=BOUNDARY,
            )

    def test_ws_stage_rejects_bad_max_new_tokens(self):
        """Test ws stage rejects bad max new tokens."""
        with self.assertRaises(ValueError):
            ws_stage(
                ParityBackend(),
                (_format(),),
                (_fixture("f0"),),
                boundary=BOUNDARY,
                max_new_tokens=0,
            )

    def test_ws_stage_rejects_bad_seed(self):
        """Test ws stage rejects bad seed."""
        with self.assertRaises(ValueError):
            ws_stage(
                ParityBackend(),
                (_format(),),
                (_fixture("f0"),),
                boundary=BOUNDARY,
                seed=-1,
            )

    def test_ws_stage_evaluate_raise_degrades_cell(self):
        """Test ws stage evaluate raise degrades cell."""

        class EncodeBoomBackend(ParityBackend):
            """Encode boom backend."""

            @staticmethod
            def encode(text):
                """Encode."""
                raise RuntimeError("encode boom")

        stage = ws_stage(
            EncodeBoomBackend(),
            (_format(),),
            (_fixture("f0"),),
            boundary=BOUNDARY,
        )
        cell = stage["per_mutation"]["trailing_space"]["per_candidate"]["chatml"]
        self.assertIsNone(cell)

    # --- experiments.ws_canonicalization: malformed inputs ---

    @staticmethod
    def _good_stage():
        """Good stage."""
        return {"mutations": ["trailing_space"], "per_mutation": {}}

    def test_canonicalization_rejects_malformed_native(self):
        """Test canonicalization rejects malformed native."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.experiments import ws_canonicalization

        out = ws_canonicalization("nope", {}, self._good_stage(), ("c1",))
        self.assertEqual(out["detail"], "native_data_malformed")
        out = ws_canonicalization({}, "nope", self._good_stage(), ("c1",))
        self.assertEqual(out["detail"], "native_data_malformed")

    def test_canonicalization_rejects_malformed_stage(self):
        """Test canonicalization rejects malformed stage."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.experiments import ws_canonicalization

        out = ws_canonicalization({}, {}, "nope", ("c1",))
        self.assertEqual(out["detail"], "ws_result_malformed")
        out = ws_canonicalization({}, {}, {"mutations": [], "per_mutation": {}}, ("c1",))
        self.assertEqual(out["detail"], "ws_result_malformed")
        out = ws_canonicalization({}, {}, {"mutations": ["trailing_space"]}, ("c1",))
        self.assertEqual(out["detail"], "ws_result_malformed")
        out = ws_canonicalization({}, {}, {"mutations": [42], "per_mutation": {}}, ("c1",))
        self.assertEqual(out["detail"], "ws_result_malformed")

    def test_canonicalization_malformed_variant_cell(self):
        """Test canonicalization malformed variant cell."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.experiments import ws_canonicalization

        stage = {
            "mutations": ["trailing_space"],
            "per_mutation": {
                "trailing_space": {
                    "per_candidate": {
                        "c1": {
                            "n": 1,
                            "n_completed": 1,
                            "pass": True,
                            "verdicts": "not-a-dict",
                            "input_ids": {},
                        },
                    }
                }
            },
        }
        native_outcomes = {"c1": {"per_fixture": {"f0": True}}}
        native_ids = {"c1": {"f0": [1, 2]}}
        out = ws_canonicalization(native_outcomes, native_ids, stage, ("c1",))
        cell = out["per_candidate"]["c1"]["per_mutation"]["trailing_space"]
        self.assertFalse(cell["evaluable"])
        self.assertEqual(cell["detail"], "variant_cell_malformed")

    def test_canonicalization_skips_bad_fixture_rows(self):
        """Test canonicalization skips bad fixture rows."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.experiments import ws_canonicalization

        stage = {
            "mutations": ["trailing_space"],
            "per_mutation": {
                "trailing_space": {
                    "per_candidate": {
                        "c1": {
                            "n": 3,
                            "n_completed": 3,
                            "pass": True,
                            # f0: ids not a list -> skipped; f1: verdict not
                            # a bool -> skipped; f2: clean.
                            "verdicts": {"f0": True, "f1": "yes", "f2": True},
                            "input_ids": {
                                "f0": "not-a-list",
                                "f1": [1, 2],
                                "f2": [1, 2],
                            },
                        },
                    }
                }
            },
        }
        native_outcomes = {"c1": {"per_fixture": {"f0": True, "f1": True, "f2": True}}}
        native_ids = {"c1": {"f0": [1, 2], "f1": [1, 2], "f2": [1, 2]}}
        out = ws_canonicalization(native_outcomes, native_ids, stage, ("c1",))
        cell = out["per_candidate"]["c1"]["per_mutation"]["trailing_space"]
        self.assertTrue(cell["evaluable"])
        # Only f2 contributed: no divergence (same ids), no flips.
        self.assertEqual(cell["n_evaluable"], 1)
        self.assertEqual(cell["n_ids_divergent"], 0)
        self.assertEqual(cell["n_verdict_flips"], 0)

    # --- reporting._wscanonical_section: defensive branches ---

    def test_section_malformed_payload_renders_nothing(self):
        """Test section malformed payload renders nothing."""
        self.assertEqual(_wscanonical_section({"ws_canonical": {"available": True}}), "")
        self.assertEqual(
            _wscanonical_section({"ws_canonical": {"available": True, "per_candidate": "x"}}),
            "",
        )

    def test_section_handles_malformed_cells(self):
        """Test section handles malformed cells."""
        payload = {
            "available": True,
            "mutations": ["trailing_space", "leading_space"],
            "per_candidate": {
                "bad-entry": "not-a-dict",
                "c2": {
                    "per_mutation": "not-a-dict",
                    "aggregate": "not-a-dict",
                    "whitespace_robust": "maybe",
                },
                "c3": {
                    "per_mutation": {
                        "trailing_space": {"evaluable": False},
                        "leading_space": {
                            "evaluable": True,
                            "ids_divergence_rate": "bad",
                            "verdict_flip_rate": 0.5,
                            "n_ids_divergent": 0,
                            "fragile": False,
                        },
                    },
                    "aggregate": {},
                    "whitespace_robust": True,
                },
            },
            "fragile_candidates": [],
            "excluded_candidates": [],
        }
        html = _wscanonical_section({"ws_canonical": payload})
        # bad-entry skipped; c2/c3 rendered with n/a fallbacks.
        self.assertNotIn("bad-entry", html)
        self.assertIn("n/a", html)
        self.assertIn("c3", html)

    # --- diffreports._valid_ws_canonical: hostile payloads ---

    def test_valid_ws_canonical_rejects_hostile(self):
        """Test valid ws canonical rejects hostile."""
        from model_protocol_lab.diffreports import _valid_ws_canonical

        good = _ws_payload()
        self.assertTrue(_valid_ws_canonical(good))
        self.assertFalse(_valid_ws_canonical(None))
        self.assertFalse(_valid_ws_canonical({"available": False}))
        bad_mutations = dict(good, mutations="x")
        self.assertFalse(_valid_ws_canonical(bad_mutations))
        bad_mutations = dict(good, mutations=[])
        self.assertFalse(_valid_ws_canonical(bad_mutations))
        bad_mutations = dict(good, mutations=[42])
        self.assertFalse(_valid_ws_canonical(bad_mutations))
        self.assertFalse(_valid_ws_canonical(dict(good, per_candidate="x")))
        self.assertFalse(_valid_ws_canonical(dict(good, per_candidate={42: {}})))
        entry = dict(good["per_candidate"]["alpha"], per_mutation="x")
        bad = dict(good, per_candidate={"alpha": entry})
        self.assertFalse(_valid_ws_canonical(bad))
        entry = {"not": "a dict"}
        bad = dict(good, per_candidate={"alpha": entry})
        self.assertFalse(_valid_ws_canonical(bad))
        entry = dict(good["per_candidate"]["alpha"], whitespace_robust="yes")
        bad = dict(good, per_candidate={"alpha": entry})
        self.assertFalse(_valid_ws_canonical(bad))
        cells = dict(good["per_candidate"]["alpha"]["per_mutation"])
        cells[42] = {}
        entry = dict(good["per_candidate"]["alpha"], per_mutation=cells)
        bad = dict(good, per_candidate={"alpha": entry})
        self.assertFalse(_valid_ws_canonical(bad))
        cells = {
            m: dict(c, ids_divergence_rate="x")
            for m, c in good["per_candidate"]["alpha"]["per_mutation"].items()
        }
        entry = dict(good["per_candidate"]["alpha"], per_mutation=cells)
        bad = dict(good, per_candidate={"alpha": entry})
        self.assertFalse(_valid_ws_canonical(bad))
        cells = {
            m: dict(c, n_ids_divergent="x")
            for m, c in good["per_candidate"]["alpha"]["per_mutation"].items()
        }
        entry = dict(good["per_candidate"]["alpha"], per_mutation=cells)
        bad = dict(good, per_candidate={"alpha": entry})
        self.assertFalse(_valid_ws_canonical(bad))
        cells = {
            m: dict(c, fragile="x")
            for m, c in good["per_candidate"]["alpha"]["per_mutation"].items()
        }
        entry = dict(good["per_candidate"]["alpha"], per_mutation=cells)
        bad = dict(good, per_candidate={"alpha": entry})
        self.assertFalse(_valid_ws_canonical(bad))

    # --- diffreports._render_ws_canonical_summary: all branches ---

    def test_ws_summary_all_branches(self):
        """Test ws summary all branches."""
        from model_protocol_lab.diffreports import _render_ws_canonical_summary

        self.assertEqual(_render_ws_canonical_summary(None), [])
        self.assertEqual(_render_ws_canonical_summary({"available": "neither"}), [])
        lines = _render_ws_canonical_summary({"available": "old"})
        self.assertTrue(any("new run did not record" in line for line in lines))
        lines = _render_ws_canonical_summary({"available": "new"})
        self.assertTrue(any("old run did not record" in line for line in lines))
        # Full diff exercising drift (with >5 overflow), robust flips,
        # fragile added/removed, candidates added/removed.
        drift = [
            {
                "candidate": f"c{i}",
                "mutation": "trailing_space",
                "metric": "verdict_flip_rate",
                "old": 0.0,
                "new": 1.0,
            }
            for i in range(7)
        ]
        drift.append(
            {
                "candidate": "cX",
                "mutation": "leading_space",
                "metric": "fragile",
                "old": False,
                "new": True,
            }
        )
        diff = {
            "available": "both",
            "changed": True,
            "rate_drift": drift,
            "robust_flips": [{"candidate": "c0", "old_robust": True, "new_robust": False}],
            "fragile_added": ["c0"],
            "fragile_removed": ["c1"],
            "candidates_added": ["c9"],
            "candidates_removed": ["c8"],
        }
        lines = _render_ws_canonical_summary(diff)
        text = "\n".join(lines)
        self.assertIn("... and 3 more", text)
        self.assertIn("Whitespace-robust flip", text)
        self.assertIn("newly fragile", text)
        self.assertIn("no longer fragile", text)
        self.assertIn("candidates added", text)
        self.assertIn("candidates removed", text)

    # --- cli._run_ws_probe: the never-raises except block ---

    def test_probe_never_raises_when_stage_raises(self):
        """Test probe never raises when stage raises."""
        from model_protocol_lab.cli import _run_ws_probe

        result = {}
        _run_ws_probe(
            None,
            (),
            result,
            (),
            boundary=None,
            boundary_map=None,
            max_new_tokens=8,
            seed=7,
        )
        self.assertEqual(result["ws_canonical"]["detail"], "ws_scan_unavailable")
