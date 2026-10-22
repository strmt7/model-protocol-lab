"""Cycle 195: choice-scoring audit (`run --choice-audit`).

The frozen contract lives in docs/CHOICE_AUDIT_DESIGN.md: imported-
fixture choice sets (the expected answer plus three planted donors
under the smallest-eligible-id-first-three-v1 rule), three rankings
computed from the reported float32 values (total = lm-eval `acc`;
per-byte = `acc_norm`; per-token mean, never labelled acc_norm), a
fixture counted evaluable only when all four cells score exactly
`ok` with finite values, a method comparison against the generation
outcome over the common evaluable population only, and a
full-recompute diff validator.

Coverage map:
- TestConstruction: the donor rule against the real block shape
  (group-wide equivalence screen, containment both directions,
  byte dedup, rung-4 class dedup, shortfall recorded not
  repaired), the blank exclusion, digest binding, and the
  numeric-answer cross-check against scorerlens's decimal parse.
- TestVerdicts: rank_verdict ties, per-ranking values, generation
  verdicts, record evaluability.
- TestAggregation: denominators and reason counting, the paired
  discordance table with its boundary split, strata, method-
  comparison agreement states and their priority, falsification
  scoping.
- TestStage: the stage against fake backends (happy path, logits
  unsupported, budget cap, cancel, unavailable candidates, prompt
  failure accounting, encode failure, non-finite demotion,
  observation-missing prompt accounting).
- TestValidatorBattery: one break at a time against
  diffreports._valid_choiceaudit, plus diff behaviours.
- TestMutationGates: tempdir package-namespace gates proving the
  recompute's float32-representability check and tie verdict can
  actually fail.
- TestReportingAndCli: the HTML section renders iff the payload is
  available, and the flag/help surface.
"""

import importlib.util
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import torch

from model_protocol_lab import choiceaudit, cli, diffreports, reporting, scorerlens
from model_protocol_lab.choiceaudit import (
    DETAIL_BUDGET_EXCEEDED,
    build_choice_sets,
    discordance_summary,
    donor_map_sha256,
    generation_verdict,
    is_numeric_answer,
    method_comparison,
    protocol_fingerprint,
    protocol_object,
    rank_verdict,
    ranking_values,
    record_evaluable,
    text_digest,
)
from model_protocol_lab.core import Fixture, Format, Message
from model_protocol_lab.experiments import choiceaudit_analysis, choiceaudit_stage

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fx(fid, expected, group):
    return Fixture(
        id=fid,
        messages=(Message(role="user", content=f"q {fid}"),),
        expected=expected,
        group=group,
    )


def _donor_ids(entry):
    return [c["source_fixture"] for c in entry["choices"][1:]]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction(unittest.TestCase):
    def test_basic_donor_order_and_shape(self):
        cases = [
            _fx("a1", "apple", "fruit"),
            _fx("a2", "pear", "fruit"),
            _fx("b1", "dog", "animal"),
            _fx("b2", "cat", "animal"),
            _fx("b3", "wolf", "animal"),
        ]
        block = build_choice_sets(cases)
        target = next(e for e in block if e["fixture"] == "a1")
        self.assertTrue(target["constructible"])
        self.assertIsNone(target["excluded"])
        # True choice first (source None), donors in ascending
        # fixture-id order; donors may share a group with each
        # other but never the target's.
        self.assertIsNone(target["choices"][0]["source_fixture"])
        self.assertEqual(target["choices"][0]["digest"], text_digest("apple"))
        self.assertEqual(_donor_ids(target), ["b1", "b2", "b3"])
        self.assertEqual(target["n_distinct_choices"], 4)

    def test_group_wide_screen_rejects_donor_matching_any_group_member(self):
        # "cat" is equivalent (folded containment) to "wild cat",
        # an expected of the target's own group, so it is
        # ineligible even though the target's own answer is "dog".
        cases = [
            _fx("t1", "dog", "pets"),
            _fx("t2", "wild cat", "pets"),
            _fx("d1", "cat", "wild"),
            _fx("d2", "fish", "wild"),
            _fx("d3", "bird", "wild"),
            _fx("d4", "frog", "wild"),
        ]
        block = build_choice_sets(cases)
        target = next(e for e in block if e["fixture"] == "t1")
        self.assertEqual(_donor_ids(target), ["d2", "d3", "d4"])

    def test_containment_rejected_both_directions(self):
        cases = [
            _fx("t1", "New York City", "geo"),
            _fx("d1", "York", "other"),
            _fx("d2", "New York City limits", "other"),
            _fx("d3", "Boston", "other"),
            _fx("d4", "Denver", "other"),
            _fx("d5", "Miami", "other"),
        ]
        block = build_choice_sets(cases)
        target = next(e for e in block if e["fixture"] == "t1")
        self.assertEqual(_donor_ids(target), ["d3", "d4", "d5"])

    def test_rung4_equivalent_donor_rejected(self):
        cases = [
            _fx("t1", "Paris!", "geo"),
            _fx("d1", "paris", "other"),
            _fx("d2", "London", "other"),
            _fx("d3", "Berlin", "other"),
            _fx("d4", "Madrid", "other"),
        ]
        block = build_choice_sets(cases)
        target = next(e for e in block if e["fixture"] == "t1")
        self.assertNotIn("d1", _donor_ids(target))
        self.assertTrue(target["constructible"])

    def test_byte_duplicate_and_class_duplicate_donors_collapse(self):
        cases = [
            _fx("t1", "sun", "sky"),
            _fx("d1", "moon", "night"),
            _fx("d2", "moon", "night"),
            _fx("d3", "MOON", "night"),
            _fx("d4", "star", "night"),
            _fx("d5", "mars", "night"),
        ]
        block = build_choice_sets(cases)
        target = next(e for e in block if e["fixture"] == "t1")
        self.assertEqual(_donor_ids(target), ["d1", "d4", "d5"])

    def test_shortfall_recorded_not_repaired(self):
        cases = [
            _fx("t1", "one", "g"),
            _fx("d1", "two", "h"),
            _fx("d2", "three", "h"),
        ]
        block = build_choice_sets(cases)
        target = next(e for e in block if e["fixture"] == "t1")
        self.assertFalse(target["constructible"])
        # The realised (partial) choices stay on the record.
        self.assertEqual(_donor_ids(target), ["d1", "d2"])
        self.assertEqual(target["n_distinct_choices"], 3)

    def test_blank_expected_excluded(self):
        cases = [
            _fx("t1", "  ", "g"),
            _fx("t2", "?", "g"),
            _fx("d1", "alpha", "h"),
            _fx("d2", "beta", "h"),
            _fx("d3", "gamma", "h"),
            _fx("d4", "delta", "h"),
        ]
        block = build_choice_sets(cases)
        by_id = {e["fixture"]: e for e in block}
        self.assertEqual(by_id["t1"]["excluded"], "blank_expected")
        self.assertEqual(by_id["t1"]["choices"], [])
        self.assertFalse(by_id["t1"]["constructible"])
        # "?" survives the rung-1 screen (the scorer-lens rule keeps
        # it there) and is constructible here.
        self.assertIsNone(by_id["t2"]["excluded"])
        self.assertTrue(by_id["t2"]["constructible"])

    def test_excluded_expected_never_serves_as_donor(self):
        cases = [
            _fx("t1", "alpha", "g"),
            _fx("e1", "  ", "h"),
            _fx("d1", "beta", "h"),
            _fx("d2", "gamma", "h"),
            _fx("d3", "delta", "h"),
        ]
        block = build_choice_sets(cases)
        target = next(e for e in block if e["fixture"] == "t1")
        self.assertNotIn("e1", _donor_ids(target))
        self.assertEqual(_donor_ids(target), ["d1", "d2", "d3"])

    def test_donor_map_digest_binds_the_block(self):
        cases = [
            _fx("t1", "alpha", "g"),
            _fx("d1", "beta", "h"),
            _fx("d2", "gamma", "h"),
            _fx("d3", "delta", "h"),
        ]
        block = build_choice_sets(cases)
        digest_a = donor_map_sha256(block)
        import copy

        altered = copy.deepcopy(block)
        target = next(e for e in altered if e["fixture"] == "t1")
        target["choices"][1]["group"] = "elsewhere"
        self.assertNotEqual(donor_map_sha256(altered), digest_a)

    def test_protocol_fingerprint_binds_backend(self):
        block = build_choice_sets([_fx("t1", "alpha", "g")])
        obj = protocol_object(["t1"], block, "backend-a")
        self.assertEqual(obj["donor_map_sha256"], donor_map_sha256(block))
        self.assertEqual(
            protocol_fingerprint(["t1"], block, "backend-a"),
            protocol_fingerprint(["t1"], block, "backend-a"),
        )
        self.assertNotEqual(
            protocol_fingerprint(["t1"], block, "backend-a"),
            protocol_fingerprint(["t1"], block, "backend-b"),
        )

    def test_numeric_flag_on_block(self):
        block = build_choice_sets([_fx("t1", "42", "g"), _fx("t2", "abc", "g")])
        by_id = {e["fixture"]: e for e in block}
        self.assertTrue(by_id["t1"]["numeric"])
        self.assertFalse(by_id["t2"]["numeric"])

    def test_numeric_answer_matches_scorerlens_decimal(self):
        corpus = [
            "42",
            "0.5",
            "1/2",
            "-3.25",
            "17%",
            "1,000",
            "3.",
            ".5",
            "abc",
            "12a",
            "",
            " 7 ",
            "0",
            "-0",
            "1000000",
            "3/4",
            "2,500.75",
            # Grammar-matching but beyond Decimal's exponent
            # range: both parses refuse it (the InvalidOperation
            # branch), so the stratum and the lens stay in step.
            "1e999999999999999999999",
        ]
        for text in corpus:
            with self.subTest(text=text):
                # scorerlens's own call pattern passes the R3 form.
                self.assertEqual(
                    is_numeric_answer(text),
                    scorerlens._decimal_value(scorerlens.ladder_forms(text)[3]) is not None,
                )


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def _cell(total, *, mean=None, tokens=2, byte_length=None, status="ok", reason=None):
    if mean is None and total is not None:
        mean = total / tokens
    return {
        "status": status,
        "reason": reason,
        "total": total,
        "mean": mean,
        "tokens": tokens,
        "bytes": byte_length if byte_length is not None else tokens * 2,
    }


def _record(
    fixture="f1", *, cells, obs_status="completed", exact=True, boundary=True, prompt_reason=None
):
    return {
        "fixture": fixture,
        "obs_status": obs_status,
        "exact_match": exact,
        "boundary_found": boundary,
        "prompt_reason": prompt_reason,
        "cells": cells,
    }


_ENTRY_OK = {"fixture": "f1", "constructible": True}
_ENTRY_SHORT = {"fixture": "f1", "constructible": False}


class TestVerdicts(unittest.TestCase):
    def test_rank_verdict_three_states(self):
        self.assertEqual(rank_verdict([3.0, 1.0, 2.0, 0.0]), "correct")
        self.assertEqual(rank_verdict([1.0, 3.0, 2.0, 0.0]), "incorrect")
        self.assertEqual(rank_verdict([3.0, 3.0, 2.0, 0.0]), "tied")
        self.assertEqual(rank_verdict([3.0, 3.0, 3.0, 3.0]), "tied")

    def test_ranking_values_per_ranking(self):
        cells = [
            _cell(-8.0, byte_length=4),
            _cell(-9.0, byte_length=3),
            _cell(-7.0, byte_length=7),
            _cell(-6.0, tokens=3, byte_length=6),
        ]
        self.assertEqual(ranking_values(cells, "total"), [-8.0, -9.0, -7.0, -6.0])
        self.assertEqual(ranking_values(cells, "byte")[0], -2.0)
        self.assertEqual(ranking_values(cells, "byte")[1], -3.0)
        self.assertEqual(ranking_values(cells, "mean")[3], -2.0)
        with self.assertRaises(ValueError):
            ranking_values(cells, "bogus")

    def test_generation_verdict_states(self):
        self.assertEqual(generation_verdict("completed", True, True), "pass")
        self.assertEqual(generation_verdict("completed", False, True), "fail")
        self.assertEqual(generation_verdict("completed", True, False), "fail")
        self.assertEqual(generation_verdict("completed", True, None), "fail")
        self.assertEqual(generation_verdict("failed", True, True), "missing")
        self.assertEqual(generation_verdict(None, None, None), "missing")

    def test_record_evaluable_rules(self):
        good = _record(cells=[_cell(-1.0) for _ in range(4)])
        self.assertTrue(record_evaluable(good, _ENTRY_OK))
        self.assertFalse(record_evaluable(good, _ENTRY_SHORT))
        short = _record(cells=[_cell(-1.0) for _ in range(3)])
        self.assertFalse(record_evaluable(short, _ENTRY_OK))
        empty = _record(cells=[_cell(0.0, status="ok_empty")] + [_cell(-1.0) for _ in range(3)])
        self.assertFalse(record_evaluable(empty, _ENTRY_OK))
        non_finite = _record(cells=[_cell(float("nan"))] + [_cell(-1.0) for _ in range(3)])
        self.assertFalse(record_evaluable(non_finite, _ENTRY_OK))
        none_total = _record(
            cells=[
                {
                    "status": "ok",
                    "reason": None,
                    "total": None,
                    "mean": None,
                    "tokens": 1,
                    "bytes": 1,
                }
            ]
            + [_cell(-1.0) for _ in range(3)]
        )
        self.assertFalse(record_evaluable(none_total, _ENTRY_OK))
        prompt_failed_shape = _record(cells=[], prompt_reason="decode_failed")
        self.assertFalse(record_evaluable(prompt_failed_shape, _ENTRY_OK))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _agg_fixtures():
    """A small real block: every fixture constructible."""
    cases = [
        _fx("f1", "42", "numbers"),
        _fx("f2", "red", "colors"),
        _fx("d1", "dog", "animals"),
        _fx("d2", "cat", "animals"),
        _fx("d3", "oak", "trees"),
        _fx("d4", "elm", "trees"),
    ]
    return build_choice_sets(cases)


def _win_cells():
    # True choice (index 0) wins every ranking outright.
    return [
        _cell(-1.0, tokens=2, byte_length=4),
        _cell(-3.0, tokens=2, byte_length=4),
        _cell(-4.0, tokens=2, byte_length=4),
        _cell(-5.0, tokens=2, byte_length=4),
    ]


def _lose_cells():
    return [
        _cell(-5.0, tokens=2, byte_length=4),
        _cell(-1.0, tokens=2, byte_length=4),
        _cell(-4.0, tokens=2, byte_length=4),
        _cell(-3.0, tokens=2, byte_length=4),
    ]


class TestAggregation(unittest.TestCase):
    def setUp(self):
        self.fixtures = _agg_fixtures()

    def _entry(self, fixture):
        return next(e for e in self.fixtures if e["fixture"] == fixture)

    def test_denominators_and_discordance(self):
        records = [
            _record("f1", cells=_win_cells()),
            _record("f2", cells=_win_cells(), exact=False),
        ]
        agg = choiceaudit.aggregate_candidate(records, self.fixtures)
        self.assertEqual(agg["n_evaluable"], 2)
        self.assertEqual(agg["accuracy"]["total"]["correct"], 2)
        self.assertEqual(agg["accuracy"]["total"]["n"], 2)
        self.assertEqual(agg["accuracy"]["total"]["accuracy"], 1.0)
        disc = agg["discordance"]
        self.assertEqual(disc["gen_pass_choice_correct"], 1)
        self.assertEqual(disc["gen_fail_choice_correct"], 1)
        self.assertEqual(disc["n_paired"], 2)
        self.assertEqual(disc["n_discordant"], 1)
        # f2's generation failed with the boundary found.
        self.assertEqual(disc["gen_fail_boundary_found"], 1)
        self.assertEqual(disc["gen_fail_boundary_not_found"], 0)

    def test_boundary_not_found_split(self):
        records = [_record("f1", cells=_lose_cells(), exact=False, boundary=False)]
        agg = choiceaudit.aggregate_candidate(records, self.fixtures)
        disc = agg["discordance"]
        self.assertEqual(disc["gen_fail_choice_incorrect"], 1)
        self.assertEqual(disc["gen_fail_boundary_not_found"], 1)
        self.assertEqual(disc["n_discordant"], 0)

    def test_tied_paired_cell_counted_beside_the_table(self):
        tied = [
            _cell(-1.0, tokens=2, byte_length=4),
            _cell(-1.0, tokens=2, byte_length=4),
            _cell(-4.0, tokens=2, byte_length=4),
            _cell(-5.0, tokens=2, byte_length=4),
        ]
        records = [_record("f1", cells=tied)]
        agg = choiceaudit.aggregate_candidate(records, self.fixtures)
        self.assertEqual(agg["discordance"]["n_choice_tied"], 1)
        self.assertEqual(agg["discordance"]["n_paired"], 0)
        self.assertEqual(agg["accuracy"]["total"]["tied"], 1)

    def test_gen_missing_not_paired(self):
        records = [
            _record("f1", cells=_win_cells(), obs_status="failed", exact=None, boundary=None)
        ]
        agg = choiceaudit.aggregate_candidate(records, self.fixtures)
        self.assertEqual(agg["n_generation_missing"], 1)
        self.assertEqual(agg["discordance"]["n_paired"], 0)
        # The choice verdict still exists (discordance needs both).
        self.assertEqual(agg["n_evaluable"], 1)

    def test_prompt_reason_counted_once_cells_not_counted(self):
        records = [_record("f1", cells=[], prompt_reason="prompt_roundtrip_mismatch")]
        agg = choiceaudit.aggregate_candidate(records, self.fixtures)
        self.assertEqual(agg["n_prompt_failures"], 1)
        self.assertEqual(agg["unavailable_reasons"], {"prompt_roundtrip_mismatch": 1})
        self.assertEqual(agg["n_evaluable"], 0)
        self.assertEqual(agg["n_choice_unavailable"], agg["n_constructible"])

    def test_cell_reasons_counted_per_cell(self):
        records = [
            _record(
                "f1",
                cells=[
                    _cell(None, status="unavailable", reason="encode_failed"),
                    _cell(-1.0),
                    _cell(-2.0),
                    _cell(-3.0),
                ],
            )
        ]
        agg = choiceaudit.aggregate_candidate(records, self.fixtures)
        self.assertEqual(agg["unavailable_reasons"], {"encode_failed": 1})
        self.assertEqual(agg["n_evaluable"], 0)

    def test_ok_empty_counted_under_its_own_status(self):
        records = [
            _record(
                "f1",
                cells=[_cell(0.0, status="ok_empty")] + [_cell(-1.0) for _ in range(3)],
            )
        ]
        agg = choiceaudit.aggregate_candidate(records, self.fixtures)
        self.assertEqual(agg["unavailable_reasons"], {"ok_empty": 1})
        self.assertEqual(agg["n_evaluable"], 0)

    def test_non_constructible_records_skip_populations(self):
        short_fixtures = build_choice_sets([_fx("s1", "one", "g"), _fx("s2", "two", "h")])
        records = [_record("s1", cells=[]), _record("s2", cells=[])]
        agg = choiceaudit.aggregate_candidate(records, short_fixtures)
        self.assertEqual(agg["n_constructible"], 0)
        self.assertEqual(agg["n_evaluable"], 0)
        self.assertIsNone(agg["accuracy"]["total"]["accuracy"])
        self.assertEqual(agg["accuracy"]["total"]["detail"], "no_evaluable_fixtures")

    def test_length_matched_stratum(self):
        spread = _win_cells()
        spread[3] = _cell(-5.0, tokens=9, byte_length=18)
        records = [_record("f1", cells=spread), _record("f2", cells=_win_cells())]
        agg = choiceaudit.aggregate_candidate(records, self.fixtures)
        self.assertEqual(agg["strata"]["length_matched"]["n"], 1)
        self.assertEqual(agg["strata"]["length_matched"]["correct"], 1)

    def test_numeric_stratum(self):
        records = [_record("f1", cells=_win_cells()), _record("f2", cells=_win_cells())]
        agg = choiceaudit.aggregate_candidate(records, self.fixtures)
        self.assertEqual(agg["strata"]["numeric"]["n"], 1)  # f1's "42"
        self.assertEqual(agg["strata"]["non_numeric"]["n"], 1)  # f2's "red"

    def test_ranking_agreement_counts(self):
        # Total: donor 1 wins (-3). Byte: the true choice wins
        # (-4/8 = -0.5). Mean: donor 2 wins (-9/9 = -1). So byte
        # disagrees with total and mean agrees (both incorrect).
        cells = [
            _cell(-4.0, tokens=2, byte_length=8),
            _cell(-3.0, tokens=1, byte_length=1),
            _cell(-9.0, tokens=9, byte_length=6),
            _cell(-8.0, tokens=2, byte_length=4),
        ]
        records = [_record("f1", cells=cells)]
        agg = choiceaudit.aggregate_candidate(records, self.fixtures)
        agreement = agg["strata"]["ranking_agreement"]
        self.assertEqual(agreement["byte_vs_total"], {"agree": 0, "disagree": 1})
        self.assertEqual(agreement["mean_vs_total"], {"agree": 1, "disagree": 0})

    def _method_aggs(self, verdicts_by_name):
        return {
            name: {"_verdicts": {fid: {"total": v} for fid, v in per.items()}}
            for name, per in verdicts_by_name.items()
        }

    def test_method_comparison_states_and_priority(self):
        big = {f"f{i}": "correct" for i in range(8)}
        # Withheld beats everything below the floor.
        out = method_comparison(
            self._method_aggs({"a": {"f1": "correct"}, "b": {"f1": "correct"}}),
            ("a", "b"),
            "compatible_on_fixture_set",
            "a",
        )
        self.assertEqual(out["agreement"], "withheld")
        self.assertEqual(out["n_common"], 1)
        self.assertTrue(out["withheld"])
        # Native abstention outranks the choice outcome.
        out = method_comparison(
            self._method_aggs({"a": big, "b": big}),
            ("a", "b"),
            "no_candidate",
            None,
        )
        self.assertEqual(out["agreement"], "native_abstain")
        # A dead heat for the top common accuracy is choice_tied.
        out = method_comparison(
            self._method_aggs({"a": big, "b": big}),
            ("a", "b"),
            "compatible_on_fixture_set",
            "a",
        )
        self.assertEqual(out["agreement"], "choice_tied")
        self.assertEqual(sorted(out["choice_top_tied"]), ["a", "b"])
        # Unique top: agree / disagree against the primary winner.
        weaker = dict(big, f3="incorrect")
        out = method_comparison(
            self._method_aggs({"a": weaker, "b": big}),
            ("a", "b"),
            "compatible_on_fixture_set",
            "b",
        )
        self.assertEqual(out["agreement"], "agree")
        self.assertEqual(out["choice_top_candidate"], "b")
        out = method_comparison(
            self._method_aggs({"a": weaker, "b": big}),
            ("a", "b"),
            "compatible_on_fixture_set",
            "a",
        )
        self.assertEqual(out["agreement"], "disagree")
        # Common accuracy is over the intersection only.
        self.assertEqual(out["common_accuracy"]["b"], {"correct": 8, "n": 8, "accuracy": 1.0})

    def test_method_comparison_primary_mapping(self):
        big = {f"f{i}": "correct" for i in range(8)}
        out = method_comparison(
            self._method_aggs({"a": big, "b": big}),
            ("a", "b"),
            "held_out_failed",
            "a",
        )
        self.assertEqual(out["primary_decision"], "selected")
        self.assertEqual(out["primary_winner"], "a")

    def test_discordance_falsification_tristate(self):
        empty = {"a": {"discordance": {"n_paired": 0, "n_discordant": 0}}}
        self.assertEqual(
            discordance_summary(empty),
            {"n_cells": 0, "n_discordant": 0, "falsified": None},
        )
        clean = {"a": {"discordance": {"n_paired": 3, "n_discordant": 0}}}
        self.assertEqual(discordance_summary(clean)["falsified"], True)
        dirty = {"a": {"discordance": {"n_paired": 3, "n_discordant": 1}}}
        self.assertEqual(discordance_summary(dirty)["falsified"], False)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Stage (fake backends)
# ---------------------------------------------------------------------------


class _CharBackend:
    """Byte-level tokenizer over a 256-symbol vocab; seeded logits."""

    supports_prompt_logprobs = True

    def __init__(self, *, mode="echo", seed=1234):
        self.mode = mode
        self._seed = seed

    def fingerprint(self):
        return {"backend": "fake-char", "mode": self.mode}

    def encode(self, text):
        return tuple(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")

    def token_bytes(self, token_id):
        return bytes([int(token_id)])

    def forward_logits(self, input_ids):
        seq = [int(i) for i in input_ids]
        gen = torch.Generator().manual_seed(self._seed + len(seq))
        rows = []
        for pos, current in enumerate(seq):
            if self.mode == "flat":
                rows.append(torch.zeros(256))
                continue
            logits = torch.randn(256, generator=gen) * 0.5
            nxt = seq[pos + 1] if pos + 1 < len(seq) else current
            logits[nxt] += 3.0
            rows.append(logits)
        return torch.stack(rows)


class _KeyedBackend(_CharBackend):
    """Noise-free prompt-keyed logits: a +8 boost on the scored
    token when it is a digit (prompts starting with byte 1) or a
    letter (prompts starting with byte 9). With the stage cases,
    the byte-1 candidate answers both numeric fixtures correctly
    and nothing else; the byte-9 candidate answers nothing
    correctly (ties at best) -- a deterministic unique top."""

    def fingerprint(self):
        return {"backend": "fake-keyed"}

    def forward_logits(self, input_ids):
        seq = [int(i) for i in input_ids]
        key = seq[0] if seq else 0
        rows = []
        for pos in range(len(seq)):
            logits = torch.zeros(256)
            if pos + 1 < len(seq):
                nxt = seq[pos + 1]
                if (key == 1 and 48 <= nxt <= 57) or (key == 9 and 97 <= nxt <= 122):
                    logits[nxt] += 8.0
            rows.append(logits)
        return torch.stack(rows)


class _NoLogitsBackend:
    """No forward_logits at all: the stage must refuse up front."""

    supports_prompt_logprobs = False

    def fingerprint(self):
        return {"backend": "fake-nologits"}

    def encode(self, text):
        return tuple(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")

    def token_bytes(self, token_id):
        return bytes([int(token_id)])


class _NonFiniteBackend(_CharBackend):
    def forward_logits(self, input_ids):
        seq = [int(i) for i in input_ids]
        rows = []
        for pos in range(len(seq)):
            logits = torch.zeros(256)
            if pos + 1 < len(seq):
                logits[seq[pos + 1]] = -3.0e38
            rows.append(logits)
        return torch.stack(rows)


class _MangleDecodeBackend(_CharBackend):
    def decode(self, ids):
        return super().decode(ids) + "!"


class _EncodeFailBackend(_CharBackend):
    def encode(self, text):
        if text == "red":
            raise ValueError("cannot encode")
        return super().encode(text)


_FORMATS = (
    Format(name="chatml", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
    Format(name="plain", bos="", user_prefix="U:", assistant_prefix="A:", turn_end="</s>"),
)


def _stage_cases():
    data = [
        ("f0", "42", "numbers"),
        ("f1", "17", "numbers"),
        ("f2", "red", "colors"),
        ("f3", "blue", "colors"),
        ("f4", "cat", "animals"),
        ("f5", "dog", "animals"),
        ("f6", "oak", "trees"),
        ("f7", "elm", "trees"),
    ]
    return tuple(
        Fixture(
            id=fid,
            messages=(Message(role="user", content=f"question {fid}"),),
            expected=expected,
            group=group,
        )
        for fid, expected, group in data
    )


def _search_entries(cases, *, exact_ids=("f0", "f2")):
    entries = []
    for fmt in _FORMATS:
        # Distinct recorded prompts per candidate (the keyed
        # backend reads the first prompt byte).
        prompt_ids = [1, 2, 3] if fmt.name == "chatml" else [9, 9, 9]
        observations = [
            {
                "fixture": case.id,
                "status": "completed",
                "exact_match": case.id in exact_ids,
                "boundary_found": True,
                "input_ids": prompt_ids,
                "answer": case.expected if case.id in exact_ids else "wrong",
            }
            for case in cases
        ]
        entries.append(
            {
                "format": {"name": fmt.name},
                "observations": observations,
                "summary": {"aggregate_eligible": True},
            }
        )
    return entries


def _run_stage(backend=None, *, cases=None, entries=None, cancel=None):
    backend = backend or _CharBackend()
    cases = tuple(cases) if cases is not None else _stage_cases()
    entries = entries if entries is not None else _search_entries(cases)
    stage = choiceaudit_stage(backend, _FORMATS, cases, entries, cancel=cancel)
    payload = choiceaudit_analysis(stage, "no_candidate", None, tuple(f.name for f in _FORMATS))
    return stage, payload


class TestStage(unittest.TestCase):
    def test_happy_path_payload_is_valid(self):
        _, payload = _run_stage()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["detail"], "")
        self.assertEqual(payload["n_fixtures"], 8)
        self.assertTrue(payload["fixtures"])
        self.assertTrue(diffreports._valid_choiceaudit(payload))
        self_diff = diffreports.diff_choiceaudit(payload, payload)
        self.assertEqual(self_diff["available"], "both")
        self.assertFalse(self_diff["changed"])
        self.assertTrue(self_diff["comparable"])

    def test_logits_not_supported(self):
        _, payload = _run_stage(_NoLogitsBackend())
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "logits_not_supported")

    def test_budget_cap_checked_first(self):
        # 25 candidates x 100 fixtures = 2,500 records > the 2,400
        # cap; the cap fires before any search entry is consulted
        # (entries are empty here).
        formats = tuple(
            Format(name=f"fmt{i}", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>")
            for i in range(25)
        )
        cases = tuple(_fx(f"x{i}", f"v{i}", f"g{i}") for i in range(100))
        stage = choiceaudit_stage(_CharBackend(), formats, cases, [], cancel=None)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], DETAIL_BUDGET_EXCEEDED)
        self.assertEqual(stage["candidates"], {})

    def test_cancel_returns_cancelled_without_partials(self):
        event = threading.Event()
        event.set()
        _, payload = _run_stage(cancel=event)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")

    def test_candidate_without_observations_is_unavailable(self):
        cases = _stage_cases()
        entries = _search_entries(cases)
        entries[1] = {"format": {"name": "plain"}, "summary": {}}
        stage, _ = _run_stage(cases=cases, entries=entries)
        self.assertEqual(
            stage["candidates"]["plain"],
            {"status": "unavailable", "detail": "native_search_unavailable", "records": []},
        )
        self.assertEqual(stage["candidates"]["chatml"]["status"], "available")

    def test_prompt_failure_accounted_once_cells_empty(self):
        _, payload = _run_stage(_MangleDecodeBackend())
        candidate = payload["candidates"]["chatml"]
        self.assertGreater(candidate["n_prompt_failures"], 0)
        self.assertEqual(candidate["n_evaluable"], 0)
        self.assertTrue(diffreports._valid_choiceaudit(payload))

    def test_encode_failure_lands_on_the_cell(self):
        _, payload = _run_stage(_EncodeFailBackend())
        reasons = payload["candidates"]["chatml"]["unavailable_reasons"]
        self.assertGreaterEqual(reasons.get("encode_failed", 0), 1)
        self.assertTrue(diffreports._valid_choiceaudit(payload))

    def test_non_finite_totals_demoted(self):
        _, payload = _run_stage(_NonFiniteBackend())
        candidate = payload["candidates"]["chatml"]
        self.assertEqual(candidate["n_evaluable"], 0)
        self.assertGreaterEqual(candidate["unavailable_reasons"].get("non_finite_score", 0), 1)
        self.assertTrue(diffreports._valid_choiceaudit(payload))

    def test_malformed_observation_ids_prompt_missing(self):
        cases = _stage_cases()
        entries = _search_entries(cases)
        for obs in entries[0]["observations"]:
            obs["input_ids"] = "not-a-list"
        _, payload = _run_stage(cases=cases, entries=entries)
        candidate = payload["candidates"]["chatml"]
        self.assertGreater(candidate["n_prompt_failures"], 0)
        self.assertEqual(
            candidate["unavailable_reasons"].get("observation_missing", 0),
            candidate["n_prompt_failures"],
        )


# ---------------------------------------------------------------------------
# Validator battery + diff behaviours
# ---------------------------------------------------------------------------


def _valid_payload():
    _, payload = _run_stage()
    assert payload["available"] and diffreports._valid_choiceaudit(payload)
    return payload


def _wide_loser_tamper(payload):
    """Shift one cell's total by -0.01 where it is a wide loser on
    ALL three rankings, so no verdict can move: only the float32
    representability check can reject the result."""
    import copy
    import struct

    def _f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    payload = copy.deepcopy(payload)
    for record in payload["candidates"]["chatml"]["records"]:
        cells = record["cells"]
        if len(cells) != 4 or any(c["status"] != "ok" for c in cells):
            continue
        totals = [c["total"] for c in cells]
        per_byte = [c["total"] / c["bytes"] for c in cells]
        for idx in range(4):
            # The shift moves the total by 0.01 and the per-byte
            # value by at most 0.01; the stored mean does not move
            # at all. A 0.05 margin on total and per-byte therefore
            # guarantees no verdict changes.
            if totals[idx] < max(totals) - 0.05 and per_byte[idx] < max(per_byte) - 0.05:
                shifted = totals[idx] - 0.01
                if _f32(shifted) == shifted:
                    continue  # freak case: still representable
                record["cells"][idx]["total"] = shifted
                return payload
    raise AssertionError("no wide-loser cell found")


class TestValidatorBattery(unittest.TestCase):
    def setUp(self):
        self.payload = _valid_payload()

    def _tampered(self, mutate):
        import copy

        payload = copy.deepcopy(self.payload)
        mutate(payload)
        return payload

    def test_aggregate_tamper_rejected(self):
        def mutate(p):
            p["candidates"]["chatml"]["accuracy"]["total"]["correct"] += 1

        self.assertFalse(diffreports._valid_choiceaudit(self._tampered(mutate)))

    def test_protocol_fingerprint_tamper_rejected(self):
        def mutate(p):
            p["protocol"]["fingerprint"] = "0" * 64

        self.assertFalse(diffreports._valid_choiceaudit(self._tampered(mutate)))

    def test_donor_digest_tamper_rejected(self):
        def mutate(p):
            entry = next(e for e in p["fixtures"] if e["constructible"])
            entry["choices"][1]["digest"] = "0" * 64

        self.assertFalse(diffreports._valid_choiceaudit(self._tampered(mutate)))

    def test_donor_group_equals_target_group_rejected(self):
        def mutate(p):
            entry = next(e for e in p["fixtures"] if e["constructible"])
            entry["choices"][1]["group"] = entry["group"]

        self.assertFalse(diffreports._valid_choiceaudit(self._tampered(mutate)))

    def test_duplicate_choice_digests_rejected(self):
        def mutate(p):
            entry = next(e for e in p["fixtures"] if e["constructible"])
            digest = entry["choices"][0]["digest"]
            for choice in entry["choices"]:
                choice["digest"] = digest

        self.assertFalse(diffreports._valid_choiceaudit(self._tampered(mutate)))

    def test_n_common_tamper_rejected(self):
        def mutate(p):
            p["method_comparison"]["n_common"] += 1

        self.assertFalse(diffreports._valid_choiceaudit(self._tampered(mutate)))

    def test_agreement_flip_rejected(self):
        def mutate(p):
            p["method_comparison"]["agreement"] = (
                "agree" if p["method_comparison"]["agreement"] != "agree" else "disagree"
            )

        self.assertFalse(diffreports._valid_choiceaudit(self._tampered(mutate)))

    def test_falsified_flip_rejected(self):
        def mutate(p):
            p["discordance"]["falsified"] = p["discordance"]["falsified"] is False

        self.assertFalse(diffreports._valid_choiceaudit(self._tampered(mutate)))

    def test_unmapped_primary_decision_rejected(self):
        # The stored primary values are the mapped ones; a stored
        # "held_out_failed" recomputes to "selected" and mismatches.
        def mutate(p):
            p["method_comparison"]["primary_decision"] = "held_out_failed"

        self.assertFalse(diffreports._valid_choiceaudit(self._tampered(mutate)))

    def test_extra_payload_key_rejected(self):
        def mutate(p):
            p["extra"] = 1

        self.assertFalse(diffreports._valid_choiceaudit(self._tampered(mutate)))

    def test_record_order_swap_rejected(self):
        def mutate(p):
            records = p["candidates"]["chatml"]["records"]
            records[0], records[1] = records[1], records[0]

        self.assertFalse(diffreports._valid_choiceaudit(self._tampered(mutate)))

    def test_raw_float_cell_rejected(self):
        forged = _wide_loser_tamper(self.payload)
        self.assertFalse(diffreports._valid_choiceaudit(forged))

    def test_diff_foreign_backend_flags_drift(self):
        _, other = _run_stage(_CharBackend(mode="flat"))
        self.assertTrue(diffreports._valid_choiceaudit(other))
        diff = diffreports.diff_choiceaudit(self.payload, other)
        self.assertEqual(diff["available"], "both")
        self.assertTrue(diff["backend_drift"])
        self.assertTrue(diff["changed"])

    def test_diff_unavailable_side(self):
        diff = diffreports.diff_choiceaudit(
            self.payload, {"available": False, "detail": "cancelled"}
        )
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])


# ---------------------------------------------------------------------------
# Mutation gates (tempdir, package-namespace exec -- cycle-67 pattern)
# ---------------------------------------------------------------------------


def _run_gate(mutant_name, replacements, scenario):
    source = (REPO_ROOT / "model_protocol_lab" / "diffreports.py").read_text(encoding="utf-8")
    for original, mutant in replacements:
        assert source.count(original) == 1, f"gate anchor not unique: {original[:60]}"
        source = source.replace(original, mutant)
    module_name = f"model_protocol_lab.diffreports_mut_{mutant_name}"
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        path = os.path.join(tmp, f"diffreports_mut_{mutant_name}.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            scenario(module)
        finally:
            sys.modules.pop(module_name, None)


class TestMutationGates(unittest.TestCase):
    def test_f32_representability_gate(self):
        honest = _valid_payload()
        forged = _wide_loser_tamper(honest)

        def scenario(module):
            self.assertTrue(module._valid_choiceaudit(honest))
            self.assertFalse(module._valid_choiceaudit(forged))

        scenario(diffreports)  # non-vacuous on the real validator
        with self.assertRaises(AssertionError):
            _run_gate(
                "ca_nof32",
                [("or _ca_f32(value) != value", "or False")],
                scenario,
            )

    def test_tie_verdict_gate(self):
        # Flat logits score strictly by length; fixture t0's true
        # answer "ab" and its donor "cd" share the minimum length,
        # so the total ranking ties them at the maximum exactly.
        cases = [
            _fx("t0", "ab", "ga"),
            _fx("t1", "cd", "gb"),
            _fx("t2", "efg", "gb"),
            _fx("t3", "hij", "gb"),
        ]
        _, payload = _run_stage(_CharBackend(mode="flat"), cases=cases)
        assert payload["available"]
        tied = any(
            agg["accuracy"]["total"]["tied"] > 0
            for agg in payload["candidates"].values()
            if agg.get("status") == "available"
        )
        self.assertTrue(tied)

        def scenario(module):
            self.assertTrue(module._valid_choiceaudit(payload))

        scenario(diffreports)
        with self.assertRaises(AssertionError):
            _run_gate(
                "ca_tie",
                [("if values[0] < top:", "if values[0] <= top:")],
                scenario,
            )


# ---------------------------------------------------------------------------
# Reporting + CLI surface
# ---------------------------------------------------------------------------


class TestReportingAndCli(unittest.TestCase):
    def test_section_renders_iff_available(self):
        _, payload = _run_stage()
        html = reporting.render_html({"choice_audit": payload})
        self.assertIn("Choice-scoring audit (second opinion)", html)
        self.assertIn("acc_norm", html)
        html_absent = reporting.render_html({})
        self.assertNotIn("Choice-scoring audit (second opinion)", html_absent)

    def test_help_states_cost_and_labelling(self):
        import argparse

        parser = cli.build_parser()
        subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        run_parser = subparsers.choices["run"]
        action = next(a for a in run_parser._actions if a.dest == "choice_audit")
        self.assertIn("9,600", action.help)
        self.assertIn("acc_norm", action.help)
        self.assertIn("second opinion", action.help)

    def test_wrapper_gates_on_incomplete_search(self):
        backend = _CharBackend()
        cases = _stage_cases()
        # Failed decision: the audit must not run at all.
        result = {"decision": "failed", "search": [], "selected_candidate": None}
        cli._run_choiceaudit_probe(backend, _FORMATS, result, cases, cancel=None)
        self.assertEqual(
            result["choice_audit"],
            {"available": False, "detail": "native_search_incomplete"},
        )
        # Truncated search (fewer entries than formats): same gate.
        result = {
            "decision": "no_candidate",
            "search": _search_entries(cases)[:1],
            "selected_candidate": None,
        }
        cli._run_choiceaudit_probe(backend, _FORMATS, result, cases, cancel=None)
        self.assertEqual(result["choice_audit"]["detail"], "native_search_incomplete")

    def test_wrapper_never_raises(self):
        class _ExplodingBackend:
            @property
            def fingerprint(self):
                raise RuntimeError("boom")

        result = {
            "decision": "no_candidate",
            "search": _search_entries(_stage_cases()),
            "selected_candidate": None,
        }
        cli._run_choiceaudit_probe(
            _ExplodingBackend(), _FORMATS, result, _stage_cases(), cancel=None
        )
        self.assertEqual(
            result["choice_audit"],
            {"available": False, "detail": "choiceaudit_unavailable"},
        )

    def test_flag_default_and_set(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "run",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "llama3",
                "--boundary-ids",
                "1",
                "--output",
                "o",
            ]
        )
        self.assertFalse(args.choice_audit)
        args = parser.parse_args(
            [
                "run",
                "--choice-audit",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "llama3",
                "--boundary-ids",
                "1",
                "--output",
                "o",
            ]
        )
        self.assertTrue(args.choice_audit)


# ---------------------------------------------------------------------------
# Gap pass: every validator guard branch exercised (cycle-174 pattern --
# the full-run coverage.json named the uncovered reject branches; each
# tamper below breaks exactly one structural identity at a time)
# ---------------------------------------------------------------------------


def _excluded_cases():
    return _stage_cases() + (_fx("f8", "  ", "misc"),)


def _special_payload(kind):
    if kind == "excluded":
        _, payload = _run_stage(cases=_excluded_cases())
    elif kind == "unavailable_candidate":
        cases = _stage_cases()
        entries = _search_entries(cases)
        entries[1] = {"format": {"name": "plain"}, "summary": {}}
        _, payload = _run_stage(cases=cases, entries=entries)
    elif kind == "encode_fail":
        _, payload = _run_stage(_EncodeFailBackend())
    elif kind == "mangle":
        _, payload = _run_stage(_MangleDecodeBackend())
    elif kind == "flat":
        _, payload = _run_stage(_CharBackend(mode="flat"))
    elif kind == "failed_obs":
        cases = _stage_cases()
        entries = _search_entries(cases)
        for entry in entries:
            for obs in entry["observations"]:
                obs["status"] = "failed"
                obs["exact_match"] = False
        _, payload = _run_stage(cases=cases, entries=entries)
    elif kind == "boundary_miss":
        cases = _stage_cases()
        entries = _search_entries(cases, exact_ids=())
        for entry in entries:
            for obs in entry["observations"]:
                obs["boundary_found"] = False
        _, payload = _run_stage(cases=cases, entries=entries)
    elif kind == "gen_all_fail":
        cases = _stage_cases()
        entries = _search_entries(cases, exact_ids=())
        _, payload = _run_stage(cases=cases, entries=entries)
    else:  # pragma: no cover - test-internal dispatch
        raise ValueError(kind)
    assert payload["available"], kind
    return payload


class TestValidatorGuardBattery(unittest.TestCase):
    """One break at a time against the mirror's structural guards."""

    @classmethod
    def setUpClass(cls):
        cls.payload = _valid_payload()
        cls.excluded = _special_payload("excluded")
        cls.unavail = _special_payload("unavailable_candidate")
        cls.encode_fail = _special_payload("encode_fail")

    def _tampered(self, payload, mutate):
        import copy

        payload = copy.deepcopy(payload)
        mutate(payload)
        return payload

    def _assert_invalid(self, payload, mutate, base=None):
        self.assertFalse(
            diffreports._valid_choiceaudit(self._tampered(base or self.payload, mutate))
        )

    def test_excluded_payload_is_valid(self):
        self.assertTrue(diffreports._valid_choiceaudit(self.excluded))
        entry = next(e for e in self.excluded["fixtures"] if e["fixture"] == "f8")
        self.assertEqual(entry["excluded"], "blank_expected")

    def test_block_guards(self):
        cases = [
            ("entry_extra_key", lambda p: p["fixtures"][0].__setitem__("x", 1)),
            ("entry_group_type", lambda p: p["fixtures"][0].__setitem__("group", 7)),
            (
                "duplicate_fixture_id",
                lambda p: p["fixtures"][1].__setitem__("fixture", p["fixtures"][0]["fixture"]),
            ),
            ("excluded_value", lambda p: p["fixtures"][0].__setitem__("excluded", "weird")),
            ("numeric_type", lambda p: p["fixtures"][0].__setitem__("numeric", "yes")),
            (
                "ndistinct_mismatch",
                lambda p: p["fixtures"][0].__setitem__("n_distinct_choices", 99),
            ),
            (
                "constructible_flag",
                lambda p: p["fixtures"][0].__setitem__("constructible", False),
            ),
            (
                "choice_extra_key",
                lambda p: p["fixtures"][0]["choices"][0].__setitem__("x", 1),
            ),
            (
                "choice_group_type",
                lambda p: p["fixtures"][0]["choices"][0].__setitem__("group", 3),
            ),
            (
                "digest_format",
                lambda p: p["fixtures"][0]["choices"][1].__setitem__("digest", "zz"),
            ),
            (
                "true_source",
                lambda p: p["fixtures"][0]["choices"][0].__setitem__("source_fixture", "zzz"),
            ),
            (
                "donor_source_type",
                lambda p: p["fixtures"][0]["choices"][1].__setitem__("source_fixture", None),
            ),
            (
                "donor_unknown",
                lambda p: p["fixtures"][0]["choices"][1].__setitem__(
                    "source_fixture", "no_such_fixture"
                ),
            ),
            (
                "donor_group_mismatch",
                lambda p: p["fixtures"][0]["choices"][1].__setitem__("group", "third_group"),
            ),
            (
                "backend_fp_type",
                lambda p: p["protocol"].__setitem__("backend_fingerprint", 42),
            ),
            ("fixtures_not_list", lambda p: p.__setitem__("fixtures", {})),
            ("candidates_empty", lambda p: p.__setitem__("candidates", {})),
            (
                "candidate_not_dict",
                lambda p: p["candidates"].__setitem__("chatml", "nope"),
            ),
            (
                "candidate_detail",
                lambda p: p["candidates"]["chatml"].__setitem__("detail", "x"),
            ),
            (
                "candidate_extra_key",
                lambda p: p["candidates"]["chatml"].__setitem__("x", 1),
            ),
            (
                "records_length",
                lambda p: p["candidates"]["chatml"]["records"].pop(),
            ),
            (
                "method_extra_key",
                lambda p: p["method_comparison"].__setitem__("x", 1),
            ),
        ]
        for name, mutate in cases:
            with self.subTest(tamper=name):
                self._assert_invalid(self.payload, mutate)

    def test_excluded_entry_constructible_guard(self):
        def mutate(p):
            entry = next(e for e in p["fixtures"] if e["fixture"] == "f8")
            entry["constructible"] = True

        self._assert_invalid(self.excluded, mutate, base=self.excluded)

    def test_unavailable_candidate_shape_guard(self):
        def mutate(p):
            p["candidates"]["plain"]["x"] = 1

        self._assert_invalid(self.unavail, mutate, base=self.unavail)

    def test_record_guards(self):
        record = lambda p: p["candidates"]["chatml"]["records"][0]  # noqa: E731
        cases = [
            ("record_extra_key", lambda p: record(p).__setitem__("x", 1)),
            ("obs_status_type", lambda p: record(p).__setitem__("obs_status", 5)),
            ("exact_type", lambda p: record(p).__setitem__("exact_match", "yes")),
            ("prompt_reason_type", lambda p: record(p).__setitem__("prompt_reason", 5)),
            ("cells_type", lambda p: record(p).__setitem__("cells", {})),
            (
                "prompt_with_cells",
                lambda p: record(p).__setitem__("prompt_reason", "decode_failed"),
            ),
            (
                "cells_length",
                lambda p: record(p).__setitem__("cells", record(p)["cells"][:3]),
            ),
        ]
        for name, mutate in cases:
            with self.subTest(tamper=name):
                self._assert_invalid(self.payload, mutate)

    def test_cell_guards(self):
        cell = lambda p: p["candidates"]["chatml"]["records"][0]["cells"][0]  # noqa: E731
        cases = [
            ("cell_extra_key", lambda p: cell(p).__setitem__("x", 1)),
            ("cell_tokens_type", lambda p: cell(p).__setitem__("tokens", "4")),
            ("ok_cell_reason", lambda p: cell(p).__setitem__("reason", "weird")),
            ("ok_cell_tokens_zero", lambda p: cell(p).__setitem__("tokens", 0)),
            ("unknown_status", lambda p: cell(p).__setitem__("status", "bogus")),
            ("ok_empty_status_flip", lambda p: cell(p).__setitem__("status", "ok_empty")),
        ]
        for name, mutate in cases:
            with self.subTest(tamper=name):
                self._assert_invalid(self.payload, mutate)

    def test_ok_empty_malformed_cell_guard(self):
        def mutate(p):
            p["candidates"]["chatml"]["records"][0]["cells"][0] = {
                "status": "ok_empty",
                "reason": None,
                "total": 0.5,
                "mean": None,
                "tokens": 0,
                "bytes": 0,
            }

        self._assert_invalid(self.payload, mutate)

    def test_unavailable_cell_guards(self):
        def first_unavailable(p):
            for record in p["candidates"]["chatml"]["records"]:
                for cell in record["cells"]:
                    if cell["status"] == "unavailable":
                        return cell
            raise AssertionError("no unavailable cell")

        cases = [
            ("reason_type", lambda c: c.__setitem__("reason", None)),
            ("total_present", lambda c: c.__setitem__("total", -1.0)),
            ("tokens_negative", lambda c: c.__setitem__("tokens", -1)),
        ]
        for name, mutate_cell in cases:
            with self.subTest(tamper=name):
                self._assert_invalid(
                    self.encode_fail,
                    lambda p, mc=mutate_cell: mc(first_unavailable(p)),
                    base=self.encode_fail,
                )

    def test_records_bytes_cap_guard(self):
        def mutate(p):
            record = p["candidates"]["chatml"]["records"][0]
            record["prompt_reason"] = "x" * 13_000_000
            record["cells"] = []

        self._assert_invalid(self.payload, mutate)

    def test_total_records_cap_guard(self):
        import copy

        payload = copy.deepcopy(self.payload)
        entry = payload["candidates"]["chatml"]
        payload["candidates"] = {f"cand{i:03d}": copy.deepcopy(entry) for i in range(301)}
        self.assertFalse(diffreports._valid_choiceaudit(payload))

    def test_valid_wrapper_guards(self):
        import copy

        payload = copy.deepcopy(self.payload)
        payload["detail"] = "ok"
        self.assertFalse(diffreports._valid_choiceaudit(payload))
        payload = copy.deepcopy(self.payload)
        payload["n_candidates"] = 99
        self.assertFalse(diffreports._valid_choiceaudit(payload))
        payload = copy.deepcopy(self.payload)
        payload["n_fixtures"] = 99
        self.assertFalse(diffreports._valid_choiceaudit(payload))
        self.assertFalse(diffreports._valid_choiceaudit({"available": False}))
        self.assertFalse(diffreports._valid_choiceaudit("not a payload"))

    def test_mirror_evaluable_nonconstructible_direct(self):
        record = {"cells": [_cell(-1.0) for _ in range(4)]}
        self.assertFalse(diffreports._ca_record_evaluable(record, {"constructible": False}))
        self.assertTrue(diffreports._ca_record_evaluable(record, {"constructible": True}))


class TestStageGuards(unittest.TestCase):
    def test_formats_and_cases_must_be_nonempty_tuples(self):
        backend = _CharBackend()
        cases = _stage_cases()
        entries = _search_entries(cases)
        with self.assertRaises(ValueError):
            choiceaudit_stage(backend, list(_FORMATS), cases, entries)
        with self.assertRaises(ValueError):
            choiceaudit_stage(backend, _FORMATS, list(cases), entries)
        with self.assertRaises(ValueError):
            choiceaudit_stage(backend, (), cases, entries)

    def test_junk_search_entries_are_skipped(self):
        cases = _stage_cases()
        entries = ["junk", {"format": {}}, {"no_format": 1}] + _search_entries(cases)
        stage, payload = _run_stage(cases=cases, entries=entries)
        self.assertTrue(stage["available"])
        self.assertTrue(payload["available"])

    def test_observation_not_dict_makes_candidate_unavailable(self):
        cases = _stage_cases()
        entries = _search_entries(cases)
        entries[0]["observations"][0] = "junk"
        stage, _ = _run_stage(cases=cases, entries=entries)
        self.assertEqual(stage["candidates"]["chatml"]["status"], "unavailable")

    def test_missing_case_observation_makes_candidate_unavailable(self):
        cases = _stage_cases()
        entries = _search_entries(cases)
        entries[0]["observations"] = entries[0]["observations"][:-1]
        stage, _ = _run_stage(cases=cases, entries=entries)
        self.assertEqual(stage["candidates"]["chatml"]["status"], "unavailable")

    def test_aggregate_ineligible_summary_makes_candidate_unavailable(self):
        cases = _stage_cases()
        entries = _search_entries(cases)
        entries[0]["summary"] = {"aggregate_eligible": False}
        stage, _ = _run_stage(cases=cases, entries=entries)
        self.assertEqual(stage["candidates"]["chatml"]["status"], "unavailable")

    def test_records_bytes_cap_in_stage(self):
        formats = tuple(
            Format(name=f"fmt{i}", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>")
            for i in range(24)
        )
        cases = tuple(_fx(f"f{'x' * 6000}{i}", f"v{i}", f"g{i}") for i in range(100))
        entries = []
        for fmt in formats:
            entries.append(
                {
                    "format": {"name": fmt.name},
                    "observations": [
                        {
                            "fixture": case.id,
                            "status": "completed",
                            "exact_match": True,
                            "boundary_found": True,
                            "input_ids": [1, 2, 3],
                        }
                        for case in cases
                    ],
                    "summary": {"aggregate_eligible": True},
                }
            )
        stage = choiceaudit_stage(_CharBackend(), formats, cases, entries, cancel=None)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], DETAIL_BUDGET_EXCEEDED)

    def test_failed_observations_payload_valid_and_counted(self):
        payload = _special_payload("failed_obs")
        self.assertTrue(diffreports._valid_choiceaudit(payload))
        candidate = payload["candidates"]["chatml"]
        self.assertEqual(candidate["n_generation_missing"], 8)
        self.assertIsNone(payload["discordance"]["falsified"])

    def test_boundary_miss_payload_valid_and_split(self):
        payload = _special_payload("boundary_miss")
        self.assertTrue(diffreports._valid_choiceaudit(payload))
        candidate = payload["candidates"]["chatml"]
        self.assertGreater(candidate["discordance"]["gen_fail_boundary_not_found"], 0)

    def test_agree_method_payload_valid(self):
        # The keyed backend gives chatml a deterministic unique
        # top (2 correct vs 0 on the common population).
        stage, abstain_payload = _run_stage(_KeyedBackend())
        self.assertEqual(abstain_payload["candidates"]["chatml"]["accuracy"]["total"]["correct"], 2)
        self.assertEqual(abstain_payload["candidates"]["plain"]["accuracy"]["total"]["correct"], 0)
        top = abstain_payload["method_comparison"]["choice_top_candidate"]
        self.assertEqual(top, "chatml")
        payload = choiceaudit_analysis(
            stage, "compatible_on_fixture_set", top, tuple(f.name for f in _FORMATS)
        )
        self.assertEqual(payload["method_comparison"]["agreement"], "agree")
        self.assertTrue(diffreports._valid_choiceaudit(payload))

    def test_all_candidates_unavailable_payload_is_not_valid(self):
        # The validator's stance (placement precedent): a payload
        # with no available candidate carries no recomputable
        # aggregates and is not a valid diff payload.
        cases = _stage_cases()
        stage = choiceaudit_stage(_CharBackend(), _FORMATS, cases, [], cancel=None)
        payload = choiceaudit_analysis(stage, "no_candidate", None, tuple(f.name for f in _FORMATS))
        self.assertTrue(payload["available"])
        self.assertFalse(diffreports._valid_choiceaudit(payload))


class TestDiffDrift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = _valid_payload()

    def test_candidate_status_drift(self):
        other = _special_payload("unavailable_candidate")
        self.assertTrue(diffreports._valid_choiceaudit(other))
        diff = diffreports.diff_choiceaudit(self.payload, other)
        self.assertEqual(diff["available"], "both")
        status_drifts = [d for d in diff["candidate_drift"] if d["metric"] == "status"]
        self.assertEqual(len(status_drifts), 1)
        self.assertEqual(status_drifts[0]["candidate"], "plain")
        self.assertTrue(diff["changed"])

    def test_candidate_metric_drift(self):
        # Keyed vs echo: plain's total correct count differs
        # deterministically (0 vs 2).
        _, keyed = _run_stage(_KeyedBackend())
        self.assertTrue(diffreports._valid_choiceaudit(keyed))
        diff = diffreports.diff_choiceaudit(self.payload, keyed)
        self.assertEqual(diff["available"], "both")
        metrics = {(d["candidate"], d["metric"]) for d in diff["candidate_drift"]}
        self.assertIn(("plain", "accuracy_total_correct"), metrics)

    def test_method_drift(self):
        stage, _ = _run_stage(_KeyedBackend())
        top = "chatml"
        agree_payload = choiceaudit_analysis(
            stage, "compatible_on_fixture_set", top, tuple(f.name for f in _FORMATS)
        )
        diff = diffreports.diff_choiceaudit(self.payload, agree_payload)
        self.assertEqual(diff["available"], "both")
        self.assertTrue(diff["method_drift"])
        self.assertTrue(diff["changed"])

    def test_discordance_drift(self):
        other = _special_payload("failed_obs")
        diff = diffreports.diff_choiceaudit(self.payload, other)
        self.assertEqual(diff["available"], "both")
        self.assertTrue(diff["method_drift"])


class TestReportingStates(unittest.TestCase):
    def test_malformed_method_block_renders_empty(self):
        html = reporting.render_html(
            {"choice_audit": {"available": True, "candidates": {}, "method_comparison": None}}
        )
        self.assertNotIn("Choice-scoring audit", html)

    def test_unavailable_payload_renders_empty(self):
        html = reporting.render_html({"choice_audit": {"available": False, "detail": "cancelled"}})
        self.assertNotIn("Choice-scoring audit", html)

    def test_no_evaluable_candidate_renders_na(self):
        payload = _special_payload("mangle")
        html = reporting.render_html({"choice_audit": payload})
        self.assertIn("n/a", html)
        self.assertIn("Evaluable 0 of", html)

    def test_unavailable_candidate_row(self):
        payload = _special_payload("unavailable_candidate")
        html = reporting.render_html({"choice_audit": payload})
        self.assertIn("unavailable", html)

    def test_non_dict_candidate_entry_skipped(self):
        payload = _valid_payload()
        payload["candidates"]["ghost"] = "nope"
        html = reporting.render_html({"choice_audit": payload})
        self.assertIn("Choice-scoring audit (second opinion)", html)
        self.assertNotIn("ghost", html)

    def test_method_lines_for_each_agreement_state(self):
        stage, base = _run_stage(_KeyedBackend())
        top = base["method_comparison"]["choice_top_candidate"]
        self.assertEqual(top, "chatml")
        other = next(n for n in ("chatml", "plain") if n != top)
        agree = choiceaudit_analysis(
            stage, "compatible_on_fixture_set", top, tuple(f.name for f in _FORMATS)
        )
        self.assertIn(
            "agrees with the primary winner",
            reporting.render_html({"choice_audit": agree}),
        )
        disagree = choiceaudit_analysis(
            stage, "compatible_on_fixture_set", other, tuple(f.name for f in _FORMATS)
        )
        self.assertIn(
            "disagrees with the primary winner",
            reporting.render_html({"choice_audit": disagree}),
        )
        _, flat_payload = _run_stage(_CharBackend(mode="flat"))
        tied_stage, _ = _run_stage(_CharBackend(mode="flat"))
        tied = choiceaudit_analysis(
            tied_stage,
            "compatible_on_fixture_set",
            "chatml",
            tuple(f.name for f in _FORMATS),
        )
        self.assertEqual(tied["method_comparison"]["agreement"], "choice_tied")
        self.assertIn("tied between", reporting.render_html({"choice_audit": tied}))
        small_cases = _stage_cases()[:4]
        small_stage, _ = _run_stage(cases=small_cases)
        withheld = choiceaudit_analysis(
            small_stage,
            "compatible_on_fixture_set",
            "chatml",
            tuple(f.name for f in _FORMATS),
        )
        self.assertEqual(withheld["method_comparison"]["agreement"], "withheld")
        self.assertIn("withheld", reporting.render_html({"choice_audit": withheld}))

    def test_falsified_false_line(self):
        payload = _special_payload("gen_all_fail")
        self.assertEqual(payload["discordance"]["falsified"], False)
        html = reporting.render_html({"choice_audit": payload})
        self.assertIn("the scoring method changes answers", html)


class TestValidatorGuardBattery2(unittest.TestCase):
    """The remaining mirror branches: canonical ok_empty counting,
    choice_tied / disagree recompute, and the diff 'new' side."""

    def test_canonical_ok_empty_cell_counted_and_rejected(self):
        # A canonically-shaped ok_empty cell passes the structural
        # cell guard, so the mirror aggregate itself must count it
        # (under its own status) and reject the payload whose
        # stored aggregates assumed an evaluable fixture.
        import copy

        payload = copy.deepcopy(_valid_payload())
        payload["candidates"]["chatml"]["records"][0]["cells"][0] = {
            "status": "ok_empty",
            "reason": None,
            "total": 0.0,
            "mean": None,
            "tokens": 0,
            "bytes": 0,
        }
        self.assertFalse(diffreports._valid_choiceaudit(payload))

    def test_choice_tied_payload_valid(self):
        stage, _ = _run_stage(_CharBackend(mode="flat"))
        payload = choiceaudit_analysis(
            stage, "compatible_on_fixture_set", "chatml", tuple(f.name for f in _FORMATS)
        )
        self.assertEqual(payload["method_comparison"]["agreement"], "choice_tied")
        self.assertTrue(diffreports._valid_choiceaudit(payload))

    def test_disagree_payload_valid(self):
        stage, _ = _run_stage(_KeyedBackend())
        payload = choiceaudit_analysis(
            stage, "compatible_on_fixture_set", "plain", tuple(f.name for f in _FORMATS)
        )
        self.assertEqual(payload["method_comparison"]["agreement"], "disagree")
        self.assertTrue(diffreports._valid_choiceaudit(payload))

    def test_fixture_set_digest_fails_closed_on_surrogate_id(self):
        # The mirror must never raise on a hostile id: an
        # unencodable fixture id yields None, so the expected
        # protocol mismatches any recorded payload (fail closed).
        self.assertIsNone(diffreports._ca_fixture_set_sha256(["fx-\ud800"]))
        expected = diffreports._ca_protocol_object(["fx-\ud800"], [], "fp")
        self.assertIsNone(expected["fixture_set_sha256"])

    def test_recompute_rejects_surrogate_fixture_id_without_raising(self):
        # Rename one fixture id consistently (every occurrence in
        # the payload), so all cross-references stay coherent; the
        # payload must then be rejected by the digest recompute --
        # and before the guard, this raised UnicodeEncodeError.
        import copy

        payload = copy.deepcopy(_valid_payload())
        target = payload["fixtures"][0]["fixture"]
        renamed = target + "\ud800"

        def _rename(node):
            if isinstance(node, dict):
                return {
                    (_rename(k) if isinstance(k, str) else k): _rename(v) for k, v in node.items()
                }
            if isinstance(node, list):
                return [_rename(v) for v in node]
            if isinstance(node, str) and node == target:
                return renamed
            return node

        payload = _rename(payload)
        self.assertFalse(diffreports._valid_choiceaudit(payload))

    def test_diff_new_side_only(self):
        diff = diffreports.diff_choiceaudit(
            {"available": False, "detail": "cancelled"}, _valid_payload()
        )
        self.assertEqual(diff["available"], "new")
        self.assertFalse(diff["changed"])
        diff = diffreports.diff_choiceaudit({"available": False}, {"available": False})
        self.assertEqual(diff["available"], "neither")


class TestReportingFalsifiedTrue(unittest.TestCase):
    def test_falsified_true_line(self):
        # Per-format generation facts aligned with the keyed
        # backend's choice verdicts (chatml correct exactly on
        # f0/f1; plain correct nowhere): zero discordance over a
        # non-empty paired population.
        cases = _stage_cases()
        entries = []
        for fmt in _FORMATS:
            exact_ids = ("f0", "f1") if fmt.name == "chatml" else ()
            prompt_ids = [1, 2, 3] if fmt.name == "chatml" else [9, 9, 9]
            entries.append(
                {
                    "format": {"name": fmt.name},
                    "observations": [
                        {
                            "fixture": case.id,
                            "status": "completed",
                            "exact_match": case.id in exact_ids,
                            "boundary_found": True,
                            "input_ids": prompt_ids,
                        }
                        for case in cases
                    ],
                    "summary": {"aggregate_eligible": True},
                }
            )
        _, payload = _run_stage(_KeyedBackend(), cases=cases, entries=entries)
        self.assertEqual(payload["discordance"]["falsified"], True)
        self.assertGreater(payload["discordance"]["n_cells"], 0)
        html = reporting.render_html({"choice_audit": payload})
        self.assertIn("Zero discordance on", html)
