"""Cycle 219: history-frame probe (research wave-11 rank 3;
fresh review BUILD-as-amended, coordinator #61).

Spec citations are to docs/HISTORYFRAME_DESIGN.md (the
candidate text of record is research section 6, incorporated
as design section 3) and to the fresh review's amendments
A1-A10 in the design doc's fresh-review adjudication section.

Every guard is proven engaged: the classification floors are
exercised on both sides, the A9 seam mutant is killed at
stage level (a corrupted hybrid fails the whole stage before
any generation), the A4 id-digest gate is shown to derive
from digests by forgery (a flipped digest changes the
validator's recompute and is rejected), and the A1 verbatim
inapplicability is derived from the recorded bos digests.
"""

from __future__ import annotations

import copy
import unittest

from model_protocol_lab import diffreports, experiments, reporting
from model_protocol_lab import historyframe as hf
from model_protocol_lab.catalog import CATALOG
from model_protocol_lab.core import Fixture, Format, Generation, Message

_CHATML = Format("chatml", "", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n")
_LLAMA3 = Format(
    "llama3",
    "<|begin_of_text|>",
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|eot_id|>",
)


def _fixtures(count: int = 12) -> tuple[Fixture, ...]:
    return tuple(
        Fixture(
            id=f"fx-{i:02d}",
            messages=(Message("user", "What color is the marker?"),),
            expected="BLUE",
            group="g",
        )
        for i in range(count)
    )


class _WordBackend:
    """Word-token scripted backend.

    Modes: ``perfect`` (the expected answer with boundary on
    every cell), ``frame_drop`` (a chatml candidate's
    foreign_llama3 cells -- llama3-framed history segment
    before a chatml tail -- answer GREEN, so the contrast
    fires), ``no_boundary`` (the expected answer with no
    boundary token), ``explode`` (generate raises).
    """

    fingerprint = "sha256:test-history-frame"

    def __init__(self, mode: str = "perfect") -> None:
        self.mode = mode
        self.calls = 0
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
        ids = []
        for token in text.replace("\n", " \n ").split(" "):
            if not token:
                continue
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                self.rev[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return tuple(ids)

    def decode(self, ids):
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        self.calls += 1
        if self.mode == "explode":
            raise RuntimeError("boom")
        text = self.decode(list(input_ids))
        answer = "BLUE"
        if (
            self.mode == "frame_drop"
            and "<|start_header_id|>user<|end_header_id|>" in text
            and "<|im_start|>user" in text
        ):
            answer = "GREEN"
        boundary = self.encode("endmark")[0]
        out = list(self.encode(answer))
        if self.mode != "no_boundary":
            out.append(boundary)
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(
    backend=None,
    formats=(_CHATML, _LLAMA3),
    fixtures=None,
    seed=0,
    primary=None,
    **stage_kwargs,
):
    backend = backend if backend is not None else _WordBackend()
    fixtures = fixtures if fixtures is not None else _fixtures()
    stage_kwargs.setdefault("max_context", 8192)
    stage = experiments.history_frame_stage(
        backend,
        formats,
        fixtures,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        **stage_kwargs,
    )
    return experiments.history_frame_analysis(stage, tuple(f.name for f in formats), primary)


def _cell(verdict: bool, boundary: bool = True, digest: str = "BLUE"):
    return {
        "verdict": verdict,
        "boundary_found": boundary,
        "boundary_text_match": None,
        "input_tokens": 40,
        "answer_digest": hf.digest_text(digest),
    }


def _record(
    fixture: str,
    *,
    own: dict | None,
    foreign: dict | None,
    render_divergent: bool = True,
    ids_identical: bool = False,
    own_bos: str = "<|begin_of_text|>",
    donor_bos: str = "",
) -> dict:
    """One synthetic record for aggregate-level tests. Cells
    for the two arms not under test are completed passes."""
    own_digest = hf.digest_text(f"ids-{fixture}-own")
    foreign_digest = own_digest if ids_identical else hf.digest_text(f"ids-{fixture}-fx")
    seg_own = hf.digest_text(f"seg-{fixture}-own")
    seg_fx = seg_own if not render_divergent else hf.digest_text(f"seg-{fixture}-fx")
    cells = {
        "own": own,
        "foreign_chatml": foreign,
        "foreign_llama3": _cell(True),
        "verbatim_chatml": _cell(True),
    }
    return {
        "fixture": fixture,
        "expected_digest": hf.digest_text("BLUE"),
        "own_bos_digest": hf.digest_text(own_bos),
        "donor_bos_digest": hf.digest_text(donor_bos),
        "segment_digests": {
            "own": seg_own,
            "foreign_chatml": seg_fx,
            "foreign_llama3": hf.digest_text(f"seg-{fixture}-fl"),
            "verbatim_chatml": seg_fx,
        },
        "hybrid_digests": {arm: hf.digest_text(f"hyb-{fixture}-{arm}") for arm in hf.ARM_KEYS},
        "arm_tokens": dict.fromkeys(hf.ARM_KEYS, 40),
        "id_digests": {
            "own": own_digest,
            "foreign_chatml": foreign_digest,
            "foreign_llama3": hf.digest_text(f"ids-{fixture}-fl"),
            "verbatim_chatml": hf.digest_text(f"ids-{fixture}-vb"),
        },
        "cells": cells,
        "cell_details": {
            arm: ("" if cells[arm] is not None else "arm_evaluation_failed") for arm in hf.ARM_KEYS
        },
    }


class FrozenMaterialTests(unittest.TestCase):
    """Design section 6.1-6.4 + A5: frozen arms, donors,
    prepend, and construction pins."""

    def test_arm_and_donor_freeze(self):
        self.assertEqual(
            hf.ARM_KEYS, ("own", "foreign_chatml", "foreign_llama3", "verbatim_chatml")
        )
        self.assertEqual(
            hf.DONOR_FAMILIES, {"foreign_chatml": "chatml", "foreign_llama3": "llama3"}
        )
        self.assertEqual(hf.HISTORY_TURNS, 6)
        self.assertEqual(hf.donor_format("foreign_chatml").bos, "")
        self.assertEqual(hf.donor_format("foreign_llama3").bos, "<|begin_of_text|>")

    def test_stage_fixtures_rewrap_original_ids(self):
        # A5: the built fixtures' namespaced ids
        # (synthhistory/history_6/<id>) are never stage ids --
        # they would change every cell's request seed (A10).
        original = _fixtures(2)
        staged = hf.stage_fixtures(original)
        self.assertEqual([c.id for c in staged], ["fx-00", "fx-01"])
        self.assertTrue(all(c.group == hf.GROUP for c in staged))
        self.assertTrue(all(c.expected == "BLUE" for c in staged))
        self.assertTrue(all(len(c.messages) == 7 for c in staged))

    def test_protocol_binds_donor_pair_and_fixture_set(self):
        protocol = hf.protocol_object(3, "fp", ["a", "b"])
        self.assertEqual(protocol["donor_families"], hf.DONOR_FAMILIES)
        self.assertEqual(protocol["history_turns"], 6)
        self.assertEqual(protocol["fixture_ids"], ["a", "b"])
        self.assertEqual(protocol["min_n"], 10)
        self.assertEqual(protocol["min_d"], 6)
        self.assertEqual(protocol["generation_budget"], 9600)
        self.assertEqual(
            hf.protocol_fingerprint(3, "fp", ["a", "b"]),
            hf.protocol_fingerprint(3, "fp", ["a", "b"]),
        )
        self.assertNotEqual(
            hf.protocol_fingerprint(3, "fp", ["a", "b"]),
            hf.protocol_fingerprint(3, "fp", ["a", "c"]),
        )


class SpliceIdentityTests(unittest.TestCase):
    """Design section 6.3 + A9: the segment/hybrid semantics
    reproduce Format.render under A == B and satisfy the
    per-arm slice-swap identity; the seam mutant dies."""

    def test_hybrid_reproduces_render_for_catalog_and_repairs(self):
        from model_protocol_lab.core import local_repairs

        formats = list(CATALOG.values())
        for base in list(CATALOG.values()):
            formats.extend(local_repairs(base))
        staged = hf.stage_fixtures(_fixtures(2))
        checked = 0
        for fmt in formats:
            for case in staged:
                native = fmt.render(case.messages)
                self.assertEqual(hf.hybrid_text(fmt, fmt, case.messages), native)
                seg_own = hf.history_segment(fmt, case.messages)
                for arm in hf.CONTRASTS:
                    donor = hf.donor_format(arm)
                    self.assertTrue(
                        hf.slice_swap_identity(
                            native,
                            hf.hybrid_text(donor, fmt, case.messages),
                            seg_own,
                            hf.history_segment(donor, case.messages),
                            fmt.bos,
                        )
                    )
                donor = hf.donor_format("foreign_chatml")
                self.assertTrue(
                    hf.verbatim_slice_identity(
                        native,
                        hf.verbatim_text(donor, fmt, case.messages),
                        seg_own,
                        hf.history_segment(donor, case.messages),
                        fmt.bos,
                        donor.bos,
                    )
                )
                checked += 1
        self.assertGreater(checked, 200)

    def test_seam_mutant_fails_the_identity(self):
        # A9's executed mutant: insert "\n" at the seam iff the
        # history segment lacks one. The diagonal-only screen
        # (round-1 S1) passed it for 51 candidates; the
        # per-arm slice-swap identity must not.
        fmt = _LLAMA3
        case = hf.stage_fixtures(_fixtures(1))[0]
        native = fmt.render(case.messages)
        donor = hf.donor_format("foreign_chatml")
        hybrid = hf.hybrid_text(donor, fmt, case.messages)
        seam = len(fmt.bos) + len(hf.history_segment(donor, case.messages))
        mutant = hybrid[:seam] + "\n" + hybrid[seam:]
        self.assertFalse(
            hf.slice_swap_identity(
                native,
                mutant,
                hf.history_segment(fmt, case.messages),
                hf.history_segment(donor, case.messages),
                fmt.bos,
            )
        )

    def test_verbatim_removes_only_the_leading_bos(self):
        # A1: a repeat_bos candidate carries internal bos
        # occurrences; the verbatim construction removes the
        # leading one only.
        fmt = Format("rb", "<B>", "u:", "a:", "\n", repeat_bos=True)
        case = hf.stage_fixtures(_fixtures(1))[0]
        donor = hf.donor_format("foreign_llama3")
        hybrid = hf.hybrid_text(donor, fmt, case.messages)
        verbatim = hf.verbatim_text(donor, fmt, case.messages)
        self.assertTrue(verbatim.startswith(donor.bos))
        self.assertEqual(verbatim, donor.bos + hybrid[len(fmt.bos) :])
        self.assertIn("<B>", verbatim[len(donor.bos) :])


class ClassificationTests(unittest.TestCase):
    """Design section 6.6: the classification is a total
    function; floors exercised on both sides."""

    def test_classification_table(self):
        cases = [
            ((9, 9, 0), "insufficient"),  # n floor binds first
            ((10, 0, 0), "no_discordance_observed"),
            ((10, 5, 0), "below_threshold"),  # d floor: 5-0 has p 0.0625
            ((10, 6, 0), "discordant_asymmetric"),  # 6-0: p 0.03125
            ((12, 6, 6), "discordant_balanced"),
            ((20, 7, 3), "below_threshold"),  # p 0.34375
            ((16, 8, 1), "discordant_asymmetric"),  # p 0.0390625
            ((24, 3, 3), "discordant_balanced"),
        ]
        for (n, a, b), want in cases:
            self.assertEqual(hf.classify_history_frame(n, a, b), want, (n, a, b))

    def test_exact_random_fire_rates_reproduce_the_design_table(self):
        # Fresh-review exact enumeration, three parties: the
        # module's enumeration must reproduce the tabulated
        # rows (A2's corrected constants derive from these).
        for n, want in ((16, 0.0196), (24, 0.0243), (30, 0.0278), (50, 0.0296), (100, 0.0367)):
            self.assertAlmostEqual(hf.random_labeller_fire_rate(n), want, delta=0.0005)
        self.assertEqual(hf.random_labeller_fire_rate(9), 0.0)

    def test_familywise_is_computed_not_frozen(self):
        # A2: at uniform n=100 over 48 decisions the computed
        # expectation is 1.76 / P 0.83 -- the printed 1.44 /
        # 0.77 of the candidate text was 48 x a rounded rate.
        rates = [hf.random_labeller_fire_rate(100)] * 48
        fw = hf.familywise_expectation(rates)
        self.assertEqual(fw["decisions"], 48)
        self.assertAlmostEqual(fw["expected_false_labels"], 1.76, delta=0.01)
        self.assertAlmostEqual(fw["p_at_least_one"], 0.83, delta=0.01)

    def test_n_floor_is_a_conservative_constant(self):
        # A3: the smallest interior-attainable n for a 6-0
        # split is 8 (one concordant pass); the floor stands
        # at 10 regardless -- n=8 and n=9 classify
        # insufficient even at a unanimous split.
        self.assertEqual(hf.classify_history_frame(8, 6, 0), "insufficient")
        self.assertEqual(hf.classify_history_frame(9, 6, 0), "insufficient")


class DecompositionTests(unittest.TestCase):
    """Design section 6.6 decomposition of discordant pairs."""

    def test_decomposition_classes(self):
        own = _cell(True, True, "BLUE")
        same_digest_no_boundary = _cell(False, False, "BLUE")
        other_digest_boundary = _cell(False, True, "GREEN")
        other_digest_no_boundary = _cell(False, False, "GREEN")
        self.assertEqual(hf.decompose_pair(own, same_digest_no_boundary), "boundary_only")
        self.assertEqual(hf.decompose_pair(own, other_digest_boundary), "exact_only")
        self.assertEqual(hf.decompose_pair(own, other_digest_no_boundary), "mixed")


class AggregateTests(unittest.TestCase):
    """Design sections 6.5/6.6 + A1/A4: eligibility and
    divergence accounting are derived from the recorded
    digests."""

    def _records(self, count: int, **kwargs):
        return [_record(f"fx-{i:02d}", **kwargs) for i in range(count)]

    def test_render_identical_contrast_is_inapplicable(self):
        records = self._records(12, own=_cell(True), foreign=_cell(True), render_divergent=False)
        block = hf.aggregate_candidate(records)["contrasts"]["foreign_chatml"]
        self.assertFalse(block["applicable"])
        self.assertIsNone(block["classification"])
        self.assertEqual(block["n_render_identical"], 12)

    def test_ids_identical_cases_are_excluded_and_flips_counted(self):
        records = self._records(
            12, own=_cell(True), foreign=_cell(False, digest="GREEN"), ids_identical=True
        )
        block = hf.aggregate_candidate(records)["contrasts"]["foreign_chatml"]
        self.assertEqual(block["n_ids_identical"], 12)
        self.assertEqual(block["n_same_ids_flips"], 12)
        self.assertEqual(block["n_paired"], 0)

    def test_equal_token_counts_do_not_fake_ids_identity(self):
        # A4: equal counts with different id digests are NOT
        # ids-identical -- the gate reads digests, not counts.
        records = self._records(12, own=_cell(True), foreign=_cell(False, digest="GREEN"))
        block = hf.aggregate_candidate(records)["contrasts"]["foreign_chatml"]
        self.assertEqual(block["n_ids_identical"], 0)
        self.assertEqual(block["n_paired"], 12)
        self.assertEqual(block["classification"], "discordant_asymmetric")
        self.assertTrue(block["falsification"])

    def test_ungated_sensitivity_counts_all_completed_pairs(self):
        records = self._records(
            12, own=_cell(True), foreign=_cell(False, digest="GREEN"), render_divergent=False
        )
        block = hf.aggregate_candidate(records)["contrasts"]["foreign_chatml"]
        self.assertEqual(block["ungated_n"], 12)
        self.assertEqual(block["ungated_own_only"], 12)

    def test_verbatim_inapplicable_when_bos_matches(self):
        # A1: own bos == chatml bos ("") -> inapplicable,
        # counted and never pooled.
        records = self._records(4, own=_cell(True), foreign=_cell(True), own_bos="", donor_bos="")
        verbatim = hf.aggregate_candidate(records)["verbatim"]
        self.assertFalse(verbatim["applicable"])
        self.assertEqual(verbatim["n_paired"], 0)

    def test_verbatim_applicable_counts_bos_share(self):
        records = self._records(4, own=_cell(True), foreign=_cell(True))
        for record in records:
            record["cells"]["verbatim_chatml"] = _cell(False, digest="GREEN")
        verbatim = hf.aggregate_candidate(records)["verbatim"]
        self.assertTrue(verbatim["applicable"])
        self.assertEqual(verbatim["n_paired"], 4)
        self.assertEqual(verbatim["twin_only"], 4)


class SelectionTests(unittest.TestCase):
    """Design section 6.6: per-arm selection on the common
    completed set + canonical-form winner headlines."""

    def _two_candidates(self):
        winner_records = [
            _record(f"fx-{i:02d}", own=_cell(True), foreign=_cell(True)) for i in range(3)
        ]
        loser_records = [
            _record(f"fx-{i:02d}", own=_cell(False, digest="GREEN"), foreign=_cell(True))
            for i in range(3)
        ]
        return {"cand-a": winner_records, "cand-b": loser_records}

    def test_arm_selection_and_winner_headline(self):
        by_candidate = self._two_candidates()
        own = hf.arm_selection(by_candidate, "own")
        foreign = hf.arm_selection(by_candidate, "foreign_chatml")
        self.assertEqual((own["decision"], own["winner"]), ("selected", "cand-a"))
        self.assertEqual(foreign["decision"], "ambiguous")
        headline = hf.winner_headline(own, foreign)
        self.assertEqual(headline["headline"], "winner_abstention_asymmetric")
        self.assertFalse(headline["winner_changed"])

    def test_winner_changed_requires_two_different_winners(self):
        baseline = {"decision": "selected", "winner": "cand-a", "withheld": False}
        other = {"decision": "selected", "winner": "cand-b", "withheld": False}
        self.assertEqual(hf.winner_headline(baseline, other)["headline"], "winner_changed")
        self.assertTrue(hf.winner_headline(baseline, other)["winner_changed"])
        withheld = {"decision": "unavailable", "winner": None, "withheld": True}
        headline = hf.winner_headline(baseline, withheld)
        self.assertEqual(headline["headline"], "winner_undefined")
        self.assertIsNone(headline["winner_changed"])

    def test_selection_withheld_on_mismatched_sets(self):
        # Canonical-form semantics: within the shared record
        # set, one candidate's completed set for the arm is
        # smaller (an uncompleted cell) -> the restricted
        # completed sets differ -> withheld.
        by_candidate = self._two_candidates()
        by_candidate["cand-b"][1]["cells"]["own"] = None
        selection = hf.arm_selection(by_candidate, "own")
        self.assertTrue(selection["withheld"])
        self.assertEqual(selection["detail"], "selection_withheld_mismatched_sets")

    def test_primary_flips_use_the_synhistory_mapping(self):
        conditions = {
            "own": {"decision": "selected", "winner": "cand-a", "withheld": False},
            "foreign_chatml": {"decision": "selected", "winner": "cand-b", "withheld": False},
            "foreign_llama3": {"decision": "unavailable", "winner": None, "withheld": True},
        }
        primary = {"decision": "compatible_on_fixture_set", "winner": "cand-a"}
        flips = hf.primary_flips(conditions, primary)
        self.assertEqual([f["arm"] for f in flips], ["foreign_chatml"])
        self.assertEqual(flips[0]["winner_to"], "cand-b")
        self.assertEqual(hf.primary_flips(conditions, None), [])


class StageTests(unittest.TestCase):
    """Design sections 6.2/6.4 + A5/A9/A10: stage behaviour
    end to end on scripted backends."""

    def test_perfect_backend_no_discordance_and_diagonal_inapplicable(self):
        payload = _payload()
        self.assertTrue(payload["available"])
        chatml = payload["candidates"]["chatml"]
        # chatml's own family IS the chatml donor: the segment
        # is render-identical, the contrast inapplicable.
        self.assertFalse(chatml["contrasts"]["foreign_chatml"]["applicable"])
        self.assertEqual(
            chatml["contrasts"]["foreign_llama3"]["classification"],
            "no_discordance_observed",
        )
        # A1: chatml's bos is "" -- verbatim inapplicable.
        self.assertFalse(chatml["verbatim"]["applicable"])
        llama3 = payload["candidates"]["llama3"]
        self.assertTrue(llama3["verbatim"]["applicable"])

    def test_frame_drop_fires_asymmetric(self):
        payload = _payload(_WordBackend("frame_drop"))
        block = payload["candidates"]["chatml"]["contrasts"]["foreign_llama3"]
        self.assertEqual((block["own_only"], block["foreign_only"]), (12, 0))
        self.assertEqual(block["classification"], "discordant_asymmetric")
        self.assertTrue(block["falsification"])

    def test_records_carry_id_digests_for_every_arm(self):
        # A4: per-arm token-id digests in every record.
        payload = _payload()
        for entry in payload["candidates"].values():
            for record in entry["records"]:
                self.assertEqual(set(record["id_digests"]), set(hf.ARM_KEYS))
                self.assertTrue(all(record["id_digests"].values()))

    def test_stage_uses_original_ids_and_frozen_group(self):
        # A5/A10: cells are requested under the fixtures'
        # original ids (request-seed identity), never the
        # synthhistory namespaced ids.
        payload = _payload()
        for entry in payload["candidates"].values():
            self.assertEqual(
                [r["fixture"] for r in entry["records"]],
                [f"fx-{i:02d}" for i in range(12)],
            )
        self.assertEqual(payload["protocol"]["fixture_ids"], [f"fx-{i:02d}" for i in range(12)])

    def test_small_panel_classifies_insufficient(self):
        payload = _payload(fixtures=_fixtures(2))
        block = payload["candidates"]["chatml"]["contrasts"]["foreign_llama3"]
        self.assertEqual(block["classification"], "insufficient")

    def test_seam_mutant_fails_the_whole_stage_before_generation(self):
        # A9 at stage level: corrupt the foreign hybrid at the
        # seam and the S1 slice-swap screen must fail the
        # stage closed with zero generations spent.
        backend = _WordBackend()
        original = hf.hybrid_text

        def mutant(hist_fmt, final_fmt, messages):
            text = original(hist_fmt, final_fmt, messages)
            if hist_fmt is not final_fmt:
                seam = len(final_fmt.bos) + len(hf.history_segment(hist_fmt, messages))
                text = text[:seam] + "\n" + text[seam:]
            return text

        hf.hybrid_text = mutant  # type: ignore[misc]
        try:
            stage = experiments.history_frame_stage(
                backend,
                (_CHATML,),
                _fixtures(2),
                boundary=tuple(backend.encode("endmark")),
                max_context=8192,
                seed=0,
            )
        finally:
            hf.hybrid_text = original  # type: ignore[misc]
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "historyframe_screen_failed")
        self.assertEqual(backend.calls, 0)

    def test_budget_exceeded_fails_closed(self):
        formats = tuple(Format(f"cand-{i:02d}", "", "u:", "a:", "\n") for i in range(25))
        backend = _WordBackend()
        stage = experiments.history_frame_stage(
            backend,
            formats,
            _fixtures(100),
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "historyframe_record_budget_exceeded")
        self.assertEqual(backend.calls, 0)

    def test_exploding_backend_yields_evaluation_failed_cells(self):
        payload = _payload(_WordBackend("explode"))
        self.assertTrue(payload["available"])
        arm = payload["candidates"]["chatml"]["arms"]["own"]
        self.assertEqual(arm["n_evaluation_failed"], 12)
        self.assertEqual(arm["n_completed"], 0)

    def test_context_exceeded_cells(self):
        payload = _payload(max_context=10)
        arm = payload["candidates"]["chatml"]["arms"]["own"]
        self.assertEqual(arm["n_context_exceeded"], 12)

    def test_cancelled_stage(self):
        import threading

        cancel = threading.Event()
        cancel.set()
        backend = _WordBackend()
        stage = experiments.history_frame_stage(
            backend,
            (_CHATML,),
            _fixtures(2),
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
            seed=0,
            cancel=cancel,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")


class StratumTests(unittest.TestCase):
    """Design section 6.5: the encoding stratum is a run-time
    measurement over the donor's framing literals."""

    def test_stratum_labels(self):
        backend = _WordBackend()
        # Word backend: multi-word literals fragment; the
        # llama3 bos token is a single word-token.
        stratum = hf.encoding_stratum(backend, hf.donor_format("foreign_llama3"))
        self.assertIn(stratum["label"], hf.STRATA)
        self.assertTrue(stratum["literals"]["<|begin_of_text|>"])

    def test_stratum_block_in_payload(self):
        payload = _payload()
        self.assertEqual(set(payload["stratum"]), set(hf.CONTRASTS))
        for block in payload["stratum"].values():
            self.assertIn(block["label"], hf.STRATA)


_PRIMARY = {"decision": "compatible_on_fixture_set", "winner": "chatml"}


class ValidatorTests(unittest.TestCase):
    """Design section 6.7 + A4: the diffreports validator
    recomputes the aggregates from the records."""

    def test_real_payload_validates_and_self_diff_is_unchanged(self):
        payload = _payload(_WordBackend("frame_drop"), primary=dict(_PRIMARY))
        self.assertTrue(diffreports._valid_history_frame(payload))
        diff = diffreports.diff_history_frame(payload, payload)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_forged_classification_is_rejected(self):
        payload = _payload(_WordBackend("frame_drop"))
        forged = copy.deepcopy(payload)
        forged["candidates"]["chatml"]["contrasts"]["foreign_llama3"]["classification"] = (
            "below_threshold"
        )
        self.assertFalse(diffreports._valid_history_frame(forged))

    def test_a4_digest_forgery_is_rejected(self):
        # The round-1 attack: fake the ids-identical gate to
        # exclude a firing (7,0)-style set by flipping a
        # carried flag. Here the gate is digest-derived, so
        # the forgery must flip digests in the records --
        # which changes the recompute and is caught.
        payload = _payload(_WordBackend("frame_drop"))
        forged = copy.deepcopy(payload)
        for record in forged["candidates"]["chatml"]["records"]:
            record["id_digests"]["foreign_llama3"] = record["id_digests"]["own"]
        self.assertFalse(diffreports._valid_history_frame(forged))

    def test_bos_pair_inconsistency_is_rejected(self):
        # A4's constancy pin: one record carrying a different
        # own-bos digest than its candidate's other records.
        payload = _payload()
        forged = copy.deepcopy(payload)
        forged["candidates"]["llama3"]["records"][0]["own_bos_digest"] = hf.digest_text("x")
        self.assertFalse(diffreports._valid_history_frame(forged))

    def test_cell_verdict_tamper_is_rejected(self):
        payload = _payload()
        forged = copy.deepcopy(payload)
        cell = forged["candidates"]["llama3"]["records"][0]["cells"]["own"]
        cell["verdict"] = not cell["verdict"]
        self.assertFalse(diffreports._valid_history_frame(forged))

    def test_familywise_tamper_is_rejected(self):
        # A2: the family-wise figures are recomputed from the
        # applicable contrasts' own rates, never carried.
        payload = _payload()
        forged = copy.deepcopy(payload)
        forged["familywise"]["expected_false_labels"] = 1.44
        self.assertFalse(diffreports._valid_history_frame(forged))

    def test_unavailable_payload_is_not_a_valid_diff_side(self):
        # Sibling convention (instruction-distance): the
        # wrapper's failure payload is not a valid diff side;
        # the diff reports the asymmetry plainly.
        unavailable = {"available": False, "detail": "historyframe_unavailable"}
        self.assertFalse(diffreports._valid_history_frame(unavailable))
        diff = diffreports.diff_history_frame(unavailable, _payload())
        self.assertEqual(diff["available"], "new")
        self.assertFalse(diff["changed"])

    def test_drift_between_runs_is_detected(self):
        perfect = _payload(_WordBackend("perfect"))
        dropped = _payload(_WordBackend("frame_drop"))
        diff = diffreports.diff_history_frame(perfect, dropped)
        self.assertTrue(diff["changed"])
        metric = next(
            entry
            for entry in diff["candidate_drift"]
            if entry["candidate"] == "chatml" and entry["metric"] == "classification_foreign_llama3"
        )
        self.assertEqual(
            (metric["old"], metric["new"]),
            ("no_discordance_observed", "discordant_asymmetric"),
        )

    def test_protocol_drift_is_detected(self):
        one = _payload()
        two = _payload(fixtures=_fixtures(10))
        diff = diffreports.diff_history_frame(one, two)
        self.assertTrue(diff["protocol_drift"])
        self.assertTrue(diff["changed"])


class ReportingTests(unittest.TestCase):
    """Design section 6.8 + A1/A2/A6/A8: the section prints
    the disclosure, the corrected family-wise figures, the
    jitter grid, the named residual channels, and the
    inapplicable accounting."""

    def test_section_renders_disclosures_and_amendments(self):
        html = reporting.render_html({"history_frame": _payload(_WordBackend("frame_drop"))})
        self.assertIn('<section class="card" id="history-frame">', html)
        # Naming condition: first-sentence prepend disclosure.
        self.assertIn("prepends the frozen six-turn synthetic history", html)
        self.assertIn("reframes only that prepended segment", html)
        # A10: no verdict-identity with the synthhistory stage.
        self.assertIn("request seeds", html)
        # A2: computed family-wise expectation at this
        # payload's own decision count (2 applicable
        # cross-contrasts, rates 0.0135 each -> 0.027).
        self.assertIn("0.027 false labels", html)
        # A6: the jitter grid is printed.
        self.assertIn("0.04 percent at epsilon 0.01", html)
        self.assertIn("2.13 percent at 0.05", html)
        # A8: residual channels named inside the claim scope.
        self.assertIn("batch composition", html)
        # A1: inapplicable accounting printed for chatml.
        self.assertIn("inapplicable", html)
        # Falsification sentence rendered for the firing
        # contrast.
        self.assertIn("discordant_asymmetric", html)

    def test_section_absent_without_payload(self):
        html = reporting.render_html({})
        self.assertNotIn('id="history-frame"', html)


class CliTests(unittest.TestCase):
    """Cycle 218 argparse rule + naming condition: the help
    renders (no bare percent) and carries the prepend
    disclosure in its first sentence."""

    def test_flag_parses_and_help_renders(self):
        from model_protocol_lab.cli import build_parser

        parser = build_parser()
        run = next(
            action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
        )
        help_text = run.choices["run"].format_help()  # raises on a bare %
        flag = next(a for a in run.choices["run"]._actions if "--history-frame" in a.option_strings)
        self.assertIsNotNone(flag)
        # Naming condition: the prepend disclosure sits in
        # the help's opening sentence (before "Per candidate").
        opening = flag.help.split("Per candidate")[0]
        self.assertIn(
            "prepends the frozen six-turn synthetic history to every search fixture",
            opening,
        )
        self.assertIn("reframes ONLY that prepended segment", opening)
        self.assertNotRegex(flag.help, r"%(?![a-z(])", "bare % in help")
        from tests.test_cycle31_cli_verify import run_argv

        args = parser.parse_args(run_argv() + ["--history-frame"])
        self.assertTrue(args.history_frame)
        args = parser.parse_args(run_argv())
        self.assertFalse(args.history_frame)
        self.assertIn("--history-frame", help_text)


if __name__ == "__main__":
    unittest.main()


class GapModuleTests(unittest.TestCase):
    """Gap pass (cycle-174 rule): the full-suite coverage
    run's missing lines in historyframe.py, each closed by
    the smallest test that moves it."""

    def test_donor_format_rejects_unknown_arm(self):
        with self.assertRaises(ValueError):
            hf.donor_format("foreign_gemma")

    def test_history_segment_digest_matches_digest_of_segment(self):
        case = hf.stage_fixtures(_fixtures(1))[0]
        self.assertEqual(
            hf.history_segment_digest(_CHATML, case.messages),
            hf.digest_text(hf.history_segment(_CHATML, case.messages)),
        )

    def test_slice_identities_fail_on_prefix_mismatch(self):
        # The identities' guard branch: a native text that
        # does not start with bos + native segment is not a
        # slice-swap of anything.
        self.assertFalse(hf.slice_swap_identity("unrelated text", "hybrid", "seg", "armseg", "<B>"))
        self.assertFalse(
            hf.verbatim_slice_identity("unrelated text", "verb", "seg", "armseg", "<B>", "<A>")
        )

    def test_stratum_all_single_and_empty(self):
        class _OneIdBackend:
            fingerprint = "sha256:test-one-id"

            def encode(self, text):
                return (1,) if text else ()

        stratum = hf.encoding_stratum(_OneIdBackend(), hf.donor_format("foreign_chatml"))
        self.assertEqual(stratum["label"], "all_single")
        empty = Format("empty", "", "", "", "")
        stratum = hf.encoding_stratum(_OneIdBackend(), empty)
        self.assertEqual(stratum["label"], "all_single")
        self.assertEqual(stratum["literals"], {})

    def test_aggregate_tolerates_malformed_record_shapes(self):
        record = _record("fx-00", own=_cell(True), foreign=_cell(True))
        record["cells"] = ["not", "a", "dict"]
        record["cell_details"] = "nope"
        aggregate = hf.aggregate_candidate([record])
        self.assertEqual(aggregate["arms"]["own"]["n_completed"], 0)

    def test_ungated_foreign_only_counts(self):
        records = self._records(3) if hasattr(self, "_records") else None
        records = [
            _record(f"fx-{i:02d}", own=_cell(False, digest="GREEN"), foreign=_cell(True))
            for i in range(3)
        ]
        block = hf.aggregate_candidate(records)["contrasts"]["foreign_chatml"]
        self.assertEqual(block["ungated_foreign_only"], 3)
        self.assertEqual(block["ungated_own_only"], 0)

    def test_paired_decomposition_counts_through_aggregate(self):
        records = [
            _record(
                f"fx-{i:02d}",
                own=_cell(True, True, "BLUE"),
                foreign=_cell(False, False, "BLUE"),
            )
            for i in range(12)
        ]
        block = hf.aggregate_candidate(records)["contrasts"]["foreign_chatml"]
        self.assertEqual(block["decomposition"]["boundary_only"], 12)
        self.assertEqual(block["n_paired"], 12)

    def test_shared_fixtures_empty_population(self):
        self.assertEqual(hf.shared_fixtures({}), frozenset())
        selection = hf.arm_selection({}, "own")
        self.assertTrue(selection["withheld"])

    def test_selection_no_candidate(self):
        records = {
            "cand-a": [
                _record(f"fx-{i:02d}", own=_cell(False, digest="GREEN"), foreign=_cell(True))
                for i in range(2)
            ],
            "cand-b": [
                _record(f"fx-{i:02d}", own=_cell(False, digest="GREEN"), foreign=_cell(True))
                for i in range(2)
            ],
        }
        selection = hf.arm_selection(records, "own")
        self.assertEqual(selection["decision"], "no_candidate")
        self.assertIsNone(selection["winner"])

    def test_winner_headline_missing_and_stable(self):
        missing = hf.winner_headline(None, {"decision": "selected", "winner": "a"})
        self.assertEqual(missing["detail"], "selection_missing")
        self.assertIsNone(missing["winner_changed"])
        baseline = {"decision": "selected", "winner": "cand-a", "withheld": False}
        same = {"decision": "selected", "winner": "cand-a", "withheld": False}
        stable = hf.winner_headline(baseline, same)
        self.assertEqual(stable["headline"], "winner_stable")
        self.assertFalse(stable["winner_changed"])

    def test_primary_flips_verbatim_decision_and_empty_decision(self):
        conditions = {
            "own": {"decision": "selected", "winner": "cand-a", "withheld": False},
            "foreign_chatml": {"decision": "selected", "winner": "cand-a", "withheld": False},
            "foreign_llama3": {"decision": "selected", "winner": "cand-a", "withheld": False},
        }
        # A non-success decision string compares verbatim.
        flips = hf.primary_flips(conditions, {"decision": "ambiguous", "winner": None})
        self.assertEqual(len(flips), 3)
        # An empty/absent decision maps to no verdict: no flips.
        self.assertEqual(hf.primary_flips(conditions, {"decision": None, "winner": None}), [])
        self.assertEqual(hf.primary_flips(conditions, {"winner": "cand-a"}), [])


class GapStageTests(unittest.TestCase):
    """Gap pass: stage validation, screen, and cancellation
    branches the happy-path tests do not reach."""

    def test_stage_argument_validation(self):
        backend = _WordBackend()
        boundary = tuple(backend.encode("endmark"))
        with self.assertRaises(ValueError):
            experiments.history_frame_stage(
                backend, (), _fixtures(1), boundary=boundary, max_context=100
            )
        with self.assertRaises(ValueError):
            experiments.history_frame_stage(
                backend,
                (_CHATML, _CHATML),
                _fixtures(1),
                boundary=boundary,
                max_context=100,
            )
        with self.assertRaises(ValueError):
            experiments.history_frame_stage(
                backend, (_CHATML,), (), boundary=boundary, max_context=100
            )
        with self.assertRaises(ValueError):
            experiments.history_frame_stage(
                backend,
                (_CHATML,),
                ("not-a-fixture",),
                boundary=boundary,
                max_context=100,
            )
        with self.assertRaises(ValueError):
            dup = _fixtures(2)
            experiments.history_frame_stage(
                backend,
                (_CHATML,),
                (dup[0], dup[0]),
                boundary=boundary,
                max_context=100,
            )

    def test_screen_failure_via_stubbed_audit(self):
        # S2's findings branch: the frozen texts are fixed, so
        # the guard is proven with a stubbed audit (the
        # cycle-171 stub-proves-the-guard pattern).
        from model_protocol_lab import fixtures as fixtures_module

        backend = _WordBackend()
        original = fixtures_module.audit_special_token_literals
        fixtures_module.audit_special_token_literals = lambda texts, formats: ["<|im_start|>"]
        try:
            stage = experiments.history_frame_stage(
                backend,
                (_CHATML,),
                _fixtures(1),
                boundary=tuple(backend.encode("endmark")),
                max_context=8192,
            )
        finally:
            fixtures_module.audit_special_token_literals = original
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "historyframe_screen_failed")
        self.assertEqual(stage["audit"]["special_token_findings"], 1)
        self.assertEqual(backend.calls, 0)

    def test_cancel_set_during_generation_cancels_the_stage(self):
        import threading

        cancel = threading.Event()

        class _CancellingBackend(_WordBackend):
            def generate(self, input_ids, **kwargs):
                out = super().generate(input_ids, **kwargs)
                cancel.set()
                return out

        backend = _CancellingBackend()
        stage = experiments.history_frame_stage(
            backend,
            (_CHATML,),
            _fixtures(2),
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
            cancel=cancel,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_analysis_unavailable_passthrough_and_candidate(self):
        backend = _WordBackend()
        stage = experiments.history_frame_stage(
            backend,
            (_CHATML,),
            _fixtures(2),
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
        )
        unavailable_stage = {
            "available": False,
            "detail": "cancelled",
            "protocol": stage["protocol"],
            "audit": stage["audit"],
            "stratum": stage["stratum"],
        }
        payload = experiments.history_frame_analysis(unavailable_stage, ("chatml",))
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")
        # A stage carrying one unavailable candidate: the
        # analysis records it and excludes it from selection.
        mixed = dict(stage)
        mixed["candidates"] = {
            "chatml": stage["candidates"]["chatml"],
            "ghost": {"status": "unavailable", "detail": "candidate_unavailable"},
        }
        payload = experiments.history_frame_analysis(mixed, ("chatml", "ghost"))
        self.assertEqual(payload["candidates"]["ghost"]["status"], "unavailable")
        self.assertEqual(payload["candidates"]["ghost"]["records"], [])
        self.assertTrue(payload["available"])


class GapValidatorBatteryTests(unittest.TestCase):
    """Gap pass: one broken identity at a time against the
    diffreports mirror (the cycle-174 validator battery)."""

    def setUp(self):
        self.payload = _payload(_WordBackend("frame_drop"), primary=dict(_PRIMARY))

    def _assert_invalid(self, mutate):
        forged = copy.deepcopy(self.payload)
        mutate(forged)
        self.assertFalse(diffreports._valid_history_frame(forged))

    def test_battery(self):
        cases = [
            ("seed type", lambda p: p["protocol"].__setitem__("seed", "0")),
            (
                "fixture ids duplicated",
                lambda p: p["protocol"]["fixture_ids"].__setitem__(
                    1, p["protocol"]["fixture_ids"][0]
                ),
            ),
            ("fixture ids emptied", lambda p: p["protocol"].__setitem__("fixture_ids", [])),
            ("audit count negative", lambda p: p["audit"].__setitem__("n_audit_excluded", -1)),
            (
                "audit by_reason negative",
                lambda p: p["audit"]["by_reason"].__setitem__("special_token_literal", -1),
            ),
            (
                "stratum label inconsistent",
                lambda p: p["stratum"]["foreign_chatml"].__setitem__("label", "mixed"),
            ),
            (
                "stratum literal non-bool",
                lambda p: p["stratum"]["foreign_chatml"]["literals"].__setitem__(
                    "<|im_end|>\n", "yes"
                ),
            ),
            ("stratum missing contrast", lambda p: p["stratum"].pop("foreign_llama3")),
            ("primary winner type", lambda p: p["primary"].__setitem__("winner", 7)),
            (
                "record order swapped",
                lambda p: p["candidates"]["chatml"]["records"].__setitem__(
                    slice(0, 2),
                    [
                        p["candidates"]["chatml"]["records"][1],
                        p["candidates"]["chatml"]["records"][0],
                    ],
                ),
            ),
            (
                "expected digest malformed",
                lambda p: p["candidates"]["chatml"]["records"][0].__setitem__(
                    "expected_digest", "zz"
                ),
            ),
            (
                "segment digest missing arm",
                lambda p: p["candidates"]["chatml"]["records"][0]["segment_digests"].pop("own"),
            ),
            (
                "arm tokens negative",
                lambda p: p["candidates"]["chatml"]["records"][0]["arm_tokens"].__setitem__(
                    "own", -1
                ),
            ),
            (
                "cell detail unknown",
                lambda p: p["candidates"]["chatml"]["records"][0]["cell_details"].__setitem__(
                    "own", "arm_lost"
                ),
            ),
            (
                "cell missing with empty detail",
                lambda p: p["candidates"]["chatml"]["records"][0]["cells"].__setitem__("own", None),
            ),
            (
                "cell boundary_text_match type",
                lambda p: p["candidates"]["chatml"]["records"][0]["cells"]["own"].__setitem__(
                    "boundary_text_match", "yes"
                ),
            ),
            (
                "cell answer digest malformed",
                lambda p: p["candidates"]["chatml"]["records"][0]["cells"]["own"].__setitem__(
                    "answer_digest", "nope"
                ),
            ),
            (
                "cell input tokens bound",
                lambda p: p["candidates"]["chatml"]["records"][0]["cells"]["own"].__setitem__(
                    "input_tokens", 999
                ),
            ),
            ("entry extra key", lambda p: p["candidates"]["chatml"].__setitem__("surprise", 1)),
            ("entry aggregate key dropped", lambda p: p["candidates"]["chatml"].pop("jitter_grid")),
            ("payload key dropped", lambda p: p.pop("flips")),
            ("detail nonempty", lambda p: p.__setitem__("detail", "historyframe_unavailable")),
        ]
        for name, mutate in cases:
            with self.subTest(case=name):
                self._assert_invalid(mutate)

    def test_candidate_cap_rejected(self):
        forged = copy.deepcopy(self.payload)
        entry = forged["candidates"]["chatml"]
        for i in range(24):
            forged["candidates"][f"extra-{i:02d}"] = {
                "status": "unavailable",
                "detail": "candidate_unavailable",
                "records": [],
            }
        self.assertFalse(diffreports._valid_history_frame(forged))
        del entry

    def test_diff_candidates_added_removed_and_stratum_drift(self):
        one = _payload(_WordBackend("perfect"), formats=(_CHATML,))
        two = _payload(_WordBackend("perfect"), formats=(_CHATML, _LLAMA3))
        diff = diffreports.diff_history_frame(one, two)
        self.assertEqual(diff["candidates_added"], ["llama3"])
        self.assertEqual(diff["candidates_removed"], [])
        self.assertTrue(diff["changed"])
        edited = copy.deepcopy(two)
        literals = edited["stratum"]["foreign_chatml"]["literals"]
        for key in literals:
            literals[key] = True
        edited["stratum"]["foreign_chatml"]["label"] = "all_single"
        self.assertTrue(diffreports._valid_history_frame(edited))
        drift = diffreports.diff_history_frame(two, edited)
        self.assertEqual(
            drift["stratum_drift"],
            [{"family": "foreign_chatml", "old": "all_fragmented", "new": "all_single"}],
        )
        self.assertTrue(drift["changed"])

    def test_diff_status_drift(self):
        backend = _WordBackend("perfect")
        stage = experiments.history_frame_stage(
            backend,
            (_CHATML, _LLAMA3),
            _fixtures(12),
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
        )
        available = experiments.history_frame_analysis(stage, ("chatml", "llama3"))
        mixed = dict(stage)
        mixed["candidates"] = {
            "chatml": stage["candidates"]["chatml"],
            "llama3": {"status": "unavailable", "detail": "candidate_unavailable"},
        }
        unavailable = experiments.history_frame_analysis(mixed, ("chatml", "llama3"))
        self.assertTrue(diffreports._valid_history_frame(unavailable))
        diff = diffreports.diff_history_frame(available, unavailable)
        self.assertTrue(diff["changed"])
        self.assertIn(
            {"candidate": "llama3", "metric": "status", "old": "available", "new": "unavailable"},
            diff["candidate_drift"],
        )


class GapReportingTests(unittest.TestCase):
    """Gap pass: section branches -- n/a fallbacks, the
    unavailable-candidate row, withheld conditions, and a
    nonempty flips list."""

    def test_section_with_attrition_and_primary_flips(self):
        payload = _payload(_WordBackend("explode"), primary=dict(_PRIMARY))
        html = reporting.render_html({"history_frame": payload})
        self.assertIn("n/a", html)  # zero-completion arms have no rate

    def test_section_unavailable_candidate_and_withheld(self):
        payload = _payload(_WordBackend("perfect"))
        forged = copy.deepcopy(payload)
        forged["candidates"]["ghost"] = {
            "status": "unavailable",
            "detail": "candidate_unavailable",
            "records": [],
        }
        for arm in forged["conditions"]:
            forged["conditions"][arm] = {
                "decision": "unavailable",
                "detail": "selection_withheld_mismatched_sets",
                "winner": None,
                "withheld": True,
            }
        html = reporting.render_html({"history_frame": forged})
        self.assertIn("unavailable", html)
        self.assertIn("selection withheld (mismatched or empty completed sets)", html)

    def test_section_prints_flips_against_primary(self):
        payload = _payload(_WordBackend("frame_drop"), primary=dict(_PRIMARY))
        self.assertTrue(payload["flips"])
        html = reporting.render_html({"history_frame": payload})
        self.assertIn("Flips against the primary run", html)
