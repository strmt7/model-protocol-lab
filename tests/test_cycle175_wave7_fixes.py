"""Cycle 175: wave-7 red-team fixes (F1, F2, F3, F7).

The wave-7 swarm record is in docs/RESEARCH_LOG.md. Every finding
was reproduced by red-team A, re-reproduced by red-team B (whose
job was disproof), and independently re-verified by the
coordinator before this cycle was built:

- F1 (HIGH): the cycle-174 refusal detector's bare prefix matching
  fired on natural non-refusals ("I will note..." -> "i will not").
  Detector v2 adds a word-boundary rule: per-opener, longest among
  the openers that also end at a boundary, letter = Unicode L*.
- F2 (LOW after cross-check): normalisation evasions -- format
  characters inside a word glued it shut (now: Cf becomes a space),
  and combining marks / Turkish dotted I survived casefold (now:
  NFKD + Mn strip). The glued missing-space refusal
  ("I cannotprovide...") is the boundary rule's documented cost.
- F3 (HIGH): the determinism audit's certificate was fail-open --
  ``recorded_ids_invalid`` was missing from the blocking skip set
  although the design doc's I5 names invalid recorded IDs as
  blocking, and ID validation ran before the status check, so an
  aborted observation's legitimately empty inputs were mislabelled.
  Fix: the code joins the blocking set, and status is judged before
  ID validity (a cell the run never claimed is scope, whatever its
  stored IDs look like).
- F7 (MEDIUM): ``pack`` accepted a report file named
  ``manifest.json`` and built a bundle its own verifier rejects.
  The name is now reserved at collection time.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from threading import Event
from typing import Any

from model_protocol_lab import determinism, pack, refusals
from model_protocol_lab.core import Generation


def _pure_output(input_ids: tuple[int, ...], seed: int, count: int) -> tuple[int, ...]:
    """Pure output."""
    digest = hashlib.sha256(bytes(b % 256 for b in input_ids) + seed.to_bytes(8, "big")).digest()
    return tuple(b % 50 for b in digest[:count])


class _PureBackend:
    """Deterministic double: output depends only on (input_ids, seed)."""

    vocab_size = 1000

    def __init__(self, fingerprint: str = "testfp") -> None:
        self.fingerprint = fingerprint
        self.calls: list[tuple[tuple[int, ...], int, int]] = []

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        """Generate."""
        self.calls.append((tuple(input_ids), max_new_tokens, seed))
        return Generation(_pure_output(tuple(input_ids), seed, max_new_tokens), "limit")


def _obs(
    fixture: str,
    input_ids: Any,
    output_ids: Any,
    status: str = "completed",
) -> dict[str, Any]:
    """Obs."""
    return {
        "candidate": "cand-a",
        "fixture": fixture,
        "input_ids": input_ids,
        "output_ids": output_ids,
        "answer": "",
        "exact_match": False,
        "boundary_found": False,
        "boundary_text_match": None,
        "raw_reason": "limit" if status == "completed" else None,
        "error": None,
        "status": status,
        "input_text": "",
    }


def _entry(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Entry."""
    return {
        "format": {"name": "cand-a"},
        "summary": {
            "aggregate_eligible": True,
            "planned": 2,
            "exact_and_boundary_passes": 1,
            "status": "completed",
        },
        "observations": observations,
    }


def _artifact(observations: list[dict[str, Any]], *, seed: int = 7, cap: int = 4) -> dict[str, Any]:
    """Artifact."""
    return {
        "schema_version": 1,
        "seed": seed,
        "max_new_tokens": cap,
        "backend_fingerprint": "testfp",
        "decision": "no_candidate",
        "search": [_entry(observations)],
        "held_out": None,
        "selected_candidate": None,
        "boundary_token_ids": [9],
        "failure_taxonomy": {},
    }


def _reproducing_obs(
    fixture: str, input_ids: tuple[int, ...], *, seed: int = 7, cap: int = 4
) -> dict[str, Any]:
    """Reproducing obs."""
    request_seed = int.from_bytes(hashlib.sha256(f"{seed}\0{fixture}".encode()).digest()[:4], "big")
    return _obs(fixture, list(input_ids), list(_pure_output(input_ids, request_seed, cap)))


class TestDetectorV2Boundaries(unittest.TestCase):
    """Test detector v2 boundaries."""

    def test_detector_version_bumped(self) -> None:
        """Test detector version bumped."""
        self.assertEqual(refusals.DETECTOR_ID, "frozen_refusal_openers_v2")

    def test_red_team_false_positive_corpus_is_silent(self) -> None:
        """Test red team false positive corpus is silent."""
        # Wave-7 A F1 + B's additional members, verbatim.
        corpus = [
            "I will note the key points for the meeting.",
            "I will notify you when the results are ready.",
            "As an Air Force pilot, I flew transports.",
            "As an airport guide, I can show you the terminal.",
            "As an air conditioner technician, I service units.",
            "It is not accurately known when this happened.",
        ]
        for answer in corpus:
            with self.subTest(answer=answer):
                self.assertIsNone(refusals.classify_refusal(answer))

    def test_boundary_corpus_still_fires(self) -> None:
        """Test boundary corpus still fires."""
        for opener in refusals.REFUSAL_OPENERS:
            with self.subTest(opener=opener):
                self.assertEqual(refusals.classify_refusal(opener), opener)
                self.assertEqual(refusals.classify_refusal(opener + ", and so on"), opener)
                self.assertEqual(refusals.classify_refusal(opener + "."), opener)
                # The glued opener itself never fires (a shorter
                # chain member may legitimately claim the answer).
                self.assertNotEqual(refusals.classify_refusal(opener + "x"), opener)

    def test_longest_among_boundary_passing(self) -> None:
        """Test longest among boundary passing."""
        # B's condition: a longest-only boundary check would lose
        # the "i cannot" attribution here.
        self.assertEqual(refusals.classify_refusal("I cannot providex the data"), "i cannot")
        self.assertEqual(refusals.classify_refusal("I cannot provide the data"), "i cannot provide")

    def test_unicode_letter_boundary(self) -> None:
        """Test unicode letter boundary."""
        # B's condition: the boundary class is Unicode L*, not
        # [a-z] -- a Cyrillic continuation also blocks the match.
        self.assertIsNone(refusals.classify_refusal("I will not\u0430"))  # Cyrillic а
        self.assertEqual(refusals.classify_refusal("I will not."), "i will not")


class TestDetectorV2Normalisation(unittest.TestCase):
    """Test detector v2 normalisation."""

    def test_format_characters_are_word_breaks(self) -> None:
        """Test format characters are word breaks."""
        self.assertEqual(refusals.classify_refusal("I\u200bcannot help."), "i cannot")
        self.assertEqual(refusals.classify_refusal("I'm\u200bsorry, no."), "i'm sorry")

    def test_combining_marks_and_dotted_capital_i(self) -> None:
        """Test combining marks and dotted capital i."""
        self.assertEqual(refusals.classify_refusal("I\u0301 cannot help."), "i cannot")
        self.assertEqual(refusals.classify_refusal("\u0130'm sorry, no."), "i'm sorry")

    def test_glued_refusal_is_the_documented_cost(self) -> None:
        """Test glued refusal is the documented cost."""
        self.assertIsNone(refusals.classify_refusal("I cannotprovide that."))

    def test_v1_behaviour_preserved_where_it_was_right(self) -> None:
        """Test v1 behaviour preserved where it was right."""
        self.assertEqual(
            refusals.classify_refusal("I'm sorry to hear about your hamster."),
            "i'm sorry",
        )
        self.assertIsNone(refusals.classify_refusal("No."))
        self.assertIsNone(refusals.classify_refusal("The answer is 7. I'm sorry."))


class TestDeterminismBlockingSet(unittest.TestCase):
    """Test determinism blocking set."""

    def test_invalid_completed_cell_blocks_certificate(self) -> None:
        """Test invalid completed cell blocks certificate."""
        # F3's fail-open shape: one reproducing cell + one claimed
        # completed cell with a non-integer token id.
        good = _reproducing_obs("f1", (1, 2, 3))
        bad = _obs("f2", [9, 8], ["not-an-int", 5, 6, 7])
        audit = determinism.audit_determinism(_artifact([good, bad]), _PureBackend())
        self.assertNotEqual(audit["verdict"], "deterministic")
        self.assertIsNone(audit["certificate"])
        bad_cell = next(c for c in audit["cells"] if c["fixture"] == "f2")
        self.assertEqual(bad_cell["detail"], "recorded_ids_invalid")

    def test_empty_input_completed_cell_blocks_certificate(self) -> None:
        """Test empty input completed cell blocks certificate."""
        good = _reproducing_obs("f1", (1, 2, 3))
        bad = _obs("f2", [], [4, 5, 6, 7])
        audit = determinism.audit_determinism(_artifact([good, bad]), _PureBackend())
        self.assertNotEqual(audit["verdict"], "deterministic")
        self.assertIsNone(audit["certificate"])

    def test_unclaimed_cell_with_empty_ids_is_scope_not_a_gap(self) -> None:
        """Test unclaimed cell with empty ids is scope not a gap."""
        # The status-first half of F3: an aborted observation
        # legitimately stores empty inputs; it must be labelled
        # observation_not_completed and must NOT block a certificate
        # earned by the claimed cells.
        good = _reproducing_obs("f1", (1, 2, 3))
        aborted = _obs("f2", [], [], status="failed")
        audit = determinism.audit_determinism(_artifact([good, aborted]), _PureBackend())
        self.assertEqual(audit["verdict"], "deterministic")
        self.assertIsNotNone(audit["certificate"])
        aborted_cell = next(c for c in audit["cells"] if c["fixture"] == "f2")
        self.assertEqual(aborted_cell["detail"], "observation_not_completed")

    def test_blocking_set_matches_design_doc_i5(self) -> None:
        """Test blocking set matches design doc i5."""
        for code in (
            "recorded_ids_invalid",
            "recorded_length_inconsistent",
            "recorded_ids_out_of_vocab",
            "replay_context_limit_exceeded",
            "audit_cell_budget_exceeded",
        ):
            self.assertIn(code, determinism._BLOCKING_SKIP_DETAILS)
        self.assertNotIn("observation_not_completed", determinism._BLOCKING_SKIP_DETAILS)


class TestPackReservedName(unittest.TestCase):
    """Test pack reserved name."""

    @staticmethod
    def _report_dir(tmp: str, *, with_manifest: bool) -> Path:
        """Report dir."""
        sample = Path(__file__).resolve().parent.parent / "examples" / "sample-report"
        directory = Path(tmp)
        for name in ("results.json", "report.html"):
            (directory / name).write_bytes((sample / name).read_bytes())
        if with_manifest:
            (directory / "manifest.json").write_text("{}", encoding="utf-8")
        return directory

    def test_manifest_named_report_file_is_refused_at_collection(self) -> None:
        """Test manifest named report file is refused at collection."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._report_dir(tmp, with_manifest=True)
            with self.assertRaises(ValueError) as caught:
                pack.collect_pack_files(directory, [])
            self.assertEqual(str(caught.exception), "pack_entry_name_reserved")
            with self.assertRaises(ValueError) as caught:
                pack.pack_bundle(directory, [])
            self.assertEqual(str(caught.exception), "pack_entry_name_reserved")

    def test_clean_report_dir_still_packs(self) -> None:
        """Test clean report dir still packs."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._report_dir(tmp, with_manifest=False)
            _data, _manifest, digest = pack.pack_bundle(directory, [])
            self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
