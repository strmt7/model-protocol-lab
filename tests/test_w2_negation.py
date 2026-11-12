"""W2 push 3 regression tests: RR-03 negation dispatch + census.

The negation section was produced by every run that stages it
but never dispatched by ``diff_reports``: a headline collapse
from 16/16 joint passes to 0/16 compared ``identical: true``
and a present-vs-absent pair compared equal too. The genuine
pair below is driven through the real stage and analysis (the
audit reproducer's driver); the admission cases tamper with a
genuine stored section and require the validator to refuse it
(the recompute standard every other section is held to); the
census pins the instrument registry against two independent
surfaces (the bare-pair diff dict and the cycle-67 oracle's
expected-identical dict) so a future producer section cannot
silently go undispatched again.
"""

from __future__ import annotations

import copy
import re
import sys
import unittest

sys.path.insert(0, "tests")

from model_protocol_lab import diffreports, experiments  # noqa: E402
from model_protocol_lab.core import Format, Generation  # noqa: E402


class ArithmeticBackend:
    """Scripted backend: correct unless it drops the negation."""

    fingerprint = "synthetic-negation-control"

    def __init__(self, drop_negation: bool):
        self.drop_negation = drop_negation
        self.calls = 0

    def encode(self, text):
        """Encode."""
        return tuple(text.encode("utf-8"))

    def decode(self, ids):
        """Decode."""
        return bytes(ids).decode("utf-8", "replace")

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        self.calls += 1
        prompt = self.decode(input_ids)
        pairs = re.findall(r"\n([0-9]+)\n([0-9]+)", prompt)
        assert pairs
        a, b = map(int, pairs[-1])
        larger = "is smaller?" not in prompt
        if self.drop_negation and "not the smaller?" in prompt:
            larger = False
        answer = "first" if ((a > b) == larger) else "second"
        output = tuple((answer + chr(7)).encode("utf-8"))
        output += (0,) * max(0, max_new_tokens - len(output))
        return Generation(token_ids=output, reason="limit")


_FORMAT = Format(name="synthetic", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>")


def _genuine_section(drop_negation: bool) -> dict:
    """Run the real negation stage + analysis for one backend."""
    backend = ArithmeticBackend(drop_negation)
    stage = experiments.negation_stage(
        backend, (_FORMAT,), boundary=(7,), max_new_tokens=16, max_context=32768, seed=0
    )
    section = experiments.negation_analysis(stage, (_FORMAT.name,))
    assert section["available"] is True
    return section


def _wrap(section):
    """Wrap a section (or None) as a minimal results payload."""
    payload = {
        "decision": "no_candidate",
        "selected_candidate": None,
        "search": [],
        "held_out": None,
    }
    if section is not None:
        payload["negation"] = section
    return payload


class NegationDispatchTests(unittest.TestCase):
    """RR-03: the negation section is dispatched and compared."""

    @classmethod
    def setUpClass(cls):
        cls.robust = _genuine_section(False)
        cls.dropped = _genuine_section(True)

    def test_genuine_pair_really_differs(self):
        """Precondition: the two genuine sections carry different headlines."""
        robust_headline = self.robust["candidates"]["synthetic"]["headline"]
        dropped_headline = self.dropped["candidates"]["synthetic"]["headline"]
        self.assertNotEqual(robust_headline["joint"], dropped_headline["joint"])
        self.assertNotEqual(robust_headline["decision"], dropped_headline["decision"])

    def test_headline_collapse_is_a_change(self):
        """16/16 -> 0/16 must not compare identical."""
        diff = diffreports.diff_reports(_wrap(self.robust), _wrap(self.dropped))
        self.assertIn("negation", diff)
        self.assertEqual(diff["negation"]["available"], "both")
        self.assertTrue(diff["negation"]["changed"])
        self.assertTrue(diff["summary"]["negation_changed"])
        self.assertFalse(diff["summary"]["identical"])

    def test_identical_pair_stays_identical(self):
        """The same genuine section on both sides compares unchanged."""
        diff = diffreports.diff_reports(_wrap(self.robust), _wrap(copy.deepcopy(self.robust)))
        self.assertEqual(diff["negation"]["available"], "both")
        self.assertFalse(diff["negation"]["changed"])
        self.assertFalse(diff["summary"]["negation_asymmetric"])
        self.assertTrue(diff["summary"]["identical"])

    def test_present_vs_absent_is_asymmetric(self):
        """A section on only one side is an asymmetry, never identical."""
        diff = diffreports.diff_reports(_wrap(self.robust), _wrap(None))
        self.assertEqual(diff["negation"]["available"], "old")
        self.assertTrue(diff["summary"]["negation_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])
        diff = diffreports.diff_reports(_wrap(None), _wrap(self.robust))
        self.assertEqual(diff["negation"]["available"], "new")
        self.assertTrue(diff["summary"]["negation_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])

    def test_absent_on_both_sides_is_neither(self):
        """No section anywhere: the axis simply does not engage."""
        diff = diffreports.diff_reports(_wrap(None), _wrap(None))
        self.assertEqual(diff["negation"], {"available": "neither", "changed": False})
        self.assertTrue(diff["summary"]["identical"])

    def test_render_summary_names_the_drift(self):
        """The text summary carries a negation drift line."""
        diff = diffreports.diff_reports(_wrap(self.robust), _wrap(self.dropped))
        text = diffreports.render_summary(diff)
        self.assertIn("Negation", text)


class NegationAdmissionTests(unittest.TestCase):
    """The validator recomputes stored aggregates from records."""

    @classmethod
    def setUpClass(cls):
        cls.genuine = _genuine_section(False)

    def _diff_with(self, section) -> dict:
        return diffreports.diff_reports(_wrap(self.genuine), _wrap(section))

    def test_tampered_aggregate_is_refused(self):
        """A stored headline count that disagrees with the records is invalid."""
        section = copy.deepcopy(self.genuine)
        section["candidates"]["synthetic"]["headline"]["joint"] += 1
        diff = self._diff_with(section)
        self.assertEqual(diff["negation"]["available"], "old")

    def test_tampered_arm_rate_is_refused(self):
        """A stored per-arm rate that disagrees with the records is invalid."""
        section = copy.deepcopy(self.genuine)
        section["candidates"]["synthetic"]["arms"]["direct"]["rate"] = 0.5
        diff = self._diff_with(section)
        self.assertEqual(diff["negation"]["available"], "old")

    def test_bool_seed_is_refused(self):
        """The protocol seed is an int gate: True is not a seed."""
        section = copy.deepcopy(self.genuine)
        section["protocol"]["seed"] = True
        diff = self._diff_with(section)
        self.assertEqual(diff["negation"]["available"], "old")

    def test_wrong_case_count_is_refused(self):
        """n_cases must equal the generated case count for the seed."""
        section = copy.deepcopy(self.genuine)
        section["n_cases"] += 1
        diff = self._diff_with(section)
        self.assertEqual(diff["negation"]["available"], "old")

    def test_wrong_candidate_count_is_refused(self):
        """n_candidates must equal the candidate map's size."""
        section = copy.deepcopy(self.genuine)
        section["n_candidates"] += 1
        diff = self._diff_with(section)
        self.assertEqual(diff["negation"]["available"], "old")

    def test_extra_top_level_key_is_refused(self):
        """The section's key set is exact (recompute standard)."""
        section = copy.deepcopy(self.genuine)
        section["extra"] = {}
        diff = self._diff_with(section)
        self.assertEqual(diff["negation"]["available"], "old")

    def test_malformed_records_never_raise(self):
        """Hostile record shapes degrade the section, never crash the diff."""
        section = copy.deepcopy(self.genuine)
        section["candidates"]["synthetic"]["records"] = [
            {"cells": {"direct": {"verdict": True}}},
            7,
        ]
        diff = self._diff_with(section)
        self.assertEqual(diff["negation"]["available"], "old")

    def test_unavailable_candidate_shape_is_compared(self):
        """A candidate recorded unavailable on both sides stays valid."""
        section = copy.deepcopy(self.genuine)
        section["candidates"]["synthetic"] = {
            "status": "unavailable",
            "detail": "backend_failed",
            "records": [],
        }
        diff = diffreports.diff_reports(_wrap(section), _wrap(copy.deepcopy(section)))
        self.assertEqual(diff["negation"]["available"], "both")
        self.assertFalse(diff["negation"]["changed"])


_CORE_DIFF_KEYS = {
    "tool",
    "old",
    "new",
    "decision",
    "winner",
    "section_states",
    "config_drift",
    "candidate_verdicts",
    "evidence",
    "fixture_diffs",
    "fixture_diffs_include_unchanged",
    "summary",
}


class SectionCensusTests(unittest.TestCase):
    """Every producer section is dispatched or has a named disposition."""

    def test_bare_pair_sections_match_registry(self):
        """The diff dict's section keys are exactly the registry's."""
        diff = diffreports.diff_reports(_wrap(None), _wrap(None))
        sections = set(diff) - _CORE_DIFF_KEYS
        self.assertEqual(sections, set(diffreports.SECTION_DIFF_KEYS.values()))

    def test_cycle67_expected_sections_match_registry(self):
        """The cycle-67 oracle's identical-diff sections agree."""
        from test_cycle67_diffreports_audit import _identical_expected

        expected = _identical_expected()
        sections = set(expected) - _CORE_DIFF_KEYS
        self.assertEqual(sections, set(diffreports.SECTION_DIFF_KEYS.values()))

    def test_undispatched_sections_are_named_with_reasons(self):
        """The opt-out set is exactly the eight documented sections."""
        self.assertEqual(
            set(diffreports.UNDISPATCHED_SECTIONS),
            {
                "sensitivity_sweep",
                "sensitivity_sweep_tokens",
                "stop_matrix",
                "logprob_diagnostic",
                "template_conformance",
                "template_firewall",
                "marker_isolation",
                "backend_divergences",
            },
        )
        for name, reason in diffreports.UNDISPATCHED_SECTIONS.items():
            self.assertTrue(reason.strip(), f"{name} needs a documented disposition")

    def test_dispatched_registry_covers_negation(self):
        """Negation itself moved from undispatched to dispatched."""
        self.assertEqual(diffreports.SECTION_DIFF_KEYS["negation"], "negation")
        self.assertNotIn("negation", diffreports.UNDISPATCHED_SECTIONS)
        self.assertEqual(len(diffreports.SECTION_DIFF_KEYS), 56)


def _genuine_two_candidate_section() -> dict:
    """Run the real stage with two candidate formats."""
    fmt_b = Format(
        name="synthetic-b", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"
    )
    backend = ArithmeticBackend(False)
    stage = experiments.negation_stage(
        backend, (_FORMAT, fmt_b), boundary=(7,), max_new_tokens=16, max_context=32768, seed=0
    )
    section = experiments.negation_analysis(stage, (_FORMAT.name, fmt_b.name))
    assert section["available"] is True
    return section


class NegationValidatorGateTests(unittest.TestCase):
    """Each shape gate of the negation validator refuses its own defect."""

    @classmethod
    def setUpClass(cls):
        """Build one genuine section for the whole battery."""
        cls.genuine = _genuine_section(False)

    def _assert_refused(self, section):
        diff = diffreports.diff_reports(_wrap(section), _wrap(self.genuine))
        self.assertTrue(diff["summary"]["negation_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])

    def test_nonempty_detail_is_refused(self):
        """An available section carries an empty detail by construction."""
        section = copy.deepcopy(self.genuine)
        section["detail"] = "partial"
        self._assert_refused(section)

    def test_non_dict_protocol_is_refused(self):
        """The protocol block is a dict."""
        section = copy.deepcopy(self.genuine)
        section["protocol"] = []
        self._assert_refused(section)

    def test_non_string_fingerprint_is_refused(self):
        """The protocol fingerprint is a string digest."""
        section = copy.deepcopy(self.genuine)
        section["protocol"]["fingerprint"] = 7
        self._assert_refused(section)

    def test_non_dict_audit_is_refused(self):
        """The audit block is None or a dict."""
        section = copy.deepcopy(self.genuine)
        section["audit"] = []
        self._assert_refused(section)

    def test_non_dict_candidates_is_refused(self):
        """The candidates block is a dict keyed by candidate name."""
        section = copy.deepcopy(self.genuine)
        section["candidates"] = []
        self._assert_refused(section)

    def test_non_dict_candidate_entry_is_refused(self):
        """Every candidate entry is a dict."""
        section = copy.deepcopy(self.genuine)
        section["candidates"]["synthetic"] = "available"
        self._assert_refused(section)

    def test_unavailable_entry_with_extra_key_is_refused(self):
        """An unavailable entry carries exactly status/detail/records."""
        section = copy.deepcopy(self.genuine)
        section["candidates"]["synthetic"] = {
            "status": "unavailable",
            "detail": "backend missing",
            "records": [],
            "arms": {},
        }
        self._assert_refused(section)

    def test_unavailable_entry_with_records_is_refused(self):
        """An unavailable entry recorded no cells."""
        section = copy.deepcopy(self.genuine)
        section["candidates"]["synthetic"] = {
            "status": "unavailable",
            "detail": "backend missing",
            "records": [{}],
        }
        self._assert_refused(section)

    def test_unknown_candidate_status_is_refused(self):
        """A candidate status is available or unavailable."""
        section = copy.deepcopy(self.genuine)
        section["candidates"]["synthetic"]["status"] = "partial"
        self._assert_refused(section)

    def test_available_entry_missing_aggregate_key_is_refused(self):
        """An available entry carries the full aggregate key set."""
        section = copy.deepcopy(self.genuine)
        del section["candidates"]["synthetic"]["arms"]
        self._assert_refused(section)

    def test_available_entry_with_detail_is_refused(self):
        """An available entry carries an empty detail by construction."""
        section = copy.deepcopy(self.genuine)
        section["candidates"]["synthetic"]["detail"] = "partial"
        self._assert_refused(section)

    def test_available_entry_with_non_list_records_is_refused(self):
        """An available entry's records are a list."""
        section = copy.deepcopy(self.genuine)
        section["candidates"]["synthetic"]["records"] = {}
        self._assert_refused(section)


class NegationDriftSurfaceTests(unittest.TestCase):
    """Status, protocol, and roster drift are named on both surfaces."""

    @classmethod
    def setUpClass(cls):
        """Build the genuine sections once."""
        cls.genuine = _genuine_section(False)
        cls.two = _genuine_two_candidate_section()

    def test_status_drift_is_named(self):
        """A candidate losing its measurement is a status drift."""
        new = copy.deepcopy(self.genuine)
        new["candidates"]["synthetic"] = {
            "status": "unavailable",
            "detail": "backend missing",
            "records": [],
        }
        diff = diffreports.diff_reports(_wrap(self.genuine), _wrap(new))
        metrics = [entry["metric"] for entry in diff["negation"]["candidate_drift"]]
        self.assertIn("status", metrics)
        self.assertFalse(diff["summary"]["identical"])
        self.assertIn("Negation measurement drift", diffreports.render_summary(diff))

    def test_protocol_drift_is_named_and_rendered(self):
        """A different instrument fingerprint is not comparable."""
        new = copy.deepcopy(self.genuine)
        new["protocol"]["fingerprint"] = "0" * 64
        diff = diffreports.diff_reports(_wrap(self.genuine), _wrap(new))
        self.assertTrue(diff["negation"]["protocol_drift"])
        self.assertFalse(diff["negation"]["comparable"])
        self.assertIn("Negation protocol drift", diffreports.render_summary(diff))

    def test_candidate_added_and_removed_are_named(self):
        """Roster changes surface in the diff and the summary."""
        forward = diffreports.diff_reports(_wrap(self.genuine), _wrap(self.two))
        self.assertEqual(forward["negation"]["candidates_added"], ["synthetic-b"])
        self.assertIn("Negation candidates added: synthetic-b", diffreports.render_summary(forward))
        backward = diffreports.diff_reports(_wrap(self.two), _wrap(self.genuine))
        self.assertEqual(backward["negation"]["candidates_removed"], ["synthetic-b"])
        self.assertIn(
            "Negation candidates removed: synthetic-b", diffreports.render_summary(backward)
        )

    def test_one_sided_render_names_the_missing_side(self):
        """A one-sided comparison says which run lacks the section."""
        diff = diffreports.diff_reports(_wrap(self.genuine), _wrap(None))
        self.assertIn(
            "Negation comparison unavailable: new run did not record",
            diffreports.render_summary(diff),
        )

    def test_renderer_overflow_and_case_count_lines(self):
        """Renderer contract: >5 drifts fold, case-count drift prints."""
        others = [
            {"candidate": "synthetic", "metric": f"arms.arm{i}.rate", "old": 1.0, "new": 0.5}
            for i in range(7)
        ]
        lines = diffreports._render_negation_summary(
            {
                "available": "both",
                "protocol_drift": False,
                "candidate_drift": others,
                "n_cases": {"old": 16, "new": 15, "changed": True},
                "candidates_added": [],
                "candidates_removed": [],
            }
        )
        text = "\n".join(lines)
        self.assertIn("Negation measurement drift (7):", text)
        self.assertIn("... and 2 more", text)
        self.assertIn("Negation case count changed: 16 -> 15", text)


if __name__ == "__main__":
    unittest.main()
