"""Cycle 28: docstring-contract regression tests for diffreports.py.

Cycle 27's differential verification found no functional bug, but it did
find two docstrings whose claims did not match the (pinned, unchanged)
code behavior. The docstrings were corrected in cycle 28; these tests pin
the code behavior so a future doc edit cannot quietly lie again:

1. Duplicate candidate names in "search": verdicts are first-wins
   (``_search_verdicts`` skips repeat names), but per-fixture evidence is
   LAST-wins (``_stage_evidence`` has no repeat-name guard).
2. Fixture-set drift gating is a single any-stage flag per side: one side
   carrying only search evidence and the other only held-out evidence
   still emits ``fixtures:search``/``fixtures:held_out`` drift entries and
   "absent" fixture diffs on the empty stages.

All tests run against the implementation (no oracle here); the goal is
the doc <-> behavior contract, not the behavior itself.
"""

from __future__ import annotations

import unittest
from typing import Any

from model_protocol_lab import diffreports


def _summary(planned: int, passes: int) -> dict[str, Any]:
    return {
        "planned": planned,
        "exact_and_boundary_passes": passes,
        "aggregate_eligible": True,
    }


class EvidenceDuplicateNameContractTests(unittest.TestCase):
    """Verdicts first-wins, evidence last-wins on duplicate candidate names."""

    def _dup_run(self) -> dict[str, Any]:
        first = {
            "format": {"name": "dup"},
            "summary": _summary(1, 1),
            "observations": [{"fixture": "f1", "exact_match": True, "boundary_found": True}],
        }
        second = {
            "format": {"name": "dup"},
            "summary": _summary(1, 0),
            "observations": [
                {"fixture": "f1", "exact_match": False, "boundary_found": True},
                {"fixture": "f2", "exact_match": True, "boundary_found": True},
            ],
        }
        return {
            "decision": "s",
            "search": [first, second],
            "selected_candidate": None,
            "held_out": None,
        }

    def test_verdicts_first_wins(self) -> None:
        """The verdict uses the FIRST entry's summary (pass)."""
        verdicts, _ = diffreports._search_verdicts(self._dup_run())
        self.assertEqual(verdicts, {"dup": "pass"})

    def test_evidence_last_wins(self) -> None:
        """The evidence map uses the SECOND entry's observations (last-wins)."""
        evidence, _, present = diffreports._stage_evidence(self._dup_run(), "search")
        self.assertTrue(present)
        self.assertEqual(evidence, {"dup": {"f1": "fail", "f2": "pass"}})

    def test_later_entry_without_usable_evidence_keeps_earlier(self) -> None:
        """A later duplicate with no usable observations leaves the earlier
        entry's outcomes in place (no repeat-name guard AND no empty-overwrite)."""
        run = {
            "decision": "s",
            "search": [
                {
                    "format": {"name": "dup"},
                    "summary": _summary(1, 1),
                    "observations": [
                        {"fixture": "f1", "exact_match": True, "boundary_found": True}
                    ],
                },
                {
                    "format": {"name": "dup"},
                    "summary": _summary(1, 0),
                    # observations present but malformed -> no usable outcomes
                    "observations": [{"fixture": "f1"}],
                },
            ],
            "selected_candidate": None,
            "held_out": None,
        }
        evidence, _, present = diffreports._stage_evidence(run, "search")
        self.assertTrue(present)
        self.assertEqual(evidence, {"dup": {"f1": "pass"}})


class FixtureSetDriftGatingContractTests(unittest.TestCase):
    """Any-stage evidence gating: per-stage entries fire on empty stages."""

    def _search_evidence_only(self) -> dict[str, Any]:
        return {
            "decision": "s",
            "search": [
                {
                    "format": {"name": "a"},
                    "summary": _summary(1, 1),
                    "observations": [
                        {
                            "fixture": "f1",
                            "exact_match": True,
                            "boundary_found": True,
                        }
                    ],
                }
            ],
            "selected_candidate": None,
            "held_out": None,
        }

    def _held_out_evidence_only(self) -> dict[str, Any]:
        return {
            "decision": "s",
            "search": [
                {
                    "format": {"name": "a"},
                    "summary": _summary(1, 1),
                    # no observations -> no usable evidence on the search stage
                }
            ],
            "selected_candidate": "a",
            "held_out": {
                "observations": [{"fixture": "f9", "exact_match": True, "boundary_found": True}]
            },
        }

    def test_fixture_set_drift_fires_per_stage(self) -> None:
        """Both stages emit drift entries even though each side has only one."""
        diff = diffreports.diff_reports(
            self._search_evidence_only(), self._held_out_evidence_only()
        )
        fields = {
            e["field"]: (e["old"], e["new"])
            for e in diff["config_drift"]
            if e["field"].startswith("fixtures:")
        }
        # old: search [f1], held_out []; new: search [], held_out [f9]
        self.assertEqual(
            fields,
            {
                "fixtures:search": (["f1"], []),
                "fixtures:held_out": ([], ["f9"]),
            },
        )

    def test_absent_diffs_on_empty_stages(self) -> None:
        """Missing-side fixtures appear as "absent" diffs on both stages."""
        diff = diffreports.diff_reports(
            self._search_evidence_only(), self._held_out_evidence_only()
        )
        cells = {(d["stage"], d["fixture"]): (d["old"], d["new"]) for d in diff["fixture_diffs"]}
        self.assertEqual(cells[("search", "f1")], ("pass", "absent"))
        self.assertEqual(cells[("held_out", "f9")], ("absent", "pass"))


if __name__ == "__main__":
    unittest.main()
