"""W2 push 4 regression tests: RR-09 + ORCH-04.

RR-09: the diff's ``identical`` verdict used to omit whole axes of
evidence drift:

* a section that is *invalid on both sides* (its differ reports
  ``available: "neither"``) could carry completely different
  payloads and the diff still read ``identical: true`` -- the
  audit's verified reproductions drift a stored rate inside
  structured / cache_profile / confidentiality / answer_collision
  while both sides fail validation. The fix publishes a
  per-section state map (``section_states``): ``compared`` /
  ``one_sided`` / ``absent`` / ``unvalidated``, with
  ``unvalidated_drift`` set when both payloads are present but
  not recomputed-equal; sections in that state are named in
  ``summary["unvalidated_drift_sections"]`` and break
  ``identical``.
* the held-out stage's planned/attempted counts were never
  compared: a 1/1 held-out pass and a 50/50 held-out pass
  compared identical, the same lie of omission the search stage's
  ``planned:{name}`` config-drift entries already fixed. The
  held-out counts now produce ``held_out_planned:{name}`` /
  ``held_out_attempted:{name}`` config-drift entries under the
  same rules.

ORCH-04: a ``capability_absent`` stop-matrix cell (the stack
does not offer the capability, so nothing could be observed)
flowed through with ``fired: false`` and rendered in HTML as a
measured "no fire", indistinguishable from a genuinely measured
no-fire cell. The producer now records ``fired: None``
(unobservable) for unavailable cells, the diff mirror recomputes
the same shape, and the HTML renderer prints "unobservable" for
unavailable cells while measured cells keep "fired"/"no fire".

Expected values come from the audit reproducer
(``batch_d_orch_rr09.py``) and the sections' own contracts, not
from the implementation's output.
"""

from __future__ import annotations

import copy
import sys
import unittest

sys.path.insert(0, "tests")

from test_cycle243_serving_backends import ServingBackendsTestBase, _FakeServer  # noqa: E402
from test_w2_negation import _genuine_section, _wrap  # noqa: E402

from model_protocol_lab import diffreports, experiments, reporting, servingbackends  # noqa: E402
from model_protocol_lab.servingbackends import StackRequest  # noqa: E402


def _base() -> dict:
    """A minimal results payload with no sections."""
    return {
        "decision": "no_candidate",
        "selected_candidate": None,
        "search": [],
        "held_out": None,
    }


class UnvalidatedDriftTests(unittest.TestCase):
    """Both-sides-invalid sections must not silently drift."""

    def test_differing_invalid_cache_profile_breaks_identical(self):
        """RR-09: an invalid section's stored gap still compares."""
        old = _base()
        new = _base()
        old["cache_profile"] = {"available": True, "sharability_gap": 0.5}
        new["cache_profile"] = {"available": True, "sharability_gap": 0.9}
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["section_states"]["cache_profile"],
            {"state": "unvalidated", "unvalidated_drift": True},
        )
        self.assertEqual(diff["summary"]["unvalidated_drift_sections"], ["cache_profile"])
        self.assertFalse(diff["summary"]["identical"])

    def test_differing_invalid_confidentiality_breaks_identical(self):
        """RR-09: the generic-mechanism section is covered too."""
        old = _base()
        new = _base()
        old["confidentiality"] = {"available": True, "gap": 0.1}
        new["confidentiality"] = {"available": True, "gap": 0.4}
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["section_states"]["confidentiality"]["state"], "unvalidated")
        self.assertTrue(diff["section_states"]["confidentiality"]["unvalidated_drift"])
        self.assertFalse(diff["summary"]["identical"])

    def test_identical_invalid_payloads_do_not_drift(self):
        """Same garbage on both sides: unvalidated, but not drift."""
        old = _base()
        new = _base()
        old["cache_profile"] = {"available": True, "sharability_gap": 0.5}
        new["cache_profile"] = {"available": True, "sharability_gap": 0.5}
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["section_states"]["cache_profile"],
            {"state": "unvalidated", "unvalidated_drift": False},
        )
        self.assertEqual(diff["summary"]["unvalidated_drift_sections"], [])
        self.assertTrue(diff["summary"]["identical"])

    def test_absent_and_one_sided_states(self):
        """The state vocabulary: absent / one_sided / compared."""
        genuine = _genuine_section(False)
        old = _wrap(genuine)
        new = _wrap(None)
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["section_states"]["negation"]["state"], "one_sided")
        self.assertEqual(diff["section_states"]["structured"]["state"], "absent")
        both = diffreports.diff_reports(_wrap(genuine), _wrap(copy.deepcopy(genuine)))
        self.assertEqual(both["section_states"]["negation"]["state"], "compared")
        self.assertEqual(both["summary"]["unvalidated_drift_sections"], [])
        self.assertTrue(both["summary"]["identical"])

    def test_one_sided_invalid_payload_is_unvalidated_not_drift(self):
        """Garbage on one side only: state names it; drift needs both."""
        old = _base()
        new = _base()
        old["structured"] = {"available": True, "adherence_gap": 2.0}
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["section_states"]["structured"],
            {"state": "unvalidated", "unvalidated_drift": False},
        )

    def test_section_states_cover_the_registry(self):
        """Every dispatched section gets a state entry."""
        diff = diffreports.diff_reports(_base(), _base())
        self.assertEqual(set(diff["section_states"]), set(diffreports.SECTION_DIFF_KEYS.values()))
        for entry in diff["section_states"].values():
            self.assertIn(entry["state"], ("compared", "one_sided", "absent", "unvalidated"))
            self.assertIs(entry["unvalidated_drift"], False)


def _held_payload(planned: object, attempted: object, passes: object) -> dict:
    """A results payload whose held-out stage has the given counts."""
    payload = _base()
    payload["selected_candidate"] = "alpha"
    payload["held_out"] = {
        "candidate": "alpha",
        "summary": {
            "status": "complete",
            "planned": planned,
            "attempted": attempted,
            "aggregate_eligible": True,
            "exact_and_boundary_passes": passes,
            "rate": None,
            "encoded_input_tokens": 0,
            "retained_output_tokens": 0,
            "failure_code": None,
        },
    }
    return payload


class HeldOutCountDriftTests(unittest.TestCase):
    """RR-09: held-out evidence-strength counts are compared."""

    def test_one_of_one_vs_fifty_of_fifty_drifts(self):
        """The audit reproducer: same verdict, different evidence."""
        diff = diffreports.diff_reports(_held_payload(1, 1, 1), _held_payload(50, 50, 50))
        fields = {entry["field"]: entry for entry in diff["config_drift"]}
        self.assertEqual(
            fields["held_out_planned:alpha"],
            {"field": "held_out_planned:alpha", "old": 1, "new": 50},
        )
        self.assertEqual(
            fields["held_out_attempted:alpha"],
            {"field": "held_out_attempted:alpha", "old": 1, "new": 50},
        )
        self.assertFalse(diff["summary"]["identical"])

    def test_equal_counts_do_not_drift(self):
        """Same counts on both sides: no held-out entries."""
        diff = diffreports.diff_reports(_held_payload(50, 50, 50), _held_payload(50, 50, 50))
        fields = [entry["field"] for entry in diff["config_drift"]]
        self.assertFalse(any(field.startswith("held_out_") for field in fields))
        self.assertTrue(diff["summary"]["identical"])

    def test_attempted_only_drift_is_named(self):
        """Attempted can drift while planned and verdict hold."""
        diff = diffreports.diff_reports(_held_payload(50, 50, 30), _held_payload(50, 30, 30))
        fields = {entry["field"] for entry in diff["config_drift"]}
        self.assertIn("held_out_attempted:alpha", fields)
        self.assertNotIn("held_out_planned:alpha", fields)

    def test_malformed_count_compares_as_none(self):
        """A non-int count is its own value (the search convention)."""
        diff = diffreports.diff_reports(_held_payload("many", 1, 1), _held_payload(50, 1, 1))
        fields = {entry["field"]: entry for entry in diff["config_drift"]}
        self.assertEqual(fields["held_out_planned:alpha"]["old"], None)
        self.assertEqual(fields["held_out_planned:alpha"]["new"], 50)


def _absent_stage() -> dict:
    """A serving stage with one capability_absent standalone record."""
    return {
        "available": True,
        "records": [
            {
                "cache_position": "standalone",
                "mirror": {
                    "hf_stream": None,
                    "observed_disposition": None,
                    "fidelity_verdict": "capability_absent",
                },
                "stop_winner": None,
                "returned_ids": [],
                "request_config": {},
                "cell_detail": "capability_absent",
                "finish_class": None,
                "fixture": "f1",
                "candidate": "c1",
            }
        ],
    }


def _measured_stage() -> dict:
    """A serving stage with one measured no-fire standalone record."""
    stage = _absent_stage()
    record = stage["records"][0]
    record["mirror"] = {
        "hf_stream": None,
        "observed_disposition": "no_stop",
        "fidelity_verdict": "agree",
    }
    record["cell_detail"] = ""
    record["finish_class"] = "stop"
    record["returned_ids"] = [1, 2]
    return stage


def _renderable(section: dict) -> dict:
    """Wrap an analysis section for the HTML renderer.

    The renderer displays identity/capabilities (populated by the
    exec-backends stage in genuine runs); the hand-built stage
    leaves them None, so supply empty display dicts. The stop
    matrix under test is the producer's own output.
    """
    section = copy.deepcopy(section)
    section["identity"] = {}
    section["capabilities"] = {}
    return {"serving_backends": section}


class Orch04Tests(unittest.TestCase):
    """ORCH-04: unobservable cells are not measured negatives."""

    def test_absent_cell_records_fired_as_none(self):
        """The producer: capability_absent => fired is None."""
        section = experiments.serving_backends_analysis(_absent_stage())
        cell = section["stop_matrix"]["rows"][0]["cells"]["llamacpp"]
        self.assertIs(cell["available"], False)
        self.assertIsNone(cell["fired"])
        self.assertIsNone(cell["firing_step"])

    def test_measured_no_fire_cell_records_fired_false(self):
        """The control: a measured no-fire keeps fired False."""
        section = experiments.serving_backends_analysis(_measured_stage())
        cell = section["stop_matrix"]["rows"][0]["cells"]["llamacpp"]
        self.assertIs(cell["available"], True)
        self.assertIs(cell["fired"], False)

    def test_absent_cell_renders_unobservable_not_no_fire(self):
        """The HTML row for an absent cell says unobservable."""
        section = experiments.serving_backends_analysis(_absent_stage())
        html = reporting.render_html(_renderable(section))
        self.assertIn("unobservable", html)
        self.assertNotIn("no fire", html)

    def test_measured_cell_still_renders_no_fire(self):
        """The control row keeps the measured negative wording."""
        section = experiments.serving_backends_analysis(_measured_stage())
        html = reporting.render_html(_renderable(section))
        self.assertIn("no fire", html)


class Orch04DifferTests(ServingBackendsTestBase):
    """The diff mirror recomputes the unobservable shape honestly."""

    def test_differ_accepts_the_unobservable_shape(self) -> None:
        """A genuine capability_absent section validates both-sides."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(prompt_ids=self.sentinel(), seed=7, min_tokens=3),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t0",
            )
            self.assertEqual(record["cell_detail"], "capability_absent")
            section = experiments.serving_backends_analysis(
                {
                    "available": True,
                    "detail": "",
                    "identity": backend.identity.as_dict(),
                    "capabilities": backend.capabilities.as_dict(),
                    "records": [record],
                    "cache_sequence": None,
                    "fidelity": None,
                    "audit": {
                        "n_records": 1,
                        "n_planned_cells": 1,
                        "n_executions": 1,
                        "budget_spent": False,
                    },
                }
            )
            cell = section["stop_matrix"]["rows"][0]["cells"]["llamacpp"]
            self.assertIs(cell["available"], False)
            self.assertIsNone(cell["fired"])
            diff = diffreports.diff_serving_backends(section, copy.deepcopy(section))
            self.assertEqual(diff["available"], "both")
            self.assertFalse(diff["changed"])


if __name__ == "__main__":
    unittest.main()
