"""W2 push 6 regression tests: bug-hunt wave 10 repairs.

Wave 10 attacked W2's own changed surfaces and proved four
defects by execution (``agents_out/bughunt10``):

* F2 -- on the both-sides ``evidence_excluded`` path (push 5),
  stored-payload drift was never compared, so a changed
  ``expected_answer``, a changed ``answer_value`` and a removed
  record all still read ``identical: true`` on default exports
  (RR-09 reopened for the seven evidence-bound sections).
* F4 -- ``_valid_negation`` type-checked the protocol seed and
  fingerprint but never recomputed
  ``protocol_fingerprint(seed, backend_fingerprint)``, so
  protocol-field tampering validated (RR-03); the records were
  likewise never checked against the ``cases_for(seed)``
  content they claim to measure (L4).
* F1 -- the ``save_report`` admission gate's refusal path
  crashed with a raw ``TypeError`` on non-string top-level keys
  instead of the documented ``ValueError`` (ORCH-D01).
* L1 -- malformed core values (hostile ``held_out``/``search``
  shapes, a non-serializable value) crashed the projection
  with ``TypeError``/``AttributeError`` after passing the gate
  (RR-02 class in the new code).

And one adjudication (F3): ``serving_backends`` is evidence-
bound by the same mechanism as the seven (its record validator
recomputes both id digests from the stripped id streams), so
the ``evidence_excluded`` degradation extends to it -- a
genuine default-export pair must not read ``unvalidated``.

Expected values come from the contracts (the admission gate's
documented ``ValueError`` vocabulary; RR-09's rule that
observably different payloads never read identical; the
recompute standard the serving validator already meets), not
from the pre-fix output.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "tests")

from model_protocol_lab import diffreports, reporting  # noqa: E402


def _wrap(section_key: str, section: dict) -> dict:
    return {
        "schema_version": 1,
        "decision": "no_candidate",
        "selected_candidate": None,
        "search": [],
        "held_out": None,
        section_key: section,
    }


class AdmissionRefusalTotalityTests(unittest.TestCase):
    """F1: the refusal path is total over key types."""

    def _refused(self, payload: dict) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report"
            with self.assertRaises(ValueError):
                reporting.save_report(payload, target)
            self.assertFalse(target.exists())

    def test_non_string_keys_refuse_with_value_error(self) -> None:
        for payload in (
            {1: "x"},
            {True: "x"},
            {(1, 2): "x"},
            {None: "x"},
            {b"numform": {}},
            {"schema_version": 1, 5: {}, "numform": {}},
        ):
            with self.subTest(payload=repr(payload)[:40]):
                self._refused(payload)

    def test_mixed_type_unknown_keys_refuse_with_value_error(self) -> None:
        # sorted() itself raises TypeError comparing an int key
        # against a str key; the refusal must still be ValueError.
        self._refused({"schema_version": 1, 5: {}, "zzz_unknown": {}})

    def test_unknown_string_key_still_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError) as caught:
            reporting.save_report({"schema_version": 1, "invented_section": {}}, Path(tmp) / "r")
        self.assertIn("invented_section", str(caught.exception))


class ProjectionTotalityTests(unittest.TestCase):
    """L1: enrichment/projection are total over admitted envelopes."""

    def test_malformed_core_shapes_write_a_report(self) -> None:
        for payload in (
            {"schema_version": 1, "held_out": [1, 2, 3]},
            {"schema_version": 1, "held_out": "junk"},
            {"schema_version": 1, "search": {"a": 1}},
            {"schema_version": 1, "search": "junk"},
            {"schema_version": 1, "search": ["junk"]},
        ):
            with self.subTest(payload=repr(payload)[:48]), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "report"
                json_path, html_path = reporting.save_report(payload, target)
                self.assertTrue(json_path.exists())
                self.assertTrue(html_path.exists())
                written = json.loads(json_path.read_text(encoding="utf-8"))
                self.assertIs(written["raw_evidence_included"], False)

    def test_non_serializable_value_refuses_with_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report"
            with self.assertRaises(ValueError):
                reporting.save_report({"schema_version": 1, "seed": object()}, target)
            self.assertFalse(target.exists())

    def test_wellformed_core_shapes_still_strip(self) -> None:
        payload = {
            "schema_version": 1,
            "search": [{"format": {"name": "chatml"}, "observations": [{"x": 1}]}],
            "held_out": {"observations": [{"y": 2}], "summary": {}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            json_path, _ = reporting.save_report(payload, Path(tmp) / "report")
            written = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertNotIn("observations", written["search"][0])
        self.assertNotIn("observations", written["held_out"])


class EvidenceExcludedDriftTests(unittest.TestCase):
    """F2: observable drift on default exports breaks identical."""

    @classmethod
    def setUpClass(cls) -> None:
        from test_cycle186_numform import _NumformBackend, _payload, _run_stage

        backend = _NumformBackend()
        section = _payload(backend, _run_stage(backend))
        cls.public = reporting._public_view(_wrap("numform", section), False)

    def _pair(self) -> tuple[dict, dict]:
        return copy.deepcopy(self.public), copy.deepcopy(self.public)

    def _assert_drift_breaks_identical(self, diff: dict) -> None:
        section = diff["numform"]
        self.assertEqual(section["detail"], "evidence_excluded")
        self.assertIs(section["changed"], True)
        state = diff["section_states"]["numform"]
        self.assertEqual(state["state"], "evidence_excluded")
        self.assertIs(state["unvalidated_drift"], True)
        self.assertIn("numform", diff["summary"]["unvalidated_drift_sections"])
        self.assertIs(diff["summary"]["identical"], False)

    def test_identical_default_pair_stays_identical(self) -> None:
        old, new = self._pair()
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["numform"]["detail"], "evidence_excluded")
        self.assertIs(diff["numform"]["changed"], False)
        state = diff["section_states"]["numform"]
        self.assertEqual(state["state"], "evidence_excluded")
        self.assertIs(state["unvalidated_drift"], False)
        self.assertIs(diff["summary"]["identical"], True)

    def test_expected_answer_drift_is_compared(self) -> None:
        old, new = self._pair()
        record = next(iter(new["numform"]["candidates"].values()))["records"][0]
        record["expected_answer"] = record["expected_answer"] + " (tampered)"
        self._assert_drift_breaks_identical(diffreports.diff_reports(old, new))

    def test_answer_value_drift_is_compared(self) -> None:
        old, new = self._pair()
        record = next(iter(new["numform"]["candidates"].values()))["records"][0]
        record["answer_value"] = record["answer_value"] + 1
        self._assert_drift_breaks_identical(diffreports.diff_reports(old, new))

    def test_record_removal_is_compared(self) -> None:
        old, new = self._pair()
        records = next(iter(new["numform"]["candidates"].values()))["records"]
        records.pop(0)
        self._assert_drift_breaks_identical(diffreports.diff_reports(old, new))


class NegationProtocolRecomputeTests(unittest.TestCase):
    """F4 + L4: the protocol and the records are recomputed."""

    @classmethod
    def setUpClass(cls) -> None:
        from test_w2_negation import _genuine_section

        cls.section = _genuine_section(False)

    def _tampered(self, mutate) -> dict:
        section = copy.deepcopy(self.section)
        mutate(section)
        return section

    def _assert_invalid(self, section: dict) -> None:
        self.assertFalse(diffreports._valid_negation(section))

    def test_genuine_payload_is_valid(self) -> None:
        self.assertTrue(diffreports._valid_negation(copy.deepcopy(self.section)))

    def test_protocol_floor_tamper_is_refused(self) -> None:
        self._assert_invalid(self._tampered(lambda s: s["protocol"].__setitem__("min_joint", 0)))

    def test_protocol_budget_tamper_is_refused(self) -> None:
        self._assert_invalid(
            self._tampered(lambda s: s["protocol"].__setitem__("generation_budget", 1))
        )

    def test_protocol_arm_keys_tamper_is_refused(self) -> None:
        self._assert_invalid(self._tampered(lambda s: s["protocol"].__setitem__("arm_keys", ["x"])))

    def test_protocol_ground_digest_removal_is_refused(self) -> None:
        self._assert_invalid(self._tampered(lambda s: s["protocol"].pop("ground_sha256")))

    def test_protocol_fingerprint_forgery_is_refused(self) -> None:
        self._assert_invalid(
            self._tampered(lambda s: s["protocol"].__setitem__("fingerprint", "0" * 64))
        )

    def test_protocol_backend_swap_is_refused(self) -> None:
        self._assert_invalid(
            self._tampered(lambda s: s["protocol"].__setitem__("backend_fingerprint", "other"))
        )

    def test_protocol_seed_swap_is_refused(self) -> None:
        self._assert_invalid(self._tampered(lambda s: s["protocol"].__setitem__("seed", 7)))

    def _first_records(self, section: dict) -> list:
        entry = next(
            entry for entry in section["candidates"].values() if entry.get("status") == "available"
        )
        return entry["records"]

    def test_record_fixture_rename_is_refused(self) -> None:
        def mutate(section: dict) -> None:
            self._first_records(section)[0]["fixture"] = "no-such-case"

        self._assert_invalid(self._tampered(mutate))

    def test_record_operand_swap_is_refused(self) -> None:
        def mutate(section: dict) -> None:
            record = self._first_records(section)[0]
            record["operands"] = [record["operands"][1], record["operands"][0]]

        self._assert_invalid(self._tampered(mutate))

    def test_record_content_digest_tamper_is_refused(self) -> None:
        def mutate(section: dict) -> None:
            record = self._first_records(section)[0]
            arm = next(iter(record["content_digests"]))
            record["content_digests"][arm] = "0" * 64

        self._assert_invalid(self._tampered(mutate))

    def test_record_expected_digest_tamper_is_refused(self) -> None:
        def mutate(section: dict) -> None:
            self._first_records(section)[0]["expected_digest"] = "0" * 64

        self._assert_invalid(self._tampered(mutate))


class ServingDegradationTests(unittest.TestCase):
    """F3: serving_backends degrades like the other evidence-bound sections."""

    @classmethod
    def setUpClass(cls) -> None:
        from test_cycle243_serving_backends import ServingBackendsTestBase, _FakeServer

        from model_protocol_lab import experiments, servingbackends
        from model_protocol_lab.servingbackends import StackRequest

        case = ServingBackendsTestBase(methodName="run")
        case.setUp()
        try:
            eos = 50256
            with _FakeServer(case.gguf) as server:
                server.state.mode = "eos"
                backend = case.probed_backend(server)
                stage = servingbackends.run_stage(
                    backend,
                    plan_cells=[
                        {
                            "candidate": "t",
                            "fixture": "t0",
                            "request": StackRequest(prompt_ids=case.sentinel(), seed=7, eos_id=eos),
                            "cache_position": "standalone",
                        }
                    ],
                    fidelity_input={
                        "observations": [
                            {
                                "fixture": "fx",
                                "input_ids": list(case.sentinel()),
                                "output_ids": [6, eos],
                            }
                        ],
                        "stop_texts": ("STOP",),
                        "eos_id": eos,
                        "min_tokens": 0,
                    },
                    cache_prompts={
                        "cold_ids": case.sentinel(),
                        "shared_prefix_ids": case.sentinel() + (9,),
                        "divergent_control_ids": (8, 12),
                    },
                    seed=0,
                    decode=backend.decode_ids,
                    capabilities=backend.capabilities,
                )
                cls.section = experiments.serving_backends_analysis(stage)
        finally:
            case.tearDown()
        cls.evidence = reporting._public_view(
            _wrap("serving_backends", copy.deepcopy(cls.section)), True
        )
        cls.public = reporting._public_view(
            _wrap("serving_backends", copy.deepcopy(cls.section)), False
        )

    def test_genuine_section_is_valid(self) -> None:
        self.assertTrue(diffreports._valid_serving_backends(copy.deepcopy(self.section)))

    def test_evidence_pair_compares(self) -> None:
        diff = diffreports.diff_reports(copy.deepcopy(self.evidence), copy.deepcopy(self.evidence))
        self.assertEqual(diff["serving_backends"]["available"], "both")
        self.assertEqual(diff["section_states"]["serving_backends"]["state"], "compared")
        self.assertIs(diff["summary"]["identical"], True)

    def test_default_pair_degrades_not_unvalidated(self) -> None:
        diff = diffreports.diff_reports(copy.deepcopy(self.public), copy.deepcopy(self.public))
        section = diff["serving_backends"]
        self.assertEqual(section["detail"], "evidence_excluded")
        self.assertEqual(section["available"], "neither")
        self.assertIs(section["changed"], False)
        state = diff["section_states"]["serving_backends"]
        self.assertEqual(state["state"], "evidence_excluded")
        self.assertIs(state["unvalidated_drift"], False)
        self.assertIs(diff["summary"]["identical"], True)

    def test_one_sided_pair_names_the_evidence_side(self) -> None:
        diff = diffreports.diff_reports(copy.deepcopy(self.public), copy.deepcopy(self.evidence))
        section = diff["serving_backends"]
        self.assertEqual(section["detail"], "evidence_excluded")
        self.assertEqual(section["available"], "new")
        self.assertEqual(diff["section_states"]["serving_backends"]["state"], "one_sided")
        self.assertIs(diff["summary"]["identical"], False)

    def test_default_pair_drift_still_breaks_identical(self) -> None:
        drifted = copy.deepcopy(self.public)
        record = drifted["serving_backends"]["records"][0]
        record["returned_text_digest"] = "0" * 64
        diff = diffreports.diff_reports(copy.deepcopy(self.public), drifted)
        self.assertIs(diff["serving_backends"]["changed"], True)
        state = diff["section_states"]["serving_backends"]
        self.assertEqual(state["state"], "evidence_excluded")
        self.assertIs(state["unvalidated_drift"], True)
        self.assertIs(diff["summary"]["identical"], False)


if __name__ == "__main__":
    unittest.main()
