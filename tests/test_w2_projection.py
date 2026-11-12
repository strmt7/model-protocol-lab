"""W2 regression tests: ORCH-01 public-view projections and the
ORCH-D01 save_report admission gate.

ORCH-01 (projection census, sentinel-proven): the default
(evidence-excluded) export carried operator-private and raw
material with ``raw_evidence_included=false`` stamped on it --
``transfer.checkpoints[].path`` (operator filesystem path),
``serving_backends.records[].prompt_ids`` / ``returned_ids``
(exact token-ID streams), and raw ``answer_text`` in eight
instrument sections; ``serving_backends.identity.config``
passed through verbatim (latent: a future stack placing an
endpoint or path there would leak on day one). The projection
now strips exactly the census leak set in the default view and
leaves the evidence view intact.

Adjudication W2-D1: ``answer_text`` is evidence-only for all
eight sections. Seven of the eight sections' diff validators
recompute from ``answer_text`` (digest-bound mirrors; only
prune_sweep's never reads it), so diffing two default exports
degrades those seven sections to a DOCUMENTED unavailable
state (``detail == "evidence_excluded"``,
``section_states == "evidence_excluded"``) -- never a silent
"invalid". The shipped CLI contract already states numform
diffing "requires --include-evidence on both runs".

ORCH-D01: ``save_report`` admitted any dict (schema_version
999 saved silently). The admission gate refuses a non-dict, an
unsupported schema_version, and unknown top-level section
keys, before anything is enriched or written.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "tests")

from model_protocol_lab import diffreports, reporting  # noqa: E402

_SENTINEL_PATH = "SENTINEL-OPERATOR-PATH-9f3c"
_SENTINEL_TEXT = "SENTINEL-ANSWER-TEXT-9f3c"
_SENTINEL_ENDPOINT = "SENTINEL-ENDPOINT-9f3c"

_CELL_SECTIONS = (
    "prune_sweep",
    "record_notation",
    "invisible_chars",
    "confusable_skeleton",
    "compatibility_form",
    "enumeration_convention",
    "typographic_emphasis",
)


def _cell_section() -> dict:
    return {
        "available": True,
        "candidates": {
            "cand": {
                "status": "available",
                "records": [{"cells": {"arm": {"answer_text": _SENTINEL_TEXT, "verdict": True}}}],
            }
        },
    }


def _leaking_result() -> dict:
    result: dict = {
        "schema_version": 1,
        "transfer": {
            "checkpoints": [{"id": "ckpt-a", "path": _SENTINEL_PATH, "fingerprint": "fp-a"}]
        },
        "serving_backends": {
            "records": [{"prompt_ids": [101, 202], "returned_ids": [303], "cell": "c1"}],
            "identity": {
                "config": {
                    "protocol": "v1",
                    "stack": "fake",
                    "model_path_digest": "d" * 64,
                    "endpoint_url": _SENTINEL_ENDPOINT,
                    "model_path": _SENTINEL_PATH,
                },
                "config_fingerprint": "f" * 64,
            },
        },
        "numform": {
            "candidates": {
                "cand": {
                    "status": "available",
                    "records": [{"answer_text": _SENTINEL_TEXT, "task": "add"}],
                }
            }
        },
    }
    for key in _CELL_SECTIONS:
        result[key] = _cell_section()
    return result


class PublicViewProjectionTests(unittest.TestCase):
    """The census leak set is stripped in the default view."""

    def test_default_view_strips_every_census_leak(self) -> None:
        public = reporting._public_view(_leaking_result(), False)
        text = json.dumps(public)
        self.assertNotIn(_SENTINEL_PATH, text)
        self.assertNotIn(_SENTINEL_TEXT, text)
        self.assertNotIn(_SENTINEL_ENDPOINT, text)
        (checkpoint,) = public["transfer"]["checkpoints"]
        self.assertNotIn("path", checkpoint)
        self.assertEqual(checkpoint["id"], "ckpt-a")
        (record,) = public["serving_backends"]["records"]
        self.assertNotIn("prompt_ids", record)
        self.assertNotIn("returned_ids", record)
        self.assertEqual(record["cell"], "c1")
        for key in _CELL_SECTIONS:
            cell = public[key]["candidates"]["cand"]["records"][0]["cells"]["arm"]
            self.assertNotIn("answer_text", cell, key)
            self.assertIs(cell["verdict"], True)
        (num_record,) = public["numform"]["candidates"]["cand"]["records"]
        self.assertNotIn("answer_text", num_record)
        self.assertIs(public["raw_evidence_included"], False)

    def test_default_view_config_is_allowlisted(self) -> None:
        public = reporting._public_view(_leaking_result(), False)
        config = public["serving_backends"]["identity"]["config"]
        self.assertEqual(
            config,
            {
                "protocol": "v1",
                "stack": "fake",
                "model_path_digest": "d" * 64,
            },
        )
        identity = public["serving_backends"]["identity"]
        self.assertEqual(identity["config_fingerprint"], "f" * 64)

    def test_evidence_view_retains_everything(self) -> None:
        public = reporting._public_view(_leaking_result(), True)
        text = json.dumps(public)
        self.assertIn(_SENTINEL_PATH, text)
        self.assertIn(_SENTINEL_TEXT, text)
        self.assertIn(_SENTINEL_ENDPOINT, text)
        self.assertIs(public["raw_evidence_included"], True)

    def test_projection_never_mutates_the_input(self) -> None:
        result = _leaking_result()
        snapshot = json.dumps(result, sort_keys=True)
        reporting._public_view(result, False)
        reporting._public_view(result, True)
        self.assertEqual(json.dumps(result, sort_keys=True), snapshot)


class EvidenceExclusionDegradationTests(unittest.TestCase):
    """Genuine sections diffed from default exports degrade to
    the documented evidence_excluded state (W2-D1)."""

    @classmethod
    def setUpClass(cls) -> None:
        from test_cycle186_numform import _NumformBackend, _payload, _run_stage
        from test_cycle221_recordnotation import _payload as _rnot_payload

        backend = _NumformBackend()
        cls.numform = _payload(backend, _run_stage(backend))
        cls.record_notation = _rnot_payload()

    def _results(self, section_key: str, section: dict) -> dict:
        return {
            "schema_version": 1,
            "decision": "no_candidate",
            "selected_candidate": None,
            "search": [],
            "held_out": None,
            section_key: section,
        }

    def test_genuine_sections_validate_from_evidence_exports(self) -> None:
        for key in ("numform", "record_notation"):
            section = getattr(self, key)
            full = reporting._public_view(self._results(key, section), True)
            diff = diffreports.diff_reports(full, full)
            self.assertEqual(diff[key]["available"], "both", key)
            self.assertNotIn("detail", diff[key])
            self.assertTrue(diff["summary"]["identical"], key)

    def test_default_exports_degrade_to_documented_unavailable(self) -> None:
        for key in ("numform", "record_notation"):
            section = getattr(self, key)
            public = reporting._public_view(self._results(key, section), False)
            diff = diffreports.diff_reports(public, public)
            self.assertEqual(diff[key]["available"], "neither", key)
            self.assertEqual(diff[key]["detail"], "evidence_excluded", key)
            self.assertIs(diff[key]["changed"], False, key)
            self.assertEqual(diff["section_states"][key]["state"], "evidence_excluded", key)
            self.assertNotIn(key, diff["summary"]["unvalidated_drift_sections"], key)
            self.assertTrue(diff["summary"]["identical"], key)

    def test_one_sided_exclusion_is_asymmetric(self) -> None:
        public = reporting._public_view(self._results("numform", self.numform), False)
        full = reporting._public_view(self._results("numform", self.numform), True)
        diff = diffreports.diff_reports(public, full)
        self.assertEqual(diff["numform"]["available"], "new")
        self.assertEqual(diff["numform"]["detail"], "evidence_excluded")
        self.assertTrue(diff["summary"]["numform_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])
        self.assertEqual(diff["section_states"]["numform"]["state"], "one_sided")

    def test_unstamped_stripped_payload_gets_no_degradation_cover(self) -> None:
        """A hand-stripped payload without the export stamp is
        judged by the validators as-is: plain unavailable, no
        evidence_excluded detail, state unvalidated."""
        import copy

        bare = copy.deepcopy(self.numform)
        for entry in bare["candidates"].values():
            for record in entry["records"]:
                record.pop("answer_text", None)
        results = self._results("numform", bare)
        diff = diffreports.diff_reports(results, results)
        self.assertEqual(diff["numform"]["available"], "neither")
        self.assertNotIn("detail", diff["numform"])
        self.assertEqual(diff["section_states"]["numform"]["state"], "unvalidated")

    def test_partial_carriage_gets_no_degradation_cover(self) -> None:
        """Some records carrying answer_text and some not is not
        the projection's shape (it strips everywhere or
        nowhere): judged as-is, no detail."""
        public = reporting._public_view(self._results("numform", self.numform), False)
        entry = next(iter(public["numform"]["candidates"].values()))
        entry["records"][0]["answer_text"] = "planted"
        diff = diffreports.diff_reports(public, public)
        self.assertEqual(diff["numform"]["available"], "neither")
        self.assertNotIn("detail", diff["numform"])


class ProjectionMalformedShapeTests(unittest.TestCase):
    """The projection walks stored payloads defensively: malformed
    members are skipped untouched, never raised on, and the
    well-formed members around them are still stripped."""

    def test_section_level_records_and_mixed_members(self) -> None:
        result = {
            "numform": {
                "records": [
                    {"answer_text": _SENTINEL_TEXT, "task": "add"},
                    "junk-member",
                    {"cells": {"arm": {"answer_text": _SENTINEL_TEXT}, "bad": "not-a-cell"}},
                ]
            }
        }
        public = reporting._public_view(result, False)
        text = json.dumps(public)
        self.assertNotIn(_SENTINEL_TEXT, text)
        records = public["numform"]["records"]
        self.assertEqual(records[1], "junk-member")
        self.assertEqual(records[0]["task"], "add")
        self.assertEqual(records[2]["cells"]["bad"], "not-a-cell")

    def test_candidate_entry_shapes(self) -> None:
        result = {
            "numform": {
                "candidates": {
                    "junk": "not-an-entry",
                    "no_records": {"records": "not-a-list"},
                    "real": {"records": [{"answer_text": _SENTINEL_TEXT}]},
                }
            }
        }
        public = reporting._public_view(result, False)
        self.assertNotIn(_SENTINEL_TEXT, json.dumps(public))
        self.assertEqual(public["numform"]["candidates"]["junk"], "not-an-entry")
        self.assertEqual(public["numform"]["candidates"]["no_records"]["records"], "not-a-list")

    def test_section_shapes_that_skip_the_walk(self) -> None:
        result = {
            "numform": "not-a-section",
            "record_notation": {"records": "not-a-list", "candidates": "not-a-dict"},
            "transfer": "not-a-section",
            "serving_backends": {"records": "not-a-list", "identity": "not-a-dict"},
        }
        public = reporting._public_view(result, False)
        self.assertEqual(public["numform"], "not-a-section")
        self.assertEqual(public["transfer"], "not-a-section")
        self.assertEqual(public["serving_backends"]["identity"], "not-a-dict")

    def test_transfer_and_serving_mixed_members(self) -> None:
        result = {
            "transfer": {"checkpoints": ["junk", {"path": _SENTINEL_PATH, "id": "ckpt-b"}]},
            "serving_backends": {
                "records": ["junk", {"prompt_ids": [1], "returned_ids": [2], "cell": "c2"}],
                "identity": {"config": "not-a-dict"},
            },
        }
        public = reporting._public_view(result, False)
        self.assertNotIn(_SENTINEL_PATH, json.dumps(public))
        self.assertEqual(public["transfer"]["checkpoints"][0], "junk")
        self.assertEqual(public["serving_backends"]["records"][0], "junk")
        self.assertEqual(public["serving_backends"]["records"][1], {"cell": "c2"})
        self.assertEqual(public["serving_backends"]["identity"]["config"], "not-a-dict")


class DegradationMalformedShapeTests(unittest.TestCase):
    """The degradation detector reads the same defensive walks:
    malformed members inside a stamped export do not defeat the
    exclusion judgement, and shapes the detector cannot vouch
    for get no degradation cover."""

    def _stamped(self, section: object) -> dict:
        return {
            "schema_version": 1,
            "decision": "no_candidate",
            "selected_candidate": None,
            "search": [],
            "held_out": None,
            "raw_evidence_included": False,
            "numform": section,
        }

    def test_non_dict_record_member_still_degrades(self) -> None:
        section = {"records": ["junk-member", {"task": "add"}]}
        diff = diffreports.diff_reports(self._stamped(section), self._stamped(section))
        self.assertEqual(diff["numform"]["detail"], "evidence_excluded")
        self.assertEqual(diff["numform"]["available"], "neither")

    def test_candidate_and_cell_shapes_still_degrade(self) -> None:
        section = {
            "candidates": {
                "junk": "not-an-entry",
                "no_records": {"records": "not-a-list"},
                "real": {
                    "records": [
                        {"cells": "not-a-dict"},
                        {"cells": {"arm": "not-a-cell"}},
                        {"cells": {"arm": {"verdict": True}}},
                    ]
                },
            }
        }
        diff = diffreports.diff_reports(self._stamped(section), self._stamped(section))
        self.assertEqual(diff["numform"]["detail"], "evidence_excluded")

    def test_non_dict_section_gets_no_cover(self) -> None:
        diff = diffreports.diff_reports(self._stamped("junk"), self._stamped("junk"))
        self.assertNotIn("detail", diff["numform"])
        self.assertEqual(diff["section_states"]["numform"]["state"], "unvalidated")

    def test_one_sided_exclusion_with_absent_other_section(self) -> None:
        old = self._stamped({"records": [{"task": "add"}]})
        new = self._stamped({"records": [{"task": "add"}]})
        del new["numform"]
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["numform"]["detail"], "evidence_excluded")
        self.assertEqual(diff["numform"]["available"], "neither")


class SaveReportAdmissionTests(unittest.TestCase):
    """ORCH-D01: the envelope/section admission gate."""

    def _save(self, result: dict) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reporting.save_report(result, Path(tmp) / "report")

    def test_schema_version_999_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._save({"schema_version": 999})

    def test_schema_version_bool_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._save({"schema_version": True})

    def test_schema_version_string_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._save({"schema_version": "1"})

    def test_unknown_section_key_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._save({"schema_version": 1, "bogus_section": {}})

    def test_refusal_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report"
            with self.assertRaises(ValueError):
                reporting.save_report({"schema_version": 999}, target)
            self.assertFalse(target.exists())

    def test_admitted_section_keys_match_the_diff_registry(self) -> None:
        """reporting's literal admitted set cannot drift from
        the diff registry it mirrors (reporting must stay
        copy-isolatable, so it cannot import the registry)."""
        registry = set(diffreports.SECTION_DIFF_KEYS) | set(diffreports.UNDISPATCHED_SECTIONS)
        self.assertEqual(set(reporting._REPORT_SECTION_KEYS), registry)

    def test_genuine_envelope_and_registry_sections_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report"
            json_path, html_path = reporting.save_report(
                {
                    "schema_version": 1,
                    "decision": "abstain",
                    "numform": {"available": False},
                    "negation": {"available": False},
                },
                target,
            )
            self.assertTrue(json_path.exists())
            self.assertTrue(html_path.exists())
            saved = json.loads(json_path.read_text())
            self.assertIs(saved["raw_evidence_included"], False)


if __name__ == "__main__":
    unittest.main()
