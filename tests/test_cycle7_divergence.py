"""Cycle 7: backend-divergence report section.

Covers the ``backend_divergences`` report plumbing:

- ``diffbackends.divergence_section_data``: per-observation
  differential testing over recorded observations, aggregated into a
  JSON-serializable section dict (agree case -> verdict "agree" with
  empty divergences; diverge case -> KEPT_VS_STRIPPED with severity,
  matrix counts and the one-line pairs verdict; graceful None when no
  observation is usable; strict validation of the other inputs);
- the pinned severity mapping (KEPT_VS_STRIPPED/EOS_PRECEDENCE=high,
  FIRING_STEP_DIFFERS/MIN_TOKENS_GATED=medium,
  HARNESS_NORMALIZATION_RELIED=low);
- ``reporting.render_html``: the "Backend divergence" section renders
  the verdict, the class x semantic count matrix with severity badges,
  and one <details> callout per divergence; hostile fixture names,
  stop texts and detail strings are escaped; rendering is
  byte-deterministic;
- the committed sample report (which has no ``backend_divergences``
  key) still regenerates byte-identically and carries no divergence
  section.

All backends are fakes decoding through fixed per-token text tables;
no torch, no network, no model weights.
"""

import importlib.util
import json
import unittest
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from model_protocol_lab.diffbackends import (
    DIVERGENCE_SEVERITY,
    divergence_section_data,
)
from model_protocol_lab.reporting import render_html

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "examples" / "make_sample_report.py"
COMMITTED = REPO_ROOT / "examples" / "sample-report"


class TableBackend:
    """Fake backend decoding token ids through a fixed per-token text table.

    Mirrors the cycle-6 diff-backend fake: whole-sequence decodes
    concatenate per-token texts.
    """

    fingerprint = "cycle7-divergence-table-backend-not-a-model"

    def __init__(self, texts: dict[int, str]) -> None:
        self._texts = dict(texts)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self._texts[token] for token in token_ids)


class DivergenceSectionDataTests(unittest.TestCase):
    def setUp(self) -> None:
        # 1->"a", 2->"b", 3->"STOP", 4->"c": stop "STOP" completes at step 3.
        self.backend = TableBackend({1: "a", 2: "b", 3: "STOP", 4: "c"})

    def _section(self, **kwargs: Any) -> dict[str, Any] | None:
        base: dict[str, Any] = {
            "observations": [{"fixture": "fx-1", "input_ids": (), "output_ids": (1, 2, 4)}],
            "stop_texts": ("STOP",),
            "eos_id": None,
            "min_tokens": 0,
            "backend": self.backend,
        }
        base.update(kwargs)
        return divergence_section_data(**base)

    def test_agree_when_no_stop_fires(self) -> None:
        section = self._section()
        self.assertIsNotNone(section)
        assert section is not None
        self.assertEqual(section["verdict"], "agree")
        self.assertEqual(
            section["pairs_verdict"], "HF vs vLLM stop semantics: 1/1 agree; 0 divergences"
        )
        self.assertEqual(len(section["per_fixture"]), 1)
        entry = section["per_fixture"][0]
        self.assertEqual(entry["fixture"], "fx-1")
        self.assertEqual(entry["verdict"], "agree")
        self.assertEqual(entry["divergences"], [])
        self.assertEqual(section["matrix"], {})
        self.assertEqual(section["stop_texts"], ["STOP"])
        self.assertIsNone(section["eos_id"])
        self.assertEqual(section["min_tokens"], 0)

    def test_diverge_kept_vs_stripped(self) -> None:
        observations = [
            {"fixture": "fx-agree", "input_ids": (), "output_ids": (1, 2, 4)},
            {"fixture": "fx-div", "input_ids": (), "output_ids": (1, 2, 3, 4)},
        ]
        section = self._section(observations=observations)
        self.assertIsNotNone(section)
        assert section is not None
        self.assertEqual(section["verdict"], "diverge")
        self.assertEqual(
            section["pairs_verdict"], "HF vs vLLM stop semantics: 1/2 agree; 1 divergence"
        )
        self.assertEqual(
            [entry["verdict"] for entry in section["per_fixture"]], ["agree", "diverge"]
        )
        findings = section["per_fixture"][1]["divergences"]
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding["class"], "KEPT_VS_STRIPPED")
        self.assertEqual(finding["stop_text"], "STOP")
        self.assertEqual(finding["severity"], "high")
        self.assertIn("vllm_holdback_strip", finding["detail"])
        self.assertIn("hf_stop_string", finding["detail"])
        self.assertEqual(
            section["matrix"],
            {"KEPT_VS_STRIPPED": {"hf_stop_string": 1, "vllm_holdback_strip": 1}},
        )
        self.assertEqual(section["severities"], {"KEPT_VS_STRIPPED": "high"})
        # The whole section dict must be JSON-serializable (it lands in
        # results.json).
        json.dumps(section, ensure_ascii=True, allow_nan=False)

    def test_observation_ids_accept_lists(self) -> None:
        # Recorded observations come back JSON-round-tripped: ids as lists.
        observations = [{"fixture": "fx-list", "input_ids": [], "output_ids": [1, 2, 4]}]
        section = self._section(observations=observations)
        self.assertIsNotNone(section)
        assert section is not None
        self.assertEqual(section["verdict"], "agree")
        self.assertEqual(section["per_fixture"][0]["fixture"], "fx-list")

    def test_none_when_no_observations_usable(self) -> None:
        self.assertIsNone(self._section(observations=[]))
        self.assertIsNone(self._section(observations=[{"fixture": "no-ids"}]))
        self.assertIsNone(
            self._section(
                observations=[{"fixture": "bad", "input_ids": "nope", "output_ids": (1,)}]
            )
        )
        self.assertIsNone(
            self._section(observations=[{"fixture": "neg", "input_ids": (), "output_ids": (-1,)}])
        )
        self.assertIsNone(self._section(observations=["not-a-dict"]))
        # One usable observation is enough; the rest are skipped.
        section = self._section(
            observations=[
                "not-a-dict",
                {"fixture": "bad", "input_ids": (), "output_ids": "nope"},
                {"fixture": "good", "input_ids": (), "output_ids": (1, 2, 4)},
            ]
        )
        self.assertIsNotNone(section)
        assert section is not None
        self.assertEqual(len(section["per_fixture"]), 1)
        self.assertEqual(section["per_fixture"][0]["fixture"], "good")

    def test_strict_validation(self) -> None:
        with self.assertRaises(ValueError):
            self._section(stop_texts=("",))
        with self.assertRaises(ValueError):
            self._section(stop_texts="STOP")
        with self.assertRaises(ValueError):
            self._section(min_tokens=-1)
        with self.assertRaises(ValueError):
            self._section(eos_id=-1)
        with self.assertRaises(ValueError):
            self._section(backend=object())
        with self.assertRaises(ValueError):
            self._section(observations="not-a-sequence")

    def test_severity_mapping_pinned(self) -> None:
        self.assertEqual(
            DIVERGENCE_SEVERITY,
            {
                "KEPT_VS_STRIPPED": "high",
                "EOS_PRECEDENCE": "high",
                "FIRING_STEP_DIFFERS": "medium",
                "MIN_TOKENS_GATED": "medium",
                "HARNESS_NORMALIZATION_RELIED": "low",
            },
        )

    def test_deterministic(self) -> None:
        observations = [
            {"fixture": "fx-agree", "input_ids": (), "output_ids": (1, 2, 4)},
            {"fixture": "fx-div", "input_ids": (), "output_ids": (1, 2, 3, 4)},
        ]
        first = self._section(observations=observations)
        second = self._section(observations=observations)
        self.assertEqual(
            json.dumps(first, sort_keys=True, ensure_ascii=True),
            json.dumps(second, sort_keys=True, ensure_ascii=True),
        )


class BackendDivergenceReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = TableBackend({1: "a", 2: "b", 3: "STOP", 4: "c"})

    def _render(self, result: dict[str, Any]) -> str:
        return render_html(result)

    def _section_result(self, section: dict[str, Any]) -> dict[str, Any]:
        return {"decision": "compatible_on_fixture_set", "backend_divergences": section}

    def _diverge_section(self) -> dict[str, Any]:
        section = divergence_section_data(
            observations=[
                {"fixture": "fx-agree", "input_ids": (), "output_ids": (1, 2, 4)},
                {"fixture": "fx-div", "input_ids": (), "output_ids": (1, 2, 3, 4)},
            ],
            stop_texts=("STOP",),
            eos_id=None,
            min_tokens=0,
            backend=self.backend,
        )
        self.assertIsNotNone(section)
        assert section is not None
        return section

    def test_section_absent_without_key(self) -> None:
        html = self._render({"decision": "compatible_on_fixture_set"})
        self.assertNotIn('id="backend-divergence"', html)
        self.assertNotIn("Backend divergence", html)

    def test_section_renders_verdict_matrix_and_callouts(self) -> None:
        html = self._render(self._section_result(self._diverge_section()))
        # TOC entry + section id + title.
        self.assertIn('href="#backend-divergence"', html)
        self.assertIn('id="backend-divergence"', html)
        self.assertIn("<h2>Backend divergence</h2>", html)
        # One-line verdict.
        self.assertIn(escape("HF vs vLLM stop semantics: 1/2 agree; 1 divergence"), html)
        self.assertIn('<span class="badge fail">diverge</span>', html)
        # Class x semantic matrix with severity badge.
        self.assertIn("KEPT_VS_STRIPPED", html)
        self.assertIn('<span class="badge fail">high</span>', html)
        self.assertIn("HF stop string", html)
        self.assertIn("vLLM strip", html)
        # One <details> callout per divergence.
        self.assertEqual(html.count("<details>"), 1)
        self.assertIn("fx-div", html)
        self.assertIn(escape("STOP"), html)

    def test_hostile_strings_escaped(self) -> None:
        hostile_fixture = "<img src=x onerror=alert(1)>"
        hostile_stop = "<script>alert(1)</script>"
        hostile_detail = 'detail with "quotes" & <tags> and \u00e9'
        section = divergence_section_data(
            observations=[
                {"fixture": hostile_fixture, "input_ids": (), "output_ids": (1, 2, 3, 4)},
            ],
            stop_texts=("STOP",),
            eos_id=None,
            min_tokens=0,
            backend=self.backend,
        )
        self.assertIsNotNone(section)
        assert section is not None
        # Hand-built section with fully hostile content (fixture names flow
        # through the builder, stop text/detail are exercised raw here).
        hostile_section: dict[str, Any] = {
            "verdict": "diverge",
            "pairs_verdict": "HF vs vLLM stop semantics: 0/1 agree; 1 divergence",
            "per_fixture": [
                {
                    "fixture": hostile_fixture,
                    "verdict": "diverge",
                    "divergences": [
                        {
                            "class": "KEPT_VS_STRIPPED",
                            "stop_text": hostile_stop,
                            "detail": hostile_detail,
                            "severity": "high",
                        }
                    ],
                }
            ],
            "matrix": {"KEPT_VS_STRIPPED": {"hf_stop_string": 1}},
            "severities": {"KEPT_VS_STRIPPED": "high"},
            "stop_texts": [hostile_stop],
            "eos_id": None,
            "min_tokens": 0,
        }
        html = self._render(self._section_result(hostile_section))
        for raw in (hostile_fixture, hostile_stop, hostile_detail):
            self.assertNotIn(raw, html)
            self.assertIn(escape(raw), html)
        # The builder-level fixture name is escaped too.
        builder_html = self._render(self._section_result(section))
        self.assertNotIn(hostile_fixture, builder_html)
        self.assertIn(escape(hostile_fixture), builder_html)

    def test_report_byte_deterministic(self) -> None:
        first = self._render(self._section_result(self._diverge_section()))
        second = self._render(self._section_result(self._diverge_section()))
        self.assertEqual(first, second)

    def test_no_remote_references(self) -> None:
        html = self._render(self._section_result(self._diverge_section()))
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)


class SampleReportUnchangedTests(unittest.TestCase):
    def test_sample_report_regenerates_byte_identically_without_divergence_section(
        self,
    ) -> None:
        spec = importlib.util.spec_from_file_location("make_sample_report", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with TemporaryDirectory() as tmp:
            module.build_sample_report(Path(tmp) / "regen")
            for name in ("results.json", "report.html"):
                committed = (COMMITTED / name).read_bytes()
                regenerated = (Path(tmp) / "regen" / name).read_bytes()
                self.assertEqual(regenerated, committed, name)
        results = json.loads((COMMITTED / "results.json").read_text(encoding="utf-8"))
        self.assertNotIn("backend_divergences", results)
        html = (COMMITTED / "report.html").read_text(encoding="utf-8")
        self.assertNotIn('id="backend-divergence"', html)


if __name__ == "__main__":
    unittest.main()
