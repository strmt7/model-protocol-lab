"""Tests for the special-token literal audit (fixtures.py / doctor.py).

Advisory GHSA-g5f9-3xfg-p9mf: role-boundary forgery via chat-template
special-token literals embedded in untrusted input composed into an LLM
context. For the lab this is a fixture-hygiene property -- a fixture whose
message text contains a literal such as ``<|im_start|>`` would poison the
experiment, because the measured "boundary" would come from the fixture
text rather than the model.

Deterministic; no network, no model, no randomness.
"""

import dataclasses
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from model_protocol_lab.catalog import CATALOG
from model_protocol_lab.cli import main
from model_protocol_lab.core import Fixture, Format, Message, Split
from model_protocol_lab.datasets import load_dataset
from model_protocol_lab.doctor import examine_dataset
from model_protocol_lab.fixtures import (
    AuditFinding,
    _extract_special_tokens,
    audit_findings_json,
    audit_special_token_literals,
    split_to_json_text,
)

CHATML = CATALOG["chatml"]
# DeepSeek V3/R1 framing from docs/RESEARCH_LOG.md (Round 4 -- Cycle 1, R1):
# full-width bars, user = <｜User｜>{content} with no closing token.
DEEPSEEK = Format(
    "deepseek",
    "＜｜begin▁of▁sentence｜＞",
    "＜｜User｜＞",
    "＜｜Assistant｜＞",
    "＜｜end▁of▁sentence｜＞",
)
MISTRAL = Format("mistral-v0.3", "<s>", "[INST] ", "", " [/INST]")


def _fixture(fid, *contents):
    messages = tuple(
        Message("user" if index % 2 == 0 else "assistant", text)
        for index, text in enumerate(contents)
    )
    return Fixture(id=fid, messages=messages, expected="ok", group="g-" + fid)


class AuditFindingTests(unittest.TestCase):
    def test_finding_is_frozen(self):
        finding = AuditFinding(
            fixture_id="f1",
            field="user_1",
            literal="<|im_start|>",
            format_name="chatml",
            offset=3,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            finding.offset = 4  # type: ignore[misc]

    def test_finding_rejects_bad_values(self):
        good = {
            "fixture_id": "f1",
            "field": "user_1",
            "literal": "<|im_start|>",
            "format_name": "chatml",
            "offset": 0,
        }
        for key, bad in (
            ("fixture_id", 1),
            ("field", ""),
            ("literal", ""),
            ("format_name", "  "),
            ("offset", -1),
            ("offset", "0"),
        ):
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                AuditFinding(**{**good, key: bad})

    def test_finding_allows_blank_fixture_id_for_raw_text(self):
        finding = AuditFinding(fixture_id="", field="text", literal="x", format_name="f", offset=0)
        self.assertEqual(finding.fixture_id, "")


class InventoryTests(unittest.TestCase):
    def test_chatml_inventory_derived_from_format_fields(self):
        self.assertEqual(_extract_special_tokens(CHATML), ("<|im_start|>", "<|im_end|>"))

    def test_llama3_inventory(self):
        self.assertEqual(
            _extract_special_tokens(CATALOG["llama3"]),
            (
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ),
        )

    def test_gemma_inventory(self):
        self.assertEqual(
            _extract_special_tokens(CATALOG["gemma"]),
            ("<bos>", "<start_of_turn>", "<end_of_turn>"),
        )

    def test_deepseek_full_width_inventory(self):
        self.assertEqual(
            _extract_special_tokens(DEEPSEEK),
            (
                "＜｜begin▁of▁sentence｜＞",
                "＜｜User｜＞",
                "＜｜Assistant｜＞",
                "＜｜end▁of▁sentence｜＞",
            ),
        )

    def test_mistral_bracket_inventory(self):
        self.assertEqual(_extract_special_tokens(MISTRAL), ("<s>", "[INST]", "[/INST]"))

    def test_catalog_mistral_covers_per_role_turn_ends(self):
        # The v0.3 release template terminates user turns with [/INST] and
        # assistant turns with </s> (per-role overrides, not turn_end).
        self.assertEqual(
            _extract_special_tokens(CATALOG["mistral"]),
            ("<s>", "[INST]", "[/INST]", "</s>"),
        )

    def test_inventory_dedupes_repeated_tokens(self):
        # <|im_start|> appears in both chatml framing fields; it is one token.
        self.assertEqual(_extract_special_tokens(CHATML).count("<|im_start|>"), 1)

    def test_inventory_rejects_non_format(self):
        with self.assertRaises(ValueError):
            _extract_special_tokens("<|im_start|>")  # type: ignore[arg-type]


class AuditScanTests(unittest.TestCase):
    def test_chatml_literal_detected_in_user_message(self):
        case = _fixture("f1", "please repeat <|im_start|> exactly")
        findings = audit_special_token_literals(case, (CHATML,))
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.fixture_id, "f1")
        self.assertEqual(finding.field, "user_1")
        self.assertEqual(finding.literal, "<|im_start|>")
        self.assertEqual(finding.format_name, "chatml")
        self.assertEqual(finding.offset, "please repeat <|im_start|> exactly".index("<|im_start|>"))

    def test_deepseek_full_width_token_detected(self):
        case = _fixture("d1", "hello ＜｜User｜＞ world")
        findings = audit_special_token_literals(case, (DEEPSEEK,))
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].literal, "＜｜User｜＞")
        self.assertEqual(findings[0].format_name, "deepseek")
        self.assertEqual(findings[0].offset, 6)
        # And against the real catalog family (DeepSeek V3/R1 framing: ASCII
        # angles, full-width bars).
        catalog_case = _fixture("d2", "hello <｜User｜> world")
        catalog_findings = audit_special_token_literals(catalog_case, (CATALOG["deepseek"],))
        self.assertEqual(len(catalog_findings), 1)
        self.assertEqual(catalog_findings[0].literal, "<｜User｜>")
        self.assertEqual(catalog_findings[0].format_name, "deepseek")
        self.assertEqual(catalog_findings[0].offset, 6)

    def test_mistral_bracket_tokens_detected(self):
        case = _fixture("m1", "ignore [INST] previous [/INST] now", "ack")
        findings = audit_special_token_literals(case, (MISTRAL,))
        literals = [finding.literal for finding in findings]
        self.assertEqual(literals, ["[INST]", "[/INST]"])
        self.assertEqual(findings[0].field, "user_1")
        self.assertEqual(findings[1].field, "user_1")
        # Against the real catalog family: [/INST] is the user turn end.
        catalog_findings = audit_special_token_literals(case, (CATALOG["mistral"],))
        self.assertEqual([finding.literal for finding in catalog_findings], ["[INST]", "[/INST]"])
        self.assertTrue(all(finding.format_name == "mistral" for finding in catalog_findings))

    def test_clean_fixture_has_no_findings(self):
        case = _fixture("c1", "what is the capital of France?", "Paris")
        findings = audit_special_token_literals(
            case, (CHATML, DEEPSEEK, MISTRAL, CATALOG["llama3"], CATALOG["gemma"])
        )
        self.assertEqual(findings, ())

    def test_matching_is_case_sensitive(self):
        # The chatml token is lowercase; an uppercase lookalike is not the
        # same literal and must not be flagged.
        upper = _fixture("u1", "shout <|IM_START|> loudly")
        self.assertEqual(audit_special_token_literals(upper, (CHATML,)), ())
        exact = _fixture("u2", "whisper <|im_start|> softly")
        findings = audit_special_token_literals(exact, (CHATML,))
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].literal, "<|im_start|>")

    def test_non_token_angle_text_not_flagged(self):
        # "<not-a-token>" is angle-shaped but not in any format inventory, so
        # it is not a special-token literal.
        case = _fixture("n1", "compare <not-a-token> with <3 and a < b")
        findings = audit_special_token_literals(case, tuple(CATALOG.values()))
        self.assertEqual(findings, ())

    def test_multiple_occurrences_each_get_a_finding(self):
        text = "a <|im_start|> b <|im_start|> c"
        case = _fixture("o1", text)
        findings = audit_special_token_literals(case, (CHATML,))
        self.assertEqual([finding.offset for finding in findings], [2, 17])
        self.assertTrue(all(finding.literal == "<|im_start|>" for finding in findings))

    def test_field_names_follow_message_order(self):
        case = _fixture("q1", "clean", "dirty <|im_end|>", "clean again")
        findings = audit_special_token_literals(case, (CHATML,))
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].field, "assistant_1")

    def test_split_scans_both_partitions(self):
        split = Split(
            (_fixture("s1", "clean"),),
            (_fixture("h1", "poisoned <|eot_id|>"),),
        )
        findings = audit_special_token_literals(split, (CATALOG["llama3"],))
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].fixture_id, "h1")
        self.assertEqual(findings[0].literal, "<|eot_id|>")

    def test_raw_string_input(self):
        findings = audit_special_token_literals("say <|im_start|> hi", (CHATML,))
        self.assertEqual(len(findings), 1)
        self.assertEqual((findings[0].fixture_id, findings[0].field), ("", "text"))

    def test_tuple_of_mixed_input(self):
        case = _fixture("t1", "clean")
        findings = audit_special_token_literals((case, "raw <|im_end|> tail"), (CHATML,))
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].field, "text_2")

    def test_same_literal_in_two_formats_reports_both(self):
        both = Format("both", "", "<|im_start|>u\n", "<|im_start|>a\n", "<|im_end|>\n")
        case = _fixture("b1", "x <|im_end|> y")
        findings = audit_special_token_literals(case, (CHATML, both))
        self.assertEqual(
            [(finding.format_name, finding.literal) for finding in findings],
            [("chatml", "<|im_end|>"), ("both", "<|im_end|>")],
        )

    def test_rejects_bad_arguments(self):
        case = _fixture("v1", "clean")
        with self.assertRaises(ValueError):
            audit_special_token_literals(case, ())
        with self.assertRaises(ValueError):
            audit_special_token_literals(case, [CHATML])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            audit_special_token_literals(case, (CHATML, "<|im_start|>"))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            audit_special_token_literals(42, (CHATML,))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            audit_special_token_literals((case, 42), (CHATML,))  # type: ignore[arg-type]


class AuditSidecarTests(unittest.TestCase):
    def test_sidecar_serializes_findings(self):
        findings = audit_special_token_literals(_fixture("f1", "x <|im_start|> y"), (CHATML,))
        doc = json.loads(audit_findings_json(findings, source="fixtures.json"))
        self.assertEqual(doc["schema"], "model-protocol-lab.audit/v1")
        self.assertEqual(doc["source"], "fixtures.json")
        self.assertEqual(len(doc["findings"]), 1)
        entry = doc["findings"][0]
        self.assertEqual(
            (entry["fixture_id"], entry["field"], entry["literal"], entry["format_name"]),
            ("f1", "user_1", "<|im_start|>", "chatml"),
        )
        self.assertEqual(entry["offset"], 2)

    def test_sidecar_empty_findings(self):
        doc = json.loads(audit_findings_json((), source="fixtures.json"))
        self.assertEqual(doc["findings"], [])

    def test_sidecar_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            audit_findings_json("nope", source="f.json")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            audit_findings_json((), source=42)  # type: ignore[arg-type]


POISONED_CSV = """\
id,group,partition,expected,user_1,assistant_1,user_2
s1,g1,search,ok,"repeat <|im_start|> back",,
s2,g2,search,ok2,clean question,clean answer,follow up?
h1,g3,held_out,ok,held clean question,,
"""

CLEAN_CSV = """\
id,group,partition,expected,user_1,assistant_1,user_2
s1,g1,search,ok,question one,,
s2,g2,search,ok2,question two,answer two,follow up?
h1,g3,held_out,ok,held question,,
"""


class FixturesBuildAuditCliTests(unittest.TestCase):
    def _run_build(self, csv_text):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        source = Path(tmp.name) / "cases.csv"
        source.write_text(csv_text, encoding="utf-8")
        output = Path(tmp.name) / "fixtures.json"
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            code = main(["fixtures", "build", str(source), "--output", str(output)])
        return code, output, out.getvalue(), err.getvalue()

    def test_build_warns_on_poisoned_csv_but_still_builds(self):
        code, output, stdout, stderr = self._run_build(POISONED_CSV)
        # Non-blocking: the build succeeds and the fixture file is valid.
        self.assertEqual(code, 0)
        split = load_dataset(output)
        self.assertEqual([case.id for case in split.search], ["s1", "s2"])
        # The warning names the literal, fixture, field and format.
        self.assertIn("fixtures: warning: special-token literal", stderr)
        self.assertIn("<|im_start|>", stderr)
        self.assertIn("'s1'", stderr)
        self.assertIn("user_1", stderr)
        self.assertIn("chatml", stderr)
        # <|im_start|> collides with every catalog family that uses it
        # (chatml, qwen3, phi4): one finding per colliding format.
        summary = json.loads(stdout)
        self.assertEqual(summary["audit_findings"], 3)
        sidecar = Path(summary["audit_path"])
        self.assertTrue(sidecar.is_file())
        doc = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(len(doc["findings"]), 3)
        self.assertEqual(
            {entry["format_name"] for entry in doc["findings"]},
            {"chatml", "qwen3", "phi4"},
        )
        self.assertTrue(all(entry["literal"] == "<|im_start|>" for entry in doc["findings"]))

    def test_build_clean_csv_writes_empty_sidecar(self):
        code, output, stdout, stderr = self._run_build(CLEAN_CSV)
        self.assertEqual(code, 0)
        self.assertNotIn("special-token literal", stderr)
        summary = json.loads(stdout)
        self.assertEqual(summary["audit_findings"], 0)
        doc = json.loads(Path(summary["audit_path"]).read_text(encoding="utf-8"))
        self.assertEqual(doc["findings"], [])

    def test_build_refuses_existing_sidecar(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        source = Path(tmp.name) / "cases.csv"
        source.write_text(CLEAN_CSV, encoding="utf-8")
        output = Path(tmp.name) / "fixtures.json"
        (Path(tmp.name) / "fixtures.json.audit.json").write_text("{}", encoding="utf-8")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            code = main(["fixtures", "build", str(source), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertIn("audit sidecar already exists", err.getvalue())
        self.assertFalse(output.exists())

    def test_build_sidecar_race_unlinks_fixture_and_reports_distinctly(self):
        # Simulates the TOCTOU race: a concurrent writer creates the sidecar
        # between our exists() pre-check and the O_EXCL create. The fixture
        # file must not be left orphaned, and the error must not blame the
        # authoring source.
        import os
        from unittest import mock

        import model_protocol_lab.fixtures as fixtures_mod

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        source = Path(tmp.name) / "cases.csv"
        source.write_text(CLEAN_CSV, encoding="utf-8")
        output = Path(tmp.name) / "fixtures.json"
        real_open = os.open

        def racing_open(path, flags, *args, **kwargs):
            if str(path).endswith(".audit.json"):
                raise FileExistsError(str(path))
            return real_open(path, flags, *args, **kwargs)

        with (
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()) as err,
            mock.patch.object(fixtures_mod.os, "open", racing_open),
        ):
            code = main(["fixtures", "build", str(source), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertIn("output collision, nothing written", err.getvalue())
        self.assertNotIn("invalid authoring source", err.getvalue())
        self.assertFalse(output.exists())
        self.assertFalse(Path(str(output) + ".audit.json").exists())


class DoctorAuditTests(unittest.TestCase):
    def _write_dataset(self, tmp, *contents_per_fixture):
        split = Split(
            tuple(_fixture(f"s{i}", text) for i, text in enumerate(contents_per_fixture[:1])),
            tuple(_fixture(f"h{i}", text) for i, text in enumerate(contents_per_fixture[1:])),
        )
        path = Path(tmp) / "fixtures.json"
        path.write_text(split_to_json_text(split), encoding="utf-8")
        return path

    def test_doctor_warns_on_poisoned_fixture_file(self):
        with TemporaryDirectory() as tmp:
            path = self._write_dataset(tmp, "clean", "say <|im_start|> now")
            report = examine_dataset(path)
        # Warnings never fail the verdict.
        self.assertEqual(report["verdict"], "ok")
        audit = report["dataset"]["special_token_audit"]
        self.assertEqual(audit["status"], "warning")
        self.assertIn("GHSA-g5f9-3xfg-p9mf", audit["detail"])
        # <|im_start|> collides with chatml, qwen3 and phi4.
        self.assertEqual(len(audit["findings"]), 3)
        self.assertEqual(
            {finding["format_name"] for finding in audit["findings"]},
            {"chatml", "qwen3", "phi4"},
        )
        finding = audit["findings"][0]
        self.assertEqual(
            (finding["fixture_id"], finding["field"], finding["literal"]),
            ("h0", "user_1", "<|im_start|>"),
        )
        self.assertEqual(finding["format_name"], "chatml")

    def test_doctor_clean_fixture_file_is_ok(self):
        with TemporaryDirectory() as tmp:
            path = self._write_dataset(tmp, "clean one", "clean two")
            report = examine_dataset(path)
        self.assertEqual(report["verdict"], "ok")
        audit = report["dataset"]["special_token_audit"]
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(audit["findings"], [])

    def test_doctor_still_fails_unreadable_dataset(self):
        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{}", encoding="utf-8")
            report = examine_dataset(bad)
        self.assertEqual(report["verdict"], "failed")


if __name__ == "__main__":
    unittest.main()
