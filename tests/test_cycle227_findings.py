"""Cycle 227: the deferred wave-2 correctness findings.

Bug-hunt wave 2 (coordinators #86/#89) and cybersecurity audit 2
(coordinator #87) confirmed sixteen defects against the cycle-225
head; cycle 226 absorbed two and defer-listed the rest, each
re-verified still open at its own final head. This file pins the
fix batch:

- F2 -- the sealed-bundle manifest parse had a ``parse_constant``
  hook but no ``parse_float`` hook, so a bundle-supplied ``1e999``
  became ``inf``, the verify verdict stayed ``ok``, and the report
  re-emitted the value as the bare token ``Infinity`` (invalid
  JSON). The manifest now carries the same hook pair the results
  parser has carried since cycle 202.
- C-02 -- ``structured.classify_response`` parsed model output with
  a plain ``json.loads``; NaN / ``1e999`` / duplicate keys in
  unchecked fields earned ``adherent``. Duplicate keys are now a
  parse failure (the shared discipline), and an undeclared field's
  subtree must be finite throughout -- the schema stage owns
  number finiteness in this module (cycle 158), so a non-finite
  number in an extra field is ``schema_failed``, the same verdict
  a checked field earns.
- C-03 -- the checkpoint JSON readers (``backends._read_json`` and
  ``doctor._check_weight_headers``) accepted duplicate keys,
  non-finite numbers, and lone surrogates; both now apply the same
  discipline and fail in their existing fixed vocabularies.
- BH2B-01/02 -- the invisible-chars and confusable-skeleton
  recompute mirrors (producer and both test oracles) subtracted
  ``arm_tokens`` before type-checking them, so a type-confused
  token count escaped ``diff_reports`` as a raw ``TypeError``.
  All four now validate first, like the older sections.
- F1 -- a truthy non-dict surface-noise ``baseline`` escaped the
  diff as a raw ``AttributeError``; the payload validator now
  checks the field like its siblings.
- F6 -- lone-surrogate fixture ids escaped five diff sections as
  raw ``UnicodeEncodeError`` at their join-encode digest sites;
  each site now fails closed like the cycle-190 site.
- BH2B-03/06/07 -- verify accepted summary-only forgeries, a vacuous
  planned == 0 winner, and a truthy-string aggregate_eligible: the
  summaries are now bound to their observations, planned must be
  >= 1, and eligibility is read as a strict bool.
- BH2B-04 -- the built-in surface-noise backends produced over-cap
  "limit" generations below max_new_tokens == 5, and the probe
  scored generations without the core contract check.
- BH2B-05 -- the cycle-67 oracle's ``identical`` conjunction (and
  the cycle-27 oracle's) omitted five sections, so the oracles
  called one-sided pairs identical while their own summaries
  reported the asymmetry; both conjunctions are complete now.
- BH2B-08 -- the cycle-67 oracle compared recomputed values with
  plain ``==`` where the producer compares type-strictly, so a
  bool-as-int or float-as-int forgery passed the oracle; the
  oracle's newest-section validators now use a type-strict
  equality with the producer's semantics.
- F3 -- a routine client disconnect killed the MCP server with a
  BrokenPipeError traceback (exit 120); it now ends the session
  cleanly. F4 -- type-confused hardening-notes reports escaped as
  -32603 instead of -32602. C-04 site 2 -- a local_hf construction
  failure on client params escaped _build_backend unwrapped;
  -32602 now, with cycle 196's -32603 pin deliberately moved.
  BH2B-09 -- the probe ate a pre-existing host SIGALRM timer;
  BH2B-10 -- a probe call from a non-main thread died as -32603.

Each test below is red on the pre-fix tree for the class it names.
"""

from __future__ import annotations

import copy
import io
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from model_protocol_lab import backends, diffreports, doctor, pack, structured, surfacenoise, verify

_SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
    "additionalProperties": True,
}


def _bundle_with_manifest(mutate: str) -> bytes:
    """Build a genuine bundle, then rewrite its manifest text."""
    src = Path(__file__).resolve().parent.parent / "examples" / "sample-report"
    bundle, _manifest, _digest = pack.pack_bundle(src, [])
    zin = zipfile.ZipFile(io.BytesIO(bundle))
    entries = [
        ("results.json", zin.read("results.json")),
        ("report.html", zin.read("report.html")),
    ]
    raw_manifest = zin.read("manifest.json").decode()
    if mutate == "context_1e999":
        mutated = raw_manifest.replace('"context":{', '"context":{"zz_probe":1e999,', 1)
    elif mutate == "toplevel_1e999":
        mutated = raw_manifest.rstrip()[:-1] + ',"zz_probe":1e999}'
    elif mutate == "context_infinity":
        mutated = raw_manifest.replace('"context":{', '"context":{"zz_probe":Infinity,', 1)
    elif mutate == "context_finite_float":
        mutated = raw_manifest.replace('"context":{', '"context":{"zz_probe":1.5,', 1)
    else:  # pragma: no cover - test-author error, not a code path
        raise AssertionError(f"unknown mutation {mutate}")
    return pack.build_bundle_bytes(mutated.encode(), entries)


def _schema_check(report: dict) -> dict:
    """Return the seal_manifest_schema check of a verify report."""
    return next(c for c in report["checks"] if c["id"] == "seal_manifest_schema")


class ManifestNonfiniteFloatTests(unittest.TestCase):
    """F2: an overflowing float literal is not manifest JSON."""

    def test_1e999_in_context_fails_manifest_schema(self):
        """Test 1e999 in context fails manifest schema."""
        report = pack.verify_bundle_bytes(_bundle_with_manifest("context_1e999"))
        check = _schema_check(report)
        self.assertEqual(check["status"], "failed")
        self.assertEqual(check["detail"], "manifest_unparseable")
        self.assertEqual(report["verdict"], "failed")

    def test_1e999_top_level_fails_manifest_schema(self):
        """Test 1e999 top level fails manifest schema."""
        report = pack.verify_bundle_bytes(_bundle_with_manifest("toplevel_1e999"))
        check = _schema_check(report)
        self.assertEqual(check["status"], "failed")
        self.assertEqual(check["detail"], "manifest_unparseable")
        self.assertEqual(report["verdict"], "failed")

    def test_literal_infinity_still_fails_manifest_schema(self):
        """Test literal Infinity still fails manifest schema."""
        report = pack.verify_bundle_bytes(_bundle_with_manifest("context_infinity"))
        check = _schema_check(report)
        self.assertEqual(check["status"], "failed")
        self.assertEqual(check["detail"], "manifest_unparseable")

    def test_finite_float_in_context_verdict_ok(self):
        """Test finite float in context verdict ok."""
        report = pack.verify_bundle_bytes(_bundle_with_manifest("context_finite_float"))
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual((report.get("context") or {}).get("zz_probe"), 1.5)

    def test_genuine_bundle_verdict_ok_and_report_strict_json(self):
        """Test genuine bundle verdict ok and report strict json."""
        src = Path(__file__).resolve().parent.parent / "examples" / "sample-report"
        bundle, _manifest, _digest = pack.pack_bundle(src, [])
        report = pack.verify_bundle_bytes(bundle)
        self.assertEqual(report["verdict"], "ok")
        # The emitted report must parse under a strict JSON reader
        # (no bare Infinity/NaN tokens) -- the pre-fix report for a
        # 1e999 manifest did not.
        rendered = json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True)

        def _reject(value: str) -> None:
            raise AssertionError(f"non-strict token in report: {value}")

        json.loads(rendered, parse_constant=_reject)


class ClassifyResponseDisciplineTests(unittest.TestCase):
    """C-02: classify_response applies the shared JSON discipline."""

    def test_nan_in_unchecked_field_is_schema_failed(self):
        """Test NaN in unchecked field is schema failed."""
        verdict = structured.classify_response('{"answer": "x", "note": NaN}', _SCHEMA)
        self.assertEqual(verdict["parse"], "parse_ok")
        self.assertEqual(verdict["schema"], "schema_failed")
        self.assertEqual(verdict["outcome"], "schema_failed")

    def test_overflowing_float_in_unchecked_field_is_schema_failed(self):
        """Test overflowing float in unchecked field is schema failed."""
        verdict = structured.classify_response('{"answer": "x", "note": 1e999}', _SCHEMA)
        self.assertEqual(verdict["parse"], "parse_ok")
        self.assertEqual(verdict["schema"], "schema_failed")
        self.assertEqual(verdict["outcome"], "schema_failed")

    def test_nonfinite_nested_in_unchecked_field_is_schema_failed(self):
        """Test nonfinite nested in unchecked field is schema failed."""
        verdict = structured.classify_response(
            '{"answer": "x", "note": {"deep": [1, Infinity]}}', _SCHEMA
        )
        self.assertEqual(verdict["outcome"], "schema_failed")

    def test_duplicate_keys_are_parse_failed(self):
        """Test duplicate keys are parse failed."""
        verdict = structured.classify_response('{"answer": "x", "answer": "y"}', _SCHEMA)
        self.assertEqual(verdict["parse"], "parse_failed")
        self.assertEqual(verdict["outcome"], "parse_failed")

    def test_finite_unchecked_field_still_adherent(self):
        """Test finite unchecked field still adherent."""
        verdict = structured.classify_response('{"answer": "x", "note": 1.5}', _SCHEMA)
        self.assertEqual(verdict["outcome"], "adherent")


class CheckpointJsonDisciplineTests(unittest.TestCase):
    """C-03: the checkpoint readers apply the shared discipline."""

    def _read(self, payload: str) -> dict:
        """Write payload as config.json in a temp dir and read it."""
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "config.json"
            target.write_text(payload, encoding="utf-8")
            return backends._read_json(target)

    def test_read_json_rejects_duplicate_keys(self):
        """Test read json rejects duplicate keys."""
        with self.assertRaises(ValueError) as ctx:
            self._read('{"a": 1, "a": 2}')
        self.assertEqual(str(ctx.exception), "invalid_asset_json")

    def test_read_json_rejects_nonfinite(self):
        """Test read json rejects nonfinite."""
        with self.assertRaises(ValueError) as ctx:
            self._read('{"a": 1e999}')
        self.assertEqual(str(ctx.exception), "invalid_asset_json")

    def test_read_json_rejects_literal_constant(self):
        """Test read json rejects literal constant."""
        with self.assertRaises(ValueError) as ctx:
            self._read('{"a": NaN}')
        self.assertEqual(str(ctx.exception), "invalid_asset_json")

    def test_read_json_rejects_lone_surrogate(self):
        """Test read json rejects lone surrogate."""
        with self.assertRaises(ValueError) as ctx:
            self._read('{"a": "\\ud800"}')
        self.assertEqual(str(ctx.exception), "invalid_asset_json")

    def test_read_json_wellformed_unchanged(self):
        """Test read json wellformed unchanged."""
        self.assertEqual(self._read('{"a": 1, "b": [1.5, "x"]}'), {"a": 1, "b": [1.5, "x"]})

    def _header_check(self, header_text: str) -> dict:
        """Run the weight-header check on a crafted shard."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            header = header_text.encode("utf-8")
            (root / "model.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header + b"\x00" * 16
            )
            return doctor._check_weight_headers(root, ("model.safetensors",))

    def test_weight_headers_reject_duplicate_keys(self):
        """Test weight headers reject duplicate keys."""
        result = self._header_check('{"__metadata__": {"a": "1", "a": "2"}}')
        self.assertEqual(result["status"], "failed")

    def test_weight_headers_reject_nonfinite(self):
        """Test weight headers reject nonfinite."""
        result = self._header_check('{"__metadata__": {"a": 1e999}}')
        self.assertEqual(result["status"], "failed")

    def test_weight_headers_reject_literal_constant(self):
        """Test weight headers reject literal constant."""
        result = self._header_check('{"__metadata__": {"a": Infinity}}')
        self.assertEqual(result["status"], "failed")

    def test_weight_headers_reject_lone_surrogate(self):
        """Test weight headers reject lone surrogate."""
        result = self._header_check('{"__metadata__": {"a": "\\ud800"}}')
        self.assertEqual(result["status"], "failed")

    def test_weight_headers_finite_float_ok(self):
        """Test weight headers finite float ok."""
        result = self._header_check('{"__metadata__": {"a": 1.5}}')
        self.assertEqual(result["status"], "ok")

    def test_weight_headers_wellformed_unchanged(self):
        """Test weight headers wellformed unchanged."""
        result = self._header_check('{"__metadata__": {"format": "pt"}}')
        self.assertEqual(result["status"], "ok")


def _t67():
    """Import the cycle-67 oracle module (builders + mirrors)."""
    from tests import test_cycle67_diffreports_audit as t67

    return t67


class ArmTokensTypeConfusionTests(unittest.TestCase):
    """BH2B-01/02: arm_tokens arithmetic happens after validation."""

    def test_invisible_chars_string_arm_tokens_invalid_not_raised(self):
        """Test invisible chars string arm tokens invalid not raised."""
        t67 = _t67()
        payload = t67.InvisibleCharsOracleAgreementTests._payload("perfect")
        cand = sorted(payload["candidates"])[0]
        record = payload["candidates"][cand]["records"][0]
        arm = sorted(k for k in record["arm_tokens"] if k != "control")[0]
        record["arm_tokens"][arm] = "x"
        self.assertFalse(diffreports._valid_invisible_chars(copy.deepcopy(payload)))
        self.assertFalse(t67._o_invisiblechars_valid(copy.deepcopy(payload)))

    def test_confusable_control_arm_tokens_list_invalid_not_raised(self):
        """Test confusable control arm tokens list invalid not raised."""
        t67 = _t67()
        from tests import test_cycle224_confusableskeleton as t224

        payload = t224._payload("perfect")[0]
        cand = sorted(payload["candidates"])[0]
        record = payload["candidates"][cand]["records"][0]
        record["arm_tokens"]["control"] = [1]
        self.assertFalse(diffreports._valid_confusable_skeleton(copy.deepcopy(payload)))
        self.assertFalse(t67._o_confusableskeleton_valid(copy.deepcopy(payload)))

    def test_diff_reports_survives_type_confused_arm_tokens(self):
        """Test diff reports survives type confused arm tokens."""
        t67 = _t67()
        payload = t67.InvisibleCharsOracleAgreementTests._payload("perfect")
        cand = sorted(payload["candidates"])[0]
        payload["candidates"][cand]["records"][0]["arm_tokens"]["control"] = "x"
        old = t67._base_report()
        new = t67._base_report()
        old["invisible_chars"] = t67.InvisibleCharsOracleAgreementTests._payload("perfect")
        new["invisible_chars"] = payload
        out = diffreports.diff_reports(old, new)
        self.assertEqual(out["invisible_chars"]["available"], "old")


class OracleIdenticalCompletenessTests(unittest.TestCase):
    """BH2B-05: the oracle identical conjunction covers all sections."""

    def test_one_sided_history_frame_not_identical_in_oracle(self):
        """Test one sided history frame not identical in oracle."""
        t67 = _t67()
        old = t67._base_report()
        new = t67._base_report()
        old["history_frame"] = t67.HistoryFrameOracleAgreementTests._payload("perfect")
        want = t67._oracle_diff(copy.deepcopy(old), copy.deepcopy(new))
        self.assertTrue(want["summary"]["history_frame_asymmetric"])
        self.assertFalse(want["summary"]["identical"])

    def test_one_sided_invisible_chars_not_identical_in_oracle(self):
        """Test one sided invisible chars not identical in oracle."""
        t67 = _t67()
        old = t67._base_report()
        new = t67._base_report()
        old["invisible_chars"] = t67.InvisibleCharsOracleAgreementTests._payload("perfect")
        want = t67._oracle_diff(copy.deepcopy(old), copy.deepcopy(new))
        self.assertTrue(want["summary"]["invisible_chars_asymmetric"])
        self.assertFalse(want["summary"]["identical"])


class OracleTypeStrictEqualityTests(unittest.TestCase):
    """BH2B-08: the oracle compares recomputed values type-strictly."""

    def test_confusable_float_delta_rejected_by_oracle(self):
        """Test confusable float delta rejected by oracle."""
        t67 = _t67()
        from tests import test_cycle224_confusableskeleton as t224

        payload = t224._payload("perfect", fold=True)[0]
        deltas = payload["candidates"]["chatml"]["ids"]["whole_word"]["id_length_delta"]
        deltas[deltas.index(0)] = 0.0
        self.assertFalse(diffreports._valid_confusable_skeleton(copy.deepcopy(payload)))
        self.assertFalse(t67._o_confusableskeleton_valid(copy.deepcopy(payload)))

    def test_invisible_bool_version_rejected_by_oracle(self):
        """Test invisible bool version rejected by oracle."""
        t67 = _t67()
        payload = t67.InvisibleCharsOracleAgreementTests._payload("perfect")
        payload["protocol"]["version"] = True
        self.assertFalse(diffreports._valid_invisible_chars(copy.deepcopy(payload)))
        self.assertFalse(t67._o_invisiblechars_valid(copy.deepcopy(payload)))

    def test_invisible_bool_n_candidates_rejected_by_oracle(self):
        """Test invisible bool n candidates rejected by oracle."""
        t67 = _t67()
        payload = t67.InvisibleCharsOracleAgreementTests._payload("perfect")
        payload["n_candidates"] = True
        self.assertFalse(diffreports._valid_invisible_chars(copy.deepcopy(payload)))
        self.assertFalse(t67._o_invisiblechars_valid(copy.deepcopy(payload)))


def _probe_payload(baseline: object) -> dict:
    """Build a minimal valid surface-noise probe payload."""
    return {
        "instrument": "surface-noise-probe",
        "status": "ok",
        "protocol": {"backend_fingerprint": "fp", "search_plan": [], "fixture_sha256": "x"},
        "prompt_sha256": "abc",
        "operators": {
            "op": {
                "stability": 1.0,
                "n_effective": 1,
                "n_total": 1,
                "nearest_observed_flip": {"distance": 1, "variant_id": "v"},
            }
        },
        "baseline": baseline,
    }


class SurfaceNoiseBaselineTests(unittest.TestCase):
    """F1: a type-confused baseline is invalid, never an exception."""

    def test_truthy_non_dict_baselines_are_invalid(self):
        """Test truthy non dict baselines are invalid."""
        for baseline in ("hello", 7, 1.5, True, [1, 2]):
            with self.subTest(baseline=baseline):
                out = diffreports.diff_surface_noise(
                    _probe_payload(baseline), _probe_payload(baseline)
                )
                self.assertEqual(out["available"], "neither")

    def test_dict_baseline_control_still_identical(self):
        """Test dict baseline control still identical."""
        baseline = {"n_pass": 3, "n_fail": 0, "n_abstain": 0}
        out = diffreports.diff_surface_noise(
            _probe_payload(baseline), _probe_payload(dict(baseline))
        )
        self.assertTrue(out["identical"])


class SurrogateFixtureIdTests(unittest.TestCase):
    """F6: lone-surrogate fixture ids fail closed in diff sections."""

    def _repetition_payload(self, fixture_id: str) -> dict:
        """Build a results payload whose repetition record uses the id."""
        base = json.loads(
            (Path(__file__).resolve().parent.parent / "examples" / "sample-report" / "results.json")
            .read_bytes()
            .decode()
        )
        cell = {
            "verdict": True,
            "boundary_found": True,
            "boundary_text_match": True,
            "input_tokens": 5,
        }
        record = {
            "fixture": fixture_id,
            "vacuous": False,
            "pre_repeated": False,
            "echo_fixture": False,
            "pad_evaluable": True,
            "construction": "id_exact",
            "seam_ids": [1],
            "delta_tokens": 3,
            "filler_tokens": 4,
            "arms": {"x1": dict(cell), "x2": dict(cell), "pad": dict(cell)},
            "arm_details": {"x2": "", "pad": ""},
        }
        base["repetition"] = {
            "available": True,
            "primary_decision": "no_candidate",
            "primary_winner": None,
            "protocol": {
                "joiner": " ",
                "filler_unit_sha256": "a" * 64,
                "sizing_rule": "x",
                "fixture_set_sha256": "b" * 64,
                "backend_fingerprint": "fp",
                "fingerprint": "c" * 64,
                "version": 1,
                "detector_version": 1,
            },
            "candidates": {
                "cand": {
                    "status": "available",
                    "think_channel": False,
                    "render_signature": None,
                    "records": [record],
                }
            },
        }
        return base

    def test_surrogate_fixture_id_repetition_neither_not_raised(self):
        """Test surrogate fixture id repetition neither not raised."""
        hostile = self._repetition_payload("\ud800")
        control = self._repetition_payload("f1")
        out = diffreports.diff_reports(hostile, control)
        self.assertEqual(out["repetition"]["available"], "neither")

    @staticmethod
    def _surrogate_fixture_ids(payload: object) -> None:
        """Append a lone surrogate to every fixture id in a payload.

        The mapping is consistent (the same id maps the same way
        everywhere, donor references included), so the payload
        stays self-consistent right up to the fixture-set digest,
        whose join-encode is the site under test.
        """
        stack: list[object] = [payload]
        while stack:
            item = stack.pop()
            if type(item) is dict:
                for key in ("fixture", "donor_fixture"):
                    value = item.get(key)
                    if type(value) is str and "\ud800" not in value:
                        item[key] = value + "\ud800"
                stack.extend(item.values())
            elif type(item) is list:
                stack.extend(item)

    def test_surrogate_fixture_id_aliasing_invalid_not_raised(self):
        """Test surrogate fixture id aliasing invalid not raised."""
        from model_protocol_lab.core import Fixture, Generation, Message, Split
        from model_protocol_lab.experiments import aliasing_profile, run_experiment

        class _StubBackend:
            """Stub backend: every prompt encodes to the same ids."""

            fingerprint = "sha256:stub"
            eos_token_id = None

            @staticmethod
            def encode(text: str) -> tuple[int, ...]:
                """Encode."""
                return (1, 2, 3)

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel):  # noqa: ANN001, ANN202
                """Generate."""
                return Generation((5, 99, 5, 99)[:max_new_tokens], "limit")

            @staticmethod
            def decode(ids) -> str:  # noqa: ANN001
                """Decode."""
                return "yo"

        from model_protocol_lab.core import Format

        fmt_a = Format(name="alpha", bos="", user_prefix="U:", assistant_prefix="A:", turn_end="\n")
        fmt_b = Format(
            name="beta",
            bos="",
            user_prefix="U:",
            assistant_prefix="A:",
            turn_end="\n",
            repeat_bos=True,
        )
        case = Fixture(
            id="s1",
            group="g1",
            messages=(Message(role="user", content="hi"),),
            expected="yo",
        )
        held = Fixture(
            id="h1",
            group="g2",
            messages=(Message(role="user", content="hey"),),
            expected="yo",
        )
        result = run_experiment(
            _StubBackend(),
            (fmt_a, fmt_b),
            Split(search=(case,), held_out=(held,)),
            boundary=(99,),
            max_new_tokens=4,
            seed=3,
        )
        payload = aliasing_profile(result)
        self.assertTrue(diffreports._valid_aliasing(copy.deepcopy(payload)))
        self._surrogate_fixture_ids(payload)
        self.assertFalse(diffreports._valid_aliasing(payload))

    def test_surrogate_fixture_id_dateline_invalid_not_raised(self):
        """Test surrogate fixture id dateline invalid not raised."""
        from tests import test_cycle188_dateline as t188

        payload = t188._base_payload()
        self.assertTrue(diffreports._valid_dateline(copy.deepcopy(payload)))
        self._surrogate_fixture_ids(payload)
        self.assertFalse(diffreports._valid_dateline(payload))

    def test_surrogate_fixture_id_leading_invalid_not_raised(self):
        """Test surrogate fixture id leading invalid not raised."""
        from tests import test_cycle185_leading as t185

        backend = t185._ClaimBackend(expected_by_marker=t185.EXPECTED_BY_MARKER)
        payload = t185._payload(backend, t185._run_stage(backend))
        self.assertTrue(diffreports._valid_leading(copy.deepcopy(payload)))
        self._surrogate_fixture_ids(payload)
        self.assertFalse(diffreports._valid_leading(payload))

    def test_surrogate_fixture_id_distraction_invalid_not_raised(self):
        """Test surrogate fixture id distraction invalid not raised."""
        from tests import test_cycle184_distraction as t184

        backend = t184._WordBackend(t184.RULES, t184.EXPECTED_BY_MARKER)
        payload = t184._payload(backend, t184._run_stage(backend))
        self.assertTrue(diffreports._valid_distraction(copy.deepcopy(payload)))
        self._surrogate_fixture_ids(payload)
        self.assertFalse(diffreports._valid_distraction(payload))

    def test_surrogate_fixture_id_placement_invalid_not_raised(self):
        """Test surrogate fixture id placement invalid not raised."""
        from tests import test_cycle193_placement as t193

        payload = t193._payload()
        self.assertTrue(diffreports._valid_placement(copy.deepcopy(payload)))
        self._surrogate_fixture_ids(payload)
        self.assertFalse(diffreports._valid_placement(payload))


def _verify_artifact() -> dict:
    """A genuine two-candidate artifact: A passes all, B passes s1."""
    from model_protocol_lab.core import Fixture, Format, Generation, Message, Split
    from model_protocol_lab.experiments import run_experiment

    boundary = (999999,)
    expected = {"s1": "one", "s2": "two", "h1": "three"}
    plan = {(name, fid): False for name in "AB" for fid in expected}
    plan[("A", "s1")] = plan[("A", "s2")] = plan[("A", "h1")] = True
    plan[("B", "s1")] = True

    class _Backend:
        """Scripted backend answering per the plan."""

        fingerprint = "cycle227-scripted"

        def encode(self, text: str) -> tuple[int, ...]:
            """Encode."""
            return tuple(ord(c) for c in text)

        def decode(self, ids) -> str:  # noqa: ANN001
            """Decode."""
            return "".join(chr(i) for i in ids if i < 0x110000)

        def generate(self, input_ids, max_new_tokens=64, seed=0, cancel=None):  # noqa: ANN001, ANN202
            """Generate."""
            prompt = self.decode(input_ids)
            cand = prompt[1 : prompt.index(">")]
            fid = prompt.split("U:")[1].split("\n")[0]
            ans = expected[fid] if plan.get((cand, fid)) else "WRONG"
            ids = self.encode(ans) + boundary
            ids = ids + (32,) * (max_new_tokens - len(ids))
            return Generation(token_ids=ids, reason="limit")

    def _fmt(name: str) -> Format:
        return Format(
            name=name, bos=f"<{name}>", user_prefix="U:", assistant_prefix="A:", turn_end="\n"
        )

    def _fx(fid: str, grp: str) -> Fixture:
        return Fixture(id=fid, messages=(Message("user", fid),), expected=expected[fid], group=grp)

    split = Split(search=(_fx("s1", "g1"), _fx("s2", "g2")), held_out=(_fx("h1", "g3"),))
    return run_experiment(
        _Backend(), (_fmt("A"), _fmt("B")), split, boundary=boundary, max_new_tokens=32
    )


def _summaries_detail(data: dict) -> str:
    """Return the summaries_consistent check detail of a verify run."""
    report = verify.verify_results(data)
    for check in report["checks"]:
        if check["id"] == "summaries_consistent":
            return check["status"] + ": " + check["detail"]
    raise AssertionError("summaries_consistent check missing")


class VerifySummaryBindingTests(unittest.TestCase):
    """BH2B-03/06/07: verify binds summaries to their observations."""

    def test_genuine_artifact_verifies_ok(self):
        """Test genuine artifact verifies ok."""
        report = verify.verify_results(_verify_artifact())
        self.assertEqual(report["verdict"], "ok")
        self.assertTrue(_summaries_detail(_verify_artifact()).startswith("passed"))

    def test_summary_only_forge_fails(self):
        """Test summary only forge fails."""
        data = _verify_artifact()
        for entry in data["search"]:
            if entry["format"]["name"] == "B":
                entry["summary"]["exact_and_boundary_passes"] = entry["summary"]["planned"]
        data["held_out"]["summary"]["exact_and_boundary_passes"] = data["held_out"]["summary"][
            "planned"
        ]
        self.assertIn("do not match the observations", _summaries_detail(data))
        self.assertNotEqual(verify.verify_results(data)["verdict"], "ok")

    def test_observation_flip_fails(self):
        """Test observation flip fails."""
        data = _verify_artifact()
        data["search"][0]["observations"][0]["exact_match"] = False
        self.assertIn("do not match the observations", _summaries_detail(data))

    def test_held_out_observation_flip_fails(self):
        """Test held out observation flip fails."""
        data = _verify_artifact()
        data["held_out"]["observations"][0]["exact_match"] = False
        detail = _summaries_detail(data)
        self.assertIn("held_out", detail)
        self.assertIn("do not match the observations", detail)

    def test_vacuous_planned_zero_fails(self):
        """Test vacuous planned zero fails."""
        data = _verify_artifact()
        for entry in data["search"]:
            entry.pop("observations", None)
            if entry["format"]["name"] == "B":
                entry["summary"]["planned"] = 0
                entry["summary"]["exact_and_boundary_passes"] = 0
                entry["summary"]["attempted"] = 0
        self.assertIn("positive integer", _summaries_detail(data))
        self.assertNotEqual(verify.verify_results(data)["verdict"], "ok")

    def test_attempted_mismatch_fails(self):
        """Test attempted mismatch fails."""
        data = _verify_artifact()
        data["search"][0]["summary"]["attempted"] = 99
        self.assertIn("observation count", _summaries_detail(data))

    def test_string_aggregate_eligible_fails(self):
        """Test string aggregate eligible fails."""
        data = _verify_artifact()
        for entry in data["search"]:
            if entry["format"]["name"] == "B":
                entry["summary"]["aggregate_eligible"] = "false"
        self.assertIn("aggregate_eligible is not a boolean", _summaries_detail(data))
        self.assertNotEqual(verify.verify_results(data)["verdict"], "ok")


class SurfaceNoiseCapContractTests(unittest.TestCase):
    """BH2B-04: built-in backends and the probe honor the cap contract."""

    def test_builtin_backends_respect_small_caps(self):
        """Test builtin backends respect small caps."""
        from model_protocol_lab.core import _valid_generation

        for backend in (surfacenoise.SelfTestBackend(), surfacenoise._ControlCalibrationBackend()):
            for cap in (1, 2, 3, 4, 5):
                gen = backend.generate((7, 7, 7), max_new_tokens=cap, seed=0, cancel=None)
                self.assertLessEqual(len(gen.token_ids), cap)
                self.assertTrue(_valid_generation(gen, cap))

    def test_probe_abstains_on_over_cap_generation(self):
        """Test probe abstains on over cap generation."""
        import threading

        from model_protocol_lab.core import Fixture, Generation, Message

        case = Fixture(
            id="f1",
            group="g",
            messages=(Message(role="user", content="q"),),
            expected="ok",
        )

        class _OverCapBackend:
            """Backend returning the right text far over the cap."""

            fingerprint = "cycle227-overcap"

            @staticmethod
            def encode(text: str) -> tuple[int, ...]:
                """Encode."""
                return (1, 2)

            @staticmethod
            def decode(ids) -> str:  # noqa: ANN001
                """Decode."""
                return "ok" if tuple(ids)[:2] == (7, 8) else "no"

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel):  # noqa: ANN001, ANN202
                """Generate."""
                return Generation((7, 8) + (9,) * 10, "limit")

        class _CompliantBackend(_OverCapBackend):
            """Backend returning the same text within the cap."""

            fingerprint = "cycle227-compliant"

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel):  # noqa: ANN001, ANN202
                """Generate."""
                return Generation((7, 8, 9, 9), "limit")

        hostile = surfacenoise._evaluate_one(
            "prompt", case, _OverCapBackend(), (999,), 4, 0, threading.Event()
        )
        compliant = surfacenoise._evaluate_one(
            "prompt", case, _CompliantBackend(), (999,), 4, 0, threading.Event()
        )
        self.assertEqual(hostile, "abstain")
        self.assertEqual(compliant, "pass")


_PROBE_ARGUMENTS = {
    "prompt": "MAGICWORD please answer now",
    "fixtures": [
        {
            "id": "f1",
            "messages": [{"role": "user", "content": "x"}],
            "expected": "BLUE",
            "group": "probe",
        }
    ],
    "boundary": [7],
}


class McpRobustnessTests(unittest.TestCase):
    """F3/F4/C-04/BH2B-09/BH2B-10: the MCP server degrades honestly."""

    @staticmethod
    def _call(name: str, arguments: dict) -> dict:
        """Call one tool in-process through handle_request."""
        from model_protocol_lab import mcp_server

        return mcp_server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )

    def test_hardening_notes_type_confusion_is_invalid_params(self):
        """Test hardening notes type confusion is invalid params."""
        cases = (
            {"report": {"operators": {"a": {"stability": 0.5, "n_effective": 1, "flips": 5}}}},
            {"report": {"operators": {}, "static_lint": "x"}},
            {"report": {"operators": {}, "static_lint": {"checks": 5}}},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                response = self._call("surface_noise_hardening_notes", arguments)
                self.assertEqual(response["error"]["code"], -32602)

    def test_local_hf_construction_failure_is_invalid_params(self):
        """Test local hf construction failure is invalid params."""
        with tempfile.TemporaryDirectory() as empty_dir:
            response = self._call(
                "probe_surface_noise_stability",
                {
                    **_PROBE_ARGUMENTS,
                    "backend": {"kind": "local_hf", "checkpoint": empty_dir},
                },
            )
        self.assertEqual(response["error"]["code"], -32602)
        self.assertEqual(
            response["error"]["message"],
            "local_hf backend could not be constructed from params.backend",
        )

    def test_probe_restores_preexisting_sigalrm_timer(self):
        """Test probe restores preexisting sigalrm timer."""
        import signal

        def _sentinel(_signum, _frame):  # noqa: ANN001, ANN202
            """Sentinel."""

        old_handler = signal.signal(signal.SIGALRM, _sentinel)
        signal.setitimer(signal.ITIMER_REAL, 45.0)
        try:
            response = self._call("probe_surface_noise_stability", dict(_PROBE_ARGUMENTS))
            remaining, _interval = signal.getitimer(signal.ITIMER_REAL)
            handler_now = signal.getsignal(signal.SIGALRM)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        self.assertIn("result", response)
        self.assertIs(handler_now, _sentinel)
        self.assertGreater(remaining, 30.0)

    def test_probe_from_worker_thread_is_not_internal_error(self):
        """Test probe from worker thread is not internal error."""
        import threading

        outcome: dict = {}

        def _run() -> None:
            """Run."""
            outcome["response"] = self._call(
                "probe_surface_noise_stability", dict(_PROBE_ARGUMENTS)
            )

        worker = threading.Thread(target=_run)
        worker.start()
        worker.join(timeout=180)
        self.assertFalse(worker.is_alive())
        self.assertIn("result", outcome["response"])

    def test_client_disconnect_ends_session_cleanly(self):
        """Test client disconnect ends session cleanly."""
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parent.parent
        proc = subprocess.Popen(
            [sys.executable, "-m", "model_protocol_lab.mcp_server"],
            cwd=repo_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            frame = (
                json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
                ).encode()
                + b"\n"
            )
            proc.stdin.write(frame)
            proc.stdin.flush()
            self.assertIn(b"tools", proc.stdout.readline())
            # Client vanishes: close our read end, then provoke one
            # more response write from the server.
            proc.stdout.close()
            try:
                proc.stdin.write(frame)
                proc.stdin.flush()
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            returncode = proc.wait(timeout=60)
            stderr = proc.stderr.read()
        finally:
            if proc.poll() is None:
                proc.kill()
        self.assertEqual(returncode, 0)
        self.assertNotIn(b"Traceback", stderr)


if __name__ == "__main__":
    unittest.main()
