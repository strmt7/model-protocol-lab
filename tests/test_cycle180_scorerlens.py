"""Cycle 180: scorer-lens regrade (wave-7 research rank 1).

The design was red-teamed before code existed (wave-7 red-team A:
DO NOT BUILD AS SPECIFIED, 12 findings; red-team B upheld all of
them, refuted one sub-claim, and added two binding amendments).
These tests pin the corrected contract:

- S1: R0 reproduces the native verdicts (status-gated exact match,
  and the boundary conjunction for verdict counts).
- S2: the ladder is monotone over an adversarial corpus.
- S3: no rung >= 1 passes an erasure-to-empty.
- S4: R4 fires only under the frozen ASCII numeric grammar;
  ``Answer: 42`` flips at no rung (there is no label-strip).

Plus the profile's fail-closed join and winner withholding, the
diff validator's recompute battery, the diff itself, the HTML
section, the CLI dispatch (helper-level and through the real run
flow), and three tempdir mutation gates on the ladder's load-
bearing constructions (union R4, the empty guard, fence strip).
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from model_protocol_lab import cli as cli_module
from model_protocol_lab import reporting, scorerlens
from model_protocol_lab.diffreports import _valid_scorerlens, diff_scorerlens
from model_protocol_lab.experiments import scorerlens_profile

_EXPECTED = {"f1": "42", "f2": "Paris", "f3": "1,000", "f4": ""}


def _obs(
    fixture: str,
    answer: str,
    *,
    status: str = "completed",
    boundary: bool = True,
) -> dict[str, Any]:
    return {
        "fixture": fixture,
        "answer": answer,
        "status": status,
        "boundary_found": boundary,
        "exact_match": status == "completed" and answer == _EXPECTED.get(fixture),
    }


def _entry(name: str, observations: list[dict[str, Any]], *, eligible: bool = True) -> dict:
    return {
        "format": {"name": name},
        "summary": {"aggregate_eligible": eligible},
        "observations": observations,
    }


class TestLadderPins(unittest.TestCase):
    """Named pins for every counterexample the cross-check produced."""

    def test_r1_whitespace_only(self):
        self.assertEqual(scorerlens.first_flip("  42\t", "42"), (1, None))
        self.assertEqual(scorerlens.first_flip("a  b", "a b"), (1, None))

    def test_r1_preserves_line_structure(self):
        # F8: collapsing newlines would equate these; R1 must not.
        self.assertEqual(scorerlens.first_flip("1\n2", "1 2"), (None, None))
        self.assertEqual(scorerlens.first_flip("a\n\nb", "a b"), (None, None))
        self.assertEqual(scorerlens.first_flip("a\n\n\nb", "a\n\nb"), (1, None))

    def test_r2_casefold_and_nfc_attribution(self):
        self.assertEqual(scorerlens.first_flip("PARIS", "Paris"), (2, "casefold"))
        nfd = "Cafe\u0301"
        self.assertEqual(scorerlens.first_flip(nfd, "Caf\u00e9"), (2, "nfc"))

    def test_r2_turkish_dotted_i_recovers(self):
        # The refusals detector's fix, adopted at R2: casefold alone
        # leaves i + combining dot; the Mn strip equates them.
        self.assertEqual(scorerlens.first_flip("\u0130", "i"), (2, "casefold"))

    def test_r2_ligature_recovers_via_casefold(self):
        # Red-team B's refutation of A's scope note: casefold maps
        # the fi ligature to "fi", so R2 does equate them.
        self.assertEqual(scorerlens.first_flip("\ufb01le", "file"), (2, "casefold"))

    def test_r3_trailing_punctuation_and_quotes(self):
        self.assertEqual(scorerlens.first_flip("42.", "42"), (3, None))
        self.assertEqual(scorerlens.first_flip('"42"', "42"), (3, None))
        self.assertEqual(scorerlens.first_flip("**42**.", "42"), (3, None))

    def test_r3_fence_strip_with_info_string(self):
        self.assertEqual(scorerlens.first_flip("```\n42\n```", "42"), (3, None))
        self.assertEqual(scorerlens.first_flip("```python\n42\n```", "42"), (3, None))

    def test_r3_sign_and_decimal_point_are_content(self):
        # F7: -42 is not 42; .5 is not 5; 50% is not 50.
        self.assertEqual(scorerlens.first_flip("-42", "42"), (None, None))
        self.assertEqual(scorerlens.first_flip(".5", "5"), (None, None))
        self.assertEqual(scorerlens.first_flip("50%", "50"), (None, None))
        self.assertEqual(scorerlens.first_flip("C++", "C"), (None, None))

    def test_r3_internal_punctuation_untouched(self):
        self.assertEqual(scorerlens.first_flip("3.5", "35"), (None, None))
        self.assertEqual(scorerlens.first_flip("don't", "dont"), (None, None))

    def test_s3_erasure_to_empty_never_passes(self):
        self.assertFalse(scorerlens.rung_match("?", "!", 3))
        self.assertFalse(scorerlens.rung_match("***", "", 1))
        self.assertFalse(scorerlens.rung_match("***", "", 4))
        self.assertEqual(scorerlens.first_flip("?", "!"), (None, None))

    def test_s4_numeric_grammar(self):
        self.assertEqual(scorerlens.first_flip("1,000", "1000"), (4, None))
        self.assertEqual(scorerlens.first_flip("1e3", "1000"), (4, None))
        self.assertEqual(scorerlens.first_flip("-0", "0"), (4, None))
        self.assertEqual(scorerlens.first_flip("+42", "42"), (4, None))
        self.assertEqual(scorerlens.first_flip(".5", "0.5"), (4, None))
        for answer, expected in (
            ("1,5", "15"),  # decimal comma is not grouping
            ("12,34", "1234"),
            ("1,2,3", "123"),
            ("1_000", "1000"),  # underscores rejected
            ("inf", "Infinity"),  # non-finite spellings never fire
            ("1e10000000000000000000", "1"),  # exponent beyond Decimal's range
            ("\uff14\uff12", "42"),  # non-ASCII digits rejected
            ("42", "17"),  # distinct numbers never match (NEW-3 pin)
        ):
            with self.subTest(answer=answer, expected=expected):
                self.assertEqual(scorerlens.first_flip(answer, expected), (None, None))

    def test_s4_no_normalize_collapse(self):
        # Decimal == is exact; normalize() under the default context
        # would collapse this pair.
        self.assertEqual(
            scorerlens.first_flip("1.00000000000000000000000000001", "1"),
            (None, None),
        )

    def test_s4_nan_survives_via_union(self):
        # F4: R4 as a replacement comparison would un-pass this pair
        # (Decimal NaN != NaN); the union definition keeps the R2 pass.
        self.assertEqual(scorerlens.first_flip("nan", "NaN"), (2, "casefold"))
        self.assertTrue(scorerlens.rung_match("nan", "NaN", 4))

    def test_no_label_strip(self):
        # F3, option (a): lead-in prose flips at no rung, by design.
        for answer in ("Answer: 42", "The answer is 42.", "**Answer:** 42"):
            with self.subTest(answer=answer):
                self.assertEqual(scorerlens.first_flip(answer, "42"), (None, None))

    def test_unknown_rung_raises(self):
        with self.assertRaises(ValueError):
            scorerlens.rung_match("a", "a", 5)


_ADVERSARIAL_PAIRS = [
    ("nan", "NaN"),
    ("inf", "Infinity"),
    ("1,000", "1000"),
    ("1,5", "15"),
    ("```\n42\n```", "42"),
    ("?", "!"),
    ("", ""),
    ("", "42"),
    ("***", "***"),
    ("\u0130", "i"),
    ("Weiß", "Weiss"),
    ("C++", "C"),
    ("-42", "42"),
    (".5", "0.5"),
    ("42.", "42"),
    ("Answer: 42", "42"),
    ("1\n2", "1 2"),
    ("\uff14\uff12", "42"),
    ("1_000", "1000"),
    ("1e3", "1000"),
    ("-0", "0"),
    ("  Paris ", "paris"),
    ("\ufb01le", "file"),
    ("1.00000000000000000000000000001", "1"),
]


class TestMonotonicity(unittest.TestCase):
    def test_s2_monotone_over_adversarial_corpus(self):
        for answer, expected in _ADVERSARIAL_PAIRS:
            for rung in range(4):
                with self.subTest(answer=answer, expected=expected, rung=rung):
                    if scorerlens.rung_match(answer, expected, rung):
                        self.assertTrue(
                            scorerlens.rung_match(answer, expected, rung + 1),
                            f"passed at R{rung} but not R{rung + 1}",
                        )

    def test_first_flip_consistent_with_rung_match(self):
        for answer, expected in _ADVERSARIAL_PAIRS:
            if answer == expected:
                continue
            rung, _via = scorerlens.first_flip(answer, expected)
            for k in range(1, 5):
                with self.subTest(answer=answer, expected=expected, k=k):
                    self.assertEqual(
                        scorerlens.rung_match(answer, expected, k),
                        rung is not None and k >= rung,
                    )


class TestS1Oracle(unittest.TestCase):
    def test_r0_reproduces_native_verdicts(self):
        observations = [
            _obs("f1", "42"),
            _obs("f1", "42."),
            _obs("f2", "Paris", boundary=False),
            _obs("f2", "paris", status="failed"),
            _obs("f3", "1000"),
            _obs("f3", "1,000"),
        ]
        record = scorerlens.analyze_candidate(observations, _EXPECTED, eligible=True)
        self.assertTrue(record["available"])
        native_exact = sum(
            1
            for o in observations
            if o["status"] == "completed" and o["answer"] == _EXPECTED[o["fixture"]]
        )
        native_verdict = sum(
            1
            for o in observations
            if o["status"] == "completed"
            and o["boundary_found"]
            and o["answer"] == _EXPECTED[o["fixture"]]
        )
        self.assertEqual(record["match_counts"]["0"], native_exact)
        self.assertEqual(record["verdict_counts"]["0"], native_verdict)


class TestAnalyzeCandidate(unittest.TestCase):
    def test_empty_expected_excluded_with_count(self):
        record = scorerlens.analyze_candidate(
            [_obs("f1", "42"), _obs("f4", "")], _EXPECTED, eligible=True
        )
        self.assertEqual(record["n_evaluable"], 1)
        self.assertEqual(record["n_excluded_empty_expected"], 1)
        excluded = [r for r in record["records"] if r["excluded"]]
        self.assertEqual([r["fixture"] for r in excluded], ["f4"])

    def test_unknown_fixture_fails_closed(self):
        record = scorerlens.analyze_candidate([_obs("nope", "42")], _EXPECTED, eligible=True)
        self.assertEqual(record, {"available": False, "detail": "unknown_fixture"})

    def test_malformed_observations_fail_closed(self):
        self.assertEqual(
            scorerlens.analyze_candidate("nope", _EXPECTED, eligible=True),
            {"available": False, "detail": "observations_not_a_list"},
        )
        self.assertEqual(
            scorerlens.analyze_candidate([42], _EXPECTED, eligible=True),
            {"available": False, "detail": "observation_malformed"},
        )
        bad = {"fixture": "f1", "answer": 42, "status": "completed", "boundary_found": True}
        self.assertEqual(
            scorerlens.analyze_candidate([bad], _EXPECTED, eligible=True),
            {"available": False, "detail": "observation_malformed"},
        )

    def test_boundary_blocked_counted_and_share(self):
        observations = [
            _obs("f1", "42."),  # flips R3, verdict passes
            _obs("f2", "Paris", boundary=False),  # exact text, boundary blocks
            _obs("f3", "999"),  # never flips
        ]
        record = scorerlens.analyze_candidate(observations, _EXPECTED, eligible=True)
        self.assertEqual(record["n_boundary_blocked"], 1)
        self.assertEqual(record["verdict_counts"]["0"], 0)
        self.assertEqual(record["verdict_counts"]["3"], 1)
        self.assertEqual(record["n_native_verdict_failures"], 3)
        self.assertEqual(record["n_recovered_by_r4"], 1)
        self.assertAlmostEqual(record["surface_failure_share"], 1 / 3)
        self.assertEqual(record["share_note"], "")

    def test_share_withheld_when_ineligible(self):
        record = scorerlens.analyze_candidate([_obs("f1", "42.")], _EXPECTED, eligible=False)
        self.assertIsNone(record["surface_failure_share"])
        self.assertEqual(record["share_note"], "candidate_not_eligible")

    def test_share_none_when_no_failures(self):
        record = scorerlens.analyze_candidate([_obs("f1", "42")], _EXPECTED, eligible=True)
        self.assertIsNone(record["surface_failure_share"])
        self.assertEqual(record["share_note"], "no_native_failures")


def _profile_result() -> dict[str, Any]:
    return {
        "search": [
            _entry(
                "alpha",
                [_obs("f1", "42."), _obs("f2", "Paris"), _obs("f3", "1,000")],
            ),
            _entry(
                "beta",
                [_obs("f1", "41"), _obs("f2", "Paris"), _obs("f3", "1,000")],
            ),
        ],
        "selected_candidate": None,
    }


class TestProfile(unittest.TestCase):
    def test_winner_by_rung(self):
        payload = scorerlens_profile(_profile_result(), dict(_EXPECTED))
        self.assertTrue(payload["available"])
        # alpha: f1 flips at R3, f3 native -> winner from R3 on.
        # beta: f1 never flips -> never winner.
        self.assertEqual(
            payload["winner_by_rung"],
            {"0": None, "1": None, "2": None, "3": "alpha", "4": "alpha"},
        )
        self.assertEqual(payload["winner_detail"], "")
        self.assertIsNone(payload["native_winner"])
        self.assertFalse(payload["evaluable_mismatch"])

    def test_winner_withheld_when_ineligible(self):
        result = _profile_result()
        result["search"][1]["summary"]["aggregate_eligible"] = False
        payload = scorerlens_profile(result, dict(_EXPECTED))
        self.assertIsNone(payload["winner_by_rung"])
        self.assertEqual(payload["winner_detail"], "candidate_not_eligible")

    def test_winner_withheld_on_evaluable_mismatch(self):
        result = _profile_result()
        result["search"][1]["observations"] = result["search"][1]["observations"][:2]
        payload = scorerlens_profile(result, dict(_EXPECTED))
        self.assertTrue(payload["evaluable_mismatch"])
        self.assertIsNone(payload["winner_by_rung"])
        self.assertEqual(payload["winner_detail"], "evaluable_mismatch")

    def test_winner_withheld_when_candidate_unavailable(self):
        result = _profile_result()
        result["search"][1]["observations"] = [_obs("ghost", "42")]
        payload = scorerlens_profile(result, dict(_EXPECTED))
        self.assertIsNone(payload["winner_by_rung"])
        self.assertEqual(payload["winner_detail"], "candidate_unavailable")
        self.assertEqual(payload["candidates"][1]["detail"], "unknown_fixture")

    def test_native_winner_passthrough(self):
        result = _profile_result()
        result["selected_candidate"] = "beta"
        payload = scorerlens_profile(result, dict(_EXPECTED))
        self.assertEqual(payload["native_winner"], "beta")

    def test_malformed_inputs(self):
        self.assertEqual(
            scorerlens_profile([], {}),  # type: ignore[arg-type]
            {"available": False, "detail": "result_not_a_dict"},
        )
        self.assertEqual(
            scorerlens_profile({"search": []}, {"f1": 42}),  # type: ignore[dict-item]
            {"available": False, "detail": "expected_map_invalid"},
        )
        self.assertEqual(
            scorerlens_profile({"search": []}, {}),
            {"available": False, "detail": "no_search_entries"},
        )
        payload = scorerlens_profile({"search": [{"format": {}}]}, dict(_EXPECTED))
        self.assertEqual(payload["candidates"][0]["detail"], "entry_malformed")
        self.assertEqual(payload["winner_detail"], "candidate_unavailable")


def _foreign_ladder_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """A coherent foreign-ladder copy: modified ladder data with its
    own correct digest (protocol drift, not corruption)."""
    other = copy.deepcopy(payload)
    ladder = copy.deepcopy(scorerlens.LADDER_SPEC)
    ladder["decisions"] = dict(ladder["decisions"], percent="foreign")
    other["method"]["ladder"] = ladder
    canonical = json.dumps(ladder, sort_keys=True, separators=(",", ":"))
    other["method"]["ladder_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return other


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.payload = scorerlens_profile(_profile_result(), dict(_EXPECTED))

    def test_producer_payload_validates(self):
        self.assertTrue(_valid_scorerlens(self.payload))

    def test_tamper_battery_rejected(self):
        def tamper(fn):
            payload = copy.deepcopy(self.payload)
            fn(payload)
            return payload

        cases = {
            "verdict count inflated": lambda p: p["candidates"][0]["verdict_counts"].__setitem__(
                "4", 99
            ),
            "share off counts": lambda p: p["candidates"][0].__setitem__(
                "surface_failure_share", 0.5
            ),
            "manufactured winner": lambda p: p["winner_by_rung"].__setitem__("0", "beta"),
            "winner withheld replaced": lambda p: p.__setitem__("winner_by_rung", None),
            "ladder digest mismatch": lambda p: p["method"].__setitem__("ladder_sha256", "0" * 64),
            "mismatch flag flipped": lambda p: p.__setitem__("evaluable_mismatch", True),
            "record aggregate drift": lambda p: p["candidates"][0].__setitem__(
                "n_boundary_blocked", 7
            ),
            "flip rung moved": lambda p: p["candidates"][0]["records"][0].__setitem__(
                "first_flip_rung", 1
            ),
        }
        for name, fn in cases.items():
            with self.subTest(case=name):
                self.assertFalse(_valid_scorerlens(tamper(fn)), name)

    def test_foreign_coherent_ladder_validates(self):
        self.assertTrue(_valid_scorerlens(_foreign_ladder_payload(self.payload)))


class TestDiff(unittest.TestCase):
    def setUp(self):
        self.payload = scorerlens_profile(_profile_result(), dict(_EXPECTED))

    def test_identical_payloads_unchanged(self):
        diff = diff_scorerlens(self.payload, self.payload)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])

    def test_winner_and_flip_drift(self):
        other_result = _profile_result()
        other_result["search"][0]["observations"][0] = _obs("f1", "42")  # alpha now native
        other = scorerlens_profile(other_result, dict(_EXPECTED))
        diff = diff_scorerlens(self.payload, other)
        self.assertTrue(diff["changed"])
        self.assertIn({"rung": 0, "old": None, "new": "alpha"}, diff["winner_drift"])
        self.assertTrue(
            any(
                c["fixture"] == "f1" and c["old_rung"] == 3 and c["new_rung"] is None
                for c in diff["flip_changes"]
            )
        )

    def test_protocol_drift_not_comparable(self):
        diff = diff_scorerlens(self.payload, _foreign_ladder_payload(self.payload))
        self.assertTrue(diff["protocol_drift"])
        self.assertFalse(diff["comparable"])
        self.assertTrue(diff["changed"])

    def test_fixture_set_drift(self):
        other = copy.deepcopy(self.payload)
        other["fixture_set_sha256"] = hashlib.sha256(b"other").hexdigest()
        diff = diff_scorerlens(self.payload, other)
        self.assertTrue(diff["fixture_set_drift"])
        self.assertFalse(diff["comparable"])

    def test_one_sided(self):
        diff = diff_scorerlens(self.payload, None)
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        diff = diff_scorerlens(None, None)
        self.assertEqual(diff["available"], "neither")


class TestReportingSection(unittest.TestCase):
    def test_section_renders_with_winner_and_honesty(self):
        payload = scorerlens_profile(_profile_result(), dict(_EXPECTED))
        html = reporting._scorerlens_section({"scorerlens": payload})
        self.assertIn('id="scorerlens"', html)
        self.assertIn("Search-phase winner under each rung", html)
        self.assertIn("no label-strip operation", html)
        self.assertIn("first passes at rung 3", html)

    def test_section_absent_without_payload(self):
        self.assertEqual(reporting._scorerlens_section({}), "")
        self.assertEqual(reporting._scorerlens_section({"scorerlens": {"available": False}}), "")

    def test_section_escapes_candidate_names(self):
        result = _profile_result()
        result["search"][0]["format"]["name"] = "<script>x</script>"
        payload = scorerlens_profile(result, dict(_EXPECTED))
        html = reporting._scorerlens_section({"scorerlens": payload})
        self.assertNotIn("<script>x</script>", html)
        self.assertIn("&lt;script&gt;", html)


class _Fixture:
    def __init__(self, fid: str, expected: str) -> None:
        self.id = fid
        self.expected = expected


class TestCliProbe(unittest.TestCase):
    def test_probe_attaches_payload(self):
        result = _profile_result()
        cli_module._run_scorerlens_probe(
            result, search_fixtures=[_Fixture(k, v) for k, v in _EXPECTED.items()]
        )
        self.assertTrue(result["scorerlens"]["available"])
        self.assertEqual(result["scorerlens"]["winner_by_rung"]["3"], "alpha")

    def test_probe_never_raises(self):
        result: dict[str, Any] = {}
        cli_module._run_scorerlens_probe(result, search_fixtures=[object()])
        self.assertEqual(
            result["scorerlens"], {"available": False, "detail": "scorerlens_unavailable"}
        )

    def test_flag_parses(self):
        parser = cli_module.build_parser()
        args = parser.parse_args(
            [
                "run",
                "--scorer-lens",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "llama3",
                "--boundary-ids",
                "1",
                "--output",
                "o",
            ]
        )
        self.assertTrue(args.scorer_lens)


class TestCliDispatchEndToEnd(unittest.TestCase):
    """The --scorer-lens branch driven through the real run flow
    (cycle-179 lesson: a flag can parse and its stage be fully
    unit-tested while the dispatch silently never runs it)."""

    def test_scorer_lens_dispatches_through_main(self):
        from model_protocol_lab import backends, doctor, experiments
        from model_protocol_lab import reporting as reporting_mod

        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "fixture.json"
            fixture_path.write_text(
                json.dumps(
                    {
                        "search": [
                            {
                                "id": "s1",
                                "group": "g1",
                                "messages": [{"role": "user", "content": "hi"}],
                                "expected": "yo",
                            }
                        ],
                        "held_out": [
                            {
                                "id": "h1",
                                "group": "g2",
                                "messages": [{"role": "user", "content": "hey"}],
                                "expected": "yo",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            outdir = Path(tmp) / "report"
            result_doc: dict[str, Any] = {
                "decision": "compatible_on_fixture_set",
                "selected_candidate": "chatml",
                "search": [
                    {
                        "format": {"name": "chatml"},
                        "summary": {"aggregate_eligible": True},
                        "observations": [
                            {
                                "fixture": "s1",
                                "answer": "yo.",
                                "status": "completed",
                                "boundary_found": True,
                                "exact_match": False,
                            }
                        ],
                    }
                ],
                "held_out": {},
            }

            class _FakeHFBackend:
                def __init__(self, model, device=None, max_context=None):
                    self.model = model

                eos_token_id = None

            def fake_prepare(path):
                Path(path).mkdir(parents=True, exist_ok=False)
                return Path(path)

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(backends, "LocalHFBackend", _FakeHFBackend))
                stack.enter_context(
                    mock.patch.object(cli_module, "run_experiment", return_value=result_doc)
                )
                stack.enter_context(
                    mock.patch.object(
                        cli_module,
                        "save_report",
                        return_value=(str(outdir / "results.json"), str(outdir / "report.md")),
                    )
                )
                stack.enter_context(
                    mock.patch.object(doctor, "template_conformance", return_value={"ok": True})
                )
                stack.enter_context(
                    mock.patch.object(
                        experiments, "budget_frontier", return_value={"available": False}
                    )
                )
                stack.enter_context(
                    mock.patch.object(reporting_mod, "prepare_report_dir", fake_prepare)
                )
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = cli_module.main(
                        [
                            "run",
                            "--model",
                            "ckpt-dir",
                            "--dataset",
                            str(fixture_path),
                            "--family",
                            "chatml",
                            "--boundary-ids",
                            "198,199",
                            "--output",
                            str(outdir),
                            "--seed",
                            "7",
                            "--scorer-lens",
                        ]
                    )
            self.assertEqual(code, 0)
            payload = result_doc["scorerlens"]
            self.assertTrue(payload["available"])
            self.assertEqual(payload["winner_by_rung"]["3"], "chatml")


class TestMutationGates(unittest.TestCase):
    """Tempdir gates (cycle-179 pattern): each mutant of the
    ladder's load-bearing constructions must be killed by a named
    scenario, and each scenario must pass on the real module."""

    def _run_gate(self, name, old_text, new_text, scenario):
        import importlib.util
        import os
        import sys

        mod = scorerlens
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c180_mut_") as tmpdir:
            mod_path = os.path.join(tmpdir, f"scorerlens_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            dotted = f"model_protocol_lab.scorerlens_mut_{name}"
            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            sys.modules[dotted] = mutant
            try:
                spec.loader.exec_module(mutant)
            finally:
                sys.modules.pop(dotted, None)
            scenario(mod)  # passes on the real module: gate is non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    def test_mutant_union_dropped_caught(self):
        def scenario(mod):
            # With R4 as a replacement comparison, the NaN pair
            # un-passes at R4 (Decimal NaN != NaN).
            assert mod.rung_match("nan", "NaN", 4) is True

        self._run_gate(
            "union_dropped",
            "    for j in (1, 2, 3):\n        if j > rung:\n            break",
            "    for j in (1, 2, 3):\n        if j > rung or rung == 4:\n"
            "            break  # MUTANT: R4 replaces instead of unioning",
            scenario,
        )

    def test_mutant_empty_guard_dropped_caught(self):
        def scenario(mod):
            assert mod.rung_match("?", "!", 3) is False

        self._run_gate(
            "empty_guard_dropped",
            "    return bool(a_form) and bool(e_form) and a_form == e_form",
            "    return a_form == e_form  # MUTANT: empty guard dropped",
            scenario,
        )

    def test_mutant_fence_strip_dropped_caught(self):
        def scenario(mod):
            # The bare-fence case is also recovered by edge backtick
            # stripping plus the final trim; the info-string fence
            # is the case only the fence operation recovers.
            assert mod.first_flip("```python\n42\n```", "42") == (3, None)

        self._run_gate(
            "fence_strip_dropped",
            "    return _punct_strip(_fence_strip(text)).strip()",
            "    return _punct_strip(text).strip()  # MUTANT: fence strip dropped",
            scenario,
        )


class TestValidatorBattery(unittest.TestCase):
    """One broken identity at a time: every reject branch of
    _valid_scorerlens must fire on its named tamper (the cycle-174
    gap pattern -- a validator branch no payload exercises is a
    claim, not a check)."""

    def setUp(self):
        self.payload = scorerlens_profile(_profile_result(), dict(_EXPECTED))

    def _tampered(self, fn):
        payload = copy.deepcopy(self.payload)
        fn(payload)
        return payload

    def test_reject_battery(self):
        cand = lambda p: p["candidates"][0]  # noqa: E731
        rec = lambda p: p["candidates"][0]["records"][0]  # noqa: E731
        cases = {
            "payload not available": lambda p: p.__setitem__("available", False),
            "method missing": lambda p: p.__setitem__("method", None),
            "protocol empty": lambda p: p["method"].__setitem__("protocol", ""),
            "ladder protocol mismatch": lambda p: p["method"]["ladder"].__setitem__(
                "protocol", "other"
            ),
            "fixture sha short": lambda p: p.__setitem__("fixture_set_sha256", "abc"),
            "fixture sha non-hex": lambda p: p.__setitem__("fixture_set_sha256", "z" * 64),
            "mismatch not bool": lambda p: p.__setitem__("evaluable_mismatch", "no"),
            "native winner wrong type": lambda p: p.__setitem__("native_winner", 7),
            "candidates empty": lambda p: p.__setitem__("candidates", []),
            "n_candidates off": lambda p: p.__setitem__("n_candidates", 99),
            "entry not dict": lambda p: p["candidates"].__setitem__(0, 42),
            "name wrong type": lambda p: cand(p).__setitem__("candidate", 7),
            "duplicate names": lambda p: p["candidates"][1].__setitem__("candidate", "alpha"),
            "available not bool": lambda p: cand(p).__setitem__("available", "yes"),
            "eligible not bool": lambda p: cand(p).__setitem__("eligible", "yes"),
            "n_observations negative": lambda p: cand(p).__setitem__("n_observations", -1),
            "n_completed over": lambda p: cand(p).__setitem__("n_completed", 99),
            "n_completed under": lambda p: cand(p).__setitem__("n_completed", 2),
            "unavailable without detail": lambda p: p["candidates"][1].__setitem__(
                "available", False
            ),
            "records wrong length": lambda p: cand(p).__setitem__("records", []),
            "record not dict": lambda p: cand(p)["records"].__setitem__(0, 42),
            "record fixture empty": lambda p: rec(p).__setitem__("fixture", ""),
            "record flag not bool": lambda p: rec(p).__setitem__("completed", "yes"),
            "flip rung zero": lambda p: rec(p).__setitem__("first_flip_rung", 0),
            "flip rung five": lambda p: rec(p).__setitem__("first_flip_rung", 5),
            "via without rung 2": lambda p: rec(p).__setitem__("flip_via", "nfc"),
            "via invalid": lambda p: p["candidates"][0]["records"][0].__setitem__(
                "flip_via", "magic"
            ),
            "exact with flip": lambda p: p["candidates"][0]["records"][1].__setitem__(
                "first_flip_rung", 2
            ),
            "altered flag not bool": lambda p: rec(p).__setitem__("expected_r3_altered", "y"),
            "match counts off": lambda p: cand(p)["match_counts"].__setitem__("4", 99),
            "n_evaluable off": lambda p: cand(p).__setitem__("n_evaluable", 99),
            "n_excluded off": lambda p: cand(p).__setitem__("n_excluded_empty_expected", 9),
            "failures off": lambda p: cand(p).__setitem__("n_native_verdict_failures", 9),
            "recovered off": lambda p: cand(p).__setitem__("n_recovered_by_r4", 9),
            "flips boundary off": lambda p: cand(p).__setitem__("n_flips_boundary_blocked", 9),
            "flips altered off": lambda p: cand(p).__setitem__("n_flips_expected_altered", 9),
            "r1 newline off": lambda p: cand(p).__setitem__("n_r1_flips_newline_changed", 9),
            "note vocabulary": lambda p: cand(p).__setitem__("share_note", "bogus"),
            "share without note mismatch": lambda p: cand(p).__setitem__(
                "surface_failure_share", None
            ),
            "share out of range": lambda p: cand(p).__setitem__("surface_failure_share", 1.5),
            "n_available off": lambda p: p.__setitem__("n_candidates_available", 9),
            "winner detail vocabulary": lambda p: p.__setitem__("winner_detail", "bogus"),
            "winners dropped silently": lambda p: p.__setitem__("winner_by_rung", None),
            "winner keys wrong": lambda p: p.__setitem__("winner_by_rung", {"0": None}),
        }
        for name, fn in cases.items():
            with self.subTest(case=name):
                self.assertFalse(_valid_scorerlens(self._tampered(fn)), name)

    def test_exact_record_with_via_consistent_flip_rejected(self):
        # flip==2 with a via passes the via check; the exact/flip
        # contradiction is the branch under test.
        payload = copy.deepcopy(self.payload)
        record = payload["candidates"][0]["records"][1]
        assert record["exact_match"] is True
        record["first_flip_rung"] = 2
        record["flip_via"] = "nfc"
        self.assertFalse(_valid_scorerlens(payload))

    def test_excluded_record_with_flip_rejected(self):
        result = {
            "search": [_entry("alpha", [_obs("f4", "something"), _obs("f1", "42")])],
            "selected_candidate": None,
        }
        payload = scorerlens_profile(result, dict(_EXPECTED))
        record = payload["candidates"][0]["records"][0]
        assert record["excluded"] is True
        record["first_flip_rung"] = 1
        self.assertFalse(_valid_scorerlens(payload))

    def test_ineligible_candidate_wrong_note_rejected(self):
        result = _profile_result()
        result["search"][1]["summary"]["aggregate_eligible"] = False
        payload = scorerlens_profile(result, dict(_EXPECTED))
        payload["candidates"][1]["share_note"] = "no_native_failures"
        self.assertFalse(_valid_scorerlens(payload))

    def test_no_failures_candidate_wrong_note_rejected(self):
        result = {
            "search": [_entry("alpha", [_obs("f1", "42"), _obs("f2", "Paris")])],
            "selected_candidate": None,
        }
        payload = scorerlens_profile(result, dict(_EXPECTED))
        assert payload["candidates"][0]["share_note"] == "no_native_failures"
        payload["candidates"][0]["share_note"] = "candidate_not_eligible"
        self.assertFalse(_valid_scorerlens(payload))

    def test_withheld_for_mismatch_validates(self):
        result = _profile_result()
        result["search"][1]["observations"] = result["search"][1]["observations"][:2]
        payload = scorerlens_profile(result, dict(_EXPECTED))
        self.assertEqual(payload["winner_detail"], "evaluable_mismatch")
        self.assertTrue(_valid_scorerlens(payload))

    def test_winners_present_despite_ineligible_candidate_rejected(self):
        result = _profile_result()
        result["search"][1]["summary"]["aggregate_eligible"] = False
        payload = scorerlens_profile(result, dict(_EXPECTED))
        payload["winner_by_rung"] = {"0": None, "1": None, "2": None, "3": None, "4": None}
        payload["winner_detail"] = ""
        self.assertFalse(_valid_scorerlens(payload))

    def test_winners_present_despite_mismatch_rejected(self):
        result = _profile_result()
        result["search"][1]["observations"] = result["search"][1]["observations"][:2]
        payload = scorerlens_profile(result, dict(_EXPECTED))
        payload["winner_by_rung"] = {"0": None, "1": None, "2": None, "3": None, "4": None}
        payload["winner_detail"] = ""
        self.assertFalse(_valid_scorerlens(payload))

    def test_withheld_reason_must_be_derivable(self):
        # Healthy payload, winner readout removed and a false
        # reason pleaded: no withholding condition holds, so the
        # payload is corruption, not a withheld readout.
        payload = copy.deepcopy(self.payload)
        payload["winner_by_rung"] = None
        payload["winner_detail"] = "candidate_not_eligible"
        self.assertFalse(_valid_scorerlens(payload))

    def test_withheld_for_unavailable_candidate_validates(self):
        result = _profile_result()
        result["search"][1]["observations"] = [_obs("ghost", "42")]
        payload = scorerlens_profile(result, dict(_EXPECTED))
        self.assertEqual(payload["winner_detail"], "candidate_unavailable")
        self.assertTrue(_valid_scorerlens(payload))

    def test_winners_present_despite_unavailable_candidate_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["candidates"][1] = {
            "candidate": "beta",
            "available": False,
            "detail": "unknown_fixture",
        }
        payload["n_candidates_available"] = 1
        self.assertFalse(_valid_scorerlens(payload))


class TestDiffGaps(unittest.TestCase):
    def setUp(self):
        self.payload = scorerlens_profile(_profile_result(), dict(_EXPECTED))

    def test_share_drift_alone(self):
        other_result = _profile_result()
        other_result["search"][0]["observations"][2] = _obs("f3", "999")
        other = scorerlens_profile(other_result, dict(_EXPECTED))
        diff = diff_scorerlens(self.payload, other)
        self.assertTrue(
            any(d["candidate"] == "alpha" for d in diff["share_drift"]),
            diff["share_drift"],
        )
        self.assertTrue(diff["changed"])

    def test_excluded_candidate_transitions(self):
        other_result = _profile_result()
        other_result["search"][1]["observations"] = [_obs("ghost", "42")]
        other = scorerlens_profile(other_result, dict(_EXPECTED))
        diff = diff_scorerlens(self.payload, other)
        self.assertEqual(diff["excluded_added"], ["beta"])
        self.assertTrue(diff["winner_readout_changed"])
        self.assertTrue(diff["changed"])

    def test_malformed_entry_candidate_in_view(self):
        result = _profile_result()
        result["search"].append({"format": {}})
        payload = scorerlens_profile(result, dict(_EXPECTED))
        diff = diff_scorerlens(payload, payload)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])


class TestReportingGaps(unittest.TestCase):
    def test_hostile_payload_defensive_branches(self):
        payload = {
            "available": True,
            "evaluable_mismatch": True,
            "winner_by_rung": None,
            "winner_detail": "candidate_not_eligible",
            "candidates": [
                42,
                {"candidate": None, "available": False},
                {
                    "candidate": "gamma",
                    "available": True,
                    "n_evaluable": 0,
                    "verdict_counts": "bogus",
                    "n_boundary_blocked": 0,
                    "surface_failure_share": None,
                    "share_note": "no_native_failures",
                },
                {
                    "candidate": "beta",
                    "available": True,
                    "n_evaluable": 2,
                    "verdict_counts": "bogus",
                    "n_boundary_blocked": 1,
                    "surface_failure_share": 0.5,
                    "share_note": "candidate_not_eligible",
                    "records": [
                        42,
                        {"fixture": "f1", "first_flip_rung": None},
                        {
                            "fixture": "f2",
                            "first_flip_rung": 2,
                            "flip_via": "nfc",
                            "boundary_found": False,
                        },
                    ],
                },
            ],
        }
        html = reporting._scorerlens_section({"scorerlens": payload})
        self.assertIn("n/a", html)
        self.assertIn("winner readout withheld", html)
        self.assertIn("candidate not eligible", html)
        self.assertIn("Caution:", html)
        self.assertIn("via nfc", html)
        self.assertIn("boundary still blocks the verdict", html)
        self.assertIn("Candidates with no usable recorded observations", html)

    def test_candidates_not_a_list(self):
        self.assertEqual(
            reporting._scorerlens_section({"scorerlens": {"available": True, "candidates": 42}}),
            "",
        )

    def test_section_without_flips(self):
        result = {
            "search": [_entry("alpha", [_obs("f1", "42"), _obs("f2", "Paris")])],
            "selected_candidate": "alpha",
        }
        payload = scorerlens_profile(result, dict(_EXPECTED))
        html = reporting._scorerlens_section({"scorerlens": payload})
        self.assertIn('id="scorerlens"', html)
        self.assertNotIn("Fixtures recovered by the ladder", html)


if __name__ == "__main__":
    unittest.main()
