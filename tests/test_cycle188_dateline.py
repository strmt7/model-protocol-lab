"""Cycle 188: date-line sensitivity x template (run --dateline).

Pins the cross-checked design (researcher + red-team A F1-F15,
red-team B corrections G1'-G12'): frozen date values x three
formats with weekday names as protocol data, a per-format
no-date control sized to the anchor arm's rendered length, one
weekday-lie arm, the component-wise eligibility screen, per-pair
discordant headlines partitioned by pair type, date-attributable
loss with its counterpart, full recompute in the diff validator
with the validator's own literal arm table, and digests only in
the records. Design: docs/DATELINE_DESIGN.md.
"""

from __future__ import annotations

import copy
import datetime
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_protocol_lab import cli, dateline
from model_protocol_lab import diffreports as _diffreports_module
from model_protocol_lab.confidentiality import mcnemar_exact_p
from model_protocol_lab.core import Fixture as Case
from model_protocol_lab.core import Format, Generation, Message
from model_protocol_lab.diffreports import (
    _dateline_classify,
    _dateline_mcnemar_p,
    _valid_dateline,
    diff_dateline,
)
from model_protocol_lab.experiments import dateline_analysis, dateline_stage
from model_protocol_lab.reporting import render_html

BOUNDARY_WORD = "endmark"


class _DateBackend:
    """Word-token backend whose date cells are scripted by substring.

    ``fail_on`` holds (marker, substring) pairs answered wrongly;
    everything else answers the fixture's expected string.
    """

    fingerprint = "sha256:dateline-test"

    def __init__(self, fail_on=(), expected_by_marker=None):
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}
        self.fail_on = set(fail_on)
        self.expected_by_marker = dict(expected_by_marker or {})
        self.generate_calls = 0

    def encode(self, text):
        ids = []
        for token in text.replace("\n", " \n ").split(" "):
            if not token:
                continue
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                self.rev[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return tuple(ids)

    def decode(self, ids):
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        self.generate_calls += 1
        text = self.decode(list(input_ids))
        marker = None
        for candidate in self.expected_by_marker:
            if candidate in text:
                marker = candidate
                break
        answer = self.expected_by_marker.get(marker, "dunno")
        if marker is not None:
            for fail_marker, substring in self.fail_on:
                if fail_marker == marker and substring in text:
                    answer = "wrong!"
                    break
        boundary = self.encode(BOUNDARY_WORD)[0]
        out = list(self.encode(answer)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


FMT_A = Format(
    name="dt-a",
    bos="<|im_start|>system\nYou are helpful.<|im_end|>\n",
    user_prefix="<|im_start|>user\n",
    user_turn_end="<|im_end|>\n",
    assistant_prefix="<|im_start|>assistant\n",
    turn_end="<|im_end|>",
)
FMT_B = Format(
    name="dt-b",
    bos="",
    user_prefix="USER: ",
    assistant_prefix="ASSISTANT: ",
    turn_end="~",
)
FORMATS = (FMT_A, FMT_B)

CASE_ONE = Case(
    id="fx-one",
    group="g1",
    messages=(Message(role="user", content="markone name a colour"),),
    expected="alpha",
)
CASE_TWO = Case(
    id="fx-two",
    group="g2",
    messages=(Message(role="user", content="marktwo name a shape"),),
    expected="beta",
)
CASE_MULTI = Case(
    id="fx-multi",
    group="g3",
    messages=(
        Message(role="user", content="markthree first part"),
        Message(role="assistant", content="noted"),
        Message(role="user", content="markthree second part"),
    ),
    expected="gamma",
)
CASE_SCREENED = Case(
    id="fx-screened",
    group="g4",
    messages=(Message(role="user", content="markfour pick a fruit"),),
    expected="2026",
)
CASE_DATEY = Case(
    id="fx-datey",
    group="g5",
    messages=(Message(role="user", content="markfive what happened on 2026-10-21 exactly"),),
    expected="delta",
)
CASE_VAC = Case(
    id="fx-vac",
    group="g6",
    messages=(Message(role="user", content="   "),),
    expected="x",
)
CASE_BLANK = Case(
    id="fx-blank",
    group="g7",
    messages=(Message(role="user", content="markblank say nothing"),),
    expected="",
)
CASES = (CASE_ONE, CASE_TWO, CASE_MULTI, CASE_SCREENED, CASE_DATEY, CASE_VAC, CASE_BLANK)
EXPECTED_BY_MARKER = {
    "markone": "alpha",
    "marktwo": "beta",
    "markthree": "gamma",
    "markfour": "2026",
    "markfive": "delta",
    "markblank": "",
}


def _obs(backend, fmt, case, answer=None, ok=True):
    return {
        "candidate": fmt.name,
        "fixture": case.id,
        "input_ids": list(backend.encode(fmt.render(case.messages))),
        "output_ids": [],
        "answer": case.expected if answer is None else answer,
        "exact_match": ok,
        "boundary_found": True,
        "boundary_text_match": True,
        "raw_reason": "limit",
        "error": "",
        "status": "completed",
        "input_text": "",
    }


def _entries(backend, formats=FORMATS, cases=CASES):
    return [
        {
            "format": {"name": fmt.name},
            "summary": {"aggregate_eligible": True},
            "observations": [_obs(backend, fmt, case) for case in cases],
        }
        for fmt in formats
    ]


def _run_stage(backend, formats=FORMATS, cases=CASES, entries=None, **kwargs):
    kwargs.setdefault("boundary", tuple(backend.encode(BOUNDARY_WORD)))
    kwargs.setdefault("max_new_tokens", 8)
    kwargs.setdefault("max_context", 8192)
    return dateline_stage(
        backend,
        formats,
        cases,
        entries if entries is not None else _entries(backend, formats, cases),
        **kwargs,
    )


def _payload(backend, stage, decision="compatible_on_fixture_set", winner="dt-a"):
    return dateline_analysis(stage, decision, winner, tuple(fmt.name for fmt in FORMATS))


def _base_payload(**backend_kwargs):
    backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER, **backend_kwargs)
    return _payload(backend, _run_stage(backend))


class TestProtocol(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(dateline.PROTOCOL_VERSION, 1)
        self.assertEqual(dateline.JOINER, "\n\n")
        self.assertEqual(dateline.GENERATION_BUDGET, 28 * 2400)
        self.assertEqual(len(dateline.TREATED_KEYS), 28)
        self.assertEqual(len(dateline.DATE_ARM_KEYS), 24)
        self.assertEqual(dateline.CONTROL_KEYS, ("nodate_iso", "nodate_long", "nodate_weekday_iso"))
        self.assertEqual(dateline.SCREEN_RULE, "dateline-components-r4-v1")
        self.assertEqual(dateline.SIZING_RULE, "whole-units-plus-one-truncated-suffix-v1")

    def test_arm_table_literals(self):
        table = dateline.date_arm_table()
        self.assertEqual(table["anchor_iso"], "2026-10-21")
        self.assertEqual(table["anchor_long"], "21 October 2026")
        self.assertEqual(table["anchor_weekday_iso"], "Wednesday, 2026-10-21")
        self.assertEqual(table["minus_day_iso"], "2026-10-20")
        self.assertEqual(table["plus_day_weekday_iso"], "Thursday, 2026-10-22")
        self.assertEqual(table["minus_year_long"], "21 October 2025")
        self.assertEqual(table["plus_year_iso"], "2027-10-21")
        self.assertEqual(table["leap_iso"], "2028-02-29")
        self.assertEqual(table["leap_long"], "29 February 2028")
        self.assertEqual(table["leap_weekday_iso"], "Tuesday, 2028-02-29")
        self.assertEqual(table["boundary_eve_long"], "31 December 2026")
        self.assertEqual(table["boundary_day_iso"], "2027-01-01")
        self.assertEqual(table["lie"], "Thursday, 2026-10-21")

    def test_weekdays_computed_from_each_date(self):
        # Independent oracle: datetime's own calendar. Every
        # weekday-bearing arm's name must be its own date's
        # weekday -- the anchor-templating bug would fail here.
        for value, (year, month, day) in dateline.DATE_VALUES:
            expected = datetime.date(year, month, day).strftime("%A")
            rendered = dateline.format_date(value, "weekday_iso")
            self.assertTrue(rendered.startswith(expected + ","), (value, rendered))

    def test_lie_differs_from_honest_anchor(self):
        self.assertNotEqual(
            dateline.lie_date_string(), dateline.format_date("anchor", "weekday_iso")
        )
        self.assertEqual(dateline.arm_date_string("lie"), "Thursday, 2026-10-21")
        with self.assertRaises(ValueError):
            dateline.arm_date_string("nodate_iso")
        with self.assertRaises(ValueError):
            dateline.format_date("anchor", "roman")

    def test_protocol_fingerprint_stable_and_bound(self):
        fp1 = dateline.protocol_fingerprint(["fx-b", "fx-a"], "sha256:x")
        fp2 = dateline.protocol_fingerprint(["fx-a", "fx-b"], "sha256:x")
        self.assertEqual(fp1, fp2)  # fixture order is sorted out
        self.assertNotEqual(fp1, dateline.protocol_fingerprint(["fx-a"], "sha256:x"))
        self.assertNotEqual(fp1, dateline.protocol_fingerprint(["fx-a", "fx-b"], "sha256:y"))
        protocol = dateline.protocol_object(["fx-a"], "sha256:x")
        self.assertEqual(protocol["arm_dates"]["lie"], "Thursday, 2026-10-21")
        self.assertEqual(protocol["treated_arms"], list(dateline.TREATED_KEYS))

    def test_no_clock_or_locale_calls(self):
        # G10 pin: the module, the stage/analysis segment of
        # experiments.py, and the reporting section must not read
        # the wall clock or the locale.
        root = Path(__file__).resolve().parent.parent / "model_protocol_lab"
        banned = (".today(", "datetime.now(", "strftime(", "strptime(", "time.time(")
        module_src = (root / "dateline.py").read_text(encoding="utf-8")
        for token in banned:
            self.assertNotIn(token, module_src)
        experiments_src = (root / "experiments.py").read_text(encoding="utf-8")
        stage_segment = experiments_src[
            experiments_src.index("def dateline_stage") : experiments_src.index("def numform_stage")
        ]
        for token in banned:
            self.assertNotIn(token, stage_segment)
        reporting_src = (root / "reporting.py").read_text(encoding="utf-8")
        section = reporting_src[
            reporting_src.index("def _dateline_section") : reporting_src.index(
                "def _cache_profile_section"
            )
        ]
        for token in banned:
            self.assertNotIn(token, section)


class TestScreen(unittest.TestCase):
    def test_battery_screened(self):
        for expected in (
            "21",
            "2026",
            "October",
            "Wednesday",
            "Thursday",
            "October 21, 2026",
            "February 29",
            "October 21",
            "Oct",
            "Dec",
            "021",
            "2",
            "2028",
            "29",
            "31 December",
            "1 January 2027",
            "2026-10-21",
            "21/10/2026",
            "31/12/2026",
        ):
            self.assertTrue(dateline.screen_expected(expected), expected)

    def test_battery_not_screened(self):
        for expected in ("March 3", "is", "26", "tomorrow", "alpha", "0", "3", "2024"):
            self.assertFalse(dateline.screen_expected(expected), expected)

    def test_parse_candidates(self):
        self.assertEqual(dateline.parse_date_candidates("2026-10-21"), {(2026, 10, 21)})
        self.assertEqual(dateline.parse_date_candidates("3/4/2026"), {(2026, 3, 4), (2026, 4, 3)})
        self.assertEqual(dateline.parse_date_candidates("29.2.2028"), {(2028, 2, 29)})
        self.assertEqual(dateline.parse_date_candidates("February 29"), {(None, 2, 29)})
        self.assertEqual(dateline.parse_date_candidates("2026-02-29"), set())  # not a real date
        self.assertEqual(dateline.parse_date_candidates("not a date"), set())

    def test_blank_expected_not_screened(self):
        self.assertFalse(dateline.screen_expected(""))
        self.assertFalse(dateline.screen_expected("   "))

    def test_message_states_date(self):
        self.assertTrue(dateline.message_states_date("what happened on 2026-10-21 exactly"))
        self.assertTrue(dateline.message_states_date("due 21 October 2026 at noon"))
        self.assertTrue(dateline.message_states_date("on October 21 we met"))
        self.assertFalse(dateline.message_states_date("name a colour"))
        self.assertFalse(dateline.message_states_date("the 21 pilots"))

    def test_fixture_screen(self):
        self.assertEqual(dateline.fixture_screen(CASE_SCREENED), (True, False))
        self.assertEqual(dateline.fixture_screen(CASE_DATEY), (False, True))
        self.assertEqual(dateline.fixture_screen(CASE_ONE), (False, False))

    def test_screen_neutral(self):
        screened_fixtures, screened_candidates = dateline.screen_neutral(
            {"fx-1": "the reference panel is mounted", "fx-2": "alpha"},
            {"cand": "on the north wall"},
        )
        self.assertEqual(screened_fixtures, frozenset({"fx-1"}))
        self.assertEqual(screened_candidates, frozenset({"cand"}))


class TestConstruction(unittest.TestCase):
    def test_build_arm_fixture_prepends_block(self):
        treated = dateline.build_arm_fixture(CASE_ONE, "anchor_iso")
        content = treated.messages[-1].content
        self.assertEqual(
            content,
            "[Context] Today's date is 2026-10-21. [/Context]\n\nmarkone name a colour",
        )
        self.assertEqual(treated.id, CASE_ONE.id)
        self.assertEqual(treated.expected, CASE_ONE.expected)

    def test_build_control_fixture(self):
        treated = dateline.build_arm_fixture(CASE_ONE, "nodate_long", "Sized neutral text. ")
        content = treated.messages[-1].content
        self.assertTrue(content.startswith("[Context] Sized neutral text. [/Context]\n\n"))

    def test_neutral_text_and_search_order(self):
        self.assertEqual(dateline.neutral_text(0, 5), dateline.NEUTRAL_UNIT[:5])
        self.assertEqual(dateline.neutral_text(2, 0), dateline.NEUTRAL_UNIT * 2)
        order = dateline.neutral_search_order(11, 23)
        self.assertLessEqual(len(order), dateline.SIZING_SEARCH_BUDGET)
        self.assertEqual(order, dateline.neutral_search_order(11, 23))  # deterministic
        self.assertNotIn((0, 0), order)

    def test_classify_construction(self):
        self.assertEqual(
            dateline.classify_construction((1, 2), (7, 8, 9, 1, 2), (7,), (8, 9)),
            ("id_exact", ()),
        )
        kind, seam = dateline.classify_construction((1, 2), (7, 5, 9, 1, 2), (7,), (8, 9))
        self.assertEqual(kind, "text_level")
        self.assertTrue(seam)
        self.assertEqual(dateline.classify_construction(None, (1,), (7,), (8,)), ("text_level", ()))

    def test_classify_pair_truth_table(self):
        self.assertEqual(dateline.classify_pair(5, 0, 0), "insufficient")
        self.assertEqual(dateline.classify_pair(6, 0, 0), "no_discordance_observed")
        self.assertEqual(dateline.classify_pair(6, 2, 2), "discordant_balanced")
        self.assertEqual(dateline.classify_pair(6, 3, 0), "discordant_asymmetric")
        self.assertEqual(dateline.classify_pair(6, 0, 2), "discordant_asymmetric")
        self.assertEqual(dateline.classify_pair(6, 1, 0), "below_threshold")
        self.assertEqual(_dateline_classify(6, 3, 0), "discordant_asymmetric")

    def test_mcnemar_mirror_matches(self):
        for a_only, b_only in ((0, 0), (1, 0), (3, 1), (5, 5), (9, 2)):
            self.assertEqual(
                _dateline_mcnemar_p(a_only, b_only),
                mcnemar_exact_p(a_only, b_only),
            )


class TestStage(unittest.TestCase):
    def test_stage_available_and_classes(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        stage = _run_stage(backend)
        self.assertTrue(stage["available"])
        records = stage["candidates"]["dt-a"]["records"]
        by_id = {record["fixture"]: record for record in records}
        self.assertEqual(by_id["fx-screened"]["cell_details"]["anchor_iso"], "screened")
        self.assertEqual(by_id["fx-datey"]["cell_details"]["anchor_iso"], "message_date_bearing")
        self.assertEqual(by_id["fx-vac"]["cell_details"]["anchor_iso"], "vacuous")
        self.assertEqual(by_id["fx-blank"]["cell_details"]["anchor_iso"], "expected_blank")
        self.assertIsNotNone(by_id["fx-one"]["cells"]["anchor_iso"])
        # 2 candidates x 3 live fixtures x 28 arms
        self.assertEqual(backend.generate_calls, 2 * 3 * 28)

    def test_control_matches_anchor_tokens(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        stage = _run_stage(backend)
        for candidate in stage["candidates"].values():
            for record in candidate["records"]:
                for fmt in dateline.FORMATS:
                    control = record["cells"][f"nodate_{fmt}"]
                    anchor = record["cells"][f"anchor_{fmt}"]
                    if control is not None and anchor is not None:
                        self.assertEqual(
                            control["input_tokens"], anchor["input_tokens"], record["fixture"]
                        )

    def test_scripted_leap_loss(self):
        backend = _DateBackend(
            fail_on={("marktwo", "2028-02-29"), ("marktwo", "29 February 2028")},
            expected_by_marker=EXPECTED_BY_MARKER,
        )
        payload = _payload(backend, _run_stage(backend))
        self.assertTrue(_valid_dateline(payload))
        candidate = payload["candidates"]["dt-a"]
        self.assertEqual(candidate["date_loss"]["leap"]["n_loss"], 1)
        self.assertEqual(candidate["date_loss"]["anchor"]["n_loss"], 0)
        self.assertFalse(candidate["date_content_falsified"])
        self.assertEqual(payload["n_multi_turn_fixtures"], 1)
        self.assertEqual(payload["n_search_fixtures"], len(CASES))

    def test_falsified_when_nothing_flips(self):
        payload = _base_payload()
        self.assertTrue(_valid_dateline(payload))
        for candidate in payload["candidates"].values():
            if candidate["status"] == "available":
                self.assertTrue(candidate["date_content_falsified"])

    def test_context_exceeded_cells(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        stage = _run_stage(backend, max_context=10)
        record = stage["candidates"]["dt-a"]["records"][0]
        self.assertEqual(record["cell_details"]["anchor_iso"], "arm_context_exceeded")
        self.assertIsNone(record["cells"]["anchor_iso"])

    def test_cancelled_stage(self):
        from threading import Event

        event = Event()
        event.set()
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        stage = _run_stage(backend, cancel=event)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_native_unavailable_candidate(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        entries = _entries(backend)
        entries[0]["summary"]["aggregate_eligible"] = False
        stage = _run_stage(backend, entries=entries)
        self.assertEqual(stage["candidates"]["dt-a"]["status"], "unavailable")
        self.assertEqual(stage["candidates"]["dt-a"]["detail"], "native_search_unavailable")

    def test_budget_exceeded(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        with mock.patch.object(dateline, "MAX_GROUND_RECORDS", 3):
            stage = _run_stage(backend)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "dateline_cell_budget_exceeded")

    def test_screen_failure_fails_closed(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        with mock.patch(
            "model_protocol_lab.fixtures.audit_special_token_literals", return_value=True
        ):
            stage = _run_stage(backend)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "dateline_screen_failed")

    def test_stage_rejects_bad_arguments(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        with self.assertRaises(ValueError):
            dateline_stage(backend, (), CASES, [], boundary=(1,), max_context=100)
        with self.assertRaises(ValueError):
            dateline_stage(backend, FORMATS, (), [], boundary=(1,), max_context=100)


class TestValidatorBattery(unittest.TestCase):
    """One break at a time against a genuine payload."""

    def setUp(self):
        self.payload = _base_payload(
            fail_on={("marktwo", "2028-02-29"), ("marktwo", "29 February 2028")}
        )
        self.assertTrue(_valid_dateline(self.payload))

    def _break(self, mutate):
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        self.assertFalse(_valid_dateline(payload))

    def test_fingerprint_zeroed(self):
        self._break(lambda p: p["protocol"].__setitem__("fingerprint", "0" * 64))

    def test_protocol_extra_key(self):
        self._break(lambda p: p["protocol"].__setitem__("extra", 1))

    def test_arm_table_weekday_templated(self):
        # The anchor-weekday attack: every weekday arm rendered
        # with Wednesday. The validator's own calendar recompute
        # must reject the table.
        def mutate(p):
            p["protocol"]["arm_dates"]["leap_weekday_iso"] = "Wednesday, 2028-02-29"

        self._break(mutate)

    def test_lie_made_honest(self):
        def mutate(p):
            p["protocol"]["arm_dates"]["lie"] = "Wednesday, 2026-10-21"

        self._break(mutate)

    def test_cell_verdict_flipped(self):
        def mutate(p):
            cell = p["candidates"]["dt-a"]["records"][0]["cells"]["anchor_iso"]
            cell["verdict"] = not cell["verdict"]

        self._break(mutate)

    def test_control_tokens_mismatched(self):
        def mutate(p):
            cell = p["candidates"]["dt-a"]["records"][0]["cells"]["nodate_iso"]
            cell["input_tokens"] += 1

        self._break(mutate)

    def test_control_evaluable_lied(self):
        def mutate(p):
            record = p["candidates"]["dt-a"]["records"][0]
            record["control_evaluable"]["iso"] = False

        self._break(mutate)

    def test_screened_flag_one_candidate_only(self):
        def mutate(p):
            p["candidates"]["dt-a"]["records"][0]["screened"] = True

        self._break(mutate)

    def test_aggregate_tampered(self):
        def mutate(p):
            p["candidates"]["dt-a"]["date_loss"]["leap"]["n_loss"] += 1

        self._break(mutate)

    def test_falsified_flag_tampered(self):
        def mutate(p):
            p["candidates"]["dt-a"]["date_content_falsified"] = True

        self._break(mutate)

    def test_arm_selection_tampered(self):
        def mutate(p):
            p["arms"]["anchor_iso"]["flip"] = None

        self._break(mutate)

    def test_common_population_tampered(self):
        def mutate(p):
            p["common_population"]["n_common"] += 1

        self._break(mutate)

    def test_multi_turn_count_tampered(self):
        def mutate(p):
            p["n_multi_turn_fixtures"] += 1

        self._break(mutate)

    def test_duplicate_fixture_id(self):
        def mutate(p):
            records = p["candidates"]["dt-a"]["records"]
            records[1]["fixture"] = records[0]["fixture"]

        self._break(mutate)

    def test_negative_seam_id(self):
        def mutate(p):
            p["candidates"]["dt-a"]["records"][0]["seam_ids"] = [-1]

        self._break(mutate)

    def test_detail_vocabulary(self):
        def mutate(p):
            p["candidates"]["dt-a"]["records"][0]["cell_details"]["anchor_iso"] = "bogus"

        self._break(mutate)

    def test_unavailable_payload_invalid(self):
        self.assertFalse(_valid_dateline({"available": False, "detail": "x", "protocol": None}))
        self.assertFalse(_valid_dateline(None))


class TestDiff(unittest.TestCase):
    def test_self_diff_unchanged(self):
        payload = _base_payload()
        diff = diff_dateline(payload, payload)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])

    def test_foreign_backend_not_comparable(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        stage = _run_stage(backend)
        payload = _payload(backend, stage)
        other_backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        other_backend.fingerprint = "sha256:other"
        other = _payload(other_backend, _run_stage(other_backend))
        diff = diff_dateline(payload, other)
        self.assertTrue(diff["backend_drift"])
        self.assertFalse(diff["comparable"])
        self.assertTrue(diff["changed"])

    def test_loss_drift_detected(self):
        calm = _base_payload()
        stormy = _base_payload(fail_on={("marktwo", "2028-02-29"), ("marktwo", "29 February 2028")})
        diff = diff_dateline(calm, stormy)
        self.assertTrue(diff["changed"])
        self.assertTrue(any(entry["metric"] == "leap_loss_rate" for entry in diff["loss_drift"]))

    def test_absent_sides(self):
        payload = _base_payload()
        self.assertEqual(diff_dateline(None, payload)["available"], "new")
        self.assertEqual(diff_dateline(payload, None)["available"], "old")
        self.assertEqual(diff_dateline(None, None)["available"], "neither")


class TestReportingAndCli(unittest.TestCase):
    def test_section_renders(self):
        payload = _base_payload(
            fail_on={("marktwo", "2028-02-29"), ("marktwo", "29 February 2028")}
        )
        html = render_html({"decision": "selected", "dateline": payload})
        self.assertIn("Date-line sensitivity", html)
        self.assertIn('id="dateline"', html)

    def test_section_absent_without_payload(self):
        html = render_html({"decision": "selected"})
        self.assertNotIn('id="dateline"', html)

    def test_cli_helper_gates_on_incomplete_search(self):
        result: dict = {"decision": "failed", "search": []}
        cli._run_dateline_probe(
            _DateBackend(),
            FORMATS,
            result,
            CASES,
            boundary=(1,),
            boundary_map=None,
            max_new_tokens=8,
            max_context=100,
            seed=0,
            cancel=None,
        )
        self.assertEqual(
            result["dateline"], {"available": False, "detail": "native_search_incomplete"}
        )


class TestParseBattery(unittest.TestCase):
    """Every closed date format, valid and invalid readings."""

    def test_all_formats(self):
        cases = {
            "21 October 2026": {(2026, 10, 21)},
            "October 21, 2026": {(2026, 10, 21)},
            "2026/10/21": {(2026, 10, 21)},
            "31 December": {(None, 12, 31)},
            "October 21": {(None, 10, 21)},
            "13/4/2026": {(2026, 4, 13)},  # only the D/M reading is a real date
            "29 February 2028": {(2028, 2, 29)},
        }
        for text, expected in cases.items():
            self.assertEqual(dateline.parse_date_candidates(text), expected, text)

    def test_parse_hits_arm_direct(self):
        # Inside screen_expected the component leg fires first
        # for every string below (their components are all
        # date-bearing arm components). The parse leg is pinned
        # directly: it is the leg that decides whenever a
        # reading's components do not rung-match, and its
        # year-less month/day rule is part of the frozen screen.
        self.assertTrue(dateline._parse_hits_arm("2026-10-21"))
        self.assertTrue(dateline._parse_hits_arm("October 21"))
        self.assertTrue(dateline._parse_hits_arm("29 February 2028"))
        self.assertFalse(dateline._parse_hits_arm("2026-10-23"))
        self.assertFalse(dateline._parse_hits_arm("March 3"))
        self.assertFalse(dateline._parse_hits_arm("not a date"))

    def test_invalid_readings(self):
        for text in (
            "32 October 2026",  # day out of range
            "October 32, 2026",
            "21 Smarch 2026",  # regex shape, unknown month name
            "Smarch 21, 2026",
            "21 Smarch",  # year-less, unknown month
            "Smarch 21",
            "31.2.2026",  # February has no 31st
            "February 30",  # year-less, checked against a leap year
            "31 April",
            "29 February 2026",  # 2026 is not a leap year
            "2026-13-01",  # month out of range
        ):
            self.assertEqual(dateline.parse_date_candidates(text), set(), text)


def _synthetic_record(fixture, cells):
    return {
        "fixture": fixture,
        "group": "g",
        "n_user_messages": 1,
        "vacuous": False,
        "expected_blank": False,
        "screened": False,
        "message_date_bearing": False,
        "cells": cells,
    }


def _verdict_cell(verdict):
    return {"verdict": verdict, "boundary_found": True, "input_tokens": 5}


class TestAggregationEdges(unittest.TestCase):
    def test_cell_helper_rejects_non_dicts(self):
        self.assertIsNone(dateline._cell({"cells": None}, "x"))
        self.assertIsNone(dateline._cell({"cells": {"x": 5}}, "x"))
        self.assertIsNone(dateline._cell({}, "x"))

    def test_aggregate_empty_records(self):
        aggregate = dateline.aggregate_candidate([])
        self.assertIsNone(aggregate["arms"]["anchor_iso"]["rate"])
        self.assertEqual(aggregate["arms"]["anchor_iso"]["n_completed"], 0)
        self.assertTrue(aggregate["date_content_falsified"])

    def test_loss_counterpart_and_lie_split(self):
        def record(fixture, control, anchor_wd, lie):
            cells = {
                **{f"nodate_{fmt}": _verdict_cell(control) for fmt in dateline.FORMATS},
                **{f"anchor_{fmt}": _verdict_cell(True) for fmt in dateline.FORMATS},
                "anchor_weekday_iso": _verdict_cell(anchor_wd),
                "lie": _verdict_cell(lie),
            }
            return _synthetic_record(fixture, cells)

        records = [
            record("fx-cp", False, True, True),  # fails control, passes value
            record("fx-any", True, False, False),  # loses anchor, fails lie too
            record("fx-spec", True, False, True),  # loses anchor, passes lie
        ]
        block = dateline._loss_block(records, "anchor")
        self.assertEqual(block["n_counterpart"], 1)
        self.assertEqual(block["n_loss"], 2)
        self.assertEqual(block["lie_split"]["n_loss_any_text"], 1)
        self.assertEqual(block["lie_split"]["n_loss_content_specific"], 1)
        self.assertEqual(block["lie_split"]["n_evaluable"], 3)

    def test_loss_block_empty(self):
        block = dateline._loss_block([], "leap")
        self.assertEqual(block["n_evaluable"], 0)
        self.assertIsNone(block["loss_rate"])
        self.assertIsNone(block["lie_split"])

    def test_common_population_empty(self):
        out = dateline.common_population({})
        self.assertEqual(out["n_common"], 0)
        self.assertEqual(out["detail"], "common_population_empty")
        disjoint = dateline.common_population(
            {
                "a": [_synthetic_record("fx-1", {})],
                "b": [_synthetic_record("fx-2", {})],
            }
        )
        self.assertEqual(disjoint["n_common"], 0)


class _DecodeRaisesBoundaryBackend(_DateBackend):
    """Raises only for the exact boundary-id tuple, which is how
    the stage asks for the boundary text; answer slices are lists
    or different ids and decode normally."""

    def decode(self, ids):
        if (
            isinstance(ids, tuple)
            and BOUNDARY_WORD in self.vocab
            and list(ids) == [self.vocab[BOUNDARY_WORD]]
        ):
            raise ValueError("undecodable boundary")
        return super().decode(ids)


class _EventSettingBackend(_DateBackend):
    def __init__(self, event, **kwargs):
        super().__init__(**kwargs)
        self._event = event

    def generate(self, input_ids, **kwargs):
        out = super().generate(input_ids, **kwargs)
        self._event.set()
        return out


class _DateRaisingBackend(_DateBackend):
    def generate(self, input_ids, **kwargs):
        if "2026-10-21" in self.decode(list(input_ids)):
            raise RuntimeError("boom")
        return super().generate(input_ids, **kwargs)


class _TripleRefBackend(_DateBackend):
    """Words cost 3 tokens whenever the text mentions the neutral
    unit's subject, which makes the control-sizing equation
    unsolvable (see cycle-188 gap analysis)."""

    def encode(self, text):
        ids = super().encode(text)
        if "reference" in text:
            tripled: list[int] = []
            for token in ids:
                tripled.extend([token] * 3)
            return tuple(tripled)
        return ids


class _TruncatingBackend(_DateBackend):
    """A tokenizer with a hard input cap: renders longer than the
    cap encode to the same length, so the empty control already
    matches the anchor arm's rendered length (delta == 0)."""

    def encode(self, text):
        return super().encode(text)[:12]


class TestStageEdges(unittest.TestCase):
    def test_junk_search_entry_ignored(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        entries = _entries(backend)
        entries.append("junk")
        stage = _run_stage(backend, entries=entries)
        self.assertTrue(stage["available"])

    def test_undecodable_boundary_screens_nothing(self):
        backend = _DecodeRaisesBoundaryBackend(expected_by_marker=EXPECTED_BY_MARKER)
        stage = _run_stage(backend)
        self.assertTrue(stage["available"])
        record = stage["candidates"]["dt-a"]["records"][0]
        self.assertIsNotNone(record["cells"]["anchor_iso"])

    def test_bad_observation_shape_unavailable(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        entries = _entries(backend)
        entries[0]["observations"][0] = {"fixture": 5}
        stage = _run_stage(backend, entries=entries)
        self.assertEqual(stage["candidates"]["dt-a"]["status"], "unavailable")

    def test_missing_observation_unavailable(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        entries = _entries(backend)
        entries[0]["observations"] = entries[0]["observations"][:-1]
        stage = _run_stage(backend, entries=entries)
        self.assertEqual(stage["candidates"]["dt-a"]["status"], "unavailable")

    def test_arm_screened_by_boundary_text(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        boundary = tuple(backend.encode("2026-10-21"))
        stage = _run_stage(backend, boundary=boundary)
        record = stage["candidates"]["dt-a"]["records"][0]
        self.assertEqual(record["cell_details"]["anchor_iso"], "arm_screened")
        self.assertIsNone(record["cells"]["anchor_iso"])
        self.assertEqual(record["cell_details"]["anchor_long"], "")

    def test_control_screened_by_boundary_and_fixture(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        boundary = tuple(backend.encode("north"))
        stage = _run_stage(backend, boundary=boundary)
        record = stage["candidates"]["dt-a"]["records"][0]
        self.assertEqual(record["cell_details"]["nodate_iso"], "control_screened")
        panel_case = Case(
            id="fx-panel",
            group="g8",
            messages=(Message(role="user", content="markpanel choose"),),
            expected="panel",
        )
        markers = dict(EXPECTED_BY_MARKER)
        markers["markpanel"] = "panel"
        backend2 = _DateBackend(expected_by_marker=markers)
        stage2 = _run_stage(backend2, cases=CASES + (panel_case,))
        by_id = {r["fixture"]: r for r in stage2["candidates"]["dt-a"]["records"]}
        self.assertEqual(by_id["fx-panel"]["cell_details"]["nodate_long"], "control_screened")
        self.assertEqual(by_id["fx-one"]["cell_details"]["nodate_long"], "")

    def test_control_unmatchable(self):
        backend = _TripleRefBackend(expected_by_marker=EXPECTED_BY_MARKER)
        stage = _run_stage(backend)
        record = stage["candidates"]["dt-a"]["records"][0]
        self.assertEqual(record["cell_details"]["nodate_iso"], "control_unmatchable")
        self.assertFalse(record["control_evaluable"]["iso"])
        payload = _payload(backend, stage)
        self.assertTrue(payload["arms"]["nodate_iso"]["withheld"])
        self.assertTrue(_valid_dateline(payload))

    def test_delta_zero_control(self):
        backend = _TruncatingBackend(expected_by_marker=EXPECTED_BY_MARKER)
        stage = _run_stage(backend)
        record = stage["candidates"]["dt-a"]["records"][0]
        self.assertTrue(record["control_evaluable"]["iso"])
        self.assertIsNotNone(record["cells"]["nodate_iso"])
        self.assertEqual(
            record["cells"]["nodate_iso"]["input_tokens"],
            record["cells"]["anchor_iso"]["input_tokens"],
        )

    def test_cancel_during_cells(self):
        from threading import Event

        event = Event()
        backend = _EventSettingBackend(event, expected_by_marker=EXPECTED_BY_MARKER)
        stage = _run_stage(backend, cancel=event)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_evaluation_failed_cells(self):
        backend = _DateRaisingBackend(expected_by_marker=EXPECTED_BY_MARKER)
        stage = _run_stage(backend)
        record = stage["candidates"]["dt-a"]["records"][0]
        self.assertEqual(record["cell_details"]["anchor_iso"], "arm_evaluation_failed")
        self.assertIsNone(record["cells"]["anchor_iso"])


SEVEN_CASES = tuple(
    Case(
        id=f"fx-{letter}",
        group=f"g-{letter}",
        messages=(Message(role="user", content=f"mark{letter} pick one"),),
        expected=f"ans-{letter}",
    )
    for letter in "abcdefg"
)
SEVEN_MARKERS = {f"mark{letter}": f"ans-{letter}" for letter in "abcdefg"}


class TestAnalysisEdges(unittest.TestCase):
    def test_analysis_unavailable_stage(self):
        out = dateline_analysis(
            {"available": False, "detail": "cancelled", "protocol": None},
            "selected",
            "dt-a",
            ("dt-a",),
        )
        self.assertFalse(out["available"])
        self.assertEqual(out["detail"], "cancelled")

    def test_arm_selection_outcomes(self):
        calm = _base_payload()
        self.assertEqual(calm["arms"]["anchor_iso"]["decision"], "ambiguous")
        one_sided = _base_payload(fail_on={("markone", "USER: ")})
        self.assertEqual(one_sided["arms"]["anchor_iso"]["decision"], "selected")
        self.assertEqual(one_sided["arms"]["anchor_iso"]["winner"], "dt-a")
        self.assertTrue(_valid_dateline(one_sided))
        none_pass = _base_payload(fail_on={("markone", "2026-10-21")})
        self.assertEqual(none_pass["arms"]["anchor_iso"]["decision"], "no_candidate")
        self.assertTrue(_valid_dateline(none_pass))

    def test_mixed_availability_payload_valid_and_withheld(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        entries = _entries(backend)
        entries[1]["summary"]["aggregate_eligible"] = False
        payload = _payload(backend, _run_stage(backend, entries=entries))
        self.assertTrue(_valid_dateline(payload))
        self.assertTrue(payload["arms"]["anchor_iso"]["withheld"])

    def test_completed_set_mismatch_withholds(self):
        bloated = Format(
            name="dt-bloated",
            bos=" ".join(["filler"] * 300),
            user_prefix="USER: ",
            assistant_prefix="ASSISTANT: ",
            turn_end="~",
        )
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        stage = _run_stage(backend, formats=(FMT_A, bloated), max_context=100)
        payload = dateline_analysis(
            stage, "compatible_on_fixture_set", "dt-a", ("dt-a", "dt-bloated")
        )
        self.assertTrue(payload["arms"]["anchor_iso"]["withheld"])
        self.assertTrue(_valid_dateline(payload))

    def test_non_selection_primary_decision_payload_valid(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        stage = _run_stage(backend)
        # Cycle 196 (oversight edge-case F3): the validator binds
        # primary_decision to the run-decision vocabulary, so this
        # uses a real abstention decision ("ambiguous"); the earlier
        # synthetic "abstain" label is not producible by any run and
        # no longer validates.
        payload = dateline_analysis(stage, "ambiguous", None, tuple(f.name for f in FORMATS))
        self.assertTrue(_valid_dateline(payload))

    def test_context_exceeded_payload_valid(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        payload = _payload(backend, _run_stage(backend, max_context=10))
        self.assertTrue(_valid_dateline(payload))

    def test_counterpart_and_lie_split_payload(self):
        backend = _DateBackend(
            fail_on={
                ("markthree", "reference panel"),
                ("marktwo", "Wednesday, 2026-10-21"),
                ("marktwo", "Thursday, 2026-10-21"),
                ("markone", "Wednesday, 2026-10-21"),
            },
            expected_by_marker=EXPECTED_BY_MARKER,
        )
        payload = _payload(backend, _run_stage(backend))
        self.assertTrue(_valid_dateline(payload))
        block = payload["candidates"]["dt-a"]["date_loss"]["anchor"]
        self.assertEqual(block["n_counterpart"], 1)
        self.assertEqual(block["n_loss"], 2)
        self.assertEqual(block["lie_split"]["n_loss_any_text"], 1)
        self.assertEqual(block["lie_split"]["n_loss_content_specific"], 1)

    def test_pair_classifications_at_scale(self):
        backend = _DateBackend(
            fail_on={
                ("marka", "20 October 2026"),
                ("markb", "20 October 2026"),
                ("markc", "22 October 2026"),
                ("markd", "22 October 2026"),
                ("marke", "29 February 2028"),
            },
            expected_by_marker=SEVEN_MARKERS,
        )
        stage = _run_stage(backend, cases=SEVEN_CASES)
        payload = _payload(backend, stage)
        self.assertTrue(_valid_dateline(payload))
        table = payload["candidates"]["dt-a"]["pairs_by_format"]["long"]
        self.assertEqual(table["minus_day|plus_day"]["classification"], "discordant_balanced")
        self.assertEqual(table["leap|boundary_eve"]["classification"], "below_threshold")
        calm_pair = payload["candidates"]["dt-a"]["pairs_by_format"]["iso"]
        self.assertEqual(
            calm_pair["anchor|minus_year"]["classification"], "no_discordance_observed"
        )


class TestValidatorBattery2(unittest.TestCase):
    """Second battery: protocol scalars, record-shape checks, and
    the record caps, one break at a time."""

    def setUp(self):
        self.payload = _base_payload()
        self.assertTrue(_valid_dateline(self.payload))

    def _break(self, mutate):
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        self.assertFalse(_valid_dateline(payload))

    def _record(self, p, index=0):
        return p["candidates"]["dt-a"]["records"][index]

    def test_protocol_not_a_dict(self):
        self._break(lambda p: p.__setitem__("protocol", "junk"))

    def test_protocol_version(self):
        self._break(lambda p: p["protocol"].__setitem__("version", 2))

    def test_protocol_block_template(self):
        self._break(lambda p: p["protocol"].__setitem__("block_template", "x"))

    def test_protocol_control_template(self):
        self._break(lambda p: p["protocol"].__setitem__("control_template", "x"))

    def test_protocol_joiner(self):
        self._break(lambda p: p["protocol"].__setitem__("joiner", " "))

    def test_protocol_neutral_unit(self):
        self._break(lambda p: p["protocol"].__setitem__("neutral_unit", "x"))

    def test_protocol_sizing_rule(self):
        self._break(lambda p: p["protocol"].__setitem__("sizing_rule", "x"))

    def test_protocol_screen_rule(self):
        self._break(lambda p: p["protocol"].__setitem__("screen_rule", "x"))

    def test_protocol_treated_arms(self):
        self._break(lambda p: p["protocol"].__setitem__("treated_arms", []))

    def test_protocol_floors(self):
        self._break(lambda p: p["protocol"].__setitem__("min_paired_evaluable", 5))

    def test_protocol_budget(self):
        self._break(lambda p: p["protocol"].__setitem__("generation_budget", 1))

    def test_protocol_records_bytes_cap_field(self):
        self._break(lambda p: p["protocol"].__setitem__("max_records_bytes", 1))

    def test_protocol_fixture_sha(self):
        self._break(lambda p: p["protocol"].__setitem__("fixture_set_sha256", "0" * 64))

    def test_candidates_not_a_dict(self):
        self._break(lambda p: p.__setitem__("candidates", []))

    def test_candidate_entry_not_a_dict(self):
        self._break(lambda p: p["candidates"].__setitem__("dt-a", "junk"))

    def test_candidate_status_vocabulary(self):
        self._break(lambda p: p["candidates"]["dt-a"].__setitem__("status", "weird"))

    def test_candidate_records_not_a_list(self):
        self._break(lambda p: p["candidates"]["dt-a"].__setitem__("records", {}))

    def test_unavailable_with_records(self):
        def mutate(p):
            p["candidates"]["dt-a"]["status"] = "unavailable"

        self._break(mutate)

    def test_candidate_record_count_cap(self):
        def mutate(p):
            entry = p["candidates"]["dt-a"]
            entry["records"] = entry["records"] * 15

        self._break(mutate)

    def test_record_not_a_dict(self):
        def mutate(p):
            p["candidates"]["dt-a"]["records"][0] = "junk"

        self._break(mutate)

    def test_record_fixture_empty(self):
        self._break(lambda p: self._record(p).__setitem__("fixture", ""))

    def test_record_group_type(self):
        self._break(lambda p: self._record(p).__setitem__("group", 5))

    def test_record_n_user_messages(self):
        self._break(lambda p: self._record(p).__setitem__("n_user_messages", 0))

    def test_record_flag_type(self):
        self._break(lambda p: self._record(p).__setitem__("vacuous", "yes"))

    def test_record_construction_vocabulary(self):
        self._break(lambda p: self._record(p).__setitem__("construction", "bogus"))

    def test_record_seam_type(self):
        self._break(lambda p: self._record(p).__setitem__("seam_ids", "x"))

    def test_record_expected_digest_shape(self):
        self._break(lambda p: self._record(p).__setitem__("expected_digest", "zz"))

    def test_record_expected_digest_none(self):
        self._break(lambda p: self._record(p).__setitem__("expected_digest", None))

    def test_record_control_evaluable_keys(self):
        def mutate(p):
            del self._record(p)["control_evaluable"]["iso"]

        self._break(mutate)

    def test_record_control_evaluable_value_type(self):
        self._break(lambda p: self._record(p)["control_evaluable"].__setitem__("iso", "x"))

    def test_record_native_verdict_flipped(self):
        def mutate(p):
            native = self._record(p)["native"]
            native["verdict"] = not native["verdict"]

        self._break(mutate)

    def test_record_cells_key_dropped(self):
        def mutate(p):
            del self._record(p)["cells"]["lie"]

        self._break(mutate)

    def test_record_details_key_dropped(self):
        def mutate(p):
            del self._record(p)["cell_details"]["lie"]

        self._break(mutate)

    def test_live_detail_set_to_class_word(self):
        self._break(lambda p: self._record(p)["cell_details"].__setitem__("anchor_iso", "screened"))

    def test_control_unmatchable_while_evaluable(self):
        def mutate(p):
            record = self._record(p)
            record["cells"]["nodate_iso"] = None
            record["cell_details"]["nodate_iso"] = "control_unmatchable"

        self._break(mutate)

    def test_treated_cell_boundary_flipped(self):
        def mutate(p):
            self._record(p)["cells"]["anchor_iso"]["boundary_found"] = False

        self._break(mutate)

    def _mega_payload(self, n_candidates, id_width):
        payload = copy.deepcopy(self.payload)
        template = copy.deepcopy(payload["candidates"]["dt-a"]["records"][0])
        candidates = {}
        for index in range(n_candidates):
            records = []
            for j in range(100):
                record = copy.deepcopy(template)
                record["fixture"] = f"fx-{j:03d}-" + "x" * id_width
                records.append(record)
            candidates[f"cand-{index:02d}"] = {
                "status": "available",
                "detail": "",
                "records": records,
            }
        payload["candidates"] = candidates
        return payload

    def test_total_records_cap(self):
        self.assertFalse(_valid_dateline(self._mega_payload(25, 0)))

    def test_records_bytes_cap(self):
        self.assertFalse(_valid_dateline(self._mega_payload(24, 12000)))

    def test_cell_none_with_empty_detail(self):
        def mutate(p):
            self._record(p)["cells"]["anchor_iso"] = None

        self._break(mutate)

    def test_reference_group_mismatch_one_candidate(self):
        self._break(lambda p: self._record(p).__setitem__("group", "g-other"))

    def test_fixture_id_order_mismatch(self):
        def mutate(p):
            records = p["candidates"]["dt-b"]["records"]
            records.reverse()

        self._break(mutate)

    def test_primary_decision_wrong_type(self):
        self._break(lambda p: p.__setitem__("primary_decision", 5))

    def test_n_candidates_tampered(self):
        self._break(lambda p: p.__setitem__("n_candidates", p["n_candidates"] + 1))

    def test_n_search_fixtures_tampered(self):
        self._break(lambda p: p.__setitem__("n_search_fixtures", p["n_search_fixtures"] + 1))

    def test_validator_lie_self_check(self):
        # The validator's own lie diff differs from the honest
        # rendering by construction; patch its table to the honest
        # string and the self-check must reject the payload even
        # though the payload's table agrees with the patched one.
        payload = copy.deepcopy(self.payload)
        table = _diffreports_module._dateline_arm_table()
        table["lie"] = "Wednesday, 2026-10-21"
        payload["protocol"]["arm_dates"]["lie"] = "Wednesday, 2026-10-21"
        with mock.patch.object(_diffreports_module, "_dateline_arm_table", return_value=table):
            self.assertFalse(_valid_dateline(payload))

    def test_all_candidates_unavailable_is_invalid(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        entries = _entries(backend)
        for entry in entries:
            entry["summary"]["aggregate_eligible"] = False
        payload = _payload(backend, _run_stage(backend, entries=entries))
        self.assertFalse(_valid_dateline(payload))


class TestReportingEdges(unittest.TestCase):
    def test_candidates_not_a_dict(self):
        html = render_html({"dateline": {"available": True, "candidates": []}})
        self.assertEqual(html.count('id="dateline"'), 0)

    def test_hostile_candidate_entries(self):
        html = render_html(
            {
                "dateline": {
                    "available": True,
                    "candidates": {"x": "junk", "y": {"status": "unavailable"}},
                }
            }
        )
        self.assertIn("unavailable", html)

    def test_falsified_banner_renders(self):
        html = render_html({"decision": "selected", "dateline": _base_payload()})
        self.assertIn("Date content falsified", html)

    def test_hostile_loss_and_pairs(self):
        payload = _base_payload()
        payload["candidates"]["dt-a"]["date_loss"]["anchor"] = "junk"
        payload["candidates"]["dt-a"]["pairs_by_format"]["iso"]["anchor|minus_day"] = "junk"
        payload["candidates"]["dt-a"]["pairs_by_format"]["long"] = "junk"
        html = render_html({"decision": "selected", "dateline": payload})
        self.assertIn('id="dateline"', html)

    def test_non_numeric_rate_renders_na(self):
        payload = _base_payload()
        payload["candidates"]["dt-a"]["date_loss"]["anchor"]["loss_rate"] = "x"
        html = render_html({"decision": "selected", "dateline": payload})
        self.assertIn("n/a", html)


class TestDiffEdges(unittest.TestCase):
    def test_pair_drift_reported(self):
        scripted = _DateBackend(
            fail_on={
                ("marka", "20 October 2026"),
                ("markb", "20 October 2026"),
                ("markc", "22 October 2026"),
                ("markd", "22 October 2026"),
            },
            expected_by_marker=SEVEN_MARKERS,
        )
        stormy = _payload(scripted, _run_stage(scripted, cases=SEVEN_CASES))
        calm_backend = _DateBackend(expected_by_marker=SEVEN_MARKERS)
        calm = _payload(calm_backend, _run_stage(calm_backend, cases=SEVEN_CASES))
        diff = diff_dateline(calm, stormy)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["pair_drift"])

    def test_tampered_side_is_one_sided(self):
        payload = _base_payload()
        tampered = copy.deepcopy(payload)
        tampered["candidates"]["dt-a"]["date_loss"]["leap"]["n_loss"] += 1
        diff = diff_dateline(payload, tampered)
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])

    def test_status_drift_reported(self):
        backend = _DateBackend(expected_by_marker=EXPECTED_BY_MARKER)
        entries = _entries(backend)
        entries[1]["summary"]["aggregate_eligible"] = False
        mixed = _payload(backend, _run_stage(backend, entries=entries))
        calm = _base_payload()
        diff = diff_dateline(calm, mixed)
        self.assertTrue(diff["changed"])
        self.assertTrue(any(entry["metric"] == "status" for entry in diff["loss_drift"]))


class TestMutationGates(unittest.TestCase):
    """Tempdir mutation gates on dateline.py's exactness-critical
    lines (cycle-169 pattern, package-namespace exec per the
    cycle-187 lesson: the scenario passes on the real module and
    must raise AssertionError on the mutant)."""

    def _run_gate(self, name, old_text, new_text, scenario):
        import importlib.util
        import os
        import tempfile

        mod = dateline
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c188_mut_") as tmpdir:
            mod_path = os.path.join(tmpdir, f"dateline_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            dotted = f"model_protocol_lab.dateline_mut_{name}"
            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            sys.modules[dotted] = mutant
            try:
                spec.loader.exec_module(mutant)
            finally:
                sys.modules.pop(dotted, None)
            scenario(mod)  # passes on the real module: non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    def _lie_scenario(self, mod):
        assert mod.lie_date_string() == "Thursday, 2026-10-21"
        assert mod.lie_date_string() != mod.format_date("anchor", "weekday_iso")

    def test_mutant_lie_shift_dropped_caught(self):
        self._run_gate(
            "lie_shift",
            "(_calendar_weekday(year, month, day) + LIE_WEEKDAY_SHIFT) % 7",
            "(_calendar_weekday(year, month, day) + 0) % 7",
            self._lie_scenario,
        )

    def _floor_scenario(self, mod):
        assert mod.classify_pair(3, 2, 0) == "insufficient"
        assert mod.classify_pair(6, 2, 0) == "discordant_asymmetric"

    def test_mutant_floor_dropped_caught(self):
        self._run_gate(
            "floor",
            "if n_paired < MIN_PAIRED_EVALUABLE:",
            "if n_paired < 0:  # MUTANT",
            self._floor_scenario,
        )

    def _screen_scenario(self, mod):
        assert mod.screen_expected("21") is True  # component leg only
        assert mod.screen_expected("26") is False

    def test_mutant_screen_component_leg_dropped_caught(self):
        self._run_gate(
            "screen_components",
            "if any(rung_match(component, arm_component, 4) for arm_component in components):",
            "if False:  # MUTANT",
            self._screen_scenario,
        )

    def _loss_scenario(self, mod):
        def cell(verdict):
            return {"verdict": verdict}

        record = {
            "fixture": "fx",
            "vacuous": False,
            "expected_blank": False,
            "screened": False,
            "message_date_bearing": False,
            "cells": {
                **{f"nodate_{fmt}": cell(True) for fmt in mod.FORMATS},
                **{f"anchor_{fmt}": cell(False) for fmt in mod.FORMATS},
                "lie": cell(True),
            },
        }
        block = mod._loss_block([record], "anchor")
        assert block["n_loss"] == 1
        assert block["n_counterpart"] == 0

    def test_mutant_loss_direction_swapped_caught(self):
        self._run_gate(
            "loss_direction",
            '            if control["verdict"] and not cell["verdict"]:\n'
            "                n_loss += 1",
            '            if cell["verdict"] and not control["verdict"]:  # MUTANT\n'
            "                n_loss += 1",
            self._loss_scenario,
        )


if __name__ == "__main__":
    unittest.main()
