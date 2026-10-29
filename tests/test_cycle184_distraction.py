"""Cycle 184: distractor dilution x template (run --distract).

Pins the amended cross-checked design (red-team A D1-D10 upheld
by B; B's NB1-NB12 amendments binding): repetition-mirror stage
with native reuse, ASCII digit-surgery plants (never Decimal),
a frozen near-miss template, capture as an upper bound beside a
neutral-control excess, the first-flip state machine, per-span
screening, per-condition selection withholding, shape
pre-checks in the diff validator, and the 10-condition scope
cut (no near-miss at 1024). Design: docs/DISTRACTION_DESIGN.md.
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from hashlib import sha256
from pathlib import Path
from threading import Event
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_protocol_lab import distraction
from model_protocol_lab.core import Fixture as Case
from model_protocol_lab.core import Format, Generation, Message
from model_protocol_lab.diffreports import _valid_distraction, diff_distraction, diff_reports
from model_protocol_lab.experiments import distraction_analysis, distraction_stage
from model_protocol_lab.reporting import render_html

BOUNDARY_WORD = "endmark"
TEMPLATE_WORDS = ("ledger", "timetable", "manifest", "kiln")
CORPUS_WORDS = ("harbour", "gardener", "ferries", "cobbles")


class _WordBackend:
    """Word-token backend with answers scripted per cell.

    ``rules`` maps (marker, kind, placement) to the answer the
    model gives for that cell; anything else answers the
    fixture's expected string (``expected_by_marker``). Filler
    detection is by vocabulary disjointness: template words
    mark near-miss spans, corpus words mark neutral spans, and
    placement is which side of the marker the filler sits on.
    Generations pad to the cap with the boundary id (the
    Backend contract: reason "limit" means exactly the cap).
    """

    fingerprint = "sha256:word"

    def __init__(self, rules=None, expected_by_marker=None, fail_markers=()):
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}
        self.rules = dict(rules or {})
        self.expected_by_marker = dict(expected_by_marker or {})
        self.fail_markers = set(fail_markers)
        self.generate_calls = 0
        self.encode_calls = 0
        self.set_cancel_on: str | None = None
        self.cancel_event: Event | None = None

    def encode(self, text):
        """Encode."""
        self.encode_calls += 1
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
        """Decode."""
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def _cell_of(self, text):
        """Cell of."""
        marker = None
        for candidate in self.expected_by_marker:
            if candidate in text:
                marker = candidate
                break
        if marker is None:
            return None, None, None
        kind = None
        filler_at = None
        for word in TEMPLATE_WORDS:
            if word in text:
                kind = "near-miss"
                filler_at = text.index(word)
                break
        if kind is None:
            for word in CORPUS_WORDS:
                if word in text:
                    kind = "neutral"
                    filler_at = text.index(word)
                    break
        if kind is None:
            return marker, None, None
        placement = "before" if filler_at < text.index(marker) else "after"
        return marker, kind, placement

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        self.generate_calls += 1
        text = self.decode(list(input_ids))
        marker, kind, placement = self._cell_of(text)
        if marker == self.set_cancel_on and self.cancel_event is not None:
            self.cancel_event.set()
        if marker in self.fail_markers:
            return Generation(token_ids=(), reason="error", error="scripted")
        # Dose band from the prompt length: native prompts here
        # are ~30 tokens, so +64 lands under 150, +256 under 600.
        total = len(input_ids)
        band = 64 if total < 150 else 256 if total < 600 else 1024
        answer = self.rules.get(
            (marker, kind, placement, band),
            self.rules.get(
                (marker, kind, placement, None),
                self.expected_by_marker.get(marker, "dunno"),
            ),
        )
        boundary = self.encode(BOUNDARY_WORD)[0]
        out = list(self.encode(answer)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


class _CharBackend:
    """Character backend; optionally merges one digraph (seam)."""

    fingerprint = "sha256:char"

    def __init__(self, rules=None, expected_by_marker=None, merge_digraph=None):
        self.rules = dict(rules or {})
        self.expected_by_marker = dict(expected_by_marker or {})
        self.merge = merge_digraph
        self.generate_calls = 0

    def encode(self, text):
        """Encode."""
        if self.merge:
            merged = 0x110000 - 1
            out = []
            i = 0
            while i < len(text):
                if text[i : i + 2] == self.merge:
                    out.append(merged)
                    i += 2
                else:
                    out.append(ord(text[i]))
                    i += 1
            return tuple(out)
        return tuple(ord(c) for c in text)

    def decode(self, ids):
        """Decode."""
        chars = []
        for i in ids:
            if self.merge and int(i) == 0x110000 - 1:
                chars.append(self.merge)
            else:
                chars.append(chr(int(i)))
        return "".join(chars)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        self.generate_calls += 1
        text = self.decode(list(input_ids))
        marker = None
        for candidate in self.expected_by_marker:
            if candidate in text:
                marker = candidate
                break
        answer = self.expected_by_marker.get(marker, "dunno")
        if marker is not None:
            kind = None
            filler_at = None
            for word in TEMPLATE_WORDS:
                if word in text:
                    kind = "near-miss"
                    filler_at = text.index(word)
                    break
            if kind is None:
                for word in CORPUS_WORDS:
                    if word in text:
                        kind = "neutral"
                        filler_at = text.index(word)
                        break
            if kind is not None:
                placement = "before" if filler_at < text.index(marker) else "after"
                answer = self.rules.get((marker, kind, placement), answer)
        boundary = self.encode("~")[0]
        out = list(self.encode(answer)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


FMT_A = Format(
    name="dst-a",
    bos="<|im_start|>system\nYou are helpful.<|im_end|>\n",
    user_prefix="<|im_start|>user\n",
    user_turn_end="<|im_end|>\n",
    assistant_prefix="<|im_start|>assistant\n",
    turn_end="<|im_end|>",
)
FMT_B = Format(
    name="dst-b",
    bos="",
    user_prefix="USER: ",
    assistant_prefix="ASSISTANT: ",
    turn_end="~",
)
FORMATS = (FMT_A, FMT_B)

CASE_NUM = Case(
    id="fx-num",
    group="g1",
    messages=(Message(role="user", content="marknum what is six plus one"),),
    expected="7",
)
CASE_COPY = Case(
    id="fx-copy",
    group="g2",
    messages=(Message(role="user", content="markcopy reply with exactly BLUE"),),
    expected="BLUE",
)
CASE_VAC = Case(
    id="fx-vac",
    group="g3",
    messages=(Message(role="user", content="   "),),
    expected="x",
)
CASES = (CASE_NUM, CASE_COPY, CASE_VAC)
EXPECTED_BY_MARKER = {"marknum": "7", "markcopy": "BLUE", "markvac": "x"}


def _obs(backend, fmt, case, answer, ok=True):
    """Obs."""
    return {
        "candidate": fmt.name,
        "fixture": case.id,
        "input_ids": list(backend.encode(fmt.render(case.messages))),
        "output_ids": [],
        "answer": answer,
        "exact_match": ok,
        "boundary_found": True,
        "boundary_text_match": True,
        "raw_reason": "limit",
        "error": "",
        "status": "completed",
        "input_text": "",
    }


def _entries(backend, formats=FORMATS, cases=CASES, answers=None):
    """Entries."""
    answers = answers or {}
    entries = []
    for fmt in formats:
        entries.append(
            {
                "format": {"name": fmt.name},
                "summary": {"aggregate_eligible": True},
                "observations": [
                    _obs(
                        backend,
                        fmt,
                        case,
                        answers.get((fmt.name, case.id), case.expected),
                        ok=answers.get((fmt.name, case.id), case.expected) == case.expected,
                    )
                    for case in cases
                ],
            }
        )
    return entries


def _run_stage(backend, formats=FORMATS, cases=CASES, entries=None, **kwargs):
    """Run stage."""
    kwargs.setdefault("boundary", tuple(backend.encode(BOUNDARY_WORD)))
    kwargs.setdefault("max_new_tokens", 8)
    kwargs.setdefault("max_context", 8192)
    return distraction_stage(
        backend,
        formats,
        cases,
        entries if entries is not None else _entries(backend, formats, cases),
        **kwargs,
    )


def _payload(backend, stage, decision="compatible_on_fixture_set", winner="dst-a"):
    """Payload."""
    return distraction_analysis(stage, decision, winner, tuple(fmt.name for fmt in FORMATS))


class TestProtocolAndPlant(unittest.TestCase):
    """Test protocol and plant."""

    def test_conditions_and_budgets(self):
        """Test conditions and budgets."""
        self.assertEqual(len(distraction.CONDITIONS), 10)
        self.assertEqual(distraction.GENERATION_BUDGET, 24000)
        self.assertEqual(distraction.MAX_GROUND_RECORDS, 2400)
        self.assertEqual(distraction.SIZING_ENCODE_BUDGET, 512)
        kinds = {(k, lv) for k, lv, _p in distraction.CONDITIONS}
        self.assertNotIn(("near-miss", 1024), kinds)
        self.assertIn(("neutral", 1024), kinds)

    def test_frozen_texts_pinned(self):
        """Test frozen texts pinned."""
        self.assertEqual(
            distraction.NEUTRAL_CORPUS_SHA256,
            "5969b0123fb992cb4457d75e128c58afffa219832e689b09c08a4ea271bb2159",
        )
        self.assertEqual(
            distraction.NEARMISS_TEMPLATES_SHA256,
            "c5e1531b671a7e88a2f0e6902874c8b2683a709671b0d946aa5530e3f80ade7b",
        )
        self.assertEqual(len(distraction.NEUTRAL_CORPUS), 71)
        self.assertEqual(len(set(distraction.NEUTRAL_CORPUS)), len(distraction.NEUTRAL_CORPUS))
        for template in distraction.NEARMISS_TEMPLATES:
            self.assertEqual(template.count("{planted}"), 1)

    def test_fingerprint_pinned(self):
        """Test fingerprint pinned."""
        self.assertEqual(
            distraction.protocol_fingerprint(["f1"], "sha256:scripted"),
            "55ee4395a37ef9bbc2a1b838cce9bfbc1f2b944e69180c430d832d9c09233c4d",
        )

    def test_plant_battery_exact(self):
        """Test plant battery exact."""
        battery = {
            "7": "8",
            "9": "10",
            "0": "1",
            "-0": "-1",
            "007": "008",
            "42.0": "42.1",
            "-42.0": "-42.1",
            "99.9": "100.0",
            "1,000": "1,001",
            "9,999": "10,000",
            "1e3": "2e3",
            "1E3": "2E3",
            ".5": ".6",
            "3.14159": "3.14160",
            "+42": "+43",
            "1,000,000": "1,000,001",
            "9" * 40: "1" + "0" * 40,
        }
        for expected, planted in battery.items():
            with self.subTest(expected=expected):
                self.assertEqual(distraction.planted_answer(expected), planted)

    def test_plant_undefined_shapes(self):
        """Test plant undefined shapes."""
        for expected in (
            "",
            " ",
            " 42 ",
            "42 ",
            "1_000",
            "٤٢",
            "inf",
            "NaN",
            "2026-10-20",
            "BLUE",
            '{"a": 1}',
            "The cat sat.",
            "café",
            "10,00",
            "1,0000",
            ".5.5",
            "1e",
            "e3",
            "--42",
            "4 2",
        ):
            with self.subTest(expected=expected):
                self.assertIsNone(distraction.planted_answer(expected))

    def test_plant_deterministic_and_never_identity(self):
        """Test plant deterministic and never identity."""
        import decimal

        samples = ["7", "42.0", "9" * 40, "1,000", ".5", "-42.0"]
        with decimal.localcontext() as ctx:
            ctx.prec = 6
            first = [distraction.planted_answer(s) for s in samples]
        with decimal.localcontext() as ctx:
            ctx.prec = 50
            second = [distraction.planted_answer(s) for s in samples]
        self.assertEqual(first, second)
        for sample in samples:
            self.assertNotEqual(distraction.planted_answer(sample), sample)


class TestPureHelpers(unittest.TestCase):
    """Test pure helpers."""

    def test_span_text_prefix_property(self):
        """Test span text prefix property."""
        full = "".join(distraction.NEUTRAL_CORPUS)
        for whole, suffix in ((0, 0), (0, 5), (3, 0), (3, 17), (70, 4)):
            span = distraction.span_text(distraction.NEUTRAL_CORPUS, whole, suffix)
            self.assertTrue(full.startswith(span))
        with self.assertRaises(ValueError):
            distraction.span_text(distraction.NEUTRAL_CORPUS, 72, 0)
        with self.assertRaises(ValueError):
            distraction.span_text(distraction.NEUTRAL_CORPUS, 0, 10_000)

    def test_span_candidates_deterministic_and_ordered(self):
        """Test span candidates deterministic and ordered."""
        first = distraction.span_candidates(
            distraction.NEUTRAL_CORPUS, target=64, chars_per_token=4.0
        )
        second = distraction.span_candidates(
            distraction.NEUTRAL_CORPUS, target=64, chars_per_token=4.0
        )
        self.assertEqual(first, second)
        estimates = [abs(chars / 4.0 - 64) for _w, _s, chars in first[:50]]
        self.assertEqual(estimates, sorted(estimates))

    def test_suffix_cuts_plant(self):
        """Test suffix cuts plant."""
        self.assertTrue(distraction.suffix_cuts_plant("...listed 4", "2 sacks", "42"))
        self.assertFalse(distraction.suffix_cuts_plant("...listed 42", " sacks", "42"))
        self.assertFalse(distraction.suffix_cuts_plant("...listed ", "42 sacks", "42"))
        self.assertFalse(distraction.suffix_cuts_plant("anything", "", "42"))

    def test_span_screened(self):
        """Test span screened."""
        self.assertTrue(distraction.span_screened("the rain fell", "rain", []))
        self.assertTrue(distraction.span_screened("anything", "x", ["<END>"]) is False)
        self.assertTrue(distraction.span_screened("has <END> in it", "x", ["<END>"]))
        self.assertFalse(distraction.span_screened("plain prose here", "zebra", ["<END>"]))
        self.assertFalse(distraction.span_screened("plain prose", "  ", []))

    def test_build_cell_fixture(self):
        """Test build cell fixture."""
        before = distraction.build_cell_fixture(CASE_NUM, "FILLER ", "before")
        after = distraction.build_cell_fixture(CASE_NUM, "FILLER ", "after")
        self.assertEqual(before.messages[-1].content, "FILLER \nmarknum what is six plus one")
        self.assertEqual(after.messages[-1].content, "marknum what is six plus one\nFILLER ")
        self.assertEqual(before.id, CASE_NUM.id)
        self.assertEqual(before.expected, CASE_NUM.expected)
        with self.assertRaises(ValueError):
            distraction.build_cell_fixture(CASE_NUM, "x", "sideways")

    def test_nearmiss_sentences(self):
        """Test nearmiss sentences."""
        sentences = distraction.nearmiss_sentences("8")
        self.assertEqual(len(sentences), distraction.NEARMISS_SEQUENCE_LENGTH)
        self.assertIn("recorded 8 crates", sentences[0])
        self.assertEqual(distraction.planted_occurrences(sentences[0], "8"), 1)
        self.assertEqual(distraction.planted_occurrences("8 and 8", "8"), 2)
        self.assertEqual(distraction.planted_occurrences("none", ""), 0)


# Capture script: marknum is captured under near-miss-before at
# both doses, answers correctly elsewhere; markcopy fails under
# neutral-256-before only (a first-flip at 256 on that curve).
RULES = {
    ("marknum", "near-miss", "before", None): "8",
    ("markcopy", "neutral", "before", 256): "dunno",
}


class TestStageWordBackend(unittest.TestCase):
    """Test stage word backend."""

    def setUp(self):
        """Set up the test fixture."""
        self.backend = _WordBackend(RULES, EXPECTED_BY_MARKER)
        self.stage = _run_stage(self.backend)
        self.assertTrue(self.stage["available"], self.stage["detail"])

    def test_zero_native_generations(self):
        """Test zero native generations."""
        # Per candidate: fx-num has 8 evaluable cells (1024 is
        # unmatchable under a word tokenizer — the corpus is 893
        # words), fx-copy has 4 (its near-miss cells are
        # undefined), fx-vac none. No native generation at all:
        # the recorded search is reused.
        self.assertEqual(self.backend.generate_calls, 2 * (8 + 4))

    def test_native_cells_reused_bit_for_bit(self):
        """Test native cells reused bit for bit."""
        for fmt in FORMATS:
            record = self.stage["candidates"][fmt.name]["records"][0]
            native = record["native"]
            self.assertTrue(native["verdict"])
            self.assertEqual(native["answer_digest"], sha256(b"7").hexdigest())
            self.assertEqual(
                native["input_tokens"],
                len(self.backend.encode(fmt.render(CASE_NUM.messages))),
            )

    def test_unmatchable_1024_cells(self):
        """Test unmatchable 1024 cells."""
        record = self.stage["candidates"]["dst-a"]["records"][0]
        for key in ("neutral_1024_before", "neutral_1024_after"):
            self.assertIsNone(record["cells"][key])
            self.assertEqual(record["cell_details"][key], "arm_unmatchable")

    def test_capture_only_before(self):
        """Test capture only before."""
        payload = _payload(self.backend, self.stage)
        entry = payload["candidates"]["dst-a"]
        self.assertEqual(entry["capture"]["before_64"]["capture_rate"], 1.0)
        self.assertEqual(entry["capture"]["before_64"]["n_captured"], 1)
        self.assertEqual(entry["capture"]["before_64"]["capture_excess_over_neutral"], 1.0)
        self.assertEqual(entry["capture"]["after_64"]["capture_rate"], 0.0)
        self.assertEqual(entry["capture"]["before_256"]["capture_rate"], 1.0)

    def test_first_flip_from_stage(self):
        """Test first flip from stage."""
        payload = _payload(self.backend, self.stage)
        entry = payload["candidates"]["dst-a"]
        flip = entry["first_flip"]["neutral_before"]
        # markcopy passes at 64, fails at 256 (scripted), and the
        # 1024 cell is unmatchable, so the curve population for
        # first-flip is empty (a curve cell is missing).
        self.assertEqual(flip["n_population"], 0)
        curve = entry["curves"]["neutral_before"]
        self.assertEqual(curve["64"]["n_paired"], 2)
        self.assertEqual(curve["64"]["rate"], 1.0)
        self.assertEqual(curve["256"]["n_pass"], 1)

    def test_vacuous_and_echo_records(self):
        """Test vacuous and echo records."""
        records = self.stage["candidates"]["dst-a"]["records"]
        vac = records[2]
        self.assertTrue(vac["vacuous"])
        self.assertTrue(all(cell is None for cell in vac["cells"].values()))
        self.assertTrue(all(detail == "vacuous" for detail in vac["cell_details"].values()))
        self.assertTrue(records[1]["echo_fixture"])
        self.assertFalse(records[0]["echo_fixture"])

    def test_near_miss_defined_flags(self):
        """Test near miss defined flags."""
        records = self.stage["candidates"]["dst-a"]["records"]
        self.assertTrue(records[0]["near_miss_defined"])
        self.assertEqual(records[0]["planted_digest"], sha256(b"8").hexdigest())
        self.assertFalse(records[1]["near_miss_defined"])
        self.assertIsNone(records[1]["planted_digest"])
        self.assertEqual(records[1]["cell_details"]["near-miss_64_before"], "near_miss_undefined")

    def test_conditions_withheld_and_flips(self):
        """Test conditions withheld and flips."""
        payload = _payload(self.backend, self.stage)
        conditions = payload["conditions"]
        # 1024 cells complete nowhere: completed sets are empty,
        # so those conditions are withheld per condition (G9).
        self.assertEqual(conditions["neutral_1024_before"]["decision"], "unavailable")
        self.assertIsNone(conditions["neutral_1024_before"]["flip"])
        # near-miss before: dst-a and dst-b both fail fx-num (the
        # only completed fixture), so no candidate passes.
        self.assertEqual(conditions["near-miss_64_before"]["decision"], "no_candidate")
        self.assertIs(conditions["near-miss_64_before"]["flip"], True)
        # The scripted backend is format-blind, so after-placement
        # passes for both candidates: ambiguous, and a flip
        # against the mapped primary ("selected", "dst-a").
        self.assertEqual(conditions["near-miss_64_after"]["decision"], "ambiguous")
        self.assertIs(conditions["near-miss_64_after"]["flip"], True)

    def test_deterministic_payload(self):
        """Test deterministic payload."""
        backend2 = _WordBackend(RULES, EXPECTED_BY_MARKER)
        stage2 = _run_stage(backend2)
        self.assertEqual(
            json.dumps(self.stage, sort_keys=True),
            json.dumps(stage2, sort_keys=True),
        )

    def test_cell_budget_exceeded(self):
        """Test cell budget exceeded."""
        backend = _WordBackend({}, EXPECTED_BY_MARKER)
        many_cases = tuple(
            Case(
                id=f"fx-{i}",
                group="g",
                messages=(Message(role="user", content=f"marknum q{i}"),),
                expected="7",
            )
            for i in range(101)
        )
        many_formats = tuple(
            Format(
                name=f"dst-{i}",
                bos="",
                user_prefix="U: ",
                assistant_prefix="A: ",
                turn_end="~",
            )
            for i in range(24)
        )
        stage = distraction_stage(
            backend,
            many_formats,
            many_cases,
            [],
            boundary=(1,),
            max_new_tokens=8,
            max_context=8192,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "distraction_cell_budget_exceeded")
        self.assertEqual(backend.generate_calls, 0)

    def test_filler_screen_failed_on_special_token(self):
        """Test filler screen failed on special token."""
        backend = _WordBackend({}, EXPECTED_BY_MARKER)
        hostile = ("A sentence carrying <|im_end|> inside it. ",)
        with mock.patch.object(distraction, "NEUTRAL_CORPUS", hostile):
            stage = _run_stage(backend)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "filler_screen_failed")
        self.assertEqual(backend.generate_calls, 0)

    def test_unavailable_candidate(self):
        """Test unavailable candidate."""
        backend = _WordBackend({}, EXPECTED_BY_MARKER)
        entries = _entries(backend)
        entries[1]["summary"] = {"aggregate_eligible": False}
        stage = _run_stage(backend, entries=entries)
        self.assertEqual(stage["candidates"]["dst-b"]["status"], "unavailable")
        payload = _payload(backend, stage)
        self.assertEqual(payload["candidates"]["dst-b"]["records"], [])
        self.assertEqual(payload["conditions"]["neutral_64_before"]["decision"], "unavailable")

    def test_cancel_mid_stage(self):
        """Test cancel mid stage."""
        backend = _WordBackend({}, EXPECTED_BY_MARKER)
        event = Event()
        backend.set_cancel_on = "markcopy"
        backend.cancel_event = event
        stage = _run_stage(backend, cancel=event)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        self.assertEqual(stage["candidates"], {})

    def test_per_candidate_boundary_map(self):
        """Test per candidate boundary map."""
        backend = _WordBackend({}, EXPECTED_BY_MARKER)
        alt = tuple(backend.encode("othermark"))
        stage = _run_stage(backend, boundary_map={"dst-b": alt})
        self.assertTrue(stage["available"])
        # dst-b's boundary never occurs in the scripted output,
        # so its cells record boundary_found False and fail.
        record_b = stage["candidates"]["dst-b"]["records"][0]
        cell = record_b["cells"]["neutral_64_before"]
        self.assertIsNotNone(cell)
        self.assertFalse(cell["boundary_found"])
        self.assertFalse(cell["verdict"])
        record_a = stage["candidates"]["dst-a"]["records"][0]
        self.assertTrue(record_a["cells"]["neutral_64_before"]["verdict"])

    def test_context_exceeded_per_cell(self):
        """Test context exceeded per cell."""
        backend = _CharBackend({}, EXPECTED_BY_MARKER)
        stage = distraction_stage(
            backend,
            FORMATS,
            CASES,
            _entries(backend),
            boundary=tuple(backend.encode("~")),
            max_new_tokens=8,
            max_context=200,
        )
        record = stage["candidates"]["dst-a"]["records"][0]
        self.assertEqual(record["cell_details"]["neutral_1024_before"], "arm_context_exceeded")
        self.assertIsNotNone(record["cells"]["neutral_64_before"])

    def test_screened_cell_per_span(self):
        """Test screened cell per span."""
        # The expected answer "harbour" appears in the corpus's
        # first sentence: every neutral span contains it, so all
        # neutral cells of that fixture screen; near-miss cells
        # (a different text) are measured.
        case = Case(
            id="fx-screen",
            group="g9",
            messages=(Message(role="user", content="marknum name the port"),),
            expected="harbour",
        )
        backend = _WordBackend({}, {"marknum": "harbour"})
        entries = [
            {
                "format": {"name": fmt.name},
                "summary": {"aggregate_eligible": True},
                "observations": [_obs(backend, fmt, case, "harbour")],
            }
            for fmt in FORMATS
        ]
        stage = distraction_stage(
            backend,
            FORMATS,
            (case,),
            entries,
            boundary=tuple(backend.encode(BOUNDARY_WORD)),
            max_new_tokens=8,
            max_context=8192,
        )
        record = stage["candidates"]["dst-a"]["records"][0]
        self.assertEqual(record["cell_details"]["neutral_64_before"], "arm_screened")
        self.assertEqual(record["cell_details"]["neutral_256_after"], "arm_screened")
        # The near-miss kind is a different text: "harbour" is
        # not numeric, so its cells are undefined, not screened —
        # screening is per span and per kind, never per corpus.
        self.assertEqual(record["cell_details"]["near-miss_64_before"], "near_miss_undefined")


class TestCharAndSeamBackends(unittest.TestCase):
    """Test char and seam backends."""

    def test_id_exact_on_clean_char_backend(self):
        """Test id exact on clean char backend."""
        backend = _CharBackend({}, EXPECTED_BY_MARKER)
        stage = distraction_stage(
            backend,
            FORMATS,
            CASES,
            _entries(backend),
            boundary=tuple(backend.encode("~")),
            max_new_tokens=8,
            max_context=8192,
        )
        record = stage["candidates"]["dst-a"]["records"][0]
        cell = record["cells"]["neutral_64_before"]
        self.assertIsNotNone(cell)
        self.assertEqual(cell["construction"], "id_exact")
        self.assertEqual(cell["seam_ids"], [])
        self.assertIsNotNone(record["cells"]["neutral_1024_before"])

    def test_text_level_on_merge_backend(self):
        """Test text level on merge backend."""
        # " T" spans the prefix/span junction of every FMT_B arm
        # render (user_prefix ends in a space, the corpus opens
        # with "The"), so the standalone prefix cannot survive.
        backend = _CharBackend({}, EXPECTED_BY_MARKER, merge_digraph=" T")
        stage = distraction_stage(
            backend,
            (FMT_B,),
            CASES,
            _entries(backend, formats=(FMT_B,)),
            boundary=tuple(backend.encode("~")),
            max_new_tokens=8,
            max_context=8192,
        )
        record = stage["candidates"]["dst-b"]["records"][0]
        constructions = {cell["construction"] for cell in record["cells"].values() if cell}
        self.assertIn("text_level", constructions)


def _record(fixture, native_verdict, cells, *, vacuous=False, defined=True, planted="8"):
    """Synthetic ground record for aggregate-level tests."""
    digests = {
        "7": sha256(b"7").hexdigest(),
        "8": sha256(b"8").hexdigest(),
        "x": sha256(b"x").hexdigest(),
    }
    built = {}
    for key in distraction.CONDITION_KEYS:
        verdict = cells.get(key)
        if verdict is None:
            built[key] = None
        else:
            built[key] = {
                "verdict": verdict[0],
                "boundary_found": True,
                "boundary_text_match": True,
                "input_tokens": 10,
                "answer_digest": digests[verdict[1]] if len(verdict) > 1 else None,
                "construction": "id_exact",
                "seam_ids": [],
                "planted_occurrences": 1 if key.startswith("near-miss") else None,
            }
    return {
        "fixture": fixture,
        "vacuous": vacuous,
        "echo_fixture": False,
        "near_miss_defined": defined,
        "planted_digest": digests[planted] if defined else None,
        "native": {
            "verdict": native_verdict,
            "boundary_found": True,
            "boundary_text_match": True,
            "input_tokens": 5,
            "answer_digest": digests["7"],
        },
        "cells": built,
        "cell_details": dict.fromkeys(distraction.CONDITION_KEYS, ""),
    }


class TestFirstFlipStateMachine(unittest.TestCase):
    """Test first flip state machine."""

    @staticmethod
    def _flip(cells, native=True):
        """Flip."""
        record = _record("f", native, cells)
        return distraction.aggregate_candidate([record])["first_flip"]["neutral_before"]

    def test_flip_64_return_256(self):
        """Test flip 64 return 256."""
        flip = self._flip(
            {
                "neutral_64_before": (False,),
                "neutral_256_before": (True,),
                "neutral_1024_before": (True,),
            }
        )
        self.assertEqual(flip["flip_counts"], {"64": 1, "256": 0, "1024": 0})
        # Returns count passing levels above the first flip
        # (256 and 1024 both pass here).
        self.assertEqual(flip["n_returns"], 2)
        self.assertEqual(flip["n_population"], 1)

    def test_fail0_gain_not_flip(self):
        """Test fail0 gain not flip."""
        flip = self._flip(
            {
                "neutral_64_before": (True,),
                "neutral_256_before": (False,),
                "neutral_1024_before": (False,),
            },
            native=False,
        )
        self.assertEqual(flip["n_population"], 0)
        self.assertEqual(flip["n_gain_population"], 1)
        self.assertEqual(flip["n_gains"], 1)

    def test_fail_everywhere_no_flip_no_gain(self):
        """Test fail everywhere no flip no gain."""
        flip = self._flip(
            {
                "neutral_64_before": (False,),
                "neutral_256_before": (False,),
                "neutral_1024_before": (False,),
            },
            native=False,
        )
        self.assertEqual(flip["n_gains"], 0)
        self.assertEqual(flip["flip_counts"], {"64": 0, "256": 0, "1024": 0})

    def test_first_flip_256_returns_1(self):
        """Test first flip 256 returns 1."""
        flip = self._flip(
            {
                "neutral_64_before": (True,),
                "neutral_256_before": (False,),
                "neutral_1024_before": (True,),
            }
        )
        self.assertEqual(flip["flip_counts"]["256"], 1)
        self.assertEqual(flip["n_returns"], 1)

    def test_partial_curve_excluded(self):
        """Test partial curve excluded."""
        flip = self._flip({"neutral_64_before": (False,), "neutral_256_before": (True,)})
        self.assertEqual(flip["n_population"], 0)
        self.assertEqual(flip["n_gain_population"], 0)


class TestCaptureRules(unittest.TestCase):
    """Test capture rules."""

    def test_captured_counts(self):
        """Test captured counts."""
        record = _record(
            "f",
            True,
            {
                "near-miss_64_before": (False, "8"),
                "neutral_64_before": (True, "7"),
            },
        )
        capture = distraction.aggregate_candidate([record])["capture"]["before_64"]
        self.assertEqual(capture["n_captured"], 1)
        self.assertEqual(capture["capture_rate"], 1.0)
        self.assertEqual(capture["capture_excess_over_neutral"], 1.0)

    def test_neutral_also_planted_no_excess(self):
        """Test neutral also planted no excess."""
        record = _record(
            "f",
            True,
            {
                "near-miss_64_before": (False, "8"),
                "neutral_64_before": (False, "8"),
            },
        )
        capture = distraction.aggregate_candidate([record])["capture"]["before_64"]
        self.assertEqual(capture["capture_rate"], 1.0)
        self.assertEqual(capture["capture_excess_over_neutral"], 0.0)

    def test_native_fail_not_in_population(self):
        """Test native fail not in population."""
        record = _record(
            "f",
            False,
            {"near-miss_64_before": (False, "8"), "neutral_64_before": (False, "x")},
        )
        capture = distraction.aggregate_candidate([record])["capture"]["before_64"]
        self.assertEqual(capture["n_population"], 0)
        self.assertIsNone(capture["capture_rate"])

    def test_undefined_not_in_population(self):
        """Test undefined not in population."""
        record = _record("f", True, {"near-miss_64_before": (False, "8")}, defined=False)
        capture = distraction.aggregate_candidate([record])["capture"]["before_64"]
        self.assertEqual(capture["n_population"], 0)

    def test_excess_population_needs_neutral_cell(self):
        """Test excess population needs neutral cell."""
        record = _record("f", True, {"near-miss_64_before": (False, "8")})
        capture = distraction.aggregate_candidate([record])["capture"]["before_64"]
        self.assertEqual(capture["n_population"], 1)
        self.assertEqual(capture["n_excess_population"], 0)
        self.assertIsNone(capture["capture_excess_over_neutral"])


class TestAnalysisMapping(unittest.TestCase):
    """Test analysis mapping."""

    def test_held_out_failed_maps_to_selected(self):
        """Test held out failed maps to selected."""
        backend = _WordBackend(RULES, EXPECTED_BY_MARKER)
        stage = _run_stage(backend)
        payload = _payload(backend, stage, decision="held_out_failed", winner="dst-b")
        # near-miss after selects dst-a (both pass there? dst-a and
        # dst-b both pass fx-num after; fx-copy's nm cells are
        # undefined, so both complete {fx-num} and both pass ->
        # ambiguous, a flip vs the mapped primary either way).
        condition = payload["conditions"]["near-miss_64_after"]
        self.assertIn(condition["decision"], ("ambiguous", "selected"))
        self.assertIs(condition["flip"], True)

    def test_verbatim_decision_mapping(self):
        """Test verbatim decision mapping."""
        backend = _WordBackend(RULES, EXPECTED_BY_MARKER)
        stage = _run_stage(backend)
        payload = _payload(backend, stage, decision="ambiguous", winner=None)
        # A verbatim-compared primary is NOT vocabulary-mapped:
        # the condition's ("ambiguous", None) equals it, so no flip.
        condition = payload["conditions"]["neutral_64_before"]
        self.assertEqual(condition["decision"], "ambiguous")
        self.assertIs(condition["flip"], False)

    def test_unavailable_stage_passthrough(self):
        """Test unavailable stage passthrough."""
        payload = distraction_analysis(
            {"available": False, "detail": "cancelled", "protocol": {"x": 1}},
            "compatible_on_fixture_set",
            "dst-a",
            ("dst-a",),
        )
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")


class TestSizingBudget(unittest.TestCase):
    """Test sizing budget."""

    def test_encode_budget_bounded(self):
        """Test encode budget bounded."""

        class _FlatBackend(_WordBackend):
            """Flat backend."""

            # skipcq: PYL-R0201 - zero-arg super() requires an instance method
            def encode(self, text):
                """Encode."""
                super().encode(text)
                return (1, 2, 3)

        backend = _FlatBackend({}, EXPECTED_BY_MARKER)
        stage = distraction_stage(
            backend,
            (FMT_A,),
            (CASE_NUM,),
            _entries(backend, formats=(FMT_A,), cases=(CASE_NUM,)),
            boundary=(1,),
            max_new_tokens=8,
            max_context=8192,
        )
        record = stage["candidates"]["dst-a"]["records"][0]
        self.assertTrue(all(d == "arm_unmatchable" for d in record["cell_details"].values()))
        self.assertLessEqual(backend.encode_calls, 1 + 10 * 512 + 40)
        self.assertEqual(backend.generate_calls, 0)


class TestValidatorAndDiff(unittest.TestCase):
    """Test validator and diff."""

    def setUp(self):
        """Set up the test fixture."""
        self.backend = _WordBackend(RULES, EXPECTED_BY_MARKER)
        self.payload = _payload(self.backend, _run_stage(self.backend))

    def test_real_payload_validates(self):
        """Test real payload validates."""
        self.assertTrue(_valid_distraction(self.payload))

    def test_identical_diff_unchanged(self):
        """Test identical diff unchanged."""
        result = diff_distraction(self.payload, copy.deepcopy(self.payload))
        self.assertEqual(result["available"], "both")
        self.assertFalse(result["changed"])
        self.assertTrue(result["comparable"])

    def test_tamper_battery(self):
        """Test tamper battery."""

        def flip_rate(payload):
            """Flip rate."""
            payload["candidates"]["dst-a"]["curves"]["neutral_before"]["64"]["rate"] = 0.5

        def flip_capture(payload):
            """Flip capture."""
            payload["candidates"]["dst-a"]["capture"]["before_64"]["n_captured"] = 0

        def flip_cell_verdict(payload):
            """Flip cell verdict."""
            record = payload["candidates"]["dst-a"]["records"][0]
            record["cells"]["neutral_64_before"]["verdict"] = False

        def break_delta(payload):
            """Break delta."""
            record = payload["candidates"]["dst-a"]["records"][0]
            record["cells"]["neutral_64_before"]["input_tokens"] += 1

        def break_digest(payload):
            """Break digest."""
            record = payload["candidates"]["dst-a"]["records"][0]
            record["planted_digest"] = "zz" * 32

        def break_fingerprint(payload):
            """Break fingerprint."""
            payload["protocol"]["fingerprint"] = "0" * 64

        def break_condition(payload):
            """Break condition."""
            payload["conditions"]["neutral_64_before"]["decision"] = "selected"

        def break_share(payload):
            """Break share."""
            payload["candidates"]["dst-a"]["nearmiss_defined_share"] = 0.25

        def seam_too_long(payload):
            """Seam too long."""
            record = payload["candidates"]["dst-a"]["records"][0]
            record["cells"]["neutral_64_before"]["seam_ids"] = [1, 2, 3]

        def wrong_detail(payload):
            """Wrong detail."""
            record = payload["candidates"]["dst-a"]["records"][0]
            record["cell_details"]["neutral_64_before"] = "bogus_detail"

        def cell_without_detail(payload):
            """Cell without detail."""
            record = payload["candidates"]["dst-a"]["records"][0]
            record["cells"]["neutral_64_before"] = None

        for mutate in (
            flip_rate,
            flip_capture,
            flip_cell_verdict,
            break_delta,
            break_digest,
            break_fingerprint,
            break_condition,
            break_share,
            seam_too_long,
            wrong_detail,
            cell_without_detail,
        ):
            with self.subTest(mutate=mutate.__name__):
                hostile = copy.deepcopy(self.payload)
                mutate(hostile)
                self.assertFalse(_valid_distraction(hostile))

    def test_shape_prechecks(self):
        """Test shape prechecks."""
        # 101 records for one candidate: rejected on shape.
        hostile = copy.deepcopy(self.payload)
        record = hostile["candidates"]["dst-a"]["records"][0]
        hostile["candidates"]["dst-a"]["records"] = [
            dict(record, fixture=f"fx-{i}") for i in range(101)
        ]
        self.assertFalse(_valid_distraction(hostile))
        # A records section over the byte bound: rejected.
        hostile = copy.deepcopy(self.payload)
        record = hostile["candidates"]["dst-a"]["records"][0]
        hostile["candidates"]["dst-a"]["records"] = [
            dict(record, fixture="fx-" + "x" * 200_000 + str(i)) for i in range(70)
        ]
        hostile["candidates"]["dst-b"]["records"] = []
        self.assertFalse(_valid_distraction(hostile))

    def test_coherent_foreign_protocol(self):
        """Test coherent foreign protocol."""
        foreign = copy.deepcopy(self.payload)
        foreign["protocol"]["version"] = 99
        # The fingerprint is rebuilt over the payload's own
        # components, exactly as the validator rebuilds it.
        import hashlib as _h

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import json as _j

        rebuilt = {k: v for k, v in foreign["protocol"].items() if k != "fingerprint"}
        canonical = _j.dumps(rebuilt, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        foreign["protocol"]["fingerprint"] = _h.sha256(canonical.encode()).hexdigest()
        self.assertTrue(_valid_distraction(foreign))
        result = diff_distraction(self.payload, foreign)
        self.assertTrue(result["protocol_drift"])
        self.assertFalse(result["comparable"])
        self.assertTrue(result["changed"])

    def test_availability_shapes(self):
        """Test availability shapes."""
        self.assertEqual(diff_distraction(None, self.payload)["available"], "new")
        self.assertEqual(diff_distraction(self.payload, None)["available"], "old")
        self.assertEqual(diff_distraction(None, None)["available"], "neither")
        self.assertFalse(diff_distraction({"available": True}, self.payload)["changed"])

    def test_curve_and_condition_drift(self):
        """Test curve and condition drift."""
        backend2 = _WordBackend({}, EXPECTED_BY_MARKER)
        other = _payload(backend2, _run_stage(backend2))
        result = diff_distraction(self.payload, other)
        self.assertTrue(result["changed"])
        metrics = {entry["metric"] for entry in result["curve_drift"]}
        self.assertIn("capture_before_64", metrics)
        self.assertTrue(result["condition_drift"])

    def test_full_diff_reports_wiring(self):
        """Test full diff reports wiring."""
        base = {
            "decision": "compatible_on_fixture_set",
            "selected_candidate": "dst-a",
            "distraction": self.payload,
        }
        same = diff_reports(base, copy.deepcopy(base))
        self.assertIn("distraction", same)
        self.assertIn("distraction_changed", same["summary"])
        self.assertFalse(same["summary"]["distraction_changed"])
        self.assertTrue(same["summary"]["identical"])


class TestReporting(unittest.TestCase):
    """Test reporting."""

    def setUp(self):
        """Set up the test fixture."""
        self.backend = _WordBackend(RULES, EXPECTED_BY_MARKER)
        self.payload = _payload(self.backend, _run_stage(self.backend))

    def test_section_renders_with_labels(self):
        """Test section renders with labels."""
        html = render_html({"distraction": self.payload})
        self.assertIn('id="distraction"', html)
        self.assertIn("upper bound", html)
        self.assertIn("terminate at 256", html)
        self.assertIn("Selection withheld", html)
        self.assertIn("neutral_1024_before", html)

    def test_absent_and_unavailable_render_nothing(self):
        """Test absent and unavailable render nothing."""
        self.assertNotIn('id="distraction"', render_html({}))
        self.assertNotIn(
            'id="distraction"',
            render_html({"distraction": {"available": False, "detail": "x"}}),
        )

    def test_hostile_candidate_name_escaped(self):
        """Test hostile candidate name escaped."""
        hostile = copy.deepcopy(self.payload)
        hostile["candidates"]["<script>alert(1)</script>"] = hostile["candidates"].pop("dst-a")
        html = render_html({"distraction": hostile})
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_malformed_payload_survives(self):
        """Test malformed payload survives."""
        html = render_html(
            {
                "distraction": {
                    "available": True,
                    "candidates": {
                        "x": {"status": "available", "curves": {"neutral_before": {"64": "junk"}}}
                    },
                }
            }
        )
        self.assertIn('id="distraction"', html)


class TestCliDispatch(unittest.TestCase):
    """Test cli dispatch."""

    def test_cancel_identity_reaches_stage(self):
        """Test cancel identity reaches stage."""
        from model_protocol_lab import cli as cli_module

        seen: dict[str, object] = {}

        import model_protocol_lab.experiments as experiments_module

        original_stage = experiments_module.distraction_stage

        def spy_stage(backend, formats, cases, entries, **kwargs):
            """Spy stage."""
            seen["cancel"] = kwargs.get("cancel")
            return original_stage(backend, formats, cases, entries, **kwargs)

        backend = _WordBackend(RULES, EXPECTED_BY_MARKER)
        formats = FORMATS
        result = {
            "decision": "compatible_on_fixture_set",
            "selected_candidate": "dst-a",
            "search": _entries(backend),
        }
        event = Event()
        with mock.patch.object(experiments_module, "distraction_stage", spy_stage):
            cli_module._run_distract_probe(
                backend,
                formats,
                result,
                CASES,
                boundary=tuple(backend.encode(BOUNDARY_WORD)),
                boundary_map=None,
                max_new_tokens=8,
                max_context=8192,
                seed=0,
                cancel=event,
            )
        self.assertIs(seen["cancel"], event)
        self.assertTrue(result["distraction"]["available"])

    def test_failed_decision_gates_to_zero_generations(self):
        """Test failed decision gates to zero generations."""
        from model_protocol_lab import cli as cli_module

        backend = _WordBackend({}, EXPECTED_BY_MARKER)
        result = {"decision": "failed", "search": []}
        cli_module._run_distract_probe(
            backend,
            FORMATS,
            result,
            CASES,
            boundary=(1,),
            boundary_map=None,
            max_new_tokens=8,
            max_context=8192,
            seed=0,
            cancel=None,
        )
        self.assertEqual(
            result["distraction"],
            {"available": False, "detail": "native_search_incomplete"},
        )
        self.assertEqual(backend.generate_calls, 0)


class TestStageGapCoverage(unittest.TestCase):
    """Test stage gap coverage."""

    def test_contract_value_errors(self):
        """Test contract value errors."""
        backend = _WordBackend({}, EXPECTED_BY_MARKER)
        with self.assertRaises(ValueError):
            distraction_stage(backend, (), CASES, [], boundary=(1,), max_context=100)
        with self.assertRaises(ValueError):
            distraction_stage(backend, FORMATS, (), [], boundary=(1,), max_context=100)

    def test_junk_search_entries(self):
        """Test junk search entries."""
        backend = _WordBackend({}, EXPECTED_BY_MARKER)
        entries = _entries(backend)
        entries.append("junk")
        entries.append({"format": "not-a-dict"})
        entries.append({"format": {"name": 42}})
        stage = _run_stage(backend, entries=entries)
        self.assertTrue(stage["available"])
        entries2 = _entries(backend)
        entries2[0]["observations"] = [{"fixture": 7}]
        stage2 = _run_stage(backend, entries=entries2)
        self.assertEqual(stage2["candidates"]["dst-a"]["status"], "unavailable")

    def test_marker_in_content(self):
        """Test marker in content."""
        # The classify probe searches token ids, not rendered
        # text, so marker bytes inside the content itself do not
        # break the measurement (the text-split approach cycle
        # 183 used for repetition had to skip such fixtures).
        case = Case(
            id="fx-marker",
            group="g7",
            messages=(
                Message(
                    role="user",
                    content="marknum has a " + chr(0) + "mpldst" + chr(0) + " inside",
                ),
            ),
            expected="7",
        )
        backend = _CharBackend({}, EXPECTED_BY_MARKER)
        entries = [
            {
                "format": {"name": fmt.name},
                "summary": {"aggregate_eligible": True},
                "observations": [_obs(backend, fmt, case, "7")],
            }
            for fmt in FORMATS
        ]
        stage = distraction_stage(
            backend,
            FORMATS,
            (case,),
            entries,
            boundary=tuple(backend.encode("~")),
            max_new_tokens=8,
            max_context=8192,
        )
        record = stage["candidates"]["dst-a"]["records"][0]
        cell = record["cells"]["neutral_64_before"]
        self.assertIsNotNone(cell)
        self.assertEqual(cell["construction"], "id_exact")


# ---------------------------------------------------------------------
# Mutation gates: each mutant is applied to a tempdir copy of
# distraction.py, exec'd under the package namespace (the module
# has relative imports), and must break a named scenario that
# passes against the real module.
# ---------------------------------------------------------------------


def _load_variant(source: str):
    """Load variant."""
    import importlib.util
    import os
    import tempfile

    with tempfile.TemporaryDirectory(prefix="c184_mut_", dir="/tmp") as tmpdir:
        mod_path = os.path.join(tmpdir, "distraction_variant.py")
        with open(mod_path, "w", encoding="utf-8") as handle:
            handle.write(source)
        spec = importlib.util.spec_from_file_location(
            "model_protocol_lab.distraction_variant", mod_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["model_protocol_lab.distraction_variant"] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop("model_protocol_lab.distraction_variant", None)
        return module


def _scenario_plant(module) -> None:
    """Scenario plant."""
    if module.planted_answer("9") != "10":
        raise AssertionError("carry lost")
    if module.planted_answer("9,999") != "10,000":
        raise AssertionError("grouped carry lost")


def _scenario_capture_population(module) -> None:
    """Scenario capture population."""
    record = _record(
        "f", False, {"near-miss_64_before": (False, "8"), "neutral_64_before": (False, "x")}
    )
    capture = module.aggregate_candidate([record])["capture"]["before_64"]
    if capture["n_population"] != 0:
        raise AssertionError("native-failing fixture entered the capture population")


def _scenario_boundary_screen(module) -> None:
    """Scenario boundary screen."""
    if not module.span_screened("span with <END> token", "zebra", ["<END>"]):
        raise AssertionError("boundary screen lost")


class TestMutationGates(unittest.TestCase):
    """Test mutation gates."""

    def _gate(self, anchor: str, mutant: str, scenario) -> None:
        """Gate."""
        source = Path(distraction.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(anchor), 1, f"anchor not unique: {anchor!r}")
        scenario(distraction)
        variant = _load_variant(source.replace(anchor, mutant))
        with self.assertRaises(AssertionError):
            scenario(variant)

    def test_gate_plant_carry(self):
        """Test gate plant carry."""
        self._gate(
            "        carry = digit // 10",
            "        carry = 0",
            _scenario_plant,
        )

    def test_gate_capture_population(self):
        """Test gate capture population."""
        self._gate(
            '                if not isinstance(native, dict) or not native.get("verdict"):',
            "                if not isinstance(native, dict) or False:",
            _scenario_capture_population,
        )

    def test_gate_boundary_screen(self):
        """Test gate boundary screen."""
        self._gate(
            "    return any(text and text in span for text in boundary_texts)",
            "    return False",
            _scenario_boundary_screen,
        )


if __name__ == "__main__":
    unittest.main()


class TestGapCoverage(unittest.TestCase):
    """Test gap coverage."""

    def test_plant_non_string(self):
        """Test plant non string."""
        self.assertIsNone(distraction.planted_answer(42))
        self.assertIsNone(distraction.planted_answer(None))

    def test_span_text_truncate_past_end(self):
        """Test span text truncate past end."""
        with self.assertRaises(ValueError):
            distraction.span_text(distraction.NEUTRAL_CORPUS, len(distraction.NEUTRAL_CORPUS), 1)

    def test_curves_for(self):
        """Test curves for."""
        curves = distraction.curves_for()
        self.assertEqual(len(curves), 4)
        self.assertEqual(curves[0], ("neutral", "before", (64, 256, 1024)))
        self.assertEqual(curves[3], ("near-miss", "after", (64, 256)))

    def test_classify_direct(self):
        """Test classify direct."""
        encode = lambda text: tuple(ord(c) for c in text)  # noqa: E731
        fmt = FMT_B
        native_ids = encode(fmt.render(CASE_NUM.messages))
        # after-placement, clean: id_exact.
        fixture_after = distraction.build_cell_fixture(CASE_NUM, "abc ", "after")
        arm_ids = encode(fmt.render(fixture_after.messages))
        construction, seam = distraction.classify_cell_construction(
            fmt, CASE_NUM, "abc ", "after", native_ids, arm_ids, encode
        )
        self.assertEqual(construction, "id_exact")
        self.assertEqual(seam, [])
        # after-placement with a corrupted inserted token (the
        # inserted region ends 12 ids before the end: the "~"
        # turn end plus the "ASSISTANT: " open-reply prefix).
        bad = list(arm_ids)
        bad[len(bad) - 13] = 1
        construction, seam = distraction.classify_cell_construction(
            fmt, CASE_NUM, "abc ", "after", native_ids, tuple(bad), encode
        )
        self.assertEqual(construction, "text_level")
        self.assertEqual(len(seam), 2)
        # before-placement with a corrupted prefix token.
        fixture_before = distraction.build_cell_fixture(CASE_NUM, "abc ", "before")
        arm_before = list(encode(fmt.render(fixture_before.messages)))
        arm_before[0] = 1
        construction, _seam = distraction.classify_cell_construction(
            fmt, CASE_NUM, "abc ", "before", native_ids, tuple(arm_before), encode
        )
        self.assertEqual(construction, "text_level")

        # Marker does not survive the encode: honest text_level.
        def dropping_encode(text):
            """Dropping encode."""
            if chr(0) in text:
                return ()
            return tuple(ord(c) for c in text)

        construction, seam = distraction.classify_cell_construction(
            fmt, CASE_NUM, "abc ", "before", native_ids, arm_ids, dropping_encode
        )
        self.assertEqual(construction, "text_level")
        self.assertEqual(seam, [])
        # Empty inserted region (arm == native): text_level, no seam.
        construction, seam = distraction.classify_cell_construction(
            fmt, CASE_NUM, "", "before", native_ids, native_ids, encode
        )
        self.assertEqual(construction, "text_level")
        self.assertEqual(seam, [])
        # Every guard fires: corrupted native prefix, corrupted
        # arm suffix (both placements), corrupted content (both
        # placements), truncated arms (both placements).
        bad_native = (1,) + native_ids[1:]
        construction, _ = distraction.classify_cell_construction(
            fmt, CASE_NUM, "abc ", "before", bad_native, arm_ids, encode
        )
        self.assertEqual(construction, "text_level")
        arm_before_ids = encode(fmt.render(fixture_before.messages))
        bad_suffix = arm_before_ids[:-1] + (1,)
        construction, _ = distraction.classify_cell_construction(
            fmt, CASE_NUM, "abc ", "before", native_ids, bad_suffix, encode
        )
        self.assertEqual(construction, "text_level")
        bad_suffix_after = arm_ids[:-1] + (1,)
        construction, _ = distraction.classify_cell_construction(
            fmt, CASE_NUM, "abc ", "after", native_ids, bad_suffix_after, encode
        )
        self.assertEqual(construction, "text_level")
        bad_content_before = list(arm_before_ids)
        bad_content_before[len(bad_content_before) - 14] = 1
        construction, _ = distraction.classify_cell_construction(
            fmt, CASE_NUM, "abc ", "before", native_ids, tuple(bad_content_before), encode
        )
        self.assertEqual(construction, "text_level")
        bad_content_after = list(arm_ids)
        bad_content_after[6] = 1
        construction, _ = distraction.classify_cell_construction(
            fmt, CASE_NUM, "abc ", "after", native_ids, tuple(bad_content_after), encode
        )
        self.assertEqual(construction, "text_level")
        truncated = arm_before_ids[:6]
        construction, _ = distraction.classify_cell_construction(
            fmt, CASE_NUM, "abc ", "before", native_ids, truncated, encode
        )
        self.assertEqual(construction, "text_level")
        truncated_after = arm_ids[:6]
        construction, _ = distraction.classify_cell_construction(
            fmt, CASE_NUM, "abc ", "after", native_ids, truncated_after, encode
        )
        self.assertEqual(construction, "text_level")
        # After-placement prefix guard.
        bad_prefix_after = (1,) + arm_ids[1:]
        construction, _ = distraction.classify_cell_construction(
            fmt, CASE_NUM, "abc ", "after", native_ids, bad_prefix_after, encode
        )
        self.assertEqual(construction, "text_level")

    def test_classify_overlap_guards(self):
        """Test classify overlap guards."""
        # Arms whose content tail overlaps the suffix head pass
        # the prefix/suffix/content checks yet leave no room for
        # the inserted region: the length guards must fire.
        encode = lambda text: tuple(ord(c) for c in text)  # noqa: E731
        mini = Format(
            name="dst-mini",
            bos="",
            user_prefix="U: ",
            assistant_prefix="ASSISTANT: ",
            turn_end="~",
        )
        case_before = Case(
            id="fx-ovb",
            group="g",
            messages=(Message(role="user", content="mar~A"),),
            expected="7",
        )
        native_b = encode(mini.render(case_before.messages))
        arm_b = encode("U: " + "\n" + "mar~A" + "SSISTANT: ")
        construction, _ = distraction.classify_cell_construction(
            mini, case_before, "x", "before", native_b, arm_b, encode
        )
        self.assertEqual(construction, "text_level")
        case_after = Case(
            id="fx-ova",
            group="g",
            messages=(Message(role="user", content="mark~"),),
            expected="7",
        )
        native_a = encode(mini.render(case_after.messages))
        arm_a = encode("U: " + "mark~" + "ASSISTANT: ")
        construction, _ = distraction.classify_cell_construction(
            mini, case_after, "x", "after", native_a, arm_a, encode
        )
        self.assertEqual(construction, "text_level")

    def test_aggregate_robustness(self):
        """Test aggregate robustness."""
        # Records with missing/malformed internals are skipped by
        # the paired loops rather than crashing the aggregate.
        weird = {
            "fixture": "w",
            "vacuous": False,
            "echo_fixture": False,
            "near_miss_defined": False,
            "planted_digest": None,
            "native": None,
            "cells": "junk",
            "cell_details": {},
        }
        aggregates = distraction.aggregate_candidate([weird])
        self.assertEqual(aggregates["n_fixtures"], 1)
        self.assertEqual(aggregates["curves"]["neutral_before"]["64"]["n_paired"], 0)
        weird2 = dict(weird, native={"verdict": None}, cells={})
        aggregates2 = distraction.aggregate_candidate([weird2])
        self.assertEqual(aggregates2["first_flip"]["neutral_before"]["n_population"], 0)
        weird3 = _record("w3", True, {"neutral_64_before": (True,)})
        weird3["native"]["verdict"] = None
        aggregates3 = distraction.aggregate_candidate([weird3])
        self.assertEqual(aggregates3["curves"]["neutral_before"]["64"]["n_paired"], 0)


# ---------------------------------------------------------------------
# Gap coverage (cycle 184 close-out): every validator reject branch
# fired one break at a time, fabricated-stage payloads for the
# analysis/diff branches the scripted stages cannot reach, and the
# stage/reporting corners the full-suite coverage run named.
# ---------------------------------------------------------------------


def _vrecord(fixture, *, native_pass=True, defined=True, cells=None, vacuous=False):
    """Validator-grade synthetic record (rendered-dose identity holds)."""
    digests = {a: sha256(a.encode()).hexdigest() for a in ("7", "8", "x")}
    cells = cells or {}
    built = {}
    details = {}
    for key in distraction.CONDITION_KEYS:
        level = int(key.split("_")[1])
        if vacuous:
            built[key] = None
            details[key] = "vacuous"
            continue
        if key.startswith("near-miss") and not defined:
            built[key] = None
            details[key] = "near_miss_undefined"
            continue
        spec = cells.get(key, (True, "7"))
        if spec is None:
            built[key] = None
            details[key] = "arm_context_exceeded"
            continue
        verdict, answer = spec
        built[key] = {
            "verdict": verdict,
            "boundary_found": True,
            "boundary_text_match": True,
            "input_tokens": 100 + level,
            "answer_digest": digests[answer],
            "construction": "id_exact",
            "seam_ids": [],
            "planted_occurrences": 1 if key.startswith("near-miss") else None,
        }
        details[key] = ""
    return {
        "fixture": fixture,
        "vacuous": vacuous,
        "echo_fixture": False,
        "near_miss_defined": defined,
        "planted_digest": digests["8"] if defined else None,
        "native": {
            "verdict": native_pass,
            "boundary_found": True,
            "boundary_text_match": True,
            "input_tokens": 100,
            "answer_digest": digests["7"],
        },
        "cells": built,
        "cell_details": details,
    }


def _fab_stage(records_by_name, ids=("fx-1", "fx-2")):
    """Fab stage."""
    protocol = {
        **distraction.protocol_object(list(ids), "sha256:fab"),
        "fingerprint": distraction.protocol_fingerprint(list(ids), "sha256:fab"),
    }
    candidates = {}
    for name, entry in records_by_name.items():
        if entry == "unavailable":
            candidates[name] = {
                "status": "unavailable",
                "detail": "native_search_unavailable",
                "records": [],
            }
        else:
            candidates[name] = {"status": "available", "detail": "", "records": entry}
    return {"available": True, "detail": "", "protocol": protocol, "candidates": candidates}


def _fab_payload(records_by_name, ids=("fx-1", "fx-2")):
    """Fab payload."""
    stage = _fab_stage(records_by_name, ids)
    return distraction_analysis(stage, "no_candidate", None, tuple(records_by_name))


class TestValidatorBatteryAndGaps(unittest.TestCase):
    """Test validator battery and gaps."""

    def setUp(self):
        """Set up the test fixture."""
        self.backend = _WordBackend(RULES, EXPECTED_BY_MARKER)
        self.payload = _payload(self.backend, _run_stage(self.backend))

    @staticmethod
    def _record0(payload):
        """Record0."""
        return payload["candidates"]["dst-a"]["records"][0]

    def test_structural_battery(self):
        """Test structural battery."""

        def breaks():
            """Breaks."""
            p = copy.deepcopy(self.payload)
            p["protocol"] = []
            yield "protocol_not_dict", p
            p = copy.deepcopy(self.payload)
            p["protocol"]["conditions"] = {}
            yield "conditions_not_list", p
            p = copy.deepcopy(self.payload)
            p["protocol"]["conditions"] = []
            yield "conditions_empty", p
            p = copy.deepcopy(self.payload)
            p["protocol"]["conditions"][0] = ["neutral", 64]
            yield "condition_short", p
            p = copy.deepcopy(self.payload)
            p["protocol"]["conditions"][0] = ("neutral", 64, "before")
            yield "condition_tuple", p
            p = copy.deepcopy(self.payload)
            p["protocol"]["conditions"][0][0] = "junk"
            yield "condition_kind", p
            p = copy.deepcopy(self.payload)
            p["protocol"]["conditions"][0][1] = "64"
            yield "condition_level_str", p
            p = copy.deepcopy(self.payload)
            p["protocol"]["conditions"][0][1] = True
            yield "condition_level_bool", p
            p = copy.deepcopy(self.payload)
            p["protocol"]["conditions"][0][1] = -64
            yield "condition_level_negative", p
            p = copy.deepcopy(self.payload)
            p["protocol"]["conditions"][0][2] = "middle"
            yield "condition_placement", p
            p = copy.deepcopy(self.payload)
            p["protocol"]["conditions"].append(list(p["protocol"]["conditions"][0]))
            yield "condition_duplicate", p
            p = copy.deepcopy(self.payload)
            p["candidates"] = []
            yield "candidates_not_dict", p
            p = copy.deepcopy(self.payload)
            p["candidates"]["dst-a"] = "junk"
            yield "entry_not_dict", p
            p = copy.deepcopy(self.payload)
            p["candidates"]["dst-a"]["records"] = {}
            yield "records_not_list", p
            p = copy.deepcopy(self.payload)
            p["candidates"]["dst-b"]["status"] = "unavailable"
            yield "unavailable_with_records", p
            p = copy.deepcopy(self.payload)
            p["candidates"]["dst-a"]["records"][0] = "junk"
            yield "record_not_dict", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["fixture"] = ""
            yield "fixture_empty", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["fixture"] = 42
            yield "fixture_not_str", p
            p = copy.deepcopy(self.payload)
            p["candidates"]["dst-a"]["records"].append(copy.deepcopy(self._record0(p)))
            yield "fixture_duplicate", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["vacuous"] = "yes"
            yield "vacuous_not_bool", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["planted_digest"] = "zz"
            yield "planted_digest_shape", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["near_miss_defined"] = False
            yield "defined_digest_mismatch", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["native"] = []
            yield "native_not_dict", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["native"]["verdict"] = 1
            yield "native_verdict_int", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["native"]["boundary_found"] = None
            yield "native_boundary_none", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["native"]["boundary_text_match"] = "yes"
            yield "native_btm_str", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["native"]["input_tokens"] = -1
            yield "native_tokens_negative", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["native"]["answer_digest"] = "x" * 63
            yield "native_digest_short", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["cells"] = []
            yield "cells_not_dict", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["cells"].pop("neutral_64_before")
            yield "cells_missing_key", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["cell_details"]["extra_key"] = ""
            yield "details_extra_key", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["cell_details"]["neutral_1024_before"] = "bogus"
            yield "detail_vocabulary", p
            p = copy.deepcopy(self.payload)
            # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
            vac = next(r for r in p["candidates"]["dst-a"]["records"] if r["vacuous"])
            vac["cells"]["neutral_64_before"] = {
                "verdict": True,
                "boundary_found": True,
                "boundary_text_match": True,
                "input_tokens": 164,
                "answer_digest": sha256(b"7").hexdigest(),
                "construction": "id_exact",
                "seam_ids": [],
                "planted_occurrences": None,
            }
            yield "vacuous_with_cell", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["cell_details"]["neutral_64_before"] = "vacuous"
            yield "detail_vacuous_on_live", p
            p = copy.deepcopy(self.payload)
            rec = self._record0(p)
            rec["cells"]["near-miss_64_before"] = {
                "verdict": True,
                "boundary_found": True,
                "boundary_text_match": True,
                "input_tokens": 164,
                "answer_digest": sha256(b"7").hexdigest(),
                "construction": "id_exact",
                "seam_ids": [],
                "planted_occurrences": 1,
            }
            rec["near_miss_defined"] = False
            rec["planted_digest"] = None
            rec["cell_details"]["near-miss_64_before"] = ""
            yield "nearmiss_cell_when_undefined", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["cells"]["neutral_64_before"]["verdict"] = "pass"
            yield "cell_verdict_str", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["cells"]["neutral_64_before"]["construction"] = "exact"
            yield "cell_construction", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["cells"]["neutral_64_before"]["seam_ids"] = [1, 2, 3]
            yield "cell_seam_long", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["cells"]["neutral_64_before"]["seam_ids"] = ["a"]
            yield "cell_seam_type", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["cells"]["neutral_64_before"]["planted_occurrences"] = 0
            yield "neutral_occurrences", p
            p = copy.deepcopy(self.payload)
            self._record0(p)["cells"]["near-miss_64_before"]["planted_occurrences"] = None
            yield "nearmiss_occurrences_none", p
            p = copy.deepcopy(self.payload)
            p["primary_decision"] = 42
            yield "primary_decision_type", p
            p = copy.deepcopy(self.payload)
            p["protocol"]["backend_fingerprint"] = "sha256:other"
            yield "fingerprint_mismatch", p
            p = copy.deepcopy(self.payload)
            for entry in p["candidates"].values():
                for record in entry["records"]:
                    if record["fixture"] == "fx-num":
                        record["fixture"] = "renamed-1"
            yield "fixture_set_sha_mismatch", p
            p = copy.deepcopy(self.payload)
            p["conditions"]["neutral_64_before"]["winner"] = "dst-b"
            yield "conditions_mismatch", p
            p = copy.deepcopy(self.payload)
            p["n_candidates"] = 99
            yield "n_candidates_mismatch", p

        for name, broken in breaks():
            with self.subTest(breaks=name):
                self.assertFalse(_valid_distraction(broken), name)

    def test_fab_capture_cell_none_and_valid(self):
        """Test fab capture cell none and valid."""
        cells = {"near-miss_64_before": None}
        records = {
            "dst-a": [
                _vrecord("fx-1", cells=cells),
                _vrecord("fx-2", cells={"near-miss_64_before": (True, "8")}),
            ],
            "dst-b": [
                _vrecord("fx-1", cells=cells),
                _vrecord("fx-2", cells={"near-miss_64_before": (True, "8")}),
            ],
        }
        payload = _fab_payload(records)
        self.assertTrue(_valid_distraction(payload))
        capture = payload["candidates"]["dst-a"]["capture"]["before_64"]
        self.assertEqual(capture["n_population"], 1)
        self.assertEqual(capture["n_captured"], 1)

    def test_fab_completed_sets_differ_withholds(self):
        """Test fab completed sets differ withholds."""
        records = {
            "dst-a": [_vrecord("fx-1"), _vrecord("fx-2")],
            "dst-b": [
                _vrecord("fx-1", cells={"neutral_64_before": None}),
                _vrecord("fx-2"),
            ],
        }
        payload = _fab_payload(records)
        self.assertTrue(payload["conditions"]["neutral_64_before"]["withheld"])
        self.assertTrue(_valid_distraction(payload))

    def test_analysis_unavailable_candidate(self):
        """Test analysis unavailable candidate."""
        records = {
            "dst-a": [_vrecord("fx-1"), _vrecord("fx-2")],
            "dst-b": "unavailable",
        }
        payload = _fab_payload(records)
        self.assertEqual(payload["candidates"]["dst-b"]["status"], "unavailable")
        self.assertTrue(payload["conditions"]["neutral_64_before"]["withheld"])
        self.assertTrue(_valid_distraction(payload))
        html = render_html({"distraction": payload})
        self.assertIn("unavailable", html)

    def test_analysis_selected_condition(self):
        """Test analysis selected condition."""
        records = {
            "dst-a": [_vrecord("fx-1"), _vrecord("fx-2")],
            "dst-b": [
                _vrecord("fx-1", cells={"neutral_64_before": (False, "x")}),
                _vrecord("fx-2"),
            ],
        }
        payload = _fab_payload(records)
        condition = payload["conditions"]["neutral_64_before"]
        self.assertEqual(condition["decision"], "selected")
        self.assertEqual(condition["winner"], "dst-a")
        self.assertTrue(_valid_distraction(payload))

    def test_curve_drift_between_fab_payloads(self):
        """Test curve drift between fab payloads."""
        base = {
            "dst-a": [_vrecord("fx-1"), _vrecord("fx-2")],
            "dst-b": [_vrecord("fx-1"), _vrecord("fx-2")],
        }
        other = {
            "dst-a": [
                _vrecord("fx-1", cells={"neutral_64_before": (False, "x")}),
                _vrecord("fx-2"),
            ],
            "dst-b": [_vrecord("fx-1"), _vrecord("fx-2")],
        }
        result = diff_distraction(_fab_payload(base), _fab_payload(other))
        self.assertTrue(result["changed"])
        self.assertTrue(result["curve_drift"])

    def test_stage_decode_failure_falls_back(self):
        """Test stage decode failure falls back."""

        class _BadDecodeBackend(_WordBackend):
            """Bad decode backend."""

            @staticmethod
            def decode(ids):
                """Decode."""
                raise RuntimeError("scripted decode failure")

        backend = _BadDecodeBackend(RULES, EXPECTED_BY_MARKER)
        stage = _run_stage(backend)
        self.assertTrue(stage["available"])

    def test_stage_sizing_skips_plant_cutting_suffix(self):
        """Test stage sizing skips plant cutting suffix."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from unittest import mock

        backend = _CharBackend()
        real = distraction.suffix_cuts_plant
        calls = {"n": 0}

        def flaky(first, second, planted):
            """Flaky."""
            calls["n"] += 1
            if calls["n"] == 1:
                return True
            return real(first, second, planted)

        with mock.patch.object(distraction, "suffix_cuts_plant", flaky):
            stage = _run_stage(backend, formats=(FMT_A,), cases=(CASE_NUM,))
        self.assertTrue(stage["available"])
        self.assertGreaterEqual(calls["n"], 1)

    def test_stage_sizing_skips_empty_span(self):
        """Test stage sizing skips empty span."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from unittest import mock

        backend = _WordBackend(RULES, EXPECTED_BY_MARKER)
        real = distraction.span_text
        calls = {"n": 0}

        def flaky(sentences, whole, suffix):
            """Flaky."""
            calls["n"] += 1
            if calls["n"] == 1:
                return ""
            return real(sentences, whole, suffix)

        with mock.patch.object(distraction, "span_text", flaky):
            stage = _run_stage(backend)
        self.assertTrue(stage["available"])
        self.assertGreaterEqual(calls["n"], 1)

    def test_stage_cancel_inside_candidate_loop(self):
        """Test stage cancel inside candidate loop."""

        # The event is set by the LAST cell generation of dst-a's
        # fx-copy case, so no per-cell check remains to catch it:
        # the next case's loop-head check must fire. (fx-vac, the
        # last case, is vacuous and never generates.)
        class _LateCancelBackend(_WordBackend):
            """Late cancel backend."""

            def generate(self, input_ids, **kwargs):
                """Generate."""
                generation = super().generate(input_ids, **kwargs)
                marker, _kind, _placement = self._cell_of(self.decode(list(input_ids)))
                if marker == "markcopy":
                    self._copy_generations = getattr(self, "_copy_generations", 0) + 1
                    # fx-copy generates exactly 4 cells per candidate
                    # on this backend (1024 is unmatchable at word
                    # level; "BLUE" has no near-miss plant).
                    if self._copy_generations == 4 and self.cancel_event is not None:
                        self.cancel_event.set()
                return generation

        event = Event()
        backend = _LateCancelBackend(RULES, EXPECTED_BY_MARKER)
        backend.cancel_event = event
        stage = _run_stage(backend, cancel=event)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        self.assertEqual(stage["candidates"], {})

    def test_reporting_malformed_shapes(self):
        """Test reporting malformed shapes."""
        self.assertNotIn(
            'id="distraction"',
            render_html({"distraction": {"available": True, "candidates": []}}),
        )
        payload = _fab_payload(
            {
                "dst-a": [_vrecord("fx-1"), _vrecord("fx-2")],
                "dst-b": [_vrecord("fx-1"), _vrecord("fx-2")],
            }
        )
        junk_conditions = copy.deepcopy(payload)
        junk_conditions["conditions"] = {"neutral_64_before": "junk"}
        self.assertIn('id="distraction"', render_html({"distraction": junk_conditions}))
        junk_candidate = copy.deepcopy(payload)
        junk_candidate["candidates"]["dst-a"] = "junk"
        self.assertIn('id="distraction"', render_html({"distraction": junk_candidate}))
        junk_capture = copy.deepcopy(payload)
        junk_capture["candidates"]["dst-a"]["capture"] = {"before_64": "junk"}
        self.assertIn('id="distraction"', render_html({"distraction": junk_capture}))


class TestValidatorBatteryRound2(unittest.TestCase):
    """Second battery round: the reject branches round 1 missed."""

    def setUp(self):
        """Set up the test fixture."""
        self.backend = _WordBackend(RULES, EXPECTED_BY_MARKER)
        self.payload = _payload(self.backend, _run_stage(self.backend))

    def test_remaining_reject_branches(self):
        """Test remaining reject branches."""

        def breaks():
            """Breaks."""
            p = copy.deepcopy(self.payload)
            p["candidates"] = {42: p["candidates"]["dst-a"]}
            yield "candidate_name_not_str", p
            p = copy.deepcopy(self.payload)
            p["candidates"]["dst-a"]["status"] = "weird"
            yield "candidate_status_vocabulary", p
            p = copy.deepcopy(self.payload)
            record0 = copy.deepcopy(p["candidates"]["dst-a"]["records"][0])
            many = []
            for i in range(100):
                clone = copy.deepcopy(record0)
                clone["fixture"] = f"bulk-{i}"
                many.append(clone)
            p["candidates"] = {
                f"cand-{i}": {
                    "status": "available",
                    "detail": "",
                    "records": copy.deepcopy(many),
                }
                for i in range(25)
            }
            yield "total_records_over_cap", p
            p = copy.deepcopy(self.payload)
            p["candidates"]["dst-a"]["records"][0]["cell_details"]["near-miss_64_before"] = (
                "near_miss_undefined"
            )
            yield "detail_undefined_on_defined", p
            p = copy.deepcopy(self.payload)
            p["candidates"]["dst-a"]["records"][0]["cells"]["neutral_64_before"] = "junk"
            yield "cell_not_dict", p
            p = copy.deepcopy(self.payload)
            p["candidates"]["dst-a"]["records"][0]["cells"]["neutral_64_before"][
                "boundary_found"
            ] = None
            yield "cell_boundary_none", p
            p = copy.deepcopy(self.payload)
            p["candidates"]["dst-a"]["records"][0]["cells"]["neutral_64_before"][
                "boundary_text_match"
            ] = "yes"
            yield "cell_btm_str", p
            p = copy.deepcopy(self.payload)
            p["candidates"]["dst-a"]["records"][0]["cells"]["neutral_64_before"][
                "input_tokens"
            ] = -1
            yield "cell_tokens_negative", p
            p = copy.deepcopy(self.payload)
            p["candidates"]["dst-a"]["records"][0]["cells"]["neutral_64_before"][
                "answer_digest"
            ] = "x" * 63
            yield "cell_digest_short", p
            p = copy.deepcopy(self.payload)
            p["candidates"]["dst-b"]["records"][0]["fixture"] = "only-in-b"
            yield "fixture_ids_differ_across_candidates", p

        for name, broken in breaks():
            with self.subTest(breaks=name):
                self.assertFalse(_valid_distraction(broken), name)

    def test_share_drift_between_fab_payloads(self):
        """Test share drift between fab payloads."""
        base = {
            "dst-a": [_vrecord("fx-1"), _vrecord("fx-2")],
            "dst-b": [_vrecord("fx-1"), _vrecord("fx-2")],
        }
        other = {
            "dst-a": [_vrecord("fx-1"), _vrecord("fx-2", defined=False)],
            "dst-b": [_vrecord("fx-1"), _vrecord("fx-2")],
        }
        result = diff_distraction(_fab_payload(base), _fab_payload(other))
        self.assertTrue(result["changed"])
        metrics = {entry["metric"] for entry in result["curve_drift"]}
        self.assertIn("nearmiss_defined_share", metrics)

    def test_stage_non_list_search_entries(self):
        """Test stage non list search entries."""
        backend = _WordBackend(RULES, EXPECTED_BY_MARKER)
        stage = distraction_stage(
            backend,
            FORMATS,
            CASES,
            {"not": "a list"},
            boundary=(99,),
            max_new_tokens=8,
            max_context=8192,
            seed=0,
        )
        self.assertTrue(stage["available"])
        self.assertEqual(stage["candidates"]["dst-a"]["status"], "unavailable")

    def test_reporting_entry_without_curves(self):
        """Test reporting entry without curves."""
        payload = _fab_payload(
            {
                "dst-a": [_vrecord("fx-1"), _vrecord("fx-2")],
                "dst-b": [_vrecord("fx-1"), _vrecord("fx-2")],
            }
        )
        payload["candidates"]["dst-a"].pop("curves")
        html = render_html({"distraction": payload})
        self.assertIn('id="distraction"', html)
