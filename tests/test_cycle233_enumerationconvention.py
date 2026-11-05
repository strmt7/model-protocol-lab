"""Cycle 233: the enumeration-convention probe (wave-12
round 4, built as amended A1-A9).

Pins, by independent execution: the frozen generator
(a from-text second implementation, the seed-0 case-0
literals, the forced-collision redraw path and the
exhaustion give-up); the section-4 validator with its
sub-rules pinned standalone (amendment A4's completed
bare-glyph set among them); the section-5 decomposition
as amended (A1 marker-prefixed wrong values, A2
ASCII-pinned glyphs and totality, A3 string-identity
echo), against the shipped battery, the collision
witnesses, and a fuzz that also cross-checks the
diffreports mirror; the completion guard (exhaustive
2,925-state audit, the executed attack worlds at the
designer's seed and at a disjoint seed, and the ids
dimension at the aggregate level); the exact operating
characteristics (Fraction enumeration, the exact paired
column, the floors frontier and plateau, and the
corrected required-N of amendment A6); the stage
payloads end to end through scripted backends; the
diffreports mirror's recompute validation and tamper
rejections; and the reporting section.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
import re
import unittest
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from statistics import NormalDist
from unittest import mock

from model_protocol_lab import diffreports as dr
from model_protocol_lab import enumerationconvention as ec
from model_protocol_lab import experiments, reporting
from model_protocol_lab.core import Format, Generation

_CHATML = Format("chatml", "", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n")
_LLAMA3 = Format(
    "llama3",
    "<|begin_of_text|>",
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|eot_id|>",
)
_REPO = Path(__file__).resolve().parent.parent
_MARKER_TOKENS = {m for arm in ec.ARM_KEYS for m in ec.MARKER_TABLE[arm]} | {
    ec.ROMAN_TABLE[t].lower() + "." for t in ec.INDICES
}


class _ListBackend:
    """Word-token scripted backend over the list panel.

    Modes: ``perfect`` (the asked topic's value read out
    of its list line, boundary on), ``echo`` (the line's
    own marker token prepended to the value, every arm),
    ``markerwrong`` (treated arms only: the line's marker
    prepended to a sibling value -- the amendment-A1
    witness behaviour), ``wrongvalue`` (treated arms
    only: a bare sibling value), ``garbage`` (a fixed
    non-answer), ``noboundary`` (the value with no
    boundary token), ``explode`` (generate raises),
    ``flipflop`` (alternates value / sibling per repeated
    identical input). With ``strip_markers=True`` the
    encoder drops every marker token (the ids-identical
    regime: treated renders tokenise exactly as control).
    """

    fingerprint = "sha256:test-enumeration-convention"

    def __init__(self, mode: str = "perfect", strip_markers: bool = False) -> None:
        """Init."""
        self.mode = mode
        self.strip_markers = strip_markers
        self.calls = 0
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}
        self.seen: dict[str, int] = {}

    def encode(self, text):
        """Encode."""
        ids = []
        for token in text.replace("\n", " \n ").split(" "):
            if not token:
                continue
            if self.strip_markers and token in _MARKER_TOKENS:
                continue
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                self.rev[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return tuple(ids)

    def decode(self, ids):
        """Decode."""
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        self.calls += 1
        if self.mode == "explode":
            raise RuntimeError("boom")
        text = self.decode(list(input_ids))
        question = re.search(r"What is the (\S+) inspection code\?", text)
        topic = question.group(1) if question else "unknown"
        match = re.search(r"The " + re.escape(topic) + r" inspection code is (\S+?)\.", text)
        value = match.group(1) if match else "unknownword-0000"
        values = re.findall(r"inspection code is (\S+?)\.", text)
        sibling = next((v for v in values if v != value), value)
        marker_match = re.search(r"(\S+) The " + re.escape(topic) + r" inspection code is", text)
        marker = marker_match.group(1) if marker_match else ""
        treated = bool(marker) and not re.fullmatch(r"[0-9]+\.", marker)
        drop_boundary = self.mode == "noboundary"
        if self.mode == "perfect":
            answer = value
        elif self.mode == "echo":
            answer = f"{marker} {value}" if marker else value
        elif self.mode == "markerwrong":
            answer = f"{marker} {sibling}" if treated and marker else value
        elif self.mode == "wrongvalue":
            answer = sibling if treated else value
        elif self.mode == "garbage":
            answer = "unknownword"
        elif self.mode == "flipflop":
            count = self.seen.get(text, 0)
            self.seen[text] = count + 1
            answer = value if count % 2 == 0 else sibling
        else:
            answer = value
        boundary = self.encode("endmark")[0]
        out = list(self.encode(answer))
        if not drop_boundary:
            out.append(boundary)
        pad = boundary if not drop_boundary else self.encode("fillerword")[0]
        out = (out + [pad] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(mode="perfect", strip_markers=False, formats=(_CHATML, _LLAMA3), **stage_kwargs):
    """Payload."""
    backend = _ListBackend(mode, strip_markers=strip_markers)
    stage_kwargs.setdefault("boundary", backend.encode("endmark"))
    stage_kwargs.setdefault("max_context", 100_000)
    stage_kwargs.setdefault("seed", 0)
    stage = experiments.enumeration_convention_stage(backend, formats, **stage_kwargs)
    names = tuple(fmt.name for fmt in formats)
    return experiments.enumeration_convention_analysis(stage, names), backend, stage


def _valid_payload():
    """A fresh valid perfect-mode payload (mirror-test
    fixture)."""
    payload, _backend, _stage = _payload("perfect")
    return payload


def _cell(text=None, *, verdict=None, boundary=True, tokens=100, expected="iris-1423"):
    """Cell (the battery's expected value is iris-1423)."""
    digest = None if text is None else sha256(text.encode("utf-8")).hexdigest()
    if verdict is None:
        verdict = bool(boundary and text is not None and text == expected)
    return {
        "verdict": verdict,
        "boundary_found": boundary,
        "boundary_text_match": True,
        "input_tokens": tokens,
        "answer_digest": digest,
        "answer_text": text,
    }


# ------------------------------------------------------------------
# Independent implementation B of the section-3 generator,
# written from the design text (not from the module): the
# freeze test compares the two over seeds, and again under
# a patched draw that forces the redraw path.
# ------------------------------------------------------------------


def _draw_b(seed, index, slot, counter):
    """Draw (implementation B)."""
    material = str(seed) + "|enumerationconvention|" + str(index) + "|" + str(slot)
    material += "|" + str(counter)
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def _case_b(seed, index, seen, draw=_draw_b):
    """Case (implementation B)."""
    n_first = draw(seed, index, 2, 0) % 24
    n_second = draw(seed, index, 3, 0) % 24
    if n_second == n_first:
        n_second = (n_first + 1) % 24
    position = 1 + draw(seed, index, 20, 0) % 9
    topics = []
    fillers = [ec.TOPICS[(index + 5 * j) % 24] for j in range(1, 9)]
    for t in range(1, 10):
        topics.append(ec.TOPICS[index] if t == position else fillers.pop(0))
    values = []
    for t in range(1, 10):
        counter = 0
        while True:
            word = ec.WORDS[draw(seed, index, 29 + t, counter) % 24]
            number = 1000 + draw(seed, index, 39 + t, counter) % 9000
            value = word + "-" + str(number)
            if value not in seen:
                break
            counter += 1
            if counter > 1000:
                raise ec.RedrawExhausted
        seen.add(value)
        values.append(value)
    return {
        "fixture_id": f"enumerationconvention-{index:02d}",
        "index": index,
        "topic": ec.TOPICS[index],
        "position": position,
        "topics": topics,
        "values": values,
        "expected": values[position - 1],
        "question": "What is the " + ec.TOPICS[index] + " inspection code?",
        "neutral_indices": [n_first, n_second],
    }


def _panel_b(seed, draw=_draw_b):
    """Panel (implementation B)."""
    seen: set[str] = set()
    return [_case_b(seed, i, seen, draw) for i in range(24)]


class TestGeneratorFreeze(unittest.TestCase):
    """The frozen panel: two implementations, the seed-0
    literals, the pool digests, and the redraw path.
    """

    def test_pool_digests_pinned(self):
        """Pools pinned by content digest (A8, TOPICS
        included per the readiness extension)."""
        self.assertEqual(
            ec.POOL_DIGESTS["topics"],
            "71d122676c1b84aa96fc7e690e2308360ccbf3448d4be75213551e3d81da236e",
        )
        self.assertEqual(
            ec.POOL_DIGESTS["words"],
            "26b9f09bc49ee096aa3d04a3951ef739acd617819af04c8be1bbbe0a5097f1b1",
        )
        self.assertEqual(
            ec.POOL_DIGESTS["neutral_statements"],
            "b2afc3e88afabd8efea564d4e5eb54e94ddc1a71c355263d8ccfc7f831e6db07",
        )
        self.assertEqual(len(ec.TOPICS), 24)
        self.assertEqual(len(ec.WORDS), 24)
        self.assertEqual(len(ec.NEUTRAL_STATEMENTS), 24)

    def test_two_implementations_agree(self):
        """Module vs implementation B over five seeds:
        cases, contents, validation."""
        for seed in (0, 1, 7, 42, 20261030):
            panel = ec.generate_cases(seed)
            panel_b = _panel_b(seed)
            self.assertEqual(panel, panel_b)
            self.assertEqual(len({v for case in panel for v in case["values"]}), 216)
            for case, case_b in zip(panel, panel_b, strict=True):
                self.assertTrue(ec.validate_case(case))
                self.assertEqual(ec.generate_case(seed, case["index"]), case)
                for arm in ec.ARM_KEYS:
                    self.assertEqual(ec.arm_content(case, arm), ec.arm_content(case_b, arm))

    def test_seed0_case0_literals(self):
        """Seed 0 case 0: topic harbor, position 9,
        expected iris-1423, sibling cedar-4159."""
        case = ec.generate_case(0, 0)
        self.assertEqual(case["topic"], "harbor")
        self.assertEqual(case["position"], 9)
        self.assertEqual(case["expected"], "iris-1423")
        self.assertEqual(case["values"][8], "iris-1423")
        self.assertIn("cedar-4159", case["values"])
        self.assertEqual(case["topics"][8], "harbor")

    def test_marker_table_and_content_shape(self):
        """The frozen marker table and one content's
        exact shape."""
        case = ec.generate_case(0, 0)
        content = ec.arm_content(case, "roman")
        lines = content.split("\n")
        self.assertEqual(lines[0].endswith("Inspection codes:"), True)
        self.assertTrue(lines[1].startswith("I. The "))
        self.assertTrue(lines[9].startswith("IX. The "))
        self.assertTrue(content.endswith("What is the harbor inspection code?"))
        bullet = ec.arm_content(case, "bullet")
        for line in bullet.split("\n")[1:10]:
            self.assertTrue(line.startswith("• The "))
        with self.assertRaises(ValueError):
            ec.arm_content(case, "nope")
        with self.assertRaises(ValueError):
            ec.marker_for("nope", 1)

    def test_forced_collision_redraw(self):
        """FG-4 discharged: a patched draw forcing item
        2's first-draw value to equal item 1's exercises
        the shared-counter redraw; both implementations
        agree on the redrawn panel."""
        real = ec._derive

        def patched(seed, index, slot, counter):
            if index == 0 and counter == 0 and slot == 31:
                return real(seed, index, 30, 0)
            if index == 0 and counter == 0 and slot == 41:
                return real(seed, index, 40, 0)
            return real(seed, index, slot, counter)

        with mock.patch.object(ec, "_derive", patched):
            panel = ec.generate_cases(0)
        self.assertEqual(panel, _panel_b(0, draw=patched))
        case = panel[0]
        self.assertNotEqual(case["values"][0], case["values"][1])
        unpatched = ec.generate_case(0, 0)
        self.assertNotEqual(case["values"][1], unpatched["values"][1])
        self.assertEqual(case["values"][0], unpatched["values"][0])

    def test_redraw_exhaustion_raises(self):
        """A draw that always collides gives up past the
        frozen limit."""
        with (
            mock.patch.object(ec, "_derive", lambda *args: 0),
            self.assertRaises(ec.RedrawExhausted),
        ):
            ec.generate_cases(0)


class TestDesignDocCompleteness(unittest.TestCase):
    """The consolidated design doc carries the amended
    contract (A6-A8 pins) and the nav row ships with it.
    """

    @classmethod
    def setUpClass(cls):
        """Set up class."""
        cls.doc = (_REPO / "docs" / "ENUMERATION_DESIGN.md").read_text(encoding="utf-8")

    def test_amendment_pins_present(self):
        """Digests, Sclar set, criterion disclosure,
        corrected N, naming rationale."""
        doc = self.doc
        for digest in (
            "71d122676c1b84aa96fc7e690e2308360ccbf3448d4be75213551e3d81da236e",
            "26b9f09bc49ee096aa3d04a3951ef739acd617819af04c8be1bbbe0a5097f1b1",
            "b2afc3e88afabd8efea564d4e5eb54e94ddc1a71c355263d8ccfc7f831e6db07",
        ):
            self.assertIn(digest, doc)
        self.assertIn("0x215F+x+1", doc)
        self.assertIn("ROMAN[x].upper()", doc)
        self.assertIn("probability-ranking", doc)
        self.assertIn("16,848", doc)
        self.assertIn("--enumeration-style", doc)
        self.assertIn("2,925", doc)
        self.assertIn("compatibilityform.py", doc)
        self.assertIn("§8.63", doc)

    def test_nav_row_present(self):
        """mkdocs nav carries the design doc (same-push
        rule)."""
        nav = (_REPO / "mkdocs.yml").read_text(encoding="utf-8")
        self.assertIn("Enumeration convention: ENUMERATION_DESIGN.md", nav)

    def test_usage_section_present(self):
        """USAGE section 8.63 exists."""
        usage = (_REPO / "docs" / "USAGE.md").read_text(encoding="utf-8")
        self.assertIn("### 8.63 Enumeration convention (`run --enumeration-convention`)", usage)


class TestValidator(unittest.TestCase):
    """Section 4: V1-V4 and the standalone sub-rules
    (A4's completed bare-glyph set pinned through V3's
    own entry point)."""

    BATTERY = [
        ("What is the harbor inspection code?", "harbor", "cedar-4159", True),
        ("What is the first harbor inspection code?", "harbor", "cedar-4159", False),
        ("What is the harbor inspection code?", "harbor", "cedar-0415", False),
        ("What is the harbor inspection code?", "harbor", "xxxxx-1234", False),
        ("What is the 3 harbor inspection code?", "harbor", "cedar-4159", False),
        ("What is the A harbor inspection code?", "harbor", "cedar-4159", False),
        ("What is the IX harbor inspection code?", "harbor", "cedar-4159", False),
        ("What is the (1) harbor inspection code?", "harbor", "cedar-4159", False),
        ("What is the • harbor inspection code?", "harbor", "cedar-4159", False),
        ("Tell me the harbor inspection code.", "harbor", "cedar-4159", False),
        ("What is the harbor inspection codes?", "harbor", "cedar-4159", False),
        ("What is the above harbor inspection code?", "harbor", "cedar-4159", False),
        ("What is item 3 of the harbor inspection code?", "harbor", "cedar-4159", False),
        ("What is the harbor inspection code?", "harbor", "cedar-10000", False),
        ("What is the harbor inspection code?", "harbor", "cedar-999", False),
        ("What is the harbor inspection code?", "harbor", "cedar-9999", True),
        ("What is the harbor inspection code?", "harbor", "cedar-1000", True),
        ("What is the marker harbor inspection code?", "harbor", "cedar-4159", False),
        ("What is the harbor [1] inspection code?", "harbor", "cedar-4159", False),
    ]

    def test_battery(self):
        """The 19-row validator battery."""
        for question, topic, expected, want in self.BATTERY:
            self.assertEqual(
                ec.validate_question(question, topic, expected), want, (question, expected)
            )

    def test_v2_forbidden_hits_standalone(self):
        """V2 names each forbidden family independently."""
        self.assertEqual(
            ec._forbidden_hits("What is the first harbor inspection code?"), ["ordinal"]
        )
        self.assertEqual(
            ec._forbidden_hits("What is item 3 of the harbor inspection code?"), ["item-ref"]
        )
        self.assertEqual(ec._forbidden_hits("Pick option B now"), ["option-ref"])
        self.assertEqual(ec._forbidden_hits("See line 4 above"), ["ordinal", "line-ref"])
        self.assertEqual(ec._forbidden_hits("What is the harbor inspection code?"), [])

    def test_v3_marker_tokens_standalone(self):
        """V3 catches every arm's marker rendering."""
        for marker in ("3.", "A.", "i.", "IX.", "(9)", "•"):
            question = f"What is the {marker} harbor inspection code?"
            self.assertTrue(ec._marker_token_hit(question), marker)

    def test_v3_bare_glyphs_a4(self):
        """A4: the bare-glyph set includes the lowercase
        Roman forms and every glyph the scorer parses
        (the F4 witnesses ix / v / iv / viii included)."""
        for glyph in ("ix", "v", "iv", "viii", "x", "X", "l", "L", "c", "C", "d", "D", "m", "M"):
            question = f"What is the {glyph} harbor inspection code?"
            self.assertTrue(ec._marker_token_hit(question), glyph)
            self.assertFalse(ec.validate_question(question, "harbor", "cedar-4159"), glyph)

    def test_v4_value_grammar_fullmatch(self):
        """V4's grammar is fullmatch: trailing newline /
        space never parse (cycle-231 F1 hygiene)."""
        self.assertTrue(ec._value_grammar_ok("cedar-4159"))
        self.assertFalse(ec._value_grammar_ok("cedar-4159\n"))
        self.assertFalse(ec._value_grammar_ok("cedar-4159 "))
        self.assertFalse(ec._value_grammar_ok(" cedar-4159"))
        self.assertFalse(ec._value_grammar_ok("cedar-0415"))
        self.assertFalse(ec._value_grammar_ok(None))
        self.assertFalse(ec._value_grammar_ok(4159))

    def test_validate_case_tampering(self):
        """The per-case battery rejects single-field
        corruptions and never raises on junk."""
        case = ec.generate_case(0, 0)
        self.assertTrue(ec.validate_case(case))
        for key, bad in (
            ("position", 3),
            ("topic", "summit"),
            ("expected", "cedar-4159"),
            ("fixture_id", "enumerationconvention-01"),
            ("index", 1),
        ):
            broken = copy.deepcopy(case)
            broken[key] = bad
            self.assertFalse(ec.validate_case(broken), key)
        broken = copy.deepcopy(case)
        broken["values"][0] = broken["values"][1]
        self.assertFalse(ec.validate_case(broken))
        broken = copy.deepcopy(case)
        broken["topics"][0] = broken["topics"][1]
        self.assertFalse(ec.validate_case(broken))
        broken = copy.deepcopy(case)
        broken["neutral_indices"] = [broken["neutral_indices"][0]] * 2
        self.assertFalse(ec.validate_case(broken))
        self.assertFalse(ec.validate_case({}))
        self.assertFalse(ec.validate_case({"fixture_id": 7}))


# The decomposition battery: (answer, arm, target, class,
# extra cell kwargs). Expected value iris-1423 throughout.
_DECOMP_BATTERY = [
    ("iris-1423", "control_arabic", 9, "pass", {"verdict": True}),
    ("iris-1423", "control_arabic", 9, "boundary_only", {"boundary": False}),
    ("cedar-4159", "control_arabic", 9, "wrong_value", {}),
    ("quill-9999", "roman", 9, "wrong_value", {}),
    ("marigold-9999", "roman", 9, "non_answer", {}),
    ("9. iris-1423", "control_arabic", 9, "echo", {}),
    ("(9) iris-1423", "wrapper", 9, "echo", {}),
    ("IX. iris-1423", "roman", 9, "echo", {}),
    ("ix. iris-1423", "roman", 9, "echo", {}),
    ("I. iris-1423", "roman", 9, "form_only", {}),
    ("I. iris-1423", "roman", 1, "echo", {}),
    ("I. iris-1423", "alpha_upper", 9, "echo", {}),
    ("C. iris-1423", "alpha_upper", 3, "echo", {}),
    ("C. iris-1423", "roman", 3, "form_only", {}),
    ("03. iris-1423", "control_arabic", 3, "form_only", {}),
    ("(03) iris-1423", "wrapper", 3, "form_only", {}),
    ("9. iris-1423", "roman", 9, "form_only", {}),
    ("iris-1423 9.", "control_arabic", 9, "form_only", {}),
    ("", "control_arabic", 9, "non_answer", {}),
    ("9.", "control_arabic", 9, "non_answer", {}),
    ("3", "control_arabic", 3, "non_answer", {}),
    ("iri-1423", "control_arabic", 9, "non_answer", {}),
    ("iris 1423", "control_arabic", 9, "non_answer", {}),
    ("iris-142", "control_arabic", 9, "non_answer", {}),
    ("iris-14230", "control_arabic", 9, "non_answer", {}),
    ("iris-1423\n", "control_arabic", 9, "non_answer", {}),
    ("iris-1423 ", "control_arabic", 9, "non_answer", {}),
    ("cedar-4159\n", "control_arabic", 9, "non_answer", {}),
    ("cedar-4159 ", "control_arabic", 9, "non_answer", {}),
    ("The iris-1423", "control_arabic", 9, "non_answer", {}),
    ("A alder-1234", "control_arabic", 1, "non_answer", {}),
    ("9.iris-1423", "control_arabic", 9, "non_answer", {}),
    ("9. 9. iris-1423", "control_arabic", 9, "non_answer", {}),
    ("[9] iris-1423", "control_arabic", 9, "form_only", {}),
    ("IIII. iris-1423", "roman", 4, "non_answer", {}),
    ("• iris-1423", "bullet", 9, "echo", {}),
    ("• iris-1423", "roman", 9, "form_only", {}),
    # A1: a wrong value wearing a marker counts toward J.
    ("IX. cedar-4159", "roman", 9, "wrong_value", {}),
    ("I. cedar-4159", "roman", 9, "wrong_value", {}),
    ("3. cedar-4159", "control_arabic", 3, "wrong_value", {}),
    ("• cedar-4159", "bullet", 9, "wrong_value", {}),
    ("(9) cedar-4159", "wrapper", 9, "wrong_value", {}),
    ("cedar-4159 3.", "control_arabic", 3, "wrong_value", {}),
    ("9. cedar-4159", "control_arabic", 9, "wrong_value", {}),
    ("[9] cedar-4159", "alpha_lower", 2, "wrong_value", {}),
    # A2: non-ASCII digit heads never parse, never raise.
    ("². cedar-4159", "control_arabic", 2, "non_answer", {}),
    ("٣. cedar-4159", "control_arabic", 3, "non_answer", {}),
    ("３. cedar-4159", "control_arabic", 3, "non_answer", {}),
    ("². iris-1423", "control_arabic", 2, "non_answer", {}),
]


class TestDecomposition(unittest.TestCase):
    """Section 5 as amended (A1-A3): the battery, the
    glyph machinery, and mirror agreement under fuzz."""

    EXPECTED = "iris-1423"

    def _classify(self, text, arm, target, **kwargs):
        """Classify via the producer."""
        cell = _cell(text, **kwargs)
        return ec.decompose_cell(cell, self.EXPECTED, ec.digest_text(self.EXPECTED), arm, target)

    def test_battery(self):
        """Every battery row, producer and mirror."""
        digest = ec.digest_text(self.EXPECTED)
        for text, arm, target, want, kwargs in _DECOMP_BATTERY:
            cell = _cell(text, **kwargs)
            self.assertEqual(
                ec.decompose_cell(cell, self.EXPECTED, digest, arm, target),
                want,
                (text, arm, target),
            )
            self.assertEqual(
                dr._enumerationconvention_decompose_cell(cell, self.EXPECTED, digest, arm, target),
                want,
                ("mirror", text, arm, target),
            )

    def test_glyph_machinery(self):
        """parse_roman / glyph_ok / split_token pins,
        ASCII-only."""
        self.assertEqual(ec.parse_roman("IX"), 9)
        self.assertEqual(ec.parse_roman("ix"), 9)
        self.assertEqual(ec.parse_roman("VIII"), 8)
        self.assertIsNone(ec.parse_roman("IIII"))
        self.assertIsNone(ec.parse_roman("VX"))
        self.assertIsNone(ec.parse_roman("Ix"))
        self.assertIsNone(ec.parse_roman(""))
        self.assertIsNone(ec.parse_roman("٣"))
        self.assertTrue(ec.glyph_ok("03"))
        self.assertFalse(ec.glyph_ok("003"))
        self.assertTrue(ec.glyph_ok("C"))
        self.assertFalse(ec.glyph_ok("AB"))
        self.assertFalse(ec.glyph_ok("²"))
        self.assertEqual(ec.split_token("(9)"), ("wrapper-parens", "9"))
        self.assertEqual(ec.split_token("IX."), ("dotted", "IX"))
        self.assertEqual(ec.split_token("3)"), ("close-paren", "3"))
        self.assertEqual(ec.split_token("[g]"), ("bracket", "g"))
        self.assertEqual(ec.split_token("•"), ("bullet", "•"))
        self.assertIsNone(ec.split_token("The"))
        self.assertIsNone(ec.split_token("A"))
        self.assertIsNone(ec.split_token("²."))
        self.assertIsNone(ec.split_token("٣."))
        self.assertEqual(ec.index_in_system("3", "arabic"), 3)
        self.assertIsNone(ec.index_in_system("٣", "arabic"))
        self.assertEqual(ec.index_in_system("C", "roman"), 100)
        self.assertEqual(ec.index_in_system("C", "alpha_upper"), 3)

    def test_strip_one(self):
        """Exactly one token, one separating space, at
        the named edge only."""
        self.assertEqual(ec.strip_one("9. iris-1423", "lead"), "iris-1423")
        self.assertEqual(ec.strip_one("iris-1423 9.", "trail"), "iris-1423")
        self.assertIsNone(ec.strip_one("iris-1423 9.", "lead"))
        self.assertIsNone(ec.strip_one("The iris-1423", "lead"))
        self.assertIsNone(ec.strip_one("iris-1423", "lead"))
        self.assertEqual(ec.strip_one("9. 9. iris-1423", "lead"), "9. iris-1423")

    def test_pair_partition(self):
        """The sibling pair partition is total."""
        same_a = _cell("iris-1423", verdict=True)
        same_b = _cell("iris-1423", verdict=True)
        self.assertEqual(ec.decompose_pair(same_a, same_b), "boundary_only")
        other = _cell("cedar-4159")
        self.assertEqual(ec.decompose_pair(same_a, other), "exact_only")
        no_boundary = _cell("cedar-4159", boundary=False)
        self.assertEqual(ec.decompose_pair(same_a, no_boundary), "mixed")

    def test_fuzz_totality_and_mirror_agreement(self):
        """20,000-case fuzz: exactly one class per cell,
        never a raise, producer == mirror (A1/A2 pins
        under fuzz, Unicode heads included)."""
        rng = random.Random(20261105)  # noqa: S311 -- deterministic test seed
        heads = [
            "9.",
            "IX.",
            "ix.",
            "I.",
            "C.",
            "(9)",
            "(03)",
            "03.",
            "•",
            "[9]",
            "3)",
            "².",
            "٣.",
            "３.",
            "IIII.",
            "The",
            "A",
            "x.",
            "M.",
            "",
        ]
        cores = [
            "iris-1423",
            "cedar-4159",
            "quill-9999",
            "marigold-9999",
            "iris-142",
            "unknownword",
            "iris-1423\n",
            "",
        ]
        digest = ec.digest_text(self.EXPECTED)
        for _ in range(20_000):
            head = rng.choice(heads)
            core = rng.choice(cores)
            text = f"{head} {core}" if head and core else head + core
            if rng.random() < 0.25:
                text = f"{core} {head}" if core and head else text
            arm = rng.choice(ec.ARM_KEYS)
            target = rng.randint(1, 9)
            cell = _cell(text)
            got = ec.decompose_cell(cell, self.EXPECTED, digest, arm, target)
            self.assertIn(got, ec.CELL_CLASSES)
            self.assertEqual(
                got,
                dr._enumerationconvention_decompose_cell(cell, self.EXPECTED, digest, arm, target),
            )

    def test_nonstring_and_missing_text(self):
        """Hostile cell shapes classify, never raise."""
        digest = ec.digest_text(self.EXPECTED)
        for cell in (
            {},
            {"verdict": False, "answer_digest": None, "answer_text": None},
            {"verdict": False, "answer_digest": "x", "answer_text": 7},
            {"verdict": False, "answer_digest": "x", "answer_text": ["iris-1423"]},
        ):
            self.assertEqual(
                ec.decompose_cell(cell, self.EXPECTED, digest, "roman", 9), "non_answer"
            )


def _oracle_classify(d, j, k):
    """The section-6 label rule restated from the design
    text (independent oracle for the guard tests)."""
    if k > 6:
        return "insufficient"
    if d < 10:
        return "insufficient"
    if j >= 6:
        return "discordant_asymmetric"
    if j == 0:
        return "no_discordance_observed" if k == 0 else "insufficient"
    return "below_threshold" if j + k < 6 else "insufficient"


class TestClassifyGuard(unittest.TestCase):
    """The guarded label: oracle grid, the exhaustive
    2,925-state audit, and the 264 cap-suppressed fires."""

    def test_oracle_grid(self):
        """Producer and mirror match the oracle on every
        reachable state."""
        for d in range(25):
            for j in range(25):
                for k in range(8):
                    if d + k <= 24:
                        want = _oracle_classify(d, j, k)
                        self.assertEqual(ec.classify_headline(d, j, k), want)
                        self.assertEqual(dr._enumerationconvention_classify(d, j, k), want)

    def test_exhaustive_audit(self):
        """No manufactured fires, no unstable acquittals,
        exactly 264 cap-suppressed stable fires."""
        states = 0
        cap_suppressed = 0
        for d in range(25):
            for j in range(d + 1):
                for k in range(24 - d + 1):
                    states += 1
                    label = ec.classify_headline(d, j, k)
                    if label == "discordant_asymmetric":
                        self.assertGreaterEqual(d, 10)
                        self.assertGreaterEqual(j, 6)
                    if label in ("no_discordance_observed", "below_threshold"):
                        self.assertEqual(ec.classify_headline(d + k, j + k, 0), label)
                    if k > 6 and d >= 10 and j >= 6:
                        cap_suppressed += 1
        self.assertEqual(states, 2925)
        self.assertEqual(cap_suppressed, 264)

    def test_label_vocabulary(self):
        """Only the four frozen labels ever issue."""
        seen = {
            ec.classify_headline(d, j, k) for d in range(25) for j in range(25) for k in range(10)
        }
        self.assertEqual(seen, set(ec.LABELS))


def _world(pc, q, remove, trials=10000, seed=20261030, k_scale=1.0):
    """The attack-world simulation, ported from the
    design evidence: per case a J cell (control pass +
    arm wrong value), a D cell (control pass + arm
    other), or a control fail; the removal rule deletes
    cells into k; labels come from the module's guard."""
    rng = random.Random(seed)  # noqa: S311 -- deterministic test seed
    out: dict[str, dict[str, int]] = {"guarded": {}, "unguarded": {}}
    for _ in range(trials):
        cells = []
        for _ in range(24):
            if rng.random() < pc:
                cells.append("J" if rng.random() < q else "D")
            else:
                cells.append(".")
        kept, k = [], 0
        for c in cells:
            if remove(c, rng):
                k += 1
            else:
                kept.append(c)
        d_obs = sum(c != "." for c in kept)
        j_obs = sum(c == "J" for c in kept)
        kk = int(round(k * k_scale))
        for key, label in (
            ("unguarded", ec.classify_headline(d_obs, j_obs, 0)),
            ("guarded", ec.classify_headline(d_obs, j_obs, kk)),
        ):
            out[key][label] = out[key].get(label, 0) + 1
    return {
        key: {label: round(n / trials, 4) for label, n in sorted(v.items())}
        for key, v in out.items()
    }


class TestGuardWorlds(unittest.TestCase):
    """Worlds A1/A2/B/C at the pinned seed reproduce the
    frozen operating characteristics exactly; the
    disjoint-seed replication and the k-misexclusion
    pricing (A9's ids-dimension battery) hold."""

    def test_worlds_at_pinned_seed(self):
        """Designer-seed worlds == OPERATING_CHARACTERISTICS."""
        oc = ec.OPERATING_CHARACTERISTICS
        a1 = _world(0.90, 0.10, lambda c, r: c == "D" and r.random() < 0.5)
        self.assertEqual(a1["guarded"]["discordant_asymmetric"], oc["world_a1_guarded_fire"])
        self.assertEqual(a1["unguarded"]["discordant_asymmetric"], oc["world_a1_unguarded_fire"])
        self.assertEqual(a1["guarded"]["insufficient"], oc["world_a1_guarded_insufficient"])
        a2 = _world(0.90, 0.10, lambda c, r: c == "." and r.random() < 0.8)
        self.assertEqual(a2["guarded"]["discordant_asymmetric"], oc["world_a2_guarded_fire"])
        self.assertEqual(a2["unguarded"]["discordant_asymmetric"], oc["world_a2_unguarded_fire"])
        self.assertEqual(a2["guarded"]["insufficient"], oc["world_a2_guarded_insufficient"])
        b = _world(0.90, 0.50, lambda c, r: c == "J" and r.random() < 0.6)
        self.assertEqual(b["unguarded"]["discordant_asymmetric"], oc["world_b_unguarded_fire"])
        self.assertEqual(b["unguarded"]["below_threshold"], oc["world_b_unguarded_below_threshold"])
        self.assertEqual(b["guarded"]["discordant_asymmetric"], oc["world_b_guarded_fire"])
        self.assertEqual(b["guarded"]["below_threshold"], oc["world_b_guarded_below_threshold"])
        self.assertEqual(b["guarded"]["insufficient"], oc["world_b_guarded_insufficient"])
        b777 = _world(0.90, 0.50, lambda c, r: c == "J" and r.random() < 0.6, seed=777)
        self.assertEqual(
            b777["guarded"]["below_threshold"], oc["world_b777_guarded_below_threshold"]
        )
        c = _world(0.90, 0.50, lambda c, r: r.random() < 0.15)
        self.assertEqual(c["guarded"]["discordant_asymmetric"], oc["world_c_guarded_fire"])
        self.assertEqual(c["unguarded"]["discordant_asymmetric"], oc["world_c_unguarded_fire"])
        self.assertEqual(c["guarded"]["below_threshold"], oc["world_c_guarded_below_threshold"])

    def test_worlds_at_disjoint_seed(self):
        """Red team A's disjoint-seed replication and its
        stronger worlds D/E/F."""
        seed = 987654321
        a1 = _world(0.90, 0.10, lambda c, r: c == "D" and r.random() < 0.5, seed=seed)
        self.assertEqual(a1["guarded"]["discordant_asymmetric"], 0.0043)
        a2 = _world(0.90, 0.10, lambda c, r: c == "." and r.random() < 0.8, seed=seed)
        self.assertEqual(a2["guarded"]["discordant_asymmetric"], 0.0192)
        self.assertEqual(a2["unguarded"]["discordant_asymmetric"], 0.0192)
        b = _world(0.90, 0.50, lambda c, r: c == "J" and r.random() < 0.6, seed=seed)
        self.assertEqual(b["guarded"]["below_threshold"], 0.0123)
        self.assertEqual(b["unguarded"]["below_threshold"], 0.7308)
        c = _world(0.90, 0.50, lambda c, r: r.random() < 0.15, seed=seed)
        self.assertEqual(c["guarded"]["discordant_asymmetric"], 0.8917)
        d = _world(0.90, 0.50, lambda c, r: c == "J" and r.random() < 0.9, seed=seed)
        self.assertEqual(d["guarded"]["below_threshold"], 0.0051)
        e = _world(0.90, 0.10, lambda c, r: c == "D" and r.random() < 0.9, seed=seed)
        self.assertEqual(e["guarded"].get("discordant_asymmetric", 0.0), 0.0)
        f = _world(0.90, 0.50, lambda c, r: r.random() < 0.30, seed=seed)
        self.assertEqual(f["guarded"]["below_threshold"], 0.0001)

    def test_k_misexclusion_pricing(self):
        """Wrongly excluding a fraction of the domain's
        incompletes from k degrades world B exactly as
        priced -- the battery is load-bearing."""
        seed = 987654321
        for fraction, want in ((0.25, 0.0383), (0.5, 0.1341)):
            result = _world(
                0.90,
                0.50,
                lambda c, r: c == "J" and r.random() < 0.6,
                seed=seed,
                k_scale=1.0 - fraction,
            )
            self.assertEqual(result["guarded"]["below_threshold"], want)

    def test_ids_dimension_at_aggregate(self):
        """The aggregate's k ranges over the id-divergent
        domain only: an ids-identical record's missing
        cell never enters n_incomplete; a divergent
        record's does."""
        cases = ec.generate_cases(0)
        case = cases[0]

        def record(divergent: bool, missing: bool):
            cells, details, tokens, digests = {}, {}, {}, {}
            for arm in ec.ARM_KEYS:
                cells[arm] = {
                    "verdict": True,
                    "boundary_found": True,
                    "boundary_text_match": True,
                    "input_tokens": 50,
                    "answer_digest": ec.digest_text(case["expected"]),
                    "answer_text": case["expected"],
                }
                details[arm] = ""
                tokens[arm] = 50
                digests[arm] = "same" if not divergent else f"digest-{arm}"
            if missing:
                cells["roman"] = None
                details["roman"] = "arm_evaluation_failed"
            return {
                "fixture": case["fixture_id"],
                "topic": case["topic"],
                "target_position": case["position"],
                "expected_digest": ec.digest_text(case["expected"]),
                "content_digests": {},
                "arm_tokens": tokens,
                "id_digests": digests,
                "ids_identical": {},
                "id_length_delta": {},
                "cells": cells,
                "cell_details": details,
            }

        agg = ec.aggregate_candidate(
            [record(divergent=False, missing=True), record(divergent=True, missing=True)],
            cases,
        )
        self.assertEqual(agg["headline"]["roman"]["n_incomplete"], 1)
        self.assertEqual(agg["headline"]["roman"]["n_ids_divergent"], 1)
        self.assertEqual(agg["headline"]["roman"]["n_population"], 0)


# ------------------------------------------------------------------
# Exact operating characteristics (A9): independent-column
# figures by Fraction enumeration; the paired column by the
# exact trinomial sum over a Simpson-quadrature Phi2.
# ------------------------------------------------------------------


def _binom_pmf(n, p):
    """Binomial pmf (exact)."""
    return [Fraction(math.comb(n, k)) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]


def _fire_exact(n, pc, q, min_d, min_j):
    """Exact independent-column fire probability."""
    total = Fraction(0)
    dpmf = _binom_pmf(n, pc)
    for d in range(min_d, n + 1):
        if dpmf[d]:
            total += dpmf[d] * sum(_binom_pmf(d, q)[min_j:], Fraction(0))
    return total


def _phi2(h, k, rho, lo=-10.0, steps=4000):
    """Bivariate-normal CDF by Simpson quadrature."""
    s = math.sqrt(1 - rho * rho)
    n = steps if steps % 2 == 0 else steps + 1
    dx = (h - lo) / n
    dist = NormalDist()
    total = 0.0
    for i in range(n + 1):
        x = lo + i * dx
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        total += w * math.exp(-x * x / 2) / math.sqrt(2 * math.pi) * dist.cdf((k - rho * x) / s)
    return total * dx / 3


def _fire_paired(pc, pa, rho=0.5, n=24, min_d=10, min_j=6):
    """Exact paired-column fire probability (trinomial
    sum over the Gaussian-copula per-case cells)."""
    dist = NormalDist()
    both = _phi2(dist.inv_cdf(pc), dist.inv_cdf(pa), rho)
    p_j, p_b, p_c = pc - both, both, 1 - pc
    total = 0.0
    for j in range(min_j, n + 1):
        for b in range(max(0, min_d - j), n + 1 - j):
            c = n - j - b
            total += math.comb(n, j) * math.comb(n - j, b) * p_j**j * p_b**b * p_c**c
    return total


class TestOperatingCharacteristics(unittest.TestCase):
    """Every section-6 figure reproduced from this
    test's own code (A9), plus the floors frontier, the
    plateau, and the corrected required-N (A6)."""

    PC = Fraction(90, 100)
    Q = Fraction(10, 100)

    def test_independent_column_exact(self):
        """The independent figures at 4dp, from exact
        Fractions, against the module's frozen table."""
        oc = ec.OPERATING_CHARACTERISTICS
        rows = {
            "fire_competent_independent": (self.PC, Fraction(10, 100)),
            "fire_weak_independent": (Fraction(80, 100), Fraction(20, 100)),
            "fire_random_independent": (Fraction(50, 100), Fraction(50, 100)),
            "power_drop5_base90": (self.PC, Fraction(15, 100)),
            "power_drop10_base90": (self.PC, Fraction(20, 100)),
            "power_drop25_base90": (self.PC, Fraction(35, 100)),
            "power_drop25_base85": (Fraction(85, 100), Fraction(40, 100)),
            "power_drop50_base90": (self.PC, Fraction(60, 100)),
            "transplant_fire_competent_independent": (self.PC, Fraction(10, 100)),
        }
        for key, (pc, q) in rows.items():
            if key.startswith("transplant"):
                continue
            self.assertEqual(round(float(_fire_exact(24, pc, q, 10, 6)), 4), oc[key], key)
        # Sensitivity rows: same floors, reduced paired count.
        self.assertEqual(
            round(float(_fire_exact(18, self.PC, Fraction(35, 100), 10, 6)), 4),
            oc["sensitivity18_drop25"],
        )
        self.assertEqual(
            round(float(_fire_exact(18, self.PC, self.Q, 10, 6)), 4),
            oc["sensitivity18_competent"],
        )
        self.assertEqual(
            round(float(_fire_exact(12, self.PC, Fraction(35, 100), 10, 6)), 4),
            oc["sensitivity12_drop25"],
        )
        self.assertEqual(
            round(float(_fire_exact(12, self.PC, self.Q, 10, 6)), 4),
            oc["sensitivity12_competent"],
        )

    def test_transplant_rule(self):
        """The rejected two-directional transplant rule
        (independent control/arm outcomes per case; fire
        iff either discordant count reaches 2, balanced
        included) fires 0.8897 on the competent null --
        the exact dynamic program, as in the sibling
        build's reproduction."""
        pc = pa = 0.90
        probs = (pc * pa, pc * (1 - pa), (1 - pc) * pa, (1 - pc) * (1 - pa))
        dp: dict[tuple[int, int], float] = {(0, 0): 1.0}
        for _ in range(24):
            nxt: dict[tuple[int, int], float] = {}
            for (c, a), w in dp.items():
                for dc, da, pi in (
                    (0, 0, probs[0]),
                    (1, 0, probs[1]),
                    (0, 1, probs[2]),
                    (0, 0, probs[3]),
                ):
                    key = (c + dc, a + da)
                    nxt[key] = nxt.get(key, 0.0) + w * pi
            dp = nxt
        fire = sum(w for (c, a), w in dp.items() if (c == a and c >= 2) or max(c, a) >= 2)
        self.assertEqual(
            round(fire, 4),
            ec.OPERATING_CHARACTERISTICS["transplant_fire_competent_independent"],
        )

    def test_paired_column_exact(self):
        """The paired figures (rho = 0.5) reproduce."""
        oc = ec.OPERATING_CHARACTERISTICS
        self.assertEqual(round(_fire_paired(0.90, 0.90), 4), oc["fire_competent_paired"])
        self.assertEqual(round(_fire_paired(0.90, 0.65), 4), oc["power_drop25_paired"])

    def test_familywise_rows(self):
        """Family-wise over 120 comparisons, from the
        unrounded exact rates."""
        oc = ec.OPERATING_CHARACTERISTICS
        p = float(_fire_exact(24, self.PC, self.Q, 10, 6))
        self.assertEqual(round(120 * p, 4), oc["familywise_competent_expected_120"])
        self.assertEqual(round(1 - (1 - p) ** 120, 4), oc["familywise_competent_any_120"])
        p_rand = float(_fire_exact(24, Fraction(50, 100), Fraction(50, 100), 10, 6))
        self.assertEqual(round(120 * p_rand, 4), oc["familywise_random_expected_120"])
        self.assertEqual(round(1 - (1 - p_rand) ** 120, 4), oc["familywise_random_any_120"])
        p_paired = _fire_paired(0.90, 0.90)
        self.assertEqual(round(120 * p_paired, 4), oc["familywise_paired_expected_120"])
        self.assertEqual(round(1 - (1 - p_paired) ** 120, 4), oc["familywise_paired_any_120"])

    def test_floors_frontier_and_plateau(self):
        """Frontier exactly (1..6, 6); printed plateau
        [1, 14], exact [1, 6]; the power-side gap."""
        q35 = Fraction(35, 100)
        best = Fraction(0)
        pairs = []
        for min_d in range(1, 17):
            for min_j in range(2, 9):
                if _fire_exact(24, self.PC, self.Q, min_d, min_j) <= Fraction(2, 100):
                    power = _fire_exact(24, self.PC, q35, min_d, min_j)
                    if power > best:
                        best, pairs = power, [(min_d, min_j)]
                    elif power == best:
                        pairs.append((min_d, min_j))
        self.assertEqual(sorted(pairs), [(d, 6) for d in range(1, 7)])
        printed = [
            md
            for md in range(1, 20)
            if round(float(_fire_exact(24, self.PC, self.Q, md, 6)), 4)
            == round(float(_fire_exact(24, self.PC, self.Q, 10, 6)), 4)
            and round(float(_fire_exact(24, self.PC, q35, md, 6)), 4)
            == round(float(_fire_exact(24, self.PC, q35, 10, 6)), 4)
        ]
        self.assertEqual(printed, list(range(1, 15)))
        exact = [
            md
            for md in range(1, 20)
            if _fire_exact(24, self.PC, self.Q, md, 6) == _fire_exact(24, self.PC, self.Q, 1, 6)
            and _fire_exact(24, self.PC, q35, md, 6) == _fire_exact(24, self.PC, q35, 1, 6)
        ]
        self.assertEqual(exact, list(range(1, 7)))
        gap = float(best - _fire_exact(24, self.PC, q35, 10, 6))
        self.assertAlmostEqual(gap, 2.796189e-11, delta=1e-13)

    def test_required_n_corrected(self):
        """A6: smallest N is 117 (floors (0, 18) in the
        tie set), verified by exact Fractions at 117 and
        by the float scan's grid values."""
        oc = ec.OPERATING_CHARACTERISTICS
        self.assertEqual(oc["required_n_smallest"], 117)
        null117 = _fire_exact(117, self.PC, self.Q, 0, 18)
        power117 = _fire_exact(117, self.PC, Fraction(20, 100), 0, 18)
        self.assertEqual(round(float(null117), 4), oc["required_n_null"])
        self.assertEqual(round(float(power117), 4), oc["required_n_power10"])
        self.assertEqual(oc["required_n_cost"], 117 * 6 * 24)

        def best_power10(n):
            mj_cap = max(24, n // 6)

            def all_fires(pc, q):
                dpmf = [math.comb(n, k) * pc**k * (1 - pc) ** (n - k) for k in range(n + 1)]
                tails = {}
                for d in range(n + 1):
                    jpmf = [math.comb(d, k) * q**k * (1 - q) ** (d - k) for k in range(d + 1)]
                    tails[d] = {mj: sum(jpmf[mj:]) for mj in range(2, mj_cap + 1)}
                out = {}
                for mj in range(2, mj_cap + 1):
                    run = 0.0
                    for d in range(n, -1, -1):
                        run += dpmf[d] * tails[d][mj]
                        out[(d, mj)] = run
                return out

            nulls = all_fires(0.90, 0.10)
            powers = all_fires(0.90, 0.20)
            return max(
                powers[(md, mj)]
                for md in range(0, n + 1)
                for mj in range(2, mj_cap + 1)
                if nulls[(md, mj)] <= 0.02
            )

        self.assertLess(best_power10(116), 0.80)
        self.assertEqual(round(best_power10(120), 4), oc["required_n_grid120_power10"])
        self.assertEqual(round(best_power10(126), 4), oc["required_n_grid126_power10"])
        self.assertEqual(round(best_power10(132), 4), oc["required_n_grid132_power10"])


class TestStagePayloads(unittest.TestCase):
    """The stage end to end through scripted backends."""

    def test_perfect_payload(self):
        """Perfect backend: full rates, acquittal labels,
        mirror-valid."""
        payload, _backend, stage = _payload("perfect")
        self.assertTrue(stage["available"])
        self.assertTrue(payload["available"])
        self.assertEqual(payload["n_cases"], 24)
        self.assertEqual(payload["n_candidates"], 2)
        for entry in payload["candidates"].values():
            for arm in ec.ARM_KEYS:
                self.assertEqual(entry["arms"][arm]["rate"], 1.0)
                self.assertEqual(entry["arms"][arm]["decomposition"]["pass"], 24)
            for arm in ec.TREATED_ARMS:
                head = entry["headline"][arm]
                self.assertEqual(head["label"], "no_discordance_observed")
                self.assertEqual(head["n_ids_divergent"], 24)
                self.assertEqual(head["control_passes"], 24)
        self.assertTrue(dr._valid_enumeration_convention(payload))

    def test_markerwrong_fires_via_a1(self):
        """Marker-prefixed wrong values fire every treated
        arm (the A1 behaviour, end to end)."""
        payload, _backend, _stage = _payload("markerwrong")
        for entry in payload["candidates"].values():
            for arm in ec.TREATED_ARMS:
                head = entry["headline"][arm]
                self.assertEqual(head["label"], "discordant_asymmetric", arm)
                self.assertEqual(head["wrong_values"], 24)
                self.assertEqual(entry["arms"][arm]["decomposition"]["wrong_value"], 24)
            self.assertEqual(entry["arms"]["control_arabic"]["rate"], 1.0)
        self.assertTrue(dr._valid_enumeration_convention(payload))

    def test_wrongvalue_fires(self):
        """Bare sibling wrong values fire too."""
        payload, _backend, _stage = _payload("wrongvalue")
        entry = payload["candidates"]["chatml"]
        for arm in ec.TREATED_ARMS:
            self.assertEqual(entry["headline"][arm]["label"], "discordant_asymmetric")

    def test_echo_mode_decomposition(self):
        """Echo answers: echo class per arm, no fire."""
        payload, _backend, _stage = _payload("echo")
        entry = payload["candidates"]["chatml"]
        for arm in ec.ARM_KEYS:
            self.assertEqual(entry["arms"][arm]["decomposition"]["echo"], 24, arm)
            self.assertEqual(entry["arms"][arm]["rate"], 0.0)
        for arm in ec.TREATED_ARMS:
            self.assertEqual(entry["headline"][arm]["wrong_values"], 0)

    def test_garbage_and_noboundary(self):
        """Garbage reads non_answer; with no boundary
        token anywhere, evaluate takes the whole padded
        output as the answer, so no-boundary cells also
        land in non_answer (the sibling suites'
        observation); true boundary_only cells are
        exercised at the aggregate level below."""
        payload, _backend, _stage = _payload("garbage")
        entry = payload["candidates"]["chatml"]
        for arm in ec.ARM_KEYS:
            self.assertEqual(entry["arms"][arm]["decomposition"]["non_answer"], 24)
        payload, _backend, _stage = _payload("noboundary")
        entry = payload["candidates"]["chatml"]
        for arm in ec.ARM_KEYS:
            self.assertEqual(entry["arms"][arm]["decomposition"]["non_answer"], 24)
            self.assertEqual(entry["arms"][arm]["rate"], 0.0)
        for arm in ec.TREATED_ARMS:
            self.assertEqual(entry["headline"][arm]["label"], "insufficient")

    def test_boundary_only_at_aggregate_level(self):
        """Hand-built cells whose digest matches but
        whose boundary flag is False decompose as
        boundary_only in both aggregates."""
        cases = ec.generate_cases(0)
        records = []
        for case in cases:
            cells, details, tokens, digests = {}, {}, {}, {}
            for arm in ec.ARM_KEYS:
                cells[arm] = {
                    "verdict": False,
                    "boundary_found": False,
                    "boundary_text_match": False,
                    "input_tokens": 50,
                    "answer_digest": ec.digest_text(case["expected"]),
                    "answer_text": case["expected"],
                }
                details[arm] = ""
                tokens[arm] = 50
                digests[arm] = f"d-{arm}"
            records.append(
                {
                    "fixture": case["fixture_id"],
                    "topic": case["topic"],
                    "target_position": case["position"],
                    "expected_digest": ec.digest_text(case["expected"]),
                    "content_digests": {},
                    "arm_tokens": tokens,
                    "id_digests": digests,
                    "ids_identical": {},
                    "id_length_delta": {},
                    "cells": cells,
                    "cell_details": details,
                }
            )
        agg = ec.aggregate_candidate(records, cases)
        for arm in ec.ARM_KEYS:
            self.assertEqual(agg["arms"][arm]["decomposition"]["boundary_only"], 24)
        self.assertEqual(agg, dr._enumerationconvention_aggregate(records, cases))

    def test_ids_identical_regime(self):
        """Marker-stripping encoder: every arm
        ids-identical, M = 0, labels withheld."""
        payload, _backend, _stage = _payload("perfect", strip_markers=True)
        entry = payload["candidates"]["chatml"]
        for arm in ec.TREATED_ARMS:
            self.assertEqual(entry["ids"][arm]["n_ids_identical"], 24)
            self.assertEqual(entry["headline"][arm]["n_ids_divergent"], 0)
            self.assertEqual(entry["headline"][arm]["label"], "insufficient")
        self.assertTrue(dr._valid_enumeration_convention(payload))

    def test_explode_mode(self):
        """A raising backend yields failed cells, not a
        failed stage."""
        payload, _backend, stage = _payload("explode")
        self.assertTrue(stage["available"])
        entry = payload["candidates"]["chatml"]
        for arm in ec.ARM_KEYS:
            self.assertEqual(entry["arms"][arm]["n_evaluation_failed"], 24)
            self.assertIsNone(entry["arms"][arm]["rate"])

    def test_cancel(self):
        """A set cancel event unwinds to the fail-closed
        cancelled payload."""
        import threading

        cancel = threading.Event()
        cancel.set()
        payload, _backend, stage = _payload("perfect", cancel=cancel)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")

    def test_format_validation(self):
        """Bad format tuples raise at the stage boundary;
        over-budget format counts degrade fail-closed."""
        backend = _ListBackend("perfect")
        with self.assertRaises(ValueError):
            experiments.enumeration_convention_stage(
                backend,
                [_CHATML],  # type: ignore[arg-type]
                boundary=(),
                max_context=1000,
            )
        with self.assertRaises(ValueError):
            experiments.enumeration_convention_stage(
                backend,
                (_CHATML, _CHATML),
                boundary=(),
                max_context=1000,
            )
        formats = tuple(Format(f"f{i}", "", "u", "a", "e") for i in range(ec.MAX_CANDIDATES + 1))
        stage = experiments.enumeration_convention_stage(
            backend,
            formats,
            boundary=backend.encode("endmark"),
            max_context=100_000,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "enumeration_convention_record_budget_exceeded")

    def test_byte_cap_degrades(self):
        """The serialised-records cap degrades the whole
        stage to the budget detail."""
        with mock.patch.object(ec, "MAX_RECORDS_BYTES", 10):
            _payload_obj, _backend, stage = _payload("perfect")
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "enumeration_convention_record_budget_exceeded")

    def test_context_exceeded_cells(self):
        """A tiny context budget marks every cell
        arm_context_exceeded."""
        payload, _backend, _stage = _payload("perfect", max_context=10)
        entry = payload["candidates"]["chatml"]
        for arm in ec.ARM_KEYS:
            self.assertEqual(entry["arms"][arm]["n_context_exceeded"], 24)

    def test_screen_failure_on_exhaustion(self):
        """Generator exhaustion inside the stage fails
        closed with the screen detail."""
        with mock.patch.object(ec, "_derive", lambda *args: 0):
            _payload_obj, _backend, stage = _payload("perfect")
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "enumeration_convention_screen_failed")

    def test_runner_fallback(self):
        """The CLI runner never raises: a stage explosion
        lands the fail-closed probe payload."""
        from model_protocol_lab import cli

        result: dict = {}
        with mock.patch.object(
            experiments, "enumeration_convention_stage", side_effect=RuntimeError("x")
        ):
            cli._run_enumeration_convention_probe(
                _ListBackend("perfect"),
                (_CHATML,),
                result,
                boundary=(),
                boundary_map=None,
                max_new_tokens=8,
                max_context=100,
                seed=0,
                cancel=None,
            )
        self.assertEqual(
            result["enumeration_convention"],
            {"available": False, "detail": "enumeration_convention_unavailable"},
        )


class TestAggregatePositions(unittest.TestCase):
    """Amendment A5 at the aggregate level: the
    by-position join carries the mean id-length delta
    beside the by-position D/J over the paired
    population."""

    def test_positions_join(self):
        """Position rows sum to the paired population and
        the mean delta is the paired records' mean."""
        cases = ec.generate_cases(0)
        records = []
        for i, case in enumerate(cases):
            cells, details, tokens, digests = {}, {}, {}, {}
            for arm in ec.ARM_KEYS:
                cells[arm] = {
                    "verdict": True,
                    "boundary_found": True,
                    "boundary_text_match": True,
                    "input_tokens": 100,
                    "answer_digest": ec.digest_text(case["expected"]),
                    "answer_text": case["expected"],
                }
                details[arm] = ""
                tokens[arm] = 100 + (i if arm == "roman" else 0)
                digests[arm] = f"d-{arm}-{i}"
            records.append(
                {
                    "fixture": case["fixture_id"],
                    "topic": case["topic"],
                    "target_position": case["position"],
                    "expected_digest": ec.digest_text(case["expected"]),
                    "content_digests": {},
                    "arm_tokens": tokens,
                    "id_digests": digests,
                    "ids_identical": {},
                    "id_length_delta": {},
                    "cells": cells,
                    "cell_details": details,
                }
            )
        agg = ec.aggregate_candidate(records, cases)
        rows = agg["positions"]["roman"]
        self.assertEqual([row["position"] for row in rows], list(range(1, 10)))
        self.assertEqual(sum(row["n_population"] for row in rows), 24)
        self.assertEqual(sum(row["control_passes"] for row in rows), 24)
        for row in rows:
            expected_deltas = [
                i for i, case in enumerate(cases) if case["position"] == row["position"]
            ]
            want = sum(expected_deltas) / len(expected_deltas) if expected_deltas else None
            self.assertEqual(row["mean_id_length_delta"], want)
        self.assertEqual(
            agg["positions"], dr._enumerationconvention_aggregate(records, cases)["positions"]
        )


class TestMirrorValidation(unittest.TestCase):
    """The diffreports mirror: valid payloads validate;
    single-point tampers never do."""

    def test_valid_payloads(self):
        """Perfect / markerwrong / ids-identical payloads
        all validate."""
        for mode, strip in (("perfect", False), ("markerwrong", False), ("perfect", True)):
            payload, _backend, _stage = _payload(mode, strip_markers=strip)
            self.assertTrue(dr._valid_enumeration_convention(payload), mode)

    def test_tamper_table(self):
        """Each tamper is rejected."""
        base = _valid_payload()

        def tampered(mutator):
            payload = copy.deepcopy(base)
            mutator(payload)
            return payload

        tampers = {
            "headline label": lambda p: p["candidates"]["chatml"]["headline"]["roman"].update(
                label="discordant_asymmetric"
            ),
            "wrong_values": lambda p: p["candidates"]["chatml"]["headline"]["roman"].update(
                wrong_values=99
            ),
            "positions row": lambda p: p["candidates"]["chatml"]["positions"]["roman"][0].update(
                control_passes=99
            ),
            "oc figure": lambda p: p["candidates"]["chatml"]["operating_characteristics"].update(
                fire_competent_independent=0.5
            ),
            "protocol seed": lambda p: p["protocol"].update(seed=9),
            "n_cases": lambda p: p.update(n_cases=23),
            "record topic": lambda p: p["candidates"]["chatml"]["records"][0].update(
                topic="summit"
            ),
            "record position": lambda p: p["candidates"]["chatml"]["records"][0].update(
                target_position=1
                if p["candidates"]["chatml"]["records"][0]["target_position"] != 1
                else 2
            ),
            "content digest": lambda p: p["candidates"]["chatml"]["records"][0][
                "content_digests"
            ].update(roman="0" * 64),
            "answer text": lambda p: p["candidates"]["chatml"]["records"][1]["cells"][
                "control_arabic"
            ].update(answer_text="cedar-4159"),
            "verdict flip": lambda p: p["candidates"]["chatml"]["records"][1]["cells"][
                "control_arabic"
            ].update(verdict=False),
            "ids flag": lambda p: p["candidates"]["chatml"]["records"][0]["ids_identical"].update(
                roman=True
            ),
            "id delta": lambda p: p["candidates"]["chatml"]["records"][0]["id_length_delta"].update(
                roman=7
            ),
            "arm tokens": lambda p: p["candidates"]["chatml"]["records"][0]["arm_tokens"].update(
                roman=12345
            ),
            "arm tokens bool": lambda p: p["candidates"]["chatml"]["records"][0][
                "arm_tokens"
            ].update(roman=True),
            "audit extra": lambda p: p["audit"].update(extra=1),
            "record extra key": lambda p: p["candidates"]["chatml"]["records"][0].update(extra=1),
        }
        for name, mutate in tampers.items():
            self.assertFalse(dr._valid_enumeration_convention(tampered(mutate)), name)

    def test_invalid_shapes(self):
        """Non-payloads and unavailable payloads are not
        valid."""
        self.assertFalse(dr._valid_enumeration_convention(None))
        self.assertFalse(dr._valid_enumeration_convention({}))
        self.assertFalse(dr._valid_enumeration_convention({"available": False, "detail": "x"}))


class TestDiff(unittest.TestCase):
    """diff_enumeration_convention: stability, drift,
    comparability."""

    def test_self_diff_unchanged(self):
        """A payload diffed against itself is unchanged
        and comparable."""
        payload = _valid_payload()
        result = dr.diff_enumeration_convention(payload, payload)
        self.assertEqual(result["available"], "both")
        self.assertFalse(result["changed"])
        self.assertTrue(result["comparable"])
        self.assertEqual(result["candidate_drift"], [])

    def test_mode_diff_changed(self):
        """Perfect vs markerwrong: label drift entries."""
        perfect = _valid_payload()
        markerwrong, _backend, _stage = _payload("markerwrong")
        result = dr.diff_enumeration_convention(perfect, markerwrong)
        self.assertTrue(result["changed"])
        metrics = {entry["metric"] for entry in result["candidate_drift"]}
        self.assertIn("roman_label", metrics)
        self.assertIn("roman_wrong_values", metrics)

    def test_seed_drift_not_comparable(self):
        """A different seed is protocol + generator
        drift: changed, not comparable."""
        perfect = _valid_payload()
        other, _backend, _stage = _payload("perfect", seed=7)
        result = dr.diff_enumeration_convention(perfect, other)
        self.assertTrue(result["protocol_drift"])
        self.assertTrue(result["generator_drift"])
        self.assertFalse(result["comparable"])
        self.assertTrue(result["changed"])

    def test_availability_shapes(self):
        """One-sided and absent payloads."""
        payload = _valid_payload()
        self.assertEqual(
            dr.diff_enumeration_convention(None, None),
            {"available": "neither", "changed": False},
        )
        self.assertEqual(dr.diff_enumeration_convention(payload, None)["available"], "old")
        self.assertEqual(dr.diff_enumeration_convention(None, payload)["available"], "new")


class TestReportingSection(unittest.TestCase):
    """The reporting section: renders the readout,
    leaks no answer text, absent without a payload, and
    the TOC/body correspondence holds."""

    def test_section_renders(self):
        """Markerwrong payload: labels and the position
        join render; no expected value leaks."""
        payload, _backend, _stage = _payload("markerwrong")
        section = reporting._enumeration_convention_section({"enumeration_convention": payload})
        self.assertIn('<section class="card" id="enumeration-convention">', section)
        self.assertIn("discordant_asymmetric", section)
        self.assertIn("By target position", section)
        self.assertIn("120 labelled", section)
        for case in ec.generate_cases(0):
            self.assertNotIn(case["expected"], section)

    def test_section_absent(self):
        """No section without an available payload."""
        self.assertEqual(reporting._enumeration_convention_section({}), "")
        self.assertEqual(
            reporting._enumeration_convention_section(
                {"enumeration_convention": {"available": False, "detail": "x"}}
            ),
            "",
        )

    def test_toc_body_correspondence(self):
        """TOC/body: the rendered report links the anchor
        and carries the section after the nav."""
        payload, _backend, _stage = _payload("perfect")
        html = reporting.render_html({"enumeration_convention": payload})
        self.assertIn('href="#enumeration-convention"', html)
        self.assertIn('<section class="card" id="enumeration-convention">', html)
        self.assertGreater(html.index('id="enumeration-convention"'), html.index("</nav>"))

    def test_section_loose_candidate_shapes(self):
        """Loose candidate shapes degrade, never crash."""
        payload, _backend, _stage = _payload("perfect")
        broken = copy.deepcopy(payload)
        broken["candidates"]["chatml"] = {"status": "available", "detail": ""}
        section = reporting._enumeration_convention_section({"enumeration_convention": broken})
        self.assertIn('id="enumeration-convention"', section)
        broken["candidates"]["chatml"] = "junk"
        section = reporting._enumeration_convention_section({"enumeration_convention": broken})
        self.assertIn('id="enumeration-convention"', section)


def _gap_digest(text):
    """Gap battery digest helper."""
    return sha256(text.encode("utf-8")).hexdigest()


class TestGapBattery(unittest.TestCase):
    """Cycle-174 gap pass: every reachable line the main
    battery left uncovered in the new code, driven by the
    input that reaches it (validator rejection rows, guard
    rows, hostile aggregate records, stage degradation
    branches, mirror rejection rows)."""

    def test_marker_for_and_unknown_arm_raises(self):
        """marker_for returns the frozen table entry;
        list_block / arm_content reject an unknown arm."""
        for arm in ec.ARM_KEYS:
            for t in ec.INDICES:
                self.assertEqual(ec.marker_for(arm, t), ec.MARKER_TABLE[arm][t - 1])
        case = ec.generate_cases(0)[0]
        with self.assertRaises(ValueError):
            ec.list_block(case, "bogus")
        with self.assertRaises(ValueError):
            ec.arm_content(case, "bogus")
        with self.assertRaises(ValueError):
            ec.marker_for("bogus", 1)

    def test_validate_question_direct_rows(self):
        """V1-V4 rejection rows hit directly: non-string
        inputs, template mismatch, a forbidden literal
        riding the topic slot, a marker token riding the
        topic slot, and an out-of-grammar expected value."""
        good_q = "What is the harbor inspection code?"
        self.assertTrue(ec.validate_question(good_q, "harbor", "iris-1423"))
        self.assertFalse(ec.validate_question(7, "harbor", "iris-1423"))
        self.assertFalse(ec.validate_question(good_q, None, "iris-1423"))
        self.assertFalse(ec.validate_question("What is the harbor code?", "harbor", "iris-1423"))
        self.assertFalse(
            ec.validate_question("What is the marker inspection code?", "marker", "iris-1423")
        )
        self.assertFalse(ec.validate_question("What is the 3. inspection code?", "3.", "iris-1423"))
        self.assertFalse(ec.validate_question(good_q, "harbor", "marigold-9999"))
        self.assertFalse(ec.validate_question(good_q, "harbor", "iris-1423\n"))

    def test_validate_case_tamper_rows(self):
        """One tamper per _validate_case rejection row,
        plus the never-raises wrapper on an unparseable
        fixture id."""
        base = ec.generate_cases(0)[0]
        self.assertTrue(ec.validate_case(copy.deepcopy(base)))

        def tampered(**changes):
            case = copy.deepcopy(base)
            case.update(changes)
            return case

        rows = [
            tampered(fixture_id="bogus-00"),
            tampered(fixture_id="enumerationconvention-24"),
            tampered(fixture_id="enumerationconvention-1"),
            tampered(index=3),
            tampered(topic="lagoon"),
            tampered(position=0),
            tampered(position="9"),
            tampered(topics="harbor"),
            tampered(topics=base["topics"][:8]),
            tampered(topics=list(reversed(base["topics"]))),
            tampered(values="iris-1423"),
            tampered(values=base["values"][:8]),
            tampered(values=["bogus-1"] + base["values"][1:]),
            tampered(values=[base["values"][0]] * 9),
            tampered(expected="cedar-4159"),
            tampered(neutral_indices=[0]),
            tampered(neutral_indices=[0, 0]),
            tampered(neutral_indices=[0, 24]),
            tampered(neutral_indices=[0, "1"]),
            tampered(question="What is the harbor inspection code"),
        ]
        for case in rows:
            self.assertFalse(ec.validate_case(case), case)
        self.assertFalse(ec.validate_case(tampered(fixture_id="enumerationconvention-x")))
        self.assertFalse(ec.validate_case({}))
        self.assertFalse(ec.validate_case(tampered(values=[None] * 9)))

    def test_glyph_and_reading_edges(self):
        """Mixed-case glyphs are not canonical (A1/A2
        machinery); the alpha_lower reading parses; an
        unknown reading system yields None; the echo
        guard rejects unknown arms and indices."""
        self.assertFalse(ec.glyph_ok("Ix"))
        self.assertFalse(ec.glyph_ok("iX"))
        cell = _cell("Ix. cedar-4159", expected="iris-1423")
        self.assertEqual(
            ec.decompose_cell(cell, "iris-1423", _gap_digest("iris-1423"), "roman", 9),
            "non_answer",
        )
        self.assertEqual(ec.index_in_system("b", "alpha_lower"), 2)
        self.assertEqual(ec.index_in_system("B", "alpha_upper"), 2)
        self.assertIsNone(ec.index_in_system("b", "bogus"))
        self.assertIsNone(ec.index_in_system("bb", "alpha_lower"))
        self.assertFalse(ec.is_echo("1.", "bogus", 1))
        self.assertFalse(ec.is_echo("1.", "roman", 10))
        stripped = ec.strip_one("b. cedar-4159", "lead")
        self.assertEqual(stripped, "cedar-4159")
        cell = _cell("b. cedar-4159", expected="iris-1423")
        self.assertEqual(
            ec.decompose_cell(cell, "iris-1423", _gap_digest("iris-1423"), "alpha_lower", 9),
            "wrong_value",
        )

    def test_aggregate_hostile_records(self):
        """Aggregate rows the main battery never built:
        a record with no token counts (delta defaults 0),
        an ids-identical verdict flip, a reverse pair in
        the divergent domain, a non-dict cell, a non-str
        detail, non-dict id digests, and a cells mapping
        that changes between the pairing filter and the
        loop (the pre-filtered continue)."""
        cases = ec.generate_cases(0)
        case = cases[0]
        expected = case["expected"]
        digest = _gap_digest(expected)

        def record(**overrides):
            rec = {
                "fixture": case["fixture_id"],
                "topic": case["topic"],
                "target_position": case["position"],
                "expected_digest": digest,
                "content_digests": {arm: _gap_digest("x") for arm in ec.ARM_KEYS},
                "arm_tokens": dict.fromkeys(ec.ARM_KEYS, 100),
                "id_digests": {arm: _gap_digest(arm) for arm in ec.ARM_KEYS},
                "ids_identical": dict.fromkeys(ec.TREATED_ARMS, False),
                "id_length_delta": dict.fromkeys(ec.TREATED_ARMS, 0),
                "cells": {
                    "control_arabic": _cell(expected, expected=expected),
                    "roman": _cell(expected, expected=expected),
                },
                "cell_details": dict.fromkeys(ec.ARM_KEYS, ""),
            }
            rec.update(overrides)
            return rec

        no_tokens = record(arm_tokens={})
        agg = ec.aggregate_candidate([no_tokens], cases)
        self.assertEqual(agg["ids"]["roman"]["id_length_delta"], [0])

        flip = record(
            id_digests={arm: _gap_digest("same") for arm in ec.ARM_KEYS},
            cells={
                "control_arabic": _cell(expected, expected=expected),
                "roman": _cell("cedar-4159", expected=expected),
            },
        )
        agg = ec.aggregate_candidate([flip], cases)
        self.assertEqual(agg["ids"]["roman"]["n_same_ids_verdict_flips"], 1)

        reverse = record(
            cells={
                "control_arabic": _cell("cedar-4159", expected=expected),
                "roman": _cell(expected, expected=expected),
            }
        )
        agg = ec.aggregate_candidate([reverse], cases)
        self.assertEqual(agg["headline"]["roman"]["label"], "insufficient")

        weird = record(
            cells={"control_arabic": "junk", "roman": _cell(expected, expected=expected)},
            cell_details={"roman": 7},
            id_digests=["not", "a", "dict"],
        )
        agg = ec.aggregate_candidate([weird], cases)
        self.assertEqual(agg["n_cases"], 1)

        shapeless = record(cells=[], cell_details=[])
        agg = ec.aggregate_candidate([shapeless], cases)
        self.assertEqual(agg["n_cases"], 1)

        class FlipCells(dict):
            """A cells mapping whose roman entry vanishes
            on its third read (hostile mutable input): the
            arms pass and the pairing filter still see it,
            the headline loop's re-fetch does not."""

            def __init__(self, inner):
                super().__init__(inner)
                self.roman_reads = 0

            def get(self, key, default=None):
                if key == "roman":
                    self.roman_reads += 1
                    if self.roman_reads >= 3:
                        return None
                return super().get(key, default)

        flip_rec = record()
        flip_rec["cells"] = FlipCells(flip_rec["cells"])
        agg = ec.aggregate_candidate([flip_rec], cases)
        self.assertEqual(agg["n_cases"], 1)


class TestGapBatteryStageAndMirror(unittest.TestCase):
    """Gap pass, part 2: stage degradation branches,
    mirror helper edges and recompute rejection rows,
    the diff status-drift row, and reporting loose
    shapes."""

    def test_cell_cancel_before_and_after(self):
        """The cell raises the cancel signal when the
        event is set before evaluate and when the
        backend sets it during evaluate."""
        import threading

        from model_protocol_lab.core import Fixture, Message

        backend = _ListBackend("perfect")
        fmt = _CHATML
        case = ec.generate_cases(0)[0]
        content = ec.arm_content(case, "control_arabic")
        fixture = Fixture(
            id=case["fixture_id"],
            messages=(Message("user", content),),
            expected=case["expected"],
            group=ec.GROUP,
        )
        ids = tuple(backend.encode(fmt.render(fixture.messages)))
        boundary = backend.encode("endmark")
        event = threading.Event()
        event.set()
        with self.assertRaises(experiments._EnumerationConventionCancelled):
            experiments._enumeration_convention_cell(
                backend,
                fmt,
                fixture,
                ids,
                boundary_ids=boundary,
                max_new_tokens=16,
                max_context=100_000,
                seed=0,
                cancel=event,
            )

        class SettingBackend(_ListBackend):
            """Backend that sets the event mid-generation."""

            def generate(self, *args, **kwargs):
                event2.set()
                return super().generate(*args, **kwargs)

        event2 = threading.Event()
        backend2 = SettingBackend("perfect")
        with self.assertRaises(experiments._EnumerationConventionCancelled):
            experiments._enumeration_convention_cell(
                backend2,
                fmt,
                fixture,
                ids,
                boundary_ids=boundary,
                max_new_tokens=16,
                max_context=100_000,
                seed=0,
                cancel=event2,
            )

    def test_stage_screen_branches(self):
        """Stage fail-closed branches: redraw exhaustion,
        an invalid case, a panel-distinctness break, a
        special-token finding, an undecodable boundary,
        and a boundary contained in the contents."""
        backend = _ListBackend("perfect")
        kwargs = {"boundary": backend.encode("endmark"), "max_context": 100_000, "seed": 0}
        with mock.patch.object(ec, "generate_cases", side_effect=ec.RedrawExhausted("x")):
            stage = experiments.enumeration_convention_stage(backend, (_CHATML,), **kwargs)
        self.assertEqual(stage["detail"], "enumeration_convention_screen_failed")
        real_generate = ec.generate_cases
        calls = {"n": 0}

        def flaky(seed):
            calls["n"] += 1
            if calls["n"] >= 3:
                raise ec.RedrawExhausted("x")
            return real_generate(seed)

        with mock.patch.object(ec, "generate_cases", side_effect=flaky):
            stage = experiments.enumeration_convention_stage(backend, (_CHATML,), **kwargs)
        self.assertEqual(stage["detail"], "enumeration_convention_screen_failed")
        self.assertEqual(stage["audit"]["by_reason"], {"redraw_exhausted": 1})

        cases = ec.generate_cases(0)
        broken = copy.deepcopy(cases)
        broken[0]["topic"] = "lagoon"
        with mock.patch.object(ec, "generate_cases", return_value=broken):
            stage = experiments.enumeration_convention_stage(backend, (_CHATML,), **kwargs)
        self.assertEqual(stage["detail"], "enumeration_convention_screen_failed")
        self.assertIn("case_invariant", stage["audit"]["by_reason"])

        dup = copy.deepcopy(cases)
        dup[1]["values"][0] = dup[0]["values"][0]
        with mock.patch.object(ec, "generate_cases", return_value=dup):
            stage = experiments.enumeration_convention_stage(backend, (_CHATML,), **kwargs)
        self.assertEqual(stage["detail"], "enumeration_convention_screen_failed")

        from model_protocol_lab import fixtures as fixtures_module

        with mock.patch.object(
            fixtures_module, "audit_special_token_literals", return_value=["hit"]
        ):
            stage = experiments.enumeration_convention_stage(backend, (_CHATML,), **kwargs)
        self.assertEqual(stage["detail"], "enumeration_convention_screen_failed")
        self.assertEqual(stage["audit"]["special_token_findings"], 1)

        class BadDecodeBackend(_ListBackend):
            """Backend whose decode raises."""

            def decode(self, ids):
                raise RuntimeError("undecodable")

        stage = experiments.enumeration_convention_stage(
            BadDecodeBackend("perfect"), (_CHATML,), **kwargs
        )
        self.assertTrue(stage["available"])

        class WordBoundaryBackend(_ListBackend):
            """Backend whose boundary decodes to a word
            the contents contain."""

            def decode(self, ids):
                return "The"

        stage = experiments.enumeration_convention_stage(
            WordBoundaryBackend("perfect"), (_CHATML,), **kwargs
        )
        self.assertEqual(stage["detail"], "enumeration_convention_screen_failed")

    def test_analysis_unavailable_candidate(self):
        """The analysis passes an unavailable candidate
        entry through with its detail."""
        stage = {
            "available": True,
            "detail": "",
            "protocol": {"seed": 0},
            "audit": {"n_audit_excluded": 0, "by_reason": {}, "special_token_findings": 0},
            "candidates": {"cand": {"status": "unavailable", "detail": "nope", "records": []}},
        }
        out = experiments.enumeration_convention_analysis(stage, ("cand",))
        self.assertEqual(out["candidates"]["cand"]["status"], "unavailable")
        self.assertEqual(out["candidates"]["cand"]["detail"], "nope")

    def test_mirror_helper_edges(self):
        """Mirror helper edges: surrogate text digest,
        digest shapes, unknown-arm raises, hostile
        record getters, roman/glyph/echo/pair rows, and
        a forced collision redraw in the mirror
        generator."""
        self.assertIsNone(dr._enumerationconvention_text_digest("\ud800"))
        self.assertFalse(dr._enumerationconvention_digest_ok("zz"))
        self.assertFalse(dr._enumerationconvention_digest_ok("a" * 63))
        case = dr._enumerationconvention_generate(0)[0]
        with self.assertRaises(ValueError):
            dr._enumerationconvention_list_block(case, "bogus")
        with self.assertRaises(ValueError):
            dr._enumerationconvention_arm_content(case, "bogus")
        self.assertIsNone(dr._enumerationconvention_cell_get({"cells": []}, "roman"))
        self.assertIsNone(dr._enumerationconvention_cell_get({"cells": {"roman": 5}}, "roman"))
        self.assertEqual(dr._enumerationconvention_detail_get({"cell_details": []}, "roman"), "")
        self.assertEqual(
            dr._enumerationconvention_detail_get({"cell_details": {"roman": 7}}, "roman"), ""
        )
        self.assertFalse(dr._enumerationconvention_ids_identical({"id_digests": []}, "roman"))
        self.assertIsNone(dr._enumerationconvention_parse_roman("IIII"))
        self.assertIsNone(dr._enumerationconvention_parse_roman("Ix"))
        self.assertIsNone(dr._enumerationconvention_parse_roman(""))
        self.assertFalse(dr._enumerationconvention_glyph_ok("Ix"))
        self.assertFalse(dr._enumerationconvention_is_echo("1.", "bogus", 1))
        self.assertEqual(
            dr._enumerationconvention_decompose_pair(
                {"answer_digest": "a" * 64, "boundary_found": False},
                {"answer_digest": "a" * 64, "boundary_found": False},
            ),
            "boundary_only",
        )
        self.assertEqual(
            dr._enumerationconvention_decompose_pair(
                {"answer_digest": "a" * 64, "boundary_found": True},
                {"answer_digest": "b" * 64, "boundary_found": False},
            ),
            "mixed",
        )
        real_derive = dr._enumerationconvention_derive

        def colliding(seed, index, slot, counter):
            if index == 0 and counter == 0 and slot in (31, 41):
                return real_derive(seed, index, slot - 1, counter)
            return real_derive(seed, index, slot, counter)

        with mock.patch.object(dr, "_enumerationconvention_derive", side_effect=colliding):
            cases = dr._enumerationconvention_generate(0)
        self.assertEqual(len(cases), 24)
        values = [v for c in cases for v in c["values"]]
        self.assertEqual(len(set(values)), 216)

    def test_mirror_cell_ok_rows(self):
        """One row per _enumerationconvention_cell_ok
        rejection branch, called directly."""
        digest = _gap_digest("iris-1423")
        good = {
            "verdict": True,
            "boundary_found": True,
            "boundary_text_match": True,
            "input_tokens": 10,
            "answer_digest": digest,
            "answer_text": "iris-1423",
        }
        self.assertTrue(dr._enumerationconvention_cell_ok(good, digest))

        def variant(**changes):
            cell = dict(good)
            cell.update(changes)
            return cell

        rows = [
            "junk",
            {k: v for k, v in good.items() if k != "verdict"},
            variant(verdict=1),
            variant(boundary_found="yes"),
            variant(boundary_text_match="yes"),
            variant(input_tokens=-1),
            variant(input_tokens="10"),
            variant(answer_digest="zz", answer_text=None),
            variant(answer_digest=None, answer_text="iris-1423"),
            variant(answer_text=7),
            variant(answer_text="cedar-4159"),
            variant(verdict=False),
            variant(answer_digest=None, answer_text=None, verdict=True, boundary_found=True),
        ]
        for cell in rows:
            self.assertFalse(dr._enumerationconvention_cell_ok(cell, digest), cell)

    def test_mirror_recompute_tamper_rows(self):
        """One tamper per recompute / valid rejection
        row, on deep copies of a valid payload."""
        base = _valid_payload()

        def check(mutate):
            payload = copy.deepcopy(base)
            mutate(payload)
            self.assertFalse(dr._valid_enumeration_convention(payload))

        check(lambda p: p.update(protocol=[]))
        check(lambda p: p["protocol"].update(seed=True))
        check(lambda p: p["protocol"].update(backend_fingerprint=7))
        check(lambda p: p["protocol"].update(generator="bogus"))
        check(lambda p: p.update(candidates={}))
        check(lambda p: p.update(candidates=[]))
        check(
            lambda p: p.update(
                candidates={
                    f"c{i}": {"status": "unavailable", "detail": "x", "records": []}
                    for i in range(25)
                }
            )
        )
        check(lambda p: p["candidates"].update(chatml=5))
        check(
            lambda p: p["candidates"].update(
                extra={"status": "unavailable", "detail": "x", "records": [{}]}
            )
        )
        check(lambda p: p["candidates"].update(extra={"status": "unavailable", "records": []}))
        check(lambda p: p["candidates"]["chatml"].update(detail="oops"))
        check(lambda p: p["candidates"]["chatml"].pop("arms"))
        check(lambda p: p["candidates"]["chatml"].update(records=[]))
        check(lambda p: p["candidates"]["chatml"]["records"].__setitem__(0, []))
        check(lambda p: p["candidates"]["chatml"]["records"][0].pop("topic"))
        check(lambda p: p["candidates"]["chatml"]["records"][0].update(fixture="zzz"))
        check(lambda p: p["candidates"]["chatml"]["records"][0].update(topic="lagoon"))
        check(lambda p: p["candidates"]["chatml"]["records"][0].update(target_position=1))
        check(lambda p: p["candidates"]["chatml"]["records"][0].update(expected_digest="0" * 64))
        for field in ("arm_tokens", "id_digests", "content_digests", "cells", "cell_details"):
            check(lambda p, f=field: p["candidates"]["chatml"]["records"][0].update({f: []}))
            check(lambda p, f=field: p["candidates"]["chatml"]["records"][0][f].pop("roman", None))
        for field in ("ids_identical", "id_length_delta"):
            check(lambda p, f=field: p["candidates"]["chatml"]["records"][0].update({f: []}))
        check(lambda p: p["candidates"]["chatml"]["records"][0]["arm_tokens"].update(roman=True))
        check(lambda p: p["candidates"]["chatml"]["records"][0]["arm_tokens"].update(roman=-1))

        def flip_ids(p):
            rec = p["candidates"]["chatml"]["records"][0]
            rec["ids_identical"]["roman"] = not rec["ids_identical"]["roman"]

        check(flip_ids)
        check(
            lambda p: p["candidates"]["chatml"]["records"][0]["id_length_delta"].update(roman=999)
        )
        check(lambda p: p["candidates"]["chatml"]["records"][0]["id_digests"].update(roman="zz"))
        check(
            lambda p: p["candidates"]["chatml"]["records"][0]["content_digests"].update(
                roman="0" * 64
            )
        )
        check(
            lambda p: p["candidates"]["chatml"]["records"][0]["cell_details"].update(roman="bogus")
        )

        def none_cell(p):
            rec = p["candidates"]["chatml"]["records"][0]
            rec["cells"]["roman"] = None
            rec["cell_details"]["roman"] = ""

        check(none_cell)

        def detailed_cell(p):
            rec = p["candidates"]["chatml"]["records"][0]
            rec["cell_details"]["roman"] = "arm_evaluation_failed"

        check(detailed_cell)
        check(
            lambda p: p["candidates"]["chatml"]["records"][0]["cells"]["roman"].update(
                input_tokens=999
            )
        )
        check(lambda p: p.update(n_candidates=99))
        check(lambda p: p.update(n_cases=99))
        check(lambda p: p["candidates"]["chatml"]["arms"]["roman"].update(rate=0.5))
        check(lambda p: p.update(audit={"n_audit_excluded": 0}))
        check(lambda p: p["audit"].update(n_audit_excluded=-1))
        check(lambda p: p["audit"].update(special_token_findings="x"))
        check(lambda p: p["audit"].update(by_reason=[]))
        check(lambda p: p["audit"].update(by_reason={"x": -1}))
        check(lambda p: p.update(detail="oops"))
        check(lambda p: p.update(available=False))
        with mock.patch.object(dr, "_ENUMERATIONCONVENTION_MAX_RECORDS_BYTES", 1):
            capped = copy.deepcopy(base)
            expected_protocol = dr._enumerationconvention_protocol(
                capped["protocol"]["seed"], capped["protocol"]["backend_fingerprint"]
            )
            capped["protocol"]["max_records_bytes"] = 1
            capped["protocol"]["fingerprint"] = expected_protocol["fingerprint"]
            self.assertFalse(dr._valid_enumeration_convention(capped))
        only_unavailable = copy.deepcopy(base)
        only_unavailable["candidates"] = {
            "solo": {"status": "unavailable", "detail": "x", "records": []}
        }
        only_unavailable["n_candidates"] = 1
        self.assertFalse(dr._valid_enumeration_convention(only_unavailable))

    def test_diff_status_drift_row(self):
        """A candidate available on one side and
        unavailable on the other produces a status
        drift entry."""
        old = _valid_payload()
        extra_records = copy.deepcopy(old["candidates"]["chatml"]["records"])
        old["candidates"]["cand2"] = {
            "status": "unavailable",
            "detail": "x",
            "records": [],
        }
        old["n_candidates"] = 3
        new = _valid_payload()
        entry = copy.deepcopy(new["candidates"]["chatml"])
        entry["records"] = extra_records
        new["candidates"]["cand2"] = entry
        new["n_candidates"] = 3
        self.assertTrue(dr._valid_enumeration_convention(old))
        self.assertTrue(dr._valid_enumeration_convention(new))
        diff = dr.diff_enumeration_convention(old, new)
        self.assertTrue(diff["changed"])
        metrics = {(d["candidate"], d["metric"]) for d in diff["candidate_drift"]}
        self.assertIn(("cand2", "status"), metrics)

    def test_reporting_loose_shapes(self):
        """Reporting rows: non-dict candidates, a
        non-numeric rate, an unavailable row, a non-dict
        entry, a non-dict arm, and non-dict ids/headline
        arm entries."""
        section = reporting._enumeration_convention_section(
            {"enumeration_convention": {"available": True, "candidates": []}}
        )
        self.assertEqual(section, "")
        payload = _valid_payload()
        entry = payload["candidates"]["chatml"]
        entry["arms"]["roman"]["rate"] = "fast"
        payload["candidates"]["gone"] = {"status": "unavailable", "detail": "x"}
        payload["candidates"]["junk"] = 5
        entry["arms"]["wrapper"] = 5
        entry["ids"]["roman"] = 5
        section = reporting._enumeration_convention_section({"enumeration_convention": payload})
        self.assertIn('id="enumeration-convention"', section)
        self.assertIn("unavailable", section)


class TestGapBatteryMirrorAggregate(unittest.TestCase):
    """Gap pass, part 3: the mirror aggregate driven
    directly (detail-bearing incomplete cells, no-token
    records, ids flips, reverse and non-answer pairs,
    the hostile mutable cells mapping), the mirror
    generator's exhaustion raise, the recompute
    exhaustion catches, and the last recompute / valid
    rows."""

    def _record(self, case, **overrides):
        """Mirror aggregate record fixture."""
        expected = case["expected"]
        rec = {
            "fixture": case["fixture_id"],
            "topic": case["topic"],
            "target_position": case["position"],
            "expected_digest": _gap_digest(expected),
            "content_digests": {arm: _gap_digest("x") for arm in ec.ARM_KEYS},
            "arm_tokens": dict.fromkeys(ec.ARM_KEYS, 100),
            "id_digests": {arm: _gap_digest(arm) for arm in ec.ARM_KEYS},
            "ids_identical": dict.fromkeys(ec.TREATED_ARMS, False),
            "id_length_delta": dict.fromkeys(ec.TREATED_ARMS, 0),
            "cells": {
                "control_arabic": _cell(expected, expected=expected),
                "roman": _cell(expected, expected=expected),
            },
            "cell_details": dict.fromkeys(ec.ARM_KEYS, ""),
        }
        rec.update(overrides)
        return rec

    def test_mirror_aggregate_rows(self):
        """Mirror aggregate rows."""
        cases = dr._enumerationconvention_generate(0)
        case = cases[0]
        expected = case["expected"]

        incomplete = self._record(
            case,
            cells={"control_arabic": _cell(expected, expected=expected), "roman": None},
            cell_details={**dict.fromkeys(ec.ARM_KEYS, ""), "roman": "arm_context_exceeded"},
        )
        agg = dr._enumerationconvention_aggregate([incomplete], cases)
        self.assertEqual(agg["arms"]["roman"]["n_context_exceeded"], 1)
        incomplete2 = self._record(
            case,
            cells={"control_arabic": _cell(expected, expected=expected), "roman": None},
            cell_details={**dict.fromkeys(ec.ARM_KEYS, ""), "roman": "arm_evaluation_failed"},
        )
        agg = dr._enumerationconvention_aggregate([incomplete2], cases)
        self.assertEqual(agg["arms"]["roman"]["n_evaluation_failed"], 1)

        agg = dr._enumerationconvention_aggregate([self._record(case, arm_tokens={})], cases)
        self.assertEqual(agg["ids"]["roman"]["id_length_delta"], [0])

        flip = self._record(
            case,
            id_digests={arm: _gap_digest("same") for arm in ec.ARM_KEYS},
            cells={
                "control_arabic": _cell(expected, expected=expected),
                "roman": _cell("cedar-4159", expected=expected),
            },
        )
        agg = dr._enumerationconvention_aggregate([flip], cases)
        self.assertEqual(agg["ids"]["roman"]["n_same_ids_verdict_flips"], 1)

        reverse = self._record(
            case,
            cells={
                "control_arabic": _cell("cedar-4159", expected=expected),
                "roman": _cell(expected, expected=expected),
            },
        )
        dr._enumerationconvention_aggregate([reverse], cases)

        garbage = self._record(
            case,
            cells={
                "control_arabic": _cell(expected, expected=expected),
                "roman": _cell("not a code at all", expected=expected),
            },
        )
        agg = dr._enumerationconvention_aggregate([garbage], cases)
        self.assertEqual(agg["headline"]["roman"]["label"], "insufficient")

        class FlipCells(dict):
            """Roman entry vanishes on its third read."""

            def __init__(self, inner):
                super().__init__(inner)
                self.roman_reads = 0

            def get(self, key, default=None):
                if key == "roman":
                    self.roman_reads += 1
                    if self.roman_reads >= 3:
                        return None
                return super().get(key, default)

        flip_rec = self._record(case)
        flip_rec["cells"] = FlipCells(flip_rec["cells"])
        dr._enumerationconvention_aggregate([flip_rec], cases)

    def test_mirror_remaining_rows(self):
        """Mirror remaining rows: roman letters outside
        the value table, a non-string answer text in
        the mirror decomposition, a None text with a
        valid digest in cell_ok, generator exhaustion,
        the recompute exhaustion catches, a bogus
        candidate status, an incomplete-cell payload
        validating through the recompute continue, and
        an extra top-level payload key."""
        self.assertIsNone(dr._enumerationconvention_parse_roman("AB"))
        digest = _gap_digest("iris-1423")
        cell = {"verdict": False, "answer_digest": "0" * 64, "answer_text": None}
        self.assertEqual(
            dr._enumerationconvention_decompose_cell(cell, "iris-1423", digest, "roman", 9),
            "non_answer",
        )
        good = {
            "verdict": True,
            "boundary_found": True,
            "boundary_text_match": True,
            "input_tokens": 10,
            "answer_digest": digest,
            "answer_text": "iris-1423",
        }
        no_text = dict(good, answer_text=None)
        self.assertFalse(dr._enumerationconvention_cell_ok(no_text, digest))
        with (
            mock.patch.object(dr, "_enumerationconvention_derive", side_effect=lambda *args: 0),
            self.assertRaises(dr._EnumerationConventionMirrorExhausted),
        ):
            dr._enumerationconvention_generate(0)
        base = _valid_payload()
        with mock.patch.object(
            dr,
            "_enumerationconvention_protocol",
            side_effect=dr._EnumerationConventionMirrorExhausted("x"),
        ):
            self.assertFalse(dr._valid_enumeration_convention(copy.deepcopy(base)))
        real_generate = dr._enumerationconvention_generate
        calls = {"n": 0}

        def flaky(seed):
            calls["n"] += 1
            if calls["n"] > 1:
                raise dr._EnumerationConventionMirrorExhausted("x")
            return real_generate(seed)

        with mock.patch.object(dr, "_enumerationconvention_generate", side_effect=flaky):
            self.assertFalse(dr._valid_enumeration_convention(copy.deepcopy(base)))
        bogus = copy.deepcopy(base)
        bogus["candidates"]["chatml"]["status"] = "bogus"
        self.assertFalse(dr._valid_enumeration_convention(bogus))
        extra = copy.deepcopy(base)
        extra["surprise"] = 1
        self.assertFalse(dr._valid_enumeration_convention(extra))
        exploded, _backend, _stage = _payload("explode")
        self.assertTrue(dr._valid_enumeration_convention(exploded))


if __name__ == "__main__":
    unittest.main()
