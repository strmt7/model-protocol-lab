"""Cycle 237: the typographic-emphasis probe (wave-12
falsification track, built as amended A1-A6).

Pins, by independent execution: the frozen generator
(a from-text second implementation -- the positive-
control pair written fresh on the main panel's A/B
pattern per amendment A4, not copied from the
designer's adjudicated tautology -- the seed-0 case-0
literals, the forced-collision redraw paths for both
the main value slots and the PC distractor slots, and
the exhaustion give-up); the section-4 validator with
the generated-text invariants (exactly one treated
span occurrence, the PC plain arm mark-free, the PC
info arm's only marks the target wrap); the section-5
decomposition against the designer's shipped battery
(20 entries, 18 distinct full tuples) plus the
fullmatch witnesses (trailing newline and trailing
space are out of grammar, never wrong-value); the
combined fire/certification label function (the
exhaustive guard audit reproducing 0/0/0 violations,
1,225 certification states, 245 cert-while-acquittal-
withholds states, and 1,471 cap-suppressed states;
the M = 23 and PC-failure demotions; the A5 PC
completion guard as a named battery); the exact
operating characteristics from this file's own code
(the fire grid by Fraction enumeration, the paired
column by Simpson quadrature, the anchored and
uniform certification curves with both delta-star
bisections, the PC floor derivation and dependence
range, the RTB fixed-assignment counterexample by
exact count DP, and the executed guard worlds at the
designer's seed plus an ids-dimension variant); the
stage payloads end to end through scripted backends
(a perfect run certifies every treated arm; a
bold-breaking run fires bold only; a PC-deaf run
demotes certificates to the sibling acquittal); the
diffreports mirror's recompute validation and tamper
rejections (including the named test that a
certificate cannot be asserted by a record whose
counts do not earn it); and the reporting section.
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
from math import comb, factorial
from statistics import NormalDist
from unittest import mock

from model_protocol_lab import cli, experiments, reporting
from model_protocol_lab import diffreports as dr
from model_protocol_lab import typographicemphasis as te
from model_protocol_lab.core import Format, Generation

_CHATML = Format("chatml", "", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n")
_LLAMA3 = Format(
    "llama3",
    "<|begin_of_text|>",
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|eot_id|>",
)
_WRONG_POOL = ("basalt-9999", "cedar-4159", "iris-1423")


class _EmphasisBackend:
    """Word-token scripted backend over the emphasis
    panel.

    Modes: ``perfect`` (the asked value read out of the
    content; on the PC panel, the bold topic's value
    when a bold topic is present, else the first fact's
    value), ``wrongvalue`` (treated main arms only: an
    in-grammar sibling value), ``echo`` (treated main
    arms answer in the arm's own echo form), ``garbage``
    (a fixed non-answer), ``noboundary`` (the value with
    no boundary token), ``explode`` (generate raises),
    ``pcdeaf`` (main arms perfect; the PC arms both
    answer the first fact's value, as if the marks
    channel carried nothing). With ``strip_marks=True``
    the encoder drops asterisks and backticks and
    lowercases, so every treated render tokenises
    exactly as control (the ids-identical regime).
    """

    fingerprint = "sha256:test-typographic-emphasis"

    def __init__(self, mode: str = "perfect", strip_marks: bool = False) -> None:
        """Init."""
        self.mode = mode
        self.strip_marks = strip_marks
        self.calls = 0
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
        """Encode."""
        if self.strip_marks:
            text = text.replace("*", "").replace("`", "").lower()
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

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        self.calls += 1
        if self.mode == "explode":
            raise RuntimeError("boom")
        text = self.decode(list(input_ids))
        values = re.findall(r"inspection code is (\S+?)\.", text)
        value = values[0] if values else "unknownword-0000"
        if "topic written in bold" in text:
            bold = re.search(r"\*\*(\S+)\*\* inspection code is (\S+?)\.", text)
            if self.mode == "pcdeaf" or bold is None:
                answer = value
            elif self.mode == "garbage":
                answer = "unknownword"
            else:
                answer = bold.group(2)
        elif self.mode == "garbage":
            answer = "unknownword"
        else:
            span_match = re.search(r"The (\S+) inspection code is", text)
            span = span_match.group(1) if span_match else ""
            if span.startswith("**") and span.endswith("**") and len(span) > 4:
                arm = "bold"
            elif span.startswith("*"):
                arm = "italic"
            elif span.startswith("`"):
                arm = "monospace"
            elif span.isupper():
                arm = "caps"
            else:
                arm = "control"
            if self.mode == "wrongvalue" and arm != "control":
                answer = next(v for v in _WRONG_POOL if v != value)
            elif self.mode == "echo" and arm != "control":
                answer = {
                    "bold": f"**{value}**",
                    "italic": f"*{value}*",
                    "monospace": f"`{value}`",
                    "caps": value.upper(),
                }[arm]
            else:
                answer = value
        boundary = self.encode("endmark")[0]
        out = list(self.encode(answer))
        if self.mode != "noboundary":
            out.append(boundary)
        pad = boundary if self.mode != "noboundary" else self.encode("fillerword")[0]
        out = (out + [pad] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(mode="perfect", strip_marks=False, formats=(_CHATML, _LLAMA3), **stage_kwargs):
    """Payload."""
    backend = _EmphasisBackend(mode, strip_marks=strip_marks)
    stage_kwargs.setdefault("boundary", backend.encode("endmark"))
    stage_kwargs.setdefault("max_context", 100_000)
    stage_kwargs.setdefault("seed", 0)
    stage = experiments.typographic_emphasis_stage(backend, formats, **stage_kwargs)
    names = tuple(fmt.name for fmt in formats)
    return experiments.typographic_emphasis_analysis(stage, names), backend, stage


def _valid_payload():
    """A fresh valid perfect-mode payload (mirror-test
    fixture)."""
    payload, _backend, _stage = _payload("perfect")
    return payload


def _cell(text=None, *, verdict=None, boundary=True, tokens=100, expected="glade-2443"):
    """Cell (the battery's expected value is the seed-0
    case-0 value glade-2443)."""
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
# Independent implementation B of the section-3
# generator, written from the design text (not from the
# module): the freeze test compares the two over seeds,
# and again under a patched draw that forces the redraw
# paths. The positive-control pair below is written
# fresh on the main panel's A/B pattern (amendment A4):
# the PC case draws its distractor and position from
# its own slots, and the PC contents are built here
# from the case fields, not via the module's builders.
# ------------------------------------------------------------------


def _draw_b(seed, index, slot, counter):
    """Draw (implementation B)."""
    material = str(seed) + "|typographicemphasis|" + str(index) + "|" + str(slot)
    material += "|" + str(counter)
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def _case_b(seed, index, seen, draw=_draw_b):
    """Case (implementation B)."""
    counter = 0
    while True:
        word = te.WORDS[draw(seed, index, 0, counter) % 24]
        number = 1000 + draw(seed, index, 1, counter) % 9000
        value = word + "-" + str(number)
        if value not in seen:
            break
        counter += 1
        if counter > 1000:
            raise te.RedrawExhausted
    seen.add(value)
    n_first = draw(seed, index, 2, 0) % 24
    n_second = draw(seed, index, 3, 0) % 24
    if n_second == n_first:
        n_second = (n_first + 1) % 24
    d_index = draw(seed, index, 4, 0) % 24
    if d_index == index:
        d_index = (index + 1) % 24
    counter = 0
    while True:
        d_word = te.WORDS[draw(seed, index, 5, counter) % 24]
        d_number = 1000 + draw(seed, index, 6, counter) % 9000
        d_value = d_word + "-" + str(d_number)
        if d_value != value:
            break
        counter += 1
        if counter > 1000:
            raise te.RedrawExhausted
    return {
        "fixture_id": f"typographicemphasis-{index:02d}",
        "index": index,
        "topic": te.TOPICS[index],
        "expected": value,
        "neutral_indices": [n_first, n_second],
        "distractor_topic": te.TOPICS[d_index],
        "distractor_value": d_value,
        "target_first": draw(seed, index, 7, 0) % 2 == 0,
    }


def _panel_b(seed, draw=_draw_b):
    """Panel (implementation B)."""
    seen: set[str] = set()
    return [_case_b(seed, i, seen, draw) for i in range(24)]


def _wrap_b(span, arm):
    """Wrap (implementation B)."""
    if arm == "bold":
        return "**" + span + "**"
    if arm == "italic":
        return "*" + span + "*"
    if arm == "monospace":
        return "`" + span + "`"
    if arm == "caps":
        return span.upper()
    return span


def _content_b(case, arm):
    """Main-arm content (implementation B)."""
    n1 = te.NEUTRAL_STATEMENTS[case["neutral_indices"][0]]
    n2 = te.NEUTRAL_STATEMENTS[case["neutral_indices"][1]]
    if arm in ("pc_plain", "pc_info"):
        target = (
            "The "
            + ("**" + case["topic"] + "**" if arm == "pc_info" else case["topic"])
            + " inspection code is "
            + case["expected"]
            + "."
        )
        distractor = (
            "The "
            + case["distractor_topic"]
            + " inspection code is "
            + case["distractor_value"]
            + "."
        )
        facts = (target, distractor) if case["target_first"] else (distractor, target)
        return (
            n1
            + " "
            + facts[0]
            + " "
            + facts[1]
            + " "
            + n2
            + " What is the inspection code of the topic written in bold?"
        )
    if arm == "pc_b":
        fact = "The ***" + case["topic"] + "*** inspection code is " + case["expected"] + "."
        question = "What is the ***" + case["topic"] + "*** inspection code?"
        return n1 + " " + fact + " " + n2 + " " + question
    fact = "The " + _wrap_b(case["topic"], arm) + " inspection code is " + case["expected"] + "."
    question = "What is the " + case["topic"] + " inspection code?"
    return n1 + " " + fact + " " + n2 + " " + question


class TestGeneratorFreeze(unittest.TestCase):
    """The frozen panel: two implementations, the seed-0
    literals, the pool digests, and the redraw paths.
    """

    def test_pool_digests_pinned(self):
        """Pools pinned by content digest (data and
        digests, never a host or library version)."""
        self.assertEqual(
            te.POOL_DIGESTS["topics"],
            "fdb85666e933495e197bb24281d0a326d1edb840931308618d7d2e9eccc2f30b",
        )
        self.assertEqual(
            te.POOL_DIGESTS["words"],
            "6acfcbc298e9d9e1bf06c585dafa9dc405e9ca5aec5747f6cb7ea399ab69d18e",
        )
        self.assertEqual(
            te.POOL_DIGESTS["neutral_statements"],
            "7b200df070672c2a4a4219969b78c6c08a8149853d312a7add931b21bf9a60cd",
        )
        self.assertEqual(len(te.TOPICS), 24)
        self.assertEqual(len(te.WORDS), 24)
        self.assertEqual(len(te.NEUTRAL_STATEMENTS), 24)

    def test_two_implementations_agree(self):
        """Module vs implementation B over five seeds:
        cases, contents for all eight record arms,
        validation."""
        for seed in (0, 1, 7, 42, 20261030):
            panel = te.generate_cases(seed)
            panel_b = _panel_b(seed)
            self.assertEqual(panel, panel_b)
            self.assertEqual(len({case["expected"] for case in panel}), 24)
            for case, case_b in zip(panel, panel_b, strict=True):
                self.assertTrue(te.validate_case(case))
                self.assertEqual(te.generate_case(seed, case["index"]), case)
                for arm in te.RECORD_ARMS:
                    self.assertEqual(te.content_for(case, arm), _content_b(case_b, arm))

    def test_seed0_case0_literals(self):
        """Seed 0 case 0: topic harbor, expected
        glade-2443, neutrals [22, 21], distractor falcon
        with jade-7376, target fact second."""
        case = te.generate_case(0, 0)
        self.assertEqual(case["topic"], "harbor")
        self.assertEqual(case["expected"], "glade-2443")
        self.assertEqual(case["neutral_indices"], [22, 21])
        self.assertEqual(case["distractor_topic"], "falcon")
        self.assertEqual(case["distractor_value"], "jade-7376")
        self.assertIs(case["target_first"], False)
        self.assertEqual(
            te.arm_content(case, "control"),
            "The soup simmered gently on the stove all morning. "
            "The harbor inspection code is glade-2443. "
            "Someone left a bicycle leaning against the railing. "
            "What is the harbor inspection code?",
        )
        self.assertEqual(
            te.arm_content(case, "bold"),
            "The soup simmered gently on the stove all morning. "
            "The **harbor** inspection code is glade-2443. "
            "Someone left a bicycle leaning against the railing. "
            "What is the harbor inspection code?",
        )
        self.assertEqual(
            te.pc_content(case, "pc_info"),
            "The soup simmered gently on the stove all morning. "
            "The falcon inspection code is jade-7376. "
            "The **harbor** inspection code is glade-2443. "
            "Someone left a bicycle leaning against the railing. "
            "What is the inspection code of the topic written in bold?",
        )

    def test_pc_value_sharing_and_marks(self):
        """A4 freeze properties over seeds: the PC
        target value is the main case's value (value
        sharing), the distractor differs in topic and
        value, pc_plain carries no mark characters, and
        pc_info differs from pc_plain only by the target
        topic's wrap."""
        for seed in (0, 3, 99):
            for case in te.generate_cases(seed):
                self.assertNotEqual(case["distractor_topic"], case["topic"])
                self.assertNotEqual(case["distractor_value"], case["expected"])
                plain = te.pc_content(case, "pc_plain")
                self.assertNotIn("*", plain)
                self.assertNotIn("`", plain)
                info = te.pc_content(case, "pc_info")
                self.assertEqual(info.count("*"), 4)
                self.assertEqual(info.replace(f"**{case['topic']}**", case["topic"], 1), plain)
                self.assertEqual(plain.count(case["expected"]), 1)

    def test_span_treatment_counts(self):
        """Build debt 2's generator asserts over seeds:
        exactly one treated span occurrence per main
        arm; the question and neutrals byte-identical
        across main arms; the expected value ASCII."""
        for seed in (0, 5):
            for case in te.generate_cases(seed):
                self.assertTrue(case["expected"].isascii())
                control = te.arm_content(case, "control")
                self.assertEqual(control.count(case["topic"]), 2)
                for arm in te.TREATED_ARMS:
                    content = te.arm_content(case, arm)
                    self.assertEqual(content.count(te.wrap_span(case["topic"], arm)), 1)
                    frame = content.replace(
                        te.fact_sentence(case, arm), te.fact_sentence(case, "control"), 1
                    )
                    self.assertEqual(frame, control)

    def test_forced_collision_redraws_value_slots(self):
        """Build debt 3: a forced main-value collision
        redraws slots 0+1 at counter+1 (the unforced
        panel's case 1 value differs from case 0's only
        through the forced draw)."""
        base = te.generate_cases(0)
        real = te._derive

        def forced(seed, index, slot, counter):
            if index == 1 and slot in (0, 1) and counter == 0:
                return real(seed, 0, slot, 0)
            return real(seed, index, slot, counter)

        with mock.patch.object(te, "_derive", forced):
            panel = te.generate_cases(0)
        self.assertEqual(panel[0], base[0])
        self.assertNotEqual(panel[1]["expected"], base[0]["expected"])
        # The redrawn value equals what slots 0+1 draw
        # at counter 1 for case 1.
        word = te.WORDS[real(0, 1, 0, 1) % 24]
        number = 1000 + real(0, 1, 1, 1) % 9000
        self.assertEqual(panel[1]["expected"], f"{word}-{number}")

    def test_forced_distractor_collision_redraws(self):
        """Build debt 3 (PC side): a forced distractor
        collision with the target value redraws slots
        5+6 at counter+1."""
        base = te.generate_cases(0)
        real = te._derive

        def forced(seed, index, slot, counter):
            if index == 0 and slot in (5, 6) and counter == 0:
                return real(seed, 0, slot - 5, 0)
            return real(seed, index, slot, counter)

        with mock.patch.object(te, "_derive", forced):
            panel = te.generate_cases(0)
        case = panel[0]
        self.assertEqual(case["expected"], base[0]["expected"])
        self.assertNotEqual(case["distractor_value"], case["expected"])
        word = te.WORDS[real(0, 0, 5, 1) % 24]
        number = 1000 + real(0, 0, 6, 1) % 9000
        self.assertEqual(case["distractor_value"], f"{word}-{number}")

    def test_redraw_exhaustion_fails_closed(self):
        """A draw that always collides exhausts the
        frozen limit and raises; the stage degrades to
        the screen-failed payload."""
        with (
            mock.patch.object(te, "_derive", lambda *args: 0),
            self.assertRaises(te.RedrawExhausted),
        ):
            te.generate_cases(0)
        backend = _EmphasisBackend("perfect")
        with mock.patch.object(te, "_derive", lambda *args: 0):
            stage = experiments.typographic_emphasis_stage(
                backend,
                (_CHATML,),
                boundary=backend.encode("endmark"),
                max_context=100_000,
                seed=0,
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "typographic_emphasis_screen_failed")
        self.assertEqual(stage["audit"]["by_reason"], {"redraw_exhausted": 1})


class TestValidator(unittest.TestCase):
    """The section-4 battery: one tamper per rejection
    row, and the never-raises property."""

    def test_panel_validates(self):
        """Panel validates."""
        for seed in (0, 1, 20261030):
            self.assertTrue(all(te.validate_case(c) for c in te.generate_cases(seed)))

    def test_tamper_rows(self):
        """Tamper rows."""
        base = te.generate_cases(0)[0]
        self.assertTrue(te.validate_case(copy.deepcopy(base)))

        def tampered(**changes):
            case = copy.deepcopy(base)
            case.update(changes)
            return case

        rows = [
            tampered(topic="summit"),
            tampered(fixture_id="typographicemphasis-01"),
            tampered(expected="glade-2443\n"),
            tampered(expected="glade-244"),
            tampered(expected="marigold-2443"),
            tampered(expected="glade-0999"),
            tampered(neutral_indices=[22, 22]),
            tampered(neutral_indices=[22]),
            tampered(neutral_indices=[22, 24]),
            tampered(distractor_topic="harbor"),
            tampered(distractor_topic="nowhere"),
            tampered(distractor_value="glade-2443"),
            tampered(distractor_value="jade-7376\n"),
            tampered(target_first=1),
            tampered(extra_key=True),
        ]
        for case in rows:
            self.assertFalse(te.validate_case(case), case)
        self.assertFalse(te.validate_case(None))
        self.assertFalse(te.validate_case("case"))
        self.assertFalse(te.validate_case({}))


class TestDecomposition(unittest.TestCase):
    """The section-5 ordered list: the designer's
    shipped battery (expected value alder-1234), the
    fullmatch witnesses this build adds, and the fuzz.
    """

    _E = "alder-1234"
    _BATTERY = [
        (_E, "bold", True, True, "pass"),
        (_E, "bold", False, False, "boundary_only"),
        ("basalt-9999", "bold", False, True, "wrong_value"),
        ("**alder**-1234", "bold", False, True, "echo_bold"),
        ("**alder-1234**", "bold", False, True, "echo_bold"),
        ("*alder*-1234", "italic", False, True, "echo_italic"),
        ("`alder`-1234", "monospace", False, True, "echo_monospace"),
        ("`alder-1234`", "monospace", False, True, "echo_monospace"),
        ("ALDER-1234", "caps", False, True, "echo_caps"),
        ("ALDER-1234", "caps", False, True, "echo_caps"),
        ("alder-1234".upper(), "caps", False, True, "echo_caps"),
        ("*alder*-1234", "bold", False, True, "form_only"),
        ("**alder-1234", "bold", False, True, "form_only"),
        ("Alder-1234", "caps", False, True, "form_only"),
        ("zebra-1234", "bold", False, True, "non_answer"),
        ("alder-0123", "bold", False, True, "non_answer"),
        ("", "bold", False, True, "non_answer"),
        ("**harbor**", "bold", False, True, "non_answer"),
        ("**ALDER**-1234", "caps", False, True, "non_answer"),
        (_E, "control", False, True, "non_answer"),
        # This build's fullmatch witnesses (the cycle-231
        # F1 rule): a trailing newline or space takes the
        # answer out of the frozen grammar entirely --
        # never wrong_value, never echo, never form_only.
        ("basalt-9999\n", "bold", False, True, "non_answer"),
        ("basalt-9999 ", "bold", False, True, "non_answer"),
        ("**alder-1234**\n", "bold", False, True, "non_answer"),
        ("ALDER-1234\n", "caps", False, True, "non_answer"),
    ]

    def test_battery(self):
        """Battery: 24 entries (the designer's 20 with
        18 distinct full tuples, plus 4 fullmatch
        witnesses), each landing in its frozen class."""
        distinct = {(a, arm, p, b) for a, arm, p, b, _want in self._BATTERY[:20]}
        self.assertEqual(len(distinct), 18)
        for text, arm, passed, boundary, want in self._BATTERY:
            cell = _cell(text, verdict=passed, boundary=boundary, expected=self._E)
            self.assertEqual(
                te.decompose_cell(cell, self._E, arm), want, (text, arm, passed, boundary)
            )

    def test_pair_partition(self):
        """Pair partition: equal digests boundary-only;
        differing digests with both boundaries exact-
        only; else mixed."""
        a = _cell("alder-1234", boundary=True)
        b = _cell("alder-1234", boundary=False)
        self.assertEqual(te.decompose_pair(a, b), "boundary_only")
        c = _cell("basalt-9999", boundary=True)
        self.assertEqual(te.decompose_pair(a, c), "exact_only")
        d = _cell("basalt-9999", boundary=False)
        self.assertEqual(te.decompose_pair(a, d), "mixed")

    def test_fuzz_totality(self):
        """Fuzz: 10,000 random answers x arms; every
        completed cell lands in the frozen vocabulary,
        and pass is exactly the verdict."""
        rng = random.Random(20261030)  # noqa: S311 -- deterministic test seed
        alphabet = "alderb-1234*`ALDER \n"
        for _ in range(10_000):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 14)))
            arm = rng.choice(te.ARM_KEYS)
            boundary = rng.random() < 0.5
            cell = _cell(text, boundary=boundary, expected=self._E)
            cls = te.decompose_cell(cell, self._E, arm)
            self.assertIn(cls, te.CELL_CLASSES)
            self.assertEqual(cls == "pass", cell["verdict"])


class TestClassification(unittest.TestCase):
    """The combined label: named witnesses for every
    branch, the certification gate demotions, and the
    exhaustive audit."""

    def test_label_witnesses(self):
        """Label witnesses."""
        c = te.classify_headline
        # Fire on observed counts, guard satisfied.
        self.assertEqual(c(24, 6, 6, 0, 0, 24, True), "discordant_asymmetric")
        self.assertEqual(c(10, 6, 9, 3, 0, 24, False), "discordant_asymmetric")
        # The certificate: E holds, full domain, guard
        # stable, PC passed.
        self.assertEqual(c(24, 0, 0, 0, 0, 24, True), "equivalence_certified")
        self.assertEqual(c(20, 1, 2, 3, 1, 24, True), "equivalence_certified")
        # The disclosed inversion: J = 0 with k > 0
        # certifies (its own guard is the stability
        # condition) where the sibling acquittal
        # withholds.
        self.assertEqual(c(20, 0, 0, 2, 2, 24, True), "equivalence_certified")
        # Demotion 1: domain short of 24 -- the same
        # counts keep only the sibling acquittal, and
        # the domain gate is named.
        self.assertEqual(c(23, 0, 0, 0, 0, 23, True), "no_discordance_observed")
        self.assertEqual(te.certification_blockers(23, 0, 0, 0, 23, True), ["domain_size"])
        # Demotion 2: the PC gate failed -- certificate
        # withheld, PC named the blocker.
        self.assertEqual(c(24, 0, 0, 0, 0, 24, False), "no_discordance_observed")
        self.assertEqual(te.certification_blockers(24, 0, 0, 0, 24, False), ["positive_control"])
        # Demotion 3: the statistic event holds but the
        # completion guard alone is unstable (H + k > 5
        # and S + k > 0) -- with J + k reaching the fire
        # floor the arm falls to insufficient, not
        # below_threshold.
        self.assertEqual(c(18, 4, 4, 4, 2, 24, True), "insufficient")
        self.assertEqual(te.certification_blockers(18, 4, 4, 2, 24, True), ["completion_guard"])
        # The statistic event itself fails (S > 0): the
        # certificate is blocked, and at J = 0, k = 0 the
        # arm keeps the sibling acquittal.
        self.assertEqual(c(20, 0, 3, 0, 0, 24, True), "no_discordance_observed")
        self.assertIn("statistic_event", te.certification_blockers(20, 3, 0, 0, 24, True))
        # Sibling acquittals and their guards.
        self.assertEqual(c(24, 0, 0, 0, 0, 24, False), "no_discordance_observed")
        self.assertEqual(c(20, 0, 7, 3, 1, 24, True), "insufficient")
        self.assertEqual(c(20, 3, 3, 0, 0, 24, True), "below_threshold")
        self.assertEqual(c(20, 5, 5, 0, 1, 24, True), "insufficient")
        # Floors and the cap.
        self.assertEqual(c(9, 0, 0, 0, 0, 24, True), "insufficient")
        self.assertEqual(c(24, 6, 6, 0, 7, 24, True), "insufficient")

    def test_exhaustive_guard_audit(self):
        """The exhaustive audit (design section 6.4,
        build debt 1's state space at J = H over M in
        {24, 23, 20}): zero manufactured-fire, zero
        unstable-acquittal, zero unstable-certification
        violations; zero certifications with the PC gate
        failed; 1,225 certification states; 245
        cert-while-acquittal-withholds states; the cap
        suppresses 1,471 completion-stable fires."""
        violations = {"fire": 0, "acquittal": 0, "cert": 0, "cert_no_pc": 0}
        cert_states = 0
        inversion_states = 0
        cap_suppressed = 0
        for domain in (24, 23, 20):
            for n_paired in range(domain + 1):
                k = domain - n_paired
                for d in range(n_paired + 1):
                    for h in range(d + 1):
                        j = h
                        for r in range(n_paired - d + 1):
                            label = te.classify_headline(d, j, h, r, k, domain, True)
                            if label == "discordant_asymmetric" and not (
                                d >= te.MIN_D and j >= te.MIN_J
                            ):
                                violations["fire"] += 1
                            if label in ("no_discordance_observed", "below_threshold"):
                                worst = te.classify_headline(
                                    d + k, j + k, h + k, r, 0, domain, True
                                )
                                if worst != label:
                                    violations["acquittal"] += 1
                            if label == "equivalence_certified":
                                cert_states += 1
                                if not (domain == 24 and h + k <= 5 and (h - r) + k <= 0):
                                    violations["cert"] += 1
                                if j == 0 and k > 0:
                                    inversion_states += 1
                            if (
                                te.classify_headline(d, j, h, r, k, domain, False)
                                == "equivalence_certified"
                            ):
                                violations["cert_no_pc"] += 1
                            if k > te.MAX_INCOMPLETE and d >= te.MIN_D and j >= te.MIN_J:
                                cap_suppressed += 1
        self.assertEqual(violations, {"fire": 0, "acquittal": 0, "cert": 0, "cert_no_pc": 0})
        self.assertEqual(cert_states, 1225)
        self.assertEqual(inversion_states, 245)
        self.assertEqual(cap_suppressed, 1471)


class TestPositiveControlGate(unittest.TestCase):
    """The PC gate and its A5 completion guard, as a
    named battery."""

    def test_gate_witnesses(self):
        """Gate witnesses."""
        g = te.pc_gate_passed
        self.assertTrue(g(24, 11, 0, 0))
        self.assertTrue(g(10, 8, 0, 0))
        # Info-pass floor.
        self.assertFalse(g(9, 20, 0, 0))
        # Margin floor on observed counts.
        self.assertFalse(g(24, 7, 0, 0))
        # A5 guard: the margin does not survive the
        # worst-case completion (13 - (4 + 2) = 7 < 8).
        self.assertFalse(g(24, 13, 4, 2))
        # A5 guard boundary: 14 - (4 + 2) = 8 passes.
        self.assertTrue(g(24, 14, 4, 2))
        # A5 cap: too many incomplete pairs.
        self.assertFalse(g(24, 20, 0, 7))
        # The guard spends incompletion only on the
        # margin; a completed pass count is untouched.
        self.assertTrue(g(10, 14, 0, 6))


# ------------------------------------------------------------------
# Figure machinery (build debt 4): every section-6
# figure recomputed here from this file's own code --
# the fire grid by Fraction enumeration, the paired
# column by Simpson quadrature over the shared factor,
# the certification curves by exact multinomial /
# trinomial enumeration, the PC gate by exact
# multinomial, the RTB counterexample by an exact
# count DP, and the guard worlds by seeded simulation.
# The heterogeneous completed-class sup itself (the
# licence figure 0.013826) is an archived coordinator
# derivation (a backward DP over (H, R, D, T) whose
# state space does not fit a unit test); it is pinned
# here against named single-assignment vertices of the
# same class, each computed exactly by the count DP
# and each dominated by the printed bound.
# ------------------------------------------------------------------

_ND = NormalDist()


def _binom_pmf(n, p):
    """Binom pmf."""
    p = Fraction(p)
    return [Fraction(comb(n, k)) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]


def _fire_exact(n, pc, q):
    """Fire exact."""
    total = Fraction(0)
    dpmf = _binom_pmf(n, pc)
    for d in range(10, n + 1):
        jpmf = _binom_pmf(d, q)
        total += dpmf[d] * sum(jpmf[6:], Fraction(0))
    return total


def _phi(x):
    """Phi."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x):
    """Phi."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bvn_both(pc, pa, rho, step=0.01, half=8.0):
    """Bvn both."""
    s, r = math.sqrt(1 - rho), math.sqrt(rho)
    tc, ta = _ND.inv_cdf(pc), _ND.inv_cdf(pa)
    n = int(2 * half / step)
    total = 0.0
    for i in range(n + 1):
        z0 = -half + i * step
        w = step * (1 if i in (0, n) else (4 if i % 2 else 2)) / 3
        total += w * _phi(z0) * _Phi((tc - r * z0) / s) * _Phi((ta - r * z0) / s)
    return total


def _fire_paired(rho, pc, pa):
    """Fire paired."""
    p_b = _bvn_both(pc, pa, rho)
    p_j, p_c = pc - p_b, 1 - pc
    total = 0.0
    for j in range(6, 25):
        for b in range(max(0, 10 - j), 25 - j):
            c = 24 - j - b
            total += comb(24, j) * comb(24 - j, b) * p_j**j * p_b**b * p_c**c
    return total


def _certify_paired(rho, pc, pa):
    """Certify paired."""
    p11 = _bvn_both(pc, pa, rho)
    p10, p01 = pc - p11, pa - p11
    p00 = 1 - pc - pa + p11
    total = 0.0
    for n10 in range(25):
        for n01 in range(25 - n10):
            for n11 in range(25 - n10 - n01):
                n00 = 24 - n10 - n01 - n11
                coef = factorial(24) // (
                    factorial(n10) * factorial(n01) * factorial(n11) * factorial(n00)
                )
                prob = coef * p10**n10 * p01**n01 * p11**n11 * p00**n00
                if n11 + n10 >= 10 and n10 <= 5 and n10 - n01 <= 0:
                    total += prob
    return total


def _full_event_prob(n, pc, delta):
    """Full event prob."""
    pa = pc - delta
    if pa < 0:
        return 0.0
    p11, p10 = pc * pa, pc * (1 - pa)
    p01, p00 = (1 - pc) * pa, (1 - pc) * (1 - pa)
    total = 0.0
    for n10 in range(n + 1):
        for n01 in range(n + 1 - n10):
            for n11 in range(n + 1 - n10 - n01):
                n00 = n - n10 - n01 - n11
                coef = factorial(n) // (
                    factorial(n10) * factorial(n01) * factorial(n11) * factorial(n00)
                )
                prob = coef * p10**n10 * p01**n01 * p11**n11 * p00**n00
                if n11 + n10 >= 10 and n10 <= 5 and n10 - n01 <= 0:
                    total += prob
    return total


def _tri_event_prob(n, p10, p01):
    """Tri event prob."""
    p0 = 1 - p10 - p01
    if p0 < -1e-12:
        return 0.0
    total = 0.0
    for h in range(min(5, n) + 1):
        for r in range(n - h + 1):
            if h - r > 0:
                continue
            n0 = n - h - r
            coef = factorial(n) // (factorial(h) * factorial(r) * factorial(n0))
            total += coef * p10**h * p01**r * p0**n0
    return total


def _sup_prob(n, delta, tstep=0.001):
    """Sup prob."""
    best = 0.0
    t = 0.0
    while t <= (1 - delta) / 2 + 1e-12:
        best = max(best, _tri_event_prob(n, t + delta, t))
        t += tstep
    return best


def _gate_prob(n, p_info, p_plain, thresh, min_info=10):
    """Gate prob."""
    p_b = p_info * (1 - p_plain)
    p_h = (1 - p_info) * p_plain
    p_pp = p_info * p_plain
    p_ff = (1 - p_info) * (1 - p_plain)
    total = 0.0
    for b in range(n + 1):
        for h in range(n + 1 - b):
            for pp in range(n + 1 - b - h):
                ff = n - b - h - pp
                coef = factorial(n) // (factorial(b) * factorial(h) * factorial(pp) * factorial(ff))
                prob = coef * p_b**b * p_h**h * p_pp**pp * p_ff**ff
                if pp + b >= min_info and b - h >= thresh:
                    total += prob
    return total


def _gate_prob_x(n, x, thresh=8, min_info=10):
    """Gate prob at discordant-cell parameter x (the
    A3 parameterization: each discordant cell carries
    probability x; both-pass and both-fail carry
    0.5 - x each)."""
    total = 0.0
    rest = 0.5 - x
    for b in range(n + 1):
        for h in range(n + 1 - b):
            for pp in range(n + 1 - b - h):
                ff = n - b - h - pp
                coef = factorial(n) // (factorial(b) * factorial(h) * factorial(pp) * factorial(ff))
                prob = coef * x**b * x**h * rest**pp * rest**ff
                if pp + b >= min_info and b - h >= thresh:
                    total += prob
    return total


def _assignment_prob(laws, n_pairs=24):
    """Exact P(E) under a fixed per-pair assignment of
    outcome laws; each law is a list of
    (dH, dR, dD, probability) outcomes. Exact count DP
    over (H, R, D)."""
    states = {(0, 0, 0): 1.0}
    for law in laws:
        nxt: dict[tuple[int, int, int], float] = {}
        for (h, r, d), prob in states.items():
            for dh, dr_, dd, p in law:
                key = (h + dh, r + dr_, d + dd)
                nxt[key] = nxt.get(key, 0.0) + prob * p
        states = nxt
    return sum(prob for (h, r, d), prob in states.items() if d >= 10 and h <= 5 and h - r <= 0)


def _vertex_laws(delta):
    """The completed class's named per-pair laws at a
    delta (coordinator derivation): degenerate
    both-pass / harm / both-fail, and the two-point
    laws b1 (harm or both-pass), b2 (harm or benefit),
    b3 (harm or both-fail)."""
    return {
        "both_pass": [(0, 0, 1, 1.0)],
        "harm": [(1, 0, 1, 1.0)],
        "both_fail": [(0, 0, 0, 1.0)],
        "b1": [(1, 0, 1, delta), (0, 0, 1, 1 - delta)],
        "b2": [(1, 0, 1, (1 + delta) / 2), (0, 1, 0, (1 - delta) / 2)],
        "b3": [(1, 0, 1, delta), (0, 0, 0, 1 - delta)],
    }


class TestFigures(unittest.TestCase):
    """Every printed figure, recomputed."""

    def test_fire_grid_independent(self):
        """Fire grid (exact Fraction enumeration)."""
        cases = [
            (("0.90", "0.10"), 0.017274),
            (("0.80", "0.20"), 0.1743),
            (("0.50", "0.50"), 0.550306),
            (("0.90", "0.20"), 0.2542),
            (("0.90", "0.35"), 0.8159),
            (("0.85", "0.40"), 0.8761),
            (("0.90", "0.60"), 0.9990),
            (("0.90", "0.2791"), 0.5831),
            (("0.90", "0.3514"), 0.8193),
            (("0.90", "0.5145"), 0.9909),
            (("0.90", "0.6544"), 0.9998),
        ]
        for (pc, q), want in cases:
            got = float(_fire_exact(24, Fraction(pc), Fraction(q)))
            self.assertEqual(round(got, 4), round(want, 4), (pc, q))
        self.assertAlmostEqual(
            float(_fire_exact(24, Fraction("0.90"), Fraction("0.10"))), 0.017274, places=5
        )
        self.assertAlmostEqual(
            float(_fire_exact(24, Fraction("0.50"), Fraction("0.50"))), 0.550306, places=5
        )

    def test_fire_paired_and_familywise(self):
        """Paired fire column (quadrature) and the
        family-wise fire price over 96 comparisons."""
        self.assertAlmostEqual(_fire_paired(0.5, 0.90, 0.90), 0.004442, delta=5e-4)
        self.assertAlmostEqual(_fire_paired(0.5, 0.90, 0.65), 0.698078, delta=5e-4)
        p = float(_fire_exact(24, Fraction("0.90"), Fraction("0.10")))
        self.assertEqual(round(96 * p, 4), 1.6583)
        self.assertEqual(round(1 - (1 - p) ** 96, 4), 0.8123)
        oc = te.OPERATING_CHARACTERISTICS
        self.assertEqual(oc["familywise_fire_expected_96"], 1.6583)
        self.assertEqual(oc["familywise_fire_any_96"], 0.8123)

    def test_certification_curves(self):
        """The anchored curve (full event, independence
        at control rate 0.90) and the uniform sup, with
        the three near-0.0122 quantities under their
        full names."""
        self.assertEqual(round(_full_event_prob(24, 0.90, 0.0), 4), 0.5974)
        self.assertEqual(round(_full_event_prob(24, 0.90, 0.05), 4), 0.3796)
        self.assertEqual(round(_full_event_prob(24, 0.90, 0.10), 4), 0.2165)
        self.assertEqual(round(_full_event_prob(24, 0.90, 0.20), 4), 0.0519)
        self.assertEqual(round(_full_event_prob(24, 0.90, 0.34), 4), 0.0034)
        # Anchored probability at the OLD delta 0.28:
        # 0.0122 -- not the licence.
        self.assertAlmostEqual(_full_event_prob(24, 0.90, 0.28), 0.012232, delta=5e-4)
        # Homogeneous uniform sup at delta 0.30: 0.012252
        # -- not the licence either.
        self.assertAlmostEqual(_sup_prob(24, 0.30), 0.012252, delta=5e-4)
        # Homogeneous uniform sup at delta 0.34 (context
        # figure): 0.005414.
        self.assertAlmostEqual(_sup_prob(24, 0.34), 0.005414, delta=5e-4)
        oc = te.OPERATING_CHARACTERISTICS
        self.assertEqual(oc["certification_bound_completed_per_comparison"], 0.013826)
        self.assertEqual(oc["certification_bound_restricted_vertex_delta_034"], 0.012253)
        self.assertEqual(oc["certify_anchored_delta_034_pc_090"], 0.0034)
        self.assertEqual(oc["certify_anchored_delta_000_pc_090"], 0.5974)
        # Family-wise certification price at the licence.
        bound = oc["certification_bound_completed_per_comparison"]
        self.assertEqual(round(96 * bound, 4), 1.3273)
        self.assertEqual(round(1 - (1 - bound) ** 96, 4), 0.7373)

    def test_delta_star_bisections(self):
        """Both delta-star bisections: anchored 0.2545,
        uniform 0.2739 (the smallest delta whose bound
        meets alpha = 0.02, on a 0.0005 grid)."""
        lo, hi = 0.20, 0.34
        for _ in range(40):
            mid = (lo + hi) / 2
            if _full_event_prob(24, 0.90, mid) <= 0.02:
                hi = mid
            else:
                lo = mid
        self.assertAlmostEqual(hi, 0.2545, delta=0.002)
        lo, hi = 0.20, 0.34
        for _ in range(24):
            mid = (lo + hi) / 2
            if _sup_prob(24, mid, tstep=0.002) <= 0.02:
                hi = mid
            else:
                lo = mid
        self.assertAlmostEqual(hi, 0.2739, delta=0.004)

    def test_certify_paired(self):
        """Certification under the paired copula:
        0.6139 at delta 0, 0.0044 at delta 0.28."""
        self.assertAlmostEqual(_certify_paired(0.5, 0.90, 0.90), 0.6139, delta=5e-3)
        self.assertAlmostEqual(_certify_paired(0.5, 0.90, 0.62), 0.0044, delta=5e-3)

    def test_pc_floor_derivation_and_range(self):
        """PC floors from the gate's own null: the net
        threshold sweep (5/6/7/8 -> 0.0966/0.0557/
        0.0297/0.0147, so 8 is the smallest threshold
        meeting alpha), the gate OC rows, and the A3
        dependence range in the discordant-cell
        parameter x."""
        for thresh, want in ((5, 0.0966), (6, 0.0557), (7, 0.0297), (8, 0.0147)):
            self.assertEqual(round(_gate_prob(24, 0.50, 0.50, thresh), 4), want)
        self.assertEqual(round(_gate_prob(24, 0.90, 0.50, 8), 4), 0.7709)
        self.assertEqual(round(_gate_prob(24, 0.80, 0.50, 8), 4), 0.4668)
        self.assertEqual(round(_gate_prob(24, 0.75, 0.50, 8), 4), 0.3250)
        self.assertEqual(round(_gate_prob(24, 0.90, 0.90, 8), 4), 0.0002)
        self.assertEqual(_gate_prob(24, 0.00, 0.50, 8), 0.0)
        for x, want in ((0.25, 0.0147), (0.30, 0.0235), (0.35, 0.0332), (0.50, 0.0758)):
            self.assertEqual(round(_gate_prob_x(24, x), 4), want, x)
        # The independence point of the x-curve is the
        # independent-arms deaf value.
        self.assertAlmostEqual(_gate_prob_x(24, 0.25), _gate_prob(24, 0.50, 0.50, 8), places=9)

    def test_rtb_counterexample_and_vertices(self):
        """The RTB fixed assignment (16 x b1, 8 x b2 at
        delta 0.28) reproduces 0.022662 exactly by the
        count DP -- the counterexample that forced the
        completed class. Named vertices of the
        completed class at delta 0.34 are each dominated
        by the printed licence bound 0.013826."""
        laws = _vertex_laws(0.28)
        got = _assignment_prob([laws["b1"]] * 16 + [laws["b2"]] * 8)
        self.assertAlmostEqual(got, 0.022662, delta=1e-6)
        self.assertEqual(
            te.OPERATING_CHARACTERISTICS["rtb_counterexample_certification_probability"],
            0.022662,
        )
        laws = _vertex_laws(0.34)
        bound = te.OPERATING_CHARACTERISTICS["certification_bound_completed_per_comparison"]
        for name in ("b1", "b2", "b3"):
            got = _assignment_prob([laws[name]] * 24)
            self.assertLessEqual(got, bound, name)
        mixed = _assignment_prob([laws["b1"]] * 16 + [laws["b2"]] * 8)
        self.assertLessEqual(mixed, bound)
        self.assertEqual(_assignment_prob([laws["harm"]] * 24), 0.0)


def _combined_unguarded(d, j, h, r, k, domain, pc):
    """The designer's combined() with guarded=False
    (test-local: the shipped module carries only the
    guarded form)."""
    s = h - r
    if d < 10:
        return "insufficient"
    if j >= 6:
        return "discordant_asymmetric"
    if h <= 5 and s <= 0 and pc:
        return "equivalence_certified"
    if j == 0:
        return "no_discordance_observed"
    if j < 6:
        return "below_threshold"
    return "insufficient"


def _guard_world(pc_, pa, remove, trials, seed, ids_dimension=False):
    """One executed guard world (the designer's
    world() shape): per-case independent arms, a
    removal rule, J set to H (maximal fire pressure).
    With ids_dimension, each case is id-divergent with
    probability 0.8 (the sibling simulation parameter)
    and only divergent cases enter the domain."""
    rng = random.Random(seed)  # noqa: S311 -- deterministic test seed
    p11, p10 = pc_ * pa, pc_ * (1 - pa)
    p01 = (1 - pc_) * pa
    guarded = {"equivalence_certified": 0, "discordant_asymmetric": 0}
    unguarded = {"equivalence_certified": 0, "discordant_asymmetric": 0}
    for _ in range(trials):
        # Stream discipline (the designer's): all 24
        # cell draws first, then the removal draws, and
        # a removal draw is consumed only where the
        # removal predicate's cell test passes.
        cells = []
        divergent_flags = []
        for _case in range(24):
            divergent_flags.append(rng.random() < 0.8 if ids_dimension else True)
            x = rng.random()
            cells.append(
                "PP" if x < p11 else "H" if x < p11 + p10 else "R" if x < p11 + p10 + p01 else "FF"
            )
        kept = []
        k = 0
        domain = 0
        for cell, divergent in zip(cells, divergent_flags, strict=True):
            if not divergent:
                continue
            domain += 1
            if remove(cell, rng):
                k += 1
            else:
                kept.append(cell)
        d = sum(cell in ("PP", "H") for cell in kept)
        h = sum(cell == "H" for cell in kept)
        r = sum(cell == "R" for cell in kept)
        label = te.classify_headline(d, h, h, r, k, domain, True)
        if label in guarded:
            guarded[label] += 1
        label_u = _combined_unguarded(d, h, h, r, k, domain, True)
        if label_u in unguarded:
            unguarded[label_u] += 1
    return {key: round(count / trials, 4) for key, count in guarded.items()}, {
        key: round(count / trials, 4) for key, count in unguarded.items()
    }


class TestGuardWorlds(unittest.TestCase):
    """The executed attack worlds (build debt 1): the
    designer's four worlds at the designer's seed, and
    the ids-dimension variant."""

    def test_worlds_at_designer_seed(self):
        """Worlds A / B-delta / B-PASTA / C reproduce
        the printed guarded rates at the designer's
        seed 20261030 (the stream the printed numbers
        came from: 24 cell draws, then removal draws
        only where the predicate's cell test passes)."""
        seed = 20261030
        guarded_a, _u = _guard_world(
            0.90, 0.90, lambda c, r: c == "PP" and r.random() < 0.5, 10_000, seed
        )
        self.assertEqual(guarded_a["equivalence_certified"], 0.0016)
        self.assertEqual(guarded_a["discordant_asymmetric"], 0.0047)
        guarded_b, unguarded_b = _guard_world(
            0.90, 0.62, lambda c, r: c == "H" and r.random() < 0.6, 10_000, seed
        )
        self.assertEqual(unguarded_b["equivalence_certified"], 0.2709)
        self.assertEqual(guarded_b["equivalence_certified"], 0.0127)
        guarded_p, _u2 = _guard_world(
            0.90, 0.4855, lambda c, r: c == "H" and r.random() < 0.6, 10_000, seed
        )
        self.assertEqual(guarded_p["equivalence_certified"], 0.0006)
        guarded_c, _u3 = _guard_world(0.90, 0.65, lambda c, r: r.random() < 0.15, 10_000, seed)
        self.assertEqual(guarded_c["discordant_asymmetric"], 0.6318)

    def test_worlds_ids_dimension(self):
        """The same worlds under the ids dimension
        (divergence probability 0.8): the guard's
        direction survives -- guarded certification
        never exceeds the unguarded rate, and the
        manufacture world's guarded fire stays under
        the licence."""
        seed = 20261030
        guarded_a, unguarded_a = _guard_world(
            0.90,
            0.90,
            lambda c, r: c == "PP" and r.random() < 0.5,
            4_000,
            seed,
            ids_dimension=True,
        )
        self.assertLessEqual(
            guarded_a["equivalence_certified"], unguarded_a["equivalence_certified"]
        )
        self.assertLess(guarded_a["discordant_asymmetric"], 0.02)
        guarded_b, unguarded_b = _guard_world(
            0.90,
            0.62,
            lambda c, r: c == "H" and r.random() < 0.6,
            4_000,
            seed,
            ids_dimension=True,
        )
        self.assertLess(guarded_b["equivalence_certified"], unguarded_b["equivalence_certified"])


class TestStagePayloads(unittest.TestCase):
    """Stage payloads end to end through the scripted
    backends."""

    def test_perfect_run_certifies(self):
        """Perfect backend: the PC gate passes and every
        treated arm is certified at M = 24."""
        payload, backend, stage = _payload("perfect")
        self.assertTrue(payload["available"])
        self.assertEqual(payload["n_candidates"], 2)
        self.assertEqual(payload["n_cases"], 24)
        self.assertGreater(backend.calls, 0)
        for name in ("chatml", "llama3"):
            entry = payload["candidates"][name]
            self.assertTrue(entry["positive_control"]["passed"])
            self.assertEqual(entry["positive_control"]["info_passes"], 24)
            for arm in te.TREATED_ARMS:
                head = entry["headline"][arm]
                self.assertEqual(head["label"], "equivalence_certified", (name, arm))
                self.assertEqual(head["n_ids_divergent"], 24)
                self.assertEqual(head["certification_blocked_by"], [])
                self.assertEqual(head["delta"], 0.34)
                self.assertEqual(head["alpha"], 0.02)
            self.assertEqual(entry["arms"]["control"]["rate"], 1.0)
            self.assertEqual(entry["arms"]["pc_b"]["rate"], 1.0)

    def test_wrongvalue_run_fires(self):
        """Wrong-value backend: every treated arm fires;
        the control decomposition stays clean."""
        payload, _backend, _stage = _payload("wrongvalue")
        entry = payload["candidates"]["chatml"]
        for arm in te.TREATED_ARMS:
            head = entry["headline"][arm]
            self.assertEqual(head["label"], "discordant_asymmetric", arm)
            self.assertEqual(head["wrong_values"], 24)
            self.assertEqual(head["harm"], 24)
        self.assertEqual(entry["arms"]["bold"]["decomposition"]["wrong_value"], 24)
        self.assertEqual(entry["arms"]["control"]["decomposition"]["pass"], 24)

    def test_echo_run_decomposes_not_fires(self):
        """Echo backend: treated answers land in the
        arm's echo class, never in J; no arm fires and
        none certifies (H counts every failure class)."""
        payload, _backend, _stage = _payload("echo")
        entry = payload["candidates"]["chatml"]
        for arm in te.TREATED_ARMS:
            self.assertEqual(entry["arms"][arm]["decomposition"][f"echo_{arm}"], 24)
            head = entry["headline"][arm]
            self.assertEqual(head["wrong_values"], 0)
            self.assertEqual(head["harm"], 24)
            self.assertNotEqual(head["label"], "equivalence_certified")
            self.assertNotEqual(head["label"], "discordant_asymmetric")

    def test_pcdeaf_run_demotes_certificates(self):
        """PC-deaf backend: the main panel is perfect
        but the positive control cannot pass, so every
        arm keeps the sibling acquittal and names the
        positive-control blocker (section 13's named
        demotion, end to end)."""
        payload, _backend, _stage = _payload("pcdeaf")
        entry = payload["candidates"]["chatml"]
        self.assertFalse(entry["positive_control"]["passed"])
        for arm in te.TREATED_ARMS:
            head = entry["headline"][arm]
            self.assertEqual(head["label"], "no_discordance_observed", arm)
            self.assertEqual(head["certification_blocked_by"], ["positive_control"])
            self.assertFalse(head["pc_gate_passed"])

    def test_ids_identical_regime(self):
        """Strip-marks encoder: treated renders are
        ids-identical to control, so the domain is
        empty and every label is insufficient (the
        instrument reports inapplicability, never a
        pass)."""
        payload, _backend, _stage = _payload("perfect", strip_marks=True)
        entry = payload["candidates"]["chatml"]
        for arm in te.TREATED_ARMS:
            head = entry["headline"][arm]
            self.assertEqual(head["n_ids_divergent"], 0)
            self.assertEqual(head["label"], "insufficient")

    def test_garbage_and_noboundary(self):
        """Garbage reads non_answer; with no boundary
        token anywhere, evaluate takes the whole padded
        output as the answer, so no-boundary cells also
        land in non_answer (the sibling suites'
        observation); true boundary_only cells are
        exercised at the aggregate level below."""
        payload, _b, _s = _payload("garbage")
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["control"]["decomposition"]["non_answer"], 24)
        for arm in te.TREATED_ARMS:
            self.assertEqual(entry["headline"][arm]["label"], "insufficient")
        payload, _b, _s = _payload("noboundary")
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["control"]["decomposition"]["non_answer"], 24)
        self.assertEqual(entry["arms"]["control"]["rate"], 0.0)

    def test_boundary_only_at_aggregate_level(self):
        """Hand-built cells whose digest matches but
        whose boundary flag is False decompose as
        boundary_only in both aggregates, and the
        producer and mirror aggregates agree exactly on
        hand-built records (including a certified
        configuration)."""
        cases = te.generate_cases(0)

        def hand_records(cell_for):
            records = []
            for case in cases:
                cells, details, tokens, digests = {}, {}, {}, {}
                for arm in te.RECORD_ARMS:
                    cells[arm] = cell_for(case, arm)
                    details[arm] = ""
                    tokens[arm] = 50
                    digests[arm] = f"d-{arm}-{case['index']}"
                records.append(
                    {
                        "fixture": case["fixture_id"],
                        "topic": case["topic"],
                        "expected_digest": te.digest_text(case["expected"]),
                        "content_digests": {},
                        "arm_tokens": tokens,
                        "id_digests": digests,
                        "ids_identical": dict.fromkeys(te.TREATED_ARMS, False),
                        "id_length_delta": dict.fromkeys(te.TREATED_ARMS, 0),
                        "cells": cells,
                        "cell_details": details,
                    }
                )
            return records

        def boundary_cell(case, arm):
            return {
                "verdict": False,
                "boundary_found": False,
                "boundary_text_match": False,
                "input_tokens": 50,
                "answer_digest": te.digest_text(case["expected"]),
                "answer_text": case["expected"],
            }

        records = hand_records(boundary_cell)
        agg = te.aggregate_candidate(records, cases)
        for arm in te.ARM_KEYS:
            self.assertEqual(agg["arms"][arm]["decomposition"]["boundary_only"], 24)
        self.assertEqual(agg, dr._typographicemphasis_aggregate(records, cases))

        def perfect_cell(case, arm):
            verdict = case["index"] >= 11 if arm == "pc_plain" else True
            return {
                "verdict": verdict,
                "boundary_found": True,
                "boundary_text_match": True,
                "input_tokens": 50,
                "answer_digest": te.digest_text(case["expected"]),
                "answer_text": case["expected"],
            }

        records = hand_records(perfect_cell)
        agg = te.aggregate_candidate(records, cases)
        self.assertTrue(agg["positive_control"]["passed"])
        self.assertEqual(agg["positive_control"]["plain_fail_info_pass"], 11)
        for arm in te.TREATED_ARMS:
            self.assertEqual(agg["headline"][arm]["label"], "equivalence_certified")
        self.assertEqual(agg, dr._typographicemphasis_aggregate(records, cases))

    def test_explode_backend_degrades_cells(self):
        """Exploding backend: cells degrade to
        arm_evaluation_failed; the stage stays
        available with insufficient labels."""
        payload, _b, _s = _payload("explode")
        self.assertTrue(payload["available"])
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["control"]["n_evaluation_failed"], 24)
        self.assertEqual(entry["arms"]["control"]["n_completed"], 0)
        for arm in te.TREATED_ARMS:
            self.assertEqual(entry["headline"][arm]["label"], "insufficient")

    def test_cancel_unwinds_stage(self):
        """A set cancel event unwinds the whole stage
        to the cancelled detail."""
        import threading

        backend = _EmphasisBackend("perfect")
        cancel = threading.Event()
        cancel.set()
        stage = experiments.typographic_emphasis_stage(
            backend,
            (_CHATML,),
            boundary=backend.encode("endmark"),
            max_context=100_000,
            seed=0,
            cancel=cancel,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_context_exceeded_cells(self):
        """A tiny context budget degrades every cell to
        arm_context_exceeded."""
        payload, _b, _s = _payload("perfect", max_context=1)
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["control"]["n_context_exceeded"], 24)
        self.assertEqual(entry["arms"]["control"]["n_completed"], 0)

    def test_stage_preconditions_and_budget(self):
        """Formats preconditions raise; 25 candidates
        exceed the ground-record budget."""
        backend = _EmphasisBackend("perfect")
        with self.assertRaises(ValueError):
            experiments.typographic_emphasis_stage(
                backend, (), boundary=(), max_context=100_000, seed=0
            )
        with self.assertRaises(ValueError):
            experiments.typographic_emphasis_stage(
                backend,
                (_CHATML, _CHATML),
                boundary=backend.encode("endmark"),
                max_context=100_000,
                seed=0,
            )
        formats = tuple(Format(f"fmt{i}", "", "u", "a", "e") for i in range(25))
        stage = experiments.typographic_emphasis_stage(
            backend,
            formats,
            boundary=backend.encode("endmark"),
            max_context=100_000,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "typographic_emphasis_record_budget_exceeded")

    def test_analysis_unavailable_passthrough(self):
        """Analysis of an unavailable stage passes the
        detail through."""
        out = experiments.typographic_emphasis_analysis(
            {"available": False, "detail": "cancelled", "protocol": {}, "audit": {}},
            ("chatml",),
        )
        self.assertFalse(out["available"])
        self.assertEqual(out["detail"], "cancelled")


class TestMirror(unittest.TestCase):
    """The diffreports mirror: recompute validation,
    tamper rejections, and diff behaviour."""

    def test_valid_payloads(self):
        """Perfect and wrong-value payloads validate."""
        self.assertTrue(dr._valid_typographic_emphasis(_valid_payload()))
        payload, _b, _s = _payload("wrongvalue")
        self.assertTrue(dr._valid_typographic_emphasis(payload))
        payload, _b, _s = _payload("perfect", strip_marks=True)
        self.assertTrue(dr._valid_typographic_emphasis(payload))

    def test_certificate_cannot_be_asserted(self):
        """Section 13(6), the named test: a record whose
        counts do not earn a certificate cannot assert
        one -- relabelling a firing arm's headline to
        equivalence_certified fails recompute
        validation, as does minting a certificate on a
        PC-deaf payload."""
        payload, _b, _s = _payload("wrongvalue")
        self.assertTrue(dr._valid_typographic_emphasis(payload))
        forged = copy.deepcopy(payload)
        forged["candidates"]["chatml"]["headline"]["bold"]["label"] = "equivalence_certified"
        self.assertFalse(dr._valid_typographic_emphasis(forged))
        payload, _b, _s = _payload("pcdeaf")
        forged = copy.deepcopy(payload)
        for arm in te.TREATED_ARMS:
            forged["candidates"]["chatml"]["headline"][arm]["label"] = "equivalence_certified"
            forged["candidates"]["chatml"]["headline"][arm]["certification_blocked_by"] = []
        self.assertFalse(dr._valid_typographic_emphasis(forged))

    def test_tamper_rows(self):
        """One tamper per recompute rejection class."""
        payload = _valid_payload()

        def tampered(mutate):
            forged = copy.deepcopy(payload)
            mutate(forged)
            return forged

        rows = [
            lambda p: p["candidates"]["chatml"]["records"][0].update(topic="summit"),
            lambda p: p["candidates"]["chatml"]["records"][0]["cells"]["bold"].update(
                answer_text="basalt-9999"
            ),
            lambda p: p["candidates"]["chatml"]["records"][0]["content_digests"].update(
                bold="0" * 64
            ),
            lambda p: p["candidates"]["chatml"]["records"][0]["ids_identical"].update(bold=True),
            lambda p: p["candidates"]["chatml"]["records"][0]["id_length_delta"].update(bold=99),
            lambda p: p["candidates"]["chatml"]["records"][0]["cells"]["bold"].update(
                verdict=False
            ),
            lambda p: p["protocol"].update(delta=0.28),
            lambda p: p["candidates"]["chatml"]["positive_control"].update(passed=False),
            lambda p: p["candidates"]["chatml"]["arms"]["bold"].update(n_pass=0),
            lambda p: p.update(n_cases=23),
        ]
        for mutate in rows:
            self.assertFalse(dr._valid_typographic_emphasis(tampered(mutate)), mutate)
        self.assertFalse(dr._valid_typographic_emphasis(None))
        self.assertFalse(dr._valid_typographic_emphasis({"available": True}))

    def test_diff_behaviour(self):
        """Diff: identical payloads unchanged; a mode
        change drifts; a seed change is protocol drift;
        one-sided shapes report plainly."""
        payload = _valid_payload()
        same = dr.diff_typographic_emphasis(payload, copy.deepcopy(payload))
        self.assertEqual(same["available"], "both")
        self.assertFalse(same["changed"])
        other, _b, _s = _payload("wrongvalue")
        drift = dr.diff_typographic_emphasis(payload, other)
        self.assertTrue(drift["changed"])
        self.assertTrue(any(row["metric"] == "bold_label" for row in drift["candidate_drift"]))
        seeded, _b, _s = _payload("perfect", seed=7)
        drift = dr.diff_typographic_emphasis(payload, seeded)
        self.assertTrue(drift["protocol_drift"])
        self.assertFalse(drift["comparable"])
        self.assertEqual(
            dr.diff_typographic_emphasis(None, None),
            {"available": "neither", "changed": False},
        )
        self.assertEqual(dr.diff_typographic_emphasis(payload, None)["available"], "old")
        self.assertEqual(dr.diff_typographic_emphasis(None, payload)["available"], "new")

    def test_diff_reports_registry(self):
        """The top-level diff carries the new section
        and summary keys."""
        payload = _valid_payload()
        old = {"decision": {}, "search": [], "typographic_emphasis": payload}
        new = {"decision": {}, "search": [], "typographic_emphasis": copy.deepcopy(payload)}
        out = dr.diff_reports(old, new)
        self.assertIn("typographic_emphasis", out)
        self.assertEqual(out["typographic_emphasis"]["available"], "both")
        self.assertFalse(out["summary"]["typographic_emphasis_changed"])
        self.assertFalse(out["summary"]["typographic_emphasis_asymmetric"])
        changed = dr.diff_reports(
            old,
            {
                "decision": {},
                "search": [],
                "typographic_emphasis": _payload("wrongvalue")[0],
            },
        )
        self.assertTrue(changed["summary"]["typographic_emphasis_changed"])


class TestReportingSection(unittest.TestCase):
    """The reporting section: renders the readout with
    its licence companions, leaks no answer text,
    absent without a payload, and the TOC/body
    correspondence holds (the cycle-231/233 precedent:
    the probe's own file carries the correspondence
    test)."""

    def test_section_renders(self):
        """Perfect payload: certificates, the licence
        figures under their full names, and the PC
        block render; no expected value leaks."""
        payload, _b, _s = _payload("perfect")
        section = reporting._typographic_emphasis_section({"typographic_emphasis": payload})
        self.assertIn('<section class="card" id="typographic-emphasis">', section)
        self.assertIn("equivalence_certified", section)
        self.assertIn("0.013826", section)
        self.assertIn("restricted-vertex", section)
        self.assertIn("Positive control", section)
        self.assertIn("delta 0.34", section)
        for case in te.generate_cases(0):
            self.assertNotIn(case["expected"], section)
            self.assertNotIn(case["distractor_value"], section)

    def test_section_absent(self):
        """No section without an available payload."""
        self.assertEqual(reporting._typographic_emphasis_section({}), "")
        self.assertEqual(
            reporting._typographic_emphasis_section(
                {"typographic_emphasis": {"available": False, "detail": "x"}}
            ),
            "",
        )

    def test_toc_body_correspondence(self):
        """TOC/body: the rendered report links the anchor
        and carries the section after the nav."""
        payload, _b, _s = _payload("perfect")
        html = reporting.render_html({"typographic_emphasis": payload})
        self.assertIn('href="#typographic-emphasis"', html)
        self.assertIn('<section class="card" id="typographic-emphasis">', html)
        self.assertGreater(html.index('id="typographic-emphasis"'), html.index("</nav>"))

    def test_section_loose_candidate_shapes(self):
        """Loose candidate shapes degrade, never crash."""
        payload, _b, _s = _payload("perfect")
        broken = copy.deepcopy(payload)
        broken["candidates"]["chatml"] = {"status": "available", "detail": ""}
        section = reporting._typographic_emphasis_section({"typographic_emphasis": broken})
        self.assertIn('id="typographic-emphasis"', section)
        broken["candidates"]["chatml"] = "junk"
        section = reporting._typographic_emphasis_section({"typographic_emphasis": broken})
        self.assertIn('id="typographic-emphasis"', section)


class TestCliDispatch(unittest.TestCase):
    """The CLI dispatcher: populates the result key on
    success and the fail-closed dict when the stage
    raises; the flag parses."""

    def test_dispatcher_success_and_failure(self):
        """Dispatcher success and failure."""
        result: dict = {}
        backend = _EmphasisBackend("perfect")
        cli._run_typographic_emphasis_probe(
            backend,
            (_CHATML,),
            result,
            boundary=backend.encode("endmark"),
            boundary_map=None,
            max_new_tokens=64,
            max_context=100_000,
            seed=0,
            cancel=None,
        )
        self.assertTrue(result["typographic_emphasis"]["available"])
        with mock.patch.object(
            experiments, "typographic_emphasis_stage", side_effect=RuntimeError("boom")
        ):
            result2: dict = {}
            cli._run_typographic_emphasis_probe(
                backend,
                (_CHATML,),
                result2,
                boundary=(),
                boundary_map=None,
                max_new_tokens=64,
                max_context=100_000,
                seed=0,
                cancel=None,
            )
        self.assertEqual(
            result2["typographic_emphasis"],
            {"available": False, "detail": "typographic_emphasis_unavailable"},
        )

    def test_flag_parses(self):
        """The --typographic-emphasis flag parses to the
        snake-case attribute, defaulting off."""
        parser = cli.build_parser()
        required = [
            "run",
            "--model",
            "m",
            "--dataset",
            "d",
            "--family",
            "chatml",
            "--boundary-ids",
            "1",
            "--output",
            "o",
        ]
        args = parser.parse_args([*required, "--typographic-emphasis"])
        self.assertTrue(args.typographic_emphasis)
        args = parser.parse_args(required)
        self.assertFalse(args.typographic_emphasis)


class TestCoverageBattery(unittest.TestCase):
    """Defence and tamper coverage for the validator's
    invariant battery, the redraw give-up paths, and the
    mirror's recompute guards (the cycle-237 coverage
    recovery: every guard in the new code is exercised,
    not asserted unreachable)."""

    def test_validator_index_range(self):
        """Out-of-range indices are invalid."""
        case = te.generate_cases(0)[0]
        for bad in (24, -1):
            tampered = dict(case, index=bad)
            self.assertFalse(te.validate_case(tampered))

    def test_validator_defence_invariants(self):
        """Each content invariant in validate_case fires
        when a builder is patched to break exactly it:
        the battery is defence in depth, exercised here
        one invariant at a time."""
        case = te.generate_cases(0)[0]
        topic = case["topic"]
        real_content = te.arm_content
        real_pc = te.pc_content

        def content_extra_topic(case_, arm):
            text = real_content(case_, arm)
            return text + f" {topic}" if arm == "control" else text

        def content_frame_break(case_, arm):
            text = real_content(case_, arm)
            if arm == "italic":
                text = text.replace("The weather changed", "The weather turned", 1)
                if text == real_content(case_, arm):
                    text = text + " Extra sentence here."
            return text

        def pc_plain_marked(case_, arm):
            text = real_pc(case_, arm)
            return text + " *" if arm == "pc_plain" else text

        def pc_info_unmarked(case_, arm):
            return real_pc(case_, "pc_plain") if arm == "pc_info" else real_pc(case_, arm)

        def pc_info_value_drift(case_, arm):
            text = real_pc(case_, arm)
            if arm == "pc_info":
                text = text.replace(case_["distractor_value"], "alder-1000", 1)
            return text

        def pc_plain_double_value(case_, arm):
            plain = real_pc(case_, "pc_plain").replace(
                case_["distractor_value"], case_["expected"], 1
            )
            if arm == "pc_plain":
                return plain
            return plain.replace(
                f"The {topic} inspection code is {case_['expected']}.",
                f"The **{topic}** inspection code is {case_['expected']}.",
                1,
            )

        scenarios = [
            ("arm_content", content_extra_topic),
            ("arm_content", content_frame_break),
            ("pc_content", pc_plain_marked),
            ("pc_content", pc_info_unmarked),
            ("pc_content", pc_info_value_drift),
            ("pc_content", pc_plain_double_value),
        ]
        for builder, fake in scenarios:
            with (
                self.subTest(builder=builder, fake=fake.__name__),
                mock.patch.object(te, builder, fake),
            ):
                self.assertFalse(te.validate_case(case))
        with mock.patch.object(te, "wrap_span", lambda span, arm: span):
            self.assertFalse(te.validate_case(case))
        with mock.patch.object(te, "pcb_content", lambda case_: real_content(case_, "control")):
            self.assertFalse(te.validate_case(case))

        def raising(case_, arm):
            raise RuntimeError("builder failure")

        with mock.patch.object(te, "arm_content", raising):
            self.assertFalse(te.validate_case(case))

    def test_value_redraw_exhaustion(self):
        """The main value loop gives up when every case
        draws the same value (distractor draws stay real,
        so the failure is the value loop's own)."""
        real = te._derive

        def colliding(seed, index, slot, counter):
            return 0 if slot in (0, 1) else real(seed, index, slot, counter)

        with (
            mock.patch.object(te, "_derive", colliding),
            self.assertRaises(te.RedrawExhausted),
        ):
            te.generate_cases(0)

    def test_decompose_non_string_text(self):
        """A completed cell whose answer text is not a
        string decomposes as non_answer."""
        cell = {
            "verdict": False,
            "boundary_found": True,
            "boundary_text_match": False,
            "input_tokens": 10,
            "answer_digest": "0" * 64,
            "answer_text": 12345,
        }
        self.assertEqual(te.decompose_cell(cell, "glade-2443", "bold"), "non_answer")

    def test_aggregate_reverse_pairs(self):
        """Control-fail / arm-pass pairs count as
        n_reverse in the producer and the mirror."""
        cases = te.generate_cases(0)

        def cell_for(case, arm):
            if arm == "control":
                text = "alder-1000" if case["expected"] != "alder-1000" else "basalt-1000"
                verdict = False
            elif arm == "pc_plain":
                text, verdict = case["expected"], True
            else:
                text, verdict = case["expected"], True
            return {
                "verdict": verdict,
                "boundary_found": True,
                "boundary_text_match": True,
                "input_tokens": 50,
                "answer_digest": te.digest_text(text),
                "answer_text": text,
            }

        records = []
        for case in cases:
            records.append(
                {
                    "fixture": case["fixture_id"],
                    "topic": case["topic"],
                    "expected_digest": te.digest_text(case["expected"]),
                    "content_digests": {},
                    "arm_tokens": dict.fromkeys(te.RECORD_ARMS, 50),
                    "id_digests": {arm: f"d-{arm}-{case['index']}" for arm in te.RECORD_ARMS},
                    "ids_identical": dict.fromkeys(te.TREATED_ARMS, False),
                    "id_length_delta": dict.fromkeys(te.TREATED_ARMS, 0),
                    "cells": {arm: cell_for(case, arm) for arm in te.RECORD_ARMS},
                    "cell_details": dict.fromkeys(te.RECORD_ARMS, ""),
                }
            )
        agg = te.aggregate_candidate(records, cases)
        for arm in te.TREATED_ARMS:
            self.assertEqual(agg["headline"][arm]["n_reverse"], 24)
            self.assertEqual(agg["headline"][arm]["label"], "insufficient")
        self.assertEqual(agg, dr._typographicemphasis_aggregate(records, cases))

    def test_mirror_valid_payload_modes(self):
        """Echo, garbage and no-boundary payloads are
        mirror-valid (their decompose paths execute in
        the mirror)."""
        for mode in ("echo", "garbage", "noboundary", "pcdeaf", "wrongvalue"):
            with self.subTest(mode=mode):
                payload, _b, _s = _payload(mode)
                self.assertTrue(dr._valid_typographic_emphasis(payload))

    def test_mirror_generate_exhaustion(self):
        """Mirror regeneration exhaustion validates
        false (both redraw loops)."""
        payload = _valid_payload()
        with mock.patch.object(dr, "_typographicemphasis_derive", lambda *args: 0):
            self.assertFalse(dr._valid_typographic_emphasis(payload))
        real = dr._typographicemphasis_derive

        def colliding(seed, index, slot, counter):
            return 0 if slot in (0, 1) else real(seed, index, slot, counter)

        with mock.patch.object(dr, "_typographicemphasis_derive", colliding):
            self.assertFalse(dr._valid_typographic_emphasis(payload))

    def test_mirror_tamper_battery(self):
        """Field-level tampers across the record,
        protocol and aggregate shapes all invalidate."""
        import copy

        base = _valid_payload()
        entry = lambda p: p["candidates"]["chatml"]  # noqa: E731
        record = lambda p: entry(p)["records"][0]  # noqa: E731

        def mutate(fn):
            payload = copy.deepcopy(base)
            fn(payload)
            return payload

        tampers = {
            "surrogate_text": lambda p: record(p)["cells"]["bold"].update(answer_text="\ud800"),
            "malformed_digest": lambda p: record(p)["content_digests"].update(bold="zz"),
            "cells_not_dict": lambda p: record(p).update(cells=[]),
            "detail_not_str": lambda p: record(p)["cell_details"].update(bold=7),
            "id_digests_not_dict": lambda p: record(p).update(id_digests=[]),
            "record_fixture": lambda p: record(p).update(fixture="nope"),
            "record_topic": lambda p: record(p).update(topic="summit"),
            "record_expected_digest": lambda p: record(p).update(expected_digest="0" * 64),
            "arm_tokens_str": lambda p: record(p)["arm_tokens"].update(bold="50"),
            "id_length_delta_wrong": lambda p: record(p)["id_length_delta"].update(bold=1),
            "ids_identical_wrong": lambda p: record(p)["ids_identical"].update(bold=True),
            "cell_verdict_flipped": lambda p: record(p)["cells"]["bold"].update(verdict=False),
            "cell_boundary_flipped": lambda p: record(p)["cells"]["bold"].update(
                boundary_found=False
            ),
            "cell_text_tampered": lambda p: record(p)["cells"]["bold"].update(
                answer_text="alder-1000"
            ),
            "cell_detail_tampered": lambda p: record(p)["cell_details"].update(
                bold="arm_evaluation_failed"
            ),
            "protocol_seed_str": lambda p: p["protocol"].update(seed="0"),
            "protocol_delta": lambda p: p["protocol"].update(delta=0.30),
            "protocol_alpha": lambda p: p["protocol"].update(alpha=0.05),
            "protocol_min_d": lambda p: p["protocol"].update(min_d=9),
            "protocol_pool_digest": lambda p: p["protocol"]["pool_digests"].update(topics="0" * 64),
            "protocol_case_material": lambda p: p["protocol"].update(case_material_sha256="0" * 64),
            "protocol_ground": lambda p: p["protocol"].update(ground_sha256="0" * 64),
            "protocol_budget": lambda p: p["protocol"].update(generation_budget=1),
            "protocol_pc_margin": lambda p: p["protocol"].update(pc_min_margin=7),
            "headline_label": lambda p: entry(p)["headline"]["bold"].update(
                label="below_threshold"
            ),
            "headline_delta": lambda p: entry(p)["headline"]["bold"].update(delta=0.3),
            "headline_blocked": lambda p: entry(p)["headline"]["bold"].update(
                certification_blocked_by=["positive_control"]
            ),
            "headline_harm": lambda p: entry(p)["headline"]["bold"].update(harm=1),
            "headline_net_harm": lambda p: entry(p)["headline"]["bold"].update(net_harm=1),
            "headline_reverse": lambda p: entry(p)["headline"]["bold"].update(n_reverse=1),
            "headline_control_passes": lambda p: entry(p)["headline"]["bold"].update(
                control_passes=23
            ),
            "pc_passed_flipped": lambda p: entry(p)["positive_control"].update(passed=False),
            "pc_margin_wrong": lambda p: entry(p)["positive_control"].update(margin=0),
            "dose_pcb_wrong": lambda p: entry(p)["dose"].update(pcb_passes=0),
            "ids_divergent_wrong": lambda p: entry(p)["ids"]["bold"].update(n_ids_divergent=23),
            "arms_completed_wrong": lambda p: entry(p)["arms"]["bold"].update(n_completed=23),
            "n_cases_wrong": lambda p: entry(p).update(n_cases=23),
            "n_incomplete_wrong": lambda p: entry(p).update(n_incomplete=1),
            "oc_delta_wrong": lambda p: entry(p)["operating_characteristics"].update(delta=0.3),
        }
        for name, fn in tampers.items():
            with self.subTest(tamper=name):
                self.assertFalse(dr._valid_typographic_emphasis(mutate(fn)), name)

    def test_diff_changed_and_availability(self):
        """Differing valid payloads diff as changed; an
        invalid side degrades availability."""
        perfect = _valid_payload()
        fired, _b, _s = _payload("wrongvalue")
        diff = dr.diff_typographic_emphasis(perfect, fired)
        self.assertEqual(diff["available"], "both")
        self.assertTrue(diff["changed"])
        one_sided = dr.diff_typographic_emphasis({"available": False}, perfect)
        self.assertEqual(one_sided["available"], "new")
        self.assertFalse(one_sided["changed"])

    def test_mirror_direct_units(self):
        """The mirror's pure functions against the
        producer's witness values (form-only and
        non-string decompose paths, pair partition, PC
        gate failures, blockers, classify witnesses)."""
        digest = te.digest_text("glade-2443")
        form = _cell("*glade-2443", verdict=False)
        self.assertEqual(
            dr._typographicemphasis_decompose_cell(form, "glade-2443", digest, "bold"),
            "form_only",
        )
        nonstr = _cell(None, verdict=False)
        nonstr["answer_text"] = 12345
        self.assertEqual(
            dr._typographicemphasis_decompose_cell(nonstr, "glade-2443", digest, "bold"),
            "non_answer",
        )
        passing = _cell("glade-2443")
        failing = _cell("alder-1000", verdict=False)
        self.assertEqual(dr._typographicemphasis_decompose_pair(passing, failing), "exact_only")
        no_boundary = _cell("alder-1000", verdict=False, boundary=False)
        self.assertEqual(dr._typographicemphasis_decompose_pair(passing, no_boundary), "mixed")
        self.assertFalse(dr._typographicemphasis_pc_gate(24, 8, 4, 0))
        self.assertFalse(dr._typographicemphasis_pc_gate(24, 20, 0, 7))
        self.assertTrue(dr._typographicemphasis_pc_gate(24, 12, 4, 0))
        self.assertEqual(
            dr._typographicemphasis_blockers(20, 4, 4, 2, 24, True), ["completion_guard"]
        )
        self.assertEqual(
            dr._typographicemphasis_classify(24, 0, 0, 0, 0, 24, True),
            "equivalence_certified",
        )
        self.assertEqual(
            dr._typographicemphasis_classify(24, 6, 6, 0, 0, 24, True),
            "discordant_asymmetric",
        )
        self.assertEqual(dr._typographicemphasis_classify(24, 3, 3, 0, 3, 24, True), "insufficient")
        self.assertIsNone(dr._typographicemphasis_rate(1, 0))

    def test_mirror_aggregate_direct_shapes(self):
        """Hand records exercising the mirror aggregate's
        shape guards (non-int tokens, ids-identical flips,
        PC harm pairs, missing cells, failed cells);
        producer and mirror agree on every one."""
        cases = te.generate_cases(0)

        def build(cell_for, **record_overrides):
            records = []
            for case in cases:
                record = {
                    "fixture": case["fixture_id"],
                    "topic": case["topic"],
                    "expected_digest": te.digest_text(case["expected"]),
                    "content_digests": {},
                    "arm_tokens": dict.fromkeys(te.RECORD_ARMS, 50),
                    "id_digests": {arm: f"d-{arm}-{case['index']}" for arm in te.RECORD_ARMS},
                    "ids_identical": dict.fromkeys(te.TREATED_ARMS, False),
                    "id_length_delta": dict.fromkeys(te.TREATED_ARMS, 0),
                    "cells": {arm: cell_for(case, arm) for arm in te.RECORD_ARMS},
                    "cell_details": dict.fromkeys(te.RECORD_ARMS, ""),
                }
                record.update(record_overrides)
                records.append(record)
            return records

        def ok_cell(case, arm):
            return {
                "verdict": True,
                "boundary_found": True,
                "boundary_text_match": True,
                "input_tokens": 50,
                "answer_digest": te.digest_text(case["expected"]),
                "answer_text": case["expected"],
            }

        def pc_harm_cell(case, arm):
            cell = ok_cell(case, arm)
            if arm == "pc_info":
                cell["verdict"] = False
                cell["answer_text"] = "alder-1000"
                cell["answer_digest"] = te.digest_text("alder-1000")
            return cell

        records = build(pc_harm_cell)
        agg = te.aggregate_candidate(records, cases)
        self.assertEqual(agg["positive_control"]["plain_pass_info_fail"], 24)
        self.assertEqual(agg, dr._typographicemphasis_aggregate(records, cases))

        def flip_cell(case, arm):
            cell = ok_cell(case, arm)
            if arm == "bold":
                cell["verdict"] = False
                cell["answer_text"] = "alder-1000"
                cell["answer_digest"] = te.digest_text("alder-1000")
            return cell

        records = build(flip_cell)
        for record in records:
            record["id_digests"]["bold"] = record["id_digests"]["control"]
            record["ids_identical"]["bold"] = True
        agg = te.aggregate_candidate(records, cases)
        self.assertEqual(agg["ids"]["bold"]["n_same_ids_verdict_flips"], 24)
        self.assertEqual(agg, dr._typographicemphasis_aggregate(records, cases))

        records = build(ok_cell)
        for record in records:
            record["arm_tokens"]["bold"] = "fifty"
            del record["cells"]["caps"]
            record["cells"]["pc_plain"] = None
            record["cell_details"]["pc_plain"] = "arm_evaluation_failed"
            record["cells"]["pc_info"] = None
            record["cell_details"]["pc_info"] = "arm_context_exceeded"
        agg = te.aggregate_candidate(records, cases)
        self.assertEqual(agg["arms"]["pc_info"]["n_context_exceeded"], 24)
        self.assertEqual(agg, dr._typographicemphasis_aggregate(records, cases))

    def test_mirror_valid_payload_modes_extended(self):
        """Explode, context-exceeded and ids-identical
        payloads are mirror-valid."""
        exploded, _b, _s = _payload("explode")
        self.assertTrue(dr._valid_typographic_emphasis(exploded))
        context, _b, _s = _payload("perfect", max_context=1)
        self.assertTrue(dr._valid_typographic_emphasis(context))
        stripped, _b, _s = _payload("perfect", strip_marks=True)
        self.assertTrue(dr._valid_typographic_emphasis(stripped))

    def test_mirror_tamper_battery_two(self):
        """Second tamper batch: key sets, cell field
        types, token bindings, audit shape, payload
        counts, candidate shapes, aggregate subfields."""
        import copy

        base = _valid_payload()
        entry = lambda p: p["candidates"]["chatml"]  # noqa: E731
        record = lambda p: entry(p)["records"][0]  # noqa: E731
        cell = lambda p: record(p)["cells"]["bold"]  # noqa: E731

        def mutate(fn):
            payload = copy.deepcopy(base)
            fn(payload)
            return payload

        tampers = {
            "record_extra_key": lambda p: record(p).update(extra=1),
            "record_missing_key": lambda p: record(p).pop("topic"),
            "ids_identical_missing_arm": lambda p: record(p)["ids_identical"].pop("caps"),
            "id_digest_malformed": lambda p: record(p)["id_digests"].update(bold="zz"),
            "arm_tokens_bool": lambda p: record(p)["arm_tokens"].update(bold=True),
            "arm_tokens_negative": lambda p: record(p)["arm_tokens"].update(bold=-1),
            "content_digest_wrong_hex": lambda p: record(p)["content_digests"].update(
                bold="0" * 64
            ),
            "cell_verdict_str": lambda p: cell(p).update(verdict="yes"),
            "cell_boundary_match_str": lambda p: cell(p).update(boundary_text_match="x"),
            "cell_input_tokens_negative": lambda p: cell(p).update(input_tokens=-1),
            "cell_answer_digest_malformed": lambda p: cell(p).update(answer_digest="zz"),
            "cell_text_none_with_digest": lambda p: cell(p).update(answer_text=None),
            "cell_input_tokens_mismatch": lambda p: cell(p).update(input_tokens=999),
            "payload_extra_key": lambda p: p.update(extra=1),
            "payload_detail": lambda p: p.update(detail="x"),
            "payload_n_cases": lambda p: p.update(n_cases=23),
            "payload_n_candidates": lambda p: p.update(n_candidates=99),
            "audit_missing_key": lambda p: p["audit"].pop("by_reason"),
            "audit_negative": lambda p: p["audit"].update(n_audit_excluded=-1),
            "audit_findings_str": lambda p: p["audit"].update(special_token_findings="x"),
            "audit_by_reason_list": lambda p: p["audit"].update(by_reason=[]),
            "audit_by_reason_negative": lambda p: p["audit"].update(by_reason={"x": -1}),
            "protocol_seed_bool": lambda p: p["protocol"].update(seed=True),
            "candidates_empty": lambda p: p.update(candidates={}),
            "candidates_too_many": lambda p: p.update(candidates={f"c{i}": {} for i in range(25)}),
            "entry_not_dict": lambda p: p["candidates"].update(chatml="x"),
            "entry_status_weird": lambda p: entry(p).update(status="weird"),
            "entry_detail_wrong": lambda p: entry(p).update(detail="x"),
            "entry_extra_key": lambda p: entry(p).update(extra=1),
            "records_not_list": lambda p: entry(p).update(records={}),
            "records_short": lambda p: entry(p).update(records=entry(p)["records"][:23]),
            "arms_decomposition_tamper": lambda p: entry(p)["arms"]["bold"]["decomposition"].update(
                {"pass": 0}
            ),
            "headline_failing_classes": lambda p: entry(p)["headline"]["bold"][
                "failing_classes"
            ].update({"pass": 1}),
            "headline_pair_decomposition": lambda p: entry(p)["headline"]["bold"][
                "pair_decomposition"
            ].update(mixed=1),
            "headline_control_rate": lambda p: entry(p)["headline"]["bold"].update(
                control_rate=0.5
            ),
            "ids_delta_list_tamper": lambda p: entry(p)["ids"]["bold"].update(
                id_length_delta=[1] * 24
            ),
            "pc_n_pairs_tamper": lambda p: entry(p)["positive_control"].update(n_pairs=23),
            "pc_info_passes_tamper": lambda p: entry(p)["positive_control"].update(info_passes=23),
        }
        for name, fn in tampers.items():
            with self.subTest(tamper=name):
                self.assertFalse(dr._valid_typographic_emphasis(mutate(fn)), name)

    def test_diff_candidate_drift(self):
        """Candidate removal and status drift surface in
        the diff; an unavailable-entry payload is itself
        mirror-valid."""
        import copy

        perfect = _valid_payload()
        trimmed = copy.deepcopy(perfect)
        del trimmed["candidates"]["llama3"]
        trimmed["n_candidates"] = 1
        self.assertTrue(dr._valid_typographic_emphasis(trimmed))
        diff = dr.diff_typographic_emphasis(perfect, trimmed)
        self.assertEqual(diff["candidates_removed"], ["llama3"])
        degraded = copy.deepcopy(perfect)
        degraded["candidates"]["llama3"] = {
            "status": "unavailable",
            "detail": "cancelled",
            "records": [],
        }
        self.assertTrue(dr._valid_typographic_emphasis(degraded))
        diff = dr.diff_typographic_emphasis(perfect, degraded)
        metrics = {(d["candidate"], d["metric"]) for d in diff["candidate_drift"]}
        self.assertIn(("llama3", "status"), metrics)

    def test_mirror_final_guards(self):
        """The mirror's last guard lines: accessor
        fallbacks on direct aggregate calls, the
        boundary-only pair class, cell field-set and
        protocol-type tampers, the regeneration
        exhaustion inside recompute, the unavailable
        shape violation, the records-bytes cap, the
        empty-aggregates rejection, the audit loop arc,
        and the both-unavailable diff arc. (Three lines
        are unreachable by construction and documented
        in VALIDATION instead: the headline loop's
        cell-None continue, subsumed by its paired
        pre-filter; the duplicated arm-token type check,
        subsumed by the before-arithmetic loop; and the
        ground-record total cap, bounded to equality by
        the candidate cap.)"""
        import copy

        cases = te.generate_cases(0)
        record = {
            "fixture": cases[0]["fixture_id"],
            "topic": cases[0]["topic"],
            "expected_digest": te.digest_text(cases[0]["expected"]),
            "content_digests": {},
            "arm_tokens": dict.fromkeys(te.RECORD_ARMS, 50),
            "id_digests": {arm: f"d-{arm}-0" for arm in te.RECORD_ARMS},
            "ids_identical": dict.fromkeys(te.TREATED_ARMS, False),
            "id_length_delta": dict.fromkeys(te.TREATED_ARMS, 0),
            "cells": [],
            "cell_details": [],
        }
        agg = dr._typographicemphasis_aggregate([record], cases)
        self.assertEqual(agg["n_cases"], 1)
        self.assertFalse(dr._typographicemphasis_verdict(record, "bold"))
        self.assertFalse(
            dr._typographicemphasis_ids_identical(
                {"id_digests": {"control": "", "bold": ""}}, "bold"
            )
        )
        ok_record = dict(record)
        ok_record["cells"] = {
            arm: {
                "verdict": arm == "control",
                "boundary_found": arm == "control",
                "boundary_text_match": True,
                "input_tokens": 50,
                "answer_digest": te.digest_text(cases[0]["expected"]),
                "answer_text": cases[0]["expected"],
            }
            for arm in te.RECORD_ARMS
        }
        ok_record["cell_details"] = dict.fromkeys(te.RECORD_ARMS, "")
        agg = dr._typographicemphasis_aggregate([ok_record], cases)
        self.assertEqual(agg["headline"]["bold"]["pair_decomposition"]["boundary_only"], 1)
        self.assertTrue(
            dr._typographicemphasis_cell_ok(
                {
                    "verdict": False,
                    "boundary_found": True,
                    "boundary_text_match": None,
                    "input_tokens": 50,
                    "answer_digest": None,
                    "answer_text": None,
                },
                te.digest_text("glade-2443"),
            )
        )

        base = _valid_payload()

        def mutate(fn):
            payload = copy.deepcopy(base)
            fn(payload)
            return payload

        entry = lambda p: p["candidates"]["chatml"]  # noqa: E731
        record0 = lambda p: entry(p)["records"][0]  # noqa: E731
        tampers = {
            "cell_not_dict": lambda p: record0(p)["cells"].update(bold=[]),
            "protocol_not_dict": lambda p: p.update(protocol=[]),
            "unavailable_records_nonempty": lambda p: p["candidates"].update(
                llama3={
                    "status": "unavailable",
                    "detail": "cancelled",
                    "records": entry(p)["records"],
                }
            ),
            "cell_none_empty_detail": lambda p: record0(p)["cells"].update(bold=None),
        }
        for name, fn in tampers.items():
            with self.subTest(tamper=name):
                self.assertFalse(dr._valid_typographic_emphasis(mutate(fn)), name)

        both_down = copy.deepcopy(base)
        for name in ("chatml", "llama3"):
            both_down["candidates"][name] = {
                "status": "unavailable",
                "detail": "cancelled",
                "records": [],
            }
        self.assertFalse(dr._valid_typographic_emphasis(both_down))

        audited = copy.deepcopy(base)
        audited["audit"]["by_reason"] = {"screen": 0, "other": 2}
        self.assertTrue(dr._valid_typographic_emphasis(audited))

        real_generate = dr._typographicemphasis_generate
        with mock.patch.object(
            dr,
            "_typographicemphasis_generate",
            side_effect=[real_generate(0), dr._TypographicEmphasisMirrorExhausted()],
        ):
            self.assertFalse(dr._valid_typographic_emphasis(copy.deepcopy(base)))

        big = copy.deepcopy(base)
        for cand in big["candidates"].values():
            for rec in cand["records"]:
                for arm_cell in rec["cells"].values():
                    if arm_cell is not None:
                        arm_cell["answer_text"] = "x" * 33000
        self.assertFalse(dr._valid_typographic_emphasis(big))

        degraded = copy.deepcopy(base)
        degraded["candidates"]["llama3"] = {
            "status": "unavailable",
            "detail": "cancelled",
            "records": [],
        }
        diff = dr.diff_typographic_emphasis(degraded, degraded)
        self.assertEqual(diff["candidate_drift"], [])


class TestStageScreens(unittest.TestCase):
    """The stage's fail-closed screens and the analysis
    passthrough shapes."""

    def _stage(self, backend=None, **kwargs):
        backend = backend or _EmphasisBackend("perfect")
        kwargs.setdefault("boundary", backend.encode("endmark"))
        kwargs.setdefault("max_context", 100_000)
        kwargs.setdefault("seed", 0)
        return experiments.typographic_emphasis_stage(backend, (_CHATML,), **kwargs)

    def test_regeneration_exhaustion_screens(self):
        """A regeneration failure screens the stage."""
        with mock.patch.object(te, "generate_cases", side_effect=te.RedrawExhausted):
            stage = self._stage()
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "typographic_emphasis_screen_failed")

    def test_invalid_case_screens(self):
        """A case failing validation screens the stage
        with the case_invariant reason."""
        real = te.validate_case
        with mock.patch.object(
            te,
            "validate_case",
            lambda case: False if case["index"] == 3 else real(case),
        ):
            stage = self._stage()
        self.assertFalse(stage["available"])
        self.assertEqual(stage["audit"]["by_reason"], {"case_invariant": 1})

    def test_panel_distinctness_screens(self):
        """Duplicated panel values screen the stage."""
        cases = te.generate_cases(0)
        cases[1] = dict(cases[1], expected=cases[0]["expected"])
        with mock.patch.object(te, "generate_cases", lambda seed: cases):
            stage = self._stage()
        self.assertFalse(stage["available"])
        self.assertEqual(stage["audit"]["by_reason"], {"panel_distinctness": 1})

    def test_special_token_literal_screens(self):
        """A special-token literal inside a generated
        content screens the stage with the audit count."""
        poisoned = ("<|im_start|> " + te.NEUTRAL_STATEMENTS[0],) + te.NEUTRAL_STATEMENTS[1:]
        with mock.patch.object(te, "NEUTRAL_STATEMENTS", poisoned):
            stage = self._stage()
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "typographic_emphasis_screen_failed")
        self.assertGreaterEqual(stage["audit"]["special_token_findings"], 1)

    def test_boundary_text_containment_screens(self):
        """Boundary text appearing inside a content
        screens the stage."""
        backend = _EmphasisBackend("perfect")
        real_decode = backend.decode
        backend.decode = lambda ids: "inspection code"  # type: ignore[method-assign]
        try:
            stage = self._stage(backend)
        finally:
            backend.decode = real_decode  # type: ignore[method-assign]
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "typographic_emphasis_screen_failed")

    def test_boundary_decode_failure_proceeds(self):
        """An undecodable boundary decodes to the empty
        string and the stage proceeds."""
        backend = _EmphasisBackend("perfect")
        real_decode = backend.decode

        def flaky(ids):
            if list(ids) == list(backend.encode("endmark")):
                raise RuntimeError("undecodable")
            return real_decode(ids)

        backend.decode = flaky  # type: ignore[method-assign]
        try:
            stage = self._stage(backend)
        finally:
            backend.decode = real_decode  # type: ignore[method-assign]
        self.assertTrue(stage["available"])

    def test_cancel_after_evaluate(self):
        """A cancel raised after generation unwinds the
        stage (the cell helper's second cancel check)."""
        import threading

        backend = _EmphasisBackend("perfect")
        cancel = threading.Event()
        real_generate = backend.generate

        def setting_generate(*args, **kwargs):
            out = real_generate(*args, **kwargs)
            cancel.set()
            return out

        backend.generate = setting_generate  # type: ignore[method-assign]
        try:
            stage = self._stage(backend, cancel=cancel)
        finally:
            backend.generate = real_generate  # type: ignore[method-assign]
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_analysis_unavailable_candidate_passthrough(self):
        """Analysis passes an unavailable candidate
        through in the frozen shape."""
        backend = _EmphasisBackend("perfect")
        stage = experiments.typographic_emphasis_stage(
            backend,
            (_CHATML, _LLAMA3),
            boundary=backend.encode("endmark"),
            max_context=100_000,
            seed=0,
        )
        stage["candidates"]["llama3"] = {
            "status": "unavailable",
            "detail": "cancelled",
            "records": [],
        }
        out = experiments.typographic_emphasis_analysis(stage, ("chatml", "llama3"))
        self.assertEqual(
            out["candidates"]["llama3"],
            {"status": "unavailable", "detail": "cancelled", "records": []},
        )


class TestReportingLooseShapes(unittest.TestCase):
    """The reporting section's defensive shapes: n/a
    fallbacks, non-dict entries and arms, unavailable
    rows."""

    def _section(self, entry):
        result = {
            "typographic_emphasis": {
                "available": True,
                "detail": "",
                "protocol": te.protocol_object(0, "fp"),
                "audit": {
                    "n_audit_excluded": 0,
                    "by_reason": {},
                    "special_token_findings": 0,
                },
                "candidates": {"cand": entry},
                "n_candidates": 1,
                "n_cases": 24,
            }
        }
        return reporting._typographic_emphasis_section(result)

    def test_non_dict_and_unavailable_entries(self):
        """Non-dict entries are skipped; unavailable
        entries render their row."""
        self.assertIn("unavailable", self._section({"status": "unavailable"}))
        payload, _b, _s = _payload("perfect")
        entry = payload["candidates"]["chatml"]
        html = self._section(entry)
        self.assertIn("equivalence_certified", html)

    def test_loose_arm_shapes(self):
        """Loose arm/dose/ids shapes skip cleanly; a
        non-numeric headline rate renders n/a."""
        import copy

        payload, _b, _s = _payload("perfect")
        entry = copy.deepcopy(payload["candidates"]["chatml"])
        entry["arms"]["bold"] = 5
        entry["dose"]["caps"] = 5
        entry["ids"]["bold"] = 5
        html = self._section(entry)
        self.assertIn("monospace headline", html)
        entry = copy.deepcopy(payload["candidates"]["chatml"])
        entry["headline"]["bold"]["control_rate"] = "fast"
        entry["headline"]["bold"]["arm_rate"] = None
        html = self._section(entry)
        self.assertIn("n/a", html)

    def test_section_candidates_not_dict(self):
        """A non-dict candidates block renders empty."""
        out = reporting._typographic_emphasis_section(
            {"typographic_emphasis": {"available": True, "candidates": []}}
        )
        self.assertEqual(out, "")

    def test_loose_headline_bits(self):
        """Non-list deltas and non-dict pair/failing/
        blocked shapes render their empty bits."""
        import copy

        payload, _b, _s = _payload("perfect")
        entry = copy.deepcopy(payload["candidates"]["chatml"])
        entry["ids"]["bold"]["id_length_delta"] = "wide"
        entry["headline"]["bold"]["pair_decomposition"] = 5
        entry["headline"]["bold"]["failing_classes"] = 5
        entry["headline"]["bold"]["certification_blocked_by"] = "positive_control"
        html = self._section(entry)
        self.assertIn("bold headline", html)

    def test_blocked_and_fired_sections_render(self):
        """PC-deaf and firing payloads render their
        blocked-gate and label text."""
        deaf, _b, _s = _payload("pcdeaf")
        html = reporting._typographic_emphasis_section({"typographic_emphasis": deaf})
        self.assertIn("positive_control", html)
        fired, _b, _s = _payload("wrongvalue")
        html = reporting._typographic_emphasis_section({"typographic_emphasis": fired})
        self.assertIn("discordant_asymmetric", html)


class TestFinalGuards(unittest.TestCase):
    """The last reachable guards: the mirror's
    ids-identical type fallback and records-bytes cap
    (on a fabricated valid huge payload), the stage's
    second regeneration handler, its post-generation
    bytes cap, and the cell helper's pre-evaluate
    cancel check."""

    def test_ids_identical_non_dict(self):
        """Non-dict id digests are never identical."""
        self.assertFalse(dr._typographicemphasis_ids_identical({"id_digests": []}, "bold"))

    def test_mirror_records_bytes_cap(self):
        """A fabricated valid payload whose records
        exceed the byte cap validates false at the cap,
        not at a record check."""
        import hashlib

        payload, backend, _s = _payload("perfect")
        cases = te.generate_cases(0)
        big_text = "x" * 70000
        big_digest = hashlib.sha256(big_text.encode("utf-8")).hexdigest()
        for entry in payload["candidates"].values():
            for record in entry["records"]:
                for arm_cell in record["cells"].values():
                    arm_cell["answer_text"] = big_text
                    arm_cell["answer_digest"] = big_digest
                    arm_cell["verdict"] = False
            aggregates = te.aggregate_candidate(entry["records"], cases)
            for key, value in aggregates.items():
                entry[key] = value
        self.assertFalse(dr._valid_typographic_emphasis(payload))

    def test_second_regeneration_handler(self):
        """Exhaustion on the stage's second panel
        regeneration screens via the second handler
        (the protocol regeneration consumes the first
        two generations)."""
        cases = te.generate_cases(0)
        backend = _EmphasisBackend("perfect")
        with mock.patch.object(
            te, "generate_cases", side_effect=[cases, cases, te.RedrawExhausted]
        ):
            stage = experiments.typographic_emphasis_stage(
                backend,
                (_CHATML,),
                boundary=backend.encode("endmark"),
                max_context=100_000,
                seed=0,
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "typographic_emphasis_screen_failed")
        self.assertNotEqual(stage["protocol"], {})

    def test_post_generation_bytes_cap(self):
        """Huge answer texts trip the stage's
        post-generation records-bytes cap."""
        backend = _EmphasisBackend("perfect")
        real_generate = backend.generate

        def huge_generate(input_ids, *, max_new_tokens, seed=None, cancel=None):
            out = real_generate(input_ids, max_new_tokens=max_new_tokens, seed=seed)
            text = "x" * 70000
            boundary = backend.encode("endmark")[0]
            ids = list(backend.encode(text)) + [boundary]
            ids = (ids + [boundary] * max_new_tokens)[:max_new_tokens]
            return type(out)(token_ids=tuple(ids), reason="limit")

        backend.generate = huge_generate  # type: ignore[method-assign]
        try:
            stage = experiments.typographic_emphasis_stage(
                backend,
                (_CHATML,),
                boundary=backend.encode("endmark"),
                max_context=100_000,
                seed=0,
            )
        finally:
            backend.generate = real_generate  # type: ignore[method-assign]
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "typographic_emphasis_record_budget_exceeded")

    def test_cancel_before_evaluate(self):
        """A cancel set by the first encode after the
        first completed generation trips the cell
        helper's pre-evaluate check on the next cell."""
        import threading

        backend = _EmphasisBackend("perfect")
        cancel = threading.Event()
        real_encode = backend.encode
        real_generate = backend.generate
        state = {"generated": False, "armed": False}

        def arming_generate(*args, **kwargs):
            out = real_generate(*args, **kwargs)
            state["generated"] = True
            return out

        def setting_encode(text):
            if state["generated"] and not state["armed"]:
                state["armed"] = True
                cancel.set()
            return real_encode(text)

        backend.generate = arming_generate  # type: ignore[method-assign]
        backend.encode = setting_encode  # type: ignore[method-assign]
        try:
            stage = experiments.typographic_emphasis_stage(
                backend,
                (_CHATML,),
                boundary=backend.encode("endmark"),
                max_context=100_000,
                seed=0,
                cancel=cancel,
            )
        finally:
            backend.generate = real_generate  # type: ignore[method-assign]
            backend.encode = real_encode  # type: ignore[method-assign]
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")


if __name__ == "__main__":
    unittest.main()
