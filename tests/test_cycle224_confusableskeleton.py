"""Cycle 224: confusable-skeleton substitution probe
(wave-12 design round 2, BUILD-as-amended, coordinator
#74).

Spec citations are to docs/CONFUSABLESKELETON_DESIGN.md
(the design as amended) and to the adjudication's nine
amendments A1-A9 in the design doc's first section.

Every amendment and build debt has named evidence here:
the consolidated design doc is checked complete (the
rarity definition in full, no truncation markers in any
shipped doc) (A1, build debt 1); the rarity assignment is
frozen by the name-prefix script rule on the regenerated
populations -- Cyrillic seats, including the Cyrillic
Supplement letters d and q, take the Cyrillic control
(A2, build debt 2); the position-aware rarity form_only
predicate is pinned by its battery witness (A3, build
debt 3); the guard battery carries an ids dimension and
the worlds are re-executed under the pinned domain (A8(i),
build debt 4); the redraw path is exercised by a forced
collision (A9, build debt 5); the operating
characteristics are re-derived exactly, including the
exact paired figures 0.004442 / 0.698078 and the
family-wise paired price over 72 comparisons (A6, build
debt 6); the attainable-state cap count is 264 (A4); the
whole-word isolation sentence is pinned verbatim in the
report (A5); the controls' relied-on properties are the
executed ones (A7); and the plateau qualifier is executed
in exact arithmetic (A8(v), with the build-phase figure
correction disclosed in the design doc).
"""

from __future__ import annotations

import copy
import re
import threading
import unicodedata
import unittest
from hashlib import sha256
from math import comb
from pathlib import Path
from statistics import NormalDist
from unittest import mock

from model_protocol_lab import canonicalform, density, diffreports, experiments, reporting
from model_protocol_lab import confusableskeleton as cs
from model_protocol_lab.core import Format, Generation

_CHATML = Format("chatml", "", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n")
_LLAMA3 = Format(
    "llama3",
    "<|begin_of_text|>",
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|eot_id|>",
)
_NORM = NormalDist()
_INVERSE = {replacement: letter for letter, replacement in cs.FROZEN_REPLACEMENTS.items()}


def _restore(word: str) -> str:
    """Map a treated word back to its Latin original: frozen
    replacements via the inverse table; a rarity control via
    the unique eligible letter that restores a pool word.
    """
    mapped = "".join(_INVERSE.get(char, char) for char in word)
    if not any(char in cs.CONTROL_CHARACTERS for char in mapped):
        return mapped
    for index, char in enumerate(mapped):
        if char in cs.CONTROL_CHARACTERS:
            for letter in cs.FROZEN_REPLACEMENTS:
                candidate = mapped[:index] + letter + mapped[index + 1 :]
                if candidate in cs.WORDS or candidate in cs.TOPICS:
                    return candidate
    return mapped


class _WordBackend:
    """Word-token scripted backend.

    Modes: ``perfect`` (the expected value read out of the
    fact sentence with the substitutions folded back to
    Latin, boundary on), ``echo`` (the value exactly as
    rendered, substitutions included), ``garbage`` (a fixed
    non-answer), ``noboundary`` (the expected answer with
    no boundary token), ``explode`` (generate raises),
    ``long_nonascii`` (a 2,048-character non-ASCII answer).
    With ``fold=True`` the encoder applies the UTS #39
    skeleton fold over the frozen tables (replacements map
    to their Latin letters), so the two confusable arms are
    token-identical to the control -- the screening-tokenizer
    regime; the rarity arm's skeleton genuinely differs, so
    it stays divergent.
    """

    fingerprint = "sha256:test-confusable-skeleton"

    def __init__(self, mode: str = "perfect", fold: bool = False) -> None:
        """Init."""
        self.mode = mode
        self.fold = fold
        self.calls = 0
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
        """Encode."""
        if self.fold:
            text = "".join(_INVERSE.get(char, char) for char in text)
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
        match = re.search(r"inspection code is (\S+?)\.", text)
        raw = match.group(1) if match else "unknownword"
        if self.mode == "perfect":
            word, _, number = raw.partition("-")
            answer = _restore(word) + "-" + number
        elif self.mode == "garbage":
            answer = "unknownword"
        elif self.mode == "long_nonascii":
            answer = "あ" * 2048
        else:
            answer = raw
        boundary = self.encode("endmark")[0]
        out = list(self.encode(answer))
        if self.mode != "noboundary":
            out.append(boundary)
        pad = boundary if self.mode != "noboundary" else self.encode("fillerword")[0]
        out = (out + [pad] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(mode="perfect", fold=False, formats=(_CHATML, _LLAMA3), **stage_kwargs):
    """Payload."""
    backend = _WordBackend(mode, fold=fold)
    stage_kwargs.setdefault("boundary", backend.encode("endmark"))
    stage_kwargs.setdefault("max_context", 100_000)
    stage_kwargs.setdefault("seed", 0)
    stage = experiments.confusable_skeleton_stage(backend, formats, **stage_kwargs)
    names = tuple(fmt.name for fmt in formats)
    return experiments.confusable_skeleton_analysis(stage, names), backend, stage


def _cell(text=None, *, verdict=None, boundary=True, tokens=100):
    """Cell."""
    digest = None if text is None else sha256(text.encode("utf-8")).hexdigest()
    if verdict is None:
        verdict = bool(boundary and text is not None and digest is not None)
    return {
        "verdict": verdict,
        "boundary_found": boundary,
        "boundary_text_match": True,
        "input_tokens": tokens,
        "answer_digest": digest,
        "answer_text": text,
    }


_EXPECTED0 = "urn-2133"
_DIGEST0 = sha256(_EXPECTED0.encode("utf-8")).hexdigest()


class TestGeneratorFreeze(unittest.TestCase):
    """The frozen derivation, panel, population, and arms
    (design sections 2-4; A1, A2, A7, A9)."""

    def test_generator_spec_text_pinned(self):
        """Generator spec text pinned."""
        self.assertIn("confusableskeleton", cs.GENERATOR_SPEC)
        self.assertIn("int.from_bytes(", cs.GENERATOR_SPEC)
        self.assertIn("digest()[:8], 'big'", cs.GENERATOR_SPEC)
        self.assertEqual(cs.GENERATOR_SPEC, diffreports._CONFUSABLESKELETON_GENERATOR_SPEC)

    def test_derive_pinned(self):
        """Derive pinned."""
        self.assertEqual(cs._derive(0, 0, 0, 0), 2236273561580164556)

    def test_seed_zero_panel_literals(self):
        """Seed zero panel literals."""
        cases = cs.generate_cases(0)
        self.assertEqual(len(cases), 24)
        self.assertEqual(cases[0]["value"], "urn-2133")
        self.assertEqual(cases[0]["neutral_indices"], [0, 14])
        self.assertEqual(cases[1]["value"], "reed-4266")
        self.assertEqual(cases[1]["neutral_indices"], [16, 4])
        self.assertEqual(cases[2]["value"], "fern-2775")
        self.assertEqual(cases[2]["neutral_indices"], [5, 19])
        self.assertEqual(cases[23]["value"], "heath-1163")
        self.assertEqual(cases[23]["neutral_indices"], [12, 8])
        self.assertEqual(
            [case["topic"] for case in cases],
            list(cs.TOPICS),
        )

    def test_five_seed_panels_literally(self):
        """Five seed panels literally."""
        pins = {
            1: ("cedar-6627", [16, 2], "kelp-6918", "moss-4627"),
            7: ("quill-2523", [7, 17], "elm-3467", "nova-7497"),
            42: ("yew-7893", [20, 9], "sage-7191", "loom-3395"),
            20261030: ("yew-4501", [22, 7], "basalt-7085", "jade-9184"),
        }
        for seed, (value0, neutrals0, value1, value23) in pins.items():
            cases = cs.generate_cases(seed)
            self.assertEqual(cases[0]["value"], value0, seed)
            self.assertEqual(cases[0]["neutral_indices"], neutrals0, seed)
            self.assertEqual(cases[1]["value"], value1, seed)
            self.assertEqual(cases[23]["value"], value23, seed)

    def test_panels_distinct_and_deterministic(self):
        """Panels distinct and deterministic."""
        self.assertEqual(cs.generate_cases(7), cs.generate_cases(7))
        self.assertNotEqual(cs.generate_cases(0), cs.generate_cases(1))
        for seed in (0, 1, 7, 42, 20261030):
            values = [case["value"] for case in cs.generate_cases(seed)]
            self.assertEqual(len(set(values)), 24, seed)

    def test_forced_collision_redraws_value_slots_only(self):
        """A9 / build debt 5: force case 1's counter-0 draws
        onto case 0's value; the redraw must land at
        counter+1 for slots 0+1 while the neutrals stay at
        counter 0."""
        original = cs._derive

        def colliding(seed, index, slot, counter):
            """Colliding."""
            if index == 1 and slot in (0, 1) and counter == 0:
                return original(seed, 0, slot, 0)
            return original(seed, index, slot, counter)

        with mock.patch.object(cs, "_derive", colliding):
            cases = cs.generate_cases(0)
        case0, case1 = cases[0], cases[1]
        self.assertNotEqual(case1["value"], case0["value"])
        word = cs.WORDS[original(0, 1, 0, 1) % len(cs.WORDS)]
        number = 1000 + original(0, 1, 1, 1) % 9000
        self.assertEqual(case1["value"], f"{word}-{number}")
        first = original(0, 1, 2, 0) % len(cs.NEUTRAL_STATEMENTS)
        second = original(0, 1, 3, 0) % len(cs.NEUTRAL_STATEMENTS)
        if second == first:
            second = (first + 1) % len(cs.NEUTRAL_STATEMENTS)
        self.assertEqual(case1["neutral_indices"], [first, second])
        self.assertEqual(len({case["value"] for case in cases}), 24)

    def test_redraw_exhaustion_fails_closed(self):
        """Redraw exhaustion fails closed."""
        with (
            mock.patch.object(cs, "_derive", lambda *args: 7),
            self.assertRaises(cs._RedrawExhausted),
        ):
            cs.generate_cases(0)

    def test_frozen_population_literals(self):
        """Frozen population literals (design section 2)."""
        self.assertEqual(cs.FROZEN_REPLACEMENTS["a"], "\u03b1")
        self.assertEqual(cs.FROZEN_REPLACEMENTS["d"], "\u0501")
        self.assertEqual(cs.FROZEN_REPLACEMENTS["q"], "\u051b")
        self.assertEqual(cs.FROZEN_REPLACEMENTS["r"], "\u0433")
        self.assertEqual(cs.FROZEN_REPLACEMENTS["y"], "\u03b3")
        self.assertEqual(len(cs.FROZEN_REPLACEMENTS), 16)
        self.assertEqual(cs.ZERO_ELIGIBLE, ("b", "f", "g", "k", "l", "m", "n", "t", "z"))
        self.assertEqual(cs.POPULATION_COUNTS["l"], (73, 0))
        self.assertEqual(cs.POPULATION_COUNTS["o"], (75, 3))
        self.assertEqual(cs.CONFUSABLES_VERSION, "15.0.0")
        self.assertEqual(unicodedata.unidata_version, "15.0.0")
        self.assertEqual(
            cs.CONFUSABLES_SHA256,
            "2b10130885c3370b101c52d7baedc452ab7f0e257b86c1e52ee657ecfc29ce64",
        )

    def test_rarity_assignment_is_the_name_prefix_rule(self):
        """A2 / build debt 2: the per-letter control is the
        same-script control BY NAME PREFIX -- the Cyrillic
        Supplement seats d and q take the Cyrillic control,
        which the design round's code-point-range proxy got
        wrong."""
        self.assertEqual(cs.RARITY_CONTROLS["d"], "\u0434")
        self.assertEqual(cs.RARITY_CONTROLS["q"], "\u0434")
        self.assertEqual(cs.RARITY_CONTROLS["a"], "\u03b6")
        for letter, replacement in cs.FROZEN_REPLACEMENTS.items():
            script = unicodedata.name(replacement).split(" ")[0]
            control = cs.RARITY_CONTROLS[letter]
            self.assertEqual(unicodedata.name(control).split(" ")[0], script, letter)
            self.assertEqual(
                control,
                "\u0434" if script == "CYRILLIC" else "\u03b6",
                letter,
            )

    def test_rarity_populations_script_agreement_all_seeds(self):
        """A2 / build debt 2: on the regenerated rarity
        populations of all five evidence seeds, every rarity
        site character's script (by name prefix) equals the
        frozen replacement's script at that site, in the
        topic and in the value word."""
        for seed in (0, 1, 7, 42, 20261030):
            for case in cs.generate_cases(seed):
                word = case["value"].partition("-")[0]
                for original_word in (case["topic"], word):
                    treated = cs.treated_word(original_word, "rarity_control")
                    single = cs.treated_word(original_word, "single_site")
                    for before, site_char, single_char in zip(
                        original_word, treated, single, strict=True
                    ):
                        if site_char != before:
                            self.assertEqual(
                                unicodedata.name(site_char).split(" ")[0],
                                unicodedata.name(single_char).split(" ")[0],
                                (seed, original_word),
                            )
                            self.assertIn(site_char, cs.CONTROL_CHARACTERS)

    def test_cyrillic_seat_words_take_cyrillic_control(self):
        """A2 witness: quarry's first eligible letter is q
        (Cyrillic seat) -- its rarity site carries U+0434,
        not the Greek control."""
        self.assertEqual(cs.treated_word("quarry", "rarity_control"), "\u0434uarry")
        self.assertEqual(cs.treated_word("quartz", "rarity_control"), "\u0434uartz")
        self.assertEqual(cs.treated_word("alder", "rarity_control"), "\u03b6lder")

    def test_population_audit_properties(self):
        """A7: the relied-on control properties, executed:
        each control is skeleton-distinct from every panel
        letter and frozen replacement under the frozen
        tables, its skeleton form is itself, and the audit
        conjunction holds."""
        audit = cs.POPULATION_AUDIT
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["unicode_version"], "15.0.0")
        for code_point, entry in audit["controls"].items():
            self.assertTrue(entry["satisfies"], code_point)
            self.assertTrue(entry["skeleton_distinct"], code_point)
        self.assertEqual(audit["controls"]["U+0434"]["name"], "CYRILLIC SMALL LETTER DE")
        self.assertEqual(audit["controls"]["U+03B6"]["name"], "GREEK SMALL LETTER ZETA")
        for letter, entry in audit["letters"].items():
            self.assertTrue(entry["satisfies"], letter)
        self.assertTrue(audit["every_word_eligible"])
        self.assertTrue(audit["skeleton_invariance"])
        self.assertTrue(audit["rarity_skeleton_differs"])
        self.assertTrue(audit["length_preserved"])
        self.assertEqual(cs.skeleton_form("m"), "rn")
        self.assertEqual(cs.skeleton_form(cs.FROZEN_REPLACEMENTS["a"]), "a")

    def test_arm_content_structure(self):
        """Arm content structure (design section 4): doses,
        skeleton invariance, byte-identical carriers."""
        for seed in (0, 42):
            for case in cs.generate_cases(seed):
                control = cs.arm_content(case, "control")
                self.assertTrue(control.isascii())
                question = cs.question_text(case)
                word = case["value"].partition("-")[0]
                for arm in cs.TREATED_ARMS:
                    content = cs.arm_content(case, arm)
                    self.assertTrue(content.endswith(question), arm)
                    self.assertNotEqual(content, control, arm)
                for arm in ("single_site", "rarity_control"):
                    for original in (case["topic"], word):
                        treated = cs.treated_word(original, arm)
                        dose = sum(1 for a, b in zip(original, treated, strict=True) if a != b)
                        self.assertEqual(dose, 1, (arm, original))
                        self.assertEqual(len(treated), len(original))
                for original in (case["topic"], word):
                    treated = cs.treated_word(original, "whole_word")
                    dose = sum(1 for a, b in zip(original, treated, strict=True) if a != b)
                    self.assertEqual(dose, cs.eligible_count(original), original)
                    self.assertEqual(cs.skeleton_form(treated), cs.skeleton_form(original))
                    rarity = cs.treated_word(original, "rarity_control")
                    self.assertNotEqual(cs.skeleton_form(rarity), cs.skeleton_form(original))

    def test_whole_word_dose_seed_zero(self):
        """The printed whole-word dose (design section 4):
        171 of 249 letters on seed 0's panel."""
        aggregate = cs.aggregate_candidate([], cs.generate_cases(0))
        self.assertEqual(aggregate["dose"]["whole_word"]["eligible_letters"], 171)
        self.assertEqual(aggregate["dose"]["whole_word"]["total_letters"], 249)
        self.assertAlmostEqual(aggregate["dose"]["whole_word"]["fraction"], 0.6867, places=4)

    def test_validate_case_accepts_and_rejects(self):
        """Validate case accepts and rejects."""
        cases = cs.generate_cases(0)
        for case in cases:
            self.assertTrue(cs.validate_case(case, 0))
        tampered = copy.deepcopy(cases[0])
        tampered["value"] = "urn-2134"
        self.assertFalse(cs.validate_case(tampered, 0))
        tampered = copy.deepcopy(cases[0])
        tampered["topic"] = "summit"
        self.assertFalse(cs.validate_case(tampered, 0))
        tampered = copy.deepcopy(cases[0])
        tampered["neutral_indices"] = [3, 3]
        self.assertFalse(cs.validate_case(tampered, 0))
        tampered = copy.deepcopy(cases[0])
        tampered["fixture_id"] = "confusableskeleton-99"
        self.assertFalse(cs.validate_case(tampered, 0))
        tampered = copy.deepcopy(cases[0])
        tampered["expected"] = "urn-9999"
        self.assertFalse(cs.validate_case(tampered, 0))
        self.assertFalse(cs.validate_case(cases[0], 1))
        self.assertFalse(cs.validate_case({"fixture_id": "bogus"}, 0))

    def test_literal_copies_pinned(self):
        """Literal copies pinned (drift alarms)."""
        self.assertEqual(tuple(canonicalform.WORDS), cs.WORDS)
        self.assertEqual(tuple(density.NEUTRAL_STATEMENTS), cs.NEUTRAL_STATEMENTS)
        self.assertEqual(cs.TOPICS, diffreports._CONFUSABLESKELETON_TOPICS)
        self.assertEqual(
            cs.FROZEN_REPLACEMENTS, diffreports._CONFUSABLESKELETON_FROZEN_REPLACEMENTS
        )
        self.assertEqual(cs.RARITY_CONTROLS, diffreports._CONFUSABLESKELETON_RARITY_CONTROLS)

    def test_protocol_fingerprint_stability(self):
        """Protocol fingerprint stability."""
        first = cs.protocol_fingerprint(0, "sha256:test")
        self.assertEqual(first, cs.protocol_fingerprint(0, "sha256:test"))
        self.assertNotEqual(first, cs.protocol_fingerprint(1, "sha256:test"))
        protocol = cs.protocol_object(0, "sha256:test")
        self.assertEqual(protocol["table_sha256"], cs.table_sha256())
        self.assertEqual(protocol["confusables_version"], "15.0.0")
        self.assertEqual(protocol["generation_budget"], 2304)
        self.assertEqual(protocol["max_ground_records"], 576)


class TestDesignDocCompleteness(unittest.TestCase):
    """A1 / build debt 1: the consolidated design doc is the
    build contract -- the rarity definition in full, and no
    truncation markers in any shipped doc."""

    def test_design_doc_rarity_definition_in_full(self):
        """Design doc rarity definition in full."""
        text = Path("docs/CONFUSABLESKELETON_DESIGN.md").read_text(encoding="utf-8")
        for needle in (
            "U+0434",
            "U+03B6",
            "CYRILLIC SMALL LETTER DE",
            "GREEK SMALL LETTER ZETA",
            "| a | α U+03B1 (Greek) | ζ U+03B6 |",
            "| d | ԁ U+0501 (Cyrillic) | д U+0434 |",
            "| q | ԛ U+051B (Cyrillic) | д U+0434 |",
            "discordant_asymmetric",
            "0.004442",
            "0.698078",
            "0.3199",
            "264",
        ):
            self.assertIn(needle, text)

    def test_no_truncation_markers_in_shipped_docs(self):
        """No truncation markers in shipped docs: the
        consolidated doc is clean, and the set of shipped
        docs carrying a marker does not grow. Two historical
        design docs carry markers from their own cycles'
        generation process (named here so the set is pinned,
        not silently absorbed): HISTORYFRAME_DESIGN.md and
        INSTRUCTIONDISTANCE_DESIGN.md -- pre-existing
        conditions outside this cycle's scope, disclosed in
        the cycle-224 final report."""
        known = {"HISTORYFRAME_DESIGN.md", "INSTRUCTIONDISTANCE_DESIGN.md"}
        carrying = set()
        for path in sorted(Path("docs").rglob("*.md")):
            if "[truncated" in path.read_text(encoding="utf-8"):
                carrying.add(path.name)
        self.assertEqual(carrying, known)
        text = Path("docs/CONFUSABLESKELETON_DESIGN.md").read_text(encoding="utf-8")
        self.assertNotIn("[truncated", text)


class TestDecomposition(unittest.TestCase):
    """The ordered decision list (design section 5; A3,
    A8(iv))."""

    def _decomp(self, text, arm, *, verdict=False, boundary=True, digest=None):
        """Decomp."""
        cell = _cell(text=text, verdict=verdict, boundary=boundary)
        if digest is not None:
            cell["answer_digest"] = digest
        return cs.decompose_cell(cell, _EXPECTED0, _DIGEST0, arm)

    def test_precedence_pass_then_boundary(self):
        """Precedence pass then boundary."""
        self.assertEqual(self._decomp(_EXPECTED0, "control", verdict=True), "pass")
        self.assertEqual(
            self._decomp(_EXPECTED0, "control", verdict=False, boundary=False),
            "boundary_only",
        )
        self.assertEqual(
            self._decomp("cedar-5678", "single_site", digest=_DIGEST0),
            "boundary_only",
        )

    def test_wrong_value_grammar_scope(self):
        """Wrong value grammar scope."""
        self.assertEqual(self._decomp("cedar-5678", "single_site"), "wrong_value")
        self.assertEqual(self._decomp("cedar-1000", "whole_word"), "wrong_value")
        for shape in ("unknown-1234", "cedar-999", "cedar-10000", "CEDAR-5678", "cedar_5678"):
            self.assertEqual(self._decomp(shape, "single_site"), "non_answer", shape)

    def test_echo_classes_per_arm(self):
        """Echo classes per arm (expected urn-2133: the
        value word is urn; its single-site echo replaces u,
        its whole-word echo replaces u and r)."""
        self.assertEqual(self._decomp("\u03c5rn-2133", "single_site"), "echo_confusable")
        self.assertEqual(self._decomp("\u03c5\u0433n-2133", "whole_word"), "echo_confusable")
        self.assertEqual(self._decomp("\u03b6rn-2133", "rarity_control"), "echo_control")
        # The single-site echo under the whole-word arm is
        # one substitution short of that arm's echo; the
        # frozen-table inverse still restores the value, so
        # it lands in form_only. The rarity echo under a
        # confusable arm is not in the frozen table at all.
        self.assertEqual(self._decomp("\u03c5rn-2133", "whole_word"), "form_only")
        self.assertEqual(self._decomp("\u03b6rn-2133", "single_site"), "non_answer")
        self.assertEqual(self._decomp("\u03c5rn-2133", "rarity_control"), "non_answer")

    def test_whole_word_echo_vs_single_site_echo(self):
        """Whole-word echo is not the single-site echo:
        expected alder-1234 (whole-word echo replaces every
        eligible letter)."""
        digest = sha256(b"alder-1234").hexdigest()
        whole_echo = cs.treated_word("alder", "whole_word") + "-1234"
        single_echo = cs.treated_word("alder", "single_site") + "-1234"
        self.assertNotEqual(whole_echo, single_echo)
        cell = _cell(text=whole_echo, verdict=False)
        self.assertEqual(
            cs.decompose_cell(cell, "alder-1234", digest, "whole_word"), "echo_confusable"
        )
        cell = _cell(text=single_echo, verdict=False)
        self.assertEqual(
            cs.decompose_cell(cell, "alder-1234", digest, "whole_word"),
            "form_only",
        )

    def test_a3_witness_position_aware_inverse(self):
        """A3 / build debt 3: the witness -- d-l-d-e-r with
        the Cyrillic control at position 0 against expected
        alder-1234 (a Greek-seat word): form_only on the
        rarity arm by the position-aware predicate (the
        control at the position stands for the expected
        value's character there), non_answer on single_site
        (its form step knows only the frozen table)."""
        digest = sha256(b"alder-1234").hexdigest()
        cell = _cell(text="\u0434lder-1234", verdict=False)
        self.assertEqual(
            cs.decompose_cell(cell, "alder-1234", digest, "rarity_control"),
            "form_only",
        )
        self.assertEqual(
            cs.decompose_cell(cell, "alder-1234", digest, "single_site"),
            "non_answer",
        )
        # A control at a non-site position also satisfies the
        # position-aware predicate (the predicate is
        # positional, not site-bound).
        cell = _cell(text="al\u0434er-1234", verdict=False)
        self.assertEqual(
            cs.decompose_cell(cell, "alder-1234", digest, "rarity_control"),
            "form_only",
        )
        # But a wrong Latin letter anywhere breaks it.
        cell = _cell(text="\u0434ldxr-1234", verdict=False)
        self.assertEqual(
            cs.decompose_cell(cell, "alder-1234", digest, "rarity_control"),
            "non_answer",
        )

    def test_form_only_frozen_table_scope(self):
        """Form-only on the confusable arms maps the frozen
        table back; A8(iv): an echo in a NON-frozen
        skeleton-equal letter (Cyrillic a, U+0430, against
        expected alder-1234) is non_answer, never form_only."""
        self.assertEqual(self._decomp("\u03c5\u0433n-2133", "single_site"), "form_only")
        digest = sha256(b"alder-1234").hexdigest()
        cell = _cell(text="\u0430lder-1234", verdict=False)
        self.assertEqual(cs.decompose_cell(cell, "alder-1234", digest, "single_site"), "non_answer")
        self.assertEqual(cs.decompose_cell(cell, "alder-1234", digest, "whole_word"), "non_answer")

    def test_control_arm_has_no_echo_or_form(self):
        """Control arm has no echo or form classes."""
        self.assertEqual(self._decomp("\u03c5rn-2133", "control"), "non_answer")
        self.assertEqual(self._decomp("\u03b6rn-2133", "control"), "non_answer")

    def test_missing_text_is_non_answer(self):
        """Missing text is non answer."""
        cell = _cell(text=None, verdict=False, boundary=False)
        cell["answer_digest"] = None
        self.assertEqual(cs.decompose_cell(cell, _EXPECTED0, _DIGEST0, "single_site"), "non_answer")

    def test_pair_decomposition(self):
        """Pair decomposition."""
        control = _cell(text=_EXPECTED0, verdict=True)
        same = _cell(text=_EXPECTED0, boundary=False)
        self.assertEqual(cs.decompose_pair(control, same), "boundary_only")
        other = _cell(text="cedar-5678")
        self.assertEqual(cs.decompose_pair(control, other), "exact_only")
        no_boundary = _cell(text="cedar-5678", boundary=False)
        self.assertEqual(cs.decompose_pair(control, no_boundary), "mixed")


def _base_rule(d: int, j: int) -> str:
    """The unguarded label rule (test-local oracle piece)."""
    if d < cs.MIN_D:
        return "insufficient"
    if j >= cs.MIN_J:
        return "discordant_asymmetric"
    if j == 0:
        return "no_discordance_observed"
    return "below_threshold"


def _oracle_label(d: int, j: int, k: int) -> str:
    """Independent worst-case-completion oracle for
    classify_headline: enumerate every completion of the
    k missing pairs and require the acquittal to be
    completion-stable; fires issue on the observed counts;
    the cap and the D floor refuse."""
    if k > cs.MAX_INCOMPLETE:
        return "insufficient"
    if d < cs.MIN_D:
        return "insufficient"
    if j >= cs.MIN_J:
        return "discordant_asymmetric"
    outcomes = set()
    for a in range(k + 1):
        for b in range(k + 1 - a):
            outcomes.add(_base_rule(d + a + b, j + a))
    if len(outcomes) == 1:
        only = outcomes.pop()
        if only in ("no_discordance_observed", "below_threshold"):
            return only
    return "insufficient"


class TestClassifyGuard(unittest.TestCase):
    """The guarded headline against an independent
    completion oracle; the cap count of amendment A4."""

    def test_classify_matches_oracle_on_grid(self):
        """Classify matches oracle on grid."""
        for d in range(0, 16):
            for j in range(0, 9):
                for k in range(0, 9):
                    self.assertEqual(
                        cs.classify_headline(d, j, k),
                        _oracle_label(d, j, k),
                        (d, j, k),
                    )

    def test_named_rows(self):
        """Named rows."""
        self.assertEqual(cs.classify_headline(24, 6, 0), "discordant_asymmetric")
        self.assertEqual(cs.classify_headline(10, 6, 6), "discordant_asymmetric")
        self.assertEqual(cs.classify_headline(24, 0, 0), "no_discordance_observed")
        self.assertEqual(cs.classify_headline(24, 0, 1), "insufficient")
        self.assertEqual(cs.classify_headline(24, 5, 0), "below_threshold")
        self.assertEqual(cs.classify_headline(24, 5, 1), "insufficient")
        self.assertEqual(cs.classify_headline(24, 1, 4), "below_threshold")
        self.assertEqual(cs.classify_headline(9, 6, 0), "insufficient")
        self.assertEqual(cs.classify_headline(24, 6, 7), "insufficient")
        self.assertEqual(cs.MIN_D, 10)
        self.assertEqual(cs.MIN_J, 6)
        self.assertEqual(cs.MAX_INCOMPLETE, 6)

    def test_a4_cap_state_count_is_264(self):
        """A4: over the reachable state space (D + k <= 24),
        exactly 264 states are completion-stable fires the
        cap suppresses (D >= MIN_D, MIN_J <= J <= D -- J
        counts among the control passes -- k > cap)."""
        count = 0
        for d in range(0, 25):
            for j in range(0, 25):
                for k in range(0, 25):
                    if d + k > 24:
                        continue
                    if d >= cs.MIN_D and cs.MIN_J <= j <= d and k > cs.MAX_INCOMPLETE:
                        count += 1
                        self.assertEqual(cs.classify_headline(d, j, k), "insufficient")
        self.assertEqual(count, 264)


def _record(case, cells, details, identical_arms=()):
    """Record."""
    digests = {arm: f"digest-{arm}" for arm in cs.ARM_KEYS}
    digest_control = digests["control"]
    for arm in identical_arms:
        digests[arm] = digest_control
    return {
        "fixture": case["fixture_id"],
        "topic": case["topic"],
        "expected_digest": cs.digest_text(case["expected"]),
        "content_digests": dict.fromkeys(cs.ARM_KEYS, "x" * 64),
        "arm_tokens": dict.fromkeys(cs.ARM_KEYS, 100),
        "id_digests": digests,
        "ids_identical": {arm: arm in identical_arms for arm in cs.TREATED_ARMS},
        "id_length_delta": dict.fromkeys(cs.TREATED_ARMS, 0),
        "table_pin": {"version": cs.CONFUSABLES_VERSION, "table_sha256": cs.table_sha256()},
        "cells": cells,
        "cell_details": details,
    }


class TestGuardBatteryIdsDimension(unittest.TestCase):
    """Build debt 4 / A8(i): the completion domain is the
    id-divergent domain; ids-identical incomplete cases
    never enter the completion."""

    def _cells(self, case, verdict=True, missing=()):
        """Cells."""
        cells = {}
        details = {}
        for arm in cs.ARM_KEYS:
            if arm in missing:
                cells[arm] = None
                details[arm] = "arm_evaluation_failed"
            else:
                cells[arm] = _cell(text=case["expected"], verdict=verdict, tokens=100)
                details[arm] = ""
        return cells, details

    def test_ids_identical_incomplete_never_enters(self):
        """Ids identical incomplete never enters."""
        cases = cs.generate_cases(0)
        records = []
        for case in cases:
            cells, details = self._cells(case, missing=("single_site",))
            records.append(_record(case, cells, details, identical_arms=("single_site",)))
        aggregate = cs.aggregate_candidate(records, cases)
        head = aggregate["headline"]["single_site"]
        self.assertEqual(head["n_population"], 0)
        self.assertEqual(head["n_incomplete"], 0)
        self.assertEqual(head["label"], "insufficient")
        self.assertEqual(aggregate["ids"]["single_site"]["n_ids_identical"], 24)
        # A divergent arm on the same records headlines
        # normally (its cells are all present).
        whole = aggregate["headline"]["whole_word"]
        self.assertEqual(whole["n_population"], 24)
        self.assertEqual(whole["label"], "no_discordance_observed")

    def test_mixed_domain_counts_only_divergent_incompletion(self):
        """Mixed domain counts only divergent incompletion."""
        cases = cs.generate_cases(0)
        records = []
        for index, case in enumerate(cases):
            if index < 6:
                cells, details = self._cells(case, missing=("single_site",))
                records.append(_record(case, cells, details, identical_arms=("single_site",)))
            elif index < 8:
                cells, details = self._cells(case, missing=("single_site",))
                records.append(_record(case, cells, details))
            else:
                cells, details = self._cells(case)
                records.append(_record(case, cells, details))
        aggregate = cs.aggregate_candidate(records, cases)
        head = aggregate["headline"]["single_site"]
        self.assertEqual(head["n_population"], 16)
        self.assertEqual(head["n_incomplete"], 2)
        self.assertEqual(head["label"], "insufficient")
        self.assertEqual(aggregate["n_incomplete"], 8)


def _worlds_ids(cap_trials: int = 4000, seed: int = 20261030) -> dict[str, dict[str, float]]:
    """Worlds A-C re-executed under the PINNED domain (build
    debt 4): each case carries an ids-divergence flag
    (divergent unless index % 5 == 4); incompletion counts
    toward the guard only on divergent cases, and the
    guarded D/J are taken over the divergent domain. The
    unguarded comparator ranges over all cases, as the
    design-phase worlds did."""
    import random

    rng = random.Random(seed)  # noqa: S311 -- deterministic test seed
    results: dict[str, dict[str, float]] = {}
    for world in ("A", "B", "C"):
        counts = {"guarded_fire": 0, "guarded_below": 0, "unguarded_fire": 0, "unguarded_below": 0}
        for _ in range(cap_trials):
            divergent = [i % 5 != 4 for i in range(24)]
            cpass = [rng.random() < 0.90 for _ in range(24)]
            q = 0.10 if world == "A" else 0.50
            afail = [cp and rng.random() < q for cp in cpass]
            if world == "A":
                inc = [i < 12 and cpass[i] and not afail[i] for i in range(24)]
            elif world == "B":
                inc = [afail[i] and rng.random() < 0.6 for i in range(24)]
            else:
                inc = [rng.random() < 0.15 for _ in range(24)]
            d_all = sum(1 for i in range(24) if cpass[i] and not inc[i])
            j_all = sum(1 for i in range(24) if afail[i] and not inc[i])
            d_dom = sum(1 for i in range(24) if divergent[i] and cpass[i] and not inc[i])
            j_dom = sum(1 for i in range(24) if divergent[i] and afail[i] and not inc[i])
            n_dom = sum(1 for i in range(24) if divergent[i] and inc[i])
            guarded = cs.classify_headline(d_dom, j_dom, n_dom)
            unguarded = _base_rule(d_all, j_all)
            counts["guarded_fire"] += guarded == "discordant_asymmetric"
            counts["guarded_below"] += guarded == "below_threshold"
            counts["unguarded_fire"] += unguarded == "discordant_asymmetric"
            counts["unguarded_below"] += unguarded == "below_threshold"
        results[world] = {key: value / cap_trials for key, value in counts.items()}
    return results


class TestWorldsAtPinnedDomain(unittest.TestCase):
    """Build debt 4: worlds A-C under the pinned
    id-divergent domain."""

    def test_worlds(self):
        """Worlds."""
        worlds = _worlds_ids()
        # World A (zero effect, sign-concentrated
        # incompletion): the guarded fire stays at the floor
        # (design-phase guarded 0.0055 over all cases).
        self.assertLess(worlds["A"]["guarded_fire"], 0.02)
        # World B (a true drop, 60 percent of would-be
        # discordants removed): the guarded acquittal nearly
        # vanishes against the unguarded (0.7277). Under the
        # pinned domain the executed guarded rate is 0.0548,
        # above the design phase's all-cases 0.0117: the
        # restricted domain shrinks the completion count
        # alongside the observed counts, so the stability
        # inequality is met slightly more often -- the
        # direction and the order of magnitude are the
        # design's, and the band is set from the execution.
        self.assertLess(worlds["B"]["guarded_below"], 0.10)
        self.assertGreater(worlds["B"]["unguarded_below"], 0.5)
        # World C (the same drop, flat incompletion): the
        # guarded fire survives (0.8893).
        self.assertGreater(worlds["C"]["guarded_fire"], 0.80)
        self.assertLessEqual(worlds["C"]["guarded_fire"], worlds["C"]["unguarded_fire"] + 0.02)


def _fire_independent(a: float, q: float, n: int = 24, min_d: int = 10) -> float:
    """Exact fire probability under the marginal
    (independent-pair) convention: D ~ Bin(n, a),
    J | D ~ Bin(D, q)."""
    total = 0.0
    for d in range(min_d, n + 1):
        p_d = comb(n, d) * a**d * (1 - a) ** (n - d)
        for j in range(cs.MIN_J, d + 1):
            total += p_d * comb(d, j) * q**j * (1 - q) ** (d - j)
    return total


def _transplant_fire(n: int, pc: float, pa: float) -> float:
    """Two-directional transplant (the sibling construction,
    as executed in the design evidence): independent
    control/arm outcomes per case; fire iff either
    discordant count reaches 2 -- evaluated by the exact
    dynamic program over (control-only, arm-only) counts."""
    probs = (pc * pa, pc * (1 - pa), (1 - pc) * pa, (1 - pc) * (1 - pa))
    dp = {(0, 0): 1.0}
    for _ in range(n):
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
    return sum(w for (c, a), w in dp.items() if (c == a and c >= 2) or max(c, a) >= 2)


def _phi2(h: float, k: float, rho: float) -> float:
    """Bivariate standard normal CDF by 1-D quadrature."""
    steps = 4000
    lo = -8.0
    if h <= lo:
        return 0.0
    width = (h - lo) / steps
    scale = (1 - rho * rho) ** 0.5

    def integrand(x: float) -> float:
        """Integrand."""
        return _NORM.pdf(x) * _NORM.cdf((k - rho * x) / scale)

    total = integrand(lo) + integrand(h)
    for i in range(1, steps):
        total += (4 if i % 2 else 2) * integrand(lo + i * width)
    return total * width / 3


def _fire_paired(pc: float, pa: float, rho: float = 0.5, n: int = 24) -> float:
    """Exact fire probability under the per-case copula:
    pJ = pc - Phi2(tc, ta; rho); the trinomial sum over
    (wrong pairs, both-pass pairs, control-fail cases)."""
    p_both = _phi2(_NORM.inv_cdf(pc), _NORM.inv_cdf(pa), rho)
    p_j = pc - p_both
    p_c = 1 - pc
    total = 0.0
    for j in range(cs.MIN_J, n + 1):
        for b in range(max(0, cs.MIN_D - j), n + 1 - j):
            c = n - j - b
            total += comb(n, j) * comb(n - j, b) * p_j**j * p_both**b * p_c**c
    return total


class TestOperatingCharacteristics(unittest.TestCase):
    """Build debt 6 / A6 / A8(v): the adjudicated figures,
    re-derived exactly."""

    def test_independent_rows(self):
        """Independent rows."""
        self.assertEqual(round(_fire_independent(0.90, 0.10), 4), 0.0173)
        self.assertEqual(round(_fire_independent(0.80, 0.20), 4), 0.1743)
        self.assertEqual(round(_fire_independent(0.50, 0.50), 4), 0.5503)
        self.assertEqual(round(_fire_independent(0.90, 0.20), 4), 0.2542)
        self.assertEqual(round(_fire_independent(0.90, 0.35), 4), 0.8159)
        self.assertEqual(round(_fire_independent(0.85, 0.40), 4), 0.8761)
        self.assertEqual(round(_fire_independent(0.90, 0.60), 4), 0.9990)

    def test_panel_sweep_and_sensitivity_rows(self):
        """Panel sweep and sensitivity rows."""
        self.assertEqual(round(_fire_independent(0.90, 0.35, n=16), 4), 0.3904)
        self.assertEqual(round(_fire_independent(0.90, 0.35, n=20), 4), 0.6395)
        self.assertEqual(round(_fire_independent(0.90, 0.10, n=18), 4), 0.0038)
        self.assertEqual(round(_fire_independent(0.90, 0.35, n=18), 4), 0.5214)

    def test_transplant_row(self):
        """Transplant row."""
        self.assertEqual(round(_transplant_fire(24, 0.90, 0.90), 4), 0.8897)

    def test_paired_rows_exact(self):
        """A6: the exact paired figures replace the design
        round's Monte Carlo values."""
        null = _fire_paired(0.90, 0.90)
        drop = _fire_paired(0.90, 0.65)
        self.assertEqual(round(null, 4), 0.0044)
        self.assertAlmostEqual(null, 0.004442, delta=5e-5)
        self.assertEqual(round(drop, 4), 0.6981)
        self.assertAlmostEqual(drop, 0.698078, delta=5e-5)

    def test_familywise_rows_72(self):
        """A6: the family-wise price over the 72 labelled
        comparisons (3 treated arms x 24 candidates)."""
        price = _fire_independent(0.90, 0.10)
        self.assertEqual(round(72 * price, 4), 1.2437)
        self.assertEqual(round(1 - (1 - price) ** 72, 4), 0.7148)
        paired = _fire_paired(0.90, 0.90)
        self.assertEqual(round(72 * paired, 4), 0.3199)
        self.assertEqual(round(1 - (1 - paired) ** 72, 4), 0.2743)
        random_price = _fire_independent(0.50, 0.50)
        self.assertEqual(round(72 * random_price, 4), 39.6221)

    def test_min_d_plateau_qualifier(self):
        """A8(v): the plateau is identical at printed
        precision over [1, 14] and exact over [1, 6]; the
        selected pair trails the exact maximum by the
        executed gap (3.3e-14 by exact rational arithmetic;
        the adjudication's 2.8e-11 did not reproduce and the
        design doc records the correction)."""
        values = {m: _fire_independent(0.90, 0.10, min_d=m) for m in range(1, 17)}
        self.assertEqual(len({round(values[m], 4) for m in range(1, 15)}), 1)
        self.assertEqual(len({values[m] for m in range(1, 7)}), 1)
        self.assertLess(values[1] - values[10], 3e-11)
        self.assertGreater(values[1] - values[10], 0.0)

    def test_module_dict_matches(self):
        """Module dict matches the reproduced figures."""
        oc = cs.OPERATING_CHARACTERISTICS
        self.assertEqual(oc["fire_competent_independent"], round(_fire_independent(0.90, 0.10), 4))
        self.assertEqual(oc["fire_competent_paired"], round(_fire_paired(0.90, 0.90), 4))
        self.assertEqual(oc["power_drop25_paired"], round(_fire_paired(0.90, 0.65), 4))
        self.assertEqual(oc["familywise_paired_expected_72"], 0.3199)
        self.assertEqual(oc["familywise_paired_any_72"], 0.2743)
        self.assertEqual(oc, diffreports._CONFUSABLESKELETON_OPERATING_CHARACTERISTICS)


class TestStagePayloads(unittest.TestCase):
    """Stage and analysis payloads over scripted backends."""

    def test_perfect_payload(self):
        """Perfect payload: every arm passes; the mirror
        validates the payload whole."""
        payload, _backend, _stage = _payload("perfect")
        self.assertTrue(payload["available"])
        self.assertEqual(payload["n_candidates"], 2)
        self.assertEqual(payload["n_cases"], 24)
        for name in ("chatml", "llama3"):
            entry = payload["candidates"][name]
            for arm in cs.ARM_KEYS:
                self.assertEqual(entry["arms"][arm]["rate"], 1.0, (name, arm))
                self.assertEqual(entry["arms"][arm]["decomposition"]["pass"], 24)
            for arm in cs.TREATED_ARMS:
                head = entry["headline"][arm]
                self.assertEqual(head["label"], "no_discordance_observed", (name, arm))
                self.assertEqual(head["control_passes"], 24)
                self.assertEqual(entry["ids"][arm]["n_ids_identical"], 0)
        self.assertTrue(diffreports._valid_confusable_skeleton(payload))

    def test_echo_payload(self):
        """Echo payload: treated answers reproduce the
        treated surface -- echo classes, never wrong values."""
        payload, _backend, _stage = _payload("echo")
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["single_site"]["decomposition"]["echo_confusable"], 24)
        self.assertEqual(entry["arms"]["whole_word"]["decomposition"]["echo_confusable"], 24)
        self.assertEqual(entry["arms"]["rarity_control"]["decomposition"]["echo_control"], 24)
        self.assertEqual(entry["arms"]["control"]["decomposition"]["pass"], 24)
        for arm in cs.TREATED_ARMS:
            self.assertEqual(entry["headline"][arm]["label"], "no_discordance_observed")
        self.assertTrue(diffreports._valid_confusable_skeleton(payload))

    def test_fold_regime_payload(self):
        """The screening-tokenizer regime: the fold makes
        both confusable arms ids-identical (headline
        withheld as instrument-inapplicable) while the
        rarity arm -- whose skeleton genuinely differs --
        stays divergent and headlines normally."""
        payload, _backend, _stage = _payload("perfect", fold=True)
        entry = payload["candidates"]["chatml"]
        for arm in cs.CONFUSABLE_ARMS:
            self.assertEqual(entry["ids"][arm]["n_ids_identical"], 24, arm)
            self.assertEqual(entry["headline"][arm]["n_population"], 0, arm)
            self.assertEqual(entry["headline"][arm]["label"], "insufficient", arm)
        self.assertEqual(entry["ids"]["rarity_control"]["n_ids_identical"], 0)
        self.assertEqual(entry["headline"]["rarity_control"]["label"], "no_discordance_observed")
        self.assertTrue(diffreports._valid_confusable_skeleton(payload))

    def test_garbage_payload(self):
        """Garbage payload."""
        payload, _backend, _stage = _payload("garbage")
        entry = payload["candidates"]["chatml"]
        for arm in cs.ARM_KEYS:
            self.assertEqual(entry["arms"][arm]["decomposition"]["non_answer"], 24, arm)
            self.assertEqual(entry["arms"][arm]["rate"], 0.0, arm)
        for arm in cs.TREATED_ARMS:
            self.assertEqual(entry["headline"][arm]["label"], "insufficient", arm)
        self.assertTrue(diffreports._valid_confusable_skeleton(payload))

    def test_noboundary_payload(self):
        """Noboundary payload."""
        payload, _backend, _stage = _payload("noboundary")
        entry = payload["candidates"]["chatml"]
        # With no boundary token anywhere, evaluate takes
        # the whole padded output as the answer, so every
        # cell's text is the answer plus filler and lands
        # in non_answer (the sibling suites' observation);
        # true boundary_only cells are exercised at the
        # aggregate level below.
        for arm in cs.ARM_KEYS:
            decomposition = entry["arms"][arm]["decomposition"]
            self.assertEqual(decomposition["non_answer"], 24, arm)
        for arm in cs.TREATED_ARMS:
            head = entry["headline"][arm]
            self.assertEqual(head["control_passes"], 0, arm)
            self.assertEqual(head["label"], "insufficient", arm)
            self.assertEqual(head["n_non_answer"], 24, arm)
        self.assertTrue(diffreports._valid_confusable_skeleton(payload))

    def test_boundary_only_at_aggregate_level(self):
        """Boundary only at aggregate level: cells carrying
        the expected text with no boundary decompose as
        boundary_only and contribute no control passes."""
        cases = cs.generate_cases(0)
        records = []
        for case in cases:
            cells = {}
            details = {}
            for arm in cs.ARM_KEYS:
                cells[arm] = _cell(text=case["expected"], verdict=False, boundary=False, tokens=100)
                details[arm] = ""
            records.append(_record(case, cells, details))
        aggregate = cs.aggregate_candidate(records, cases)
        for arm in cs.ARM_KEYS:
            self.assertEqual(aggregate["arms"][arm]["decomposition"]["boundary_only"], 24, arm)
        for arm in cs.TREATED_ARMS:
            head = aggregate["headline"][arm]
            self.assertEqual(head["control_passes"], 0, arm)
            self.assertEqual(head["label"], "insufficient", arm)

    def test_explode_payload(self):
        """Explode payload: every cell an evaluation
        failure; the stage still completes."""
        payload, _backend, _stage = _payload("explode")
        entry = payload["candidates"]["chatml"]
        for arm in cs.ARM_KEYS:
            self.assertEqual(entry["arms"][arm]["n_evaluation_failed"], 24, arm)
            self.assertEqual(entry["arms"][arm]["n_completed"], 0, arm)
        self.assertEqual(entry["n_incomplete"], 24)
        self.assertTrue(diffreports._valid_confusable_skeleton(payload))

    def test_context_exceeded_payload(self):
        """Context exceeded payload."""
        payload, _backend, _stage = _payload("perfect", max_context=10)
        entry = payload["candidates"]["chatml"]
        for arm in cs.ARM_KEYS:
            self.assertEqual(entry["arms"][arm]["n_context_exceeded"], 24, arm)
        self.assertTrue(diffreports._valid_confusable_skeleton(payload))

    def test_cancelled_stage(self):
        """Cancelled stage."""
        event = threading.Event()
        event.set()
        payload, _backend, stage = _payload("perfect", cancel=event)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")

    def test_screen_tamper_stage(self):
        """Screen tamper stage: a broken case invariant
        fails the whole stage closed before generation."""
        with mock.patch.object(cs, "validate_case", lambda case, seed: False):
            payload, _backend, stage = _payload("perfect")
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "confusable_skeleton_screen_failed")
        self.assertEqual(stage["audit"]["by_reason"], {"case_invariant": 24})
        self.assertFalse(payload["available"])

    def test_stage_format_budget(self):
        """Stage format budget: 25 candidates exceed the
        ground-record cap before any generation."""
        formats = tuple(Format(f"fmt{i}", "", "u", "a", "e") for i in range(cs.MAX_CANDIDATES + 1))
        payload, _backend, stage = _payload("perfect", formats=formats)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "confusable_skeleton_record_budget_exceeded")

    def test_record_budget_payload(self):
        """Record budget: eleven formats of long non-ASCII
        answers cross the 12 MiB serialised cap (the
        four-arm panel carries fewer cells per candidate
        than the sibling's seven-arm panel, so the count is
        higher)."""
        formats = tuple(Format(f"fmt{i}", "", "u", "a", "e") for i in range(11))
        payload, _backend, stage = _payload("long_nonascii", formats=formats)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "confusable_skeleton_record_budget_exceeded")
        self.assertFalse(payload["available"])

    def test_stage_rejects_bad_format_tuples(self):
        """Stage rejects bad format tuples at its own
        boundary."""
        backend = _WordBackend("perfect")
        with self.assertRaises(ValueError):
            experiments.confusable_skeleton_stage(
                backend, (), boundary=(), max_context=1000, seed=0
            )
        with self.assertRaises(ValueError):
            experiments.confusable_skeleton_stage(
                backend, (_CHATML, _CHATML), boundary=(), max_context=1000, seed=0
            )


class TestMirrorValidation(unittest.TestCase):
    """The diffreports mirror: recompute validation and
    drift."""

    def test_valid_payload_accepted(self):
        """Valid payload accepted."""
        payload, _backend, _stage = _payload("perfect")
        self.assertTrue(diffreports._valid_confusable_skeleton(payload))

    def test_tampers_rejected(self):
        """Tampers rejected."""
        payload, _backend, _stage = _payload("perfect")

        def tamper(mutator):
            """Tamper."""
            broken = copy.deepcopy(payload)
            mutator(broken)
            return diffreports._valid_confusable_skeleton(broken)

        def spoil(path, value):
            """Spoil."""

            def apply(payload):
                """Apply."""
                node = payload
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value

            return apply

        chat = ("candidates", "chatml")
        record0 = (*chat, "records", 0)
        head = (*chat, "headline", "single_site")
        for path, value in (
            ((*head, "label"), "below_threshold"),
            ((*head, "control_passes"), 23),
            ((*head, "n_population"), 23),
            ((*head, "control_rate"), 0.5),
            ((*head, "pair_decomposition", "mixed"), 1),
            ((*chat, "ids", "single_site", "n_ids_identical"), 1),
            ((*chat, "dose", "whole_word", "eligible_letters"), 170),
            ((*chat, "operating_characteristics"), {"fire_competent_independent": 0.5}),
            (("protocol", "fingerprint"), "0" * 64),
            ((*record0, "content_digests", "single_site"), "0" * 64),
            ((*record0, "cells", "control", "answer_text"), "cedar-5678"),
            ((*record0, "cells", "control", "input_tokens"), 999),
            ((*record0, "cells", "control", "verdict"), False),
            (
                (*record0, "table_pin"),
                {"version": "15.0.0", "table_sha256": "0" * 64},
            ),
            (
                (*record0, "ids_identical"),
                {"single_site": True, "whole_word": False, "rarity_control": False},
            ),
        ):
            self.assertFalse(tamper(spoil(path, value)), path)

    def test_diff_self_unchanged(self):
        """Diff self unchanged."""
        payload, _backend, _stage = _payload("perfect")
        result = diffreports.diff_confusable_skeleton(payload, payload)
        self.assertEqual(result["available"], "both")
        self.assertFalse(result["changed"])
        self.assertTrue(result["comparable"])

    def test_diff_echo_vs_perfect_changed(self):
        """Diff echo vs perfect changed, carrying the
        per-arm rate metrics."""
        perfect, _b1, _s1 = _payload("perfect")
        echo, _b2, _s2 = _payload("echo")
        result = diffreports.diff_confusable_skeleton(perfect, echo)
        self.assertTrue(result["changed"])
        metrics = {entry["metric"] for entry in result["candidate_drift"]}
        self.assertIn("single_site_rate", metrics)
        self.assertIn("whole_word_rate", metrics)
        self.assertIn("rarity_control_rate", metrics)

    def test_diff_missing_side(self):
        """Diff missing side."""
        payload, _backend, _stage = _payload("perfect")
        result = diffreports.diff_confusable_skeleton(payload, None)
        self.assertEqual(result["available"], "old")
        self.assertFalse(result["changed"])
        result = diffreports.diff_confusable_skeleton(None, None)
        self.assertEqual(result["available"], "neither")


class TestReportingSection(unittest.TestCase):
    """The HTML section (design sections 6-8 printed
    matter; A5, A6)."""

    def test_section_renders(self):
        """Section renders."""
        payload, _backend, _stage = _payload("perfect")
        html = reporting._confusable_skeleton_section({"confusable_skeleton": payload})
        self.assertIn("Confusable skeleton", html)
        self.assertIn("no_discordance_observed", html)
        self.assertIn("U+0434", html)
        self.assertIn("U+03B6", html)
        self.assertIn("0.3199", html)
        self.assertIn("0.2743", html)
        self.assertIn("0.0044", html)
        # A5: the whole-word isolation sentence, verbatim.
        self.assertIn(
            "skeleton-class substitution at whole-word dose "
            "disrupts this candidate; rarity at that dose is "
            "unmeasured by this design",
            html,
        )
        self.assertIn("171", html)
        self.assertIn("CYRILLIC SMALL LETTER DE", html)

    def test_section_absent(self):
        """Section absent."""
        self.assertEqual(reporting._confusable_skeleton_section({}), "")
        self.assertEqual(
            reporting._confusable_skeleton_section({"confusable_skeleton": {"available": False}}),
            "",
        )

    def test_section_legend_fallback(self):
        """Section legend fallback: a protocol without the
        population audit still renders (empty legend)."""
        payload, _backend, _stage = _payload("perfect")
        broken = copy.deepcopy(payload)
        broken["protocol"] = {}
        html = reporting._confusable_skeleton_section({"confusable_skeleton": broken})
        self.assertIn("Confusable skeleton", html)


class TestCoverageGaps(unittest.TestCase):
    """Targeted branch coverage for the stage screens, the
    aggregate's hostile shapes, and the mirror's deep
    rejections."""

    def test_population_screen_stage(self):
        """Population screen stage."""
        with mock.patch.dict(cs.POPULATION_AUDIT, {"ok": False}):
            payload, _backend, stage = _payload("perfect")
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "confusable_skeleton_screen_failed")
        self.assertEqual(stage["audit"]["by_reason"].get("arm_population"), 1)
        self.assertFalse(payload["available"])

    def test_special_token_screen_stage(self):
        """Special-token screen stage: a special-token
        literal finding against the run's formats fails the
        stage closed before any generation."""
        with mock.patch(
            "model_protocol_lab.fixtures.audit_special_token_literals",
            return_value=["finding"],
        ):
            payload, _backend, stage = _payload("perfect")
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "confusable_skeleton_screen_failed")
        self.assertEqual(stage["audit"]["by_reason"], {"special_token_literal": 1})
        self.assertEqual(stage["audit"]["special_token_findings"], 1)

    def test_boundary_containment_screen(self):
        """Boundary containment screen: a boundary text that
        appears inside arm content fails the stage closed."""
        backend = _WordBackend("perfect")
        boundary = backend.encode("inspection code is")
        stage = experiments.confusable_skeleton_stage(
            backend,
            (_CHATML,),
            boundary=boundary,
            max_context=100_000,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "confusable_skeleton_screen_failed")

    def test_undecodable_boundary_screens_nothing(self):
        """Undecodable boundary screens nothing: decode
        raising is treated as an empty boundary text and the
        stage proceeds."""

        class _BadDecode(_WordBackend):
            """Bad decode backend: undecodable only for the
            boundary id tuple the stage decodes."""

            def decode(self, ids):
                """Decode."""
                if tuple(ids) == (1, 2, 3):
                    raise RuntimeError("no decode")
                return super().decode(ids)

        backend = _BadDecode("perfect")
        stage = experiments.confusable_skeleton_stage(
            backend,
            (_CHATML,),
            boundary=(1, 2, 3),
            max_context=100_000,
            seed=0,
        )
        self.assertTrue(stage["available"])

    def test_redraw_exhaustion_stage(self):
        """Redraw exhaustion stage: the protocol build
        itself surfaces the exhaustion as the screen
        failure with no protocol."""
        with mock.patch.object(cs, "_derive", lambda *args: 7):
            payload, _backend, stage = _payload("perfect")
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "confusable_skeleton_screen_failed")
        self.assertEqual(stage["protocol"], {})
        self.assertEqual(stage["audit"]["by_reason"], {"redraw_exhausted": 1})

    def test_analysis_unavailable_candidate(self):
        """Analysis unavailable candidate."""
        payload, _backend, _stage = _payload("perfect")
        stage = {
            "available": True,
            "detail": "",
            "protocol": payload["protocol"],
            "audit": payload["audit"],
            "candidates": {
                "chatml": {"status": "unavailable", "detail": "gone", "records": []},
            },
        }
        result = experiments.confusable_skeleton_analysis(stage, ("chatml",))
        self.assertEqual(result["candidates"]["chatml"]["status"], "unavailable")
        self.assertEqual(result["candidates"]["chatml"]["records"], [])

    def test_analysis_unavailable_stage(self):
        """Analysis unavailable stage."""
        result = experiments.confusable_skeleton_analysis(
            {"available": False, "detail": "confusable_skeleton_unavailable"},
            ("chatml",),
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["detail"], "confusable_skeleton_unavailable")
        result = experiments.confusable_skeleton_analysis(
            {"available": False, "detail": "cancelled"}, ("chatml",)
        )
        self.assertEqual(result["detail"], "cancelled")
        result = experiments.confusable_skeleton_analysis({"available": False}, ("chatml",))
        self.assertEqual(result["detail"], "confusable_skeleton_unavailable")

    def test_aggregate_hostile_shapes(self):
        """Aggregate hostile shapes: non-dict cells and
        details degrade to incompletion, never a crash."""
        cases = cs.generate_cases(0)
        records = []
        for case in cases:
            record = _record(case, {}, {})
            record["cells"] = {"control": "junk"}
            record["cell_details"] = {"control": 7}
            record["id_digests"] = "junk"
            record["arm_tokens"] = "junk"
            records.append(record)
        aggregate = cs.aggregate_candidate(records, cases)
        self.assertEqual(aggregate["n_cases"], 24)
        self.assertEqual(aggregate["n_incomplete"], 24)
        for arm in cs.TREATED_ARMS:
            self.assertEqual(aggregate["headline"][arm]["label"], "insufficient")

    def test_validate_case_hostile_patch(self):
        """Validate rejects every case while the population
        audit is broken, and a frozen-table drift moves the
        table and case-material digests (the binding the
        protocol and the mirror check)."""
        case = cs.generate_cases(0)[0]
        with mock.patch.dict(cs.POPULATION_AUDIT, {"ok": False}):
            self.assertFalse(cs.validate_case(case, 0))
        before = cs.table_sha256()
        with mock.patch.dict(cs.FROZEN_REPLACEMENTS, {"r": "\u0436"}):
            self.assertNotEqual(cs.table_sha256(), before)

    def test_mirror_exhaustion_rejected(self):
        """Mirror exhaustion rejected."""
        payload, _backend, _stage = _payload("perfect")
        with mock.patch.object(diffreports, "_confusableskeleton_derive", lambda *args: 7):
            self.assertFalse(diffreports._valid_confusable_skeleton(payload))

    def test_mirror_unavailable_entry_shapes(self):
        """Mirror unavailable entry shapes."""
        payload, _backend, _stage = _payload("perfect")
        broken = copy.deepcopy(payload)
        broken["candidates"]["chatml"] = {"status": "unavailable", "detail": "x", "records": []}
        broken["candidates"].pop("llama3")
        # An unavailable entry beside no available entries
        # leaves no aggregates: invalid.
        self.assertFalse(diffreports._valid_confusable_skeleton(broken))
        result = diffreports.diff_confusable_skeleton(payload, payload)
        self.assertEqual(result["available"], "both")


class TestGapPassModule(unittest.TestCase):
    """Gap pass (cycle 224 coverage): the module's reachable
    defensive branches, exercised directly -- the treated-word
    identity paths, the script helper's empty result, the
    population audit's failure branches under a corrupted
    frozen table, validate_case's deep invariants (reachable
    once the regeneration oracle is patched to return the
    case under test, which is exactly what those invariants
    guard), and the aggregate's flip / wrong-value / reverse
    counters on crafted records."""

    def test_treated_word_identity_paths(self):
        """Treated word identity paths."""
        self.assertEqual(cs.treated_word("cedar", "control"), "cedar")
        # m is zero-eligible in v1 (its skeleton is "rn"): the
        # single-site and rarity scans fall through unchanged.
        self.assertEqual(cs.treated_word("m", "single_site"), "m")
        self.assertEqual(cs.treated_word("m", "rarity_control"), "m")
        self.assertEqual(cs.treated_word("m", "whole_word"), "m")

    def test_script_of_non_panel_scripts(self):
        """Script of non panel scripts."""
        self.assertEqual(cs._script_of("5"), "")
        self.assertEqual(cs._script_of("a"), "")
        self.assertEqual(cs._script_of("ж"), "CYRILLIC")

    def test_audit_failure_branches(self):
        """Audit failure branches under corrupted tables."""
        with mock.patch.dict(cs.FROZEN_REPLACEMENTS, {"a": "ж"}):
            self.assertFalse(cs.audit_population()["ok"])
        with mock.patch.object(cs, "CONTROL_CHARACTERS", ("a",)):
            self.assertFalse(cs.audit_population()["ok"])
        self.assertTrue(cs.audit_population()["ok"])

    def _validate_with_regeneration(self, case):
        """Validate a case whose regeneration returns itself,
        so the deep invariants are what decide."""
        with mock.patch.object(cs, "generate_case", lambda seed, index: case):
            return cs.validate_case(case, 0)

    def test_validate_deep_invariants(self):
        """Validate case deep invariants."""
        genuine = cs.generate_case(0, 0)
        self.assertTrue(self._validate_with_regeneration(copy.deepcopy(genuine)))
        dup = copy.deepcopy(genuine)
        dup["neutral_indices"] = [3, 3]
        self.assertFalse(self._validate_with_regeneration(dup))
        out_of_range = copy.deepcopy(genuine)
        out_of_range["neutral_indices"] = [0, 999]
        self.assertFalse(self._validate_with_regeneration(out_of_range))

    def test_validate_content_invariants(self):
        """Validate case content invariants."""
        case = cs.generate_case(0, 0)
        real_content = cs.arm_content

        def nonascii(case_, arm):
            text = real_content(case_, arm)
            return "é" + text[1:] if arm == "control" else text

        with mock.patch.object(cs, "arm_content", nonascii):
            self.assertFalse(cs.validate_case(case, 0))

        def echo_control(case_, arm):
            return real_content(case_, "control")

        with mock.patch.object(cs, "arm_content", echo_control):
            self.assertFalse(cs.validate_case(case, 0))

    def test_validate_treated_invariants(self):
        """Validate case treated-word invariants."""
        case = cs.generate_case(0, 0)
        real_treated = cs.treated_word

        def longer(word, arm):
            text = real_treated(word, arm)
            return text + "x" if arm == "single_site" else text

        with mock.patch.object(cs, "treated_word", longer):
            self.assertFalse(cs.validate_case(case, 0))

        def wrong_char(word, arm):
            if arm != "single_site":
                return real_treated(word, arm)
            for index, char in enumerate(word):
                if char in cs.FROZEN_REPLACEMENTS:
                    return word[:index] + "ж" + word[index + 1 :]
            return word

        with mock.patch.object(cs, "treated_word", wrong_char):
            self.assertFalse(cs.validate_case(case, 0))

        def double_dose(word, arm):
            if arm != "single_site":
                return real_treated(word, arm)
            out = list(word)
            hits = 0
            for index, char in enumerate(out):
                if char in cs.FROZEN_REPLACEMENTS and hits < 2:
                    out[index] = cs.FROZEN_REPLACEMENTS[char]
                    hits += 1
            return "".join(out)

        with mock.patch.object(cs, "treated_word", double_dose):
            self.assertFalse(cs.validate_case(case, 0))
        self.assertTrue(cs.validate_case(case, 0))

    def _crafted_records(self):
        """Records exercising the aggregate's counters."""
        cases = cs.generate_cases(0)
        case0, case1, case2 = cases[0], cases[1], cases[2]
        details = dict.fromkeys(cs.ARM_KEYS)
        flip = _record(
            case0,
            {
                "control": _cell(case0["expected"]),
                "single_site": _cell("zzz", verdict=False),
                "whole_word": _cell(case0["expected"]),
                "rarity_control": _cell(case0["expected"]),
            },
            details,
            identical_arms=("single_site",),
        )
        wrong = _record(
            case1,
            {
                "control": _cell(case1["expected"]),
                "single_site": _cell("cedar-1000", verdict=False),
                "whole_word": _cell(case1["expected"]),
                "rarity_control": _cell(case1["expected"]),
            },
            details,
        )
        reverse = _record(
            case2,
            {
                "control": _cell("cedar-1000", verdict=False),
                "single_site": _cell(case2["expected"]),
                "whole_word": _cell(case2["expected"]),
                "rarity_control": _cell(case2["expected"]),
            },
            details,
        )
        return cases, [flip, wrong, reverse]

    def test_aggregate_counters(self):
        """Aggregate flip, wrong-value and reverse counters."""
        cases, records = self._crafted_records()
        aggregate = cs.aggregate_candidate(records, cases)
        ids = aggregate["ids"]["single_site"]
        self.assertEqual(ids["n_same_ids_verdict_flips"], 1)
        headline = aggregate["headline"]["single_site"]
        self.assertEqual(headline["control_passes"], 1)
        self.assertEqual(headline["wrong_values"], 1)
        self.assertEqual(headline["n_reverse"], 1)

    def test_aggregate_hostile_cell_shapes(self):
        """Aggregate tolerates non-dict cells and details."""
        cases, records = self._crafted_records()
        hostile = _record(cases[3], {}, {})
        hostile["cells"] = "nope"
        hostile["cell_details"] = 5
        aggregate = cs.aggregate_candidate([*records, hostile], cases)
        self.assertIn("headline", aggregate)
        details_hostile = _record(
            cases[4],
            {
                "control": _cell(cases[4]["expected"]),
                "single_site": _cell("cedar-1000", verdict=False),
                "whole_word": _cell(cases[4]["expected"]),
                "rarity_control": _cell(cases[4]["expected"]),
            },
            {},
        )
        details_hostile["cell_details"] = 5
        aggregate = cs.aggregate_candidate([details_hostile], cases)
        self.assertEqual(aggregate["headline"]["single_site"]["wrong_values"], 1)


class TestGapPassStage(unittest.TestCase):
    """Gap pass: the stage's cancel-after-evaluate raise, the
    evaluation-failed cell path, and the stage-level redraw
    exhaustion return (protocol already built)."""

    def test_cancel_set_during_generation(self):
        """Cancel set during generation."""

        class _CancelBackend(_WordBackend):
            """Backend that cancels mid-generation."""

            def __init__(self, event):
                super().__init__("perfect")
                self._event = event

            def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
                self._event.set()
                return super().generate(
                    input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
                )

        event = threading.Event()
        backend = _CancelBackend(event)
        stage = experiments.confusable_skeleton_stage(
            backend,
            (_CHATML,),
            boundary=backend.encode("endmark"),
            max_context=100_000,
            seed=0,
            cancel=event,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_evaluation_failed_cells(self):
        """Evaluation-failed cells land as arm_evaluation_failed."""
        backend = _WordBackend("perfect")
        stub = mock.Mock()
        stub.aggregate_eligible = False
        stub.observations = []
        with mock.patch.object(experiments, "evaluate", return_value=stub):
            stage = experiments.confusable_skeleton_stage(
                backend,
                (_CHATML,),
                boundary=backend.encode("endmark"),
                max_context=100_000,
                seed=0,
            )
        self.assertTrue(stage["available"])
        record = stage["candidates"]["chatml"]["records"][0]
        self.assertEqual(record["cell_details"]["control"], "arm_evaluation_failed")
        self.assertIsNone(record["cells"]["control"])

    def test_stage_level_redraw_exhaustion(self):
        """Stage redraw exhaustion after the protocol built."""
        backend = _WordBackend("perfect")
        real_cases = cs.generate_cases(0)
        calls = {"n": 0}

        def flaky(seed):
            calls["n"] += 1
            if calls["n"] <= 2:
                return real_cases
            raise cs._RedrawExhausted("forced")

        with mock.patch.object(cs, "generate_cases", flaky):
            stage = experiments.confusable_skeleton_stage(
                backend,
                (_CHATML,),
                boundary=backend.encode("endmark"),
                max_context=100_000,
                seed=0,
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "confusable_skeleton_screen_failed")
        self.assertEqual(stage["audit"]["by_reason"], {"redraw_exhausted": 1})
        self.assertNotEqual(stage["protocol"], {})


class TestGapPassReporting(unittest.TestCase):
    """Gap pass: the section's loose-shape branches (non-dict
    candidates, unavailable rows, non-dict arms, n/a rates,
    non-dict ids entries)."""

    def test_candidates_not_a_dict(self):
        """Candidates not a dict renders empty."""
        result = {"confusable_skeleton": {"available": True, "candidates": [1]}}
        self.assertEqual(reporting._confusable_skeleton_section(result), "")

    def test_loose_candidate_shapes(self):
        """Loose candidate shapes render with fallbacks."""
        probe = {
            "available": True,
            "protocol": {},
            "candidates": {
                "bad": 5,
                "down": {"status": "failed"},
                "odd": {
                    "status": "available",
                    "arms": {
                        "control": 5,
                        "single_site": {
                            "rate": None,
                            "n_pass": 1,
                            "n_completed": 2,
                            "n_context_exceeded": 0,
                            "n_evaluation_failed": 0,
                            "decomposition": None,
                        },
                    },
                    "dose": {
                        "whole_word": {
                            "eligible_letters": 3,
                            "total_letters": 9,
                            "fraction": None,
                        }
                    },
                    "ids": {"single_site": 5},
                    "headline": {"single_site": {"label": "insufficient"}},
                },
            },
        }
        html = reporting._confusable_skeleton_section({"confusable_skeleton": probe})
        self.assertIn("unavailable", html)
        self.assertIn("n/a", html)
        self.assertIn("odd", html)


class TestGapPassMirror(unittest.TestCase):
    """Gap pass: the diffreports mirror's reachable defensive
    branches -- helper identities, audit failures under
    corrupted mirror tables, aggregate counters and classes
    on crafted records, the classify fire label, deep
    validate invariants under a patched mirror regeneration,
    and the diff's new-only and status-drift paths."""

    def test_mirror_helper_identities(self):
        """Mirror helper identities."""
        self.assertEqual(diffreports._confusableskeleton_treated_word("cedar", "control"), "cedar")
        self.assertEqual(diffreports._confusableskeleton_treated_word("m", "single_site"), "m")
        self.assertEqual(diffreports._confusableskeleton_script_of("5"), "")
        self.assertFalse(diffreports._confusableskeleton_digest_ok("zz"))
        self.assertFalse(diffreports._confusableskeleton_digest_ok(123))
        self.assertTrue(diffreports._confusableskeleton_digest_ok("a" * 64))
        self.assertIsNone(diffreports._confusableskeleton_cell_get({"cells": "x"}, "control"))
        self.assertEqual(diffreports._confusableskeleton_detail_get({"cell_details": 5}, "c"), "")

    def test_mirror_audit_failure_branches(self):
        """Mirror audit failure branches."""
        with mock.patch.dict(diffreports._CONFUSABLESKELETON_FROZEN_REPLACEMENTS, {"a": "ж"}):
            self.assertFalse(diffreports._confusableskeleton_population_audit()["ok"])
        with mock.patch.object(diffreports, "_CONFUSABLESKELETON_CONTROL_CHARACTERS", ("a",)):
            self.assertFalse(diffreports._confusableskeleton_population_audit()["ok"])
        self.assertTrue(diffreports._confusableskeleton_population_audit()["ok"])

    def test_mirror_aggregate_counters_and_classes(self):
        """Mirror aggregate counters and decomposition classes."""
        cases = cs.generate_cases(0)
        details = dict.fromkeys(cs.ARM_KEYS)

        def rec(case, control, single, identical=False):
            return _record(
                case,
                {
                    "control": control,
                    "single_site": single,
                    "whole_word": _cell(case["expected"]),
                    "rarity_control": _cell(case["expected"]),
                },
                details,
                identical_arms=("single_site",) if identical else (),
            )

        records = []
        # 12 divergent control passes; 6 single-site wrong
        # values (a guarded fire), 1 boundary-only, 1 form-only
        # (a non-echo skeleton form: the LAST eligible letter
        # substituted, where the arm's echo substitutes the
        # first), the rest passes.
        form_index = next(
            i
            for i, case in enumerate(cases[:12])
            if i >= 7
            and sum(
                1 for char in case["expected"].partition("-")[0] if char in cs.FROZEN_REPLACEMENTS
            )
            >= 2
        )
        for index, case in enumerate(cases[:12]):
            expected = case["expected"]
            if index < 6:
                arm_cell = _cell("cedar-1000", verdict=False)
            elif index == 6:
                arm_cell = _cell(expected, verdict=False)
            elif index == form_index:
                word, _, number = expected.partition("-")
                positions = [i for i, char in enumerate(word) if char in cs.FROZEN_REPLACEMENTS]
                site = positions[-1]
                formed = word[:site] + cs.FROZEN_REPLACEMENTS[word[site]] + word[site + 1 :]
                arm_cell = _cell(formed + "-" + number, verdict=False)
            else:
                arm_cell = _cell(expected)
            records.append(rec(case, _cell(expected), arm_cell))
        # A rarity form-only cell (position-aware, A3): the
        # OTHER script seat's control at the first position --
        # not the arm's own echo (which would classify
        # echo_control), but the position-aware inverse still
        # yields the expected value.
        case = cases[12]
        word, _, number = case["expected"].partition("-")
        own_control = cs.RARITY_CONTROLS[word[0]]
        other_control = "ζ" if own_control == "д" else "д"
        rarity_record = _record(
            case,
            {
                "control": _cell(case["expected"]),
                "single_site": _cell(case["expected"]),
                "whole_word": _cell(case["expected"]),
                "rarity_control": _cell(other_control + case["expected"][1:], verdict=False),
            },
            details,
        )
        records.append(rarity_record)
        # A reverse discordant and an ids-identical flip.
        records.append(
            rec(cases[13], _cell("cedar-1000", verdict=False), _cell(cases[13]["expected"]))
        )
        records.append(
            rec(
                cases[14], _cell(cases[14]["expected"]), _cell("zzz", verdict=False), identical=True
            )
        )
        aggregate = diffreports._confusableskeleton_aggregate(records, cases)
        headline = aggregate["headline"]["single_site"]
        self.assertEqual(headline["label"], "discordant_asymmetric")
        self.assertEqual(headline["wrong_values"], 6)
        self.assertEqual(headline["n_reverse"], 1)
        self.assertEqual(aggregate["ids"]["single_site"]["n_same_ids_verdict_flips"], 1)
        decomp = aggregate["arms"]["single_site"]["decomposition"]
        self.assertEqual(decomp["boundary_only"], 1)
        self.assertEqual(decomp["form_only"], 1)
        rarity_decomp = aggregate["arms"]["rarity_control"]["decomposition"]
        self.assertEqual(rarity_decomp["form_only"], 1)
        pairs = aggregate["headline"]["single_site"]["pair_decomposition"]
        # The 9 discordant pairs decompose by the failing
        # cell's nature: 8 exactness failures (6 arm wrong
        # values + 1 arm form-only + the reverse pair's
        # control wrong value) and 1 boundary-only.
        self.assertEqual(pairs["exact_only"], 8)
        self.assertEqual(pairs["boundary_only"], 1)
        self.assertEqual(pairs["mixed"], 0)

    def test_mirror_validate_deep_invariants(self):
        """Mirror validate deep invariants."""
        genuine = diffreports._confusableskeleton_generate(0)
        tampered = copy.deepcopy(genuine)
        tampered[0]["neutral_indices"] = [3, 3]

        def fake_generate(seed):
            return tampered if seed == 0 else genuine

        payload, _backend, _stage = _payload("perfect")
        with mock.patch.object(diffreports, "_confusableskeleton_generate", fake_generate):
            self.assertFalse(diffreports._valid_confusable_skeleton(payload))

    def test_mirror_type_hostile_payloads(self):
        """Mirror type-hostile payload shapes are invalid."""
        payload, _backend, _stage = _payload("perfect")

        def spoil(mutator):
            broken = copy.deepcopy(payload)
            mutator(broken["candidates"]["chatml"])
            self.assertFalse(diffreports._valid_confusable_skeleton(broken))

        spoil(lambda entry: entry["records"].__setitem__(0, [1]))
        spoil(lambda entry: entry["records"][0].__setitem__("cells", [1]))
        spoil(lambda entry: entry["records"][0]["cells"].__setitem__("control", 5))
        spoil(lambda entry: entry["records"][0].__setitem__("arm_tokens", []))
        spoil(lambda entry: entry["records"][0].__setitem__("id_digests", None))
        spoil(lambda entry: entry["records"][0].__setitem__("content_digests", 7))
        spoil(lambda entry: entry["dose"]["whole_word"].__setitem__("fraction", 0.5))
        spoil(lambda entry: entry["ids"]["single_site"].__setitem__("n_same_ids_verdict_flips", 9))
        spoil(lambda entry: entry["arms"]["control"]["decomposition"].__setitem__("pass", -1))
        spoil(
            lambda entry: entry["headline"]["single_site"]["pair_decomposition"].__setitem__(
                "mixed", True
            )
        )
        spoil(lambda entry: entry["records"][0].__setitem__("table_pin", []))
        spoil(lambda entry: entry["records"][0]["table_pin"].__setitem__("table_sha256", 5))
        spoil(lambda entry: entry["records"][0].__setitem__("id_length_delta", []))
        spoil(lambda entry: entry["records"][0].__setitem__("ids_identical", []))
        spoil(lambda entry: entry["records"][0]["cells"]["control"].__setitem__("verdict", "yes"))
        spoil(lambda entry: entry["records"][0]["cells"]["control"].__setitem__("answer_digest", 5))
        spoil(lambda entry: entry["records"][0].__setitem__("expected_digest", 5))
        spoil(lambda entry: entry["records"][0].__setitem__("fixture", 5))

    def test_diff_new_only_and_status_drift(self):
        """Diff new-only availability and status drift."""
        payload, _backend, _stage = _payload("perfect")
        new_only = diffreports.diff_confusable_skeleton({}, payload)
        self.assertEqual(new_only["available"], "new")
        mixed = copy.deepcopy(payload)
        mixed["candidates"]["llama3"] = {"status": "unavailable", "detail": "x", "records": []}
        if diffreports._valid_confusable_skeleton(mixed):
            drift = diffreports.diff_confusable_skeleton(payload, mixed)
            self.assertEqual(drift["available"], "both")
        shrunk = copy.deepcopy(payload)
        shrunk["candidates"].pop("llama3")
        if diffreports._valid_confusable_skeleton(shrunk):
            one_sided = diffreports.diff_confusable_skeleton(payload, shrunk)
            self.assertEqual(one_sided["available"], "both")


class TestGapPass2(unittest.TestCase):
    """Gap pass 2 (cycle 224 coverage): validate_case's
    early reject order, the cell's post-evaluate cancel
    raise, and the mirror's remaining aggregate branches
    (mixed pair class, non-answer decomposition, the
    classify tail return, tokenless-record id deltas, and
    the ids-identical helper's hostile shape)."""

    def test_validate_early_reject_order(self):
        """Validate early reject order."""
        case = cs.generate_case(0, 0)
        out_of_range = copy.deepcopy(case)
        out_of_range["fixture_id"] = "confusableskeleton-99"
        self.assertFalse(cs.validate_case(out_of_range, 0))
        tampered = copy.deepcopy(case)
        tampered["topic"] = cs.TOPICS[1]
        self.assertFalse(cs.validate_case(tampered, 0))
        with mock.patch.object(cs, "generate_case", lambda seed, index: tampered):
            self.assertFalse(cs.validate_case(tampered, 0))
        bad_value = copy.deepcopy(case)
        bad_value["value"] = "BAD VALUE!"
        bad_value["expected"] = "BAD VALUE!"
        with mock.patch.object(cs, "generate_case", lambda seed, index: bad_value):
            self.assertFalse(cs.validate_case(bad_value, 0))

    def test_validate_word_and_eligible_returns(self):
        """Validate word-membership and eligibility rejects."""
        case = cs.generate_case(0, 0)
        alien = copy.deepcopy(case)
        alien["value"] = "zebra-1234"
        alien["expected"] = "zebra-1234"
        with mock.patch.object(cs, "generate_case", lambda seed, index: alien):
            self.assertFalse(cs.validate_case(alien, 0))
        dull = copy.deepcopy(case)
        dull["topic"] = "bbb"
        with (
            mock.patch.object(cs, "generate_case", lambda seed, index: dull),
            mock.patch.object(cs, "TOPICS", ("bbb", *cs.TOPICS[1:])),
        ):
            self.assertFalse(cs.validate_case(dull, 0))

    def test_cell_pre_evaluate_cancel(self):
        """The cell's pre-evaluate cancel raise (the
        multiturn rule, at the cell level)."""
        event = threading.Event()
        event.set()
        backend = _WordBackend("perfect")
        with self.assertRaises(experiments._ConfusableSkeletonCancelled):
            experiments._confusable_skeleton_cell(
                backend,
                _CHATML,
                object(),
                (1, 2, 3),
                boundary_ids=backend.encode("endmark"),
                max_new_tokens=64,
                max_context=100_000,
                seed=0,
                cancel=event,
            )

    def test_stage_post_evaluate_cancel(self):
        """Stage post-evaluate cancel raise."""
        event = threading.Event()
        backend = _WordBackend("perfect")
        observation = mock.Mock()
        observation.exact_match = True
        observation.boundary_found = True
        observation.boundary_text_match = True
        observation.input_ids = (1, 2, 3)
        observation.answer = "urn-2133"
        report = mock.Mock()
        report.aggregate_eligible = True
        report.observations = [observation]

        def fake_evaluate(*args, **kwargs):
            event.set()
            return report

        with mock.patch.object(experiments, "evaluate", fake_evaluate):
            stage = experiments.confusable_skeleton_stage(
                backend,
                (_CHATML,),
                boundary=backend.encode("endmark"),
                max_context=100_000,
                seed=0,
                cancel=event,
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_mirror_ids_identical_hostile(self):
        """Mirror ids-identical hostile shapes."""
        self.assertFalse(
            diffreports._confusableskeleton_ids_identical({"id_digests": "x"}, "single_site")
        )
        self.assertFalse(diffreports._confusableskeleton_ids_identical({}, "single_site"))
        self.assertFalse(
            diffreports._confusableskeleton_ids_identical(
                {"id_digests": {"control": "", "single_site": ""}}, "single_site"
            )
        )

    def test_mirror_aggregate_remaining_branches(self):
        """Mirror aggregate mixed pair, non-answer, classify
        tail, and tokenless deltas."""
        cases = cs.generate_cases(0)
        details = dict.fromkeys(cs.ARM_KEYS)
        records = []
        for index, case in enumerate(cases[:12]):
            expected = case["expected"]
            single = _cell(expected)
            whole = _cell(expected)
            if index == 0:
                # Mixed pair: digests differ and the arm cell
                # has no boundary.
                single = _cell("cedar-1000", verdict=False, boundary=False)
            elif index == 1:
                single = _cell("zzz", verdict=False)
            if index in (2, 3):
                whole = _cell("cedar-1000", verdict=False)
            records.append(
                _record(
                    case,
                    {
                        "control": _cell(expected),
                        "single_site": single,
                        "whole_word": whole,
                        "rarity_control": _cell(expected),
                    },
                    details,
                )
            )
        tokenless = _record(
            cases[13],
            {
                "control": _cell(cases[13]["expected"]),
                "single_site": _cell(cases[13]["expected"]),
                "whole_word": _cell(cases[13]["expected"]),
                "rarity_control": _cell(cases[13]["expected"]),
            },
            details,
        )
        tokenless.pop("arm_tokens")
        records.append(tokenless)
        aggregate = diffreports._confusableskeleton_aggregate(records, cases)
        pairs = aggregate["headline"]["single_site"]["pair_decomposition"]
        self.assertEqual(pairs["mixed"], 1)
        decomp = aggregate["arms"]["single_site"]["decomposition"]
        self.assertEqual(decomp["non_answer"], 1)
        self.assertEqual(aggregate["headline"]["whole_word"]["label"], "below_threshold")
        self.assertEqual(len(aggregate["ids"]["single_site"]["id_length_delta"]), 13)


if __name__ == "__main__":
    unittest.main()
