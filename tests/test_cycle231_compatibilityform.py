"""Cycle 231: compatibility-form substitution probe
(wave-12 design round 3, BUILD-as-amended, coordinator
#96).

Spec citations are to docs/COMPATIBILITYFORM_DESIGN.md
(the design as amended) and to the adjudication's nine
amendments A1-A9 in the design doc's first section.

Every amendment and build debt has named evidence here:
A1 (the cap restated over the divergent domain; the
M-conditioned operating characteristics re-derived
exactly; the headline carries M) (design section 6,
build debt 5); A2 (the screen-failure detail is
compatibility_form_screen_failed everywhere; the redraw
give-up counting pinned by a forced-collision test)
(build debt 3); A3 (the control arm's class-4 table is
the empty set; the folded control answer classifies
form_only) (build debt 6); A4 (the staged counts carry
their universe and operator, re-derived from the pinned
file) (build debt 4); A5 (the decomposition battery
lists all 15 executed cases including the halfwidth-kana
non_answer case, and the exhaustive echo-reachability
search over all 216,000 expected values is executed as
a test) (build debt 6); A6 (the excluded mirror families
are named with their counts re-derived); A7 (the
arithmetic screen's operator set is named and executed);
the guard battery with the ids dimension and the worlds
re-executed under the pinned domain, with the exhaustive
guard audit over all reachable states (build debt 1);
the generator asserts with freeze tests (build debt 2);
and the doc-completeness test that the shipped design
doc's section-2 tables, the module's tables, and a fresh
derivation from the pinned UnicodeData.txt (by its
SHA-256) all agree (build debt 4).
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
from model_protocol_lab import compatibilityform as cf
from model_protocol_lab import diffreports as dr
from model_protocol_lab.core import Format, Generation

_SENTINEL = object()

_CHATML = Format("chatml", "", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n")
_LLAMA3 = Format(
    "llama3",
    "<|begin_of_text|>",
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|eot_id|>",
)
_NORM = NormalDist()
_ENCLOSED_BY_DIGIT = {reading: char for char, reading in cf.ENCLOSED}
_SUPERSCRIPT_BY_DIGIT = {reading: char for char, reading in cf.SUPERSCRIPTS}
_TABLE_CHARS = frozenset().union(*(cf.CLASS_CHARS[arm] for arm in cf.ARM_KEYS[1:]))


def _swap_value(raw: str) -> str:
    """A different in-grammar value (same number, next word)."""
    word, _, number = raw.partition("-")
    if word in cf.WORDS and number.isdigit():
        alt = cf.WORDS[(cf.WORDS.index(word) + 1) % len(cf.WORDS)]
        return f"{alt}-{number}"
    return "basalt-9999"


class _WordBackend:
    """Word-token scripted backend.

    Modes: ``perfect`` (the expected value read out of the
    fact sentence, boundary on), ``echo`` (the value with
    its digits rendered as enclosed characters -- an
    echo_compat answer under the enclosed arm, a form_only
    answer under every other arm), ``garbage`` (a fixed
    non-answer), ``noboundary`` (the expected answer with
    no boundary token), ``explode`` (generate raises),
    ``long`` (a 2,048-character answer). With ``fold=True``
    the encoder applies full NFKC before tokenising (the
    NFKC-pipeline regime of design section 6): every arm
    whose fold lands exactly on the control text is
    token-identical to control; the fraction arm folds to
    the fraction-slash third form and stays divergent.
    """

    fingerprint = "sha256:test-compatibility-form"

    def __init__(self, mode: str = "perfect", fold: bool = False) -> None:
        """Init."""
        self.mode = mode
        self.fold = fold
        self.calls = 0
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}
        self.seen: dict[str, int] = {}

    def encode(self, text):
        """Encode."""
        if self.fold:
            text = unicodedata.normalize("NFKC", text)
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
        treated = any(char in _TABLE_CHARS for char in text)
        drop_boundary = self.mode == "noboundary"
        if self.mode == "perfect":
            answer = raw
        elif self.mode == "echo":
            answer = "".join(_ENCLOSED_BY_DIGIT.get(char, char) for char in raw)
        elif self.mode == "garbage":
            answer = "unknownword"
        elif self.mode == "long":
            answer = "あ" * 2048
        elif self.mode == "discordant":
            answer = _swap_value(raw) if treated else raw
        elif self.mode == "somediscordant":
            number = raw.partition("-")[2]
            wrong = treated and number.isdigit() and int(number) % 8 == 0
            answer = _swap_value(raw) if wrong else raw
        elif self.mode == "reverse":
            answer = raw if treated else _swap_value(raw)
        elif self.mode == "boundarydrop":
            answer = raw
            drop_boundary = treated
        elif self.mode == "flipflop":
            count = self.seen.get(text, 0)
            self.seen[text] = count + 1
            answer = raw if count % 2 == 0 else _swap_value(raw)
        else:
            answer = raw
        boundary = self.encode("endmark")[0]
        out = list(self.encode(answer))
        if not drop_boundary:
            out.append(boundary)
        pad = boundary if not drop_boundary else self.encode("fillerword")[0]
        out = (out + [pad] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(mode="perfect", fold=False, formats=(_CHATML, _LLAMA3), **stage_kwargs):
    """Payload."""
    backend = _WordBackend(mode, fold=fold)
    stage_kwargs.setdefault("boundary", backend.encode("endmark"))
    stage_kwargs.setdefault("max_context", 100_000)
    stage_kwargs.setdefault("seed", 0)
    stage = experiments.compatibility_form_stage(backend, formats, **stage_kwargs)
    names = tuple(fmt.name for fmt in formats)
    return experiments.compatibility_form_analysis(stage, names), backend, stage


def _valid_payload():
    """A fresh valid perfect-mode payload (mirror-test
    fixture)."""
    payload, _backend, _stage = _payload("perfect")
    return payload


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


_EXPECTED0 = "cedar-8160"
_DIGEST0 = sha256(_EXPECTED0.encode("utf-8")).hexdigest()


class TestGeneratorFreeze(unittest.TestCase):
    """Design section 3 and build debt 2: the frozen
    generator, its asserts, and the panel literals.
    """

    def test_generator_spec_text_pinned(self):
        """Generator spec text pinned."""
        self.assertIn("compatfold", cf.GENERATOR_SPEC)
        self.assertIn("slots 0 and 1 only at counter+1", cf.GENERATOR_SPEC)
        self.assertIn("exceeds 1000", cf.GENERATOR_SPEC)
        self.assertEqual(cf.GENERATOR_SPEC, diffreports._COMPATIBILITYFORM_GENERATOR_SPEC)

    def test_derive_pinned(self):
        """Derive pinned to the frozen conversion."""
        material = "0|compatfold|0|0|0"
        expect = int.from_bytes(sha256(material.encode("ascii")).digest()[:8], "big")
        self.assertEqual(cf._derive(0, 0, 0, 0), expect)
        self.assertEqual(diffreports._compatibilityform_derive(0, 0, 0, 0), expect)

    def test_seed_zero_panel_literals(self):
        """The design section 3 worked example, byte-exact:
        seed 0, case 0.
        """
        case = cf.generate_cases(0)[0]
        self.assertEqual(case["fixture_id"], "compatibilityform-00")
        self.assertEqual(case["topic"], "harbor")
        self.assertEqual(
            cf.marker_sentence(case),
            "The marker line reads script fi, circle 7, share 1/7, "
            "power x9, regnal Henry V, kana カ.",
        )
        self.assertEqual(
            cf.marker_sentence(case, "fraction"),
            "The marker line reads script fi, circle 7, share ⅐, "
            "power x9, regnal Henry V, kana カ.",
        )
        self.assertEqual(
            cf.marker_sentence(case, "katakana"),
            "The marker line reads script fi, circle 7, share 1/7, "
            "power x9, regnal Henry V, kana ｶ.",
        )

    def test_five_seed_panels_literally(self):
        """Five evidence seeds: valid cases, pairwise
        distinct values, every treated arm byte-divergent
        from control on 24/24 cases (design section 3).
        """
        for seed in (0, 1, 7, 42, 20261030):
            cases = cf.generate_cases(seed)
            self.assertEqual(len({case["value"] for case in cases}), 24)
            for case in cases:
                self.assertTrue(cf.validate_case(case), (seed, case["fixture_id"]))
                control = cf.arm_content(case, "control")
                for arm in cf.TREATED_ARMS:
                    self.assertNotEqual(cf.arm_content(case, arm), control)

    def test_panels_distinct_and_deterministic(self):
        """Panels distinct and deterministic."""
        self.assertEqual(cf.generate_cases(7), cf.generate_cases(7))
        self.assertNotEqual(cf.generate_cases(0), cf.generate_cases(1))

    def test_dose_deltas_match_design(self):
        """Per-arm codepoint/byte delta ranges equal the
        design section 3 executed ranges over five seeds.
        """
        ranges = {
            "ligature": ((-2, -1), (0, 1)),
            "enclosed": ((0, 0), (2, 2)),
            "fraction": ((-3, -2), (-1, 0)),
            "superscript": ((0, 0), (1, 2)),
            "roman": ((-3, 0), (-1, 2)),
            "katakana": ((0, 0), (0, 0)),
        }
        for arm, ((cp_lo, cp_hi), (b_lo, b_hi)) in ranges.items():
            cps, byts = [], []
            for seed in (0, 1, 7, 42, 20261030):
                for case in cf.generate_cases(seed):
                    cps.append(
                        len(cf.arm_content(case, arm)) - len(cf.arm_content(case, "control"))
                    )
                    byts.append(
                        len(cf.arm_content(case, arm).encode("utf-8"))
                        - len(cf.arm_content(case, "control").encode("utf-8"))
                    )
            self.assertEqual((min(cps), max(cps)), (cp_lo, cp_hi), arm)
            self.assertEqual((min(byts), max(byts)), (b_lo, b_hi), arm)

    def test_forced_collision_redraws_value_slots_only(self):
        """A2 / build debt 3: a forced value collision
        redraws slots 0+1 at counter+1 while slots 2-9 stay
        at counter 0.
        """
        seed = 3
        cases = cf.generate_cases(seed)
        first_value = cases[0]["value"]
        word0, _, number0 = first_value.partition("-")
        real_derive = cf._derive

        def forced(sd, index, slot, counter):
            """Forced derive: case 1 collides at counter 0."""
            if index == 1 and counter == 0 and slot in (0, 1):
                if slot == 0:
                    return cf.WORDS.index(word0)
                return int(number0) - 1000
            return real_derive(sd, index, slot, counter)

        with mock.patch.object(cf, "_derive", forced):
            panel = cf.generate_cases(seed)
        self.assertNotEqual(panel[1]["value"], first_value)
        # Slots 2-9 (neutrals and all six members) identical
        # to the unforced case 1 at its redrawn value.
        expect = cf.generate_case(seed, 1, panel[1]["value"])
        self.assertEqual(panel[1]["neutral_indices"], expect["neutral_indices"])
        self.assertEqual(panel[1]["members"], expect["members"])
        self.assertEqual(panel[1]["members"], cases[1]["members"])
        self.assertEqual(panel[1]["neutral_indices"], cases[1]["neutral_indices"])

    def test_redraw_exhaustion_fails_closed(self):
        """A2: the counter increments once per colliding
        redraw and exhaustion raises past REDRAW_LIMIT;
        the stage maps it to compatibility_form_screen_failed.
        """

        def constant(sd, index, slot, counter):
            """Constant derive: every case collides forever."""
            return 0

        with mock.patch.object(cf, "_derive", constant):
            with self.assertRaises(cf.RedrawExhausted):
                cf.generate_cases(0)
            backend = _WordBackend()
            stage = experiments.compatibility_form_stage(
                backend,
                (_CHATML,),
                boundary=backend.encode("endmark"),
                max_context=100_000,
                seed=0,
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "compatibility_form_screen_failed")
        self.assertEqual(stage["audit"]["by_reason"], {"redraw_exhausted": 1})

    def test_frozen_population_literals(self):
        """The 117 frozen members (design section 2)."""
        self.assertEqual(
            {name: len(table) for name, table in cf.CLASS_TABLES.items()},
            {
                "ligature": 7,
                "enclosed": 10,
                "fraction": 19,
                "superscript": 10,
                "roman": 16,
                "katakana": 55,
            },
        )
        self.assertEqual(cf.LIGATURES[0], ("ĳ", "ij"))
        self.assertEqual(cf.LIGATURES[-1], ("ﬆ", "st"))
        self.assertEqual(cf.ENCLOSED[0], ("①", "1"))
        self.assertEqual(cf.ENCLOSED[-1], ("⓪", "0"))
        self.assertEqual(cf.FRACTIONS[0], ("¼", "1/4"))
        self.assertEqual(cf.FRACTIONS[-1], ("↉", "0/3"))
        self.assertEqual(cf.SUPERSCRIPTS[0], ("²", "2"))
        self.assertEqual(cf.ROMAN[0], ("Ⅰ", "I"))
        self.assertEqual(cf.ROMAN[-1], ("Ⅿ", "M"))
        self.assertEqual(cf.KATAKANA[0], ("ｦ", "ヲ"))
        self.assertEqual(cf.KATAKANA[-1], ("ﾝ", "ン"))
        all_chars = [char for table in cf.CLASS_TABLES.values() for char, _r in table]
        self.assertEqual(len(all_chars), 117)
        self.assertEqual(len(set(all_chars)), 117)

    def test_population_audit_properties(self):
        """Population audit properties (design section 2):
        every member satisfies the general rule and its
        class clause; readings equal raw NFKC for exactly
        98 members, the 19 fractions diverging by the
        U+2044 rule only.
        """
        audit = cf.POPULATION_AUDIT
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["total"], 117)
        self.assertTrue(audit["pairwise_disjoint"])
        # Audit F3 (cycle 224 rule): the HOST Unicode
        # version is never pinned -- the audit reports the
        # host's own version back, and the PINNED data
        # version is asserted on the data field instead.
        self.assertEqual(audit["unicode_version"], unicodedata.unidata_version)
        self.assertEqual(audit["unicodedata_version"], "15.0.0")
        self.assertEqual(audit["unicodedata_sha256"], cf.UNICODEDATA_SHA256)
        equal = 0
        for name, table in cf.CLASS_TABLES.items():
            for char, reading in table:
                folded = unicodedata.normalize("NFKC", char)
                if reading == folded:
                    equal += 1
                else:
                    self.assertEqual(name, "fraction")
                    self.assertEqual(reading, folded.replace("\u2044", "/"))
        self.assertEqual(equal, 98)

    def test_validate_case_accepts_and_rejects(self):
        """Validate case accepts and rejects."""
        case = cf.generate_cases(0)[0]
        self.assertTrue(cf.validate_case(case))
        for key, bad in (
            ("topic", "birch"),
            ("value", "cedar-816"),
            ("expected", "cedar-8161"),
            ("neutral_indices", [0, 0]),
        ):
            broken = copy.deepcopy(case)
            broken[key] = bad
            if key == "value":
                broken["expected"] = bad
            self.assertFalse(cf.validate_case(broken), key)
        broken = copy.deepcopy(case)
        broken["members"] = dict(case["members"], fraction=0x215F)
        self.assertFalse(cf.validate_case(broken))
        broken = copy.deepcopy(case)
        broken["members"] = dict(case["members"], katakana="x")
        self.assertFalse(cf.validate_case(broken))
        self.assertFalse(cf.validate_case({}))
        self.assertFalse(cf.validate_case({"fixture_id": 7}))

    def test_arm_content_structure(self):
        """Content structure: fact and question frozen;
        the question byte-identical across arms; compat
        characters only in the marker sentence.
        """
        case = cf.generate_cases(0)[3]
        control = cf.arm_content(case, "control")
        self.assertTrue(control.endswith(cf.question_text(case)))
        self.assertIn(cf.fact_sentence(case), control)
        for arm in cf.TREATED_ARMS:
            content = cf.arm_content(case, arm)
            self.assertTrue(content.endswith(cf.question_text(case)))
            self.assertIn(cf.fact_sentence(case), content)
            treated_char = chr(case["members"][arm])
            self.assertIn(treated_char, cf.marker_sentence(case, arm))
            self.assertNotIn(treated_char, cf.fact_sentence(case))
        non_ascii = {char for char in control if not char.isascii()}
        self.assertEqual(non_ascii, {cf._READING_BY_CP["katakana"][case["members"]["katakana"]]})

    def test_literal_copies_pinned(self):
        """Literal copies pinned (drift alarms): canonical
        pools and the diffreports mirror literals.
        """
        self.assertEqual(tuple(canonicalform.WORDS), cf.WORDS)
        self.assertEqual(tuple(density.NEUTRAL_STATEMENTS), cf.NEUTRAL_STATEMENTS)
        self.assertEqual(cf.TOPICS, diffreports._COMPATIBILITYFORM_TOPICS)
        self.assertEqual(cf.WORDS, diffreports._COMPATIBILITYFORM_WORDS)
        self.assertEqual(cf.NEUTRAL_STATEMENTS, diffreports._COMPATIBILITYFORM_NEUTRAL_STATEMENTS)
        self.assertEqual(cf.CLASS_TABLES, diffreports._COMPATIBILITYFORM_CLASS_TABLES)
        self.assertEqual(cf.MARKER_TEMPLATE, diffreports._COMPATIBILITYFORM_MARKER_TEMPLATE)
        self.assertEqual(
            cf.OPERATING_CHARACTERISTICS,
            diffreports._COMPATIBILITYFORM_OPERATING_CHARACTERISTICS,
        )
        self.assertEqual(cf.table_sha256(), diffreports._compatibilityform_table_sha256())
        self.assertEqual(cf.audit_population(), diffreports._compatibilityform_population_audit())

    def test_protocol_fingerprint_stability(self):
        """Protocol fingerprint stability."""
        first = cf.protocol_fingerprint(0, "sha256:test")
        self.assertEqual(first, cf.protocol_fingerprint(0, "sha256:test"))
        self.assertNotEqual(first, cf.protocol_fingerprint(1, "sha256:test"))
        protocol = cf.protocol_object(0, "sha256:test")
        self.assertEqual(protocol["table_sha256"], cf.table_sha256())
        self.assertEqual(protocol["unicode_version"], "15.0.0")
        self.assertEqual(protocol["unicodedata_sha256"], cf.UNICODEDATA_SHA256)
        self.assertEqual(protocol["generation_budget"], 4032)
        self.assertEqual(protocol["max_ground_records"], 576)


def _unicode_rows() -> dict[int, tuple[str, str]]:
    """All (name, decomposition) rows, parsed from the
    pinned UnicodeData.txt (15.0.0) vendored into this
    repository at tests/data/UnicodeData.txt, its SHA-256
    verified against cf.UNICODEDATA_SHA256 before use.

    Cycle 234: the loader previously read the pinned file
    from a workspace-local path when present and silently
    fell back to the host unicodedata when absent, so the
    census below asserted whatever the host's UCD version
    carried and failed deterministically on CI's 3.11
    (UCD 14) and 3.13 (UCD 15.1) legs. The vendored file
    is the only source now: absence or a digest mismatch
    raises, failing the census loudly -- never a host
    fallback, never a skip.
    """
    pinned = Path(__file__).resolve().parent / "data" / "UnicodeData.txt"
    if not pinned.is_file():
        raise AssertionError(
            f"pinned UnicodeData.txt missing from the repository: {pinned}; "
            "the census derives from the vendored file only (cycle 234)"
        )
    data = pinned.read_bytes()
    digest = sha256(data).hexdigest()
    if digest != cf.UNICODEDATA_SHA256:
        raise AssertionError(
            f"vendored UnicodeData.txt digest {digest} != pinned "
            f"{cf.UNICODEDATA_SHA256}; refusing to derive the census "
            "from unverified data (cycle 234)"
        )
    rows: dict[int, tuple[str, str]] = {}
    for line in data.decode("utf-8").splitlines():
        fields = line.split(";")
        rows[int(fields[0], 16)] = (fields[1], fields[5] if len(fields) > 5 else "")
    return rows


def _derive_classes(rows: dict[int, tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    """The design section 2 derivation, executed from the
    rows alone.
    """

    def tag(decomp: str) -> str:
        """Tag."""
        first = decomp.split()[0] if decomp.split() else ""
        return first if first.startswith("<") else ""

    def target(decomp: str) -> str:
        """Target."""
        return "".join(chr(int(token, 16)) for token in decomp.split()[1:])

    def eligible(codepoint: int) -> bool:
        """Eligible."""
        _name, decomp = rows[codepoint]
        char = chr(codepoint)
        if not tag(decomp) or tag(decomp) == "<wide>":
            return False
        if unicodedata.normalize("NFC", char) != char:
            return False
        if unicodedata.normalize("NFD", char) != char:
            return False
        return unicodedata.normalize("NFKC", char) != char

    def reading(codepoint: int) -> str:
        """Reading."""
        folded = unicodedata.normalize("NFKC", chr(codepoint))
        if rows[codepoint][0].startswith("VULGAR FRACTION"):
            return folded.replace("\u2044", "/")
        return folded

    classes: dict[str, list[tuple[str, str]]] = {
        "ligature": [
            (chr(cp), reading(cp))
            for cp in sorted(rows)
            if eligible(cp)
            and rows[cp][0].startswith("LATIN SMALL LIGATURE")
            and target(rows[cp][1]).isascii()
        ],
        "enclosed": [
            (chr(cp), reading(cp))
            for cp in sorted(rows)
            if eligible(cp) and rows[cp][0].startswith("CIRCLED DIGIT")
        ],
        "fraction": [
            (chr(cp), reading(cp))
            for cp in sorted(rows)
            if eligible(cp) and rows[cp][0].startswith("VULGAR FRACTION")
        ],
        "superscript": [
            (chr(cp), reading(cp))
            for cp in sorted(rows)
            if eligible(cp)
            and tag(rows[cp][1]) == "<super>"
            and len(target(rows[cp][1])) == 1
            and target(rows[cp][1]) in "0123456789"
        ],
        "roman": [
            (chr(cp), reading(cp))
            for cp in sorted(rows)
            if eligible(cp) and rows[cp][0].startswith("ROMAN NUMERAL")
        ],
        "katakana": [
            (chr(cp), reading(cp))
            for cp in sorted(rows)
            if eligible(cp) and rows[cp][0].startswith("HALFWIDTH KATAKANA LETTER")
        ],
    }
    return classes


class TestDesignDocCompleteness(unittest.TestCase):
    """Build debt 4 / A4 / A6: the shipped design doc's
    section-2 tables, the module's tables, and a fresh
    derivation from the pinned data all agree; the staged
    counts carry their universe and operator.
    """

    @classmethod
    def setUpClass(cls):
        """Set up class."""
        cls.rows = _unicode_rows()
        cls.derived = _derive_classes(cls.rows)

    def test_doc_tables_match_module_and_derivation(self):
        """Doc tables match module and derivation."""
        text = Path("docs/COMPATIBILITYFORM_DESIGN.md").read_text(encoding="utf-8")
        doc_tables: dict[str, list[tuple[str, str]]] = {}
        current = None
        headings = {
            "**Ligatures (7)**": "ligature",
            "**Enclosed alphanumerics (10)**": "enclosed",
            "**Superscripts (10)**": "superscript",
            "**Roman numerals (16)**": "roman",
            "**Vulgar fractions (19)**": "fraction",
            "**Halfwidth katakana (55)**": "katakana",
        }
        for line in text.splitlines():
            for heading, name in headings.items():
                if line.startswith(heading):
                    current = name
                    doc_tables[current] = []
            match = re.match(r"\| U\+([0-9A-F]{4,6}) \| (.) \| (.+?) \|", line)
            if match and current:
                codepoint = int(match.group(1), 16)
                doc_tables[current].append((chr(codepoint), match.group(3)))
        for name in cf.CLASS_TABLES:
            expect = [(char, reading) for char, reading in self.derived[name]]
            self.assertEqual(doc_tables.get(name), expect, f"doc table {name}")
            self.assertEqual(list(cf.CLASS_TABLES[name]), expect, f"module table {name}")
            self.assertEqual(
                list(diffreports._COMPATIBILITYFORM_CLASS_TABLES[name]),
                expect,
                f"mirror table {name}",
            )

    def test_census_and_staged_counts(self):
        """A4/A6: the census and the staged counts,
        re-derived with their universes and operators.
        """
        rows = self.rows
        decomp_rows = {cp: value for cp, value in rows.items() if value[1]}

        def tag(decomp: str) -> str:
            """Tag."""
            first = decomp.split()[0] if decomp.split() else ""
            return first if first.startswith("<") else ""

        self.assertEqual(len(decomp_rows), 5857)
        self.assertEqual(sum(1 for _n, d in decomp_rows.values() if tag(d)), 3796)
        self.assertEqual(sum(1 for _n, d in decomp_rows.values() if tag(d) == "<wide>"), 104)
        derived_total = sum(len(table) for table in self.derived.values())
        self.assertEqual(derived_total, 117)
        # Ligature staging (A4): prefix over
        # decomposition-bearing rows = 8; file-wide by
        # prefix = 9 (U+0153 carries no decomposition).
        self.assertEqual(
            sum(1 for n, _d in decomp_rows.values() if n.startswith("LATIN SMALL LIGATURE")),
            8,
        )
        self.assertEqual(
            sum(1 for n, _d in rows.values() if n.startswith("LATIN SMALL LIGATURE")),
            9,
        )
        # Roman staging (A4): substring over
        # decomposition-bearing rows = 32; file-wide 24 by
        # prefix, 40 by substring.
        self.assertEqual(sum(1 for n, _d in decomp_rows.values() if "ROMAN NUMERAL" in n), 32)
        self.assertEqual(sum(1 for n, _d in rows.values() if n.startswith("ROMAN NUMERAL")), 24)
        self.assertEqual(sum(1 for n, _d in rows.values() if "ROMAN NUMERAL" in n), 40)
        # A6 mirror families: <sub> rows = 64 with the
        # single-digit members U+2080-2089 = 10;
        # parenthesized digits (prefix, decomposition
        # rows) = 9; circled numbers (prefix, decomposition
        # rows) = 41.
        self.assertEqual(sum(1 for _n, d in decomp_rows.values() if tag(d) == "<sub>"), 64)
        self.assertEqual(
            sum(
                1
                for cp, (_n, d) in decomp_rows.items()
                if tag(d) == "<sub>" and 0x2080 <= cp <= 0x2089
            ),
            10,
        )
        self.assertEqual(
            sum(1 for n, _d in decomp_rows.values() if n.startswith("PARENTHESIZED DIGIT")),
            9,
        )
        self.assertEqual(
            sum(1 for n, _d in decomp_rows.values() if n.startswith("CIRCLED NUMBER")),
            41,
        )

    def test_doc_carries_amendments_and_figures(self):
        """The consolidated doc carries A1-A9's substance
        and the section-6 figures.
        """
        text = Path("docs/COMPATIBILITYFORM_DESIGN.md").read_text(encoding="utf-8")
        for needle in (
            "compatibility_form_screen_failed",
            "264",
            "144 comparisons",
            "0.004442",
            "0.698078",
            "2.4875",
            "M = 12, 0.0113",
            "U+2080–2089",
            "empty set",
            "216,000",
            "surfacenoise.py",
            "83",
        ):
            self.assertIn(needle, text)

    def test_no_truncation_markers_in_shipped_docs(self):
        """No truncation markers in shipped docs: the set
        of shipped docs carrying a marker does not grow
        (the two historical carriers are named in the
        cycle-224 test and pinned there).
        """
        known = {"HISTORYFRAME_DESIGN.md", "INSTRUCTIONDISTANCE_DESIGN.md"}
        carrying = set()
        for path in sorted(Path("docs").rglob("*.md")):
            if "[truncated" in path.read_text(encoding="utf-8"):
                carrying.add(path.name)
        self.assertEqual(carrying, known)
        text = Path("docs/COMPATIBILITYFORM_DESIGN.md").read_text(encoding="utf-8")
        self.assertNotIn("[truncated", text)


class TestDecomposition(unittest.TestCase):
    """The ordered decision list (design section 5; A3,
    A5; build debt 6): the 15-case battery, the precedence
    witnesses, A3's control witness, and the exhaustive
    echo-reachability search.
    """

    def _decompose(self, text, arm, *, verdict=False, boundary=True, expected=_EXPECTED0):
        """Decompose."""
        cell = _cell(text, verdict=verdict, boundary=boundary)
        digest = sha256(expected.encode("utf-8")).hexdigest()
        return cf.decompose_cell(cell, expected, digest, arm)

    def test_precedence_pass_then_boundary(self):
        """Precedence: the exact expected answer is pass,
        not form_only (its literal form predicate holds);
        a digest-equal boundary failure is boundary_only
        before all form classes.
        """
        self.assertEqual(self._decompose(_EXPECTED0, "enclosed", verdict=True), "pass")
        self.assertEqual(
            self._decompose(_EXPECTED0, "enclosed", verdict=False, boundary=False),
            "boundary_only",
        )
        self.assertEqual(
            self._decompose(_EXPECTED0, "control", verdict=False, boundary=False),
            "boundary_only",
        )

    def test_wrong_value_grammar_scope(self):
        """Wrong value: in-grammar wrong values only."""
        self.assertEqual(self._decompose("basalt-9999", "roman"), "wrong_value")
        self.assertEqual(self._decompose("cedar-8161", "ligature"), "wrong_value")
        # Out-of-grammar value changes are non_answer,
        # never toward J (design section 5, class 3 scope).
        self.assertEqual(self._decompose("unknown-1234", "roman"), "non_answer")
        self.assertEqual(self._decompose("cedar-0816", "roman"), "non_answer")
        self.assertEqual(self._decompose("cedar-816", "roman"), "non_answer")

    def test_battery_fifteen_cases(self):
        """A5(ii): the battery enumeration is complete --
        all 15 executed cases.
        """
        enclosed_echo = "".join(_ENCLOSED_BY_DIGIT.get(c, c) for c in _EXPECTED0)
        superscript_echo = "".join(_SUPERSCRIPT_BY_DIGIT.get(c, c) for c in _EXPECTED0)
        fullwidth = "".join(
            chr(0xFF41 + ord(c) - ord("a"))
            if "a" <= c <= "z"
            else chr(0xFF10 + ord(c) - ord("0"))
            if c.isdigit()
            else c
            for c in _EXPECTED0
        )
        cases = [
            ("basalt-9999", "fraction", "wrong_value"),
            ("cedar-8161", "katakana", "wrong_value"),
            (enclosed_echo, "enclosed", "echo_compat"),
            (superscript_echo, "superscript", "echo_compat"),
            (superscript_echo, "ligature", "form_only"),
            (fullwidth, "roman", "form_only"),
            ("½", "fraction", "non_answer"),
            ("unknown-1234", "enclosed", "non_answer"),
            ("cedar-0816", "enclosed", "non_answer"),
            ("", "enclosed", "non_answer"),
            ("cedar-816", "enclosed", "non_answer"),
            (_EXPECTED0 + " ", "enclosed", "non_answer"),
            ("ｾﾀ-8160", "katakana", "non_answer"),
        ]
        # The enumeration's remaining two cases -- the
        # correct shape (pass) and the digest-equal
        # boundary failure (boundary_only) -- are pinned in
        # test_precedence_pass_then_boundary; the
        # halfwidth-kana answer above is the enumeration's
        # 15th case: it folds to fullwidth kana, never to
        # an ASCII value.
        for text, arm, expect in cases:
            self.assertEqual(self._decompose(text, arm), expect, (text, arm))
        self.assertEqual(len(cases) + 2, 15)

    def test_a3_control_witness(self):
        """A3: a fullwidth rendering of the expected value
        under the control arm classifies form_only (the
        control arm's echo table is the empty set).
        """
        fullwidth = "".join(
            chr(0xFF41 + ord(c) - ord("a"))
            if "a" <= c <= "z"
            else chr(0xFF10 + ord(c) - ord("0"))
            if c.isdigit()
            else c
            for c in _EXPECTED0
        )
        self.assertEqual(self._decompose(fullwidth, "control"), "form_only")
        enclosed_echo = "".join(_ENCLOSED_BY_DIGIT.get(c, c) for c in _EXPECTED0)
        self.assertEqual(self._decompose(enclosed_echo, "control"), "form_only")
        self.assertEqual(cf.CLASS_CHARS["control"], frozenset())

    def test_case_changing_rendering_is_non_answer(self):
        """A5(i): case-changing renderings fold to a
        different string and decompose non_answer.
        """
        circled_capitals = "ⒶⓁⒹⒺⓇ-1234"
        self.assertEqual(unicodedata.normalize("NFKC", circled_capitals), "ALDER-1234")
        cell = _cell(circled_capitals, verdict=False)
        digest = sha256(b"alder-1234").hexdigest()
        self.assertEqual(
            cf.decompose_cell(cell, "alder-1234", digest, "enclosed"),
            "non_answer",
        )

    def test_missing_text_is_non_answer(self):
        """Missing text is non_answer."""
        cell = _cell(None, verdict=False, boundary=False)
        cell["answer_digest"] = None
        self.assertEqual(cf.decompose_cell(cell, _EXPECTED0, _DIGEST0, "roman"), "non_answer")

    def test_pair_decomposition(self):
        """Pair decomposition."""
        control = _cell(text=_EXPECTED0)
        same = _cell(text=_EXPECTED0, boundary=False)
        self.assertEqual(cf.decompose_pair(control, same), "boundary_only")
        other = _cell(text="cedar-5678")
        self.assertEqual(cf.decompose_pair(control, other), "exact_only")
        no_boundary = _cell(text="cedar-5678", boundary=False)
        self.assertEqual(cf.decompose_pair(control, no_boundary), "mixed")

    def test_exhaustive_echo_reachability(self):
        """A5(iii) / build debt 6: the exhaustive search
        over all 216,000 expected values -- echo_compat is
        reachable only for the enclosed and superscript
        arms.

        A single-character substitution at position i can
        fold back only when the substituted character's
        NFKC equals the replaced character (the other
        positions are unchanged and none of the frozen
        forms composes across a boundary); the search
        enumerates exactly those candidate positions for
        every expected value and executes the literal
        predicate on each.
        """
        singles: dict[str, dict[str, str]] = {}
        for arm in cf.TREATED_ARMS:
            by_fold: dict[str, str] = {}
            for char, _reading in cf.CLASS_TABLES[arm]:
                folded = unicodedata.normalize("NFKC", char)
                if len(folded) == 1:
                    by_fold[folded] = char
            singles[arm] = by_fold
        reachable = dict.fromkeys(cf.TREATED_ARMS, False)
        witnesses = dict.fromkeys(cf.TREATED_ARMS, 0)
        for word in cf.WORDS:
            for number in range(1000, 10000):
                expected = f"{word}-{number}"
                digest = sha256(expected.encode("utf-8")).hexdigest()
                for arm in cf.TREATED_ARMS:
                    by_fold = singles[arm]
                    if not by_fold:
                        continue
                    for position, original in enumerate(expected):
                        char = by_fold.get(original)
                        if char is None:
                            continue
                        rendered = expected[:position] + char + expected[position + 1 :]
                        cell = _cell(rendered, verdict=False)
                        outcome = cf.decompose_cell(cell, expected, digest, arm)
                        self.assertEqual(outcome, "echo_compat", (expected, arm, position))
                        reachable[arm] = True
                        witnesses[arm] += 1
        self.assertTrue(reachable["enclosed"])
        self.assertTrue(reachable["superscript"])
        for arm in ("ligature", "fraction", "roman", "katakana"):
            self.assertFalse(reachable[arm], arm)
            self.assertEqual(witnesses[arm], 0, arm)
        # The 5/0 digit witnesses the design round's sample
        # could not exhibit: digits 5 and 0 echo under both
        # reachable arms.
        expected = "alder-1050"
        digest = sha256(expected.encode("utf-8")).hexdigest()
        arm_tables = (("enclosed", _ENCLOSED_BY_DIGIT), ("superscript", _SUPERSCRIPT_BY_DIGIT))
        for arm, table in arm_tables:
            rendered = "".join(table.get(c, c) for c in expected)
            cell = _cell(rendered, verdict=False)
            self.assertEqual(
                cf.decompose_cell(cell, expected, digest, arm),
                "echo_compat",
                arm,
            )


def _base_rule(d: int, j: int) -> str:
    """The unguarded label rule (test-local oracle piece)."""
    if d < cf.MIN_D:
        return "insufficient"
    if j >= cf.MIN_J:
        return "discordant_asymmetric"
    if j == 0:
        return "no_discordance_observed"
    return "below_threshold"


def _oracle_label(d: int, j: int, k: int) -> str:
    """Independent worst-case-completion oracle for
    classify_headline: enumerate every completion of the
    k missing pairs and require the acquittal to be
    completion-stable; fires issue on the observed counts;
    the cap and the D floor refuse.
    """
    if k > cf.MAX_INCOMPLETE:
        return "insufficient"
    if d < cf.MIN_D:
        return "insufficient"
    if j >= cf.MIN_J:
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
    completion oracle; the cap count; the exhaustive
    guard audit (build debt 1).
    """

    def test_classify_matches_oracle_on_grid(self):
        """Classify matches oracle on grid."""
        for d in range(0, 16):
            for j in range(0, 9):
                for k in range(0, 9):
                    self.assertEqual(
                        cf.classify_headline(d, j, k),
                        _oracle_label(d, j, k),
                        (d, j, k),
                    )

    def test_named_rows(self):
        """Named rows."""
        self.assertEqual(cf.classify_headline(24, 6, 0), "discordant_asymmetric")
        self.assertEqual(cf.classify_headline(10, 6, 6), "discordant_asymmetric")
        self.assertEqual(cf.classify_headline(24, 0, 0), "no_discordance_observed")
        self.assertEqual(cf.classify_headline(24, 0, 1), "insufficient")
        self.assertEqual(cf.classify_headline(24, 5, 0), "below_threshold")
        self.assertEqual(cf.classify_headline(24, 5, 1), "insufficient")
        self.assertEqual(cf.classify_headline(24, 1, 4), "below_threshold")
        self.assertEqual(cf.classify_headline(9, 6, 0), "insufficient")
        self.assertEqual(cf.classify_headline(24, 6, 7), "insufficient")
        self.assertEqual(cf.MIN_D, 10)
        self.assertEqual(cf.MIN_J, 6)
        self.assertEqual(cf.MAX_INCOMPLETE, 6)

    def test_cap_state_count_is_264(self):
        """Over the reachable state space (D + k <= 24),
        exactly 264 states are completion-stable fires the
        cap suppresses (D >= MIN_D, MIN_J <= J <= D -- J
        counts among the control passes -- k > cap).
        """
        count = 0
        for d in range(0, 25):
            for j in range(0, 25):
                for k in range(0, 25):
                    if d + k > 24:
                        continue
                    if d >= cf.MIN_D and cf.MIN_J <= j <= d and k > cf.MAX_INCOMPLETE:
                        count += 1
                        self.assertEqual(cf.classify_headline(d, j, k), "insufficient")
        self.assertEqual(count, 264)

    def test_exhaustive_guard_audit(self):
        """Build debt 1: over every reachable state
        (D + k <= 24, J <= D) and every completion
        (dc in [0, k], jc in [0, dc]): 0 manufactured
        fires (a completion fires only when the observed
        counts already meet the fire floors) and 0
        unstable acquittals (an acquittal label's every
        completion carries the same label under the base
        rule).
        """
        states = 0
        for d in range(0, 25):
            for k in range(0, 25 - d):
                for j in range(0, d + 1):
                    states += 1
                    label = cf.classify_headline(d, j, k)
                    if label == "discordant_asymmetric":
                        # The fire is carried by the observed
                        # counts alone: the floors hold on
                        # (d, j), and every completion -- counts
                        # only grow -- fires under the base rule
                        # too. Removing generations can never
                        # create the fire.
                        self.assertGreaterEqual(d, cf.MIN_D, (d, j, k))
                        self.assertGreaterEqual(j, cf.MIN_J, (d, j, k))
                    for dc in range(0, k + 1):
                        for jc in range(0, dc + 1):
                            completed = _base_rule(d + dc, j + jc)
                            if label == "discordant_asymmetric":
                                self.assertEqual(
                                    completed,
                                    "discordant_asymmetric",
                                    (d, j, k, dc, jc),
                                )
                            if label in ("no_discordance_observed", "below_threshold"):
                                self.assertEqual(completed, label, (d, j, k, dc, jc))
        self.assertEqual(states, 2925)


def _record(case, cells, details, identical_arms=()):
    """Record."""
    digests = {arm: f"digest-{arm}" for arm in cf.ARM_KEYS}
    digest_control = digests["control"]
    for arm in identical_arms:
        digests[arm] = digest_control
    return {
        "fixture": case["fixture_id"],
        "topic": case["topic"],
        "expected_digest": cf.digest_text(case["expected"]),
        "content_digests": dict.fromkeys(cf.ARM_KEYS, "x" * 64),
        "arm_tokens": dict.fromkeys(cf.ARM_KEYS, 100),
        "id_digests": digests,
        "ids_identical": {arm: arm in identical_arms for arm in cf.TREATED_ARMS},
        "id_length_delta": dict.fromkeys(cf.TREATED_ARMS, 0),
        "table_pin": {
            "version": cf.UNICODE_VERSION,
            "unicodedata_sha256": cf.UNICODEDATA_SHA256,
            "table_sha256": cf.table_sha256(),
        },
        "members": dict(case["members"]),
        "cells": cells,
        "cell_details": details,
    }


class TestGuardBatteryIdsDimension(unittest.TestCase):
    """Build debt 1: the completion domain is the
    id-divergent domain; ids-identical incomplete cases
    never enter the completion.
    """

    @staticmethod
    def _cells(case, verdict=True, missing=()):
        """Cells."""
        cells = {}
        details = {}
        for arm in cf.ARM_KEYS:
            if arm in missing:
                cells[arm] = None
                details[arm] = "arm_evaluation_failed"
            else:
                cells[arm] = _cell(text=case["expected"], verdict=verdict)
                details[arm] = ""
        return cells, details

    def test_ids_identical_incomplete_never_enters(self):
        """Ids identical incomplete never enters."""
        cases = cf.generate_cases(0)
        records = []
        for case in cases:
            cells, details = self._cells(case, missing=("fraction",))
            records.append(_record(case, cells, details, identical_arms=("fraction",)))
        aggregate = cf.aggregate_candidate(records, cases)
        head = aggregate["headline"]["fraction"]
        self.assertEqual(head["n_population"], 0)
        self.assertEqual(head["n_incomplete"], 0)
        self.assertEqual(head["n_ids_divergent"], 0)
        self.assertEqual(head["label"], "insufficient")
        self.assertEqual(aggregate["ids"]["fraction"]["n_ids_identical"], 24)
        # A divergent arm on the same records headlines
        # normally (its cells are all present).
        whole = aggregate["headline"]["roman"]
        self.assertEqual(whole["n_population"], 24)
        self.assertEqual(whole["n_ids_divergent"], 24)
        self.assertEqual(whole["label"], "no_discordance_observed")

    def test_mixed_domain_counts_only_divergent_incompletion(self):
        """Mixed domain counts only divergent incompletion."""
        cases = cf.generate_cases(0)
        records = []
        for index, case in enumerate(cases):
            if index < 6:
                cells, details = self._cells(case, missing=("fraction",))
                records.append(_record(case, cells, details, identical_arms=("fraction",)))
            elif index < 8:
                cells, details = self._cells(case, missing=("fraction",))
                records.append(_record(case, cells, details))
            else:
                cells, details = self._cells(case)
                records.append(_record(case, cells, details))
        aggregate = cf.aggregate_candidate(records, cases)
        head = aggregate["headline"]["fraction"]
        self.assertEqual(head["n_population"], 16)
        self.assertEqual(head["n_incomplete"], 2)
        self.assertEqual(head["n_ids_divergent"], 18)
        self.assertEqual(head["label"], "insufficient")
        self.assertEqual(aggregate["n_incomplete"], 8)


def _worlds_ids(cap_trials: int = 4000, seed: int = 20261030) -> dict[str, dict[str, float]]:
    """Worlds A-C re-executed under the PINNED domain
    (build debt 1): each case carries an ids-divergence
    flag (divergent unless index % 5 == 4, the executed
    stand-in for the design's divergent-with-probability-
    0.8 parameter); incompletion counts toward the guard
    only on divergent cases, and the guarded D/J are taken
    over the divergent domain. The unguarded comparator
    ranges over all cases, as the design-phase worlds did.
    """
    import random

    rng = random.Random(seed)  # noqa: S311 -- deterministic test seed
    results: dict[str, dict[str, float]] = {}
    for world in ("A", "B", "C"):
        counts = {
            "guarded_fire": 0,
            "guarded_below": 0,
            "guarded_insufficient": 0,
            "unguarded_fire": 0,
            "unguarded_below": 0,
        }
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
            guarded = cf.classify_headline(d_dom, j_dom, n_dom)
            unguarded = _base_rule(d_all, j_all)
            counts["guarded_fire"] += guarded == "discordant_asymmetric"
            counts["guarded_below"] += guarded == "below_threshold"
            counts["guarded_insufficient"] += guarded == "insufficient"
            counts["unguarded_fire"] += unguarded == "discordant_asymmetric"
            counts["unguarded_below"] += unguarded == "below_threshold"
        results[world] = {key: value / cap_trials for key, value in counts.items()}
    return results


class TestWorldsAtPinnedDomain(unittest.TestCase):
    """Build debt 1: worlds A-C under the pinned
    id-divergent domain (design section 6; bands set from
    the executed harness, direction and order of magnitude
    the design's).
    """

    def test_worlds(self):
        """Worlds."""
        worlds = _worlds_ids()
        # World A (zero effect, sign-concentrated
        # incompletion): the guarded fire stays at the
        # floor (design-phase guarded 0.0037) and the
        # guarded rule withholds the overwhelming majority
        # (design-phase insufficient 0.9765).
        self.assertLess(worlds["A"]["guarded_fire"], 0.02)
        self.assertGreater(worlds["A"]["guarded_insufficient"], 0.85)
        # World B (a true drop, 60 percent of would-be
        # discordants removed): the guarded acquittal
        # nearly vanishes against the unguarded
        # (design-phase 0.7231 unguarded, 0.0763 guarded).
        self.assertLess(worlds["B"]["guarded_below"], 0.15)
        self.assertGreater(worlds["B"]["unguarded_below"], 0.5)
        # World C (the same drop, flat incompletion): the
        # guarded fire survives near the unguarded
        # (design-phase 0.7719 guarded, 0.7824 unguarded).
        self.assertGreater(worlds["C"]["guarded_fire"], 0.60)
        self.assertLessEqual(worlds["C"]["guarded_fire"], worlds["C"]["unguarded_fire"] + 0.02)


def _fire_independent(a: float, q: float, n: int = 24, min_d: int = 10) -> float:
    """Exact fire probability under the marginal
    (independent-pair) convention: D ~ Bin(n, a),
    J | D ~ Bin(D, q).
    """
    total = 0.0
    for d in range(min_d, n + 1):
        p_d = comb(n, d) * a**d * (1 - a) ** (n - d)
        for j in range(cf.MIN_J, d + 1):
            total += p_d * comb(d, j) * q**j * (1 - q) ** (d - j)
    return total


def _transplant_fire(n: int, pc: float, pa: float) -> float:
    """Two-directional transplant (the rejected rule):
    independent control/arm outcomes per case; fire iff
    either discordant count reaches 2 -- evaluated by the
    exact dynamic program over (control-only, arm-only)
    counts.
    """
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
    (wrong pairs, both-pass pairs, control-fail cases).
    """
    p_both = _phi2(_NORM.inv_cdf(pc), _NORM.inv_cdf(pa), rho)
    p_j = pc - p_both
    p_c = 1 - pc
    total = 0.0
    for j in range(cf.MIN_J, n + 1):
        for b in range(max(0, cf.MIN_D - j), n + 1 - j):
            c = n - j - b
            total += comb(n, j) * comb(n - j, b) * p_j**j * p_both**b * p_c**c
    return total


class TestOperatingCharacteristics(unittest.TestCase):
    """Build debt 5 / A1: every section-6 figure
    re-derived exactly, including the M-conditioned rows
    and the exact paired values.
    """

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

    def test_a1_m_conditioned_rows(self):
        """A1: the M-conditioned rows (drop-25 / drop-10 /
        competent null) at each printed domain size.
        """
        rows = {
            24: (0.8159, 0.2542, 0.0173),
            20: (0.6395, 0.1356, 0.0068),
            18: (0.5214, 0.0889, 0.0038),
            16: (0.3904, 0.0527, 0.0019),
            12: (0.1379, 0.0113, 0.0003),
            10: (0.0331, 0.0022, 0.0001),
        }
        for m, (drop25, drop10, null) in rows.items():
            self.assertEqual(round(_fire_independent(0.90, 0.35, n=m), 4), drop25, m)
            self.assertEqual(round(_fire_independent(0.90, 0.20, n=m), 4), drop10, m)
            self.assertEqual(round(_fire_independent(0.90, 0.10, n=m), 4), null, m)
        oc = cf.OPERATING_CHARACTERISTICS
        for m in (20, 18, 16, 12, 10):
            drop25, drop10, null = rows[m]
            self.assertEqual(oc[f"m{m}_drop25"], drop25)
            self.assertEqual(oc[f"m{m}_drop10"], drop10)
            self.assertEqual(oc[f"m{m}_competent"], null)

    def test_transplant_row(self):
        """Transplant row."""
        self.assertEqual(round(_transplant_fire(24, 0.90, 0.90), 4), 0.8897)

    def test_paired_rows_exact(self):
        """The exact paired figures (never Monte Carlo)."""
        null = _fire_paired(0.90, 0.90)
        drop = _fire_paired(0.90, 0.65)
        self.assertEqual(round(null, 4), 0.0044)
        self.assertAlmostEqual(null, 0.004442, delta=5e-5)
        self.assertEqual(round(drop, 4), 0.6981)
        self.assertAlmostEqual(drop, 0.698078, delta=5e-5)

    def test_familywise_rows_144(self):
        """The family-wise price over the 144 labelled
        comparisons (6 treated arms x 24 candidates).
        """
        price = _fire_independent(0.90, 0.10)
        self.assertAlmostEqual(price, 0.017274, delta=5e-6)
        self.assertEqual(round(144 * price, 4), 2.4875)
        self.assertEqual(round(1 - (1 - price) ** 144, 4), 0.9187)
        paired = _fire_paired(0.90, 0.90)
        self.assertEqual(round(144 * paired, 4), 0.6397)
        self.assertEqual(round(1 - (1 - paired) ** 144, 4), 0.4733)
        random_price = _fire_independent(0.50, 0.50)
        self.assertEqual(round(144 * random_price, 4), 79.2441)

    def test_min_d_plateau_qualifier(self):
        """The MIN_D plateau: identical at printed
        precision over [1, 14], exact over [1, 6]; the
        selected pair trails the exact maximum by less
        than 3e-11 (the adjudication's exact-arithmetic
        gap is 2.796189e-11).
        """
        values = {m: _fire_independent(0.90, 0.10, min_d=m) for m in range(1, 17)}
        self.assertEqual(len({round(values[m], 4) for m in range(1, 15)}), 1)
        self.assertEqual(len({values[m] for m in range(1, 7)}), 1)
        self.assertLess(values[1] - values[10], 3e-11)
        self.assertGreater(values[1] - values[10], 0.0)

    def test_module_dict_matches(self):
        """Module dict matches the reproduced figures."""
        oc = cf.OPERATING_CHARACTERISTICS
        self.assertEqual(oc["fire_competent_independent"], round(_fire_independent(0.90, 0.10), 4))
        self.assertEqual(oc["fire_competent_paired"], round(_fire_paired(0.90, 0.90), 4))
        self.assertEqual(oc["power_drop25_paired"], round(_fire_paired(0.90, 0.65), 4))
        self.assertEqual(oc["familywise_competent_expected_144"], 2.4875)
        self.assertEqual(oc["familywise_paired_expected_144"], 0.6397)
        self.assertEqual(oc, diffreports._COMPATIBILITYFORM_OPERATING_CHARACTERISTICS)


class TestStagePayloads(unittest.TestCase):
    """Stage and analysis payloads over scripted backends."""

    def test_perfect_payload(self):
        """Perfect payload: every arm passes; the mirror
        validates the payload whole.
        """
        payload, _backend, _stage = _payload("perfect")
        self.assertTrue(payload["available"])
        self.assertEqual(payload["n_candidates"], 2)
        self.assertEqual(payload["n_cases"], 24)
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["control"]["rate"], 1.0)
        for arm in cf.TREATED_ARMS:
            head = entry["headline"][arm]
            self.assertEqual(head["label"], "no_discordance_observed", arm)
            self.assertEqual(head["control_passes"], 24, arm)
            self.assertEqual(head["n_ids_divergent"], 24, arm)
        self.assertTrue(diffreports._valid_compatibility_form(payload))

    def test_echo_payload(self):
        """Echo payload: enclosed-digit answers decompose
        echo_compat under the enclosed arm and form_only
        under every other arm including control (A3).
        """
        payload, _backend, _stage = _payload("echo")
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["enclosed"]["decomposition"]["echo_compat"], 24)
        self.assertEqual(entry["arms"]["ligature"]["decomposition"]["form_only"], 24)
        self.assertEqual(entry["arms"]["control"]["decomposition"]["form_only"], 24)
        self.assertEqual(entry["arms"]["control"]["decomposition"]["echo_compat"], 0)
        self.assertTrue(diffreports._valid_compatibility_form(payload))

    def test_fold_regime_payload(self):
        """The section-6 regime expectation, executed: an
        NFKC-folding encoder makes five arms ids-identical
        (instrument-inapplicable, never passes) while the
        fraction arm stays divergent on the third form.
        """
        payload, _backend, _stage = _payload("perfect", fold=True)
        entry = payload["candidates"]["chatml"]
        for arm in ("ligature", "enclosed", "superscript", "roman", "katakana"):
            self.assertEqual(entry["ids"][arm]["n_ids_identical"], 24, arm)
            head = entry["headline"][arm]
            self.assertEqual(head["n_population"], 0, arm)
            self.assertEqual(head["n_ids_divergent"], 0, arm)
            self.assertEqual(head["label"], "insufficient", arm)
        self.assertEqual(entry["ids"]["fraction"]["n_ids_divergent"], 24)
        self.assertEqual(entry["headline"]["fraction"]["label"], "no_discordance_observed")
        self.assertTrue(diffreports._valid_compatibility_form(payload))

    def test_garbage_payload(self):
        """Garbage payload: every completed cell is a
        non-answer; nothing counts toward J.
        """
        payload, _backend, _stage = _payload("garbage")
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["control"]["decomposition"]["non_answer"], 24)
        self.assertEqual(entry["arms"]["control"]["rate"], 0.0)
        for arm in cf.TREATED_ARMS:
            self.assertEqual(entry["headline"][arm]["wrong_values"], 0, arm)
        self.assertTrue(diffreports._valid_compatibility_form(payload))

    def test_noboundary_payload(self):
        """No-boundary payload: with no boundary token
        anywhere, evaluate takes the whole padded output
        as the answer, so every cell lands in non_answer
        (the sibling suites' observation); true
        boundary_only cells are exercised at the aggregate
        level below.
        """
        payload, _backend, _stage = _payload("noboundary")
        entry = payload["candidates"]["chatml"]
        for arm in cf.ARM_KEYS:
            decomposition = entry["arms"][arm]["decomposition"]
            self.assertEqual(decomposition["non_answer"], 24, arm)
        for arm in cf.TREATED_ARMS:
            head = entry["headline"][arm]
            self.assertEqual(head["control_passes"], 0, arm)
            self.assertEqual(head["label"], "insufficient", arm)
            self.assertEqual(head["n_non_answer"], 24, arm)
        self.assertTrue(diffreports._valid_compatibility_form(payload))

    def test_boundary_only_at_aggregate_level(self):
        """Boundary only at aggregate level: cells carrying
        the expected text with no boundary decompose as
        boundary_only and contribute no control passes.
        """
        cases = cf.generate_cases(0)
        records = []
        for case in cases:
            cells = {}
            details = {}
            for arm in cf.ARM_KEYS:
                cells[arm] = _cell(text=case["expected"], verdict=False, boundary=False)
                details[arm] = ""
            records.append(_record(case, cells, details))
        aggregate = cf.aggregate_candidate(records, cases)
        for arm in cf.ARM_KEYS:
            self.assertEqual(aggregate["arms"][arm]["decomposition"]["boundary_only"], 24, arm)
        for arm in cf.TREATED_ARMS:
            head = aggregate["headline"][arm]
            self.assertEqual(head["control_passes"], 0, arm)
            self.assertEqual(head["label"], "insufficient", arm)

    def test_explode_payload(self):
        """Explode payload: raising generations become
        evaluation-failed cells, never a stage failure.
        """
        payload, _backend, _stage = _payload("explode")
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["control"]["n_evaluation_failed"], 24)
        self.assertEqual(entry["n_incomplete"], 24)
        self.assertTrue(diffreports._valid_compatibility_form(payload))

    def test_context_exceeded_payload(self):
        """Context-exceeded payload."""
        payload, _backend, _stage = _payload("perfect", max_context=10)
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["control"]["n_context_exceeded"], 24)
        self.assertTrue(diffreports._valid_compatibility_form(payload))

    def test_cancelled_stage(self):
        """Cancelled stage."""
        cancel = threading.Event()
        cancel.set()
        _payload_unused, _backend, stage = _payload("perfect", cancel=cancel)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_screen_tamper_stage(self):
        """Screen tamper stage: a patched-out case battery
        fails the stage closed.
        """
        with mock.patch.object(cf, "validate_case", return_value=False):
            _payload_unused, _backend, stage = _payload("perfect")
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "compatibility_form_screen_failed")
        self.assertEqual(stage["audit"]["by_reason"], {"case_invariant": 24})

    def test_stage_format_budget(self):
        """Stage format budget: 25 candidates exceed the
        ground-record cap before any generation.
        """
        formats = tuple(Format(f"fmt{i}", "", "u\n", "a\n", "<|end|>\n") for i in range(25))
        backend = _WordBackend()
        stage = experiments.compatibility_form_stage(
            backend,
            formats,
            boundary=backend.encode("endmark"),
            max_context=100_000,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "compatibility_form_record_budget_exceeded")

    def test_record_budget_payload(self):
        """Record budget: the serialised-records byte cap
        degrades the whole stage.
        """
        with mock.patch.object(cf, "MAX_RECORDS_BYTES", 10):
            _payload_unused, _backend, stage = _payload("perfect")
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "compatibility_form_record_budget_exceeded")

    def test_stage_rejects_bad_format_tuples(self):
        """Stage rejects bad format tuples."""
        backend = _WordBackend()
        with self.assertRaises(ValueError):
            experiments.compatibility_form_stage(
                backend,
                (),
                boundary=backend.encode("endmark"),
                max_context=100_000,
                seed=0,
            )
        with self.assertRaises(ValueError):
            experiments.compatibility_form_stage(
                backend,
                [_CHATML],
                boundary=backend.encode("endmark"),
                max_context=100_000,
                seed=0,
            )
        with self.assertRaises(ValueError):
            experiments.compatibility_form_stage(
                backend,
                (_CHATML, _CHATML),
                boundary=backend.encode("endmark"),
                max_context=100_000,
                seed=0,
            )


class TestMirrorValidation(unittest.TestCase):
    """The diffreports mirror: recompute validation and
    the diff readout.
    """

    def test_valid_payload_accepted(self):
        """Valid payload accepted."""
        payload, _backend, _stage = _payload("perfect")
        self.assertTrue(diffreports._valid_compatibility_form(payload))

    def test_tampers_rejected(self):
        """Tampers rejected: every committed field is
        rebound by the mirror.
        """
        payload, _backend, _stage = _payload("perfect")

        def tamper(fn):
            """Tamper."""
            broken = copy.deepcopy(payload)
            fn(broken["candidates"]["chatml"]["records"][0], broken)
            return broken

        tampers = [
            lambda record, _p: record.update({"topic": "birch"}),
            lambda record, _p: record.update({"expected_digest": "0" * 64}),
            lambda record, _p: record["members"].update({"fraction": 0x215F}),
            lambda record, _p: record["table_pin"].update({"table_sha256": "0" * 64}),
            lambda record, _p: record["table_pin"].update({"unicodedata_sha256": "0" * 64}),
            lambda record, _p: record["content_digests"].update({"roman": "0" * 64}),
            lambda record, _p: record["ids_identical"].update({"roman": True}),
            lambda record, _p: record["id_length_delta"].update({"roman": 99}),
            lambda record, _p: record["arm_tokens"].update({"roman": "100"}),
            lambda record, _p: record["arm_tokens"].update({"roman": True}),
            lambda record, _p: record["cells"]["control"].update({"verdict": False}),
            lambda record, _p: record["cells"]["control"].update({"answer_text": "cedar-9999"}),
            lambda record, _p: record["cells"]["control"].update({"input_tokens": 3}),
            lambda record, _p: record["cell_details"].update({"control": "bogus"}),
            lambda record, _p: record.update({"cells": dict(record["cells"], control=None)}),
            lambda _r, p: p["candidates"]["chatml"]["headline"]["roman"].update(
                {"label": "discordant_asymmetric"}
            ),
            lambda _r, p: p["candidates"]["chatml"]["headline"]["roman"].update(
                {"control_passes": 23}
            ),
            lambda _r, p: p["candidates"]["chatml"]["ids"]["roman"].update({"n_ids_identical": 1}),
            lambda _r, p: p["candidates"]["chatml"]["dose"]["roman"].update({"byte_delta_max": 99}),
            lambda _r, p: p["candidates"]["chatml"].update(
                {"operating_characteristics": {"fire_competent_independent": 0.5}}
            ),
            lambda _r, p: p["candidates"]["chatml"].update({"n_cases": 23}),
            lambda _r, p: p["protocol"].update({"seed": 1}),
            lambda _r, p: p.update({"n_cases": 23}),
            lambda _r, p: p["audit"].update({"special_token_findings": -1}),
            lambda _r, p: p["audit"].update({"by_reason": {"x": "y"}}),
        ]
        for index, fn in enumerate(tampers):
            self.assertFalse(
                diffreports._valid_compatibility_form(tamper(fn)),
                f"tamper {index} accepted",
            )

    def test_diff_self_unchanged(self):
        """Diff self unchanged."""
        payload, _backend, _stage = _payload("perfect")
        result = diffreports.diff_compatibility_form(payload, payload)
        self.assertEqual(result["available"], "both")
        self.assertFalse(result["changed"])
        self.assertTrue(result["comparable"])
        self.assertEqual(result["candidate_drift"], [])

    def test_diff_echo_vs_perfect_changed(self):
        """Diff echo vs perfect changed."""
        perfect, _b1, _s1 = _payload("perfect")
        echo, _b2, _s2 = _payload("echo")
        result = diffreports.diff_compatibility_form(perfect, echo)
        self.assertEqual(result["available"], "both")
        self.assertTrue(result["changed"])
        metrics = {entry["metric"] for entry in result["candidate_drift"]}
        self.assertIn("control_rate", metrics)
        self.assertIn("enclosed_label", metrics)

    def test_diff_missing_side(self):
        """Diff missing side."""
        payload, _backend, _stage = _payload("perfect")
        self.assertEqual(
            diffreports.diff_compatibility_form(payload, None),
            {"available": "old", "changed": False},
        )
        self.assertEqual(
            diffreports.diff_compatibility_form(None, payload),
            {"available": "new", "changed": False},
        )
        self.assertEqual(
            diffreports.diff_compatibility_form(None, None),
            {"available": "neither", "changed": False},
        )


class TestReportingSection(unittest.TestCase):
    """The reporting section and its TOC/body wiring."""

    def test_section_renders(self):
        """Section renders with the A1 M beside the label."""
        payload, _backend, _stage = _payload("perfect")
        section = reporting._compatibility_form_section({"compatibility_form": payload})
        self.assertIn('<section class="card" id="compatibility-form">', section)
        self.assertIn("divergent-domain size M is 24", section)
        self.assertIn("label no_discordance_observed", section)
        self.assertIn("144 labelled", section)
        self.assertIn("U+FF66", section)

    def test_section_absent(self):
        """Section absent without an available payload."""
        self.assertEqual(reporting._compatibility_form_section({}), "")
        self.assertEqual(
            reporting._compatibility_form_section(
                {"compatibility_form": {"available": False, "detail": "x"}}
            ),
            "",
        )

    def test_toc_body_correspondence(self):
        """TOC/body: the rendered report links the anchor
        and carries the section after the nav.
        """
        payload, _backend, _stage = _payload("perfect")
        html = reporting.render_html({"compatibility_form": payload})
        self.assertIn('href="#compatibility-form"', html)
        self.assertIn('<section class="card" id="compatibility-form">', html)
        self.assertGreater(html.index('id="compatibility-form"'), html.index("</nav>"))

    def test_section_loose_candidate_shapes(self):
        """Loose candidate shapes degrade, never crash."""
        payload, _backend, _stage = _payload("perfect")
        broken = copy.deepcopy(payload)
        broken["candidates"]["chatml"] = {"status": "available", "detail": ""}
        section = reporting._compatibility_form_section({"compatibility_form": broken})
        self.assertIn('id="compatibility-form"', section)
        broken["candidates"]["chatml"] = "junk"
        section = reporting._compatibility_form_section({"compatibility_form": broken})
        self.assertIn('id="compatibility-form"', section)
        broken["protocol"] = {}
        section = reporting._compatibility_form_section({"compatibility_form": broken})
        self.assertIn('id="compatibility-form"', section)


class TestCoverageGaps(unittest.TestCase):
    """Targeted branch coverage for the stage screens, the
    aggregate's hostile shapes, and the mirror's deep
    rejections.
    """

    def test_population_screen_stage(self):
        """Population screen stage."""
        with mock.patch.dict(cf.POPULATION_AUDIT, {"ok": False}):
            _payload_unused, _backend, stage = _payload("perfect")
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "compatibility_form_screen_failed")
        self.assertEqual(stage["audit"]["by_reason"].get("arm_population"), 1)

    def test_special_token_screen_stage(self):
        """Special-token screen stage."""
        with mock.patch(
            "model_protocol_lab.fixtures.audit_special_token_literals",
            return_value=["finding"],
        ):
            _payload_unused, _backend, stage = _payload("perfect")
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "compatibility_form_screen_failed")
        self.assertEqual(stage["audit"]["by_reason"], {"special_token_literal": 1})
        self.assertEqual(stage["audit"]["special_token_findings"], 1)

    def test_boundary_containment_screen(self):
        """Boundary containment screen."""
        backend = _WordBackend("perfect")
        boundary = backend.encode("inspection code is")
        stage = experiments.compatibility_form_stage(
            backend,
            (_CHATML,),
            boundary=boundary,
            max_context=100_000,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "compatibility_form_screen_failed")

    def test_undecodable_boundary_screens_nothing(self):
        """Undecodable boundary screens nothing."""

        class _BadDecode(_WordBackend):
            """Bad decode backend."""

            def decode(self, ids):
                """Decode."""
                if tuple(ids) == (1, 2, 3):
                    raise RuntimeError("no decode")
                return super().decode(ids)

        backend = _BadDecode("perfect")
        stage = experiments.compatibility_form_stage(
            backend,
            (_CHATML,),
            boundary=(1, 2, 3),
            max_context=100_000,
            seed=0,
        )
        self.assertTrue(stage["available"])

    def test_redraw_exhaustion_stage(self):
        """Redraw exhaustion stage: the protocol build
        itself surfaces the exhaustion, with no protocol.
        """
        with mock.patch.object(cf, "_derive", lambda *args: 7):
            _payload_unused, _backend, stage = _payload("perfect")
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "compatibility_form_screen_failed")
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
        result = experiments.compatibility_form_analysis(stage, ("chatml",))
        self.assertEqual(result["candidates"]["chatml"]["status"], "unavailable")
        self.assertEqual(result["candidates"]["chatml"]["records"], [])

    def test_analysis_unavailable_stage(self):
        """Analysis unavailable stage."""
        result = experiments.compatibility_form_analysis(
            {"available": False, "detail": "compatibility_form_unavailable"},
            ("chatml",),
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["detail"], "compatibility_form_unavailable")
        result = experiments.compatibility_form_analysis({"available": False}, ("chatml",))
        self.assertEqual(result["detail"], "compatibility_form_unavailable")

    def test_aggregate_hostile_shapes(self):
        """Aggregate hostile shapes."""
        cases = cf.generate_cases(0)
        records = []
        for case in cases:
            record = _record(case, {}, {})
            record["cells"] = {"control": "junk"}
            record["cell_details"] = {"control": 7}
            record["id_digests"] = "junk"
            record["arm_tokens"] = "junk"
            records.append(record)
        aggregate = cf.aggregate_candidate(records, cases)
        self.assertEqual(aggregate["n_cases"], 24)
        self.assertEqual(aggregate["n_incomplete"], 24)
        for arm in cf.TREATED_ARMS:
            self.assertEqual(aggregate["headline"][arm]["label"], "insufficient")

    def test_table_drift_moves_digests(self):
        """A frozen-table drift moves the table and
        case-material digests (the binding the protocol
        and the mirror check).
        """
        before = cf.table_sha256()
        with mock.patch.dict(cf.CLASS_TABLES, {"roman": (("Ⅰ", "I"),)}):
            self.assertNotEqual(cf.table_sha256(), before)

    def test_mirror_exhaustion_rejected(self):
        """Mirror exhaustion rejected."""
        payload, _backend, _stage = _payload("perfect")
        with mock.patch.object(diffreports, "_compatibilityform_derive", lambda *args: 7):
            self.assertFalse(diffreports._valid_compatibility_form(payload))

    def test_mirror_unavailable_entry_shapes(self):
        """Mirror unavailable entry shapes."""
        payload, _backend, _stage = _payload("perfect")
        broken = copy.deepcopy(payload)
        broken["candidates"]["chatml"] = {"status": "unavailable", "detail": "x", "records": []}
        broken["candidates"].pop("llama3")
        # An unavailable entry beside no available entries
        # leaves no aggregates: invalid.
        self.assertFalse(diffreports._valid_compatibility_form(broken))
        result = diffreports.diff_compatibility_form(payload, payload)
        self.assertEqual(result["available"], "both")

    def test_cancel_set_during_generation(self):
        """Cancel set during generation unwinds the stage
        to the cancelled payload.
        """
        cancel = threading.Event()

        class _CancellingBackend(_WordBackend):
            """Cancelling backend."""

            def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
                """Generate."""
                out = super().generate(
                    input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
                )
                cancel_event.set()
                return out

        cancel_event = cancel
        backend = _CancellingBackend("perfect")
        stage = experiments.compatibility_form_stage(
            backend,
            (_CHATML,),
            boundary=backend.encode("endmark"),
            max_context=100_000,
            seed=0,
            cancel=cancel,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_long_answer_payload_validates(self):
        """Long non-ASCII answers stay digest-bound and the
        payload validates.
        """
        payload, _backend, _stage = _payload("long")
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["control"]["decomposition"]["non_answer"], 24)
        self.assertTrue(diffreports._valid_compatibility_form(payload))


class TestAuditFixWitnesses(unittest.TestCase):
    """Compatfold adversarial-audit witnesses (cycle 231):
    F1 (trailing newline), F2 (lone-surrogate answer
    text), F4 (unknown arm in decompose), F5 (missing
    verdict key), F6 (unknown arm in arm_content). Each
    was confirmed by the coordinator's own execution
    before the fix; these tests pin the fixed behaviour
    in the producer and the mirror alike."""

    def test_trailing_newline_is_non_answer(self):
        """Audit F1: "expected-value + newline" is a
        non-answer on both sides of the mirror, never a
        wrong value counting toward J."""
        cell = _cell(_EXPECTED0 + "\n", verdict=False)
        expected_digest = cf.digest_text(_EXPECTED0)
        self.assertEqual(
            cf.decompose_cell(cell, _EXPECTED0, expected_digest, "roman"), "non_answer"
        )
        self.assertEqual(
            dr._compatibilityform_decompose_cell(cell, _EXPECTED0, expected_digest, "roman"),
            "non_answer",
        )

    def test_surrogate_answer_text_rejected_not_raised(self):
        """Audit F2: a lone-surrogate answer text is a
        validation failure, never an exception."""
        payload = _valid_payload()
        record = payload["candidates"]["chatml"]["records"][0]
        rogue = _EXPECTED0 + "\ud800"
        record["cells"]["control"]["answer_text"] = rogue
        record["cells"]["control"]["answer_digest"] = sha256(
            rogue.encode("utf-8", "surrogatepass")
        ).hexdigest()
        self.assertFalse(dr._valid_compatibility_form(payload))
        result = dr.diff_compatibility_form(payload, payload)
        self.assertEqual(result["available"], "neither")

    def test_decompose_unknown_arm_is_form_only(self):
        """Audit F4: an arm outside the frozen seven
        decomposes as form_only, never a KeyError."""
        cell = _cell(
            "".join(_ENCLOSED_BY_DIGIT.get(char, char) for char in _EXPECTED0), verdict=False
        )
        expected_digest = cf.digest_text(_EXPECTED0)
        self.assertEqual(cf.decompose_cell(cell, _EXPECTED0, expected_digest, "bogus"), "form_only")
        self.assertEqual(
            dr._compatibilityform_decompose_cell(cell, _EXPECTED0, expected_digest, "bogus"),
            "form_only",
        )

    def test_aggregate_tolerates_records_missing_verdict(self):
        """Audit F5: a cell dict without a verdict key
        aggregates as a non-pass, never a KeyError."""
        cell = _cell(_EXPECTED0)
        del cell["verdict"]
        records = [{"fixture": "compatibilityform-00", "cells": {"control": cell, "roman": cell}}]
        aggregate = cf.aggregate_candidate(records, cf.generate_cases(0))
        self.assertEqual(aggregate["arms"]["control"]["n_completed"], 1)
        self.assertEqual(aggregate["arms"]["control"]["n_pass"], 0)

    def test_arm_content_unknown_arm_raises(self):
        """Audit F6: an unknown arm is a ValueError in the
        producer (the mirror twin is in TestGapMirrorPaths)."""
        case = cf.generate_cases(0)[0]
        with self.assertRaises(ValueError):
            cf.arm_content(case, "bogus")


class TestGapGeneratorValidate(unittest.TestCase):
    """Coverage gap battery (cycle 231): the generator's
    default-value path, the validate_case guard table, and
    the population-clause probes. The two NFC/NFD and
    NFKC-fixed returns of _member_satisfies are NOT here:
    an exhaustive scan over all 0x110000 code points found
    no character with a tagged decomposition that is
    NFC/NFD-unstable or NFKC-fixed, so those returns are
    unreachable for any real character data (VALIDATION.md
    documents the scan)."""

    def test_generate_case_default_value(self):
        """Spec: design section 3 (the generator reads the
        case value from its own slots when none is given)."""
        self.assertEqual(cf.generate_case(3, 5), cf.generate_cases(3)[5])

    def test_validate_case_corruption_table(self):
        """Spec: design section 4 (the validator's guards);
        each single-field corruption is rejected."""
        base = cf.generate_cases(0)[0]

        def corrupt(**changes):
            case = copy.deepcopy(base)
            case.update(changes)
            return case

        bad_cases = [
            corrupt(fixture_id="compatibilityform-x"),
            corrupt(fixture_id="compatibilityform-99"),
            corrupt(fixture_id="compatibilityform-1"),
            corrupt(fixture_id=7),
            corrupt(topic="not the topic"),
            corrupt(value=12),
            corrupt(expected="other-1234"),
            corrupt(value="Cedar-8160", expected="Cedar-8160"),
            corrupt(value="zebra-8160", expected="zebra-8160"),
            corrupt(value="cedar-0999", expected="cedar-0999"),
            corrupt(neutral_indices="x"),
            corrupt(neutral_indices=[0]),
            corrupt(neutral_indices=[0, 0]),
            corrupt(neutral_indices=[0, 24]),
            corrupt(neutral_indices=[True, 2]),
            corrupt(members="x"),
            corrupt(members={**base["members"], "roman": True}),
            corrupt(members={**base["members"], "roman": 0x41}),
        ]
        missing = copy.deepcopy(base)
        del missing["members"]["roman"]
        bad_cases.append(missing)
        extra = copy.deepcopy(base)
        extra["members"]["bogus"] = 0x41
        bad_cases.append(extra)
        for case in bad_cases:
            with self.subTest(case=case.get("fixture_id")):
                self.assertFalse(cf.validate_case(case))

    def test_member_satisfies_probes(self):
        """Spec: design section 2 (the population clauses)."""
        self.assertFalse(cf._member_satisfies("ligature", "A", "A"))
        self.assertFalse(cf._member_satisfies("bogus", "ﬀ", "ff"))
        self.assertTrue(cf._member_satisfies("ligature", "ﬀ", "ff"))


class TestGapAggregatePaths(unittest.TestCase):
    """Coverage gap battery: aggregate paths the scripted
    payloads cannot reach (same-ids flips need a stateful
    backend through the stage; hostile record shapes are
    unit-level)."""

    def test_ids_flip_counted(self):
        """Spec: design section 7 (a verdict flip at
        identical ids is counted, never paired)."""
        case = cf.generate_cases(0)[0]
        cells = dict.fromkeys(cf.ARM_KEYS)
        cells["control"] = _cell(case["expected"])
        cells["roman"] = _cell(case["expected"], verdict=False)
        details = dict.fromkeys(cf.ARM_KEYS, "")
        record = _record(case, cells, details, identical_arms=("roman",))
        aggregate = cf.aggregate_candidate([record], cf.generate_cases(0))
        self.assertEqual(aggregate["ids"]["roman"]["n_same_ids_verdict_flips"], 1)
        self.assertEqual(aggregate["ids"]["roman"]["n_ids_identical"], 1)
        self.assertEqual(aggregate["headline"]["roman"]["n_population"], 0)

    def test_aggregate_hostile_record_shapes(self):
        """Spec: design section 5/7 (totality of the cell
        and detail readers)."""
        records = [
            {
                "fixture": "compatibilityform-00",
                "cells": "junk",
                "cell_details": {"control": 5},
                "arm_tokens": {"control": "x"},
            }
        ]
        aggregate = cf.aggregate_candidate(records, cf.generate_cases(0))
        self.assertEqual(aggregate["arms"]["control"]["n_completed"], 0)


class TestGapStagePaths(unittest.TestCase):
    """Coverage gap battery: the stage's cancellation and
    redraw-exhaustion outcomes, and payloads whose paired
    populations are discordant in each direction (the
    headline's fire paths and the pair decomposition)."""

    def test_stage_cancelled(self):
        """Spec: design section 8 (cancellation is a stage
        outcome, never an evaluation failure)."""
        backend = _WordBackend("perfect")
        event = threading.Event()
        event.set()
        stage = experiments.compatibility_form_stage(
            backend,
            (_CHATML,),
            boundary=backend.encode("endmark"),
            max_context=100_000,
            seed=0,
            cancel=event,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_stage_redraw_exhausted(self):
        """Spec: design section 3 (panel exhaustion fails
        the stage closed)."""
        backend = _WordBackend("perfect")
        with mock.patch.object(cf, "generate_cases", side_effect=cf.RedrawExhausted("x")):
            stage = experiments.compatibility_form_stage(
                backend,
                (_CHATML,),
                boundary=backend.encode("endmark"),
                max_context=100_000,
                seed=0,
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "compatibility_form_screen_failed")

    def test_discordant_payload_fires(self):
        """Spec: design sections 5/7 (treated arms answer
        wrong in-grammar values: the asymmetric fire with
        wrong_value failing cells)."""
        payload, _, _ = _payload("discordant")
        self.assertTrue(dr._valid_compatibility_form(payload))
        entry = payload["candidates"]["chatml"]
        headline = entry["headline"]["roman"]
        self.assertEqual(headline["label"], "discordant_asymmetric")
        self.assertEqual(headline["n_population"], 24)
        self.assertEqual(headline["wrong_values"], 24)
        self.assertEqual(headline["pair_decomposition"]["exact_only"], 24)
        self.assertEqual(headline["failing_classes"]["wrong_value"], 24)

    def test_reverse_payload_counts_reverse(self):
        """Spec: design section 7 (control wrong, treated
        right on every pair: the reverse count is full and
        the headline is insufficient -- with zero control
        passes the estimand has no anchor)."""
        payload, _, _ = _payload("reverse")
        self.assertTrue(dr._valid_compatibility_form(payload))
        headline = payload["candidates"]["chatml"]["headline"]["roman"]
        self.assertEqual(headline["label"], "insufficient")
        self.assertEqual(headline["n_reverse"], 24)
        self.assertEqual(headline["control_passes"], 0)

    def test_boundarydrop_payload_pairs(self):
        """Spec: design section 5 (with no boundary token
        the whole generation is the answer: treated arms
        decompose as non_answer, which the headline
        reports on its own axis, never as discordance)."""
        payload, _, _ = _payload("boundarydrop")
        self.assertTrue(dr._valid_compatibility_form(payload))
        entry = payload["candidates"]["chatml"]
        headline = entry["headline"]["roman"]
        self.assertEqual(headline["label"], "no_discordance_observed")
        self.assertEqual(headline["n_non_answer"], 24)
        self.assertEqual(entry["arms"]["roman"]["decomposition"]["non_answer"], 24)

    def test_boundary_only_pair_via_record_splice(self):
        """Spec: design section 5 (a completed cell whose
        digest matches the expected value but whose
        boundary dropped pairs as boundary_only; the
        payload is rebuilt by the producer analysis from
        the spliced stage record and stays mirror-valid,
        exercising the mirror's pair decomposition)."""
        _payload0, _backend, stage = _payload("perfect")
        record = stage["candidates"]["chatml"]["records"][0]
        case = cf.generate_cases(0)[0]
        cell = record["cells"]["roman"]
        cell["verdict"] = False
        cell["boundary_found"] = False
        cell["boundary_text_match"] = False
        cell["answer_text"] = case["expected"]
        cell["answer_digest"] = cf.digest_text(case["expected"])
        payload = experiments.compatibility_form_analysis(stage, ("chatml", "llama3"))
        self.assertTrue(dr._valid_compatibility_form(payload))
        headline = payload["candidates"]["chatml"]["headline"]["roman"]
        self.assertEqual(headline["pair_decomposition"]["boundary_only"], 1)
        self.assertEqual(headline["failing_classes"]["boundary_only"], 1)

    def test_somediscordant_below_threshold(self):
        """Spec: design section 7 (a small discordant count
        under the licence is below_threshold)."""
        payload, _, _ = _payload("somediscordant")
        self.assertTrue(dr._valid_compatibility_form(payload))
        headline = payload["candidates"]["chatml"]["headline"]["roman"]
        self.assertGreaterEqual(headline["wrong_values"], 1)
        self.assertLessEqual(headline["wrong_values"], 5)
        self.assertEqual(headline["label"], "below_threshold")

    def test_flipflop_fold_flips(self):
        """Spec: design section 7 (a stateful backend flips
        verdicts at identical ids under the fold regime;
        the flips are counted, and the payload stays
        valid)."""
        payload, _, _ = _payload("flipflop", fold=True)
        self.assertTrue(dr._valid_compatibility_form(payload))
        entry = payload["candidates"]["chatml"]
        flips = sum(entry["ids"][arm]["n_same_ids_verdict_flips"] for arm in cf.TREATED_ARMS)
        self.assertGreaterEqual(flips, 1)


class TestGapMirrorPaths(unittest.TestCase):
    """Coverage gap battery: mirror entry points and the
    recompute's deep reject branches."""

    def test_mirror_arm_content_unknown_arm_raises(self):
        """Spec: audit F6 (the mirror's content builder is
        strict, like the producer's)."""
        case = cf.generate_cases(0)[0]
        with self.assertRaises(ValueError):
            dr._compatibilityform_arm_content(case, "bogus")

    def test_mirror_member_and_validate(self):
        """Spec: design sections 2/3 (mirror clauses and
        the mirror's independent generator)."""
        self.assertFalse(dr._compatibilityform_member_satisfies("ligature", "A", "A"))
        self.assertEqual(dr._compatibilityform_generate(0), cf.generate_cases(0))
        self.assertEqual(dr._compatibilityform_generate(7), cf.generate_cases(7))

    def test_recompute_deep_rejects(self):
        """Spec: design section 8 (the recompute rejects
        each deep shape)."""
        tampers = [
            lambda p: p["protocol"].update({"seed": True}),
            lambda p: p["candidates"].update({"chatml": "junk"}),
            lambda p: p["candidates"]["chatml"].update({"status": "bogus"}),
            lambda p: p["candidates"]["chatml"].update({"records": "junk"}),
            lambda p: p["candidates"]["chatml"]["records"].append("junk"),
            lambda p: p["n_candidates"].__class__ and p.update({"n_candidates": 99}),
            lambda p: p["candidates"]["chatml"]["records"][0]["arm_tokens"].pop("roman"),
            lambda p: p["candidates"]["chatml"]["records"][0]["id_digests"].update({"roman": "zz"}),
            lambda p: p["candidates"]["chatml"]["records"][0]["ids_identical"].update(
                {"roman": "yes"}
            ),
            lambda p: p["candidates"]["chatml"]["records"][0]["cells"]["roman"].update(
                {"boundary_text_match": "yes"}
            ),
            lambda p: p["candidates"]["chatml"]["records"][0]["cells"]["roman"].update(
                {"input_tokens": True}
            ),
            lambda p: p["candidates"]["chatml"]["records"][0]["cells"]["roman"].update(
                {"answer_digest": "abc"}
            ),
            lambda p: p["candidates"]["chatml"]["records"][0]["cell_details"].update(
                {"roman": "arm_context_exceeded"}
            ),
        ]
        for index, tamper in enumerate(tampers):
            with self.subTest(tamper=index):
                payload = _valid_payload()
                tamper(payload)
                self.assertFalse(dr._valid_compatibility_form(payload))

    def test_diff_status_drift(self):
        """Spec: design section 9 (a candidate's status
        change is a drift entry)."""
        old = _valid_payload()
        new = _valid_payload()
        new["candidates"]["chatml"] = {
            "status": "unavailable",
            "detail": "compatibility_form_unavailable",
            "records": [],
        }
        result = dr.diff_compatibility_form(old, new)
        self.assertTrue(result["available"])
        self.assertTrue(any(drift.get("metric") == "status" for drift in result["candidate_drift"]))


class TestGapReporting(unittest.TestCase):
    """Coverage gap battery: the section's defensive
    render paths."""

    def test_candidates_not_a_dict(self):
        """Spec: design section 9 (a malformed payload
        renders nothing)."""
        section = reporting._compatibility_form_section(
            {"compatibility_form": {"available": True, "candidates": "junk"}}
        )
        self.assertEqual(section, "")

    def test_crafted_entries_render(self):
        """Spec: design section 9 (unavailable rows and
        malformed sub-entries are skipped or rendered as
        unavailable, never raised on)."""
        crafted = {
            "compatibility_form": {
                "available": True,
                "candidates": {
                    "junk": "not-a-dict",
                    "gone": {
                        "status": "unavailable",
                        "detail": "compatibility_form_unavailable",
                        "records": [],
                    },
                    "odd": {
                        "status": "available",
                        "protocol": {},
                        "records": [],
                        "arms": {"control": "junk"},
                        "dose": {"roman": "junk"},
                        "ids": {"roman": "junk"},
                        "headline": {"roman": {"label": "insufficient_data"}},
                    },
                },
            }
        }
        section = reporting._compatibility_form_section(crafted)
        self.assertIn("unavailable", section)
        self.assertIn("odd", section)

    def test_fold_payload_renders_na_rates(self):
        """Spec: design section 9 (zero-population arms
        render n/a rates)."""
        payload, _, _ = _payload(fold=True)
        section = reporting._compatibility_form_section({"compatibility_form": payload})
        self.assertIn("n/a", section)
