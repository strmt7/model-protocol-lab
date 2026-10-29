"""Cycle 114: bidi-control sanitization audit.

The four operator-visible sanitizers (``cli._display_id``,
``debug._sanitize_display``, ``diffreports._display_id``, ``fixtures._safe``)
strip C0/C1 controls and lone surrogates so hostile input files cannot inject
terminal escape sequences. Cycle 114 found they missed the Unicode bidi
controls: a hostile fixture ID like ``"safe-\\u202e-evil"`` (RIGHT-TO-LEFT
OVERRIDE) renders right-to-left in the operator's terminal -- Trojan Source
spoofing (CVE-2021-42574) -- and U+2028/U+2029 break line discipline. All four
sanitizers now strip the bidi-control set (U+200E/U+200F, U+2028/U+2029,
U+202A-U+202E, U+2066-U+2069).

Deliberately preserved: other invisible format characters (BOM, zero-width
space, soft hyphen, ...). They cannot reorder displayed text, and the BOM in
particular is diagnostically valuable -- ``test_bom_prefixed_header_rejected``
asserts a BOM-prefixed CSV header shows the operator exactly why it was
rejected. Stripping it would make the error display a seemingly-valid header.
"""

from __future__ import annotations

import io
import unittest

from model_protocol_lab import cli as cli_mod
from model_protocol_lab import debug as debug_mod
from model_protocol_lab import diffreports as dr_mod
from model_protocol_lab import fixtures as fixtures_mod

_SANITIZERS = (
    ("cli._display_id", cli_mod._display_id),
    ("debug._sanitize_display", debug_mod._sanitize_display),
    ("diffreports._display_id", dr_mod._display_id),
    ("fixtures._safe", fixtures_mod._safe),
)

# Bidi controls: the Trojan Source reordering set, plus LRM/RLM and the
# line/paragraph separators that break terminal line discipline.
_BIDI = [
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u2028",  # line separator
    "\u2029",  # paragraph separator
    "\u202a",  # LRE
    "\u202b",  # RLE
    "\u202c",  # PDF
    "\u202d",  # LRO
    "\u202e",  # RLO
    "\u2066",  # LRI
    "\u2067",  # RLI
    "\u2068",  # FSI
    "\u2069",  # PDI
]

# Invisible format characters with diagnostic value (or benign): must survive.
_PRESERVED_INVISIBLE = [
    "\ufeff",  # BOM -- shown in CSV-header rejection diagnostics
    "\u200b",  # zero-width space
    "\u00ad",  # soft hyphen
    "\u200c",  # ZWNJ (Persian, Indic)
    "\u200d",  # ZWJ (emoji sequences)
]

# Must survive: legitimate text.
_PRESERVED_TEXT = [
    "\u00e9",  # accented latin
    "\u4e2d",  # CJK
    "\u0645",  # arabic letter
    "\U0001f600",  # emoji
    "\U0001d11e",  # musical symbol (astral)
    " ",
]


class TestBidiControlsStripped(unittest.TestCase):
    """Every bidi control must be stripped by every sanitizer."""

    def test_bidi_controls_stripped(self) -> None:
        """Test bidi controls stripped."""
        for name, sanitize in _SANITIZERS:
            for ch in _BIDI:
                with self.subTest(sanitizer=name, char=f"U+{ord(ch):04X}"):
                    self.assertEqual(sanitize(f"safe-{ch}evil"), "safe-evil")

    def test_trojan_source_spoof_neutralized(self) -> None:
        """Test trojan source spoof neutralized."""
        # The canonical spoof: an ID that displays as a trusted name while
        # being logically different. After sanitization the override is gone,
        # so display order == logical order.
        spoof = "trusted-fixture-\u202e-evil"
        for name, sanitize in _SANITIZERS:
            with self.subTest(sanitizer=name):
                cleaned = sanitize(spoof)
                self.assertNotIn("\u202e", cleaned)
                self.assertEqual(cleaned, "trusted-fixture--evil")

    def test_isolate_spoof_neutralized(self) -> None:
        """Test isolate spoof neutralized."""
        spoof = "\u2067evil\u2069-trusted"
        for name, sanitize in _SANITIZERS:
            with self.subTest(sanitizer=name):
                self.assertEqual(sanitize(spoof), "evil-trusted")


class TestDiagnosticInvisiblePreserved(unittest.TestCase):
    """Invisible-but-meaningful characters must survive sanitization.

    The BOM is the proof: ``test_bom_prefixed_header_rejected`` asserts the
    rejection message shows ``\\ufeff`` so the operator sees *why* the header
    failed. Stripping it would display a seemingly-valid header.
    """

    def test_invisible_formats_preserved(self) -> None:
        """Test invisible formats preserved."""
        for name, sanitize in _SANITIZERS:
            for ch in _PRESERVED_INVISIBLE:
                with self.subTest(sanitizer=name, char=f"U+{ord(ch):04X}"):
                    self.assertEqual(sanitize(f"a{ch}b"), f"a{ch}b")

    def test_bom_survives_in_fixtures_safe(self) -> None:
        """Test bom survives in fixtures safe."""
        # Direct regression guard for the cycle-114 design decision: the BOM
        # must reach the ``!r`` diagnostic in the CSV-header error message.
        self.assertEqual(fixtures_mod._safe("\ufeffid"), "\ufeffid")


class TestLegitimateTextPreserved(unittest.TestCase):
    """Legitimate text must survive sanitization."""

    def test_text_preserved(self) -> None:
        """Test text preserved."""
        for name, sanitize in _SANITIZERS:
            for ch in _PRESERVED_TEXT:
                with self.subTest(sanitizer=name, char=f"U+{ord(ch):04X}"):
                    self.assertEqual(sanitize(f"a{ch}b"), f"a{ch}b")

    def test_zwj_emoji_sequence_preserved(self) -> None:
        """Test zwj emoji sequence preserved."""
        # Family emoji: man + ZWJ + woman + ZWJ + girl.
        family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
        for name, sanitize in _SANITIZERS:
            with self.subTest(sanitizer=name):
                self.assertEqual(sanitize(family), family)


class TestExistingBehaviorPreserved(unittest.TestCase):
    """The C0/C1/surrogate stripping contract is unchanged."""

    def test_c0_c1_stripped_everywhere(self) -> None:
        """Test c0 c1 stripped everywhere."""
        for name, sanitize in _SANITIZERS:
            with self.subTest(sanitizer=name):
                self.assertEqual(sanitize("a\x1bb"), "ab")  # ESC
                self.assertEqual(sanitize("a\x00b"), "ab")  # NUL
                self.assertEqual(sanitize("a\x7fb"), "ab")  # DEL
                self.assertEqual(sanitize("a\x9bb"), "ab")  # C1 CSI

    def test_lone_surrogates_stripped_everywhere(self) -> None:
        """Test lone surrogates stripped everywhere."""
        for name, sanitize in _SANITIZERS:
            with self.subTest(sanitizer=name):
                self.assertEqual(sanitize("A\ud800B"), "AB")

    def test_debug_keeps_tab_lf_cr_others_strip(self) -> None:
        """Test debug keeps tab lf cr others strip."""
        # debug keeps \t \n \r so multi-line renders stay readable; the
        # ID-oriented sanitizers strip all C0.
        self.assertEqual(debug_mod._sanitize_display("a\tb\nc\rd"), "a\tb\nc\rd")
        self.assertEqual(cli_mod._display_id("a\tb"), "ab")
        self.assertEqual(dr_mod._display_id("a\nb"), "ab")
        self.assertEqual(fixtures_mod._safe("a\rb"), "ab")


class TestSanitizerParity(unittest.TestCase):
    """All four sanitizers agree on the hostile battery (modulo the
    documented \\t\\n\\r exception for the REPL renderer).
    """

    def test_parity_on_bidi_battery(self) -> None:
        """Test parity on bidi battery."""
        battery = "".join(_BIDI)
        hostile = f"fixture-{battery}-id"
        results = {name: sanitize(hostile) for name, sanitize in _SANITIZERS}
        for name, cleaned in results.items():
            with self.subTest(sanitizer=name):
                for ch in _BIDI:
                    self.assertNotIn(ch, cleaned)
                self.assertEqual(cleaned, "fixture--id")
        # The ID sanitizers fully agree; debug differs only by design.
        self.assertEqual(
            results["cli._display_id"],
            results["diffreports._display_id"],
        )
        self.assertEqual(
            results["cli._display_id"],
            results["fixtures._safe"],
        )


class TestEndToEnd(unittest.TestCase):
    """Hostile IDs must not reach the operator's terminal with bidi intact."""

    def test_repl_fixtures_listing_strips_bidi(self) -> None:
        """Test repl fixtures listing strips bidi."""
        from model_protocol_lab.core import Fixture, Message, Split
        from model_protocol_lab.debug import DebugShell

        evil = Fixture(
            id="trusted-\u202e-evil",
            messages=(Message(role="user", content="hello"),),
            expected="yo",
            group="g",
        )
        other = Fixture(
            id="held-1",
            messages=(Message(role="user", content="world"),),
            expected="yo",
            group="gh",
        )
        shell = DebugShell(Split(search=(evil,), held_out=(other,)))
        buf = io.StringIO()
        shell.stdout = buf
        shell.onecmd(".fixtures")
        out = buf.getvalue()
        self.assertNotIn("\u202e", out)
        out.encode("utf-8", errors="strict")

    def test_fixtures_safe_keeps_bom_for_diagnostics(self) -> None:
        """Test fixtures safe keeps bom for diagnostics."""
        cleaned = fixtures_mod._safe("\ufeffid,group,partition,expected")
        self.assertEqual(cleaned, "\ufeffid,group,partition,expected")

    def test_diffreports_display_id_strips_bidi_candidate(self) -> None:
        """Test diffreports display id strips bidi candidate."""
        cleaned = dr_mod._display_id("cand\u202eevil")
        self.assertEqual(cleaned, "candevil")


if __name__ == "__main__":
    unittest.main()
