"""Cycle 100: HTML report injection audit.

``render_html`` interpolates experiment data (candidate names, fixture
labels, format names, stop strings, divergence details, config values)
into a self-contained HTML report that the user opens in a browser.
Some of that data can originate from third parties (shared dataset CSVs,
checkpoint metadata). This audit feeds a hostile payload battery through
every render path and pins the escaping discipline:

- no raw payload may survive anywhere in the output;
- the escaped form must be present (data is shown, but inert);
- no raw ``<``/``>`` may appear in text content;
- the inline script must contain no ``</script`` breakout;
- the Content-Security-Policy meta tag must be present;
- a hostile ``decision`` value must map to the neutral CSS class
  (no class-attribute injection);
- the output must parse as well-formed HTML.
"""

from __future__ import annotations

import re
import unittest
from html import escape
from html.parser import HTMLParser

from model_protocol_lab.reporting import _CSP, _SCRIPT, render_html

PAYLOADS = [
    "<script>alert(1)</script>",
    '"><img src=x onerror=alert(2)>',
    "<svg onload=alert(3)>",
    "javascript:alert(4)",
    "</title><script>alert(5)</script>",
    "<iframe src=javascript:alert(6)>",
    "\u2028\u2029<script>alert(7)</script>",
]


def _hostile_result() -> dict:
    """A result dict with a hostile payload in every string field the
    HTML renderer reads.
    """
    p0, p1, p2, p3, p4, p5, p6 = PAYLOADS
    return {
        "decision": p0,
        "failure_taxonomy": {
            "per_candidate": {"cand": {p1: 2, p2: 1}},
            "held_out": {p3: 1},
        },
        "search": [
            {
                "format": {"name": p4},
                "summary": {
                    "status": p5,
                    "attempted": 1,
                    "planned": 2,
                    "exact_and_boundary_passes": 1,
                },
                "candidate_fingerprint": p6,
            }
        ],
        "config": {"dataset": p0, "model": p1, "note": p2},
        "held_out": {"summary": {"status": p3}},
        "backend_divergences": [
            {"class": p4, "stop_text": p5, "detail": p6},
        ],
        "template_conformance": {
            "divergences": [{"format": p0, "fixture": p1}],
        },
        "stop_matrix": {"stages": [{"stage": p2, "rows": []}]},
    }


class _TextExtractor(HTMLParser):
    """Extract visible text without case-sensitive tag regexes.

    Cycle 217 (CodeQL py/bad-tag-filter): stripping tags with
    ``re.sub(r"<[^>]*>", ...)`` after lowercase-only script/style
    pre-filters misses uppercase elements; the parser lowercases tag
    names, so every script/style element is excluded regardless of
    case. Character references stay undecoded (convert_charrefs=False)
    so escaped payload text remains in its inert entity form, as the
    previous helper left it. Only well-formed renderer output is fed
    here; an unclosed script element drops trailing text, where the
    old regex kept it minus tags.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track entry into script and style elements so their text is skipped."""
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        """Track exit from script and style elements."""
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        """Collect character data outside script and style elements."""
        if not self._skip:
            self.parts.append(data)


def _text_content(html: str) -> str:
    """Extract the visible text content of the HTML, excluding script and style."""
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    return "".join(extractor.parts)


class _WellFormedChecker(HTMLParser):
    """HTML parser that records well-formedness errors instead of raising."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.errors: list[str] = []

    def error(self, message: str) -> None:  # pragma: no cover - py3.12 compat
        """Record a well-formedness error reported by the parser."""
        self.errors.append(message)


class HtmlInjectionAuditTests(unittest.TestCase):
    """Hostile payloads never survive into the rendered HTML unescaped."""

    def test_no_raw_payload_survives(self):
        """No raw hostile payload survives in the rendered HTML."""
        html = render_html(_hostile_result())
        for payload in PAYLOADS:
            if escape(payload) == payload:
                # No HTML metacharacters: raw and escaped forms are
                # identical, and the value can only ever appear as inert
                # text (no data-driven URL attributes exist in the report).
                continue
            self.assertNotIn(
                payload,
                html,
                f"hostile payload survived unescaped: {payload!r}",
            )

    def test_metachar_free_payload_only_as_text(self):
        """``javascript:alert(4)`` has no metacharacters, so it is
        identical escaped or not. Pin that it renders as text content,
        never inside a URL attribute (the report has no data-driven
        href/src attributes at all).
        """
        html = render_html(_hostile_result())
        self.assertNotIn('href="javascript:', html)
        self.assertNotIn('src="javascript:', html)
        self.assertIn("javascript:alert(4)", html)

    def test_escaped_payload_present(self):
        """Escaping must not silently drop the data: the inert escaped
        form has to be in the output.
        """
        html = render_html(_hostile_result())
        for payload in PAYLOADS:
            self.assertIn(
                escape(payload),
                html,
                f"escaped payload missing: {payload!r}",
            )

    def test_no_raw_angle_brackets_in_text(self):
        """The rendered text content contains no raw angle brackets."""
        html = render_html(_hostile_result())
        text = _text_content(html)
        self.assertNotIn("<", text)
        self.assertNotIn(">", text)

    def test_inline_script_has_no_breakout(self):
        """The inline script contains no script-breakout sequences."""
        self.assertNotIn("</script", _SCRIPT.lower())
        self.assertNotIn("<!--", _SCRIPT)

    def test_csp_meta_present(self):
        """The rendered HTML carries the CSP meta with its script-src policy."""
        html = render_html(_hostile_result())
        self.assertIn("Content-Security-Policy", html)
        self.assertIn("script-src", _CSP)
        self.assertIn("default-src 'none'", _CSP)

    def test_hostile_decision_maps_to_neutral_class(self):
        """A hostile decision string maps to a neutral CSS class."""
        html = render_html({"decision": '"><script>alert(1)</script>'})
        # The class attribute must only ever carry a fixed-vocabulary value.
        m = re.search(r'<section class="decision ([^"]*)"', html)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "neutral")
        self.assertNotIn("<script>alert(1)", html)

    def test_output_is_well_formed_html(self):
        """The rendered hostile report is well-formed HTML."""
        html = render_html(_hostile_result())
        checker = _WellFormedChecker()
        checker.feed(html)
        checker.close()
        self.assertEqual(checker.errors, [])
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn("</html>", html)

    def test_attribute_contexts_escaped(self):
        """Payloads inside quoted attribute values (data-sort) must not
        break out of the attribute.
        """
        html = render_html(_hostile_result())
        # A raw '"><' sequence would indicate attribute breakout.
        self.assertNotIn('">\n<', html.replace("&quot;", ""))
        for payload in PAYLOADS:
            if '"' in payload:
                self.assertNotIn(payload, html)


if __name__ == "__main__":
    unittest.main()
