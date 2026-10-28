"""Cycle 213: TOC <-> body correspondence for rendered report sections.

Cycle 209 shipped the canonical-form section computed and linked from
the table of contents but never concatenated into render_html's body:
the TOC link rendered (the section HTML was non-empty, so _toc linked
it) and landed nowhere. The cycle-209 tests called
_canonical_form_section directly, so the gap was invisible; the
cycle-33 correspondence check rendered a result with no probe payloads
populated, so no probe section rendered at all there either.

These tests render through the real render_html path:

- the canonical-form section BODY (not just its TOC link) is present
  when the payload is populated;
- on a populated render, the TOC hrefs and the body section ids
  correspond exactly (same set, same order);
- a source-level guard over render_html itself: every variable passed
  to _toc is concatenated into the returned document, and every
  section variable concatenated is listed in the TOC. This guards the
  whole defect class for sections added later, whose payloads a
  fixture here cannot populate.
"""

import ast
import re
import unittest
from pathlib import Path

from model_protocol_lab import reporting as reporting_mod
from tests.test_cycle33_reporting_verify import _full_result
from tests.test_cycle209_canonicalform import _payload as _canonical_payload
from tests.test_cycle212_negation import _payload as _negation_payload
from tests.test_cycle212_negation import _WordBackend
from tests.test_cycle218_instructiondistance import _payload as _instruction_distance_payload
from tests.test_cycle218_instructiondistance import _WordBackend as _IdWordBackend

_SECTION_ID_RE = re.compile(r'<section[^>]*? id="([^"]+)"')


def _toc_hrefs(html):
    nav = re.search(r'<nav class="toc".*?</nav>', html, re.S)
    assert nav is not None, "rendered report has no TOC nav"
    return re.findall(r'href="#([^"]+)"', nav.group(0))


def _body_section_ids(html):
    body = html.split("</nav>", 1)[1]
    return _SECTION_ID_RE.findall(body)


def _populated_result():
    result = _full_result()
    result["canonical_form"] = _canonical_payload()
    result["negation"] = _negation_payload(_WordBackend("negation_drop"))
    result["instruction_distance"] = _instruction_distance_payload(_IdWordBackend("drop"))
    return result


class CanonicalFormBodyTests(unittest.TestCase):
    def test_canonical_form_body_present_in_rendered_report(self):
        html = reporting_mod.render_html(_populated_result())
        # The TOC link alone is not the section: pre-fix it rendered
        # while the body below was absent.
        self.assertIn('href="#canonical-form"', html)
        self.assertIn('<section class="card" id="canonical-form">', html)
        # Body-only copy from inside the section (the TOC carries the
        # title as link text, so the title alone cannot prove this).
        self.assertIn("Twenty-four generated cases each bury one", html)
        # The section sits in the body, after the TOC nav.
        self.assertGreater(html.index('id="canonical-form"'), html.index("</nav>"))


class InstructionDistanceBodyTests(unittest.TestCase):
    def test_instruction_distance_body_present_in_rendered_report(self):
        html = reporting_mod.render_html(_populated_result())
        self.assertIn('href="#instruction-distance"', html)
        self.assertIn('<section class="card" id="instruction-distance">', html)
        # Body-only copy from inside the section.
        self.assertIn("excess-of-between-losses label", html)
        self.assertGreater(html.index('id="instruction-distance"'), html.index("</nav>"))


class TocBodyCorrespondenceTests(unittest.TestCase):
    def test_populated_sections_match_toc_exactly(self):
        html = reporting_mod.render_html(_populated_result())
        hrefs = _toc_hrefs(html)
        body_ids = _body_section_ids(html)
        # Both probe sections populated for this render are linked.
        self.assertIn("canonical-form", hrefs)
        self.assertIn("negation", hrefs)
        self.assertIn("instruction-distance", hrefs)
        # Every link lands on a rendered section, every rendered
        # section is linked, in the same order.
        self.assertEqual(hrefs, body_ids)


class RenderHtmlSourceGuardTests(unittest.TestCase):
    """Every _toc entry's variable must reach the returned document."""

    def _render_html_parts(self):
        source = Path(reporting_mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        render = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "render_html"
        )
        section_vars = set()
        toc_vars = []
        for node in ast.walk(render):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id.endswith("_section"):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            section_vars.add(target.id)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_toc"
            ):
                for entry in node.args[0].elts:
                    toc_vars.append(entry.elts[2].id)
        return_names = []
        for node in ast.walk(render):
            if isinstance(node, ast.Return) and node.value is not None:
                stack = [node.value]
                while stack:
                    current = stack.pop()
                    if isinstance(current, ast.BinOp) and isinstance(current.op, ast.Add):
                        stack.append(current.left)
                        stack.append(current.right)
                    elif isinstance(current, ast.Name):
                        return_names.append(current.id)
        return section_vars, toc_vars, return_names

    def test_every_toc_section_is_concatenated(self):
        section_vars, toc_vars, return_names = self._render_html_parts()
        self.assertTrue(toc_vars, "no TOC entries extracted from render_html")
        for var in toc_vars:
            self.assertIn(
                var,
                return_names,
                f"TOC entry variable {var} is never concatenated into the body",
            )
        for var in section_vars:
            self.assertIn(
                var,
                toc_vars,
                f"section variable {var} is computed but missing from the TOC",
            )
            self.assertIn(
                var,
                return_names,
                f"section variable {var} is computed but never concatenated into the body",
            )


if __name__ == "__main__":
    unittest.main()
