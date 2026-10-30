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
from tests.test_cycle219_historyframe import _payload as _history_frame_payload
from tests.test_cycle219_historyframe import _WordBackend as _HfWordBackend
from tests.test_cycle220_prunesweep import _payload as _prune_sweep_payload
from tests.test_cycle220_prunesweep import _WordBackend as _PsWordBackend
from tests.test_cycle221_recordnotation import _payload as _record_notation_payload
from tests.test_cycle221_recordnotation import _WordBackend as _RnWordBackend
from tests.test_cycle223_invisiblechars import _payload as _invisible_chars_payload
from tests.test_cycle223_invisiblechars import _WordBackend as _IcWordBackend
from tests.test_cycle224_confusableskeleton import _payload as _confusable_skeleton_payload

_SECTION_ID_RE = re.compile(r'<section[^>]*? id="([^"]+)"')


def _toc_hrefs(html):
    """Toc hrefs."""
    nav = re.search(r'<nav class="toc".*?</nav>', html, re.S)
    assert nav is not None, "rendered report has no TOC nav"
    return re.findall(r'href="#([^"]+)"', nav.group(0))


def _body_section_ids(html):
    """Body section ids."""
    body = html.split("</nav>", 1)[1]
    return _SECTION_ID_RE.findall(body)


def _populated_result():
    """Populated result."""
    result = _full_result()
    result["canonical_form"] = _canonical_payload()
    result["negation"] = _negation_payload(_WordBackend("negation_drop"))
    result["instruction_distance"] = _instruction_distance_payload(_IdWordBackend("drop"))
    result["history_frame"] = _history_frame_payload(_HfWordBackend("frame_drop"))
    result["prune_sweep"] = _prune_sweep_payload(_PsWordBackend("dose72_drop"))
    result["record_notation"] = _record_notation_payload(_RnWordBackend("json_drop"))
    result["invisible_chars"] = _invisible_chars_payload(_IcWordBackend("echo"))
    result["confusable_skeleton"] = _confusable_skeleton_payload("echo")[0]
    return result


class CanonicalFormBodyTests(unittest.TestCase):
    """Canonical form body tests."""

    def test_canonical_form_body_present_in_rendered_report(self):
        """Canonical form body present in rendered report."""
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
    """Instruction distance body tests."""

    def test_instruction_distance_body_present_in_rendered_report(self):
        """Instruction distance body present in rendered report."""
        html = reporting_mod.render_html(_populated_result())
        self.assertIn('href="#instruction-distance"', html)
        self.assertIn('<section class="card" id="instruction-distance">', html)
        # Body-only copy from inside the section.
        self.assertIn("excess-of-between-losses label", html)
        self.assertGreater(html.index('id="instruction-distance"'), html.index("</nav>"))


class HistoryFrameBodyTests(unittest.TestCase):
    """History frame body tests."""

    def test_history_frame_body_present_in_rendered_report(self):
        """History frame body present in rendered report."""
        html = reporting_mod.render_html(_populated_result())
        self.assertIn('href="#history-frame"', html)
        self.assertIn('<section class="card" id="history-frame">', html)
        # Body-only copy from inside the section.
        self.assertIn("verdict-identity with the synth-history stage is not claimed", html)
        self.assertGreater(html.index('id="history-frame"'), html.index("</nav>"))


class PruneSweepBodyTests(unittest.TestCase):
    """Prune sweep body tests."""

    def test_prune_sweep_body_present_in_rendered_report(self):
        """Prune sweep body present in rendered report."""
        html = reporting_mod.render_html(_populated_result())
        self.assertIn('href="#prune-sweep"', html)
        self.assertIn('<section class="card" id="prune-sweep">', html)
        # Body-only copy from inside the section.
        self.assertIn("partially broken at dose72", html)
        self.assertGreater(html.index('id="prune-sweep"'), html.index("</nav>"))


class RecordNotationBodyTests(unittest.TestCase):
    """Record notation body tests."""

    def test_record_notation_body_present_in_rendered_report(self):
        """Record notation body present in rendered report."""
        html = reporting_mod.render_html(_populated_result())
        self.assertIn('href="#record-notation"', html)
        self.assertIn('<section class="card" id="record-notation">', html)
        # Body-only copy from inside the section.
        self.assertIn("no classification, no winner", html)
        self.assertGreater(html.index('id="record-notation"'), html.index("</nav>"))


class InvisibleCharsBodyTests(unittest.TestCase):
    """Invisible chars body tests."""

    def test_invisible_chars_body_present_in_rendered_report(self):
        """Invisible chars body present in rendered report."""
        html = reporting_mod.render_html(_populated_result())
        self.assertIn('href="#invisible-chars"', html)
        self.assertIn('<section class="card" id="invisible-chars">', html)
        # Body-only copy from inside the section.
        self.assertIn("U+200B ZERO WIDTH SPACE", html)
        self.assertGreater(html.index('id="invisible-chars"'), html.index("</nav>"))


class ConfusableSkeletonBodyTests(unittest.TestCase):
    """Confusable skeleton body tests."""

    def test_confusable_skeleton_body_present_in_rendered_report(self):
        """Confusable skeleton body present in rendered report."""
        html = reporting_mod.render_html(_populated_result())
        self.assertIn('href="#confusable-skeleton"', html)
        self.assertIn('<section class="card" id="confusable-skeleton">', html)
        # Body-only copy from inside the section.
        self.assertIn("U+0434 CYRILLIC SMALL LETTER DE", html)
        self.assertGreater(html.index('id="confusable-skeleton"'), html.index("</nav>"))


class TocBodyCorrespondenceTests(unittest.TestCase):
    """Toc body correspondence tests."""

    def test_populated_sections_match_toc_exactly(self):
        """Populated sections match toc exactly."""
        html = reporting_mod.render_html(_populated_result())
        hrefs = _toc_hrefs(html)
        body_ids = _body_section_ids(html)
        # Both probe sections populated for this render are linked.
        self.assertIn("canonical-form", hrefs)
        self.assertIn("negation", hrefs)
        self.assertIn("instruction-distance", hrefs)
        self.assertIn("history-frame", hrefs)
        self.assertIn("prune-sweep", hrefs)
        self.assertIn("record-notation", hrefs)
        self.assertIn("invisible-chars", hrefs)
        self.assertIn("confusable-skeleton", hrefs)
        # Every link lands on a rendered section, every rendered
        # section is linked, in the same order.
        self.assertEqual(hrefs, body_ids)


class RenderHtmlSourceGuardTests(unittest.TestCase):
    """Every _toc entry's variable must reach the returned document."""

    @staticmethod
    def _render_html_parts():
        """Render html parts."""
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
        """Every toc section is concatenated."""
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
