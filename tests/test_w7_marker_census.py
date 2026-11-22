"""W7-4 marker census gate (RR-17): scanner exclusions are pinned.

The repo's scanner-exclusion policy — every exclusion marker
(noqa / nosec / type: ignore / pragma: no cover) is deliberate,
reason-carrying, and enumerated; the withdrawn DeepSource marker
class stays at zero everywhere, permanently — was, before this
test, a convention audited by hand. This gate makes it mechanical:
it recounts every marker occurrence in the five source trees with
the W1 inventory method and compares against the pinned census in
``tests/marker_census.json`` (per-tree totals for all trees, plus
a per-file map for the package, where the scanners' product-code
surface lives). Any addition, removal, or redistribution inside
the package fails here until the census is regenerated and the
change reviewed — the census update is the deliberate act the
policy requires.

The census also counts ruff's file-level directives
(``# ruff: noqa`` / ``# flake8: noqa``, bare and coded): one such
line suppresses findings across a whole file while matching no
per-line marker pattern, so without its own classes it would be
an exclusion the gate cannot see (wave-15 BH15-4).

This file is excluded from its own census: it defines the marker
patterns as string literals, which the line-based count cannot
distinguish from markers themselves. It carries no markers.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF = "tests/test_w7_marker_census.py"
TREES = ["model_protocol_lab", "tests", "examples", "tools", "fuzz"]
PATTERNS = {
    "noqa": re.compile(r"#\s*noqa\b"),
    "nosec": re.compile(r"#\s*nosec\b"),
    "type_ignore": re.compile(r"#\s*type:\s*ignore"),
    "pragma_no_cover": re.compile(r"#\s*pragma:\s*no cover"),
    "skipcq": re.compile(r"skip" + "cq"),
    # File-level directives: the bare class excludes the coded
    # form (a colon after noqa), the coded class requires it.
    "file_noqa_ruff": re.compile(r"#\s*ruff:\s*noqa\b(?!\s*:)"),
    "file_noqa_ruff_coded": re.compile(r"#\s*ruff:\s*noqa\s*:\s*[A-Z]"),
    "file_noqa_flake8": re.compile(r"#\s*flake8:\s*noqa\b(?!\s*:)"),
    "file_noqa_flake8_coded": re.compile(r"#\s*flake8:\s*noqa\s*:\s*[A-Z]"),
}


def _count_text(text: str) -> dict[str, int]:
    """Count marker occurrences in one file's text."""
    counts = dict.fromkeys(PATTERNS, 0)
    for line in text.splitlines():
        for name, pat in PATTERNS.items():
            counts[name] += len(pat.findall(line))
    return counts


def _count_tree(tree: str) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Count marker occurrences in one tree (the W1 method)."""
    totals = dict.fromkeys(PATTERNS, 0)
    per_file: dict[str, dict[str, int]] = {}
    base = ROOT / tree
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(ROOT))
        if rel == SELF:
            continue
        counts = _count_text(path.read_text(encoding="utf-8", errors="replace"))
        if any(counts.values()):
            per_file[rel] = counts
        for name in PATTERNS:
            totals[name] += counts[name]
    return totals, per_file


class MarkerCensusTests(unittest.TestCase):
    """The live marker census equals the pinned census."""

    def test_census_matches_pinned(self):
        """Recount every tree and compare totals + package per-file."""
        pinned = json.loads((ROOT / "tests" / "marker_census.json").read_text())
        for tree in TREES:
            totals, per_file = _count_tree(tree)
            with self.subTest(tree=tree):
                self.assertEqual(
                    totals,
                    pinned["totals"][tree],
                    f"{tree}: marker totals drifted from the pinned census; "
                    "regenerate tests/marker_census.json in the same commit "
                    "that deliberately adds or removes an exclusion",
                )
                if tree == "model_protocol_lab":
                    self.assertEqual(
                        per_file,
                        pinned["package_per_file"],
                        "package marker distribution drifted from the "
                        "pinned census (see tests/marker_census.json)",
                    )

    def test_withdrawn_marker_class_is_zero_everywhere(self):
        """The withdrawn DeepSource marker class stays at zero."""
        for tree in TREES:
            totals, _per_file = _count_tree(tree)
            with self.subTest(tree=tree):
                self.assertEqual(totals["skipcq"], 0)


class FileLevelDirectiveCensusTests(unittest.TestCase):
    """Planted file-level noqa directives are counted (BH15-4).

    Negative control for the census itself: every spelling ruff
    honors as a file-level directive (executed against real ruff
    in the wave-15 hunt: bare at the top, mid-file, and after a
    docstring; the flake8 spelling; the coded form) must land in
    its census class, and in no other — in particular not in the
    per-line ``noqa`` class, whose pattern it evaded before.
    """

    SPELLINGS = {
        "# ruff: noqa\n": "file_noqa_ruff",
        "import os\n# ruff: noqa\n": "file_noqa_ruff",
        '"""Doc."""\n# ruff: noqa\n': "file_noqa_ruff",
        "# flake8: noqa\n": "file_noqa_flake8",
        "# ruff: noqa: F401\n": "file_noqa_ruff_coded",
        "# flake8: noqa: F401\n": "file_noqa_flake8_coded",
    }

    def test_each_spelling_counts_in_its_own_class(self):
        """Every honored spelling is visible to the census."""
        file_classes = [name for name in PATTERNS if name.startswith("file_noqa")]
        for text, expected in self.SPELLINGS.items():
            with self.subTest(spelling=text):
                counts = _count_text(text)
                self.assertEqual(counts[expected], 1)
                self.assertEqual(counts["noqa"], 0)
                for name in file_classes:
                    if name != expected:
                        self.assertEqual(counts[name], 0, name)

    def test_plain_noqa_comment_is_not_a_file_directive(self):
        """The per-line marker does not leak into the new classes."""
        counts = _count_text("import os  # noqa: F401\n")
        self.assertEqual(counts["noqa"], 1)
        for name in PATTERNS:
            if name.startswith("file_noqa"):
                self.assertEqual(counts[name], 0, name)


if __name__ == "__main__":
    unittest.main()
