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
}


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
        counts = dict.fromkeys(PATTERNS, 0)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            for name, pat in PATTERNS.items():
                counts[name] += len(pat.findall(line))
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


if __name__ == "__main__":
    unittest.main()
