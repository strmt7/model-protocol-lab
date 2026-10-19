"""Cycle 84: release-readiness audit pins.

The release-readiness definition of done requires that a few numbers the
docs cite stay pinned to the actual tree: the 29 catalog families and the
118 mutate candidates (README), the eight CLI subcommands, and version
consistency across pyproject / __init__ / CITATION.cff. None of these
reads the environment, the network, or the calendar.

METHOD
------
Pure stdlib imports; ``tomllib`` reads pyproject; the catalog is counted
through the public ``candidates()`` API (mutate=True), the same
hypothesis space ``doctor --portability-matrix`` measures.
"""

import re
import tomllib
import unittest
from pathlib import Path

import model_protocol_lab
from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.cli import build_parser

ROOT = Path(__file__).resolve().parent.parent


class VersionConsistencyTest(unittest.TestCase):
    def test_pyproject_version_matches_package(self) -> None:
        with open(ROOT / "pyproject.toml", "rb") as fh:
            declared = tomllib.load(fh)["project"]["version"]
        self.assertEqual(model_protocol_lab.__version__, declared)

    def test_citation_cff_version_matches(self) -> None:
        text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        match = re.search(r"^version:\s*(\S+)\s*$", text, re.MULTILINE)
        self.assertIsNotNone(match, "CITATION.cff carries no version field")
        assert match is not None
        self.assertEqual(match.group(1), model_protocol_lab.__version__)

    def test_version_is_final_release(self) -> None:
        # Release decision (cycle 84, owner-ordered): 0.1.0 -- the honest
        # pre-release number for a first public release, NOT 1.0.0 (no
        # long-term API-stability commitment yet) and NOT an alpha suffix
        # (the suite executes, CI is green, packaging verifies). Pin the
        # final-release shape so a future alpha/dev label is deliberate.
        self.assertRegex(model_protocol_lab.__version__, r"^\d+\.\d+\.\d+$")


class CatalogCountTest(unittest.TestCase):
    def test_twenty_nine_families(self) -> None:
        self.assertEqual(len(CATALOG), 29)

    def test_one_hundred_eighteen_mutate_candidates(self) -> None:
        total = sum(len(candidates(family, mutate=True)) for family in CATALOG)
        self.assertEqual(total, 118)

    def test_every_family_has_a_base_candidate(self) -> None:
        for family in CATALOG:
            self.assertEqual(len(candidates(family, mutate=False)), 1)


class SubcommandSurfaceTest(unittest.TestCase):
    def test_ten_subcommands(self) -> None:
        parser = build_parser()
        actions = list(parser._subparsers._group_actions)
        names = sorted(actions[0].choices.keys())
        self.assertEqual(
            names,
            [
                "catalog",
                "debug",
                "diff-backends",
                "diff-reports",
                "doctor",
                "fixtures",
                "inspect",
                "run",
                "surface-noise",
                "verify",
            ],
        )


class ChangelogCoverageTest(unittest.TestCase):
    def test_unreleased_covers_cycle_84_span(self) -> None:
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        unreleased = text.split("## [0.1.0a0]")[0]
        for marker in (
            "cycles 14–15, 52, 54",
            "cycles 45–83",
            "missing_or_unsafe_asset",
            "cycles 36–83",
        ):
            self.assertIn(marker, unreleased)


if __name__ == "__main__":
    unittest.main()
