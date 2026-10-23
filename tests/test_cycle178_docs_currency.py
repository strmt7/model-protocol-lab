"""Docs-currency audit pins (Cycle 178, 2026-10-20).

Cycle 178 brought the user-facing docs back in line with the shipped
tree after a read-only swarm audit (researcher/verifier briefs in the
wave-7 round) found systematic drift: README Capabilities omitted
shipped probes and the `pack` command, the README sample-report
paragraph claimed sections the committed sample does not render,
USAGE sections 6 and 12 carried stale inventories, USAGE 8.33-8.36
sat physically after section 14, the FAQ stopped at the cycle-130
era, the CHANGELOG had no entries for cycles 164/166/167/172 (and no
Added entry for 165), VALIDATION's evidence header claimed
"cycles 10-153" while citing cycles 176-177, and its ruff line
stated a file count without the scope that produces it.

These tests pin the corrected state so the drift cannot silently
return. They do NOT pin historical snapshot sections (older
VALIDATION evidence blocks keep their own era's numbers, per the
standing snapshot rule).
"""

import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# NB: no .resolve() on sys.executable -- the project .venv python is a
# symlink, and resolving it would escape into the system bin directory.
BIN_DIR = Path(sys.executable).parent

README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
USAGE = (REPO_ROOT / "docs" / "USAGE.md").read_text(encoding="utf-8")
CHANGELOG = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
VALIDATION = (REPO_ROOT / "VALIDATION.md").read_text(encoding="utf-8")
SEALED = (REPO_ROOT / "docs" / "SEALED_BUNDLE_DESIGN.md").read_text(encoding="utf-8")

# Every stage/probe argument on the `run` subcommand (cli.py run
# block, lines 185-483): the README Capabilities section must name
# each one. Core plumbing (--model, --dataset, ...) is documented in
# Quick start / USAGE section 5 and intentionally not pinned here.
RUN_PROBE_FLAGS = [
    "--transfer-model",
    "--perturb",
    "--token-healing",
    "--hierarchy",
    "--prefill",
    "--synth-history",
    "--few-shot",
    "--batch-invariance",
    "--position",
    "--stop-cause",
    "--ws-scan",
    "--structured",
    "--deliforge",
    "--fake-turn",
    "--tool-call",
    "--verbosity",
    "--refusals",
    "--scorer-lens",
    "--aliasing",
    "--repeat",
    "--distract",
    "--leading",
    "--numform",
    "--dateline",
    "--multiturn",
    "--needle",
    "--placement",
    "--choice-audit",
    "--cache-profile",
    "--cache-profile-block-size",
    "--confidentiality",
    "--answer-collision",
    "--template-robustness",
    "--ablate-fields",
    "--include-evidence",
    "--sensitivity-sweep",
    "--max-drop-bytes",
    "--logprob-diagnostic",
]


class DocsCurrencyTests(unittest.TestCase):
    def test_readme_names_every_run_probe_flag(self):
        for flag in RUN_PROBE_FLAGS:
            self.assertIn(
                flag,
                README,
                f"README does not name run probe flag {flag}",
            )

    def test_readme_names_pack_and_fixtures_fuzz(self):
        # Both shipped without any README mention before cycle 178.
        self.assertIn("model-protocol-lab pack", README)
        self.assertIn("fixtures fuzz", README)

    def test_readme_canonicality_bullet_not_inverted(self):
        # boundary_audit.rate is n_problem / n_usable: the fraction
        # of pairs that do NOT re-encode. The pre-178 bullet called
        # the complementary (success) fraction the PBP rate.
        self.assertIn("does *not* re-encode to the recorded ids", README)
        self.assertNotIn(
            "whose concatenated text re-encodes to the recorded ids",
            README,
        )

    def test_changelog_backfilled_cycles(self):
        # Cycles that shipped without entries, found by the audit.
        for cycle in (164, 166, 167, 172):
            self.assertIn(f"(cycle {cycle} ", CHANGELOG)
        # Cycle 165 had a Fixed heading but no Added heading for its
        # surface-noise feature.
        self.assertIn("### Added (cycle 165 ", CHANGELOG)

    def test_sealed_bundle_design_scopes_strict_claims(self):
        # F8: the pre-parse enforces disk/ZIP64 properties at the
        # EOCD record only; the design doc must say so.
        self.assertIn("ZIP64 sentinel", SEALED)
        self.assertNotIn("and ZIP64 is rejected", SEALED)

    def test_usage_stage_sections_inside_section_8(self):
        # Sections 8.33-8.36 sat after section 14 until cycle 178.
        idx_833 = USAGE.index("### 8.33 ")
        idx_836 = USAGE.index("### 8.36 ")
        idx_sec9 = USAGE.index("## 9. ")
        self.assertLess(idx_833, idx_sec9)
        self.assertLess(idx_836, idx_sec9)

    def test_validation_header_covers_latest_cycle(self):
        # The first "## " section (index 1; index 0 is the title).
        # Bump the pinned range whenever a cycle updates the top
        # evidence section (179 -> 180 in cycle 180; 192 -> 195 in
        # cycle 195; 195 -> 199 in cycle 199; 199 -> 200 in cycle 200).
        top = VALIDATION.split("\n## ")[1]
        self.assertIn("cycles 10–201", top)


class ValidationRuffCountPinTests(unittest.TestCase):
    def test_validation_ruff_count_matches_live_ci_scope(self):
        # The claim lives in VALIDATION's top evidence section only;
        # older sections are historical snapshots with their own
        # era's counts and must not be compared to the live tree.
        top = VALIDATION.split("\n## ")[1]
        match = re.search(r"\((\d+) files in that CI scope", top)
        self.assertIsNotNone(match, "VALIDATION top section lost its scoped ruff count")
        claimed = int(match.group(1))  # type: ignore[union-attr]
        exe = BIN_DIR / "ruff"
        self.assertTrue(exe.exists(), f"expected tool at {exe}")
        proc = subprocess.run(
            [
                str(exe),
                "format",
                "--check",
                "model_protocol_lab",
                "tests",
                "examples",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout[-1000:])
        live = re.search(r"(\d+) files already formatted", proc.stdout)
        self.assertIsNotNone(live)
        self.assertEqual(
            claimed,
            int(live.group(1)),  # type: ignore[union-attr]
            "VALIDATION's ruff count drifted from the live CI-scope run",
        )


if __name__ == "__main__":
    unittest.main()
