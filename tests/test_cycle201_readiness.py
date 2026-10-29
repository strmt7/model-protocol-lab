"""Cycle 201 — release-readiness gates.

Specs (docs/RESEARCH_LOG.md cycle-201 entry; the release-readiness audit
briefs in the coordinator's evidence pack): the changelog must cover the
shipped CLI surface mechanically (the cycle-196/197/198 gap — a whole
user-facing flag shipped with no changelog entry — must be impossible
to repeat silently), the release build's tar normalization must be
deterministic (the member-level diagnosis that found a build-time PAX
mtime leaking into the sdist), and the publish path must stay
tag-gated and token-free (Trusted Publishing; no long-lived PyPI token).
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
import unittest
from pathlib import Path

from model_protocol_lab.cli import build_parser
from tools import release_build

REPO_ROOT = Path(__file__).resolve().parent.parent


def _cli_surface() -> tuple[set[str], set[str]]:
    subcommands: set[str] = set()
    flags: set[str] = set()

    def walk(parser: object) -> None:
        for action in parser._actions:  # noqa: SLF001 - argparse has no public walk
            if action.__class__.__name__ == "_SubParsersAction":
                for name, sub in action.choices.items():
                    subcommands.add(name)
                    walk(sub)
            elif action.option_strings:
                for option in action.option_strings:
                    if option.startswith("--"):
                        flags.add(option)

    walk(build_parser())
    return subcommands, flags


def _mentioned(name: str, text: str) -> bool:
    """Whole-token mention: the name must appear as its own token.

    Plain substring matching (the cycle-201 original) is satisfiable
    by accident: ``--include`` "appeared" only inside
    ``--include-evidence``, ``--seed`` inside ``--seeded-output``,
    and ``cycle 202`` inside ``cycle 2025`` (wave-9 red-team F5,
    confirmed by the cross-check). A mention counts only when the
    characters around the name cannot extend it into a longer token.
    """
    return re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", text) is not None


def _documented_cycles(log_text: str) -> set[int]:
    """Cycles the research log *documents*: those with a cycle entry.

    A cycle entry is a level-2/3 heading naming the cycle ("## Cycle
    188 — ...", "### Build record (cycle 169)"). Body mentions do not
    count: the log forward-references future cycles in slate prose
    ("rank 1 (cycle 203)") and quotes example strings when dissecting
    other defects ("`cycle 202` inside `cycle 2025`"), and neither is a
    shipped cycle a changelog can entry. Extracting every mention
    (the cycle-201 original) made the gate demand changelog entries
    for cycles that have not happened — it failed on exactly
    [203, 2025] in cycle 202's first gate run.
    """
    return {
        int(n)
        for n in re.findall(r"^#{2,3}\s[^\n]*?[Cc]ycle (\d+)\b", log_text, flags=re.MULTILINE)
    }


class TestChangelogGate(unittest.TestCase):
    """The changelog covers the shipped surface — measured, not sampled."""

    def test_every_subcommand_and_flag_appears_in_changelog(self) -> None:
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        subcommands, flags = _cli_surface()
        missing = [name for name in sorted(subcommands | flags) if not _mentioned(name, changelog)]
        self.assertEqual(missing, [], f"CLI surface missing from CHANGELOG.md: {missing}")

    def test_every_recent_cycle_appears_in_changelog(self) -> None:
        # The per-cycle changelog convention holds from cycle 165 onward
        # (earlier cycles are covered by the rolled [0.1.0]/[0.1.0a0]
        # sections); every cycle the research log documents in that
        # window must have a changelog entry. Fails on the pre-201 tree
        # (cycles 196, 197, 198 absent).
        log = (REPO_ROOT / "docs" / "RESEARCH_LOG.md").read_text(encoding="utf-8")
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8").lower()
        recent = sorted(c for c in _documented_cycles(log) if c >= 165)
        missing = [c for c in recent if not _mentioned(f"cycle {c}", changelog)]
        self.assertEqual(missing, [], f"cycles missing from CHANGELOG.md: {missing}")

    def test_documented_cycles_ignores_mentions_without_an_entry(self) -> None:
        # Pins the extraction semantics above: a forward reference in
        # prose and a quoted example string are not documented cycles;
        # a cycle-entry heading is.
        synthetic = (
            "## Cycle 170 (2026-10-20) -- answer-collision audit\n"
            "Body text: rank 1 is cycle 203, not yet built.\n"
            "The gate dissects `cycle 202` inside `cycle 2025`.\n"
            "### Build record (cycle 170)\n"
        )
        self.assertEqual(_documented_cycles(synthetic), {170})


class TestSdistNormalization(unittest.TestCase):
    """normalize_sdist_bytes: deterministic, idempotent, content-preserving."""

    @staticmethod
    def _synthetic_sdist() -> bytes:
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="x.tar.gz", mode="wb", fileobj=buffer, mtime=999) as gz:
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
                info = tarfile.TarInfo("pkg-0.1/file.txt")
                info.mtime = 1792776432
                info.uname = "somebuilder"
                info.gname = "somegroup"
                info.uid = 1000
                info.gid = 1000
                # A build-time float mtime in the PAX header — the exact
                # leak the cycle-201 member-level diagnosis found.
                info.pax_headers = {"mtime": "1792776432.3124018"}
                payload = b"hello release\n"
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            gz.write(tar_buffer.getvalue())
        return buffer.getvalue()

    def test_normalization_is_deterministic_and_scrubs_metadata(self) -> None:
        epoch = 1700000000
        once = release_build.normalize_sdist_bytes(self._synthetic_sdist(), epoch)
        twice = release_build.normalize_sdist_bytes(once, epoch)
        self.assertEqual(once, twice, "normalization must be idempotent")
        with tarfile.open(fileobj=io.BytesIO(once), mode="r:gz") as tar:
            members = tar.getmembers()
            self.assertEqual(len(members), 1)
            member = members[0]
            self.assertEqual(member.mtime, epoch)
            self.assertEqual(member.uid, 0)
            self.assertEqual(member.gid, 0)
            self.assertEqual(member.uname, "")
            self.assertNotIn("mtime", member.pax_headers)
            extracted = tar.extractfile(member)
            self.assertIsNotNone(extracted)
            self.assertEqual(extracted.read(), b"hello release\n")


class TestReleaseSurface(unittest.TestCase):
    def test_py_typed_marker_present(self) -> None:
        self.assertTrue((REPO_ROOT / "model_protocol_lab" / "py.typed").is_file())

    def test_release_workflow_is_tag_gated_and_token_free(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        self.assertIn("tags:", workflow)
        self.assertIn("gh-action-pypi-publish", workflow)
        self.assertIn("id-token: write", workflow)
        # Trusted Publishing means no long-lived PyPI credential anywhere.
        self.assertNotIn("PYPI_API_TOKEN", workflow)
        self.assertNotIn("password:", workflow)

    def test_metadata_story_is_consistent(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("Development Status :: 4 - Beta", pyproject)
        self.assertNotIn("Pre-Alpha", pyproject)
        self.assertIn("torch>=2.6,<3", pyproject)
        init = (REPO_ROOT / "model_protocol_lab" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('__version__ = "0.1.0"', init)
        self.assertIn('version = "0.1.0"', pyproject)


if __name__ == "__main__":
    unittest.main()
