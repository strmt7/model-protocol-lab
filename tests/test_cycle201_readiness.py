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


class TestChangelogGate(unittest.TestCase):
    """The changelog covers the shipped surface — measured, not sampled."""

    def test_every_subcommand_and_flag_appears_in_changelog(self) -> None:
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        subcommands, flags = _cli_surface()
        missing = [name for name in sorted(subcommands | flags) if name not in changelog]
        self.assertEqual(missing, [], f"CLI surface missing from CHANGELOG.md: {missing}")

    def test_every_recent_cycle_appears_in_changelog(self) -> None:
        # The per-cycle changelog convention holds from cycle 165 onward
        # (earlier cycles are covered by the rolled [0.1.0]/[0.1.0a0]
        # sections); every cycle the research log documents in that
        # window must have a changelog entry. Fails on the pre-201 tree
        # (cycles 196, 197, 198 absent).
        log = (REPO_ROOT / "docs" / "RESEARCH_LOG.md").read_text(encoding="utf-8")
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8").lower()
        cycles = {int(n) for n in re.findall(r"[Cc]ycle (\d+)", log)}
        recent = sorted(c for c in cycles if c >= 165)
        missing = [c for c in recent if f"cycle {c}" not in changelog]
        self.assertEqual(missing, [], f"cycles missing from CHANGELOG.md: {missing}")


class TestSdistNormalization(unittest.TestCase):
    """normalize_sdist_bytes: deterministic, idempotent, content-preserving."""

    def _synthetic_sdist(self) -> bytes:
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
