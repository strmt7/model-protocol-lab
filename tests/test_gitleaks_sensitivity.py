"""Scanner-sensitivity tests for the gitleaks configuration (RR-01, W1).

Spec: audit PLAN.md W1 item 2 and its acceptance block. The repository's
secret scan is only evidence if it can catch a secret. These tests plant
a never-issued canary in a disposable directory/repository and require
the exact configured CI command to detect it, in directory mode and in
history mode, while a benign control is not detected.

The CI command is ``gitleaks detect --redact --config .gitleaks.toml``
(.github/workflows/gitleaks.yml; the job downloads gitleaks 8.30.1 and
verifies its SHA-256 before running). History mode below is that
command with ``--source`` pointed at the scratch repository; directory
mode adds ``--no-git``, the same command's working-tree form.

The canary is a fabricated private-key block in the shape the upstream
default ``private-key`` rule catches, assembled from fragments at
runtime: the literal never appears in this file, so the repository's
own history scan can never flag the test itself, and the key material
was generated from a repeated sentence — it was never issued to anyone.
A control run with no ``--config`` (pure default rules) must also
detect the canary; that control is what makes a green repo-config run
mean "the default rules are loaded" rather than "the canary went
stale". Before the W1 repair, the repo-config runs below missed the
canary in both modes (the allowlist-only config loaded zero rules);
that failure is the regression evidence for RR-01.

Binary resolution: ``MPL_GITLEAKS_BIN``, else ``gitleaks`` on PATH.
Absent, the tests skip with a reason — the Secret scan workflow runs
this module with the pinned binary (MPL_GITLEAKS_BIN=/tmp/gitleaks),
so the proof executes in CI on every push. A present binary must
report version 8.30.1, the workflow's pin: gitleaks config semantics
are version-dependent, so another version's verdict is not evidence
about the CI scan.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / ".gitleaks.toml"
PINNED_VERSION = "8.30.1"

_HEADER = "-----BEGIN " + "RSA PRIVATE " + "KEY-----"
_FOOTER = "-----END " + "RSA PRIVATE " + "KEY-----"


def _canary_text() -> str:
    """Fabricated private-key block; synthetic body, never issued."""
    raw = base64.b64encode(
        b"mpl gitleaks sensitivity canary: synthetic, never issued. " * 30
    ).decode("ascii")
    body = "\n".join(raw[i : i + 64] for i in range(0, len(raw), 64))
    return f"planted test material\n{_HEADER}\n{body}\n{_FOOTER}\n"


_BENIGN_TEXT = (
    "release notes: the quick brown fox jumps over the lazy dog.\n"
    "build id 12345678-1234-1234-1234-123456789012; contact docs@example.invalid\n"
)


def _resolve_binary() -> Path | None:
    candidate = os.environ.get("MPL_GITLEAKS_BIN")
    if candidate:
        path = Path(candidate)
        return path if path.is_file() else None
    found = shutil.which("gitleaks")
    return Path(found) if found else None


class GitleaksSensitivityTests(unittest.TestCase):
    """The configured scan detects a planted canary; the control is clean."""

    binary: Path

    @classmethod
    def setUpClass(cls) -> None:
        resolved = _resolve_binary()
        if resolved is None:
            raise unittest.SkipTest(
                "gitleaks binary not available (set MPL_GITLEAKS_BIN); the "
                "Secret scan workflow runs this module with the pinned binary"
            )
        cls.binary = resolved
        version = subprocess.run(
            [str(resolved), "version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout.strip()
        if version != PINNED_VERSION:
            raise unittest.SkipTest(
                f"gitleaks {version} present; the CI contract pins "
                f"{PINNED_VERSION}, whose config semantics this test certifies"
            )

    def _scan(self, source: Path, *, config: Path | None, no_git: bool) -> tuple[int, list[str]]:
        """Run the CI-shaped detect command; return (exit, rule ids)."""
        with tempfile.TemporaryDirectory() as report_dir:
            report = Path(report_dir) / "report.json"
            command = [str(self.binary), "detect", "--redact"]
            if no_git:
                command.append("--no-git")
            if config is not None:
                command += ["--config", str(config)]
            command += [
                "--report-format",
                "json",
                "--report-path",
                str(report),
                "--source",
                str(source),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
            rules: list[str] = []
            if report.is_file() and report.stat().st_size > 0:
                findings = json.loads(report.read_text(encoding="utf-8"))
                rules = sorted({str(f["RuleID"]) for f in findings})
            return completed.returncode, rules

    def _scratch_dir(self, files: dict[str, str]) -> tempfile.TemporaryDirectory[str]:
        scratch = tempfile.TemporaryDirectory()
        root = Path(scratch.name)
        for name, text in files.items():
            (root / name).write_text(text, encoding="utf-8")
        return scratch

    def _scratch_repo(self, commits: list[dict[str, str]]) -> tempfile.TemporaryDirectory[str]:
        scratch = tempfile.TemporaryDirectory()
        root = Path(scratch.name)
        git_bin = shutil.which("git")
        assert git_bin is not None  # callers skip when git is unavailable

        def git(*args: str) -> None:
            subprocess.run(
                [git_bin, "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )

        git("init", "-q", ".")
        git("config", "user.email", "sensitivity@example.invalid")
        git("config", "user.name", "gitleaks sensitivity test")
        for index, files in enumerate(commits):
            for name, text in files.items():
                (root / name).write_text(text, encoding="utf-8")
            git("add", ".")
            git("commit", "-q", "-m", f"scratch commit {index}")
        return scratch

    def test_canary_detected_by_default_rules(self) -> None:
        """Control: the canary is a shape the upstream default rules catch."""
        with self._scratch_dir({"planted.txt": _canary_text()}) as root:
            code, rules = self._scan(Path(root), config=None, no_git=True)
        self.assertEqual(code, 1)
        self.assertIn("private-key", rules)

    def test_canary_detected_with_repo_config_dir_mode(self) -> None:
        """The repo config loads the default rules (working-tree scan)."""
        with self._scratch_dir({"planted.txt": _canary_text()}) as root:
            code, rules = self._scan(Path(root), config=CONFIG, no_git=True)
        self.assertEqual(code, 1)
        self.assertIn("private-key", rules)

    def test_canary_detected_with_repo_config_history_mode(self) -> None:
        """The exact CI command (history scan) detects the committed canary."""
        if shutil.which("git") is None:
            self.skipTest("git not available for the history-mode scratch repo")
        commits = [{"planted.txt": _canary_text()}, {"notes.txt": _BENIGN_TEXT}]
        with self._scratch_repo(commits) as root:
            code, rules = self._scan(Path(root), config=CONFIG, no_git=False)
        self.assertEqual(code, 1)
        self.assertIn("private-key", rules)

    def test_benign_control_not_detected_dir_mode(self) -> None:
        """Ordinary prose plus a UUID is not a finding under the repo config."""
        with self._scratch_dir({"notes.txt": _BENIGN_TEXT}) as root:
            code, rules = self._scan(Path(root), config=CONFIG, no_git=True)
        self.assertEqual(code, 0)
        self.assertEqual(rules, [])

    def test_benign_control_not_detected_history_mode(self) -> None:
        """The benign control stays clean through the full history scan."""
        if shutil.which("git") is None:
            self.skipTest("git not available for the history-mode scratch repo")
        with self._scratch_repo([{"notes.txt": _BENIGN_TEXT}]) as root:
            code, rules = self._scan(Path(root), config=CONFIG, no_git=False)
        self.assertEqual(code, 0)
        self.assertEqual(rules, [])

    def test_allowlist_does_not_cover_canary(self) -> None:
        """No allowlist regex may match the canary (no fixing by exemption)."""
        with CONFIG.open("rb") as handle:
            doc = tomllib.load(handle)
        patterns = doc.get("allowlist", {}).get("regexes", [])
        canary = _canary_text()
        for pattern in patterns:
            self.assertIsNone(
                re.search(pattern, canary),
                f"allowlist pattern {pattern!r} matches the sensitivity canary",
            )


if __name__ == "__main__":
    unittest.main()
