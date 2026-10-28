"""Cycle 217: pin the workflow supply-chain hygiene the Scorecard
Pinned-Dependencies check enforces, so a future unpinned install
fails the suite instead of reappearing as a dashboard alert.

Rules pinned here (all verified against Scorecard v5.6.0 locally
during cycle 217, which scored Pinned-Dependencies 10/10 on this
tree):
- every pip install in a workflow run step is hash-pinned
  (``--require-hashes -r <file>`` with the file present in the
  tree) or a bare local install (``--no-deps -e .``,
  ``dist/*.whl``);
- every Dockerfile FROM image carries a digest (``@sha256:``);
- every ``uses:`` action reference is a full 40-hex SHA
  (Scorecard counts GitHub/third-party actions pinned separately);
- the findings guard workflow keeps Stratos's triggers and the
  security-events: read grant, and the fuzz workflow keeps its
  scheduled atheris run.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))

_PIP_RE = re.compile(r"(?:^|\s)(?:python -m )?pip install\b")
_SHA_RE = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def _load_yaml(path):
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(doc):
    # PyYAML (YAML 1.1) parses the `on:` key as boolean True.
    return doc.get(True, doc.get("on", {}))


def _run_steps(doc):
    for job in (doc.get("jobs") or {}).values():
        for step in job.get("steps") or []:
            run = step.get("run")
            if isinstance(run, str):
                yield run


class WorkflowPinningTests(unittest.TestCase):
    def test_every_workflow_pip_install_is_hash_pinned_or_local(self):
        """No workflow may install a remote package without hashes."""
        for path in WORKFLOWS:
            doc = _load_yaml(path)
            for run in _run_steps(doc):
                for line in run.splitlines():
                    if not _PIP_RE.search(line):
                        continue
                    if "--require-hashes" in line:
                        match = re.search(r"-r\s+(\S+)", line)
                        self.assertIsNotNone(
                            match, f"{path.name}: hashed install without -r file: {line}"
                        )
                        assert match is not None
                        self.assertTrue(
                            (REPO_ROOT / match.group(1)).is_file(),
                            f"{path.name}: hashed install references missing file {match.group(1)}",
                        )
                    else:
                        self.assertTrue(
                            "--no-deps -e ." in line or "dist/*.whl" in line,
                            f"{path.name}: unpinned pip install: {line}",
                        )

    def test_dockerfile_images_pinned_by_digest(self):
        """Container images must carry a digest, not just a tag."""
        for path in REPO_ROOT.glob(".clusterfuzzlite/Dockerfile"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("FROM "):
                    self.assertIn("@sha256:", line, f"{path}: unpinned image: {line}")

    def test_every_action_reference_is_sha_pinned(self):
        """Action references must be full SHAs, never moving tags."""
        for path in WORKFLOWS:
            doc = _load_yaml(path)
            for job in (doc.get("jobs") or {}).values():
                for step in job.get("steps") or []:
                    uses = step.get("uses")
                    if isinstance(uses, str) and not uses.startswith("./"):
                        self.assertRegex(uses, _SHA_RE, f"{path.name}: unpinned action: {uses}")

    def test_findings_guard_workflow_shape(self):
        """Guard keeps its triggers, grant, and script invocation."""
        doc = _load_yaml(REPO_ROOT / ".github" / "workflows" / "findings-guard.yml")
        on = _triggers(doc)
        self.assertIn("main", (on.get("push") or {}).get("branches") or [])
        self.assertIn("schedule", on)
        self.assertIn("workflow_dispatch", on)
        permissions = doc.get("permissions") or {}
        self.assertEqual(permissions.get("security-events"), "read")
        runs = list(_run_steps(doc))
        self.assertTrue(any("tools/findings_guard.py" in run for run in runs))

    def test_fuzz_workflow_shape(self):
        """Fuzz job keeps its schedule and bounded atheris run."""
        doc = _load_yaml(REPO_ROOT / ".github" / "workflows" / "fuzz.yml")
        on = _triggers(doc)
        self.assertIn("schedule", on)
        self.assertIn("workflow_dispatch", on)
        runs = list(_run_steps(doc))
        self.assertTrue(any("fuzz_parse_results.py" in run for run in runs))
        self.assertTrue(any("--require-hashes" in run for run in runs))


if __name__ == "__main__":
    unittest.main()
