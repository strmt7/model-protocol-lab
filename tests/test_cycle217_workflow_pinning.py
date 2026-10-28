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

Fix round (cycle 217): the first hashed files were resolved on one
host (Linux / CPython 3.12), which silently produced two
cross-target defects CI caught — a pin whose Requires-Python
excludes a matrix interpreter can never install there, and a
marker-gated dependency of another platform never enters the
closure at all. The multi-target rules pinned below keep both
defect classes from returning:

- every pin block in a workflow-consumed hashed file carries at
  least two distinct hashes (a complete release set), except named
  single-file releases;
- build.txt carries the Windows marker deps its parents gate
  (colorama for build, pywin32-ctypes for keyring);
- dev-hf.txt marker-splits the pins whose latest releases exclude
  Python 3.11 (numpy, networkx) into >=3.12 / <3.12 branches.
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


def _pin_blocks(path):
    """Parse a hashed requirements file into (name, version, marker, hashes)."""
    blocks = []
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)(?:;\s*(.*?))?\s*\\?$", line)
        if match:
            current = [match.group(1).lower(), match.group(2), match.group(3) or "", set()]
            blocks.append(current)
        elif current is not None and "--hash=sha256:" in line:
            current[3].add(re.search(r"--hash=sha256:([0-9a-f]+)", line).group(1))
    return [(name, version, marker, hashes) for name, version, marker, hashes in blocks]


def _hashed_workflow_files():
    """Every requirements file a workflow installs with --require-hashes."""
    found = set()
    for path in WORKFLOWS:
        doc = _load_yaml(path)
        for run in _run_steps(doc):
            for line in run.splitlines():
                if "--require-hashes" in line:
                    match = re.search(r"-r\s+(\S+)", line)
                    if match:
                        found.add(REPO_ROOT / match.group(1))
    return sorted(found)


# Releases that ship exactly one distribution file on PyPI (verified
# during the cycle-217 fix round): a complete hash set for them is a
# single hash. Any OTHER pin with fewer than two hashes means its
# set was generated from a partial resolution — the defect class
# that broke the CI 3.11 and Windows legs.
_SINGLE_HASH_RELEASES = {("cuda-pathfinder", "1.8.4"), ("cuda-toolkit", "13.0.3.0")}


class HashedRequirementsMultiTargetTests(unittest.TestCase):
    def test_pin_blocks_carry_multi_hash_sets(self):
        """Each pinned release lists its complete (multi-file) hash set."""
        files = _hashed_workflow_files()
        self.assertTrue(files, "no hashed workflow requirements files found")
        for path in files:
            blocks = _pin_blocks(path)
            self.assertTrue(blocks, f"{path.name}: no pin blocks parsed")
            for name, version, _marker, hashes in blocks:
                if (name, version) in _SINGLE_HASH_RELEASES:
                    self.assertEqual(
                        len(hashes), 1, f"{path.name}: {name}=={version} allowlist drift"
                    )
                else:
                    self.assertGreaterEqual(
                        len(hashes),
                        2,
                        f"{path.name}: {name}=={version} carries {len(hashes)} hash(es); "
                        "a complete release set has one per distribution file",
                    )

    def test_build_txt_carries_windows_marker_deps(self):
        """Deps gated to Windows by their parents appear with win markers.

        build requires colorama only on os_name == "nt" and keyring
        requires pywin32-ctypes only on sys_platform == "win32"; a
        resolution produced on Linux never activates those markers,
        so without these lines the Windows smoke leg resolves two
        requirements the hashed file does not pin (cycle-217 fix).
        """
        blocks = _pin_blocks(REPO_ROOT / "tools" / "requirements" / "build.txt")
        by_name = {name: (marker, hashes) for name, _v, marker, hashes in blocks}
        for dep, gate in (("colorama", "os_name"), ("pywin32-ctypes", "sys_platform")):
            self.assertIn(dep, by_name, f"build.txt is missing the Windows dep {dep}")
            marker, hashes = by_name[dep]
            self.assertIn(gate, marker, f"build.txt: {dep} lost its Windows marker")
            self.assertTrue(hashes, f"build.txt: {dep} carries no hashes")

    def test_dev_hf_python_version_split_pins(self):
        """Pins whose latest releases exclude 3.11 are marker-split.

        numpy 2.5.x and networkx 3.7 declare Requires-Python >=3.12,
        so a single exact pin can never install on the matrix's 3.11
        leg. Each carries a >=3.12 branch (the latest pin) and a
        <3.12 branch (the newest release still admitting 3.11), and
        the older branch must sort below the latest (cycle-217 fix).
        """
        from packaging.version import Version

        blocks = _pin_blocks(REPO_ROOT / "tools" / "requirements" / "dev-hf.txt")
        for name in ("numpy", "networkx"):
            branches = [(v, m) for n, v, m, _h in blocks if n == name]
            self.assertEqual(len(branches), 2, f"dev-hf.txt: {name} must have two marker branches")
            by_marker = {marker: version for version, marker in branches}
            latest = by_marker.get('python_version >= "3.12"')
            legacy = by_marker.get('python_version < "3.12"')
            self.assertIsNotNone(latest, f"dev-hf.txt: {name} lost its >=3.12 branch")
            self.assertIsNotNone(legacy, f"dev-hf.txt: {name} lost its <3.12 branch")
            assert latest is not None and legacy is not None
            self.assertLess(
                Version(legacy),
                Version(latest),
                f"dev-hf.txt: {name} <3.12 branch must be the older release",
            )


if __name__ == "__main__":
    unittest.main()
