"""Cycle 228 — release machinery: sdist mode normalization, pinned backend.

Pins for the deployment-preparation findings F-A2/F-A1 and the
security-audit findings B-02/B-05/A-L1:

* ``normalize_sdist_bytes`` canonicalizes tar member modes, so an
  sdist's hash no longer depends on the checkout's umask (F-A2).
* The PEP 517 backend is pinned exactly (pyproject
  ``setuptools==84.2.1``, the same version the hash-pinned
  pip-setuptools.txt installs) and every in-repo build path builds
  with ``--no-isolation``, which refuses a different ambient backend
  (B-02).
* The release workflow bootstraps the pinned pip like every other
  workflow before installing anything else (B-05).
* The CI test job no longer holds ``actions: write``; the docs build
  job performs both the pip cache and an artifact upload under
  ``contents: read`` alone, disproving the grant's old justification
  (A-L1).
"""

from __future__ import annotations

import gzip
import io
import tarfile
import tomllib
import unittest
from pathlib import Path

from tools import release_build

REPO_ROOT = Path(__file__).resolve().parent.parent


def _synthetic_sdist(modes: dict[str, int]) -> bytes:
    """Build a tiny .tar.gz whose members carry the given modes."""
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="x.tar.gz", mode="wb", fileobj=buffer, mtime=999) as gz:
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, mode in modes.items():
                info = tarfile.TarInfo(name)
                info.mode = mode
                info.mtime = 1792776432
                if name.endswith("/"):
                    info.type = tarfile.DIRTYPE
                    tar.addfile(info)
                else:
                    payload = b"payload\n"
                    info.size = len(payload)
                    tar.addfile(info, io.BytesIO(payload))
        gz.write(tar_buffer.getvalue())
    return buffer.getvalue()


def _member_modes(data: bytes) -> dict[str, int]:
    """Read back {member name: mode} from a .tar.gz byte string."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        return {member.name: member.mode for member in tar.getmembers()}


class TestSdistModeNormalization(unittest.TestCase):
    """F-A2: member modes are canonicalized, umask fingerprints erased."""

    def test_canonical_mode_mapping(self) -> None:
        """Test dirs/executables map to 0o755, plain files to 0o644."""
        plain = tarfile.TarInfo("pkg-0.1/file.txt")
        plain.mode = 0o660
        self.assertEqual(release_build._canonical_mode(plain), 0o644)
        script = tarfile.TarInfo("pkg-0.1/tool.py")
        script.mode = 0o777
        self.assertEqual(release_build._canonical_mode(script), 0o755)
        group_exec_only = tarfile.TarInfo("pkg-0.1/tool2.py")
        group_exec_only.mode = 0o610
        self.assertEqual(release_build._canonical_mode(group_exec_only), 0o755)
        directory = tarfile.TarInfo("pkg-0.1/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o700
        self.assertEqual(release_build._canonical_mode(directory), 0o755)

    def test_modes_normalized_and_idempotent(self) -> None:
        """Test host modes (0660/0777/0770) normalize, and stay normalized."""
        source = _synthetic_sdist(
            {"pkg-0.1/": 0o770, "pkg-0.1/file.txt": 0o660, "pkg-0.1/tool.py": 0o777}
        )
        once = release_build.normalize_sdist_bytes(source, 1700000000)
        self.assertEqual(
            _member_modes(once),
            {"pkg-0.1": 0o755, "pkg-0.1/file.txt": 0o644, "pkg-0.1/tool.py": 0o755},
        )
        twice = release_build.normalize_sdist_bytes(once, 1700000000)
        self.assertEqual(once, twice, "mode normalization must be idempotent")

    def test_mode_only_difference_normalizes_away(self) -> None:
        """Test two sdists differing only in modes normalize to one bytes."""
        umask_022 = _synthetic_sdist({"pkg-0.1/": 0o755, "pkg-0.1/file.txt": 0o644})
        umask_002 = _synthetic_sdist({"pkg-0.1/": 0o775, "pkg-0.1/file.txt": 0o664})
        self.assertNotEqual(umask_022, umask_002)
        self.assertEqual(
            release_build.normalize_sdist_bytes(umask_022, 1700000000),
            release_build.normalize_sdist_bytes(umask_002, 1700000000),
        )


class TestPinnedBuildBackend(unittest.TestCase):
    """B-02/B-05: one exact backend pin, installed hashed, used ambient."""

    def test_pyproject_backend_pin_matches_hashed_requirements(self) -> None:
        """Test pyproject's exact pin equals pip-setuptools.txt's pin."""
        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            pyproject = tomllib.load(handle)
        self.assertEqual(pyproject["build-system"]["requires"], ["setuptools==84.2.1"])
        requirements = (REPO_ROOT / "tools" / "requirements" / "pip-setuptools.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("setuptools==84.2.1", requirements)

    def test_release_build_uses_no_isolation(self) -> None:
        """Test release_build.py builds both artifacts with --no-isolation."""
        source = (REPO_ROOT / "tools" / "release_build.py").read_text(encoding="utf-8")
        self.assertEqual(source.count('"--no-isolation"'), 2)

    def test_release_workflow_bootstraps_pinned_pip_first(self) -> None:
        """Test release.yml installs pip.txt before any other requirements."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        self.assertIn("pip-setuptools.txt", workflow)
        pip_bootstrap = workflow.index("tools/requirements/pip.txt")
        builder_install = workflow.index("tools/requirements/build.txt")
        self.assertLess(pip_bootstrap, builder_install)

    def test_ci_and_smoke_build_against_the_pinned_backend(self) -> None:
        """Test the CI package job and the smoke job use --no-isolation."""
        for name in ("ci.yml", "smoke.yml"):
            workflow = (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
            self.assertIn("pip-setuptools.txt", workflow, name)
            self.assertIn("--no-isolation", workflow, name)


class TestCiLeastPrivilege(unittest.TestCase):
    """A-L1: no workflow job holds actions: write without a true need."""

    def test_ci_test_job_has_no_actions_write(self) -> None:
        """Test ci.yml grants actions: write nowhere."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertNotIn("actions: write", workflow)
        # The grants the test job actually needs stay.
        self.assertIn("id-token: write # Codecov OIDC upload", workflow)


if __name__ == "__main__":
    unittest.main()
