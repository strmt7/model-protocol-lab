"""W5-1 (RR-11): the release smoke must prove the installed wheel works.

The pre-fix smoke in release.yml installed the wheel into a venv and
then ran its console scripts from the checkout with the checkout
reachable on PYTHONPATH: with no artifact installed at all, every
command still exited 0 against the checkout (readiness lane, sabotage
arms A and B). The replacement (tools/artifact_smoke.py) creates its
own venv, scrubs the environment it passes to every subprocess, and
asserts — before trusting any command output — that the imported
package resolves under that venv's site-packages and that its version
equals the wheel's own METADATA version.

These tests build a wheel by zipping the real package directory (the
ambient venv's setuptools is not the pinned backend, so a PEP 517
build is not available to the suite; the smoke's contract is about
install layout and import origin, not about how the zip was made) and
run the tool end to end: the genuine arm passes, the PYTHONPATH
sabotage cannot reach the smoke, a gutted wheel fails, and a wheel
whose metadata version disagrees with the package fails. The probe
level reproduces the readiness sabotage arms exactly: an interpreter
whose import resolves to the checkout is rejected by the origin
assertion whether or not a wheel was installed into it.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import model_protocol_lab  # noqa: E402
from tools import artifact_smoke  # noqa: E402

PACKAGE_VERSION = model_protocol_lab.__version__
DIST_INFO = f"model_protocol_lab-{PACKAGE_VERSION}.dist-info"


def _record_row(name: str, data: bytes) -> str:
    """One RECORD row with a real sha256, as pip writes them."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return f"{name},sha256={digest},{len(data)}"


def build_test_wheel(
    dest_dir: Path,
    *,
    omit: frozenset[str] = frozenset(),
    metadata_version: str | None = None,
) -> Path:
    """Zip the real package tree into an installable wheel.

    ``omit`` drops package members by arcname (the gutted-wheel arm);
    ``metadata_version`` rewrites the dist-info METADATA version (the
    version-mismatch arm). Everything else is the tree's real bytes.
    """
    version = metadata_version or PACKAGE_VERSION
    dist_info = f"model_protocol_lab-{version}.dist-info"
    members: dict[str, bytes] = {}
    package_dir = REPO_ROOT / "model_protocol_lab"
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in (".py", ".typed"):
            continue
        arcname = path.relative_to(REPO_ROOT).as_posix()
        if arcname in omit:
            continue
        members[arcname] = path.read_bytes()
    members[f"{dist_info}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: model-protocol-lab\nVersion: {version}\n"
    ).encode()
    members[f"{dist_info}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: test_w5_artifact_smoke\n"
        b"Root-Is-Purelib: true\nTag: py3-none-any\n"
    )
    members[f"{dist_info}/entry_points.txt"] = (
        b"[console_scripts]\n"
        b"model-protocol-lab = model_protocol_lab.cli:main\n"
        b"model-protocol-lab-mcp = model_protocol_lab.mcp_server:serve\n"
    )
    record_lines = [_record_row(name, data) for name, data in sorted(members.items())]
    record_lines.append(f"{dist_info}/RECORD,,")
    members[f"{dist_info}/RECORD"] = ("\n".join(record_lines) + "\n").encode()
    wheel = dest_dir / f"model_protocol_lab-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)
    return wheel


def _run_main(wheel: Path, work: Path) -> int:
    """Run the smoke tool in-process, swallowing its report output."""
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return artifact_smoke.main(
            ["--wheel", str(wheel), "--source-tree", str(REPO_ROOT), "--work", str(work)]
        )


class OriginAssertionTests(unittest.TestCase):
    """The origin and version assertions bite on their own."""

    def test_origin_accepts_files_under_purelib(self) -> None:
        """Test origin accepts files under purelib."""
        purelib = "/venv/lib/python3.12/site-packages"
        files = {
            "package": f"{purelib}/model_protocol_lab/__init__.py",
            "cli": f"{purelib}/model_protocol_lab/cli.py",
            "mcp_server": f"{purelib}/model_protocol_lab/mcp_server.py",
        }
        artifact_smoke.check_origin(files, purelib)

    def test_origin_rejects_a_checkout_file_and_names_it(self) -> None:
        """Test origin rejects a checkout file and names it."""
        purelib = "/venv/lib/python3.12/site-packages"
        files = {
            "package": "/checkout/model_protocol_lab/__init__.py",
            "cli": f"{purelib}/model_protocol_lab/cli.py",
            "mcp_server": f"{purelib}/model_protocol_lab/mcp_server.py",
        }
        with self.assertRaises(artifact_smoke.SmokeFailure) as caught:
            artifact_smoke.check_origin(files, purelib)
        self.assertIn("package", str(caught.exception))
        self.assertIn("/checkout/model_protocol_lab/__init__.py", str(caught.exception))

    def test_version_equality(self) -> None:
        """Test version equality."""
        artifact_smoke.check_version(PACKAGE_VERSION, PACKAGE_VERSION)
        with self.assertRaises(artifact_smoke.SmokeFailure):
            artifact_smoke.check_version(PACKAGE_VERSION, "0.0.0")


class SmokeEnvironmentTests(unittest.TestCase):
    """The environment handed to smoke subprocesses is scrubbed."""

    def test_pythonpath_cannot_leak_into_the_smoke_env(self) -> None:
        """Test pythonpath cannot leak into the smoke env."""
        poisoned = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONHOME": "/nonexistent",
            "PYTHONSTARTUP": "/nonexistent/startup.py",
        }
        with mock.patch.dict(os.environ, poisoned):
            env = artifact_smoke.smoke_env()
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("PYTHONSTARTUP", env)
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")


class WheelMetadataTests(unittest.TestCase):
    """The expected version comes from the wheel itself."""

    def test_wheel_version_reads_dist_info_metadata(self) -> None:
        """Test wheel version reads dist info metadata."""
        with TemporaryDirectory() as tmp:
            wheel = build_test_wheel(Path(tmp))
            self.assertEqual(artifact_smoke.wheel_version(wheel), PACKAGE_VERSION)

    def test_wheel_version_rejects_a_wheel_without_metadata(self) -> None:
        """Test wheel version rejects a wheel without metadata."""
        with TemporaryDirectory() as tmp:
            bogus = Path(tmp) / "bogus-0.1-py3-none-any.whl"
            with zipfile.ZipFile(bogus, "w") as archive:
                archive.writestr("README.txt", "not a wheel")
            with self.assertRaises(artifact_smoke.SmokeFailure):
                artifact_smoke.wheel_version(bogus)


class ProbeSabotageTests(unittest.TestCase):
    """Readiness §4 arms A and B, reproduced against the probe."""

    def _installed_python(self, tmp: Path, *, install: bool) -> Path:
        """A venv, optionally with the genuine test wheel installed."""
        venv = tmp / "venv"
        python = artifact_smoke.create_venv(venv)
        if install:
            wheel = build_test_wheel(tmp)
            artifact_smoke.install_wheel(python, wheel, artifact_smoke.smoke_env())
        return python

    def test_arm_a_nothing_installed_checkout_on_pythonpath(self) -> None:
        """Test arm A nothing installed checkout on pythonpath."""
        with TemporaryDirectory() as tmp:
            python = self._installed_python(Path(tmp), install=False)
            env = artifact_smoke.smoke_env()
            env["PYTHONPATH"] = str(REPO_ROOT)
            probe = artifact_smoke.probe_installed(python, cwd=Path(tmp), env=env)
            with self.assertRaises(artifact_smoke.SmokeFailure):
                artifact_smoke.check_origin(probe["files"], probe["purelib"])

    def test_arm_b_wheel_installed_checkout_on_pythonpath(self) -> None:
        """Test arm B wheel installed checkout on pythonpath."""
        with TemporaryDirectory() as tmp:
            python = self._installed_python(Path(tmp), install=True)
            env = artifact_smoke.smoke_env()
            env["PYTHONPATH"] = str(REPO_ROOT)
            probe = artifact_smoke.probe_installed(python, cwd=Path(tmp), env=env)
            with self.assertRaises(artifact_smoke.SmokeFailure):
                artifact_smoke.check_origin(probe["files"], probe["purelib"])

    def test_genuine_probe_resolves_under_purelib(self) -> None:
        """Test genuine probe resolves under purelib."""
        with TemporaryDirectory() as tmp:
            python = self._installed_python(Path(tmp), install=True)
            probe = artifact_smoke.probe_installed(
                python, cwd=Path(tmp), env=artifact_smoke.smoke_env()
            )
            artifact_smoke.check_origin(probe["files"], probe["purelib"])
            artifact_smoke.check_version(probe["version"], PACKAGE_VERSION)
            self.assertIn("catalog", probe["commands"])
            self.assertEqual(len(probe["commands"]), 12)


class ArtifactSmokeEndToEndTests(unittest.TestCase):
    """The full tool: genuine passes, sabotage and damage fail."""

    def test_genuine_wheel_passes(self) -> None:
        """Test genuine wheel passes."""
        with TemporaryDirectory() as tmp:
            wheel = build_test_wheel(Path(tmp))
            self.assertEqual(_run_main(wheel, Path(tmp) / "work"), 0)

    def test_pythonpath_sabotage_cannot_reach_the_smoke(self) -> None:
        """Test pythonpath sabotage cannot reach the smoke."""
        with TemporaryDirectory() as tmp:
            wheel = build_test_wheel(Path(tmp))
            with mock.patch.dict(os.environ, {"PYTHONPATH": str(REPO_ROOT)}):
                self.assertEqual(_run_main(wheel, Path(tmp) / "work"), 0)

    def test_gutted_wheel_fails(self) -> None:
        """Test gutted wheel fails."""
        with TemporaryDirectory() as tmp:
            wheel = build_test_wheel(
                Path(tmp), omit=frozenset({"model_protocol_lab/mcp_server.py"})
            )
            self.assertEqual(_run_main(wheel, Path(tmp) / "work"), 1)

    def test_version_mismatch_wheel_fails(self) -> None:
        """Test version mismatch wheel fails."""
        with TemporaryDirectory() as tmp:
            wheel = build_test_wheel(Path(tmp), metadata_version="0.0.0")
            self.assertEqual(_run_main(wheel, Path(tmp) / "work"), 1)

    def test_missing_wheel_fails(self) -> None:
        """Test missing wheel fails."""
        with TemporaryDirectory() as tmp:
            self.assertEqual(_run_main(Path(tmp) / "absent.whl", Path(tmp) / "work"), 1)


if __name__ == "__main__":
    unittest.main()
