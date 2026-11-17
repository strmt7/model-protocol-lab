"""W5-6: the from-sdist rebuild verifies against the manifest.

Reproducibility has two tiers, and the manifest's per-member digests
keep them distinguishable (readiness §3, measured): under the
declared conditions — tarfile ``filter="data"`` extraction, mtimes
normalized to SOURCE_DATE_EPOCH, an epoch-stamped build — the wheel
rebuilt from the published sdist is byte-identical to the direct
wheel. A naive rebuild differs only in zip entry timestamps: every
member's content digest still matches, so the check names the delta
metadata-only (and which members were restamped) instead of
reporting an opaque hash mismatch. A member whose content actually
changed is named as a content mismatch.

The build step is injected here (the real ``python -m build``
invocation is exercised against the real artifacts in the W5
acceptance, not in the unit suite); the extraction, the artifact
binding and the comparison logic are the real code.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import artifact_manifest as manifest  # noqa: E402

_VERSION = "0.2.0"
_DIST_INFO = f"model_protocol_lab-{_VERSION}.dist-info"
_METADATA = f"Metadata-Version: 2.4\nName: model-protocol-lab\nVersion: {_VERSION}\n"
_WHEEL_MEMBERS = {
    "model_protocol_lab/a.py": "# a\n",
    "model_protocol_lab/py.typed": "",
    f"{_DIST_INFO}/METADATA": _METADATA,
    f"{_DIST_INFO}/WHEEL": "Wheel-Version: 1.0\n",
    f"{_DIST_INFO}/RECORD": "",
}
_EPOCH = 1794481907
_STAMP = (2026, 11, 17, 12, 0, 0)
_EPOCH_STAMP = (1980, 1, 1, 0, 0, 0)


def _write_wheel(path: Path, members: dict[str, str], stamp: tuple[int, ...]) -> None:
    """Write the wheel."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            info = zipfile.ZipInfo(name, date_time=stamp)
            archive.writestr(info, content)


def _write_sdist(path: Path, extra: dict[str, str] | None = None) -> None:
    """Write the sdist."""
    top = f"model_protocol_lab-{_VERSION}"
    members = {f"{top}/PKG-INFO": "Name: model-protocol-lab\nVersion: 0.2.0\n"}
    members.update(extra or {})
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _sha256(path: Path) -> str:
    """Return the sha256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _member_digests(members: dict[str, str]) -> dict[str, str]:
    """Return the member digests."""
    return {name: hashlib.sha256(content.encode()).hexdigest() for name, content in members.items()}


class RebuildCheckTests(unittest.TestCase):
    """The check accepts byte-identity and names every other outcome."""

    def setUp(self) -> None:
        """Set up the fixture."""
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dist = Path(self._tmp.name) / "dist"
        self.dist.mkdir()
        self.wheel = self.dist / f"model_protocol_lab-{_VERSION}-py3-none-any.whl"
        self.sdist = self.dist / f"model_protocol_lab-{_VERSION}.tar.gz"
        _write_wheel(self.wheel, _WHEEL_MEMBERS, _EPOCH_STAMP)
        _write_sdist(self.sdist)
        self.manifest_path = self.dist / "artifact-manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "schema": "artifact-manifest-1",
                    "artifacts": [
                        {
                            "name": self.wheel.name,
                            "sha256": _sha256(self.wheel),
                            "members": _member_digests(_WHEEL_MEMBERS),
                        },
                        {"name": self.sdist.name, "sha256": _sha256(self.sdist), "members": {}},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def _check(self, builder: object) -> list[str]:
        """Return the check."""
        return manifest.rebuild_check(
            dist=self.dist,
            manifest_path=self.manifest_path,
            epoch=_EPOCH,
            work=Path(self._tmp.name) / "work",
            builder=builder,  # type: ignore[arg-type]
        )

    @staticmethod
    def _copy_builder(source_wheel: Path) -> object:
        """Return the copy builder."""

        def build(_package_dir: Path, out_dir: Path, _epoch: int) -> Path:
            """Build."""
            target = out_dir / source_wheel.name
            target.write_bytes(source_wheel.read_bytes())
            return target

        return build

    def test_byte_identical_rebuild_passes(self) -> None:
        """Test byte identical rebuild passes."""
        lines = self._check(self._copy_builder(self.wheel))
        self.assertIn("byte-identical", lines[0])

    def test_metadata_only_delta_is_named_not_passed(self) -> None:
        """Test metadata only delta is named not passed."""
        restamped = Path(self._tmp.name) / "restamped.whl"
        _write_wheel(restamped, _WHEEL_MEMBERS, _STAMP)
        with self.assertRaises(manifest.ManifestFailure) as caught:
            self._check(self._copy_builder(restamped))
        message = str(caught.exception)
        self.assertIn("metadata-only", message)
        self.assertIn("METADATA", message)

    def test_content_mismatch_names_the_member(self) -> None:
        """Test content mismatch names the member."""
        tampered = dict(_WHEEL_MEMBERS)
        tampered["model_protocol_lab/a.py"] = "# tampered\n"
        target = Path(self._tmp.name) / "tampered.whl"
        _write_wheel(target, tampered, _EPOCH_STAMP)
        with self.assertRaises(manifest.ManifestFailure) as caught:
            self._check(self._copy_builder(target))
        self.assertIn("content mismatch", str(caught.exception))
        self.assertIn("model_protocol_lab/a.py", str(caught.exception))

    def test_unbound_sdist_is_refused(self) -> None:
        """Test unbound sdist is refused."""
        _write_sdist(self.sdist, {f"model_protocol_lab-{_VERSION}/extra.txt": "x\n"})
        with self.assertRaises(manifest.ManifestFailure) as caught:
            self._check(self._copy_builder(self.wheel))
        self.assertIn("does not match the manifest", str(caught.exception))

    def test_traversal_member_fails_closed(self) -> None:
        """Test traversal member fails closed."""
        _write_sdist(self.sdist, {"../evil.txt": "x\n"})
        # Rebind the manifest to the hostile sdist so extraction is reached.
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        data["artifacts"][1]["sha256"] = _sha256(self.sdist)
        self.manifest_path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(manifest.ManifestFailure):
            self._check(self._copy_builder(self.wheel))


class WorkflowTests(unittest.TestCase):
    """The release workflow runs the rebuild check in the build job."""

    def test_release_yml_runs_rebuild_check(self) -> None:
        """Test release yml runs rebuild check."""
        text = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        self.assertIn("tools/artifact_manifest.py rebuild-check", text)


if __name__ == "__main__":
    unittest.main()
