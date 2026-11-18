"""W5-5: the artifact manifest + SBOM bind the readiness receipt to the artifacts.

The release artifacts are identified by their SHA-256 digests (PEP 740
and build-provenance attestations bind those digests). The manifest
(dist/artifact-manifest.json, emitted by the build job and attested
with the artifacts) binds the *evidence* to the same digests: the
double-build record, the archive census, the origin smoke and the
sdist-scope results, each naming the artifact hash it ran against, so
an independent verifier given only the artifacts + manifest + SBOM can
reproduce every claim or name the one that fails.

These tests pin that contract with synthetic artifacts and a canned
PyPI fetcher (no network): the genuine set verifies; a wrong-version
manifest, an altered artifact, a swapped manifest, a tampered SBOM
and a member-level tamper with a re-written file hash are each
rejected by the named claim. The SBOM pins: the dev closure is
exactly pylock.toml's package set; hf licenses resolve through the
documented PyPI cascade (license_expression, then the license field,
then unambiguous OSI classifiers); a license the primary source does
not state unambiguously is NOASSERTION in the SBOM and named in the
manifest — never silently dropped and never guessed.
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
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import artifact_manifest as manifest  # noqa: E402

_VERSION = "0.2.0"
_DIST_INFO = f"model_protocol_lab-{_VERSION}.dist-info"
_DIST_FILES = (
    "METADATA",
    "WHEEL",
    "RECORD",
    "entry_points.txt",
    "top_level.txt",
    "licenses/LICENSE",
)
_ROOT_FILES = (
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
    "setup.cfg",
)
_EGG_INFO_FILES = (
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "entry_points.txt",
    "requires.txt",
    "top_level.txt",
)
_METADATA = f"Metadata-Version: 2.4\nName: model-protocol-lab\nVersion: {_VERSION}\n"
_COMMIT = "a" * 40
_EPOCH = 1794481907

# Canned PyPI metadata, shaped like the responses measured on
# 2026-11-17 for the real packages (see the W5 work log): torch's
# composite license_expression, transformers' free-text license field,
# safetensors'/tokenizers' classifier-only Apache, jinja2's ambiguous
# "BSD License" classifier.
_PYPI: dict[str, dict] = {
    "alpha": {
        "info": {
            "name": "alpha",
            "version": "1.0",
            "license_expression": "Apache-2.0 AND MIT",
            "license": None,
            "classifiers": [],
            "requires_dist": ["delta>=1", "gamma>=1; extra == 'docs'"],
        }
    },
    "beta": {
        "info": {
            "name": "beta",
            "version": "2.0",
            "license_expression": None,
            "license": "Apache 2.0 License",
            "classifiers": [],
            "requires_dist": ["zeta>=1", "eta>=1"],
        }
    },
    "delta": {
        "info": {
            "name": "delta",
            "version": "3.0",
            "license_expression": None,
            "license": None,
            "classifiers": ["License :: OSI Approved :: Apache Software License"],
            "requires_dist": ["epsilon>=1"],
        }
    },
    "epsilon": {
        "info": {
            "name": "epsilon",
            "version": "4.0",
            "license_expression": None,
            "license": None,
            "classifiers": ["License :: OSI Approved :: BSD License"],
            "requires_dist": [],
        }
    },
    "gamma": {
        "info": {
            "name": "gamma",
            "version": "5.0",
            "license_expression": "MIT",
            "license": None,
            "classifiers": [],
            "requires_dist": [],
        }
    },
    "zeta": {
        "info": {
            "name": "zeta",
            "version": "6.0",
            "license_expression": "BSD-3-Clause AND 0BSD AND MIT",
            "license": None,
            "classifiers": [],
            "requires_dist": [],
        }
    },
    "eta": {
        "info": {
            "name": "eta",
            "version": "7.0",
            "license_expression": None,
            "license": "ISC License",
            "classifiers": [],
            "requires_dist": [],
        }
    },
    "devfake1": {
        "info": {
            "name": "devfake1",
            "version": "0.1",
            "license_expression": "MIT",
            "license": None,
            "classifiers": [],
            "requires_dist": [],
        }
    },
    "devfake2": {
        "info": {
            "name": "devfake2",
            "version": "0.2",
            "license_expression": None,
            "license": "MIT",
            "classifiers": [],
            "requires_dist": [],
        }
    },
}


def _fetch(url: str) -> dict:
    """Canned PyPI JSON: /pypi/<name>[/<version>]/json from _PYPI."""
    name = url.split("/pypi/", 1)[1].split("/", 1)[0].lower()
    if name not in _PYPI:
        raise AssertionError(f"unexpected PyPI fetch: {url}")
    return _PYPI[name]


def _fake_source(root: Path) -> Path:
    """Minimal source tree the census accepts, plus manifest inputs."""
    package = root / "model_protocol_lab"
    package.mkdir(parents=True)
    for name in ("a.py", "b.py"):
        (package / name).write_text(f"# {name}\n", encoding="utf-8")
    (package / "py.typed").write_text("", encoding="utf-8")
    (package / "pack.py").write_text(
        "MAX_BUNDLE_ZIP_BYTES = 288 * 1024 * 1024\n"
        "MAX_BUNDLE_ENTRIES = 1024\n"
        "MAX_BUNDLE_FILE_BYTES = 64 * 1024 * 1024\n"
        "MAX_BUNDLE_TOTAL_BYTES = 256 * 1024 * 1024\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "model-protocol-lab"\n'
        f'version = "{_VERSION}"\n'
        "[project.optional-dependencies]\n"
        'hf = ["alpha>=1", "beta>=2"]\n',
        encoding="utf-8",
    )
    (root / "pylock.toml").write_text(
        'lock-version = "1.0"\n'
        "[[packages]]\n"
        'name = "devfake1"\n'
        'version = "0.1"\n'
        "[[packages]]\n"
        'name = "devfake2"\n'
        'version = "0.2"\n',
        encoding="utf-8",
    )
    requirements = root / "tools" / "requirements"
    requirements.mkdir(parents=True)
    (requirements / "pip.txt").write_text("pip==26.3 \\\n    --hash=sha256:aa\n", encoding="utf-8")
    (requirements / "pip-setuptools.txt").write_text(
        "setuptools==84.2.1 \\\n    --hash=sha256:bb\n", encoding="utf-8"
    )
    (requirements / "build.txt").write_text(
        "build==1.3.0 \\\n    --hash=sha256:cc\n", encoding="utf-8"
    )
    (root / "tools" / "release_build.py").write_text("# builder\n", encoding="utf-8")
    tests = root / "tests"
    (tests / "data").mkdir(parents=True)
    (tests / "test_a.py").write_text("# test\n", encoding="utf-8")
    (tests / "data" / "UnicodeData.txt").write_text("data\n", encoding="utf-8")
    (tests / "CHECKOUT_ONLY").write_text("none\n", encoding="utf-8")
    (tests / "test_layers.json").write_text("{}\n", encoding="utf-8")
    return root


def _wheel_members() -> list[str]:
    """Return the wheel members."""
    members = [f"model_protocol_lab/{name}" for name in ("a.py", "b.py", "pack.py", "py.typed")]
    members += [f"{_DIST_INFO}/{name}" for name in _DIST_FILES]
    return members


def _write_wheel(path: Path, content_overrides: dict[str, str] | None = None) -> None:
    """Write the wheel."""
    overrides = content_overrides or {}
    with zipfile.ZipFile(path, "w") as archive:
        for member in _wheel_members():
            if member.endswith("/METADATA"):
                archive.writestr(member, overrides.get(member, _METADATA))
            else:
                archive.writestr(member, overrides.get(member, f"content of {member}\n"))


def _write_sdist(path: Path) -> None:
    """Write the sdist."""
    top = f"model_protocol_lab-{_VERSION}"
    members = [f"{top}/{name}" for name in _ROOT_FILES]
    members += [
        f"{top}/model_protocol_lab/{name}" for name in ("a.py", "b.py", "pack.py", "py.typed")
    ]
    members += [f"{top}/model_protocol_lab.egg-info/{name}" for name in _EGG_INFO_FILES]
    members += [
        f"{top}/tests/test_a.py",
        f"{top}/tests/data/UnicodeData.txt",
        f"{top}/tests/CHECKOUT_ONLY",
        f"{top}/tests/test_layers.json",
    ]
    with tarfile.open(path, "w:gz") as archive:
        for member in members:
            data = (
                _METADATA.encode()
                if member.endswith("/PKG-INFO")
                else f"content of {member}\n".encode()
            )
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _sha256(path: Path) -> str:
    """Return the sha256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_json(path: Path, tool: str, artifact: Path) -> Path:
    """Write a result json."""
    path.write_text(
        json.dumps(
            {
                "tool": tool,
                "artifact": {"name": artifact.name, "sha256": _sha256(artifact)},
                "report": [f"{tool}: synthetic report line"],
                "result": "passed",
            }
        ),
        encoding="utf-8",
    )
    return path


class _Fixture(unittest.TestCase):
    """A fake source tree + dist pair with matching result records."""

    def setUp(self) -> None:
        """Set up the fixture."""
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.source = _fake_source(root / "tree")
        self.dist = root / "dist"
        self.dist.mkdir()
        self.wheel = self.dist / f"model_protocol_lab-{_VERSION}-py3-none-any.whl"
        self.sdist = self.dist / f"model_protocol_lab-{_VERSION}.tar.gz"
        _write_wheel(self.wheel)
        _write_sdist(self.sdist)
        self.build_record = root / "build-record.json"
        self.build_record.write_text(
            json.dumps(
                {
                    "tool": "release_build",
                    "version": _VERSION,
                    "epoch": _EPOCH,
                    "sdist": {
                        "name": self.sdist.name,
                        "sha256_run1": _sha256(self.sdist),
                        "sha256_run2": _sha256(self.sdist),
                    },
                    "wheel": {
                        "name": self.wheel.name,
                        "sha256_run1": _sha256(self.wheel),
                        "sha256_run2": _sha256(self.wheel),
                    },
                    "verdict": "identical",
                }
            ),
            encoding="utf-8",
        )
        self.smoke_result = _result_json(root / "smoke.json", "artifact_smoke", self.wheel)
        self.scope_result = _result_json(root / "scope.json", "sdist_scope", self.sdist)

    def emit(self, **overrides: object) -> Path:
        """Emit."""
        kwargs: dict[str, object] = {
            "dist": self.dist,
            "source": self.source,
            "commit": _COMMIT,
            "epoch": _EPOCH,
            "build_record_path": self.build_record,
            "smoke_result_path": self.smoke_result,
            "scope_result_path": self.scope_result,
            "fetch": _fetch,
        }
        kwargs.update(overrides)
        return manifest.emit_manifest(**kwargs)  # type: ignore[arg-type]

    def verify(self, **overrides: object) -> list[tuple[str, bool, str]]:
        """Verify."""
        kwargs: dict[str, object] = {
            "dist": self.dist,
            "manifest_path": self.dist / "artifact-manifest.json",
            "source": self.source,
        }
        kwargs.update(overrides)
        return manifest.verify_manifest(**kwargs)  # type: ignore[arg-type]

    @staticmethod
    def failed_claims(claims: list[tuple[str, bool, str]]) -> list[str]:
        """Return the failed claims."""
        return [name for name, ok, _detail in claims if not ok]


class EmitTests(_Fixture):
    """The emitted manifest carries every block, bound to the artifacts."""

    def test_emit_writes_manifest_and_sbom(self) -> None:
        """Test emit writes manifest and sbom."""
        path = self.emit()
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema"], "artifact-manifest-1")
        self.assertEqual(data["package"], {"name": "model-protocol-lab", "version": _VERSION})
        by_name = {entry["name"]: entry for entry in data["artifacts"]}
        self.assertEqual(set(by_name), {self.wheel.name, self.sdist.name})
        for artifact in (self.wheel, self.sdist):
            entry = by_name[artifact.name]
            self.assertEqual(entry["sha256"], _sha256(artifact))
            self.assertEqual(entry["size"], artifact.stat().st_size)
        self.assertEqual(data["source"]["commit"], _COMMIT)
        self.assertEqual(data["source"]["source_date_epoch"], _EPOCH)
        self.assertTrue(data["source"]["tree_digest"].startswith("sha256:"))
        self.assertEqual(data["reproducibility"]["verdict"], "identical")
        self.assertEqual(data["verification"]["census"]["result"], "passed")
        self.assertEqual(data["verification"]["census"]["unexpected"], 0)
        self.assertEqual(
            data["verification"]["pack_ceilings"]["MAX_BUNDLE_ENTRIES"],
            1024,
        )
        self.assertEqual(
            data["verification"]["origin_smoke"]["artifact"]["sha256"],
            _sha256(self.wheel),
        )
        self.assertEqual(
            data["verification"]["sdist_scope"]["artifact"]["sha256"],
            _sha256(self.sdist),
        )
        sbom_path = self.dist / data["sbom"]["file"]
        self.assertTrue(sbom_path.is_file())
        self.assertEqual(data["sbom"]["sha256"], _sha256(sbom_path))
        self.assertEqual(data["sbom"]["unresolved_licenses"], ["epsilon"])

    def test_emit_is_deterministic(self) -> None:
        """Test emit is deterministic."""
        first = self.emit().read_bytes()
        second = self.emit().read_bytes()
        self.assertEqual(first, second)

    def test_emit_refuses_smoke_bound_to_another_wheel(self) -> None:
        """Test emit refuses smoke bound to another wheel."""
        self.smoke_result.write_text(
            json.dumps(
                {
                    "tool": "artifact_smoke",
                    "artifact": {"name": self.wheel.name, "sha256": "0" * 64},
                    "report": [],
                    "result": "passed",
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(manifest.ManifestFailure) as caught:
            self.emit()
        self.assertIn("smoke", str(caught.exception))

    def test_emit_refuses_missing_scope_result(self) -> None:
        """Test emit refuses missing scope result."""
        self.scope_result.unlink()
        with self.assertRaises(manifest.ManifestFailure):
            self.emit()

    def test_emit_refuses_mismatch_build_record(self) -> None:
        """Test emit refuses mismatch build record."""
        record = json.loads(self.build_record.read_text(encoding="utf-8"))
        record["verdict"] = "mismatch"
        self.build_record.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(manifest.ManifestFailure) as caught:
            self.emit()
        self.assertIn("reproducibility", str(caught.exception))

    def test_emit_refuses_version_disagreement(self) -> None:
        """Test emit refuses version disagreement."""
        bad_metadata = "Metadata-Version: 2.4\nName: model-protocol-lab\nVersion: 9.9.9\n"
        _write_wheel(self.wheel, {f"{_DIST_INFO}/METADATA": bad_metadata})
        with self.assertRaises(manifest.ManifestFailure) as caught:
            self.emit()
        self.assertIn("version", str(caught.exception))


class VerifyTests(_Fixture):
    """The verifier accepts the genuine set and rejects each control."""

    def test_verify_accepts_genuine(self) -> None:
        """Test verify accepts genuine."""
        self.emit()
        claims = self.verify()
        self.assertEqual(self.failed_claims(claims), [], claims)

    def test_verify_rejects_wrong_version(self) -> None:
        """Test verify rejects wrong version."""
        path = self.emit()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["package"]["version"] = "9.9.9"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertIn("package-version", self.failed_claims(self.verify()))

    def test_verify_rejects_altered_artifact(self) -> None:
        """Test verify rejects altered artifact."""
        self.emit()
        with self.wheel.open("ab") as handle:
            handle.write(b"tamper")
        failed = self.failed_claims(self.verify())
        self.assertIn("artifact-hashes", failed)

    def test_verify_rejects_swapped_manifest(self) -> None:
        """Test verify rejects swapped manifest."""
        self.emit()
        other = Path(self._tmp.name) / "other"
        other.mkdir()
        other_dist = other / "dist"
        other_dist.mkdir()
        other_wheel = other_dist / self.wheel.name
        _write_wheel(other_wheel, {"model_protocol_lab/a.py": "# different content\n"})
        other_sdist = other_dist / self.sdist.name
        _write_sdist(other_sdist)
        claims = manifest.verify_manifest(
            dist=other_dist,
            manifest_path=self.dist / "artifact-manifest.json",
            source=self.source,
        )
        self.assertIn("artifact-hashes", self.failed_claims(claims))

    def test_verify_rejects_sbom_tamper(self) -> None:
        """Test verify rejects sbom tamper."""
        self.emit()
        sbom = self.dist / "sbom.cdx.json"
        sbom.write_text(sbom.read_text(encoding="utf-8") + " ", encoding="utf-8")
        self.assertIn("sbom-digest", self.failed_claims(self.verify()))

    def test_verify_rejects_member_tamper_with_rewritten_hash(self) -> None:
        """Test verify rejects member tamper with rewritten hash."""
        path = self.emit()
        _write_wheel(self.wheel, {"model_protocol_lab/a.py": "# tampered member\n"})
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data["artifacts"]:
            if entry["name"] == self.wheel.name:
                entry["sha256"] = _sha256(self.wheel)
                entry["size"] = self.wheel.stat().st_size
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertIn("artifact-members", self.failed_claims(self.verify()))


class SbomTests(_Fixture):
    """The SBOM's closures and license cascade are pinned."""

    def _sbom(self) -> dict:
        """Return the sbom."""
        sbom, _unresolved = manifest.build_sbom(self.source, _VERSION, _fetch)
        return sbom

    def test_sbom_shape_and_root(self) -> None:
        """Test sbom shape and root."""
        sbom = self._sbom()
        self.assertEqual(sbom["bomFormat"], "CycloneDX")
        self.assertEqual(sbom["specVersion"], "1.5")
        root = sbom["metadata"]["component"]
        self.assertEqual(root["name"], "model-protocol-lab")
        self.assertEqual(root["licenses"], [{"license": {"id": "MIT"}}])

    def test_dev_closure_is_exactly_pylock(self) -> None:
        """Test dev closure is exactly pylock."""
        sbom = self._sbom()
        dev = {
            component["name"] for component in sbom["components"] if _closure(component) == "dev"
        }
        self.assertEqual(dev, {"devfake1", "devfake2"})

    def test_hf_cascade(self) -> None:
        """Test hf cascade."""
        sbom = self._sbom()
        by_name = {component["name"]: component for component in sbom["components"]}
        self.assertEqual(
            by_name["alpha"]["licenses"],
            [{"expression": "Apache-2.0 AND MIT"}],
        )
        self.assertEqual(by_name["beta"]["licenses"], [{"license": {"id": "Apache-2.0"}}])
        self.assertEqual(by_name["delta"]["licenses"], [{"license": {"id": "Apache-2.0"}}])
        # Measured PyPI forms (2026-11-17): numpy's composite carries
        # 0BSD; shellingham's free-text field is "ISC License".
        self.assertEqual(
            by_name["zeta"]["licenses"],
            [{"expression": "BSD-3-Clause AND 0BSD AND MIT"}],
        )
        self.assertEqual(by_name["eta"]["licenses"], [{"license": {"id": "ISC"}}])
        # gamma is extra-gated in alpha's requires_dist: not in the closure.
        self.assertNotIn("gamma", by_name)

    def test_unresolved_license_is_noassertion_and_named(self) -> None:
        """Test unresolved license is noassertion and named."""
        sbom, unresolved = manifest.build_sbom(self.source, _VERSION, _fetch)
        by_name = {component["name"]: component for component in sbom["components"]}
        self.assertEqual(
            by_name["epsilon"]["licenses"],
            [{"license": {"name": "NOASSERTION"}}],
        )
        self.assertEqual(unresolved, ["epsilon"])


def _closure(component: dict) -> str:
    """Return the closure."""
    for prop in component.get("properties", []):
        if prop["name"] == "mpl:closure":
            return str(prop["value"])
    return ""


class VerifyTotalityTests(_Fixture):
    """BH13-4: malformed manifests fail named claims, never tracebacks.

    The verifier's user story is checking a downloaded release's
    receipt, so type-confused JSON is the expected hostile case:
    every malformed shape must surface as a failed named claim.
    """

    def test_non_object_manifest_fails_readable_claim(self) -> None:
        """Test non object manifest fails readable claim."""
        path = self.dist / "artifact-manifest.json"
        path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
        claims = self.verify()
        self.assertEqual(self.failed_claims(claims), ["manifest-readable"])

    def test_string_artifact_entries_fail_present_claim(self) -> None:
        """Test string artifact entries fail present claim."""
        path = self.emit()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["artifacts"] = ["wheel-name", "sdist-name"]
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertIn("artifacts-present", self.failed_claims(self.verify()))

    def test_int_artifact_name_fails_present_claim(self) -> None:
        """Test int artifact name fails present claim."""
        path = self.emit()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["artifacts"][0]["name"] = 7
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertIn("artifacts-present", self.failed_claims(self.verify()))

    def test_string_package_fails_version_claims(self) -> None:
        """Test string package fails version claims."""
        path = self.emit()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["package"] = "model-protocol-lab"
        path.write_text(json.dumps(data), encoding="utf-8")
        failed = self.failed_claims(self.verify())
        self.assertIn("package-version", failed)
        self.assertIn("sbom-package", failed)

    def test_non_utf8_wheel_metadata_fails_version_claim(self) -> None:
        """Test non utf8 wheel metadata fails version claim."""
        self.emit()
        with zipfile.ZipFile(self.wheel, "w") as archive:
            for member in _wheel_members():
                payload: bytes | str = (
                    b"\xff\xfe not utf-8"
                    if member.endswith("/METADATA")
                    else f"content of {member}\n"
                )
                archive.writestr(member, payload)
        self.assertIn("package-version", self.failed_claims(self.verify()))


class VerificationBindingTests(_Fixture):
    """BH13-5: verify re-validates the embedded results it displays.

    Emit enforces each embedded result's tool, pass status, artifact
    name and hash; verify must enforce the same, so a manifest cannot
    carry a failed (or foreign) verification record under a
    fully-verified receipt.
    """

    def test_embedded_failed_result_fails_bindings(self) -> None:
        """Test embedded failed result fails bindings."""
        path = self.emit()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["verification"]["origin_smoke"]["result"] = "failed"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertIn("verification-bindings", self.failed_claims(self.verify()))

    def test_embedded_foreign_tool_fails_bindings(self) -> None:
        """Test embedded foreign tool fails bindings."""
        path = self.emit()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["verification"]["sdist_scope"]["tool"] = "something_else"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertIn("verification-bindings", self.failed_claims(self.verify()))

    def test_embedded_wrong_artifact_name_fails_bindings(self) -> None:
        """Test embedded wrong artifact name fails bindings."""
        path = self.emit()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["verification"]["origin_smoke"]["artifact"]["name"] = "other.whl"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertIn("verification-bindings", self.failed_claims(self.verify()))


class MemberCeilingTests(unittest.TestCase):
    """BH13-6: _member_digests reads are ceiling-bound, refusing by name.

    A crafted archive member must not amplify into unbounded memory:
    the reader streams under ceilings mirroring pack.py's RTI-02
    producer ceilings and refuses by name over a ceiling.
    """

    def test_member_ceiling_names_the_member(self) -> None:
        """Test member ceiling names the member."""
        with TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "x.whl"
            with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("model_protocol_lab/big.py", b"\0" * 4096)
            with (
                mock.patch.object(manifest, "_MAX_MEMBER_BYTES", 1024),
                self.assertRaises(manifest.ManifestFailure) as caught,
            ):
                manifest._member_digests(wheel)
        self.assertIn("model_protocol_lab/big.py", str(caught.exception))

    def test_aggregate_ceiling_names_the_archive(self) -> None:
        """Test aggregate ceiling names the archive."""
        with TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "x.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("model_protocol_lab/a.py", b"a" * 800)
                archive.writestr("model_protocol_lab/b.py", b"b" * 800)
            with (
                mock.patch.object(manifest, "_MAX_MEMBER_BYTES", 4096),
                mock.patch.object(manifest, "_MAX_TOTAL_MEMBER_BYTES", 1024),
                self.assertRaises(manifest.ManifestFailure) as caught,
            ):
                manifest._member_digests(wheel)
        self.assertIn("x.whl", str(caught.exception))

    def test_count_ceiling_names_the_archive(self) -> None:
        """Test count ceiling names the archive."""
        with TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "x.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                for index in range(5):
                    archive.writestr(f"model_protocol_lab/m{index}.py", b"x")
            with (
                mock.patch.object(manifest, "_MAX_MEMBERS", 3),
                self.assertRaises(manifest.ManifestFailure) as caught,
            ):
                manifest._member_digests(wheel)
        self.assertIn("x.whl", str(caught.exception))

    def test_sdist_member_ceiling_names_the_member(self) -> None:
        """Test sdist member ceiling names the member."""
        with TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "x.tar.gz"
            data = b"\0" * 4096
            with tarfile.open(sdist, "w:gz") as archive:
                info = tarfile.TarInfo("model_protocol_lab-0.2.0/model_protocol_lab/big.py")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            with (
                mock.patch.object(manifest, "_MAX_MEMBER_BYTES", 1024),
                self.assertRaises(manifest.ManifestFailure) as caught,
            ):
                manifest._member_digests(sdist)
        self.assertIn("model_protocol_lab/big.py", str(caught.exception))


class CeilingParityTests(_Fixture):
    """BH13-6: the manifest reader's ceilings equal pack.py's RTI-02 values."""

    def test_ceilings_match_pack(self) -> None:
        """Test ceilings match pack."""
        ceilings = manifest._pack_ceilings(self.source)
        self.assertEqual(manifest._MAX_MEMBER_BYTES, ceilings["MAX_BUNDLE_FILE_BYTES"])
        self.assertEqual(manifest._MAX_TOTAL_MEMBER_BYTES, ceilings["MAX_BUNDLE_TOTAL_BYTES"])
        self.assertEqual(manifest._MAX_MEMBERS, ceilings["MAX_BUNDLE_ENTRIES"])

    def test_verify_fails_members_claim_over_ceiling(self) -> None:
        """Test verify fails members claim over ceiling."""
        self.emit()
        with mock.patch.object(manifest, "_MAX_MEMBER_BYTES", 8):
            failed = self.failed_claims(self.verify())
        self.assertIn("artifact-members", failed)


class SpdxExpressionTests(unittest.TestCase):
    """BH13-7: the SPDX validator implements the expression grammar.

    Dangling, doubled or bare operators are not expressions; a
    malformed license_expression falls through the cascade to
    classifiers / NOASSERTION instead of landing in the SBOM as an
    invalid license id.
    """

    def test_valid_expressions(self) -> None:
        """Test valid expressions."""
        for text in (
            "MIT",
            "MIT+",
            "MIT OR Apache-2.0",
            "Apache-2.0 AND MIT",
            "(MIT OR Apache-2.0) AND BSD-3-Clause",
            "BSD-3-Clause AND 0BSD AND MIT",
            "Apache-2.0 WITH LLVM-exception",
            "GPL-2.0-only WITH Classpath-exception-2.0",
        ):
            self.assertTrue(manifest._is_spdx_expression(text), text)

    def test_invalid_expressions(self) -> None:
        """Test invalid expressions."""
        for text in (
            "",
            "OR",
            "AND",
            "WITH MIT",
            "MIT OR",
            "OR MIT",
            "MIT AND",
            "MIT AND AND MIT",
            "MIT OR OR MIT",
            "MIT WITH",
            "MIT WITH AND",
            "MIT OR (Apache-2.0",
            "MIT)",
            "()",
            "FooBar",
        ):
            self.assertFalse(manifest._is_spdx_expression(text), text)

    def test_bare_operator_resolves_unresolved(self) -> None:
        """Test bare operator resolves unresolved."""
        info = {"license_expression": "OR", "license": None, "classifiers": []}
        self.assertEqual(manifest._resolve_license(info), (None, "unresolved"))

    def test_bare_operator_falls_through_to_classifier(self) -> None:
        """Test bare operator falls through to classifier."""
        info = {
            "license_expression": "OR",
            "license": None,
            "classifiers": ["License :: OSI Approved :: MIT License"],
        }
        self.assertEqual(manifest._resolve_license(info), ("MIT", "pypi:classifier"))


class WorkflowTests(unittest.TestCase):
    """The release workflow emits the manifest and records the recipe."""

    def test_release_yml_wires_the_manifest(self) -> None:
        """Test release yml wires the manifest."""
        text = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        self.assertIn("tools/artifact_manifest.py emit", text)
        self.assertIn("--build-record", text)
        self.assertIn("--result-json", text)
        self.assertIn("artifact-manifest.json", text)
        self.assertIn("gh attestation verify", text)


if __name__ == "__main__":
    unittest.main()
