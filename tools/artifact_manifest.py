"""Artifact manifest + SBOM: the release readiness receipt, bound to the artifacts.

Who uses this on Monday morning: the person verifying a downloaded
release before installing it. The attestations (PEP 740 and build
provenance) bind the artifacts' SHA-256 digests; this manifest binds
the *evidence* to the same digests — the reproducible double build,
the archive census, the installed-origin smoke and the sdist-scope
run, each result naming the artifact hash it ran against — plus a
CycloneDX SBOM whose license inventory is enumerated from PyPI JSON
metadata at generation time. ``verify`` re-checks every claim from
the artifacts alone and names the claim that fails.

License resolution cascade (per package, from the PyPI JSON ``info``
of the release whose metadata was read):

1. ``license_expression``, when it parses as an SPDX expression over
   known identifiers;
2. the ``license`` field, when it is an SPDX expression or one of the
   measured free-text forms in ``_LICENSE_TEXT``;
3. an unambiguous OSI classifier (``_CLASSIFIER_LICENSES``); the
   generic "BSD License" classifier is deliberately NOT resolved —
   it does not say which BSD variant applies;
4. otherwise the license is ``NOASSERTION`` in the SBOM and the
   package is named in the manifest's ``sbom.unresolved_licenses`` —
   never silently dropped, never guessed.

The hf closure is range-resolved, so its inventory is a walk of
``requires_dist`` from PyPI JSON (extra-gated requirements excluded);
the recorded version of an hf component is the release whose metadata
was read, not a resolved environment. The dev closure is exactly
``pylock.toml``'s package set, read at its locked versions.

Usage:
    python tools/artifact_manifest.py emit --dist dist --source . \
        --commit <sha> --epoch <SOURCE_DATE_EPOCH> \
        --build-record build-record.json \
        --smoke-result smoke-result.json --scope-result scope-result.json
    python tools/artifact_manifest.py verify --dist dist [--source .]
    python tools/artifact_manifest.py rebuild-check --dist dist \
        --epoch <SOURCE_DATE_EPOCH>

``rebuild-check`` is the from-sdist rebuild under the declared
conditions (W5-6): the sdist is extracted with tarfile's ``data``
filter, the extracted tree's mtimes are normalized to the epoch,
and the wheel is rebuilt with ``SOURCE_DATE_EPOCH`` set. Under
those conditions the rebuilt wheel must be byte-identical to the
manifest's wheel. If the file hash differs but every member's
content digest matches the manifest, the delta is metadata-only
(zip entry timestamps) and the check says so, naming the members
whose timestamps differ — the distinction is checkable from the
manifest's per-member digests instead of being arguable.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

_SCHEMA = "artifact-manifest-1"
_SBOM_NAME = "sbom.cdx.json"
_REPOSITORY = "strmt7/model-protocol-lab"
_PYPI_BASE = "https://pypi.org"
_MAX_WALK = 512

_TREE_EXCLUDED = {
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "site",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".coverage",
}

_SPDX_IDS = {
    "0BSD",
    "Apache-2.0",
    "CNRI-Python",
    "MIT",
    "MIT-0",
    "BSD-1-Clause",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSL-1.0",
    "ISC",
    "Zlib",
    "MPL-2.0",
    "PSF-2.0",
    "LGPL-2.1-only",
    "LGPL-2.1-or-later",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "GPL-2.0-only",
    "GPL-2.0-or-later",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "AGPL-3.0-only",
    "AGPL-3.0-or-later",
    "Unlicense",
    "CC0-1.0",
    "Apache-1.1",
    "Python-2.0",
}

_LICENSE_TEXT = {
    "Apache 2.0 License": "Apache-2.0",
    "Apache License 2.0": "Apache-2.0",
    "Apache License, Version 2.0": "Apache-2.0",
    "MIT License": "MIT",
    "The MIT License": "MIT",
    "ISC License": "ISC",
    "BSD 3-Clause License": "BSD-3-Clause",
    "BSD-2-Clause License": "BSD-2-Clause",
    "Mozilla Public License 2.0": "MPL-2.0",
}

_CLASSIFIER_LICENSES = {
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: Zope Public License": "ZPL-2.1",
    # Deliberately absent: "License :: OSI Approved :: BSD License"
    # does not state which BSD variant applies.
}

_PACK_CEILINGS = (
    "MAX_BUNDLE_ZIP_BYTES",
    "MAX_BUNDLE_ENTRIES",
    "MAX_BUNDLE_FILE_BYTES",
    "MAX_BUNDLE_TOTAL_BYTES",
)


class ManifestFailure(Exception):
    """A manifest input or claim does not hold; the message names it."""


def _sha256_bytes(data: bytes) -> str:
    """Return the sha256 bytes."""
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the sha256 file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_digests(path: Path) -> dict[str, str]:
    """Per-member content digests of a wheel (zip) or sdist (tar.gz)."""
    members: dict[str, str] = {}
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                members[info.filename] = _sha256_bytes(archive.read(info.filename))
    else:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        members[member.name] = _sha256_bytes(extracted.read())
    return members


def _metadata_fields(text: str) -> dict[str, str]:
    """First-occurrence header fields of a METADATA / PKG-INFO document."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            break
        if ":" in line:
            key, value = line.split(":", 1)
            fields.setdefault(key.strip(), value.strip())
    return fields


def _wheel_identity(wheel: Path) -> tuple[str, str]:
    """(name, version) from the wheel's own METADATA."""
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith(".dist-info/METADATA"):
                fields = _metadata_fields(archive.read(name).decode("utf-8"))
                return fields.get("Name", ""), fields.get("Version", "")
    raise ManifestFailure(f"wheel carries no dist-info METADATA: {wheel.name}")


def _sdist_identity(sdist: Path) -> tuple[str, str]:
    """(name, version) from the sdist's top-level PKG-INFO."""
    with tarfile.open(sdist, "r:gz") as archive:
        for member in archive.getmembers():
            if (
                member.isfile()
                and member.name.count("/") == 1
                and member.name.endswith("/PKG-INFO")
            ):
                extracted = archive.extractfile(member)
                if extracted is not None:
                    fields = _metadata_fields(extracted.read().decode("utf-8"))
                    return fields.get("Name", ""), fields.get("Version", "")
    raise ManifestFailure(f"sdist carries no top-level PKG-INFO: {sdist.name}")


def _tree_digest(source: Path) -> str:
    """Content digest of the source tree (generated dirs excluded).

    sha256 over the sorted lines ``<file sha256>  <relative path>``
    for every regular file outside the excluded generated/vendored
    directories — reproducible from any identical tree, e.g. a fresh
    checkout of the bound commit.
    """
    lines: list[str] = []
    for root, dirs, files in os.walk(source):
        dirs[:] = sorted(d for d in dirs if d not in _TREE_EXCLUDED and not d.endswith(".egg-info"))
        for name in sorted(files):
            path = Path(root) / name
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            lines.append(f"{_sha256_file(path)}  {relative}")
    lines.sort()
    return "sha256:" + _sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def _load_census() -> ModuleType:
    """Load tools/artifact_census.py by file location (as release_build does)."""
    path = Path(__file__).with_name("artifact_census.py")
    spec = importlib.util.spec_from_file_location("artifact_census", path)
    if spec is None or spec.loader is None:
        raise ManifestFailure("cannot load tools/artifact_census.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pinned_requirements(source: Path) -> dict[str, list[str]]:
    """The pinned builder requirements (name==version) per requirements file."""
    pinned: dict[str, list[str]] = {}
    for name in ("pip.txt", "pip-setuptools.txt", "build.txt"):
        path = source / "tools" / "requirements" / name
        specs: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip().rstrip("\\").strip()
            match = re.match(r"^([A-Za-z0-9._-]+)==([A-Za-z0-9._+!-]+)$", line)
            if match:
                specs.append(f"{match.group(1)}=={match.group(2)}")
        pinned[f"tools/requirements/{name}"] = sorted(specs)
    return pinned


def _const_value(node: ast.expr) -> int:
    """Evaluate an integer constant expression (literals, * and + only)."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _const_value(node.left) * _const_value(node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _const_value(node.left) + _const_value(node.right)
    raise ManifestFailure("pack ceiling is not an integer constant expression")


def _pack_ceilings(source: Path) -> dict[str, int]:
    """The RTI-02 producer ceilings as shipped in model_protocol_lab/pack.py."""
    tree = ast.parse((source / "model_protocol_lab" / "pack.py").read_text(encoding="utf-8"))
    found: dict[str, int] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in _PACK_CEILINGS
        ):
            found[node.targets[0].id] = _const_value(node.value)
    missing = [name for name in _PACK_CEILINGS if name not in found]
    if missing:
        raise ManifestFailure(f"pack ceilings not found in pack.py: {', '.join(missing)}")
    return {name: found[name] for name in _PACK_CEILINGS}


def _is_spdx_expression(text: str) -> bool:
    """True when text parses as an SPDX expression over known identifiers."""
    parts = re.findall(r"\(|\)|[^\s()]+", text)
    if not parts:
        return False
    previous: str | None = None
    for part in parts:
        if part in ("AND", "OR", "(", ")"):
            previous = part
            continue
        if part == "WITH":
            if previous is None or previous not in _SPDX_IDS:
                return False
            previous = "WITH"
            continue
        if previous == "WITH":
            if not re.fullmatch(r"[A-Za-z0-9.-]+", part):
                return False
            previous = part
            continue
        base = part.rstrip("+")
        if base not in _SPDX_IDS:
            return False
        previous = base
    return True


def _resolve_license(info: dict) -> tuple[str | None, str]:
    """Resolve (license, source) from PyPI JSON info via the documented cascade."""
    expression = info.get("license_expression")
    if type(expression) is str and expression.strip() and _is_spdx_expression(expression.strip()):
        return expression.strip(), "pypi:license_expression"
    text = info.get("license")
    if type(text) is str and text.strip():
        candidate = text.strip()
        if candidate in _LICENSE_TEXT:
            return _LICENSE_TEXT[candidate], "pypi:license"
        if _is_spdx_expression(candidate):
            return candidate, "pypi:license"
    for classifier in info.get("classifiers") or []:
        if classifier in _CLASSIFIER_LICENSES:
            return _CLASSIFIER_LICENSES[classifier], "pypi:classifier"
    return None, "unresolved"


def _pypi_fetch(url: str) -> dict:
    """Fetch and parse a PyPI JSON document."""
    # S310: every URL this tool opens is built from the fixed https
    # PyPI base plus a package name validated by _checked_name, so no
    # other scheme or host can reach urlopen here. The startswith
    # guard keeps that true if the construction ever changes.
    if not url.startswith("https://pypi.org/"):
        raise ManifestFailure("PyPI URL left the fixed PyPI host")
    request = urllib.request.Request(url, headers={"User-Agent": "model-protocol-lab-manifest"})  # noqa: S310  # nosec B310 -- URL asserted to the fixed https PyPI host above
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310  # nosec B310 -- URL asserted to the fixed https PyPI host above
        data = json.load(response)
    if not isinstance(data, dict):
        raise ManifestFailure(f"PyPI returned a non-object document: {url}")
    return data


def _normalize(name: str) -> str:
    """Return the normalize."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _checked_name(name: str) -> str:
    """Validate a distribution name before it enters a PyPI URL."""
    normalized = _normalize(name)
    if not re.fullmatch(r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?", normalized):
        raise ManifestFailure(f"invalid distribution name: {name!r}")
    return normalized


def _requirement_name(requirement: str) -> str | None:
    """The distribution name of a requirement string, or None if unparsable."""
    match = re.match(r"^\s*([A-Za-z0-9._-]+)", requirement)
    return match.group(1) if match else None


def _component(
    name: str,
    version: str,
    closure: str,
    scope: str,
    license_value: str | None,
    license_source: str,
) -> dict:
    """One CycloneDX component with its license and provenance properties."""
    if license_value is None:
        licenses: list[dict] = [{"license": {"name": "NOASSERTION"}}]
    elif " AND " in license_value or " OR " in license_value or " WITH " in license_value:
        licenses = [{"expression": license_value}]
    else:
        licenses = [{"license": {"id": license_value}}]
    return {
        "type": "library",
        "name": name,
        "version": version,
        "scope": scope,
        "licenses": licenses,
        "purl": f"pkg:pypi/{_normalize(name)}@{version}",
        "properties": [
            {"name": "mpl:closure", "value": closure},
            {"name": "mpl:license-source", "value": license_source},
        ],
    }


def build_sbom(
    source: Path,
    package_version: str,
    fetch: Callable[[str], dict],
) -> tuple[dict, list[str]]:
    """Build the CycloneDX SBOM; returns (sbom, unresolved license names)."""
    unresolved: list[str] = []
    components: list[dict] = []
    seen: set[str] = set()

    def add(name: str, version: str, closure: str, scope: str, info: dict) -> None:
        """Add."""
        license_value, license_source = _resolve_license(info)
        if license_value is None:
            unresolved.append(name)
        components.append(_component(name, version, closure, scope, license_value, license_source))
        seen.add(_normalize(name))

    # Dev closure: exactly pylock.toml's package set, at locked versions.
    with (source / "pylock.toml").open("rb") as handle:
        lock = tomllib.load(handle)
    for package in lock.get("packages", []):
        name, version = str(package["name"]), str(package["version"])
        info = fetch(f"{_PYPI_BASE}/pypi/{_checked_name(name)}/{version}/json")["info"]
        add(name, version, "dev", "excluded", info)

    # Hf closure: direct deps from pyproject, then a requires_dist walk.
    with (source / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    direct = project["project"]["optional-dependencies"]["hf"]
    queue: list[str] = []
    for requirement in direct:
        name = _requirement_name(requirement)
        if name is not None:
            queue.append(name)
    while queue:
        if len(seen) >= _MAX_WALK:
            raise ManifestFailure(f"hf closure walk exceeds {_MAX_WALK} packages")
        name = queue.pop(0)
        if _normalize(name) in seen:
            continue
        info = fetch(f"{_PYPI_BASE}/pypi/{_checked_name(name)}/json")["info"]
        add(str(info.get("name") or name), str(info.get("version") or ""), "hf", "optional", info)
        for requirement in info.get("requires_dist") or []:
            if "extra ==" in requirement:
                continue  # extra-gated: not part of the default closure
            dep = _requirement_name(requirement)
            if dep is not None and _normalize(dep) not in seen:
                queue.append(dep)

    components.sort(key=lambda c: str(c["name"]).lower())
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "model-protocol-lab",
                "version": package_version,
                "licenses": [{"license": {"id": "MIT"}}],
                "purl": f"pkg:pypi/model-protocol-lab@{package_version}",
            }
        },
        "components": components,
    }
    return sbom, sorted(unresolved)


def _read_result(path: Path, tool: str, artifact: Path) -> dict:
    """Load a verification tool's result JSON and check its artifact binding."""
    if not path.is_file():
        raise ManifestFailure(f"{tool} result is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestFailure(f"{tool} result is unreadable: {error}") from error
    if data.get("tool") != tool or data.get("result") != "passed":
        raise ManifestFailure(f"{tool} result does not record a pass: {path}")
    bound = data.get("artifact") or {}
    if bound.get("name") != artifact.name or bound.get("sha256") != _sha256_file(artifact):
        raise ManifestFailure(
            f"{tool} result is bound to a different artifact than {artifact.name}"
        )
    return data


def _dump(data: dict) -> bytes:
    """Deterministic JSON bytes (sorted keys, 2-space indent, final newline)."""
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def emit_manifest(
    *,
    dist: Path,
    source: Path,
    commit: str,
    epoch: int,
    build_record_path: Path,
    smoke_result_path: Path,
    scope_result_path: Path,
    fetch: Callable[[str], dict],
) -> Path:
    """Emit dist/artifact-manifest.json + dist/sbom.cdx.json; returns the manifest path."""
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ManifestFailure(f"dist must hold exactly one wheel and one sdist: {dist}")
    wheel, sdist = wheels[0], sdists[0]
    wheel_name, wheel_version = _wheel_identity(wheel)
    sdist_name, sdist_version = _sdist_identity(sdist)
    with (source / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    tree_name, tree_version = str(project["name"]), str(project["version"])
    if not (wheel_version == sdist_version == tree_version):
        raise ManifestFailure(
            f"version disagreement: wheel {wheel_version}, sdist {sdist_version}, "
            f"tree {tree_version}"
        )
    if _normalize(wheel_name) != _normalize(tree_name) or _normalize(sdist_name) != _normalize(
        tree_name
    ):
        raise ManifestFailure(
            f"package name disagreement: wheel {wheel_name}, sdist {sdist_name}, tree {tree_name}"
        )
    try:
        record = json.loads(build_record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestFailure(f"build record is unreadable: {error}") from error
    if record.get("verdict") != "identical":
        raise ManifestFailure(
            "build record does not record the reproducibility verdict 'identical'"
        )
    for label, artifact in (("wheel", wheel), ("sdist", sdist)):
        entry = record.get(label) or {}
        digest = _sha256_file(artifact)
        if (
            entry.get("name") != artifact.name
            or entry.get("sha256_run1") != digest
            or entry.get("sha256_run2") != digest
        ):
            raise ManifestFailure(
                f"build record's {label} hashes do not match the artifact in dist"
            )
    smoke = _read_result(smoke_result_path, "artifact_smoke", wheel)
    scope = _read_result(scope_result_path, "sdist_scope", sdist)
    census = _load_census()
    findings = (
        census.check_source_tree(source)
        + census.check_wheel(wheel, source)
        + census.check_sdist(sdist, source)
    )
    if findings:
        raise ManifestFailure(f"artifact census failed: {'; '.join(findings)}")
    wheel_members = _member_digests(wheel)
    sdist_members = _member_digests(sdist)
    sbom, unresolved = build_sbom(source, tree_version, fetch)
    sbom_bytes = _dump(sbom)
    (dist / _SBOM_NAME).write_bytes(sbom_bytes)
    manifest = {
        "schema": _SCHEMA,
        "package": {"name": tree_name, "version": tree_version},
        "artifacts": [
            {
                "name": wheel.name,
                "sha256": _sha256_file(wheel),
                "size": wheel.stat().st_size,
                "media_type": "application/zip",
                "members": wheel_members,
            },
            {
                "name": sdist.name,
                "sha256": _sha256_file(sdist),
                "size": sdist.stat().st_size,
                "media_type": "application/gzip",
                "members": sdist_members,
            },
        ],
        "source": {
            "repository": _REPOSITORY,
            "commit": commit,
            "source_date_epoch": epoch,
            "tree_digest": _tree_digest(source),
        },
        "builder": {
            "tool": "tools/release_build.py",
            "tool_sha256": _sha256_file(source / "tools" / "release_build.py"),
            "interpreter": sys.version.split()[0],
            "pinned_requirements": _pinned_requirements(source),
        },
        "reproducibility": {
            "verdict": "identical",
            "epoch": epoch,
            "wheel_sha256": record["wheel"]["sha256_run1"],
            "sdist_sha256": record["sdist"]["sha256_run1"],
            "conditions": (
                "two builds from independent copies of the same tree on one host, "
                "SOURCE_DATE_EPOCH-normalized (tools/release_build.py); "
                "cross-platform reproducibility is not claimed"
            ),
        },
        "verification": {
            "census": {
                "result": "passed",
                "unexpected": 0,
                "wheel_members": len(wheel_members),
                "sdist_members": len(sdist_members),
            },
            "origin_smoke": smoke,
            "sdist_scope": scope,
            "pack_ceilings": _pack_ceilings(source),
        },
        "sbom": {
            "file": _SBOM_NAME,
            "format": "CycloneDX 1.5",
            "sha256": _sha256_bytes(sbom_bytes),
            "unresolved_licenses": unresolved,
            "notes": (
                "runtime closure of the wheel is empty (no Requires-Dist outside "
                "extras); dev closure is exactly pylock.toml; hf inventory is "
                "enumerated from PyPI JSON at generation time and each hf "
                "component's version is the release whose metadata was read"
            ),
        },
    }
    manifest_path = dist / "artifact-manifest.json"
    manifest_path.write_bytes(_dump(manifest))
    return manifest_path


def verify_manifest(
    *,
    dist: Path,
    manifest_path: Path,
    source: Path | None = None,
) -> list[tuple[str, bool, str]]:
    """Re-check every manifest claim; returns (claim, ok, detail) rows."""
    claims: list[tuple[str, bool, str]] = []

    def claim(name: str, ok: bool, detail: str = "") -> None:
        """Claim."""
        claims.append((name, ok, detail))

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [("manifest-readable", False, str(error))]
    claim("manifest-readable", True)
    claim("schema", data.get("schema") == _SCHEMA, str(data.get("schema")))
    package = data.get("package") or {}
    artifacts = {entry.get("name"): entry for entry in data.get("artifacts") or []}
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    present = (
        len(wheels) == 1
        and len(sdists) == 1
        and set(artifacts)
        == {
            wheels[0].name,
            sdists[0].name,
        }
    )
    claim("artifacts-present", present, ", ".join(sorted(artifacts)))
    if present:
        wheel, sdist = wheels[0], sdists[0]
        sizes_ok = all(
            (dist / name).stat().st_size == entry.get("size") for name, entry in artifacts.items()
        )
        claim("artifact-sizes", sizes_ok)
        hashes_ok = all(
            _sha256_file(dist / name) == entry.get("sha256") for name, entry in artifacts.items()
        )
        claim("artifact-hashes", hashes_ok)
        members_ok = all(
            _member_digests(dist / name) == entry.get("members")
            for name, entry in artifacts.items()
        )
        claim("artifact-members", members_ok)
        try:
            _wn, wheel_version = _wheel_identity(wheel)
            _sn, sdist_version = _sdist_identity(sdist)
            claim(
                "package-version",
                package.get("version") == wheel_version == sdist_version,
                f"manifest {package.get('version')}, wheel {wheel_version}, sdist {sdist_version}",
            )
        except ManifestFailure as error:
            claim("package-version", False, str(error))
        verification = data.get("verification") or {}
        bindings_ok = True
        for block, artifact in (("origin_smoke", wheel), ("sdist_scope", sdist)):
            bound = (verification.get(block) or {}).get("artifact") or {}
            if bound.get("sha256") != artifacts[artifact.name].get("sha256"):
                bindings_ok = False
        claim("verification-bindings", bindings_ok)
        reproducibility = data.get("reproducibility") or {}
        claim(
            "reproducibility-binding",
            reproducibility.get("verdict") == "identical"
            and reproducibility.get("wheel_sha256") == artifacts[wheel.name].get("sha256")
            and reproducibility.get("sdist_sha256") == artifacts[sdist.name].get("sha256"),
        )
    sbom_block = data.get("sbom") or {}
    sbom_path = dist / str(sbom_block.get("file", _SBOM_NAME))
    if sbom_path.is_file():
        claim("sbom-digest", _sha256_file(sbom_path) == sbom_block.get("sha256"))
        try:
            sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
            root = (sbom.get("metadata") or {}).get("component") or {}
            claim(
                "sbom-package",
                root.get("name") == package.get("name")
                and root.get("version") == package.get("version"),
            )
        except json.JSONDecodeError as error:
            claim("sbom-package", False, str(error))
    else:
        claim("sbom-digest", False, f"missing {sbom_path.name}")
    if source is not None:
        claim(
            "source-tree",
            (data.get("source") or {}).get("tree_digest") == _tree_digest(source),
        )
    return claims


def _build_wheel(package_dir: Path, out_dir: Path, epoch: int) -> Path:
    """Build the wheel from an extracted sdist tree (the pinned invocation)."""
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = str(epoch)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(out_dir),
            str(package_dir),
        ],
        check=True,
        env=env,
        capture_output=True,
    )
    wheels = sorted(out_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ManifestFailure(f"from-sdist rebuild produced {len(wheels)} wheels")
    return wheels[0]


def _zip_timestamps(path: Path) -> dict[str, tuple[int, ...]]:
    """Zip entry timestamps per member of a wheel."""
    with zipfile.ZipFile(path) as archive:
        return {info.filename: info.date_time for info in archive.infolist()}


def rebuild_check(
    *,
    dist: Path,
    manifest_path: Path,
    epoch: int,
    work: Path | None = None,
    builder: Callable[[Path, Path, int], Path] = _build_wheel,
) -> list[str]:
    """Rebuild the wheel from the sdist under the declared conditions.

    Returns the report lines on byte-identity; raises ManifestFailure
    naming the differing members on a content mismatch, or carrying
    the metadata-only explanation when only archive metadata differs.
    """
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestFailure(f"manifest is unreadable: {error}") from error
    entries = {entry.get("name"): entry for entry in data.get("artifacts") or []}
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ManifestFailure(f"dist must hold exactly one wheel and one sdist: {dist}")
    wheel, sdist = wheels[0], sdists[0]
    for artifact in (wheel, sdist):
        entry = entries.get(artifact.name)
        if entry is None or entry.get("sha256") != _sha256_file(artifact):
            raise ManifestFailure(
                f"{artifact.name} does not match the manifest; refusing to rebuild"
            )
    if work is None:
        work = Path(tempfile.mkdtemp(prefix="mpl_rebuild_check_"))
    extract = work / "extract"
    extract.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(sdist, "r:gz") as archive:
            archive.extractall(extract, filter="data")
    except (tarfile.TarError, OSError) as error:
        raise ManifestFailure(f"sdist extraction refused (data filter): {error}") from error
    roots = sorted(p for p in extract.iterdir() if p.is_dir())
    if len(roots) != 1:
        raise ManifestFailure("sdist does not hold exactly one top-level directory")
    package_dir = roots[0]
    for root, dirs, files in os.walk(package_dir):
        for name in dirs + files:
            os.utime(Path(root) / name, (epoch, epoch))
    os.utime(package_dir, (epoch, epoch))
    out_dir = work / "out"
    out_dir.mkdir(exist_ok=True)
    rebuilt = builder(package_dir, out_dir, epoch)
    expected_members = entries[wheel.name].get("members") or {}
    rebuilt_members = _member_digests(rebuilt)
    differing = sorted(
        name
        for name in set(expected_members) | set(rebuilt_members)
        if expected_members.get(name) != rebuilt_members.get(name)
    )
    if differing:
        raise ManifestFailure(f"from-sdist rebuild content mismatch: {', '.join(differing)}")
    if _sha256_file(rebuilt) == entries[wheel.name].get("sha256"):
        return [
            "from-sdist rebuild: byte-identical to the manifest wheel "
            "under the declared conditions (tarfile data-filter extraction, "
            "mtimes normalized to SOURCE_DATE_EPOCH, epoch-stamped build)"
        ]
    reference_times = _zip_timestamps(wheel)
    rebuilt_times = _zip_timestamps(rebuilt)
    restamped = sorted(
        name for name in reference_times if reference_times.get(name) != rebuilt_times.get(name)
    )
    raise ManifestFailure(
        "from-sdist rebuild is not byte-identical, but every member's content "
        "digest matches the manifest: the delta is metadata-only (zip entry "
        f"timestamps differ on {len(restamped)} member(s): {', '.join(restamped)})"
    )


def main(argv: list[str] | None = None) -> int:
    """Return the main."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    emit = sub.add_parser("emit", help="emit the manifest + SBOM into --dist")
    emit.add_argument("--dist", type=Path, required=True)
    emit.add_argument("--source", type=Path, default=Path("."))
    emit.add_argument("--commit", required=True, help="commit SHA the artifacts were built from")
    emit.add_argument("--epoch", type=int, required=True, help="SOURCE_DATE_EPOCH the build used")
    emit.add_argument("--build-record", type=Path, required=True)
    emit.add_argument("--smoke-result", type=Path, required=True)
    emit.add_argument("--scope-result", type=Path, required=True)
    verify = sub.add_parser("verify", help="re-check every claim of a manifest")
    verify.add_argument("--dist", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, default=None)
    verify.add_argument("--source", type=Path, default=None)
    rebuild = sub.add_parser(
        "rebuild-check", help="rebuild the wheel from the sdist under the declared conditions"
    )
    rebuild.add_argument("--dist", type=Path, required=True)
    rebuild.add_argument("--manifest", type=Path, default=None)
    rebuild.add_argument(
        "--epoch", type=int, required=True, help="SOURCE_DATE_EPOCH of the release"
    )
    args = parser.parse_args(argv)
    if args.command == "emit":
        try:
            path = emit_manifest(
                dist=args.dist,
                source=args.source,
                commit=args.commit,
                epoch=args.epoch,
                build_record_path=args.build_record,
                smoke_result_path=args.smoke_result,
                scope_result_path=args.scope_result,
                fetch=_pypi_fetch,
            )
        except ManifestFailure as error:
            print(f"artifact manifest refused: {error}", file=sys.stderr)
            return 2
        print(f"{_sha256_file(path)}  {path}")
        sbom_path = args.dist / _SBOM_NAME
        print(f"{_sha256_file(sbom_path)}  {sbom_path}")
        print("artifact manifest: emitted")
        return 0
    if args.command == "rebuild-check":
        try:
            lines = rebuild_check(
                dist=args.dist,
                manifest_path=args.manifest or (args.dist / "artifact-manifest.json"),
                epoch=args.epoch,
            )
        except ManifestFailure as error:
            print(f"rebuild check FAILED: {error}", file=sys.stderr)
            return 2
        for line in lines:
            print(f"  {line}")
        print("rebuild check: OK")
        return 0
    manifest_path = args.manifest or (args.dist / "artifact-manifest.json")
    claims = verify_manifest(dist=args.dist, manifest_path=manifest_path, source=args.source)
    failed = 0
    for name, ok, detail in claims:
        print(f"  claim {name}: {'ok' if ok else 'FAILED'} {detail}".rstrip())
        failed += 0 if ok else 1
    if failed:
        print(f"artifact manifest: {failed} claim(s) FAILED", file=sys.stderr)
        return 1
    print("artifact manifest: all claims verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
