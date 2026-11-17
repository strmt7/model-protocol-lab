"""Build-time census gate for release artifacts (W5-4).

A release artifact carries exactly the declared members, and the
documented surface cannot silently diverge from the shipped one.
Who uses this on Monday morning: the release engineer, before a
publish-capable job, to prove the wheel and sdist contain nothing
but what the build declares — no tests in the wheel, no weights,
no caches, no personal paths, no checkout content in the sdist.

Rules (all derived by execution from the current tree's real
artifacts, not from the readiness prose — the readiness census
counted 75 package modules; the tree now carries 78, and the gate
checks classes, not a frozen member list):

- Source tree (checked before building): every file under
  ``model_protocol_lab/`` is a ``.py`` module or ``py.typed``.
  Anything else (weights, notes, loose ``.pyc``) is named and the
  build is refused, whether or not the backend would have packaged
  it. Build-excluded directories (``__pycache__`` and the rest of
  ``release_build._EXCLUDED_NAMES``, pinned equal by test) do not
  count.
- Wheel: members are the package ``.py`` files + ``py.typed`` + the
  six dist-info files (``METADATA``, ``WHEEL``, ``RECORD``,
  ``entry_points.txt``, ``top_level.txt``,
  ``licenses/LICENSE``) under the dist-info directory for the
  source tree's version. Completeness runs both ways: no unexpected
  member, no missing module, no duplicate member, no wrong-version
  dist-info directory.
- Sdist: members are the nine named root files, the package files,
  the six regenerated egg-info files, the ``test_*.py`` modules,
  and the three named non-code test files
  (``tests/data/UnicodeData.txt``, ``tests/CHECKOUT_ONLY``,
  ``tests/test_layers.json``). No ``docs/``, ``examples/``,
  ``tools/``, ``.github/``, ``site/``, or any other checkout
  content; the ``tests/__init__.py`` and underscore-prefixed test
  helpers the checkout carries are not shipped by setuptools and
  are not required (the rule is ``test_*.py``, measured, not a
  hardcoded exception list).

``tools/release_build.py`` calls these checks on every build: the
source tree before building, the built wheel and sdist before their
hashes are published to the output directory. This module is also
runnable standalone for inspection:

Usage: python tools/artifact_census.py --source . --wheel dist/*.whl \\
    --sdist dist/*.tar.gz
With neither --wheel nor --sdist, only the source tree is checked.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

# Twin of release_build._EXCLUDED_NAMES: the directories the build
# copy prunes, so the source check sees the same tree the build
# would copy. Kept in sync by test (test_w5_census_gate) instead of
# by import, so this module stays runnable as a plain script.
_EXCLUDED_NAMES = {
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    ".tox",
}

_PACKAGE = "model_protocol_lab"
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
_TEST_FILES = (
    "tests/data/UnicodeData.txt",
    "tests/CHECKOUT_ONLY",
    "tests/test_layers.json",
)


def _package_version(source: Path) -> str:
    """Return the source tree's package version."""
    with (source / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    version = data["project"]["version"]
    if type(version) is not str or not version:
        raise SystemExit("artifact census failed: pyproject.toml has no project.version")
    return version


def _pruned_walk(root: Path) -> list[Path]:
    """Files under root, with build-excluded directories pruned."""
    found: list[Path] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_NAMES and not d.endswith(".egg-info")]
        for name in files:
            found.append(Path(current) / name)
    return sorted(found)


def _package_files(source: Path) -> set[str]:
    """Relative paths of the package files the build would copy."""
    package = source / _PACKAGE
    if not package.is_dir():
        return set()
    return {path.relative_to(source).as_posix() for path in _pruned_walk(package)}


def _test_modules(source: Path) -> set[str]:
    """Relative paths of the shipped test modules (test_*.py)."""
    tests = source / "tests"
    if not tests.is_dir():
        return set()
    return {
        path.relative_to(source).as_posix()
        for path in sorted(tests.glob("test_*.py"))
        if path.is_file()
    }


def check_source_tree(source: Path) -> list[str]:
    """Unexpected files in the package tree, as census findings."""
    findings: list[str] = []
    for relative in sorted(_package_files(source)):
        name = relative.rsplit("/", 1)[-1]
        if name != "py.typed" and not name.endswith(".py"):
            findings.append(f"source:unexpected: {relative}")
    return findings


def check_wheel(wheel: Path, source: Path) -> list[str]:
    """Census findings for a wheel against the source tree."""
    findings: list[str] = []
    version = _package_version(source)
    expected_dist = f"{_PACKAGE}-{version}.dist-info"
    expected = _package_files(source) | {f"{expected_dist}/{name}" for name in _DIST_FILES}
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
    seen: set[str] = set()
    prefixes: set[str] = set()
    for member in members:
        if member in seen:
            findings.append(f"wheel:duplicate: {member}")
            continue
        seen.add(member)
        parts = member.split("/")
        if (
            len(parts) == 2
            and parts[0] == _PACKAGE
            and (parts[1] == "py.typed" or parts[1].endswith(".py"))
        ):
            continue
        if len(parts) >= 2 and parts[0].endswith(".dist-info"):
            prefixes.add(parts[0])
            rest = "/".join(parts[1:])
            if rest not in _DIST_FILES:
                findings.append(f"wheel:unexpected: {member}")
            continue
        findings.append(f"wheel:unexpected: {member}")
    for prefix in sorted(prefixes):
        if prefix != expected_dist:
            findings.append(f"wheel:version: {prefix} != {expected_dist}")
    for member in sorted(expected - seen):
        findings.append(f"wheel:missing: {member}")
    return findings


def check_sdist(sdist: Path, source: Path) -> list[str]:
    """Census findings for an sdist against the source tree."""
    findings: list[str] = []
    version = _package_version(source)
    expected_top = f"{_PACKAGE}-{version}"
    expected = set(_ROOT_FILES)
    expected |= _package_files(source)
    expected |= {f"{_PACKAGE}.egg-info/{name}" for name in _EGG_INFO_FILES}
    expected |= _test_modules(source)
    expected |= set(_TEST_FILES)
    with tarfile.open(sdist, "r:gz") as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
    tops: set[str] = set()
    relative: list[str] = []
    for member in members:
        parts = member.split("/")
        tops.add(parts[0])
        relative.append("/".join(parts[1:]) if len(parts) > 1 else parts[0])
    for top in sorted(tops):
        if top != expected_top:
            findings.append(f"sdist:version: {top} != {expected_top}")
    seen: set[str] = set()
    for name in relative:
        if name in seen:
            findings.append(f"sdist:duplicate: {name}")
            continue
        seen.add(name)
        if name in expected:
            continue
        findings.append(f"sdist:unexpected: {name}")
    for member in sorted(expected - seen):
        findings.append(f"sdist:missing: {member}")
    return findings


def check_artifacts(
    source: Path, wheel: Path | None = None, sdist: Path | None = None
) -> list[str]:
    """All census findings for a source tree and its artifacts."""
    findings = check_source_tree(source)
    if wheel is not None:
        findings += check_wheel(wheel, source)
    if sdist is not None:
        findings += check_sdist(sdist, source)
    return findings


def main(argv: list[str] | None = None) -> int:
    """Return the main."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--wheel", type=Path, default=None)
    parser.add_argument("--sdist", type=Path, default=None)
    args = parser.parse_args(argv)
    source = args.source.resolve()
    findings = check_artifacts(source, args.wheel, args.sdist)
    if findings:
        for finding in findings:
            print(f"artifact census failed: {finding}", file=sys.stderr)
        return 2
    print("artifact census passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
