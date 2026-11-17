"""W5-4 census gate: archive members and surface counts are pinned.

The release artifacts carry exactly the declared members, and the
documented surface cannot silently diverge from the shipped one:

- Wheel members are the package ``.py`` files + ``py.typed`` + the
  six dist-info files. Nothing else (no tests, no data files, no
  weights, no caches).
- Sdist members are the package files, the ``test_*.py`` modules, the
  three named non-code test files, the six regenerated egg-info files,
  and the nine named root files. No ``docs/``, ``examples/``, ``tools/``,
  ``.github/``, ``site/``, or any other checkout content.
- The source tree itself is checked before building: a stray file in
  ``model_protocol_lab/`` fails the build with the member named,
  whether or not the build backend would have packaged it.
- Surface counts are pinned from the source registries, not from
  docs: 12 CLI commands, 4 MCP tools, 65 report sections (57
  dispatched + 8 undispatched-with-reasons; the W0 audit's "62" was
  the count at that head, and the W7 census measured 64). The 29
  catalog families are pinned by their exact list in
  test_cycle39/test_cycle41/test_cycle46, not repeated here.

Every rule below has a failing variant that names the offending
member, so removing a rule fails a named test (mutant-kill by
construction). Archives in these tests are synthetic; the real
wheel/sdist are checked by ``tools/release_build.py`` itself, which
runs this census on every build before publishing its output.
"""

from __future__ import annotations

import io
import shutil
import tarfile
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

from model_protocol_lab import catalog as catalog_module
from model_protocol_lab import cli as cli_module
from model_protocol_lab import diffreports as diffreports_module
from model_protocol_lab import mcp_server as mcp_module
from model_protocol_lab import reporting as reporting_module
from tools import artifact_census, release_build

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
_EXPECTED_COMMANDS = (
    "catalog",
    "debug",
    "diff-backends",
    "diff-reports",
    "doctor",
    "exec-backends",
    "fixtures",
    "inspect",
    "pack",
    "run",
    "surface-noise",
    "verify",
)
_EXPECTED_MCP_TOOLS = (
    "probe_surface_noise_stability",
    "surface_noise_controls",
    "surface_noise_hardening_notes",
    "surface_noise_protocol_info",
)


def _fake_source(root: Path) -> Path:
    """Minimal source tree: package + pyproject + one test module."""
    package = root / "model_protocol_lab"
    package.mkdir(parents=True)
    for name in ("a.py", "b.py"):
        (package / name).write_text(f"# {name}\n", encoding="utf-8")
    (package / "py.typed").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "model-protocol-lab"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )
    tests = root / "tests"
    (tests / "data").mkdir(parents=True)
    (tests / "test_a.py").write_text("# test\n", encoding="utf-8")
    (tests / "data" / "UnicodeData.txt").write_text("data\n", encoding="utf-8")
    (tests / "CHECKOUT_ONLY").write_text("none\n", encoding="utf-8")
    (tests / "test_layers.json").write_text("{}\n", encoding="utf-8")
    return root


def _wheel_members(
    source: Path, extra: tuple[str, ...] = (), drop: tuple[str, ...] = ()
) -> list[str]:
    """Member list of the wheel the fake source tree should produce."""
    members = [f"model_protocol_lab/{name}" for name in ("a.py", "b.py", "py.typed")]
    members += [f"{_DIST_INFO}/{name}" for name in _DIST_FILES]
    members += list(extra)
    return [m for m in members if m not in drop]


def _write_wheel(path: Path, members: list[str]) -> None:
    """Write a synthetic wheel with the given members."""
    with warnings.catch_warnings():
        # The duplicate-member test writes one name twice on purpose;
        # zipfile warns about that, and the warning is the test's point.
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            for member in members:
                archive.writestr(member, f"content of {member}\n")


def _sdist_members(
    source: Path, extra: tuple[str, ...] = (), drop: tuple[str, ...] = ()
) -> list[str]:
    """Member list of the sdist the fake source tree should produce."""
    top = f"model_protocol_lab-{_VERSION}"
    members = [f"{top}/{name}" for name in _ROOT_FILES]
    members += [f"{top}/model_protocol_lab/{name}" for name in ("a.py", "b.py", "py.typed")]
    members += [f"{top}/model_protocol_lab.egg-info/{name}" for name in _EGG_INFO_FILES]
    members += [
        f"{top}/tests/test_a.py",
        f"{top}/tests/data/UnicodeData.txt",
        f"{top}/tests/CHECKOUT_ONLY",
        f"{top}/tests/test_layers.json",
    ]
    members += list(extra)
    return [m for m in members if m not in drop]


def _write_sdist(path: Path, members: list[str]) -> None:
    """Write a synthetic sdist with the given members."""
    with tarfile.open(path, "w:gz") as archive:
        for member in members:
            data = f"content of {member}\n".encode()
            info = tarfile.TarInfo(member)
            info.size = len(data)
            info.mtime = 1700000000
            archive.addfile(info, io.BytesIO(data))


class TestSourceTreeCensus(unittest.TestCase):
    """Test source tree census."""

    def setUp(self) -> None:
        """Set up."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.source = _fake_source(self.tmp / "src")

    def tearDown(self) -> None:
        """Tear down."""
        self._tmpdir.cleanup()

    def test_clean_tree_passes(self) -> None:
        """Test clean tree passes."""
        self.assertEqual(artifact_census.check_source_tree(self.source), [])

    def test_stray_binary_named(self) -> None:
        """Test stray binary named."""
        stray = self.source / "model_protocol_lab" / "weights.bin"
        stray.write_bytes(b"\x00\x01\x02")
        findings = artifact_census.check_source_tree(self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("model_protocol_lab/weights.bin", findings[0])

    def test_stray_text_and_loose_pyc_named(self) -> None:
        """Test stray text and loose pyc named."""
        package = self.source / "model_protocol_lab"
        (package / "notes.txt").write_text("notes\n", encoding="utf-8")
        (package / "stray.cpython-312.pyc").write_bytes(b"\x00")
        findings = artifact_census.check_source_tree(self.source)
        self.assertEqual(len(findings), 2)
        joined = "\n".join(findings)
        self.assertIn("model_protocol_lab/notes.txt", joined)
        self.assertIn("model_protocol_lab/stray.cpython-312.pyc", joined)

    def test_pycache_ignored(self) -> None:
        """Test pycache ignored."""
        cache = self.source / "model_protocol_lab" / "__pycache__"
        cache.mkdir()
        (cache / "a.cpython-312.pyc").write_bytes(b"\x00")
        self.assertEqual(artifact_census.check_source_tree(self.source), [])

    def test_excluded_names_match_release_build(self) -> None:
        """Test excluded names match release build."""
        self.assertEqual(artifact_census._EXCLUDED_NAMES, release_build._EXCLUDED_NAMES)


class TestWheelCensus(unittest.TestCase):
    """Test wheel census."""

    def setUp(self) -> None:
        """Set up."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.source = _fake_source(self.tmp / "src")
        self.wheel = self.tmp / "model_protocol_lab-0.2.0-py3-none-any.whl"

    def tearDown(self) -> None:
        """Tear down."""
        self._tmpdir.cleanup()

    def test_clean_wheel_passes(self) -> None:
        """Test clean wheel passes."""
        _write_wheel(self.wheel, _wheel_members(self.source))
        self.assertEqual(artifact_census.check_wheel(self.wheel, self.source), [])

    def test_stray_member_named(self) -> None:
        """Test stray member named."""
        members = _wheel_members(self.source, extra=("model_protocol_lab/weights.bin",))
        _write_wheel(self.wheel, members)
        findings = artifact_census.check_wheel(self.wheel, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("model_protocol_lab/weights.bin", findings[0])

    def test_test_in_wheel_named(self) -> None:
        """Test test in wheel named."""
        members = _wheel_members(self.source, extra=("tests/test_x.py",))
        _write_wheel(self.wheel, members)
        findings = artifact_census.check_wheel(self.wheel, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("tests/test_x.py", findings[0])

    def test_missing_pytyped_named(self) -> None:
        """Test missing pytyped named."""
        members = _wheel_members(self.source, drop=("model_protocol_lab/py.typed",))
        _write_wheel(self.wheel, members)
        findings = artifact_census.check_wheel(self.wheel, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("model_protocol_lab/py.typed", findings[0])

    def test_missing_module_named(self) -> None:
        """Test missing module named."""
        members = _wheel_members(self.source, drop=("model_protocol_lab/b.py",))
        _write_wheel(self.wheel, members)
        findings = artifact_census.check_wheel(self.wheel, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("model_protocol_lab/b.py", findings[0])

    def test_extra_dist_info_named(self) -> None:
        """Test extra dist info named."""
        members = _wheel_members(self.source, extra=(f"{_DIST_INFO}/extra.txt",))
        _write_wheel(self.wheel, members)
        findings = artifact_census.check_wheel(self.wheel, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn(f"{_DIST_INFO}/extra.txt", findings[0])

    def test_wrong_version_named(self) -> None:
        """Test wrong version named."""
        wrong = "model_protocol_lab-9.9.9.dist-info"
        members = [m.replace(_DIST_INFO, wrong) for m in _wheel_members(self.source)]
        _write_wheel(self.wheel, members)
        findings = artifact_census.check_wheel(self.wheel, self.source)
        self.assertIn(wrong, "\n".join(findings))

    def test_duplicate_member_named(self) -> None:
        """Test duplicate member named."""
        members = _wheel_members(self.source) + ["model_protocol_lab/a.py"]
        _write_wheel(self.wheel, members)
        findings = artifact_census.check_wheel(self.wheel, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("model_protocol_lab/a.py", findings[0])


class TestSdistCensus(unittest.TestCase):
    """Test sdist census."""

    def setUp(self) -> None:
        """Set up."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.source = _fake_source(self.tmp / "src")
        self.sdist = self.tmp / "model_protocol_lab-0.2.0.tar.gz"

    def tearDown(self) -> None:
        """Tear down."""
        self._tmpdir.cleanup()

    def test_clean_sdist_passes(self) -> None:
        """Test clean sdist passes."""
        _write_sdist(self.sdist, _sdist_members(self.source))
        self.assertEqual(artifact_census.check_sdist(self.sdist, self.source), [])

    def test_docs_member_named(self) -> None:
        """Test docs member named."""
        top = f"model_protocol_lab-{_VERSION}"
        members = _sdist_members(self.source, extra=(f"{top}/docs/USAGE.md",))
        _write_sdist(self.sdist, members)
        findings = artifact_census.check_sdist(self.sdist, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("docs/USAGE.md", findings[0])

    def test_tools_member_named(self) -> None:
        """Test tools member named."""
        top = f"model_protocol_lab-{_VERSION}"
        members = _sdist_members(self.source, extra=(f"{top}/tools/x.py",))
        _write_sdist(self.sdist, members)
        findings = artifact_census.check_sdist(self.sdist, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("tools/x.py", findings[0])

    def test_stray_test_member_named(self) -> None:
        """Test stray test member named."""
        top = f"model_protocol_lab-{_VERSION}"
        members = _sdist_members(self.source, extra=(f"{top}/tests/stray.bin",))
        _write_sdist(self.sdist, members)
        findings = artifact_census.check_sdist(self.sdist, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("tests/stray.bin", findings[0])

    def test_non_test_py_in_tests_named(self) -> None:
        """Test non test py in tests named."""
        top = f"model_protocol_lab-{_VERSION}"
        members = _sdist_members(self.source, extra=(f"{top}/tests/helper.py",))
        _write_sdist(self.sdist, members)
        findings = artifact_census.check_sdist(self.sdist, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("tests/helper.py", findings[0])

    def test_missing_test_module_named(self) -> None:
        """Test missing test module named."""
        top = f"model_protocol_lab-{_VERSION}"
        members = _sdist_members(self.source, drop=(f"{top}/tests/test_a.py",))
        _write_sdist(self.sdist, members)
        findings = artifact_census.check_sdist(self.sdist, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("tests/test_a.py", findings[0])

    def test_missing_root_file_named(self) -> None:
        """Test missing root file named."""
        top = f"model_protocol_lab-{_VERSION}"
        members = _sdist_members(self.source, drop=(f"{top}/README.md",))
        _write_sdist(self.sdist, members)
        findings = artifact_census.check_sdist(self.sdist, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("README.md", findings[0])

    def test_unexpected_root_file_named(self) -> None:
        """Test unexpected root file named."""
        top = f"model_protocol_lab-{_VERSION}"
        members = _sdist_members(self.source, extra=(f"{top}/SECURITY.md",))
        _write_sdist(self.sdist, members)
        findings = artifact_census.check_sdist(self.sdist, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("SECURITY.md", findings[0])

    def test_missing_egg_info_named(self) -> None:
        """Test missing egg info named."""
        top = f"model_protocol_lab-{_VERSION}"
        members = _sdist_members(
            self.source, drop=(f"{top}/model_protocol_lab.egg-info/requires.txt",)
        )
        _write_sdist(self.sdist, members)
        findings = artifact_census.check_sdist(self.sdist, self.source)
        self.assertEqual(len(findings), 1)
        self.assertIn("model_protocol_lab.egg-info/requires.txt", findings[0])


class TestReleaseBuildWiring(unittest.TestCase):
    """Test release build wiring."""

    def test_stray_in_package_tree_refuses_build(self) -> None:
        """Test stray in package tree refuses build."""
        tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(tmpdir.name)
        try:
            source = tmp / "src"
            source.mkdir()
            repo = Path(__file__).resolve().parent.parent
            shutil.copytree(repo / "model_protocol_lab", source / "model_protocol_lab")
            shutil.copyfile(repo / "pyproject.toml", source / "pyproject.toml")
            stray = source / "model_protocol_lab" / "weights.bin"
            stray.write_bytes(b"\x00\x01\x02")
            out = tmp / "out"
            import contextlib
            import io as _io

            stderr = _io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = release_build.main(
                    [
                        "--source",
                        str(source),
                        "--output",
                        str(out),
                        "--epoch",
                        "1700000000",
                        "--expect-version",
                        "0.2.0",
                    ]
                )
            self.assertNotEqual(code, 0)
            self.assertIn("model_protocol_lab/weights.bin", stderr.getvalue())
            self.assertEqual(list(out.glob("*")), [])
        finally:
            tmpdir.cleanup()


class TestSurfaceCensus(unittest.TestCase):
    """Test surface census."""

    def test_cli_12_commands_exact(self) -> None:
        """Test cli 12 commands exact."""
        import argparse

        parser = cli_module.build_parser()
        subs = [
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        ]
        self.assertEqual(len(subs), 1)
        self.assertEqual(sorted(subs[0].choices), list(_EXPECTED_COMMANDS))

    def test_mcp_4_tools_exact(self) -> None:
        """Test mcp 4 tools exact."""
        self.assertEqual(sorted(mcp_module.TOOLS), list(_EXPECTED_MCP_TOOLS))

    def test_catalog_29_families(self) -> None:
        """Test catalog 29 families."""
        # The exact family list is pinned by test_cycle39/test_cycle41/
        # test_cycle46; this asserts the count here so the census file
        # is the single place that names all four surface counts.
        self.assertEqual(len(catalog_module.CATALOG), 29)

    def test_sections_65_and_registries_agree(self) -> None:
        """Test sections 65 and registries agree."""
        section_keys = set(reporting_module._REPORT_SECTION_KEYS)
        diff_keys = set(diffreports_module.SECTION_DIFF_KEYS) | set(
            diffreports_module.UNDISPATCHED_SECTIONS
        )
        self.assertEqual(section_keys, diff_keys)
        self.assertEqual(len(section_keys), 65)
        self.assertEqual(len(diffreports_module._SECTION_PAIRS), 57)
        self.assertEqual(len(diffreports_module.UNDISPATCHED_SECTIONS), 8)


if __name__ == "__main__":
    unittest.main()
