"""W5 push 3 (RR-12) — the sdist scope is declared, and the declaration is pinned.

The finding: the sdist shipped the full test suite while part of it
audits the checkout itself, so a recipient running the tarball's
tests met unexplained collection errors and failures. The fix is a
declared scope: ``tests/CHECKOUT_ONLY`` names every module that
needs checkout content, with the reason per module, and
``tools/sdist_scope.py`` executes the remaining declared scope from
the sdist alone.

These tests pin the contract from the checkout side (the runner
itself is the artifact-side execution, wired into the release
workflow):

* the manifest parses under the runner's own parser, is sorted, and
  every entry names a real test-layer module (a module that is both
  heavy and checkout-bound appears in both declarations);
* every base/posix module that imports the checkout's ``tools``
  package at module or class level — the collection-blocking class —
  is listed (a module that gains such an import without joining the
  manifest fails here, before the runner ever runs);
* every listed module's source carries at least one checkout-content
  reference, so a listing cannot be fabricated for a module that
  does not touch the checkout;
* the parser refuses reasonless, duplicated, or malformed entries;
* MANIFEST.in actually ships the two scope contracts, and the
  release workflow runs the runner against the built sdist.
"""

from __future__ import annotations

import ast
import io
import json
import sys
import tarfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import sdist_scope as scope  # noqa: E402

TESTS_DIR = REPO_ROOT / "tests"

# Substrings that evidence a checkout-content reference in a module's
# source. Deliberately broad: this is an anti-fabrication tripwire
# for manifest entries, not the derivation itself (the derivation is
# the executed sdist-only run recorded for RR-12).
_CHECKOUT_TOKENS = (
    "examples/",
    "docs/",
    ".github/",
    "tools/",
    "from tools",
    "DEVELOPMENT_PLAN",
    "CITATION.cff",
    "VALIDATION.md",
    "mkdocs.yml",
    "pylock.toml",
    "_collect_cache",
    "test_layers",
    "installed distribution",
    "make_sample_report",
    "sample-report",
    "fixtures.json",
    "fixtures.csv",
    "workflows/",
    "USAGE.md",
    "marker_census.json",
)


def _manifest_entries() -> list[tuple[str, str]]:
    """Parse the shipped manifest with the runner's own parser."""
    return scope.parse_checkout_only((TESTS_DIR / "CHECKOUT_ONLY").read_text(encoding="utf-8"))


def _layers() -> tuple[list[str], list[str]]:
    """(in-scope, hf) module lists from the layer manifest."""
    return scope.layer_scope((TESTS_DIR / "test_layers.json").read_text(encoding="utf-8"))


def _top_level_tools_imports(path: Path) -> list[str]:
    """tools.* imports at module or class level (the layer semantics:
    what an import of the module itself would execute).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []

    def visit(body: list[ast.stmt]) -> None:
        """Visit."""
        for node in body:
            if isinstance(node, ast.Import):
                found.extend(a.name for a in node.names if a.name.split(".")[0] == "tools")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "tools":
                    found.append(node.module or "")
            elif isinstance(node, ast.ClassDef):
                visit(node.body)

    visit(tree.body)
    return found


class ManifestShapeTests(unittest.TestCase):
    """The manifest is a contract: sorted, reasoned, in-scope."""

    def test_manifest_parses_sorted_and_nonempty(self) -> None:
        """Test manifest parses sorted and nonempty."""
        entries = _manifest_entries()
        modules = [module for module, _reason in entries]
        self.assertEqual(modules, sorted(modules))
        self.assertGreater(len(modules), 0)

    def test_every_entry_is_a_shipped_layer_module(self) -> None:
        """Test every entry is a shipped layer module."""
        in_scope, hf = _layers()
        for module, _reason in _manifest_entries():
            self.assertTrue((TESTS_DIR / module).is_file(), module)
            self.assertTrue(module in in_scope or module in hf, module)

    def test_every_top_level_tools_importer_is_listed(self) -> None:
        """Test every top level tools importer is listed."""
        in_scope, _hf = _layers()
        listed = {module for module, _reason in _manifest_entries()}
        missing = []
        for module in in_scope:
            if module in listed:
                continue
            if _top_level_tools_imports(TESTS_DIR / module):
                missing.append(module)
        self.assertEqual(missing, [])

    def test_every_entry_references_checkout_content(self) -> None:
        """Test every entry references checkout content."""
        for module, _reason in _manifest_entries():
            source = (TESTS_DIR / module).read_text(encoding="utf-8")
            self.assertTrue(
                any(token in source for token in _CHECKOUT_TOKENS),
                f"{module} is listed but references no checkout content",
            )


class ParserTests(unittest.TestCase):
    """The runner's manifest parser fails closed."""

    def test_reasonless_line_rejected(self) -> None:
        """Test reasonless line rejected."""
        with self.assertRaises(scope.ScopeFailure):
            scope.parse_checkout_only("test_x.py\n")

    def test_duplicate_rejected(self) -> None:
        """Test duplicate rejected."""
        with self.assertRaises(scope.ScopeFailure):
            scope.parse_checkout_only("test_x.py | one\ntest_x.py | two\n")

    def test_non_module_name_rejected(self) -> None:
        """Test non module name rejected."""
        with self.assertRaises(scope.ScopeFailure):
            scope.parse_checkout_only("notes.txt | not a module\n")

    def test_empty_manifest_rejected(self) -> None:
        """Test empty manifest rejected."""
        with self.assertRaises(scope.ScopeFailure):
            scope.parse_checkout_only("# only comments\n")

    def test_layer_scope_rejects_malformed_json(self) -> None:
        """Test layer scope rejects malformed json."""
        with self.assertRaises(scope.ScopeFailure):
            scope.layer_scope("{not json")

    def test_backslash_module_rejected(self) -> None:
        """BH13-9: a backslash path is not a module name."""
        with self.assertRaises(scope.ScopeFailure):
            scope.parse_checkout_only("..\\test_a.py | windows path, not a module\n")

    def test_space_module_rejected(self) -> None:
        """BH13-9: a space-containing name is not a module name."""
        with self.assertRaises(scope.ScopeFailure):
            scope.parse_checkout_only("test a.py | not a module name\n")

    def test_layer_scope_rejects_dict_layer(self) -> None:
        """BH13-9: a layer given as an object is malformed, not a key list."""
        text = json.dumps({"layers": {"base": {"test_a.py": 1}, "posix": [], "hf": []}})
        with self.assertRaises(scope.ScopeFailure):
            scope.layer_scope(text)

    def test_layer_scope_rejects_string_layer(self) -> None:
        """BH13-9: a layer given as a string is not a list of modules."""
        text = json.dumps({"layers": {"base": "test_a.py", "posix": [], "hf": []}})
        with self.assertRaises(scope.ScopeFailure):
            scope.layer_scope(text)

    def test_layer_scope_rejects_int_elements(self) -> None:
        """BH13-9: layer elements must be module-name strings."""
        text = json.dumps({"layers": {"base": [1, 2], "posix": [], "hf": []}})
        with self.assertRaises(scope.ScopeFailure):
            scope.layer_scope(text)

    def test_layer_scope_accepts_genuine_shape(self) -> None:
        """Test layer scope accepts genuine shape."""
        text = json.dumps(
            {
                "layers": {
                    "base": ["test_b.py", "test_a.py"],
                    "posix": ["test_c.py"],
                    "hf": ["test_h.py"],
                }
            }
        )
        in_scope, hf = scope.layer_scope(text)
        self.assertEqual(in_scope, ["test_a.py", "test_b.py", "test_c.py"])
        self.assertEqual(hf, ["test_h.py"])


class RunnerGateTests(unittest.TestCase):
    """The runner refuses artifacts that do not carry the contract."""

    def test_missing_sdist_rejected(self) -> None:
        """Test missing sdist rejected."""
        with self.assertRaises(scope.ScopeFailure):
            scope.run_scope(
                REPO_ROOT / "dist" / "absent.tar.gz",
                REPO_ROOT,
                TESTS_DIR / "_w5_nowhere",
                "collect",
            )

    def test_sdist_without_checkout_only_rejected(self) -> None:
        """Test sdist without checkout only rejected."""
        import tempfile

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo("model_protocol_lab-0.2.0/tests/test_dummy.py")
            payload = b"# dummy\n"
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake.tar.gz"
            fake.write_bytes(buffer.getvalue())
            with self.assertRaises(scope.ScopeFailure) as caught:
                scope.run_scope(fake, REPO_ROOT, Path(tmp) / "work", "collect")
        self.assertIn("CHECKOUT_ONLY", str(caught.exception))

    def test_traversal_sdist_is_a_scope_failure(self) -> None:
        """BH13-8: a traversal member fails as ScopeFailure, not a traceback.

        The data filter refuses the member; run_scope wraps tarfile's
        refusal the way artifact_manifest.rebuild_check does, so the
        CLI reports a named failure instead of a raw traceback.
        """
        import tempfile

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo("../evil.py")
            payload = b"# evil\n"
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake.tar.gz"
            fake.write_bytes(buffer.getvalue())
            with self.assertRaises(scope.ScopeFailure) as caught:
                scope.run_scope(fake, REPO_ROOT, Path(tmp) / "work", "collect")
        self.assertIn("extraction refused", str(caught.exception))

    def test_non_gzip_sdist_is_a_scope_failure(self) -> None:
        """BH13-8: non-gzip input fails as ScopeFailure, not a traceback."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake.tar.gz"
            fake.write_bytes(b"this is not a gzip stream")
            with self.assertRaises(scope.ScopeFailure) as caught:
                scope.run_scope(fake, REPO_ROOT, Path(tmp) / "work", "collect")
        self.assertIn("extraction refused", str(caught.exception))

    def test_escaping_symlink_sdist_is_a_scope_failure(self) -> None:
        """BH13-8: an escaping symlink member fails as ScopeFailure."""
        import tempfile

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo("model_protocol_lab-0.2.0/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            archive.addfile(info)
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake.tar.gz"
            fake.write_bytes(buffer.getvalue())
            with self.assertRaises(scope.ScopeFailure) as caught:
                scope.run_scope(fake, REPO_ROOT, Path(tmp) / "work", "collect")
        self.assertIn("extraction refused", str(caught.exception))


class WiringTests(unittest.TestCase):
    """The scope contracts ship, and the release workflow runs the runner."""

    def test_manifest_in_ships_both_scope_contracts(self) -> None:
        """Test manifest in ships both scope contracts."""
        manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertIn("include tests/CHECKOUT_ONLY", manifest)
        self.assertIn("include tests/test_layers.json", manifest)

    def test_release_workflow_runs_the_scope_runner(self) -> None:
        """Test release workflow runs the scope runner."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        self.assertIn("tools/sdist_scope.py --sdist dist/*.tar.gz", workflow)

    def test_clean_env_drops_pythonpath_and_sets_tmpdir(self) -> None:
        """Test clean env drops pythonpath and sets tmpdir."""
        import os

        sentinel = object()
        previous = os.environ.get("PYTHONPATH", sentinel)
        previous_proxy = os.environ.get("https_proxy", sentinel)
        os.environ["PYTHONPATH"] = "/checkout"
        os.environ["https_proxy"] = "http://proxy.invalid:3128"
        try:
            env = scope._clean_env(Path("/work/tmp"))
        finally:
            if previous is sentinel:
                del os.environ["PYTHONPATH"]
            else:
                os.environ["PYTHONPATH"] = previous  # type: ignore[assignment]
            if previous_proxy is sentinel:
                del os.environ["https_proxy"]
            else:
                os.environ["https_proxy"] = previous_proxy  # type: ignore[assignment]
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertEqual(env["TMPDIR"], "/work/tmp")
        # The host's egress configuration is not checkout state: it
        # passes through, or the hashed installs cannot reach an index.
        self.assertEqual(env["https_proxy"], "http://proxy.invalid:3128")


if __name__ == "__main__":
    unittest.main()
