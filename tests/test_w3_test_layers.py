"""W3-3 contract: the test-layer partition (RR-08, collection half).

The suite's modules are partitioned in tests/test_layers.json into
base (collects and runs on .[dev] alone), hf (needs .[dev,hf]) and
posix (needs a POSIX platform). This contract keeps the manifest
honest:

* it re-derives the partition from an AST import scan on every run
  -- an import is top-level iff reachable from the module body
  without crossing a function/class/lambda boundary (module-level
  try:/if: counts as top-level, conservatively); hf = a heavy name
  (torch/transformers/tokenizers/numpy) at top level, posix = only
  resource at top level, and the requirement is transitive through
  top-level test-tree imports (a base module importing an hf test
  module is hf -- test_cycle244_fixwave8 -> test_cycle195_choiceaudit
  is the standing example);
* it pins the POSIX guards and the hf pytest marks the manifest
  records;
* it productizes the simulated-absence collection: a subprocess
  child (tests/_blocked_collect.py) blocks torch/numpy/
  transformers/tokenizers/resource and must collect the base layer
  with zero errors, and -- as the negative control -- must error on
  exactly the hf modules when collecting the excluded set.

What the simulation proves: base modules *collect* when the blocked
modules are unimportable (with find_spec probes answered None, as in
a real absence). What it does not prove: that base tests *pass* in
that environment (function-level HF tests rely on their existing
skipUnless guards; the W3-6 Windows CI cell is the honest check for
execution), nor anything about other platform absences.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
MANIFEST_PATH = TESTS_DIR / "test_layers.json"
TRACKED = {"torch", "numpy", "transformers", "tokenizers", "resource"}
HEAVY = {"torch", "numpy", "transformers", "tokenizers"}
CONTROL = "test_workflows.py"

_CHILD_RESULTS: dict[str, subprocess.CompletedProcess[str]] = {}


def _load_manifest() -> dict:
    """Load manifest."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _classify(path: Path) -> tuple[set[str], set[str], set[str]]:
    """Return (top-level tracked, function-level tracked, test-tree edges)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    top: set[str] = set()
    func: set[str] = set()
    edges: set[str] = set()

    def visit(node: ast.AST, nested: bool) -> None:
        """Visit."""
        for child in ast.iter_child_nodes(node):
            inner = nested or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            )
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                if isinstance(child, ast.Import):
                    names = [a.name.split(".")[0] for a in child.names]
                    if not inner:
                        for alias in child.names:
                            if alias.name.startswith("tests.test_"):
                                edges.add(alias.name.split(".")[1] + ".py")
                else:
                    names = [child.module.split(".")[0]] if child.module else []
                    if not inner and child.module:
                        if child.module.startswith("tests.test_"):
                            edges.add(child.module.split(".")[1] + ".py")
                        elif child.module == "tests":
                            for alias in child.names:
                                if alias.name.startswith("test_"):
                                    edges.add(alias.name + ".py")
                        elif child.level and child.module.startswith("test_"):
                            edges.add(child.module + ".py")
                for name in names:
                    if name in TRACKED:
                        (func if inner else top).add(name)
            visit(child, inner)

    visit(tree, False)
    return top, func - top, edges


def _derive_layers() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Re-derive the partition and func_level map from the live tree."""
    scanned = {p.name: _classify(p) for p in sorted(TESTS_DIR.glob("test_*.py"))}
    label: dict[str, str] = {}
    for name, (top, _func, _edges) in scanned.items():
        if top & HEAVY:
            label[name] = "hf"
        elif top == {"resource"}:
            label[name] = "posix"
        else:
            label[name] = "base"
    changed = True
    while changed:
        changed = False
        for name, (_top, _func, edges) in scanned.items():
            if label[name] != "base":
                continue
            targets = {label[e] for e in edges if e in label}
            if "hf" in targets:
                label[name] = "hf"
                changed = True
            elif "posix" in targets:
                label[name] = "posix"
                changed = True
    layers: dict[str, list[str]] = {"base": [], "hf": [], "posix": []}
    func_level: dict[str, list[str]] = {}
    for name in sorted(scanned):
        layers[label[name]].append(name)
        if label[name] == "base" and scanned[name][1]:
            func_level[name] = sorted(scanned[name][1])
    return layers, func_level


def _run_blocked_child(layer: str) -> subprocess.CompletedProcess[str]:
    """Run the blocked-collection child once per layer (memoized)."""
    if layer not in _CHILD_RESULTS:
        _CHILD_RESULTS[layer] = subprocess.run(
            [sys.executable, str(TESTS_DIR / "_blocked_collect.py"), "--layer", layer],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    return _CHILD_RESULTS[layer]


class ManifestContractTests(unittest.TestCase):
    """The checked-in manifest must equal the re-derived partition."""

    def test_partition_matches_disk_and_counts(self):
        """Test partition matches disk and counts."""
        manifest = _load_manifest()
        layers = manifest["layers"]
        on_disk = {p.name for p in TESTS_DIR.glob("test_*.py")}
        union = set(layers["base"]) | set(layers["hf"]) | set(layers["posix"])
        self.assertEqual(
            union,
            on_disk,
            "manifest partition != on-disk test modules; a new test file must be "
            "added to tests/test_layers.json in the same push (base unless it "
            "imports torch/transformers/tokenizers/numpy/resource at top level)",
        )
        total = sum(len(v) for v in layers.values())
        self.assertEqual(total, len(on_disk), "layers overlap or miss a module")
        counts = manifest["module_counts"]
        self.assertEqual(counts["total"], len(on_disk))
        for key in ("base", "hf", "posix"):
            self.assertEqual(counts[key], len(layers[key]), key)
        self.assertEqual(set(manifest["tracked_imports"]), TRACKED)
        self.assertEqual(
            set(manifest["blocked_in_simulation"]),
            {"torch", "numpy", "transformers", "tokenizers", "resource"},
        )

    def test_manifest_matches_rescan(self):
        """Test manifest matches rescan."""
        manifest = _load_manifest()
        layers, func_level = _derive_layers()
        self.assertEqual(
            manifest["layers"],
            layers,
            "manifest drifted from the AST import scan; re-derive the layer of "
            "the differing modules (see this file's docstring for the rules) "
            "and update tests/test_layers.json in the same push",
        )
        self.assertEqual(manifest["func_level"], func_level)

    def test_posix_guards_present(self):
        """Test posix guards present."""
        manifest = _load_manifest()

        def decorator_text(path: Path, class_name: str, method_name: str | None) -> str:
            """Decorator text."""
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    targets = (
                        [node]
                        if method_name is None
                        else [
                            n
                            for n in node.body
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and n.name == method_name
                        ]
                    )
                    text = ""
                    for target in targets:
                        for dec in target.decorator_list:
                            segment = ast.get_source_segment(path.read_text(encoding="utf-8"), dec)
                            text += (segment or "") + "\n"
                    return text
            return ""

        for filename, classes in manifest["posix_classes"].items():
            for class_name in classes:
                text = decorator_text(TESTS_DIR / filename, class_name, None)
                self.assertIn("win32", text, f"{filename}:{class_name} lost its POSIX guard")
                self.assertIn("resource", text, f"{filename}:{class_name} guard lost its reason")
        for filename, methods in manifest["posix_methods"].items():
            for dotted in methods:
                needle = "mkfifo" if "fifo" in dotted else "win32"
                class_name, method_name = dotted.split(".")
                text = decorator_text(TESTS_DIR / filename, class_name, method_name)
                self.assertIn(needle, text, f"{filename}:{dotted} lost its POSIX guard")

    def test_geteuid_root_guards_are_attribute_guarded(self):
        """Regression (W3-6, second Windows portable run): the two
        root-skip decorators in cycle72/cycle83 evaluated
        os.geteuid() at class-definition time; the attribute does
        not exist on Windows, so both base modules errored during
        Windows collection (9181 collected, 2 errors) while every
        other base module collected cleanly. The guards must test
        hasattr(os, "geteuid") before evaluating it."""
        for filename in (
            "test_cycle72_doctor_integration_audit.py",
            "test_cycle83_cli_wiring_audit.py",
        ):
            path = TESTS_DIR / filename
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            guarded = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    for dec in node.decorator_list:
                        segment = ast.get_source_segment(source, dec) or ""
                        if "geteuid" in segment:
                            self.assertIn(
                                "hasattr", segment, f"{filename}: unguarded geteuid decorator"
                            )
                            guarded += 1
            self.assertEqual(guarded, 1, f"{filename}: expected exactly one geteuid root guard")

    def test_hf_marks_and_marker_registration(self):
        """Test hf marks and marker registration."""
        import tomllib

        manifest = _load_manifest()
        for name in manifest["layers"]["hf"]:
            source = (TESTS_DIR / name).read_text(encoding="utf-8")
            self.assertIn("pytestmark = pytest.mark.hf", source, name)
        with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
            config = tomllib.load(handle)
        markers = config["tool"]["pytest"]["ini_options"]["markers"]
        self.assertTrue(any(m.startswith("hf:") for m in markers), markers)
        self.assertTrue(any(m.startswith("posix:") for m in markers), markers)

    def test_layer_paths_tool_agrees(self):
        """Test layer paths tool agrees."""
        manifest = _load_manifest()
        layers = manifest["layers"]
        expectations = {
            "base": sorted(layers["base"]),
            "hf": sorted(layers["hf"]),
            "posix": sorted(layers["posix"]),
            "excluded": sorted(layers["hf"] + layers["posix"]),
            "all": sorted(n for members in layers.values() for n in members),
        }
        for layer, names in expectations.items():
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "tools" / "layer_paths.py"), "--layer", layer],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
            self.assertEqual(proc.stdout.split(), [f"tests/{n}" for n in names], layer)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "layer_paths.py"),
                    "--layer",
                    layer,
                    "--format",
                    "modules",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
            self.assertEqual(
                proc.stdout.split(),
                [f"tests.{Path(n).stem}" for n in names],
                layer,
            )

    def test_layer_paths_output_is_byte_exact_under_translating_stdout(self):
        """Regression (W3-6, first Windows portable run): the tool's
        output feeds shell command substitution, so it must be
        byte-exact even when sys.stdout is a Windows text-mode
        stream that translates newlines to CRLF on write(). The
        pre-fix tool printed through write(), and pytest on Windows
        rejected every printed path as not found (a carriage return
        glued to each name). The tool must write bytes to the raw
        buffer instead; this fake reproduces the translation so the
        defect is exercised on any host."""

        class _TranslatingStdout:
            """A Windows text-mode stdout stand-in: write() encodes
            with CRLF translation; .buffer is the raw byte sink."""

            def __init__(self) -> None:
                self.buffer = io.BytesIO()

            def write(self, text: str) -> int:
                self.buffer.write(text.replace("\n", "\r\n").encode("utf-8"))
                return len(text)

            def flush(self) -> None:
                pass

        spec = importlib.util.spec_from_file_location(
            "layer_paths_under_test", REPO_ROOT / "tools" / "layer_paths.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fake = _TranslatingStdout()
        with mock.patch.object(sys, "stdout", fake):
            rc = module.main(["--layer", "base"])
        self.assertEqual(rc, 0)
        raw = fake.buffer.getvalue()
        self.assertNotIn(b"\r", raw)
        manifest = _load_manifest()
        self.assertEqual(
            raw.decode("utf-8").split(),
            [f"tests/{n}" for n in sorted(manifest["layers"]["base"])],
        )


@unittest.skipIf(
    os.environ.get("MPL_BLOCKED_COLLECT_CHILD"),
    "never re-enter the blocked simulation from inside itself",
)
class BlockedCollectionTests(unittest.TestCase):
    """The simulated-absence collection, run in a fresh interpreter."""

    def test_negative_control_errors_exactly_the_hf_modules(self):
        """Test negative control errors exactly the hf modules."""
        manifest = _load_manifest()
        proc = _run_blocked_child("excluded")
        output = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0, output[-1000:])
        errored = set(re.findall(r"^ERROR tests/(test_\S+\.py)", output, flags=re.M))
        self.assertEqual(
            errored,
            set(manifest["layers"]["hf"]),
            "the blocker must error on exactly the hf modules; if an hf "
            "module's import became guarded, move it in the manifest "
            "instead of weakening this control",
        )
        for collected in (CONTROL, *manifest["layers"]["posix"]):
            self.assertIn(f"tests/{collected}::", output, collected)
            self.assertNotIn(f"ERROR tests/{collected}", output, collected)

    def test_base_layer_collects_with_zero_errors(self):
        """Test base layer collects with zero errors."""
        manifest = _load_manifest()
        proc = _run_blocked_child("base")
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, output[-2000:])
        self.assertNotIn("ERROR", output)
        match = re.search(r"(\d+) tests collected", output)
        self.assertIsNotNone(match, output[-500:])
        from tests._collect_cache import pytest_collect_counts

        counts, total = pytest_collect_counts(REPO_ROOT)
        expected = total
        for name in manifest["layers"]["hf"] + manifest["layers"]["posix"]:
            expected -= counts.get(name, 0)
        self.assertEqual(
            int(match.group(1)),
            expected,
            "base-layer collected count must equal the full collection "
            "minus the hf/posix per-file counts (derived, never hardcoded)",
        )


if __name__ == "__main__":
    unittest.main()
