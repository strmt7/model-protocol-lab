"""Cycle 85: README quick-start dependency audit.

The cycle-84 README fix changed the quick-start install line from
``python -m pip install -e ".[dev]"`` to
``python -m pip install -e ".[dev,hf]"`` because 9 test modules import
torch/transformers at module level, which the ``[dev]`` extra alone does
not provide. This audit pins the invariant the fix relies on: every
module-level third-party import in ``tests/`` must be provided by the
distributions named (directly or transitively) in the extras the README
install lines reference. A future test module that imports a new
third-party package without a declared home fails here instead of
failing at collection time on a fresh ``.[dev,hf]`` install.

METHOD
------
Pure stdlib (``ast``, ``tomllib``, ``importlib.metadata``); no network,
no subprocess, no environment reads beyond the installed distribution
metadata of the running interpreter. The allowed set is the recursive
``Requires-Dist`` closure over the extras named by the README's
``pip install -e ".[...]"`` lines -- so genuinely transitive providers
(e.g. ``numpy`` via ``torch``, ``tokenizers`` via ``transformers``) are
accepted without hard-coding their names. Intra-repo imports (the
``model_protocol_lab`` package itself and sibling ``tests`` modules) are
resolved by file location and skipped.

FINDINGS
--------
CLEAN. All module-level third-party imports across ``tests/`` resolve
to the ``[dev,hf]`` closure: torch/transformers/safetensors via ``hf``;
hypothesis/pytest/ruff/mypy/coverage via ``dev``; numpy via torch's
metadata; tokenizers via transformers' metadata.

Verified by running: 2026-10-16 (repo .venv, ``[dev,hf]`` installed).
"""

import ast
import importlib.metadata
import importlib.util
import re
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "tests"


def _normalize(name: str) -> str:
    """Normalize."""
    return re.sub(r"[-_.]+", "_", name).strip().lower()


def _requirement_name(requirement: str) -> str:
    """Distribution name out of a Requires-Dist entry, markers/specifiers dropped."""
    return _normalize(re.split(r"[;<>!=\s()~\[]", requirement, maxsplit=1)[0])


def _pyproject() -> dict:
    """Pyproject."""
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def _readme_install_extras() -> set[str]:
    """Extras named by the README's ``pip install -e ".[...]"`` lines."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    extras: set[str] = set()
    for match in re.finditer(r'pip install[^\n]*"\.\[([^\]]+)\]"', text):
        extras.update(_normalize(part) for part in match.group(1).split(","))
    return extras


def _declared_distributions() -> set[str]:
    """Direct requirements from install_requires plus the README-named extras."""
    project = _pyproject()["project"]
    extras = _readme_install_extras()
    declared: set[str] = set()
    for req in project.get("dependencies", []):
        declared.add(_requirement_name(req))
    optional = project.get("optional-dependencies", {})
    for extra in extras:
        for req in optional.get(extra, []):
            declared.add(_requirement_name(req))
    return declared


def _dependency_closure(declared: set[str]) -> set[str]:
    """Recursive Requires-Dist closure over locally installed metadata."""
    closure = set(declared)
    stack = list(declared)
    while stack:
        name = stack.pop()
        try:
            requires = importlib.metadata.requires(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for req in requires or ():
            dep = _requirement_name(req)
            if dep and dep not in closure:
                closure.add(dep)
                stack.append(dep)
    return closure


def _module_level_third_party_imports() -> dict[str, set[str]]:
    """Map test module name -> module-level third-party import roots."""
    stdlib = sys.stdlib_module_names
    found: dict[str, set[str]] = {}
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_bytes())
        roots: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
        third_party = set()
        pkg_dir = (ROOT / "model_protocol_lab").resolve()
        tools_dir = (ROOT / "tools").resolve()
        for root in roots:
            if root in stdlib:
                continue
            # Sibling test modules: unittest discover inserts the start dir
            # (tests/) into sys.path, so `import test_core` resolves there;
            # `import tests` / `from tests import ...` resolves to the tests
            # package itself.
            # The tools/ namespace package (dev CLIs) is also repo-local.
            sibling = TESTS_DIR / root
            if (
                (TESTS_DIR / f"{root}.py").exists()
                or (sibling.is_dir() and (sibling / "__init__.py").exists())
                or (root == TESTS_DIR.name and (TESTS_DIR / "__init__.py").exists())
            ):
                continue
            spec = importlib.util.find_spec(root)
            if spec is not None:
                if spec.origin is not None:
                    origin = Path(spec.origin).resolve()
                    # The package itself. (The repo .venv lives under ROOT too,
                    # so match the package directory specifically, not ROOT.)
                    if pkg_dir in origin.parents:
                        continue
                elif spec.submodule_search_locations:
                    # Namespace package (e.g. tools/ with no __init__.py):
                    # local if its locations resolve under the repo.
                    locations = [Path(p).resolve() for p in spec.submodule_search_locations]
                    if any(tools_dir == loc or tools_dir in loc.parents for loc in locations):
                        continue
            third_party.add(root)
        found[path.name] = third_party
    return found


class ReadmeInstallLineTest(unittest.TestCase):
    """Readme install line test."""

    def test_install_extras_exist_in_pyproject(self) -> None:
        """Test install extras exist in pyproject."""
        extras = _readme_install_extras()
        self.assertTrue(extras, 'no pip install -e ".[...]" line found in README')
        declared_extras = {_normalize(e) for e in _pyproject()["project"]["optional-dependencies"]}
        for extra in sorted(extras):
            self.assertIn(extra, declared_extras, f"README names unknown extra [{extra}]")

    def test_hf_extra_provides_torch_and_transformers(self) -> None:
        """The premise of the cycle-84 fix: [hf] must name the ML stack."""
        hf_reqs = _pyproject()["project"]["optional-dependencies"]["hf"]
        names = {_requirement_name(r) for r in hf_reqs}
        self.assertIn("torch", names)
        self.assertIn("transformers", names)

    def test_dev_extra_still_present(self) -> None:
        """Test dev extra still present."""
        dev_reqs = _pyproject()["project"]["optional-dependencies"]["dev"]
        names = {_requirement_name(r) for r in dev_reqs}
        self.assertIn("pytest", names)
        self.assertIn("hypothesis", names)


class TestImportsCoveredTest(unittest.TestCase):
    """Test imports covered test."""

    def test_module_level_imports_within_declared_closure(self) -> None:
        """Test module level imports within declared closure."""
        closure = _dependency_closure(_declared_distributions())
        dist_for_module = importlib.metadata.packages_distributions()
        uncovered: list[str] = []
        for module, roots in _module_level_third_party_imports().items():
            for root in sorted(roots):
                providers = {_normalize(d) for d in dist_for_module.get(root, set())}
                if not providers:
                    uncovered.append(f"{module}: no installed distribution provides {root!r}")
                elif providers.isdisjoint(closure):
                    uncovered.append(
                        f"{module}: {root!r} provided by {sorted(providers)}, "
                        "outside the [dev,hf] dependency closure"
                    )
        self.assertEqual([], uncovered, "\n".join(uncovered))

    def test_torch_transformers_importers_are_covered(self) -> None:
        """Regression pin for the original failure: torch/transformers must be
        importable by every test module that needs them at module level.
        """
        importers = [
            module
            for module, roots in _module_level_third_party_imports().items()
            if roots & {"torch", "transformers"}
        ]
        self.assertTrue(importers, "expected at least one torch/transformers importer")
        for module in importers:
            self.assertTrue(
                importlib.util.find_spec("torch") is not None
                and importlib.util.find_spec("transformers") is not None,
                f"{module} needs torch/transformers but they are not importable",
            )


if __name__ == "__main__":
    unittest.main()
