"""Standing guard: no exact-version assertions that break CI matrix jobs.

Regression guard for the cycle-79/80 CI red: three tests pinned the local
environment exactly (``sys.version_info[:2] == (3, 12)``,
``torch.__version__ == "2.14.1+cpu"`` with the local build tag,
``transformers.__version__ == "5.19.0"``) and failed on CI's other Python
versions, the CUDA wheel, and newer dependency resolves. A fourth shelled
out to a ``bandit`` binary that only exists in the security-audit job.

Rules enforced here (AST scan over every test module):
  * no ``==``/``!=`` comparison between ``sys.version_info[...]`` and a
    literal version tuple -- the interpreter version must be checked
    against pyproject's ``requires-python`` range instead;
  * no ``==``/``!=`` comparison of a bare ``torch.__version__`` /
    ``transformers.__version__`` attribute against a string literal --
    compare a build-tag-agnostic base (``.split("+")[0]``) or check
    membership in the pyproject pin's SpecifierSet instead.

This file itself is excluded from the scan.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
_GUARD_NAME = Path(__file__).name
_VERSIONED_MODULES = ("torch", "transformers")


def _is_version_info_subscript(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "version_info"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "sys"
    )


def _is_bare_dunder_version(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "__version__"
        and isinstance(node.value, ast.Name)
        and node.value.id in _VERSIONED_MODULES
    )


def _is_const_seq(node: ast.AST) -> bool:
    return isinstance(node, (ast.Tuple, ast.List)) and all(
        isinstance(elt, ast.Constant) for elt in node.elts
    )


def _is_const_str(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


class _ExactVersionVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []

    def _check_pair(self, a: ast.AST, b: ast.AST, lineno: int) -> None:
        for operand, other in ((a, b), (b, a)):
            if _is_version_info_subscript(operand) and _is_const_seq(other):
                self.hits.append((lineno, "exact sys.version_info tuple comparison"))
            if _is_bare_dunder_version(operand) and _is_const_str(other):
                self.hits.append((lineno, "exact torch/transformers __version__ comparison"))

    def visit_Compare(self, node: ast.Compare) -> None:
        if any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
            operands = [node.left, *node.comparators]
            for i, operand in enumerate(operands):
                for other in operands[:i] + operands[i + 1 :]:
                    self._check_pair(operand, other, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # The assertEqual / assertNotEqual spellings, as attribute calls.
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("assertEqual", "assertNotEqual")
            and len(node.args) == 2
        ):
            self._check_pair(node.args[0], node.args[1], node.lineno)
        self.generic_visit(node)


def _scan(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = _ExactVersionVisitor()
    visitor.visit(tree)
    return [f"{path.name}:{lineno}: {kind}" for lineno, kind in visitor.hits]


class TestNoExactVersionPins(unittest.TestCase):
    def test_no_exact_version_assertions(self) -> None:
        hits: list[str] = []
        for path in sorted(TESTS_DIR.glob("test_*.py")):
            if path.name == _GUARD_NAME:
                continue
            hits.extend(_scan(path))
        self.assertEqual(
            hits,
            [],
            "exact-version assertions break CI matrix jobs; derive from "
            "pyproject pins instead:\n" + "\n".join(hits),
        )


def _is_lone_surrogate(char: str) -> bool:
    return 0xD800 <= ord(char) <= 0xDFFF


def _docstring_surrogate_hits(path: Path) -> list[str]:
    """Find docstrings containing lone-surrogate characters.

    Regression guard for the cycle-92 CI red: CPython 3.13's compile()
    encodes ``__doc__`` as strict UTF-8, so a module whose docstring
    holds a real lone-surrogate character (e.g. a single-backslash
    ``'\\ud800'`` escape in the docstring source) fails at import time
    with ``UnicodeEncodeError`` -- while 3.11/3.12 accept it, which is
    why the local suite stayed green. Docstrings must show the escape
    as literal text (double the backslash); surrogate *test data*
    belongs in runtime-constructed strings (``chr(0xD800)``), never in
    docstring prose.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []

    def _check(label: str, lineno: int, doc: str | None) -> None:
        if doc and any(_is_lone_surrogate(c) for c in doc):
            hits.append(f"{path.name}:{lineno}: {label} docstring holds a lone surrogate")

    _check("module", 1, ast.get_docstring(tree))
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            _check(f"{type(node).__name__} {node.name}", node.lineno, ast.get_docstring(node))
    return hits


class TestNoDocstringSurrogates(unittest.TestCase):
    def test_no_docstring_lone_surrogates(self) -> None:
        hits: list[str] = []
        for path in sorted(TESTS_DIR.glob("test_*.py")):
            if path.name == _GUARD_NAME:
                continue
            hits.extend(_docstring_surrogate_hits(path))
        self.assertEqual(
            hits,
            [],
            "lone surrogates in docstrings make the module unimportable on "
            "Python 3.13 (compile() encodes __doc__ as strict UTF-8); "
            "double the backslash so the escape renders as text:\n" + "\n".join(hits),
        )


class TestPublicApiDocstrings(unittest.TestCase):
    """Every public class/function in the shipped package needs a docstring.

    A 0.1.0 release documents its public API; an undocumented public name is
    a docs regression. Only top-level defs are checked (methods inherit their
    class contract); dunder and private names are excluded.
    """

    def test_all_public_api_has_docstrings(self) -> None:
        pkg = TESTS_DIR.parent / "model_protocol_lab"
        missing: list[str] = []
        for path in sorted(pkg.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if node.name.startswith("_"):
                    continue
                if not ast.get_docstring(node):
                    missing.append(f"{path.name}:{node.lineno} {node.name}")
        self.assertEqual(
            missing,
            [],
            "public API without docstrings (release docs gap):\n" + "\n".join(missing),
        )


class TestMypyStrictPinned(unittest.TestCase):
    """pyproject must keep mypy strict mode enabled.

    Cycle 104 annotated the whole package for --strict (37 errors fixed);
    flipping strict back to false would silently re-admit untyped defs and
    bare generics. The CI lint job runs `mypy model_protocol_lab`, which
    reads this setting.
    """

    def test_mypy_strict_enabled(self) -> None:
        import tomllib

        pyproject = TESTS_DIR.parent / "pyproject.toml"
        with open(pyproject, "rb") as handle:
            config = tomllib.load(handle)
        self.assertTrue(
            config.get("tool", {}).get("mypy", {}).get("strict") is True,
            "mypy strict mode must stay enabled in pyproject.toml",
        )


class TestDualRunnerContribution(unittest.TestCase):
    """Every test file must contribute tests under BOTH runners.

    Cycle 106 found 9 pytest-style files (2533 test executions) that CI's
    `unittest discover` silently skipped; cycle 107/108 converted 5 more.
    VALIDATION.md documents both runners, so a file that collects zero tests
    under either runner is a coverage lie. This guard fails if any
    tests/test_*.py file yields zero tests from unittest's TestLoader or from
    pytest's collector.

    Cycle 113 note: test_cycle66_fixture_loading_audit.py's TestCase
    conversion is being re-attempted as a diagnostic experiment (cycle 112
    added CI failure-artifact upload so a CI-only failure is now
    diagnosable). The exemption below is empty while the experiment runs.
    """

    # Files exempt from the unittest-contribution check, with reasons.
    # Currently empty: the cycle-108 test_cycle66 exemption was lifted for
    # the cycle-113 diagnostic re-attempt (CI failure artifacts now make a
    # CI-only failure diagnosable). Reinstate if the experiment fails.
    _UNITTEST_EXEMPT: dict[str, str] = {}

    def test_every_test_file_collected_by_unittest(self) -> None:
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import unittest

        loader = unittest.TestLoader()
        zero: list[str] = []
        for path in sorted(TESTS_DIR.glob("test_*.py")):
            if path.name == "test_ci_portability_guard.py":
                continue  # self-reference; this file is TestCase by construction
            if path.name in self._UNITTEST_EXEMPT:
                continue  # documented exemption; see class docstring
            module = f"tests.{path.stem}"
            try:
                count = loader.loadTestsFromName(module).countTestCases()
            except Exception as exc:  # noqa: BLE001 -- report as zero, don't crash guard
                zero.append(f"{path.name} (load error: {exc})")
                continue
            if count == 0:
                zero.append(path.name)
        self.assertEqual(
            zero,
            [],
            "test files contributing zero tests under unittest (CI gate skips them):\n"
            + "\n".join(zero),
        )

    def test_every_test_file_collected_by_pytest(self) -> None:
        # The collection subprocess is shared with test_cycle4's README
        # count pin via tests/_collect_cache.py (one ~30-60s spawn per
        # session instead of two).
        from tests._collect_cache import pytest_collect_counts

        try:
            counts, _ = pytest_collect_counts(TESTS_DIR.parent)
        except RuntimeError as exc:
            self.fail(str(exc))
        zero = [p.name for p in sorted(TESTS_DIR.glob("test_*.py")) if counts.get(p.name, 0) == 0]
        self.assertEqual(
            zero,
            [],
            "test files contributing zero tests under pytest:\n" + "\n".join(zero),
        )


class CollectCacheTests(unittest.TestCase):
    def test_memoized_within_process(self):
        from tests._collect_cache import pytest_collect_counts

        first = pytest_collect_counts(TESTS_DIR.parent)
        second = pytest_collect_counts(TESTS_DIR.parent)
        self.assertIs(first, second)
        counts, total = first
        self.assertGreater(total, 0)
        self.assertGreater(sum(counts.values()), 0)


if __name__ == "__main__":
    unittest.main()
