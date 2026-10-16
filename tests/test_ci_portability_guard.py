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
        # self.assertEqual(a, b) / self.assertNotEqual(a, b)
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


if __name__ == "__main__":
    unittest.main()
