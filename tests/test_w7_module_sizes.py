"""W7-2 module size pins.

Each split family module has a hard line budget derived from the
W7 re-census (lane A) cluster masses: budget = the family's
measured def-lines x 1.15 + 400 lines for imports, docstrings and
spacing. A module that outgrows its budget must split its largest
cluster out (the census's rule) instead of silently re-becoming a
monolith. Facades get a flat budget: they carry imports and
re-exports only, never implementation.
"""

from __future__ import annotations

import unittest
from pathlib import Path

PKG = Path(__file__).parent.parent / "model_protocol_lab"

# module -> (def-lines measured at the W7 baseline, budget)
FAMILY_BUDGETS: dict[str, tuple[int, int]] = {
    "reporting_core": (2786, 3604),
    "reporting_text": (2748, 3560),
    "reporting_forms": (2136, 2856),
    "reporting_probes": (2778, 3595),
    "reporting_runlevel": (3833, 4808),
    "experiments_core": (2091, 2804),
    "experiments_text": (2549, 3331),
    "experiments_forms": (2840, 3665),
    "experiments_probes": (4365, 5419),
    "experiments_runlevel": (4260, 5299),
}
# Facade budgets: lines measured at split time + 300, rounded to the
# nearest 50 (reporting: 311 -> 600; experiments: 719 -> 1000).
FACADE_BUDGETS: dict[str, int] = {
    "reporting": 600,
    "experiments": 1000,
}


def _lines(module: str) -> int:
    return len((PKG / f"{module}.py").read_text(encoding="utf-8").splitlines())


class ModuleSizeTests(unittest.TestCase):
    """Split modules stay within their census-derived budgets."""

    def test_family_budgets(self) -> None:
        for module, (_measured, budget) in FAMILY_BUDGETS.items():
            self.assertLessEqual(
                _lines(module),
                budget,
                f"{module} exceeds its budget of {budget} lines; "
                "split its largest cluster out instead of growing it",
            )

    def test_facade_budgets(self) -> None:
        for module, budget in FACADE_BUDGETS.items():
            self.assertLessEqual(
                _lines(module),
                budget,
                f"facade {module} exceeds {budget} lines; implementation "
                "belongs in a family module, not the facade",
            )


if __name__ == "__main__":
    unittest.main()
