"""W7 registry pins.

Pre-registry invariants (W7-0): the section tables the typed
registry is built from — ``SECTION_DIFF_KEYS`` /
``UNDISPATCHED_SECTIONS`` in diffreports — cover exactly the
section-key universe reporting admits (``_REPORT_SECTION_KEYS``),
disjointly, and every dispatched key resolves its mirror validator
and its differ by the tree's naming conventions (with the measured
exceptions below). W7-1 replaces the convention-based resolution
with the registry's explicit dotted paths; these invariants are
what the registry must preserve.
"""

from __future__ import annotations

import inspect
import re
import unittest

from model_protocol_lab import diffreports, reporting

# Measured naming exceptions at the W7 baseline (every other
# dispatched key follows diff_<diff_key> / _valid_<diff_key>):
# the budget differ takes whole results dicts and validates by
# recomputation (_budget_side), and three sections keep the
# instrument module's fused spelling.
_DIFFER_EXCEPTIONS = {
    "budget": "diff_budgets",
    "frontier": "diff_frontiers",
    "batch_invariance": "diff_batchinv",
    "canonical_form": "diff_canonicalform",
    "choice_audit": "diff_choiceaudit",
}
_VALIDATOR_EXCEPTIONS = {
    "budget": "_budget_side",
    "batch_invariance": "_valid_batchinv",
    "canonical_form": "_valid_canonicalform",
    "choice_audit": "_valid_choiceaudit",
}


def differ_name(diff_key: str) -> str:
    return _DIFFER_EXCEPTIONS.get(diff_key, f"diff_{diff_key}")


def validator_name(diff_key: str) -> str:
    return _VALIDATOR_EXCEPTIONS.get(diff_key, f"_valid_{diff_key}")


class RegistryTableInvariantTests(unittest.TestCase):
    """The dispatched/undispatched tables cover the universe once."""

    def test_universe_is_65_and_matches_reporting(self) -> None:
        dispatched = set(diffreports.SECTION_DIFF_KEYS)
        undispatched = set(diffreports.UNDISPATCHED_SECTIONS)
        self.assertEqual(len(dispatched), 57)
        self.assertEqual(len(undispatched), 8)
        self.assertFalse(dispatched & undispatched)
        self.assertEqual(dispatched | undispatched, set(reporting._REPORT_SECTION_KEYS))

    def test_diff_keys_are_the_pair_projection(self) -> None:
        self.assertEqual(diffreports.SECTION_DIFF_KEYS, dict(diffreports._SECTION_PAIRS))
        self.assertEqual(len(diffreports._SECTION_PAIRS), len(diffreports.SECTION_DIFF_KEYS))

    def test_undispatched_reasons_are_nonempty(self) -> None:
        for key, reason in diffreports.UNDISPATCHED_SECTIONS.items():
            self.assertTrue(reason.strip(), f"empty undispatched reason: {key}")


class RegistryResolutionTests(unittest.TestCase):
    """Every dispatched key resolves its validator and differ."""

    def test_validators_resolve(self) -> None:
        for diff_key in diffreports.SECTION_DIFF_KEYS.values():
            name = validator_name(diff_key)
            target = getattr(diffreports, name, None)
            self.assertTrue(callable(target), f"{diff_key}: {name} not callable")

    def test_differs_resolve_and_are_dispatched(self) -> None:
        source = inspect.getsource(diffreports)
        for diff_key in diffreports.SECTION_DIFF_KEYS.values():
            name = differ_name(diff_key)
            target = getattr(diffreports, name, None)
            self.assertTrue(callable(target), f"{diff_key}: {name} not callable")
            # Dispatch reachability: the differ's name is called
            # somewhere in the module (diff_reports or a sibling
            # entry point), so a rename cannot silently orphan it.
            self.assertTrue(
                re.search(rf"\b{re.escape(name)}\(", source),
                f"{diff_key}: {name} is never called in diffreports",
            )


if __name__ == "__main__":
    unittest.main()
