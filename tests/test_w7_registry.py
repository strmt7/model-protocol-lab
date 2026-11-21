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

import importlib
import inspect
import re
import unittest
from pathlib import Path

from model_protocol_lab import diffreports, registry, reporting

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


def _resolve(dotted: str) -> object:
    module_name, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


class RegistryContentTests(unittest.TestCase):
    """The registry table itself: coverage, resolution, anchors."""

    def test_registry_covers_the_universe_exactly_once(self) -> None:
        keys = [i.key for i in registry.INSTRUMENTS]
        self.assertEqual(len(keys), 65)
        self.assertEqual(len(set(keys)), 65)
        self.assertEqual(set(keys), set(reporting._REPORT_SECTION_KEYS))
        self.assertEqual(set(registry.INSTRUMENT_BY_KEY), set(keys))

    def test_derived_views_match_the_registry(self) -> None:
        pairs = [(i.key, i.diff_key) for i in registry.INSTRUMENTS if i.diff_key is not None]
        self.assertEqual([tuple(p) for p in diffreports._SECTION_PAIRS], pairs)
        self.assertEqual(diffreports.SECTION_DIFF_KEYS, dict(pairs))
        reasons = {
            i.key: i.undispatched_reason
            for i in registry.INSTRUMENTS
            if i.undispatched_reason is not None
        }
        self.assertEqual(diffreports.UNDISPATCHED_SECTIONS, reasons)

    def test_exactly_one_of_diff_key_or_reason(self) -> None:
        for i in registry.INSTRUMENTS:
            if i.diff_key is None:
                self.assertIsNotNone(i.undispatched_reason, i.key)
                self.assertIsNone(i.differ, i.key)
                self.assertIsNone(i.validator, i.key)
            else:
                self.assertIsNone(i.undispatched_reason, i.key)
                self.assertIsNotNone(i.differ, i.key)
                self.assertIsNotNone(i.validator, i.key)

    def test_every_dotted_path_resolves_to_a_callable(self) -> None:
        for i in registry.INSTRUMENTS:
            for path in (i.producer, i.stage, i.differ, i.validator):
                if path is not None:
                    self.assertTrue(callable(_resolve(path)), f"{i.key}: {path}")

    def test_docs_anchors_exist_as_usage_headings(self) -> None:
        usage = (Path(__file__).parent.parent / "docs" / "USAGE.md").read_text(encoding="utf-8")
        headings = {line.lstrip("#").strip() for line in usage.splitlines() if line.startswith("#")}
        for i in registry.INSTRUMENTS:
            self.assertIn(i.docs_anchor, headings, i.key)

    def test_cli_flags_exist_in_the_cli_parser(self) -> None:
        cli_source = (Path(__file__).parent.parent / "model_protocol_lab" / "cli.py").read_text(
            encoding="utf-8"
        )
        for i in registry.INSTRUMENTS:
            if i.cli_flag is not None:
                self.assertIn(f'"{i.cli_flag}"', cli_source, i.key)

    def test_projection_rules_are_complete(self) -> None:
        for i in registry.INSTRUMENTS:
            self.assertTrue(i.projection_reason.strip(), i.key)
            if i.projection is registry.ProjectionKind.STRIP_FIELDS:
                self.assertTrue(i.projection_fields, i.key)
            else:
                self.assertEqual(i.projection_fields, frozenset(), i.key)
            if i.key == "serving_backends":
                self.assertEqual(
                    i.config_allowlist,
                    frozenset(reporting._SERVING_CONFIG_ALLOWLIST),
                )
            else:
                self.assertIsNone(i.config_allowlist, i.key)

    def test_answer_text_rows_match_the_shipped_section_list(self) -> None:
        rows = {
            i.key for i in registry.INSTRUMENTS if i.projection_fields == frozenset({"answer_text"})
        }
        self.assertEqual(rows, set(reporting._ANSWER_TEXT_SECTIONS))


class RegistryProjectionBehaviourTests(unittest.TestCase):
    """The registry's projection rows describe _public_view's behaviour.

    Each strip rule is exercised against a synthetic section payload
    carrying sentinel values in the named fields; verbatim rows must
    pass a sentinel field through untouched.
    """

    def _view(self, key: str, section: object) -> dict:
        doc = reporting._public_view({key: section}, False)
        return doc[key]

    def test_answer_text_is_stripped(self) -> None:
        for i in registry.INSTRUMENTS:
            if i.projection_fields == frozenset({"answer_text"}):
                section = self._view(
                    i.key,
                    {"records": [{"answer_text": "RAW", "answer_digest": "d"}]},
                )
                record = section["records"][0]
                self.assertNotIn("answer_text", record, i.key)
                self.assertEqual(record["answer_digest"], "d", i.key)

    def test_template_conformance_ids_are_stripped(self) -> None:
        section = self._view(
            "template_conformance",
            {"diffs": [{"diff": [{"hand_ids": [1], "template_ids": [2], "v": 3}]}]},
        )
        segment = section["diffs"][0]["diff"][0]
        self.assertNotIn("hand_ids", segment)
        self.assertNotIn("template_ids", segment)
        self.assertEqual(segment["v"], 3)

    def test_backend_divergences_are_stripped_and_withheld(self) -> None:
        section = self._view(
            "backend_divergences",
            {
                "stop_texts": ["S"],
                "per_fixture": [{"divergences": [{"stop_text": "S", "detail": "RAW", "k": 1}]}],
            },
        )
        self.assertNotIn("stop_texts", section)
        finding = section["per_fixture"][0]["divergences"][0]
        self.assertNotIn("stop_text", finding)
        self.assertEqual(finding["detail"], reporting._WITHHELD_DIAGNOSTIC_DETAIL)
        self.assertEqual(finding["k"], 1)

    def test_transfer_paths_are_stripped(self) -> None:
        section = self._view("transfer", {"checkpoints": [{"path": "/host/x", "id": "a"}]})
        checkpoint = section["checkpoints"][0]
        self.assertNotIn("path", checkpoint)
        self.assertEqual(checkpoint["id"], "a")

    def test_serving_backends_ids_stripped_and_config_allowlisted(self) -> None:
        section = self._view(
            "serving_backends",
            {
                "records": [{"prompt_ids": [1], "returned_ids": [2], "v": 3}],
                "identity": {"config": {"protocol": "p", "unlisted": "S"}},
            },
        )
        record = section["records"][0]
        self.assertNotIn("prompt_ids", record)
        self.assertNotIn("returned_ids", record)
        self.assertEqual(record["v"], 3)
        self.assertEqual(section["identity"]["config"], {"protocol": "p"})

    def test_verbatim_rows_pass_sentinels_through(self) -> None:
        for i in registry.INSTRUMENTS:
            if i.projection is registry.ProjectionKind.VERBATIM:
                section = self._view(i.key, {"sentinel_w7": "kept"})
                self.assertEqual(section.get("sentinel_w7"), "kept", i.key)


if __name__ == "__main__":
    unittest.main()
