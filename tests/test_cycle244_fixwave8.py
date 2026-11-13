"""Cycle 244: bug-hunt wave-7 DISPROOF fix regressions (wave-8).

Every test pins one of the six execution-verified findings of the
wave-7 disproof lane (``~/workspace/agents_out/bughunt7_verify/
report.md`` section 2, F1-F6), each parent-spot-verified for F1 and
F5. One meta-class: a stored value is consumed before its effective
type/magnitude validation, so a malformed-but-loadable payload
crashes the differ/renderer with a raw traceback (or misclassifies
an MCP error) instead of degrading to the frozen degradation
contract -- section differs report ``available`` in {both, old,
new, neither} and never raise on stored data; renderers degrade,
never crash; the MCP layer answers invalid client data with the
tools' frozen invalid-input error (-32602), never an internal
error (-32603).

Fix level, per the codebase's own conventions (recorded in the
cycle work log): entry VALUES are gated at the section validators
(the sibling validators already dict-gate ``per_variant`` values,
contamination pair entries, and transfer's own ``matrix`` values);
the frontier validator shape-gates before ``len()``/iteration (the
renderer-side ``reporting._fixture_total`` already reads a non-list
``fixture_ids`` as zero); float-CONVERTIBILITY of stored ints is
gated at validation, tested by conversion rather than a literal
bound, because the loader's contract deliberately admits any
JSON integer up to its 4,300-digit cap and the consumers must
cope; the renderer's own safe-rate helper completes its
documented never-crash contract; and the MCP fix sits at the
validity layer, where ``hardening_notes`` already raises
``ValueError`` for structural violations, which the tool's frozen
catch tuple maps to -32602.

Each hostile case raised (or misclassified) on the pre-fix code
and degrades on the fixed code; each control passes on BOTH
trees. Boundary magnitudes are exact: ``10**308`` is finite and
stays valid, ``10**309`` is the first overflowing power, and
``2**1024 - 2**970 - 1`` / ``2**1024 - 2**970`` straddle the
exact float-conversion edge.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

import pytest

from model_protocol_lab import diffreports, mcp_server, reporting, surfacenoise
from tests.test_cycle195_choiceaudit import _run_stage as _choiceaudit_run_stage

pytestmark = pytest.mark.hf

REPO_ROOT = Path(__file__).resolve().parent.parent

BIG = 10**400  # loadable (loader digit cap 4,300), beyond float range
FINITE_BIG = 10**308  # largest finite power of ten
EDGE_OK = 2**1024 - 2**970 - 1  # largest float-convertible int
EDGE_OVER = 2**1024 - 2**970  # first non-convertible int

RAISERS: tuple[Any, ...] = ("x", 1, 1.5, True, ["a"], [[]])
FALSY_ENTRIES: tuple[Any, ...] = (None, 0, "", [])


def _sample_results() -> dict:
    """Load the committed sample results payload."""
    path = REPO_ROOT / "examples" / "sample-report" / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _section(name: str) -> Any:
    """One genuine section payload from the committed sample report."""
    return copy.deepcopy(_sample_results()[name])


def _full_diff(name: str, old_section: Any, new_section: Any, *, diff_key: str = "") -> dict:
    """Run the public diff_reports entry point over a one-section swap."""
    old = _sample_results()
    old[name] = old_section
    new = _sample_results()
    new[name] = new_section
    return diffreports.diff_reports(old, new)[diff_key or name]


class BoundaryAuditPerFamilyGateTests(unittest.TestCase):
    """Wave-8 F1: boundary_audit gates per_family entry values.

    Spec: bughunt7_verify F1 -- the validator checked ``per_family``
    is a dict but never its values, while the differ reads
    ``entry.get("rate")``; sibling validators gate entry values
    (``_valid_perturbation`` per_variant, ``_valid_contamination``
    pair entries, ``_valid_transfer`` matrix values).
    """

    def _mutated(self, value: Any) -> dict:
        payload = _section("boundary_audit")
        payload["per_family"]["llama3"] = value
        return payload

    def test_genuine_pair_both(self):
        """Control (both trees): the genuine payload compares clean."""
        diff = diffreports.diff_boundary_audit(
            _section("boundary_audit"), _section("boundary_audit")
        )
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_truthy_non_dict_entries_invalidate(self):
        """Every truthy non-dict entry raised AttributeError pre-fix."""
        for hostile in RAISERS:
            with self.subTest(hostile=repr(hostile)):
                self.assertFalse(diffreports._valid_boundary_audit(self._mutated(hostile)))

    def test_differ_degrades_one_sided(self):
        """The differ reports the hostile side unavailable, never raises."""
        genuine = _section("boundary_audit")
        for hostile in RAISERS:
            with self.subTest(hostile=repr(hostile)):
                mutated = self._mutated(hostile)
                self.assertEqual(
                    diffreports.diff_boundary_audit(mutated, genuine)["available"], "new"
                )
                self.assertEqual(
                    diffreports.diff_boundary_audit(genuine, mutated)["available"], "old"
                )

    def test_public_diff_reports_degrades(self):
        """The public entry point degrades the section, never raises."""
        genuine = _section("boundary_audit")
        out = _full_diff("boundary_audit", self._mutated("x"), genuine)
        self.assertEqual(out["available"], "new")

    def test_falsy_entries_invalidate_like_siblings(self):
        """Stored falsy non-dict entries are invalid data too.

        Deliberate consequence of the value gate: the sibling
        validators reject non-dict entries of any truthiness; a
        falsy entry was previously compared against an empty dict
        by accident of the differ's ``or {}`` (which exists for
        cross-side key absence, not for stored values).
        """
        for falsy in FALSY_ENTRIES:
            with self.subTest(falsy=repr(falsy)):
                self.assertFalse(diffreports._valid_boundary_audit(self._mutated(falsy)))

    def test_dict_entry_with_none_rate_stays_valid(self):
        """Control (both trees): a dict entry is in-grammar at any rate."""
        payload = _section("boundary_audit")
        payload["per_family"]["chatml"] = {"rate": None}
        self.assertTrue(diffreports._valid_boundary_audit(payload))


class SelectionRobustnessPerSemanticGateTests(unittest.TestCase):
    """Wave-8 F2: selection_robustness gates per_semantic values.

    Spec: bughunt7_verify F2 -- same shape as F1 on the
    ``per_semantic.<axis>`` values, whose ``decision``/``winner``
    the differ reads via ``.get``.
    """

    def _mutated(self, value: Any) -> dict:
        payload = _section("selection_robustness")
        payload["per_semantic"]["native"] = value
        return payload

    def test_genuine_pair_both(self):
        """Control (both trees): the genuine payload compares clean."""
        diff = diffreports.diff_selection_robustness(
            _section("selection_robustness"), _section("selection_robustness")
        )
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_truthy_non_dict_entries_invalidate(self):
        """Every truthy non-dict axis value raised AttributeError pre-fix."""
        for hostile in RAISERS:
            with self.subTest(hostile=repr(hostile)):
                self.assertFalse(diffreports._valid_selection_robustness(self._mutated(hostile)))

    def test_differ_degrades_one_sided(self):
        """The differ reports the hostile side unavailable, never raises."""
        genuine = _section("selection_robustness")
        for hostile in RAISERS:
            with self.subTest(hostile=repr(hostile)):
                mutated = self._mutated(hostile)
                self.assertEqual(
                    diffreports.diff_selection_robustness(mutated, genuine)["available"],
                    "new",
                )
                self.assertEqual(
                    diffreports.diff_selection_robustness(genuine, mutated)["available"],
                    "old",
                )

    def test_public_diff_reports_degrades(self):
        """The public entry point degrades the section, never raises."""
        genuine = _section("selection_robustness")
        out = _full_diff("selection_robustness", self._mutated(1), genuine)
        self.assertEqual(out["available"], "new")

    def test_falsy_entries_invalidate_like_siblings(self):
        """Stored falsy non-dict axis values are invalid data too."""
        for falsy in FALSY_ENTRIES:
            with self.subTest(falsy=repr(falsy)):
                self.assertFalse(diffreports._valid_selection_robustness(self._mutated(falsy)))


class TransferPerCheckpointGateTests(unittest.TestCase):
    """Wave-8 F3: transfer gates per_checkpoint values like matrix.

    Spec: bughunt7_verify F3 -- ``_valid_transfer`` dict-gated the
    ``matrix`` values but not the ``per_checkpoint`` values, whose
    ``decision``/``winner`` the differ reads via ``.get``.
    """

    def _mutated(self, value: Any) -> dict:
        payload = _section("transfer")
        payload["per_checkpoint"]["demo-self"] = value
        return payload

    def test_genuine_pair_both(self):
        """Control (both trees): the genuine payload compares clean."""
        diff = diffreports.diff_transfer(_section("transfer"), _section("transfer"))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_truthy_non_dict_entries_invalidate(self):
        """Every truthy non-dict checkpoint value raised pre-fix."""
        for hostile in RAISERS:
            with self.subTest(hostile=repr(hostile)):
                self.assertFalse(diffreports._valid_transfer(self._mutated(hostile)))

    def test_differ_degrades_one_sided(self):
        """The differ reports the hostile side unavailable, never raises."""
        genuine = _section("transfer")
        for hostile in RAISERS:
            with self.subTest(hostile=repr(hostile)):
                mutated = self._mutated(hostile)
                self.assertEqual(diffreports.diff_transfer(mutated, genuine)["available"], "new")
                self.assertEqual(diffreports.diff_transfer(genuine, mutated)["available"], "old")

    def test_public_diff_reports_degrades(self):
        """The public entry point degrades the section, never raises."""
        genuine = _section("transfer")
        out = _full_diff("transfer", self._mutated(["a"]), genuine)
        self.assertEqual(out["available"], "new")

    def test_falsy_entries_invalidate_like_matrix(self):
        """Stored falsy non-dict values are invalid, as matrix's are."""
        for falsy in FALSY_ENTRIES:
            with self.subTest(falsy=repr(falsy)):
                self.assertFalse(diffreports._valid_transfer(self._mutated(falsy)))


class FrontierScalarShapeGateTests(unittest.TestCase):
    """Wave-8 F4: the frontier validator never raises on stored data.

    Spec: bughunt7_verify F4 -- ``len(data.get("fixture_ids") or [])``
    and ``for point in data.get("pass_steps") or []`` raise TypeError
    on truthy scalars (``or []`` rescues only falsy values). The
    validator shape-gates to ``list`` before len()/iteration, the
    same reading ``reporting._fixture_total`` already applies.
    """

    def _mutated(self, key: str, value: Any) -> dict:
        # Frontier validity is any-candidate: the hostile candidate
        # must be the ONLY candidate, or a genuine sibling would
        # qualify the side (the disproof reproducer's shape).
        payload = _section("budget_frontier")
        candidate = copy.deepcopy(payload["candidates"]["llama3"])
        candidate[key] = value
        payload["candidates"] = {"llama3": candidate}
        return payload

    def _one_candidate_mutated(self, key: str, value: Any) -> dict:
        payload = _section("budget_frontier")
        payload["candidates"]["llama3"][key] = value
        return payload

    def test_genuine_pair_both(self):
        """Control (both trees): the genuine payload compares clean."""
        diff = diffreports.diff_frontiers(_section("budget_frontier"), _section("budget_frontier"))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_truthy_scalar_fixture_ids_invalidate(self):
        """Scalar fixture_ids raised TypeError inside the validator."""
        for hostile in (1, True, 1.5):
            with self.subTest(hostile=repr(hostile)):
                self.assertFalse(diffreports._valid_frontier(self._mutated("fixture_ids", hostile)))

    def test_truthy_scalar_pass_steps_invalidate(self):
        """Scalar pass_steps raised TypeError inside the validator."""
        for hostile in (1, True, 1.5):
            with self.subTest(hostile=repr(hostile)):
                self.assertFalse(diffreports._valid_frontier(self._mutated("pass_steps", hostile)))

    def test_differ_degrades_one_sided(self):
        """The differ reports the hostile side unavailable, never raises."""
        genuine = _section("budget_frontier")
        for key in ("fixture_ids", "pass_steps"):
            mutated = self._mutated(key, 1)
            self.assertEqual(diffreports.diff_frontiers(mutated, genuine)["available"], "new")
            self.assertEqual(diffreports.diff_frontiers(genuine, mutated)["available"], "old")

    def test_public_diff_reports_degrades(self):
        """The public entry point degrades the section, never raises."""
        genuine = _section("budget_frontier")
        out = _full_diff(
            "budget_frontier",
            self._mutated("fixture_ids", True),
            genuine,
            diff_key="frontier",
        )
        self.assertEqual(out["available"], "new")

    def test_non_list_fixture_ids_invalidate(self):
        """A string/dict fixture_ids passed len() pre-fix by accident.

        The producer only ever emits lists; a non-empty string or
        dict satisfied ``len(...) > 0`` and qualified the candidate
        pre-fix, and is invalid data under the shape gate.
        """
        for hostile in ("abc", {"steps": 3}):
            with self.subTest(hostile=repr(hostile)):
                self.assertFalse(diffreports._valid_frontier(self._mutated("fixture_ids", hostile)))

    def test_non_list_pass_steps_never_qualified(self):
        """Control (both trees): string/dict steps contain no points.

        Iterating a string yields characters and a dict yields
        keys; neither is a [step, budget] point, so the candidate
        never qualified -- the same verdict the shape gate gives.
        """
        self.assertFalse(diffreports._valid_frontier(self._mutated("pass_steps", "ab")))
        self.assertFalse(diffreports._valid_frontier(self._mutated("pass_steps", {"steps": 3})))

    def test_falsy_and_empty_never_qualified(self):
        """Control (both trees): falsy/empty shapes never qualified.

        As the only candidate they leave the payload invalid; as
        one candidate among genuine siblings they change nothing --
        identical verdicts before and after the fix.
        """
        for value in (None, 0, "", []):
            with self.subTest(value=repr(value)):
                self.assertFalse(diffreports._valid_frontier(self._mutated("fixture_ids", value)))
                self.assertFalse(diffreports._valid_frontier(self._mutated("pass_steps", value)))
                self.assertTrue(
                    diffreports._valid_frontier(self._one_candidate_mutated("fixture_ids", value))
                )


class FloatConvertibilityEdgeTests(unittest.TestCase):
    """Wave-8 F5 predicate edges: conversion, not a literal bound.

    Spec: bughunt7_verify F5 -- the boundary is float's own.
    ``10**308`` converts (finite), ``10**309`` does not, and the
    exact edge sits between ``2**1024 - 2**970 - 1`` (converts)
    and ``2**1024 - 2**970`` (raises OverflowError).
    """

    def test_convertible_magnitudes(self):
        """Finite magnitudes of every size stay convertible."""
        for value in (0, 1, -1, FINITE_BIG, EDGE_OK):
            with self.subTest(value=value):
                self.assertTrue(diffreports._float_convertible(value))

    def test_non_convertible_magnitudes(self):
        """Beyond-range magnitudes are not convertible (fix cases)."""
        for value in (10**309, BIG, EDGE_OVER):
            with self.subTest(value=value):
                self.assertFalse(diffreports._float_convertible(value))

    def test_valid_rate_edges(self):
        """The renderer's safe-rate helper honours the same edges."""
        self.assertEqual(reporting._valid_rate(0), 0.0)
        self.assertEqual(reporting._valid_rate(1), 1.0)
        self.assertEqual(reporting._valid_rate(FINITE_BIG), float(FINITE_BIG))
        self.assertEqual(reporting._valid_rate(EDGE_OK), float(EDGE_OK))
        # Fix cases: each raised OverflowError pre-fix.
        self.assertIsNone(reporting._valid_rate(BIG))
        self.assertIsNone(reporting._valid_rate(EDGE_OVER))
        # Controls (both trees): non-rates were always rejected.
        self.assertIsNone(reporting._valid_rate("x"))
        self.assertIsNone(reporting._valid_rate(True))
        self.assertIsNone(reporting._valid_rate(None))


class ChoiceAuditOverflowGateTests(unittest.TestCase):
    """Wave-8 F5: choice_audit ok-cell bytes are conversion-gated.

    Spec: bughunt7_verify F5 -- the byte ranking divides
    ``float(total)`` by ``int(bytes)``; an oversized stored bytes
    is consumed by the construction check first, whose vocabulary
    is the frozen one-sided degradation.
    """

    def _genuine(self) -> dict:
        _, payload = _choiceaudit_run_stage()
        assert payload["available"] is True
        return payload

    def _mutated(self, value: Any) -> dict:
        payload = self._genuine()
        for entry in payload["candidates"].values():
            if entry.get("status") != "available":
                continue
            for record in entry["records"]:
                for cell in record["cells"]:
                    if cell.get("status") == "ok":
                        cell["bytes"] = value
                        return payload
        raise AssertionError("genuine payload has no ok cell")

    def test_genuine_pair_both(self):
        """Control (both trees): the producer payload compares clean."""
        diff = diffreports.diff_choiceaudit(self._genuine(), self._genuine())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_oversized_bytes_invalidate(self):
        """A 10**400 bytes raised OverflowError in the recompute."""
        self.assertFalse(diffreports._valid_choiceaudit(self._mutated(BIG)))

    def test_differ_degrades_one_sided(self):
        """The differ reports the hostile side unavailable, never raises."""
        genuine = self._genuine()
        mutated = self._mutated(BIG)
        self.assertEqual(diffreports.diff_choiceaudit(mutated, genuine)["available"], "new")
        self.assertEqual(diffreports.diff_choiceaudit(genuine, mutated)["available"], "old")

    def test_public_diff_reports_degrades(self):
        """The public entry point degrades the section, never raises."""
        out = _full_diff("choice_audit", self._mutated(BIG), self._genuine())
        self.assertEqual(out["available"], "new")


class CacheProfileOverflowGateTests(unittest.TestCase):
    """Wave-8 F5: cache_profile counts are conversion-gated.

    Spec: bughunt7_verify F5 -- the validator's own recompute
    divides and multiplies the stored counts (block fractions,
    mean prompt tokens); every count the recompute touches is
    gated, uniformly, because which division overflows depends on
    the quotient, not on the site.
    """

    _SCALED_KEYS = (
        "total_tokens",
        "cacheable_tokens",
        "shared_prefixes",
        "achievable_tokens",
        "block_cacheable_blocks",
        "distinct_block_prefixes",
        "achievable_blocks",
    )

    def _scaled(self, factor: int) -> dict:
        """Scale one candidate's token counts, identities preserved.

        Uniform scaling preserves every ratio and difference
        identity the validator recomputes; the stored mean is
        recomputed by the validator's own expression.
        """
        payload = _section("cache_profile")
        for entry in payload["per_candidate"].values():
            if entry is None:
                continue
            for key in self._SCALED_KEYS:
                entry[key] = entry[key] * factor
            try:
                entry["mean_prompt_tokens"] = entry["total_tokens"] / entry["n_evaluable"]
            except OverflowError:
                # The oversized case: no float mean exists, and the
                # count gate rejects the payload before the mean is
                # ever recomputed, so any placeholder is unreachable.
                entry["mean_prompt_tokens"] = 1.0
            return payload
        raise AssertionError("genuine payload has no candidate entry")

    def test_genuine_pair_both(self):
        """Control (both trees): the genuine payload compares clean."""
        diff = diffreports.diff_cache_profile(_section("cache_profile"), _section("cache_profile"))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_finite_scaling_stays_valid(self):
        """Control (both trees): counts near 10**303 remain in domain."""
        scaled = self._scaled(10**300)
        self.assertTrue(diffreports._valid_cache_profile(scaled))
        diff = diffreports.diff_cache_profile(scaled, copy.deepcopy(scaled))
        self.assertEqual(diff["available"], "both")

    def test_oversized_scaling_invalidates(self):
        """Counts near 10**403 overflowed the recompute pre-fix."""
        scaled = self._scaled(BIG)
        self.assertFalse(diffreports._valid_cache_profile(scaled))
        genuine = _section("cache_profile")
        self.assertEqual(diffreports.diff_cache_profile(scaled, genuine)["available"], "new")
        self.assertEqual(diffreports.diff_cache_profile(genuine, scaled)["available"], "old")

    def test_oversized_block_size_invalidates(self):
        """A 10**400 block_size overflowed the block fractions."""
        payload = _section("cache_profile")
        payload["block_size"] = BIG
        self.assertFalse(diffreports._valid_cache_profile(payload))
        genuine = _section("cache_profile")
        self.assertEqual(diffreports.diff_cache_profile(payload, genuine)["available"], "new")

    def test_public_diff_reports_degrades(self):
        """The public entry point degrades the section, never raises."""
        out = _full_diff("cache_profile", self._scaled(BIG), _section("cache_profile"))
        self.assertEqual(out["available"], "new")


class AnswerCollisionOverflowGateTests(unittest.TestCase):
    """Wave-8 F5: answer_collision max_collision_rate is gated.

    Spec: bughunt7_verify F5 -- the validator compares the stored
    max rate against the recomputed maximum via ``float(...)``;
    an oversized int max rate is invalid data, not an exception.
    """

    def _mutated(self, value: Any) -> dict:
        payload = _section("answer_collision")
        payload["max_collision_rate"] = value
        return payload

    def test_genuine_pair_both(self):
        """Control (both trees): the genuine payload compares clean."""
        diff = diffreports.diff_answer_collision(
            _section("answer_collision"), _section("answer_collision")
        )
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_oversized_max_rate_invalidates(self):
        """A 10**400 max rate raised OverflowError in the validator."""
        self.assertFalse(diffreports._valid_answer_collision(self._mutated(BIG)))

    def test_differ_degrades_one_sided(self):
        """The differ reports the hostile side unavailable, never raises."""
        genuine = _section("answer_collision")
        mutated = self._mutated(BIG)
        self.assertEqual(diffreports.diff_answer_collision(mutated, genuine)["available"], "new")
        self.assertEqual(diffreports.diff_answer_collision(genuine, mutated)["available"], "old")

    def test_wrong_magnitude_rate_invalid_on_both_trees(self):
        """Control (both trees): a finite but wrong max rate is invalid."""
        self.assertFalse(diffreports._valid_answer_collision(self._mutated(0)))

    def test_public_diff_reports_degrades(self):
        """The public entry point degrades the section, never raises."""
        out = _full_diff("answer_collision", self._mutated(BIG), _section("answer_collision"))
        self.assertEqual(out["available"], "new")


class NumformGroupingOverflowTests(unittest.TestCase):
    """The grouping block's stored form means/deltas feed a float
    tolerance product in the recompute; oversized values stored at
    those float positions must degrade the side, not raise."""

    def _genuine(self) -> dict:
        from tests.test_cycle186_numform import _base_payload

        return _base_payload()

    def test_grouping_mean_overflow_degrades(self):
        old, new = self._genuine(), self._genuine()
        for block in ("form_token_means", "token_deltas_vs_plain"):
            for form in ("comma", "space", "scientific"):
                if type(new["grouping"][block][form]) is float:
                    new["grouping"][block][form] = BIG
        self.assertEqual(diffreports.diff_numform(old, new)["available"], "old")
        self.assertEqual(diffreports.diff_numform(new, old)["available"], "new")


class HeldOutRateRendererTests(unittest.TestCase):
    """Wave-8 F5 (renderer): an oversized held_out rate degrades.

    Spec: bughunt7_verify F5 -- ``render_html`` raised OverflowError
    on a stored 10**400 held_out rate. ``_valid_rate`` exists so a
    hostile rate "degrades to the incomplete badge, never a crash";
    the fix completes that documented contract.
    """

    def _render_section(self, payload: dict) -> str:
        out = reporting.render_html(payload)
        start = out.index('id="held-out-stage"')
        return out[start : out.index("</section>", start)]

    def test_genuine_rate_renders_as_percent(self):
        """Control (both trees): the genuine rate renders formatted."""
        section = self._render_section(_sample_results())
        self.assertIn("100%", section)
        self.assertNotIn("incomplete", section)

    def test_oversized_rate_degrades_to_incomplete(self):
        """The hostile rate renders the incomplete badge, never raises."""
        payload = _sample_results()
        payload["held_out"]["summary"]["rate"] = BIG
        section = self._render_section(payload)
        self.assertIn("incomplete", section)


class McpHardeningNotesOverflowTests(unittest.TestCase):
    """Wave-8 F6: a huge flip distance is invalid input (-32602).

    Spec: bughunt7_verify F6 + the wave-5 F4 error-class
    discipline -- client-supplied invalid data yields the tool's
    frozen invalid-input error, never an internal error.
    ``hardening_notes`` raises ``ValueError`` for a numeric
    distance that is not float-convertible, in its own structural
    vocabulary; the tool's catch tuple maps it to -32602.
    """

    def _report(self, distance: Any) -> dict:
        return {
            "operators": {
                "op_a": {
                    "stability": 0.5,
                    "n_effective": 3,
                    "flips": [
                        {
                            "variant_id": "v1",
                            "distance": distance,
                            "changed_fixtures": ["f1"],
                        }
                    ],
                }
            },
            "static_lint": {},
        }

    def _call(self, distance: Any) -> dict:
        return mcp_server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {
                    "name": "surface_noise_hardening_notes",
                    "arguments": {"report": self._report(distance)},
                },
            }
        )

    def test_hardening_notes_raises_value_error(self):
        """The validity layer rejects the distance as ValueError."""
        with self.assertRaises(ValueError):
            surfacenoise.hardening_notes(self._report(BIG))

    def test_tool_raises_invalid_params(self):
        """The tool surfaces MCPError -32602, not a raw OverflowError."""
        with self.assertRaises(mcp_server.MCPError) as caught:
            mcp_server.tool_hardening_notes({"report": self._report(BIG)})
        self.assertEqual(caught.exception.code, -32602)

    def test_handle_request_returns_invalid_params(self):
        """End to end: -32602 for the hostile distance (was -32603)."""
        for hostile in (BIG, 10**309):
            with self.subTest(hostile=hostile):
                response = self._call(hostile)
                self.assertEqual(response["error"]["code"], -32602)

    def test_sane_distances_resolve_as_before(self):
        """Controls (both trees): sane distances produce notes."""
        for distance, rendered in ((0.25, "0.2500"), (1, "1.0000"), (0, "0.0000")):
            with self.subTest(distance=distance):
                response = self._call(distance)
                self.assertIn("result", response)
                self.assertIn(rendered, json.dumps(response["result"]))

    def test_finite_huge_distance_resolves(self):
        """Control (both trees): 10**308 is finite and formats fine."""
        response = self._call(FINITE_BIG)
        self.assertIn("result", response)

    def test_wrong_type_distances_resolve_as_before(self):
        """Controls (both trees): non-numeric distances take str()."""
        for distance in ("x", None, [1], {"a": 1}):
            with self.subTest(distance=repr(distance)):
                response = self._call(distance)
                self.assertNotEqual(response.get("error", {}).get("code"), -32603)
                self.assertIn("result", response)


class LoaderPathIntegrationTests(unittest.TestCase):
    """Wave-8 integration: the loader path degrades end to end.

    Spec: the disproof lane's part-4 shape -- a 401-digit integer
    is legal input to ``parse_results_bytes`` (digit cap 4,300);
    every consumer downstream of the loader must cope.
    """

    def _roundtrip(self, payload: dict) -> dict:
        return diffreports.parse_results_bytes(json.dumps(payload).encode("utf-8"))

    def test_hostile_pair_degrades_everywhere(self):
        """Loaded hostile reports diff and render without raising."""
        old = _sample_results()
        new = _sample_results()
        new["boundary_audit"]["per_family"]["llama3"] = "x"
        candidate = copy.deepcopy(new["budget_frontier"]["candidates"]["llama3"])
        candidate["fixture_ids"] = 1
        new["budget_frontier"]["candidates"] = {"llama3": candidate}
        new["held_out"]["summary"]["rate"] = BIG
        old_loaded = self._roundtrip(old)
        new_loaded = self._roundtrip(new)
        out = diffreports.diff_reports(old_loaded, new_loaded)
        self.assertEqual(out["boundary_audit"]["available"], "old")
        self.assertEqual(out["frontier"]["available"], "old")
        self.assertIsInstance(reporting.render_html(new_loaded), str)


def _set_first(node: Any, key: str, value: Any) -> bool:
    """Set the first occurrence of ``key`` anywhere under ``node``."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                node[k] = value
                return True
            if _set_first(v, key, value):
                return True
    elif isinstance(node, list):
        for item in node:
            if _set_first(item, key, value):
                return True
    return False


def _set_all(node: Any, key: str, value: Any) -> int:
    """Set every occurrence of ``key`` under ``node``; return the count."""
    count = 0
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                node[k] = value
                count += 1
            else:
                count += _set_all(v, key, value)
    elif isinstance(node, list):
        for item in node:
            count += _set_all(item, key, value)
    return count


def _bulk_big(node: Any) -> Any:
    """Deep-copy with every int leaf replaced by the oversized int."""
    if isinstance(node, dict):
        return {k: _bulk_big(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_bulk_big(v) for v in node]
    if type(node) is int:
        return BIG
    return node


class LeadingFamilyOverflowGateTests(unittest.TestCase):
    """Wave-8 F5 (census extension): leading/numform token means.

    Spec: the cycle's own pattern sweep over all 53 sections (the
    disproof census covered 14) -- the leading differ's framing
    deltas and the numform differ's token means divide sums of
    stored counts; the family admission predicate
    ``_leading_int_ok`` now requires float-convertibility, and
    every mean/delta is bounded by the largest admitted count.
    """

    def _leading_genuine(self) -> dict:
        from tests.test_cycle185_leading import _fab_payload, _vrecords

        return _fab_payload(_vrecords())

    def _numform_genuine(self) -> dict:
        from tests.test_cycle186_numform import _base_payload

        return _base_payload()

    def test_leading_genuine_pair_both(self):
        """Control (both trees): the fabricated payload compares clean."""
        diff = diffreports.diff_leading(self._leading_genuine(), self._leading_genuine())
        self.assertEqual(diff["available"], "both")

    def test_leading_oversized_input_tokens_invalidate(self):
        """A 10**400 cell input_tokens overflowed the framing deltas."""
        mutated = self._leading_genuine()
        self.assertTrue(_set_first(mutated, "input_tokens", BIG))
        self.assertFalse(diffreports._valid_leading(mutated))
        genuine = self._leading_genuine()
        self.assertEqual(diffreports.diff_leading(mutated, genuine)["available"], "new")
        self.assertEqual(diffreports.diff_leading(genuine, mutated)["available"], "old")

    def test_numform_genuine_pair_both(self):
        """Control (both trees): the fabricated payload compares clean."""
        diff = diffreports.diff_numform(self._numform_genuine(), self._numform_genuine())
        self.assertEqual(diff["available"], "both")

    def test_numform_oversized_input_tokens_invalidate(self):
        """A 10**400 cell input_tokens overflowed the token mean."""
        mutated = self._numform_genuine()
        self.assertTrue(_set_first(mutated, "input_tokens", BIG))
        self.assertFalse(diffreports._valid_numform(mutated))
        genuine = self._numform_genuine()
        self.assertEqual(diffreports.diff_numform(mutated, genuine)["available"], "new")
        self.assertEqual(diffreports.diff_numform(genuine, mutated)["available"], "old")

    def test_numform_oversized_answer_tokens_invalidate(self):
        """Oversized expected_answer_tokens overflowed their mean."""
        mutated = self._numform_genuine()
        self.assertGreater(_set_all(mutated, "expected_answer_tokens", BIG), 0)
        self.assertFalse(diffreports._valid_numform(mutated))
        genuine = self._numform_genuine()
        self.assertEqual(diffreports.diff_numform(mutated, genuine)["available"], "new")


class CacheProfileComposedProductTests(unittest.TestCase):
    """Wave-8 F5 (composed values): block_size x blocks is gated.

    Spec: two individually float-convertible factors (10**200
    each) compose past float range, and the recompute converts
    the product -- the gate covers the composed values the
    consumer actually converts.
    """

    def test_composed_block_product_invalidates(self):
        """The 10**200 x 10**200 product raised on the ungated tree."""
        payload = _section("cache_profile")
        _, entry = next((k, v) for k, v in payload["per_candidate"].items() if v is not None)
        entry["block_cacheable_blocks"] = 10**200
        entry["distinct_block_prefixes"] = 10**200 - entry["achievable_blocks"]
        payload["block_size"] = 10**200
        self.assertFalse(diffreports._valid_cache_profile(payload))
        genuine = _section("cache_profile")
        self.assertEqual(diffreports.diff_cache_profile(payload, genuine)["available"], "new")

    def test_convertible_product_stays_valid(self):
        """Control (both trees): a large but convertible product is fine."""
        payload = _section("cache_profile")
        _, entry = next((k, v) for k, v in payload["per_candidate"].items() if v is not None)
        # block_size 10**150 with the genuine block counts: the
        # stored block fractions no longer match the recompute, so
        # the payload is invalid on both trees for the ordinary
        # mismatch reason -- the verdict, not the mechanism, is
        # what this control pins (no raise either way).
        payload["block_size"] = 10**150
        self.assertFalse(diffreports._valid_cache_profile(payload))


class RendererOverflowSiteTests(unittest.TestCase):
    """Wave-8 F5 (renderer): every proven format site degrades.

    Spec: the pattern sweep's render hits -- the section-local
    numeric formatters (_rate/_num/_share/_pct/_gap/_pvalue/
    _index/_points/_sci/_mean/_delta/_as_rate) and the inline
    prose sites format stored numbers; each now degrades to its
    own fallback ("n/a", omission, or the incomplete badge)
    instead of raising OverflowError on an oversized stored int.
    """

    def _render_with(self, section: str, key: str) -> str:
        payload = _sample_results()
        self.assertTrue(_set_first(payload[section], key, BIG), f"{section}.{key} not found")
        return reporting.render_html(payload)

    def test_sample_section_sites_render(self):
        """One proven stored field per sample-present section renders."""
        cases = [
            ("boundary_audit", "overall_rate"),
            ("marker_isolation", "overall_isolation_rate"),
            ("hierarchy", "mean_system_rate"),
            ("deliforge", "mean_authority_gap"),
            ("faketurn", "transfer_gap"),
            ("faketurn", "max_transfer_gap"),
            ("synthhistory", "mean_pass_rate"),
            ("prefill", "mean_honor_rate"),
            ("verbosity", "max_excess"),
            ("verbosity", "median_excess_gap"),
            ("cache_profile", "sharability_gap"),
            ("confidentiality", "gap"),
            ("answer_collision", "collision_rate"),
        ]
        for section, key in cases:
            with self.subTest(section=section, key=key):
                self.assertIsInstance(self._render_with(section, key), str)

    def test_boundary_family_rate_text_degrades(self):
        """A per-family rate stored as an oversized int renders n/a."""
        payload = _sample_results()
        payload["boundary_audit"]["per_family"]["llama3"]["rate"] = BIG
        out = reporting.render_html(payload)
        start = out.index('id="boundary-canonicality"')
        section = out[start : out.index("</section>", start)]
        self.assertIn("n/a", section)

    def test_faketurn_top_level_max_flip_rate_line_omitted(self):
        """The optional largest-flip-rate line is omitted, not a crash."""
        payload = _sample_results()
        payload["faketurn"]["max_flip_rate"] = BIG
        out = reporting.render_html(payload)
        self.assertNotIn("Largest per-cell verdict-flip rate", out)

    def test_structured_adherence_gap_line_omitted(self):
        """The optional adherence-gap line is omitted, not a crash."""
        payload = _sample_results()
        payload["structured"]["adherence_gap"] = BIG
        out = reporting.render_html(payload)
        self.assertNotIn("Adherence gap (strongest minus weakest template)", out)

    def test_invisible_chars_screen_probability_degrades(self):
        """The operating-characteristics _sci format degrades to n/a."""
        from tests.test_cycle223_invisiblechars import _payload as _ic_payload

        section = _ic_payload()
        entry = next(iter(section["candidates"].values()))
        entry["operating_characteristics"]["screen_guess_probability"] = BIG
        report = _sample_results()
        report["invisible_chars"] = section
        self.assertIsInstance(reporting.render_html(report), str)

    def test_frontier_oversized_count_point_is_skipped(self):
        """The frontier percent point degrades by omission, not a crash."""
        payload = _sample_results()
        point = payload["budget_frontier"]["candidates"]["llama3"]["pass_steps"][0]
        point[1] = BIG
        out = reporting.render_html(payload)
        self.assertIn('id="token-budget-frontier"', out)

    def test_factory_section_sites_render(self):
        """Factory-built sections render with an oversized field."""
        from tests.test_cycle198_precision import _valid_payload as _precision_payload
        from tests.test_cycle199_confidence import _stage_payload as _confidence_payload
        from tests.test_cycle204_density import _payload as _density_payload
        from tests.test_cycle205_fieldorder import _payload as _field_order_payload
        from tests.test_cycle221_recordnotation import (
            _payload as _record_notation_payload,
        )

        cases = [
            ("confidence", _confidence_payload(), "gap"),
            ("density", _density_payload(), "sign_p"),
            ("density", _density_payload(), "sham_block_words_mean"),
            ("field_order", _field_order_payload(), "sign_p"),
            ("record_notation", _record_notation_payload(), "resolution_points"),
            ("precision", _precision_payload(), "mean_delta"),
        ]
        for section, section_payload, key in cases:
            with self.subTest(section=section, key=key):
                self.assertTrue(
                    _set_first(section_payload, key, BIG),
                    f"{section}.{key} not found in factory payload",
                )
                report = _sample_results()
                report[section] = section_payload
                self.assertIsInstance(reporting.render_html(report), str)

    def test_genuine_report_renders_unchanged(self):
        """Control (both trees): the genuine sample renders fully."""
        out = reporting.render_html(_sample_results())
        self.assertIn('id="held-out-stage"', out)
        self.assertIn("100%", out)

    def test_bulk_oversized_report_renders(self):
        """Every int leaf oversized at once still renders (bulk surface)."""
        self.assertIsInstance(reporting.render_html(_bulk_big(_sample_results())), str)

    def test_bulk_widened_report_renders(self):
        """Every float leaf stored as an oversized int still renders.

        The type-widening arm of the class: renderers format
        stored numbers whose genuine form is a float; a hostile
        payload may carry the same field as a several-hundred-
        digit int literal instead.
        """

        def widen(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: widen(v) for k, v in node.items()}
            if isinstance(node, list):
                return [widen(v) for v in node]
            if type(node) is float:
                return BIG
            return node

        self.assertIsInstance(reporting.render_html(widen(_sample_results())), str)


class SummaryOverflowSiteTests(unittest.TestCase):
    """Wave-8 F5 (text summary): _fmt_* helpers degrade to (none).

    Spec: the corrected pattern sweep's summary hits -- differs
    pass some stored magnitudes through without range gates
    (boundary overall_rate is type-checked only), and
    ``render_summary`` formatted them unguarded.
    """

    def test_boundary_overall_rate_summary_degrades(self):
        """An oversized overall_rate renders as (none), never raises."""
        old = _sample_results()
        new = _sample_results()
        new["boundary_audit"]["overall_rate"] = BIG
        text = diffreports.render_summary(diffreports.diff_reports(old, new))
        self.assertIn("(none)", text)

    def test_stopcause_oversized_cause_rate_summary_degrades(self):
        """An oversized stored cause rate renders as (none).

        The stop-cause differ passes the stored per-cause rates
        into its drift entries; the summary formatted them
        unguarded pre-fix.
        """
        from tests.test_cycle155_stopcause import _stop_payload

        old = _sample_results()
        new = _sample_results()
        old["stop_cause"] = _stop_payload()
        hostile = _stop_payload()
        hostile["per_candidate"]["alpha"]["cause_rates"]["eos_id"] = BIG
        new["stop_cause"] = hostile
        text = diffreports.render_summary(diffreports.diff_reports(old, new))
        self.assertIn("(none)", text)

    def test_hierarchy_oversized_system_rate_summary_degrades(self):
        """An oversized stored system rate renders as (none).

        The hierarchy differ passes per-condition system rates
        into its drift entries; the summary formatted them
        unguarded pre-fix.
        """
        from tests.test_cycle147_hierarchy import _hierarchy_payload

        old = _sample_results()
        new = _sample_results()
        old["hierarchy"] = _hierarchy_payload()
        hostile = _hierarchy_payload()
        mutated = False
        for condition in hostile["per_condition"].values():
            for entry in condition["per_candidate"].values():
                entry["system_rate"] = BIG
                mutated = True
            if mutated:
                break
        self.assertTrue(mutated)
        new["hierarchy"] = hostile
        text = diffreports.render_summary(diffreports.diff_reports(old, new))
        self.assertIn("(none)", text)

    def test_genuine_summary_unchanged(self):
        """Control (both trees): the genuine pair summarizes cleanly."""
        report = _sample_results()
        text = diffreports.render_summary(diffreports.diff_reports(report, report))
        self.assertIn("Prompt Boundary Problem rate: 0.0% -> 0.0%", text)


class RendererAggregateOverflowTests(unittest.TestCase):
    """Wave-8 F5, aggregate arm: the per-section summary helpers.

    Spec: bughunt7_verify F5 -- the renderer's safe-conversion
    convention is uniform, so each section's aggregate helpers
    (the nested _rate / _gap / _num / _share / _pct / _delta
    closures that format the section's headline values) degrade
    an oversized stored aggregate to the helper's own "n/a"
    instead of raising OverflowError. Every section payload is
    rendered under both mutation arms: every int leaf oversized,
    and every float leaf stored as an oversized int (the
    type-widening arm).
    """

    def _sections(self) -> dict:
        from model_protocol_lab.experiments import scorerlens_profile
        from tests.test_cycle180_scorerlens import _EXPECTED as _lens_expected
        from tests.test_cycle180_scorerlens import _profile_result as _lens_result
        from tests.test_cycle183_repetition import (
            TestValidatorAndDiff as _RepetitionFactory,
        )
        from tests.test_cycle184_distraction import _fab_payload as _distraction_fab
        from tests.test_cycle184_distraction import _vrecord as _distraction_vrecord
        from tests.test_cycle185_leading import _fab_payload, _vrecords
        from tests.test_cycle186_numform import _base_payload as _numform_payload
        from tests.test_cycle188_dateline import _base_payload as _dateline_payload
        from tests.test_cycle189_multiturn import _payload as _multiturn_payload
        from tests.test_cycle191_needle import _payload as _needle_payload
        from tests.test_cycle193_placement import _payload as _placement_payload
        from tests.test_cycle195_choiceaudit import _run_stage as _choice_stage
        from tests.test_cycle198_precision import _valid_payload as _precision_payload
        from tests.test_cycle199_confidence import _stage_payload as _confidence_payload
        from tests.test_cycle203_rotation import _payload as _rotation_payload
        from tests.test_cycle204_density import _payload as _density_payload
        from tests.test_cycle205_fieldorder import _payload as _field_order_payload
        from tests.test_cycle209_canonicalform import _payload as _canonical_payload
        from tests.test_cycle212_negation import _payload as _negation_payload
        from tests.test_cycle218_instructiondistance import (
            _payload as _instruction_payload,
        )
        from tests.test_cycle219_historyframe import _payload as _history_payload
        from tests.test_cycle220_prunesweep import _payload as _prune_payload
        from tests.test_cycle221_recordnotation import (
            _payload as _record_notation_payload,
        )
        from tests.test_cycle223_invisiblechars import _payload as _invisible_payload
        from tests.test_cycle224_confusableskeleton import (
            _payload as _confusable_payload,
        )
        from tests.test_cycle231_compatibilityform import (
            _valid_payload as _compatibility_payload,
        )
        from tests.test_cycle233_enumerationconvention import (
            _valid_payload as _enumeration_payload,
        )
        from tests.test_cycle237_typographicemphasis import (
            _valid_payload as _emphasis_payload,
        )

        _distraction_cells = {"near-miss_64_before": None}
        _distraction_records = {
            name: [
                _distraction_vrecord("fx-1", cells=_distraction_cells),
                _distraction_vrecord("fx-2", cells={"near-miss_64_before": (True, "8")}),
            ]
            for name in ("dst-a", "dst-b")
        }

        return {
            "structured": _section("structured"),
            "deliforge": _section("deliforge"),
            "faketurn": _section("faketurn"),
            "toolcall": _section("toolcall"),
            "leading": _fab_payload(_vrecords()),
            "numform": _numform_payload(),
            "choice_audit": _choice_stage()[1],
            "precision": _precision_payload(),
            "confidence": _confidence_payload(),
            "density": _density_payload(),
            "field_order": _field_order_payload(),
            "record_notation": _record_notation_payload(),
            "invisible_chars": _invisible_payload(),
            "dateline": _dateline_payload(),
            "multiturn": _multiturn_payload(),
            "needle": _needle_payload(),
            "placement": _placement_payload(),
            "rotation": _rotation_payload(),
            "canonical_form": _canonical_payload(),
            "negation": _negation_payload(),
            "instruction_distance": _instruction_payload(),
            "history_frame": _history_payload(),
            "prune_sweep": _prune_payload(),
            "confusable_skeleton": _confusable_payload()[0],
            "compatibility_form": _compatibility_payload(),
            "enumeration_convention": _enumeration_payload(),
            "typographic_emphasis": _emphasis_payload(),
            "scorerlens": scorerlens_profile(_lens_result(), dict(_lens_expected)),
            "repetition": _RepetitionFactory._valid_payload(),
            "distraction": _distraction_fab(_distraction_records),
        }

    def _widened(self, node: Any) -> Any:
        if isinstance(node, dict):
            return {k: self._widened(v) for k, v in node.items()}
        if isinstance(node, list):
            return [self._widened(v) for v in node]
        if type(node) is float:
            return BIG
        return node

    def test_confusable_arm_rates_render_oversized(self):
        """The skeleton arms' rates are formatted by the section _rate."""
        from tests.test_cycle224_confusableskeleton import _payload as _confusable

        section = _confusable()[0]
        self.assertGreater(_set_all(section, "rate", BIG), 0)
        report = _sample_results()
        report["confusable_skeleton"] = section
        self.assertIsInstance(reporting.render_html(report), str)

    def test_section_aggregates_render_oversized(self):
        """Every section's headline helpers degrade, never raise."""
        for name, section in self._sections().items():
            for label, hostile in (
                ("int-bulk", _bulk_big(copy.deepcopy(section))),
                ("widened", self._widened(copy.deepcopy(section))),
            ):
                with self.subTest(section=name, arm=label):
                    report = _sample_results()
                    report[name] = hostile
                    self.assertIsInstance(reporting.render_html(report), str)
