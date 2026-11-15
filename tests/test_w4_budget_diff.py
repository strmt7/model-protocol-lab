"""W4-2 (RR-14a) — the diff-reports mirror of the budget ledger.

The ledger is the first results section whose recorded numbers are
only meaningful against other parts of the same document, so its
differ recomputes each side's actuals from that side's own
search/held-out observations before comparing. These tests pin the
mirror contract:

* two genuine runs compare field-by-field (cap, estimate, actuals,
  planned vs executed cells) and identical runs stay identical;
* a ledger whose recorded actuals contradict its own observations
  is not compared — it surfaces as one-sided availability, which
  breaks the identical verdict (a tampered report must not pass as
  a match);
* producer-invariant violations (estimate above cap, estimate
  recorded without a cap) are likewise rejected per side.
"""

from __future__ import annotations

import copy
import unittest

from model_protocol_lab import core, diffreports
from model_protocol_lab.experiments import run_experiment

BOUNDARY = (999,)
MAX_NEW_TOKENS = 40
EXPECTED = {"S1": "GREEN", "S2": "BLUE", "H1": "RED", "H2": "CYAN"}
BEHAVIOR = {("cand-B", "S2"): "fail"}


class ScriptedBackend:
    """Deterministic backend: ord-encode; pass text + boundary, padded."""

    fingerprint = "w4-budget-diff"

    def __init__(self):
        """Init."""
        self._expected = dict(EXPECTED)
        self._behavior = dict(BEHAVIOR)

    def encode(self, text):
        """Encode."""
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids):
        """Decode."""
        return "".join(map(chr, token_ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        prompt = self.decode(input_ids)
        candidate = next((n for n in ("cand-A", "cand-B") if f"[{n}]" in prompt), None)
        fixture = next((fid for fid in self._expected if f"question-{fid}" in prompt), None)
        if candidate is None or fixture is None:
            return core.Generation((), "error", "unknown_request")
        mode = self._behavior.get((candidate, fixture), "pass")
        expected = self._expected[fixture]
        text = expected + chr(BOUNDARY[0]) if mode == "pass" else expected + "WRONG"
        ids = tuple(map(ord, text))
        assert len(ids) <= max_new_tokens
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


def _split():
    """Split."""

    def fixtures(ids):
        """Fixtures."""
        return tuple(
            core.Fixture(
                id=fid,
                messages=(core.Message(role="user", content=f"question-{fid}"),),
                expected=EXPECTED[fid],
                group=f"group-{fid}",
            )
            for fid in ids
        )

    return core.Split(search=fixtures(("S1", "S2")), held_out=fixtures(("H1", "H2")))


def _formats():
    """Formats."""
    return tuple(
        core.Format(
            name=name,
            bos="",
            user_prefix=f"[{name}]U:",
            assistant_prefix="A:",
            turn_end="\n",
        )
        for name in ("cand-A", "cand-B")
    )


def _run(**kwargs):
    """Run."""
    return run_experiment(
        ScriptedBackend(),
        _formats(),
        _split(),
        boundary=BOUNDARY,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=7,
        **kwargs,
    )


class BudgetDiffTests(unittest.TestCase):
    """The budget section mirror."""

    def test_identical_capped_runs_compare_and_stay_identical(self):
        """Test identical capped runs compare and stay identical."""
        old = _run(max_total_tokens=100_000)
        new = _run(max_total_tokens=100_000)
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["budget"]["available"], "both")
        self.assertFalse(diff["budget"]["changed"])
        self.assertEqual(
            diff["budget"]["actual_total_tokens"]["old"], old["budget"]["actual_total_tokens"]
        )
        self.assertTrue(diff["summary"]["identical"])
        self.assertEqual(diff["section_states"]["budget"]["state"], "compared")

    def test_uncapped_runs_compare(self):
        """Test uncapped runs compare."""
        diff = diffreports.diff_reports(_run(), _run())
        self.assertEqual(diff["budget"]["available"], "both")
        self.assertIsNone(diff["budget"]["max_total_tokens"]["old"])
        self.assertFalse(diff["budget"]["changed"])
        self.assertTrue(diff["summary"]["identical"])

    def test_differing_caps_change_the_section(self):
        """Test differing caps change the section."""
        old = _run(max_total_tokens=100_000)
        new = _run(max_total_tokens=200_000)
        diff = diffreports.diff_reports(old, new)
        self.assertTrue(diff["budget"]["max_total_tokens"]["changed"])
        self.assertTrue(diff["budget"]["changed"])
        self.assertFalse(diff["budget"]["actual_total_tokens"]["changed"])
        self.assertTrue(diff["summary"]["budget_changed"])
        self.assertFalse(diff["summary"]["identical"])

    def test_tampered_actual_is_one_sided_and_breaks_identical(self):
        """Test tampered actual is one sided and breaks identical."""
        old = _run(max_total_tokens=100_000)
        new = copy.deepcopy(old)
        new["budget"]["actual_total_tokens"] += 1
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["budget"]["available"], "old")
        self.assertTrue(diff["summary"]["budget_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])
        self.assertEqual(diff["section_states"]["budget"]["state"], "one_sided")

    def test_estimate_above_cap_is_rejected_per_side(self):
        """Test estimate above cap is rejected per side."""
        old = _run(max_total_tokens=100_000)
        new = copy.deepcopy(old)
        new["budget"]["max_total_tokens"] = new["budget"]["estimated_total_tokens"] - 1
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["budget"]["available"], "old")
        self.assertFalse(diff["summary"]["identical"])

    def test_estimate_without_cap_is_rejected_per_side(self):
        """Test estimate without cap is rejected per side."""
        old = _run(max_total_tokens=100_000)
        new = copy.deepcopy(old)
        new["budget"]["max_total_tokens"] = None
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["budget"]["available"], "old")
        self.assertFalse(diff["summary"]["identical"])

    def test_absent_ledgers_are_neither(self):
        """Test absent ledgers are neither."""
        diff = diffreports.diff_budgets(None, None, {}, {})
        self.assertEqual(diff, {"available": "neither", "changed": False})


if __name__ == "__main__":
    unittest.main()
