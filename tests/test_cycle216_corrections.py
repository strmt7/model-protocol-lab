"""Cycle 216: verification/correction round — item 2 (the
`_valid_primary_pair` latent sibling) and item 3 (dateline oracle
caveat evidence).

Item 2. Cycle 207's A-F1 fixed a TypeError crash in verify.py's
schema check: an unhashable `decision` value (list/dict) reached a
frozenset membership test without a type guard. The cycle-207
record queued the latent sibling in
`diffreports._valid_primary_pair` with the note "every caller
pre-guards". Cycle 216 investigated by execution: the function
itself raised TypeError on an unhashable `primary_decision`, and
two of its six callers — the precision and confidence recompute
paths — reach it without the pre-guard the other four callers
carry, so `_valid_precision` / `_valid_confidence` (and the
`diff_precision` / `diff_confidence` entry points) raised on a
hostile payload instead of failing closed. The fix type-checks
the decision inside `_valid_primary_pair` itself (the shared
helper, so the class closes for every present and future
caller). The precision/confidence tests below FAIL on the
pre-fix code (they raise TypeError); the direct-function tests
fail pre-fix the same way. The four pre-guarded validators'
tests pass both ways — they pin the fail-closed surface, not
the mutant.

Item 3. The cycle-188 verifier recorded that neither test
oracle carries its own literal dateline arm table or weekday
computation (both recompute through the producer module, the
oracle family's standing producer-as-ground-truth pattern).
The shipped mitigation is the production validator's own
literal tables in `diffreports`: the drift test below corrupts
the producer's weekday table, rebuilds a full payload through
the producer path (payload, protocol and fingerprint drift
together, self-consistently), and pins that the literal
validator still rejects it. This test passes before and after
the cycle-216 fix — it is the execution evidence for the
DATELINE_DESIGN.md wording the cycle adds, not a regression
test for a code change.
"""

from __future__ import annotations

import copy
import sys
import unittest

sys.path.insert(0, "tests")

from test_cycle188_dateline import _base_payload  # noqa: E402
from test_cycle198_precision import _valid_payload as _precision_payload  # noqa: E402
from test_cycle199_confidence import _stage_payload as _confidence_payload  # noqa: E402

from model_protocol_lab import dateline, diffreports  # noqa: E402
from model_protocol_lab.diffreports import (  # noqa: E402
    _valid_confidence,
    _valid_dateline,
    _valid_distraction,
    _valid_leading,
    _valid_precision,
    _valid_primary_pair,
    _valid_repetition,
    diff_confidence,
    diff_precision,
)

_UNHASHABLE_DECISIONS = ([], {}, ["compatible_on_fixture_set"], {"decision": "ambiguous"})


class TestPrimaryPairTotality(unittest.TestCase):
    """The pair binding itself must be total: bool out, never a raise."""

    def test_unhashable_decision_fails_closed(self):
        """An unhashable primary decision makes the pair validator fail closed."""
        for bad in _UNHASHABLE_DECISIONS:
            with self.subTest(bad=bad):
                self.assertFalse(
                    _valid_primary_pair(
                        {"primary_decision": bad, "primary_winner": None, "candidates": {}}
                    )
                )

    def test_unhashable_winner_fails_closed(self):
        """An unhashable primary winner makes the pair validator fail closed."""
        self.assertFalse(
            _valid_primary_pair(
                {
                    "primary_decision": "compatible_on_fixture_set",
                    "primary_winner": ["cand-a"],
                    "candidates": {"cand-a": {}},
                }
            )
        )

    def test_valid_pairs_still_bind(self):
        """Valid primary pairs still bind; mismatched ones do not."""
        self.assertTrue(
            _valid_primary_pair(
                {
                    "primary_decision": "compatible_on_fixture_set",
                    "primary_winner": "cand-a",
                    "candidates": {"cand-a": {}},
                }
            )
        )
        self.assertTrue(
            _valid_primary_pair(
                {"primary_decision": "ambiguous", "primary_winner": None, "candidates": {}}
            )
        )
        self.assertFalse(
            _valid_primary_pair(
                {"primary_decision": "ambiguous", "primary_winner": "cand-a", "candidates": {}}
            )
        )


class TestPrecisionConfidencePairGuard(unittest.TestCase):
    """The two recompute paths that reached the pair unguarded (live pre-fix)."""

    def test_precision_unhashable_decision_fails_closed(self):
        """An unhashable decision fails the precision pair validator closed."""
        base = _precision_payload()
        self.assertTrue(_valid_precision(base))
        for bad in _UNHASHABLE_DECISIONS:
            hostile = copy.deepcopy(base)
            hostile["primary_decision"] = bad
            with self.subTest(bad=bad):
                self.assertFalse(_valid_precision(hostile))

    def test_precision_diff_reports_hostile_side_unavailable(self):
        """A hostile precision side diffs as unavailable."""
        base = _precision_payload()
        hostile = copy.deepcopy(base)
        hostile["primary_decision"] = []
        result = diff_precision(base, hostile)
        self.assertEqual(result["available"], "old")

    def test_confidence_unhashable_decision_fails_closed(self):
        """An unhashable decision fails the confidence pair validator closed."""
        base = _confidence_payload()
        self.assertTrue(_valid_confidence(base))
        for bad in _UNHASHABLE_DECISIONS:
            hostile = copy.deepcopy(base)
            hostile["primary_decision"] = bad
            with self.subTest(bad=bad):
                self.assertFalse(_valid_confidence(hostile))

    def test_confidence_diff_reports_hostile_side_unavailable(self):
        """A hostile confidence side diffs as unavailable."""
        base = _confidence_payload()
        hostile = copy.deepcopy(base)
        hostile["primary_decision"] = {}
        result = diff_confidence(base, hostile)
        self.assertEqual(result["available"], "old")


class TestPreGuardedPairValidators(unittest.TestCase):
    """The four callers that already pre-guard: the surface stays fail-closed."""

    def test_minimal_payloads_fail_closed_without_raising(self):
        """The pre-guarded pair validators fail closed on minimal payloads without raising."""
        validators = (_valid_repetition, _valid_distraction, _valid_leading, _valid_dateline)
        for validator in validators:
            for bad in ([], {}):
                payload = {"available": True, "primary_decision": bad, "primary_winner": None}
                with self.subTest(validator=validator.__name__, bad=bad):
                    self.assertFalse(validator(payload))


class TestDatelineLiteralValidatorDrift(unittest.TestCase):
    """Item 3 evidence: producer weekday drift is caught by the literal validator."""

    def test_producer_weekday_drift_rejected(self):
        """Producer weekday-name drift is rejected by the dateline literal validator."""
        self.assertTrue(_valid_dateline(_base_payload()))
        original = dateline.WEEKDAY_NAMES
        swapped = list(original)
        swapped[2], swapped[3] = swapped[3], swapped[2]
        dateline.WEEKDAY_NAMES = tuple(swapped)
        try:
            drifted = _base_payload()
        finally:
            dateline.WEEKDAY_NAMES = original
        self.assertEqual(
            drifted["protocol"]["arm_dates"]["anchor_weekday_iso"], "Thursday, 2026-10-21"
        )
        self.assertFalse(_valid_dateline(drifted))
        self.assertFalse(diffreports.diff_dateline(_base_payload(), drifted)["available"] == "both")


if __name__ == "__main__":
    unittest.main()
