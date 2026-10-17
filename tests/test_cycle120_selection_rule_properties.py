"""Cycle 120 -- Hypothesis property tests for the selection rule.

Scope
-----
``model_protocol_lab.experiments._select_winner`` is the pure core of the
lab's conservative selection rule: exactly one candidate passing every
search fixture wins; ties and failures abstain; a cancelled run reports
"cancelled". It was extracted from ``run_experiment`` (cycle 120) so the
decision table -- previously reachable only through heavyweight
backend-backed integration tests -- is exhaustively testable in
isolation.

Properties
----------
1. Vocabulary: the decision is always one of the four documented strings.
2. Cancel dominates: cancelled=True always yields ("cancelled", None),
   regardless of the winner list.
3. Unique winner: exactly one winner yields ("selected", that winner).
4. All fail: zero winners yields ("no_candidate", None).
5. Tie: two or more winners yields ("ambiguous", None).
6. Order invariance: permuting the winner list never changes the decision,
   and the selected candidate is identical.
7. Selection identity: the returned candidate is the same object as the
   unique winner passed in.

Mutation gate: 4 mutants -- (1) ``== 1`` -> ``>= 1`` (selects on ties),
(2) "ambiguous"/"no_candidate" swapped, (3) cancelled branch removed,
(4) always ("selected", winners[0]) when winners nonempty -- each killed
by >=1 property test.
"""

import unittest
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from model_protocol_lab.core import Format
from model_protocol_lab.experiments import _select_winner

_DECISIONS = frozenset({"selected", "ambiguous", "no_candidate", "cancelled"})


def _make_format(idx: int) -> Format:
    return Format(
        name=f"cand-{idx}",
        bos="",
        user_prefix="",
        assistant_prefix="",
        turn_end="",
    )


def _winner_lists() -> Any:
    """Lists of 0-6 distinct candidate Formats."""
    return st.lists(
        st.integers(min_value=0, max_value=20), min_size=0, max_size=6, unique=True
    ).map(lambda ids: [_make_format(i) for i in ids])


class TestSelectWinnerProperties(unittest.TestCase):
    @given(winners=_winner_lists(), cancelled=st.booleans())
    @settings(max_examples=200)
    def test_decision_vocabulary(self, winners: Any, cancelled: bool) -> None:
        """The decision is always one of the four documented strings."""
        decision, selected = _select_winner(winners, cancelled)
        self.assertIn(decision, _DECISIONS)
        if decision == "selected":
            self.assertIsNotNone(selected)
        else:
            self.assertIsNone(selected)

    @given(winners=_winner_lists())
    @settings(max_examples=200)
    def test_cancel_dominates(self, winners: Any) -> None:
        """cancelled=True always yields ('cancelled', None)."""
        decision, selected = _select_winner(winners, True)
        self.assertEqual(decision, "cancelled")
        self.assertIsNone(selected)

    @given(winners=_winner_lists())
    @settings(max_examples=200)
    def test_unique_winner_selected(self, winners: Any) -> None:
        """Exactly one winner is selected (not cancelled)."""
        if len(winners) != 1:
            return
        decision, selected = _select_winner(winners, False)
        self.assertEqual(decision, "selected")
        self.assertIs(selected, winners[0])

    @given(winners=_winner_lists())
    @settings(max_examples=200)
    def test_all_fail_abstains(self, winners: Any) -> None:
        """Zero winners yields 'no_candidate' (not cancelled)."""
        if winners:
            return
        decision, selected = _select_winner(winners, False)
        self.assertEqual(decision, "no_candidate")
        self.assertIsNone(selected)

    @given(winners=_winner_lists())
    @settings(max_examples=200)
    def test_tie_abstains(self, winners: Any) -> None:
        """Two or more winners yields 'ambiguous' (not cancelled)."""
        if len(winners) < 2:
            return
        decision, selected = _select_winner(winners, False)
        self.assertEqual(decision, "ambiguous")
        self.assertIsNone(selected)

    @given(winners=_winner_lists(), cancelled=st.booleans())
    @settings(max_examples=200)
    def test_order_invariance(self, winners: Any, cancelled: bool) -> None:
        """Permuting the winner list never changes the decision."""
        reversed_winners = list(reversed(winners))
        d1, s1 = _select_winner(winners, cancelled)
        d2, s2 = _select_winner(reversed_winners, cancelled)
        self.assertEqual(d1, d2)
        # The selected candidate must be the same object, not merely equal.
        if d1 == "selected":
            self.assertIs(s1, s2)
            self.assertIn(s1, winners)
            self.assertIn(s2, reversed_winners)

    @given(winners=_winner_lists())
    @settings(max_examples=200)
    def test_exhaustive_decision_table(self, winners: Any) -> None:
        """The not-cancelled decision is a pure function of winner count."""
        decision, _ = _select_winner(winners, False)
        if len(winners) == 0:
            expected = "no_candidate"
        elif len(winners) == 1:
            expected = "selected"
        else:
            expected = "ambiguous"
        self.assertEqual(decision, expected)


if __name__ == "__main__":
    unittest.main()
