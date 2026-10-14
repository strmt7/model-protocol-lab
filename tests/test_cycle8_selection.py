"""Cycle 8: selection-rule veto pins for ``run_experiment``.

Two behavioral properties of the conservative unique-winner rule that the
existing suite did not pin:

- a failed search evaluation in ANY candidate vetoes selection, even when
  an earlier-listed candidate would otherwise be the unique winner
  (``test_later_candidate_failure_vetoes_earlier_winner``);
- when the held-out phase fails after a unique search winner was found,
  the winner's name is still recorded while the decision records where it
  failed -- selection and decision are separate records
  (``test_held_out_failure_still_names_search_winner``).

Both pin the "failures abstain" contract against the two most plausible
regressions: skipping failed candidates and continuing to a selection, or
wiping the winner's name on held-out failure (which would destroy the
diagnostic that distinguishes "no candidate worked" from "the winner
failed held-out").
"""

import sys
import unittest
from dataclasses import replace

sys.path.insert(0, "tests")

from test_core import BASE, FakeBackend, ids  # noqa: E402
from test_round3 import split  # noqa: E402

from model_protocol_lab.core import Generation  # noqa: E402
from model_protocol_lab.experiments import run_experiment  # noqa: E402

OK = Generation(ids("ok|"), "limit")


class SelectionVetoTests(unittest.TestCase):
    def test_later_candidate_failure_vetoes_earlier_winner(self) -> None:
        """A failure in any candidate's search phase vetoes the whole run.

        The first candidate passes every search fixture (it would be the
        unique winner on its own); the second candidate's evaluation fails.
        The conservative rule must report ``"failed"`` -- not select the
        first candidate, not call it ambiguous, and never run held-out.
        """
        second = replace(BASE, name="second")
        backend = FakeBackend(
            generation=OK,
            generate_script=[OK, OK, Generation((), "error")],
        )
        result = run_experiment(
            backend, (BASE, second), split(), boundary=ids("|"), max_new_tokens=3
        )
        self.assertEqual(result["decision"], "failed")
        self.assertIsNone(result["selected_candidate"])
        self.assertIsNone(result["held_out"])
        self.assertIn("failure_taxonomy", result)

    def test_held_out_failure_still_names_search_winner(self) -> None:
        """Selection and decision are separate records.

        The unique search winner is named in ``selected_candidate`` even
        when the held-out phase then fails; the ``decision`` field records
        where it failed. This distinguishes "no candidate worked" from
        "the winner failed held-out" in saved reports.
        """
        bad = Generation(ids("BAD"), "limit")
        backend = FakeBackend(generation=OK, generate_script=[OK, OK, bad, bad])
        result = run_experiment(backend, (BASE,), split(), boundary=ids("|"), max_new_tokens=3)
        self.assertEqual(result["decision"], "held_out_failed")
        self.assertEqual(result["selected_candidate"], "base")
        self.assertIsNotNone(result["held_out"])
        self.assertIn("failure_taxonomy", result)


if __name__ == "__main__":
    unittest.main()
