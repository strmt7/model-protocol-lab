"""Cycle 138: cross-checkpoint transfer experiment (IDEAS #1).

``experiments.transfer_portability`` (pure) re-applies the conservative
selection rule per transfer checkpoint and reports whether the primary
winner transfers, ties, or fails there. ``experiments.transfer_stage``
re-runs ``core.evaluate()`` for every candidate against each checkpoint
(search split only). Wired into ``mpl run --transfer-model`` (max 4),
the HTML "Cross-checkpoint transfer" section, and results.json.
"""

import unittest

from model_protocol_lab import core, diffreports, experiments
from model_protocol_lab.core import Fixture, Format, Message
from model_protocol_lab.reporting import _transfer_section

CANDIDATES = ("alpha", "beta")


def _portability(transfer, winner="alpha", decision="selected"):
    return experiments.transfer_portability(winner, decision, CANDIDATES, transfer)


class TransferPortabilityTest(unittest.TestCase):
    def test_full_transfer(self):
        out = _portability(
            {
                "ckptA": {"alpha": True, "beta": False},
                "ckptB": {"alpha": True, "beta": False},
            }
        )
        self.assertTrue(out["available"])
        self.assertEqual(out["transfer_verdict"], "full")
        self.assertEqual(
            out["per_checkpoint"]["ckptA"],
            {"decision": "selected", "winner": "alpha"},
        )
        self.assertEqual(out["matrix"]["alpha"]["ckptA"], "transfers")
        self.assertEqual(out["matrix"]["beta"]["ckptA"], "fails_there")

    def test_partial_transfer(self):
        out = _portability(
            {
                "ckptA": {"alpha": True, "beta": False},
                "ckptB": {"alpha": False, "beta": True},
            }
        )
        self.assertEqual(out["transfer_verdict"], "partial")
        self.assertEqual(out["matrix"]["alpha"]["ckptB"], "fails_there")
        self.assertEqual(out["matrix"]["beta"]["ckptB"], "wins_there")

    def test_no_transfer(self):
        out = _portability(
            {
                "ckptA": {"alpha": False, "beta": False},
                "ckptB": {"alpha": False, "beta": True},
            }
        )
        self.assertEqual(out["transfer_verdict"], "none")
        self.assertEqual(
            out["per_checkpoint"]["ckptA"],
            {"decision": "no_candidate", "winner": None},
        )

    def test_ties_there(self):
        out = _portability({"ckptA": {"alpha": True, "beta": True}})
        self.assertEqual(out["matrix"]["alpha"]["ckptA"], "ties_there")
        self.assertEqual(
            out["per_checkpoint"]["ckptA"],
            {"decision": "ambiguous", "winner": None},
        )
        # A tie is not a transfer.
        self.assertEqual(out["transfer_verdict"], "none")

    def test_incompatible_checkpoint(self):
        out = _portability(
            {
                "ckptA": {"alpha": True, "beta": False},
                "ckptB": {"alpha": None, "beta": None},
            }
        )
        self.assertEqual(out["matrix"]["alpha"]["ckptB"], "checkpoint_incompatible")
        self.assertEqual(
            out["per_checkpoint"]["ckptB"],
            {"decision": "unavailable", "winner": None},
        )
        # Only the evaluable checkpoint counts: full transfer.
        self.assertEqual(out["transfer_verdict"], "full")

    def test_no_primary_winner_unavailable(self):
        out = _portability({"ckptA": {"alpha": True, "beta": False}}, winner=None)
        self.assertEqual(out["transfer_verdict"], "unavailable")

    def test_all_incompatible_unavailable(self):
        out = _portability({"ckptA": {"alpha": None, "beta": None}})
        self.assertEqual(out["transfer_verdict"], "unavailable")

    def test_checkpoint_order_invariant(self):
        t1 = {
            "ckptA": {"alpha": True, "beta": False},
            "ckptB": {"alpha": False, "beta": True},
        }
        t2 = {
            "ckptB": {"alpha": False, "beta": True},
            "ckptA": {"alpha": True, "beta": False},
        }
        self.assertEqual(_portability(t1), _portability(t2))

    def test_malformed_input_unavailable(self):
        for bad in (
            ("alpha", "", ("alpha", "beta"), {"c": {"alpha": True}}),
            ("gamma", "selected", CANDIDATES, {"c": {"alpha": True}}),
            ("alpha", "selected", ("alpha", "alpha"), {"c": {"alpha": True}}),
            ("alpha", "selected", CANDIDATES, {"c": {"alpha": "yes"}}),
            ("alpha", "selected", CANDIDATES, {"c": "nope"}),
        ):
            winner, decision, cands, transfer = bad
            out = experiments.transfer_portability(winner, decision, cands, transfer)
            self.assertFalse(out["available"], bad)


def _fixture(fid, expected):
    return Fixture(
        id=fid,
        messages=(Message(role="user", content=f"q-{fid}"),),
        expected=expected,
        group=f"g-{fid}",
    )


def _format(name):
    return Format(name=name, bos="", user_prefix=f"[{name}]", assistant_prefix="A:", turn_end="\n")


class ScriptedBackend:
    """Deterministic backend; behavior per candidate: pass/fail/error."""

    fingerprint = "cycle138-transfer-scripted"

    def __init__(self, expected, behavior):
        self._expected = dict(expected)
        self._behavior = dict(behavior)

    @staticmethod
    def encode(text):
        return tuple(map(ord, text))

    @staticmethod
    def decode(ids):
        return "".join(map(chr, ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        candidate = next((n for n in CANDIDATES if f"[{n}]" in prompt), None)
        fixture = next((fid for fid in self._expected if f"q-{fid}" in prompt), None)
        if candidate is None or fixture is None:
            return core.Generation((), "error", "unknown")
        mode = self._behavior.get(candidate, "pass")
        if mode == "error":
            return core.Generation((), "error", "boom")
        text = self._expected[fixture]
        if mode == "fail":
            text += "WRONG"
        ids = self.encode(text + chr(7))
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


class TransferStageTest(unittest.TestCase):
    def _stage(self, behavior):
        expected = {"S1": "GREEN", "S2": "BLUE"}
        cases = tuple(_fixture(fid, ans) for fid, ans in expected.items())
        formats = tuple(_format(n) for n in CANDIDATES)

        def factory(cid: str) -> ScriptedBackend:
            return ScriptedBackend(expected, behavior.get(cid, {}))

        return experiments.transfer_stage(
            factory,
            tuple(behavior),
            formats,
            cases,
            boundary=(7,),
            max_new_tokens=32,
            seed=1,
        )

    def test_all_pass(self):
        out = self._stage({"ckptA": {"alpha": "pass", "beta": "fail"}})
        self.assertEqual(out, {"ckptA": {"alpha": True, "beta": False}})

    def test_backend_construction_failure_records_none(self):
        def factory(cid):
            raise RuntimeError("no such checkpoint")

        cases = (_fixture("S1", "GREEN"),)
        formats = (_format("alpha"),)
        out = experiments.transfer_stage(factory, ("ckptX",), formats, cases, boundary=(7,))
        self.assertEqual(out, {"ckptX": {"alpha": None}})

    def test_evaluation_error_records_none(self):
        out = self._stage({"ckptA": {"alpha": "error", "beta": "pass"}})
        self.assertIsNone(out["ckptA"]["alpha"])
        self.assertTrue(out["ckptA"]["beta"])

    def test_end_to_end_portability(self):
        outcomes = self._stage(
            {
                "ckptA": {"alpha": "pass", "beta": "fail"},
                "ckptB": {"alpha": "pass", "beta": "pass"},
            }
        )
        port = experiments.transfer_portability("alpha", "selected", CANDIDATES, outcomes)
        self.assertTrue(port["available"])
        # ckptA: alpha alone -> transfers; ckptB: tie -> ties_there.
        self.assertEqual(port["matrix"]["alpha"]["ckptA"], "transfers")
        self.assertEqual(port["matrix"]["alpha"]["ckptB"], "ties_there")
        self.assertEqual(port["transfer_verdict"], "partial")

    def test_validation(self):
        cases = (_fixture("S1", "GREEN"),)
        formats = (_format("alpha"),)

        def factory(cid: str) -> ScriptedBackend:
            return ScriptedBackend({"S1": "GREEN"}, {})

        for kwargs in (
            {"checkpoint_ids": ()},
            {"checkpoint_ids": ("a", "b", "c", "d", "e")},
            {"checkpoint_ids": ("a", "a")},
            {"boundary": ()},
            {"max_new_tokens": 0},
        ):
            base = {
                "checkpoint_ids": ("a",),
                "formats": formats,
                "cases": cases,
                "boundary": (7,),
            }
            base.update(kwargs)
            with self.assertRaises(ValueError, msg=str(kwargs)):
                experiments.transfer_stage(factory, **base)
        with self.assertRaises(ValueError):
            experiments.transfer_stage(None, ("a",), formats, cases, boundary=(7,))


class TransferHtmlTest(unittest.TestCase):
    def _transfer(self):
        return {
            "available": True,
            "primary_winner": "alpha",
            "checkpoints": [
                {"id": "ckptA", "path": "/models/a", "fingerprint": "fp123"},
                {"id": "ckptB", "path": "/models/b", "fingerprint": None},
            ],
            "per_checkpoint": {
                "ckptA": {"decision": "selected", "winner": "alpha"},
                "ckptB": {"decision": "ambiguous", "winner": None},
            },
            "matrix": {
                "alpha": {"ckptA": "transfers", "ckptB": "ties_there"},
                "beta": {"ckptA": "fails_there", "ckptB": "ties_there"},
            },
            "transfer_verdict": "partial",
        }

    def test_section_renders(self):
        html = _transfer_section({"transfer": self._transfer()})
        self.assertIn("Cross-checkpoint transfer", html)
        self.assertIn("ckptA", html)
        self.assertIn("transfers", html)
        self.assertIn("transfers to some checkpoints", html)

    def test_section_absent_when_missing(self):
        self.assertEqual(_transfer_section({}), "")
        self.assertEqual(_transfer_section({"transfer": {"available": False}}), "")
        self.assertEqual(_transfer_section({"transfer": {"available": True}}), "")


def _transfer_payload(verdict="full", winner="alpha"):
    return {
        "available": True,
        "primary_winner": "alpha",
        "transfer_verdict": verdict,
        "per_checkpoint": {
            "ckptA": {"decision": "selected", "winner": winner},
        },
        "matrix": {
            "alpha": {"ckptA": "transfers" if winner == "alpha" else "fails_there"},
            "beta": {"ckptA": "fails_there"},
        },
    }


class DiffTransferTest(unittest.TestCase):
    def test_identical_unchanged(self):
        diff = diffreports.diff_transfer(_transfer_payload(), _transfer_payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_verdict_change_detected(self):
        old = _transfer_payload(verdict="full")
        new = _transfer_payload(verdict="partial")
        diff = diffreports.diff_transfer(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["transfer_verdict"]["changed"])

    def test_checkpoint_drift_detected(self):
        old = _transfer_payload()
        new = _transfer_payload(winner="beta")
        new["matrix"]["alpha"]["ckptA"] = "fails_there"
        new["matrix"]["beta"]["ckptA"] = "wins_there"
        diff = diffreports.diff_transfer(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["checkpoint_drift"]), 1)
        self.assertEqual(diff["checkpoint_drift"][0]["checkpoint"], "ckptA")
        self.assertEqual(len(diff["cell_drift"]), 2)

    def test_checkpoint_added_removed(self):
        old = _transfer_payload()
        new = _transfer_payload()
        new["per_checkpoint"]["ckptB"] = {"decision": "selected", "winner": "alpha"}
        new["matrix"]["alpha"]["ckptB"] = "transfers"
        new["matrix"]["beta"]["ckptB"] = "fails_there"
        diff = diffreports.diff_transfer(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["checkpoints_added"], ["ckptB"])
        self.assertEqual(diff["checkpoints_removed"], [])

    def test_asymmetric(self):
        diff = diffreports.diff_transfer(_transfer_payload(), {"available": False})
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        diff = diffreports.diff_transfer(None, None)
        self.assertEqual(diff["available"], "neither")

    def test_feeds_identical(self):
        old = {"decision": "selected", "transfer": _transfer_payload()}
        new = {"decision": "selected", "transfer": _transfer_payload()}
        diff = diffreports.diff_reports(old, new)
        self.assertFalse(diff["summary"]["transfer_changed"])
        self.assertFalse(diff["summary"]["transfer_asymmetric"])
        # A changed transfer breaks identical.
        new2 = {"decision": "selected", "transfer": _transfer_payload(verdict="none")}
        diff2 = diffreports.diff_reports(old, new2)
        self.assertTrue(diff2["summary"]["transfer_changed"])
        self.assertFalse(diff2["summary"]["identical"])

    def test_mixed_key_types_fail_closed(self):
        # Hostile results.json with non-string checkpoint keys must not
        # crash the comparator; the malformed side degrades to asymmetric.
        good = _transfer_payload()
        hostile = _transfer_payload()
        hostile["per_checkpoint"][1] = {"decision": "selected", "winner": "alpha"}
        diff = diffreports.diff_transfer(good, hostile)
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        hostile2 = _transfer_payload()
        hostile2["matrix"]["alpha"][2] = "transfers"
        diff2 = diffreports.diff_transfer(good, hostile2)
        self.assertEqual(diff2["available"], "old")

    def test_transfer_portability_mixed_keys_fail_closed(self):
        # Never raises on malformed input, per the docstring contract.
        result = experiments.transfer_portability(
            "alpha",
            "selected",
            ("alpha", "beta"),
            {1: {"alpha": True, "beta": False}, "b": {"alpha": True, "beta": False}},
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["detail"], "invalid_candidates_or_transfer")


if __name__ == "__main__":
    unittest.main()
