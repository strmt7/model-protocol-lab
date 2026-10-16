"""Cycle-78 adversarial audit: ``run_experiment`` decision orchestration.

Target: ``model_protocol_lab.experiments.run_experiment`` (line 942 to end
of file) -- the decision heart of the product: per-candidate search
evaluation, the ``aggregate_eligible`` early-return gates, the conservative
unique-winner rule (``len(winners) != 1``), cancellation handling, the
held-out confirmation stage, and the final decision mapping. Cycle 8 covered
selection basics; this cycle audits the orchestration branches deeply.

BRANCH MAP (hand-derived from the source, not from running it):

    for fmt in formats:                                   # candidate loop, in order
        report = evaluate(backend, fmt, split.search, ...)  # per-candidate boundary
        output["search"].append({...})                    # record always appended
        if not report.aggregate_eligible:                 # GATE 1 (early return)
            decision = report.status                      # "cancelled" | "failed"
            return finish()
        if summary["exact_and_boundary_passes"] == report.planned_count:
            winners.append(fmt)                           # needs EVERY search fixture
    if event.is_set():                                    # GATE 2 (post-loop race)
        decision = "cancelled"; return finish()
    if len(winners) != 1:                                 # GATE 3 (unique-winner rule)
        decision = "ambiguous" if winners else "no_candidate"; return finish()
    selected = winners[0]                                 # first (only) winner
    held = evaluate(backend, selected, split.held_out, ...)  # winner's own boundary
    if not held.aggregate_eligible:                       # GATE 4 (held passthrough)
        decision = held.status                             # "cancelled" | "failed"
    else:
        decision = ("compatible_on_fixture_set"           # every held-out obs passes
                    if all(o.exact_match and o.boundary_found for o in held.observations)
                    else "held_out_failed")
    finish() attaches failure_taxonomy(output, eos_id) on EVERY return path.

Method:
  1. Scripted fake backends (chr-codec, deterministic, no models/network)
     control per-(candidate, fixture) search outcomes and held-out outcomes
     independently.
  2. Decision-branch coverage: every gate above is hit with a hand-derived
     expected decision.
  3. Minimal pairs: flipping one candidate's one observation flips the
     decision; the ambiguous/unique-winner boundary is exactly one winner.
  4. Adversarial edges: empty candidate list, duplicate names (must raise),
     single candidate, cancellation racing the held-out stage, backend
     raising mid-evaluation (fail-closed fixed codes, no exception text
     leaks).
  5. Independence: expected decisions are hand-derived from the branch map.
     The tests never re-run the source's decision logic to build
     expectations.

All backends/tokenizers here are synthetic and scripted; no model
weights, no network, no downloads.
"""

import json
import re
import unittest
from threading import Event
from typing import Any

from model_protocol_lab.core import Fixture, Format, Generation, Message, Split
from model_protocol_lab.experiments import run_experiment

BOUNDARY = (777001,)  # default boundary; huge ids never emitted by the chr-codec
ALT_BOUNDARY = (111,)  # per-candidate override used in the boundary-wiring test
PAD_ID = 32  # chr(32) == " ": padding never collides with the boundary ids
CAP = 24  # max_new_tokens: comfortably above any scripted answer + boundary

# Outcome vocabulary for the scripted backend, keyed by (candidate, fixture):
#   "pass"        -> exact answer + boundary present (counts toward winner)
#   "wrong"       -> boundary present, wrong answer
#   "no_boundary" -> right answer text, boundary absent
#   "both"        -> wrong answer, boundary absent
#   ("error", code)        -> generate() returns reason="error"
#   ("raise", stage)       -> stage in {"encode", "generate", "decode"} raises
#   ("pass_then_set_event",) -> behaves like "pass", then sets the cancel event


def make_candidate(name: str) -> Format:
    """A candidate whose rendered prompt carries a routable ``<cand:NAME>`` marker."""
    return Format(
        name=name,
        bos=f"<cand:{name}>",
        user_prefix="U:",
        assistant_prefix="A:",
        turn_end=";",
    )


def make_fixtures(tag: str, count: int) -> tuple[Fixture, ...]:
    """Fixtures whose message content carries a routable ``<fx:ID>`` marker."""
    return tuple(
        Fixture(
            id=f"{tag}{j}",
            messages=(Message(role="user", content=f"<fx:{tag}{j}> probe {j}"),),
            expected=f"ANS-{tag}{j}",
            group=f"cycle78-{tag}{j}",
        )
        for j in range(count)
    )


def make_split(n_search: int = 2, n_held: int = 2) -> Split:
    return Split(search=make_fixtures("s", n_search), held_out=make_fixtures("h", n_held))


def expected_map(split: Split) -> dict[str, str]:
    return {case.id: case.expected for case in split.search + split.held_out}


class ScriptedBackend:
    """Deterministic fake backend: generate() is a pure function of the script.

    encode/decode are a chr-codec (no implicit BOS/EOS). The rendered prompt
    carries ``<cand:NAME>`` (from the format's bos marker) and ``<fx:ID>``
    (from the fixture message), so every backend call is routed to the
    scripted outcome for that (candidate, fixture) pair. Search and held-out
    outcomes are scripted independently through the same table.
    """

    fingerprint = "cycle78-audit-scripted"

    def __init__(
        self,
        outcomes: dict[tuple[str, str], Any],
        expected: dict[str, str],
        emit_boundary: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        self.outcomes = outcomes
        self.expected = expected
        self.emit_boundary = emit_boundary or {}
        self.generate_calls: list[tuple[str, str]] = []
        self.encode_calls = 0
        self.decode_calls = 0
        self._last: tuple[str, str] | None = None

    @staticmethod
    def _route(text: str) -> tuple[str, str]:
        cand = re.search(r"<cand:([^>]+)>", text)
        fx = re.search(r"<fx:([^>]+)>", text)
        assert cand is not None and fx is not None, f"unroutable prompt: {text!r}"
        return cand.group(1), fx.group(1)

    def encode(self, text: str) -> tuple[int, ...]:
        self.encode_calls += 1
        outcome = self.outcomes[self._route(text)]
        if isinstance(outcome, tuple) and outcome[:2] == ("raise", "encode"):
            raise RuntimeError("scripted encode boom")
        return tuple(ord(c) for c in text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        self.decode_calls += 1
        outcome = self.outcomes.get(self._last) if self._last else None
        if isinstance(outcome, tuple) and outcome[:2] == ("raise", "decode"):
            raise RuntimeError("scripted decode boom")
        # Tolerant codec: boundary ids are huge; map anything outside the
        # chr range instead of raising (the diagnostic path degrades anyway).
        return "".join(chr(t) if 0 <= t <= 0x10FFFF else "\ufffd" for t in token_ids)

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        text = "".join(chr(t) for t in input_ids)  # raw decode: no counting, no raising
        cand, fx = self._route(text)
        self._last = (cand, fx)
        self.generate_calls.append((cand, fx))
        outcome = self.outcomes[(cand, fx)]
        if isinstance(outcome, tuple):
            if outcome[:2] == ("raise", "generate"):
                raise RuntimeError("scripted generate boom")
            if outcome[0] == "error":
                return Generation(
                    token_ids=(PAD_ID,) * max_new_tokens, reason="error", error=outcome[1]
                )
            if outcome[0] == "pass_then_set_event":
                cancel.set()
                outcome = "pass"
        answer = self.expected[fx] if outcome in ("pass", "no_boundary") else f"WRONG-{fx}"
        ids = tuple(ord(c) for c in answer)
        if outcome in ("pass", "wrong"):
            ids = ids + self.emit_boundary.get(cand, BOUNDARY)
        assert len(ids) <= max_new_tokens, "scripted answer exceeds the token cap"
        return Generation(token_ids=ids + (PAD_ID,) * (max_new_tokens - len(ids)), reason="limit")


class ArmingEvent(Event):
    """is_set() returns False exactly ``budget`` times, then True forever.

    evaluate() calls event.is_set() exactly 5 times per completed fixture
    (top-of-loop, after render, after encode, after generate, after decode)
    plus once more for its final status check, and the scripted backend never
    touches the event. A budget of C*(5*S+1) therefore arms the event for the
    first check AFTER the search phase: run_experiment's own post-loop
    ``if event.is_set()``. Deterministic -- no threads, no sleeps.
    """

    def __init__(self, budget: int) -> None:
        super().__init__()
        self._budget = budget

    def is_set(self) -> bool:
        if self._budget > 0:
            self._budget -= 1
            return False
        return True


def run_audit(
    outcomes: dict[tuple[str, str], Any],
    names: tuple[str, ...] = ("cand-a", "cand-b"),
    n_search: int = 2,
    n_held: int = 2,
    **kwargs: Any,
) -> tuple[dict, ScriptedBackend]:
    """Build fixtures/candidates/backend from a script and run the experiment."""
    split = make_split(n_search, n_held)
    backend = ScriptedBackend(outcomes, expected_map(split))
    formats = tuple(make_candidate(n) for n in names)
    result = run_experiment(
        backend, formats, split, boundary=BOUNDARY, max_new_tokens=CAP, **kwargs
    )
    return result, backend


def full_pass(names: tuple[str, ...], split: Split) -> dict[tuple[str, str], Any]:
    """Every (candidate, fixture) passes, in both phases."""
    return {(n, c.id): "pass" for n in names for c in split.search + split.held_out}


class DecisionBranchTests(unittest.TestCase):
    """GATE 3 + held-out stage: the unique-winner rule and decision mapping."""

    def test_unique_winner_held_out_pass_maps_compatible(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-a", "cand-b"), split)
        outcomes[("cand-b", "s1")] = "wrong"  # cand-a the unique all-search passer
        result, backend = run_audit(outcomes)
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        self.assertEqual(result["selected_candidate"], "cand-a")
        self.assertIsNotNone(result["held_out"])
        self.assertEqual(len(result["search"]), 2)
        # Winner evaluated on search AND held-out; loser on search only.
        self.assertEqual(
            backend.generate_calls.count(("cand-a", "h0")),
            1,
            "winner must reach the held-out stage",
        )
        self.assertNotIn(("cand-b", "h0"), backend.generate_calls)

    def test_unique_winner_held_out_fail_maps_held_out_failed(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-a", "cand-b"), split)
        outcomes[("cand-b", "s1")] = "wrong"
        outcomes[("cand-a", "h1")] = "wrong"  # winner diverges on held-out
        result, _ = run_audit(outcomes)
        self.assertEqual(result["decision"], "held_out_failed")
        self.assertEqual(result["selected_candidate"], "cand-a")
        self.assertIsNotNone(result["held_out"])

    def test_zero_winners_maps_no_candidate(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-a", "cand-b"), split)
        outcomes[("cand-a", "s0")] = "wrong"
        outcomes[("cand-b", "s1")] = "no_boundary"  # partial passers are not winners
        result, backend = run_audit(outcomes)
        self.assertEqual(result["decision"], "no_candidate")
        self.assertIsNone(result["selected_candidate"])
        self.assertIsNone(result["held_out"])
        # No winner -> the held-out stage never runs for anyone.
        self.assertFalse(any(fx.startswith("h") for _, fx in backend.generate_calls))

    def test_two_winners_maps_ambiguous(self) -> None:
        split = make_split()
        result, backend = run_audit(full_pass(("cand-a", "cand-b"), split))
        self.assertEqual(result["decision"], "ambiguous")
        self.assertIsNone(result["selected_candidate"])
        self.assertIsNone(result["held_out"])
        self.assertFalse(any(fx.startswith("h") for _, fx in backend.generate_calls))

    def test_three_winners_maps_ambiguous(self) -> None:
        split = make_split()
        result, _ = run_audit(
            full_pass(("cand-a", "cand-b", "cand-c"), split), names=("cand-a", "cand-b", "cand-c")
        )
        self.assertEqual(result["decision"], "ambiguous")
        self.assertIsNone(result["selected_candidate"])

    def test_held_out_requires_every_observation(self) -> None:
        # Held-out uses all(): one wrong answer among passes -> held_out_failed.
        split = make_split(n_held=3)
        outcomes = full_pass(("cand-a", "cand-b"), split)
        outcomes[("cand-b", "s0")] = "both"
        outcomes[("cand-a", "h2")] = "no_boundary"
        result, _ = run_audit(outcomes, n_held=3)
        self.assertEqual(result["decision"], "held_out_failed")


class EarlyReturnTests(unittest.TestCase):
    """GATE 1: an ineligible aggregate short-circuits the candidate loop."""

    def test_failed_first_candidate_skips_rest(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-bad", "cand-a"), split)
        outcomes[("cand-bad", "s0")] = ("raise", "generate")
        result, backend = run_audit(outcomes, names=("cand-bad", "cand-a"))
        # Hand-derived: cand-bad's report is failed -> not aggregate_eligible ->
        # decision = report.status = "failed"; cand-a is never evaluated.
        self.assertEqual(result["decision"], "failed")
        self.assertIsNone(result["selected_candidate"])
        self.assertIsNone(result["held_out"])
        self.assertEqual(len(result["search"]), 1)
        self.assertFalse(any(c == "cand-a" for c, _ in backend.generate_calls))

    def test_failed_last_candidate_discards_would_be_winner(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-a", "cand-bad"), split)
        outcomes[("cand-bad", "s1")] = ("error", "boom")  # sanitized backend error
        result, backend = run_audit(outcomes, names=("cand-a", "cand-bad"))
        # Hand-derived: cand-a passed everything and would win, but cand-bad's
        # failed report hits GATE 1 first -> decision "failed", no held-out.
        self.assertEqual(result["decision"], "failed")
        self.assertIsNone(result["selected_candidate"])
        self.assertIsNone(result["held_out"])
        self.assertFalse(any(fx.startswith("h") for _, fx in backend.generate_calls))
        summary = result["search"][1]["summary"]
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["failure_code"], "backend_reported_error")

    def test_error_code_allowlist_passthrough(self) -> None:
        # An allowlisted backend error code survives; anything else collapses.
        split = make_split()
        outcomes = full_pass(("cand-a",), split)
        outcomes[("cand-a", "s0")] = ("error", "context_limit_exceeded")
        result, _ = run_audit(outcomes, names=("cand-a",))
        self.assertEqual(result["decision"], "failed")
        self.assertEqual(result["search"][0]["summary"]["failure_code"], "context_limit_exceeded")

    def test_cancelled_report_midloop_early_return(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-a", "cand-b"), split)
        outcomes[("cand-a", "s1")] = ("pass_then_set_event",)
        result, backend = run_audit(outcomes)
        # Hand-derived: cand-a's report is cancelled mid-evaluation ->
        # GATE 1 -> decision = report.status = "cancelled"; cand-b skipped.
        self.assertEqual(result["decision"], "cancelled")
        self.assertIsNone(result["selected_candidate"])
        self.assertFalse(any(c == "cand-b" for c, _ in backend.generate_calls))


class CancellationTests(unittest.TestCase):
    """GATE 2 and the cancel=Event plumbing: pre-set, mid-loop, post-loop race."""

    def test_cancel_set_before_loop(self) -> None:
        split = make_split()
        event = Event()
        event.set()
        result, backend = run_audit(full_pass(("cand-a", "cand-b"), split), cancel=event)
        # Hand-derived: the first evaluate() sees the set event before any
        # fixture -> cancelled report with zero observations -> GATE 1.
        self.assertEqual(result["decision"], "cancelled")
        self.assertEqual(backend.encode_calls, 0, "no backend work may happen")
        self.assertEqual(result["search"][0]["summary"]["attempted"], 0)

    def test_cancel_between_loop_and_decision(self) -> None:
        # The event arms AFTER the search phase: evaluate() returned a
        # completed report, so GATE 1 did not fire; GATE 2 must catch it.
        # 1 candidate x (5 calls x 2 fixtures + 1 final check) = 11.
        split = make_split()
        event = ArmingEvent(budget=11)
        result, backend = run_audit(full_pass(("cand-a",), split), names=("cand-a",), cancel=event)
        self.assertEqual(result["decision"], "cancelled")
        self.assertIsNone(result["selected_candidate"])
        self.assertIsNone(result["held_out"])
        self.assertEqual(len(backend.generate_calls), 2, "held-out must never run")
        self.assertTrue(event.is_set())

    def test_cancel_racing_held_out_stage(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-a", "cand-b"), split)
        outcomes[("cand-b", "s0")] = "wrong"
        outcomes[("cand-a", "h0")] = ("pass_then_set_event",)  # winner selected, then cancel
        result, backend = run_audit(outcomes)
        # Hand-derived: held-out evaluate() terminates cancelled -> GATE 4
        # passthrough -> decision = held.status = "cancelled".
        self.assertEqual(result["decision"], "cancelled")
        self.assertEqual(result["selected_candidate"], "cand-a")
        self.assertIsNotNone(result["held_out"])
        self.assertEqual(result["held_out"]["summary"]["status"], "cancelled")
        # Only the first held-out fixture ran before cancellation hit.
        self.assertNotIn(("cand-a", "h1"), backend.generate_calls)

    def test_invalid_cancel_type_raises(self) -> None:
        split = make_split()
        backend = ScriptedBackend(full_pass(("cand-a",), split), expected_map(split))
        with self.assertRaises(ValueError):
            run_experiment(
                backend,
                (make_candidate("cand-a"),),
                split,
                boundary=BOUNDARY,
                cancel="not-an-event",  # type: ignore[arg-type]
            )


class HeldOutPassthroughTests(unittest.TestCase):
    """GATE 4: an ineligible held-out report passes its status through."""

    def test_held_out_backend_raise_passthrough(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-a", "cand-b"), split)
        outcomes[("cand-b", "s1")] = "wrong"
        outcomes[("cand-a", "h0")] = ("raise", "generate")
        result, _ = run_audit(outcomes)
        self.assertEqual(result["decision"], "failed")
        self.assertEqual(result["selected_candidate"], "cand-a")
        self.assertIsNotNone(result["held_out"])
        self.assertEqual(result["held_out"]["summary"]["status"], "failed")

    def test_held_out_error_reason_passthrough(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-a", "cand-b"), split)
        outcomes[("cand-b", "s1")] = "wrong"
        outcomes[("cand-a", "h1")] = ("error", "boom")
        result, _ = run_audit(outcomes)
        self.assertEqual(result["decision"], "failed")
        self.assertEqual(result["held_out"]["summary"]["failure_code"], "backend_reported_error")


class MinimalPairTests(unittest.TestCase):
    """One observation flips the decision; ambiguous <-> unique is one winner."""

    def test_one_observation_flips_ambiguous_to_unique_winner(self) -> None:
        split = make_split()
        both_pass = full_pass(("cand-a", "cand-b"), split)
        before, _ = run_audit(dict(both_pass))
        self.assertEqual(before["decision"], "ambiguous")
        flipped = dict(both_pass)
        flipped[("cand-b", "s1")] = "wrong"  # exactly one observation changes
        after, _ = run_audit(flipped)
        self.assertEqual(after["decision"], "compatible_on_fixture_set")
        self.assertEqual(after["selected_candidate"], "cand-a")

    def test_one_observation_flips_no_candidate_to_unique_winner(self) -> None:
        split = make_split()
        neither = full_pass(("cand-a", "cand-b"), split)
        neither[("cand-a", "s1")] = "wrong"
        neither[("cand-b", "s1")] = "wrong"
        before, _ = run_audit(dict(neither))
        self.assertEqual(before["decision"], "no_candidate")
        flipped = dict(neither)
        flipped[("cand-a", "s1")] = "pass"  # exactly one observation changes
        after, _ = run_audit(flipped)
        self.assertEqual(after["decision"], "compatible_on_fixture_set")
        self.assertEqual(after["selected_candidate"], "cand-a")

    def test_ambiguous_boundary_is_exactly_one_winner(self) -> None:
        split = make_split()
        names = ("cand-a", "cand-b", "cand-c")
        all_pass = full_pass(names, split)
        three, _ = run_audit(dict(all_pass), names=names)
        self.assertEqual(three["decision"], "ambiguous")  # 3 winners
        two = dict(all_pass)
        two[("cand-c", "s0")] = "wrong"
        two_winners, _ = run_audit(two, names=names)
        self.assertEqual(two_winners["decision"], "ambiguous")  # 2 winners: still ambiguous
        one = dict(two)
        one[("cand-b", "s0")] = "wrong"
        one_winner, _ = run_audit(one, names=names)
        self.assertEqual(one_winner["decision"], "compatible_on_fixture_set")  # 1 winner
        self.assertEqual(one_winner["selected_candidate"], "cand-a")

    def test_one_held_out_observation_flips_compatible_to_failed(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-a", "cand-b"), split)
        outcomes[("cand-b", "s0")] = "wrong"
        before, _ = run_audit(dict(outcomes))
        self.assertEqual(before["decision"], "compatible_on_fixture_set")
        flipped = dict(outcomes)
        flipped[("cand-a", "h1")] = "wrong"  # exactly one observation changes
        after, _ = run_audit(flipped)
        self.assertEqual(after["decision"], "held_out_failed")


class AdversarialEdgeTests(unittest.TestCase):
    """Validation gates, single candidate, fail-closed backends, boundary wiring."""

    def test_empty_candidate_tuple_raises(self) -> None:
        split = make_split()
        backend = ScriptedBackend({}, expected_map(split))
        with self.assertRaises(ValueError):
            run_experiment(backend, (), split, boundary=BOUNDARY)

    def test_candidate_list_not_tuple_raises(self) -> None:
        split = make_split()
        backend = ScriptedBackend({}, expected_map(split))
        with self.assertRaises(ValueError):
            run_experiment(backend, [make_candidate("cand-a")], split, boundary=BOUNDARY)  # type: ignore[arg-type]

    def test_duplicate_candidate_names_raise(self) -> None:
        split = make_split()
        backend = ScriptedBackend({}, expected_map(split))
        with self.assertRaises(ValueError):
            run_experiment(
                backend,
                (make_candidate("dup"), make_candidate("dup")),
                split,
                boundary=BOUNDARY,
            )

    def test_single_candidate_compatible(self) -> None:
        split = make_split()
        result, _ = run_audit(full_pass(("only",), split), names=("only",))
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        self.assertEqual(result["selected_candidate"], "only")

    def test_single_candidate_no_candidate(self) -> None:
        split = make_split()
        outcomes = full_pass(("only",), split)
        outcomes[("only", "s0")] = "wrong"
        result, _ = run_audit(outcomes, names=("only",))
        self.assertEqual(result["decision"], "no_candidate")

    def test_single_candidate_failed(self) -> None:
        split = make_split()
        outcomes = full_pass(("only",), split)
        outcomes[("only", "s1")] = ("raise", "generate")
        result, _ = run_audit(outcomes, names=("only",))
        self.assertEqual(result["decision"], "failed")

    def test_generate_raise_is_fail_closed(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-a", "cand-b"), split)
        outcomes[("cand-a", "s1")] = ("raise", "generate")
        result, _ = run_audit(outcomes)
        self.assertEqual(result["decision"], "failed")
        self.assertEqual(result["search"][0]["summary"]["failure_code"], "generate_failed")
        # The raw exception text must not leak into the exported result.
        self.assertNotIn("scripted generate boom", json.dumps(result))

    def test_encode_raise_is_fail_closed(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-a",), split)
        outcomes[("cand-a", "s0")] = ("raise", "encode")
        result, _ = run_audit(outcomes, names=("cand-a",))
        self.assertEqual(result["decision"], "failed")
        self.assertEqual(result["search"][0]["summary"]["failure_code"], "encode_failed")
        self.assertNotIn("scripted encode boom", json.dumps(result))

    def test_decode_raise_is_fail_closed(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-a",), split)
        outcomes[("cand-a", "s0")] = ("raise", "decode")
        result, _ = run_audit(outcomes, names=("cand-a",))
        self.assertEqual(result["decision"], "failed")
        self.assertEqual(result["search"][0]["summary"]["failure_code"], "decode_failed")
        self.assertNotIn("scripted decode boom", json.dumps(result))

    def test_held_out_uses_winner_own_boundary(self) -> None:
        # boundary_map overrides cand-a's boundary; the backend emits the
        # overridden boundary for cand-a. If the held-out stage used the
        # default boundary instead, no boundary would be found -> held_out_failed.
        split = make_split()
        backend = ScriptedBackend(
            full_pass(("cand-a", "cand-b"), split),
            expected_map(split),
            emit_boundary={"cand-a": ALT_BOUNDARY, "cand-b": BOUNDARY},
        )
        backend.outcomes[("cand-b", "s1")] = "wrong"
        formats = (make_candidate("cand-a"), make_candidate("cand-b"))
        result = run_experiment(
            backend,
            formats,
            split,
            boundary=BOUNDARY,
            boundary_map={"cand-a": ALT_BOUNDARY},
            max_new_tokens=CAP,
        )
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        self.assertEqual(result["selected_candidate"], "cand-a")
        self.assertEqual(result["selected_boundary_token_ids"], [111])
        self.assertEqual(result["search"][0]["boundary_token_ids"], [111])
        self.assertEqual(result["search"][1]["boundary_token_ids"], [777001])

    def test_nonstring_backend_fingerprint_raises(self) -> None:
        split = make_split()

        class BadFingerprint(ScriptedBackend):
            fingerprint = 123  # type: ignore[assignment]

        backend = BadFingerprint(full_pass(("cand-a",), split), expected_map(split))
        with self.assertRaises(ValueError):
            # Intentional protocol violation: non-string fingerprint.
            run_experiment(
                backend,  # type: ignore
                (make_candidate("cand-a"),),
                split,
                boundary=BOUNDARY,
            )

    def test_invalid_boundary_raises(self) -> None:
        split = make_split()
        backend = ScriptedBackend(full_pass(("cand-a",), split), expected_map(split))
        with self.assertRaises(ValueError):
            run_experiment(backend, (make_candidate("cand-a"),), split, boundary=())


class AssemblyTests(unittest.TestCase):
    """Result assembly: per-candidate records and finish() on every path."""

    def test_search_record_shape(self) -> None:
        split = make_split()
        outcomes = full_pass(("cand-a", "cand-b"), split)
        outcomes[("cand-b", "s1")] = "wrong"
        result, _ = run_audit(outcomes)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["selection_rule"], "unique_all_search_pass")
        entry = result["search"][0]
        self.assertEqual(entry["format"]["name"], "cand-a")
        self.assertTrue(entry["candidate_fingerprint"].startswith("sha256:"))
        self.assertEqual(entry["boundary_token_ids"], [777001])
        summary = entry["summary"]
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["planned"], 2)
        self.assertEqual(summary["attempted"], 2)
        self.assertTrue(summary["aggregate_eligible"])
        self.assertEqual(summary["exact_and_boundary_passes"], 2)
        self.assertEqual(len(entry["observations"]), 2)
        obs = entry["observations"][0]
        self.assertTrue(obs["exact_match"])
        self.assertTrue(obs["boundary_found"])
        self.assertEqual(obs["status"], "completed")

    def test_finish_attaches_taxonomy_on_every_path(self) -> None:
        scenarios = []
        split = make_split()
        scenarios.append(full_pass(("cand-a", "cand-b"), split))  # ambiguous
        no_cand = full_pass(("cand-a", "cand-b"), split)
        no_cand[("cand-a", "s0")] = "wrong"
        no_cand[("cand-b", "s0")] = "wrong"
        scenarios.append(no_cand)  # no_candidate
        compat = full_pass(("cand-a", "cand-b"), split)
        compat[("cand-b", "s1")] = "wrong"
        scenarios.append(compat)  # compatible_on_fixture_set
        held_fail = dict(compat)
        held_fail[("cand-a", "h0")] = "wrong"
        scenarios.append(held_fail)  # held_out_failed
        failed = full_pass(("cand-a",), split)
        failed[("cand-a", "s0")] = ("raise", "generate")
        scenarios.append(failed)  # failed
        decisions = set()
        for i, outcomes in enumerate(scenarios):
            names = ("cand-a",) if i == 4 else ("cand-a", "cand-b")
            result, _ = run_audit(dict(outcomes), names=names)
            self.assertIn("failure_taxonomy", result, f"scenario {i} missing taxonomy")
            self.assertNotEqual(result["decision"], "not_started")
            decisions.add(result["decision"])
        self.assertEqual(
            decisions,
            {"ambiguous", "no_candidate", "compatible_on_fixture_set", "held_out_failed", "failed"},
        )


if __name__ == "__main__":
    unittest.main()
