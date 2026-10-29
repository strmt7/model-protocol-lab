"""Cycle 69: adversarial audit of the run-command diagnostic paths.

Scope: the three diagnostics wired into ``mpl run`` (cli.py) that Cycle 68
skipped as "compute-heavy". This audit drives all of them through the Python
API with tiny synthetic inputs so each stays fast (no model downloads, no
heavy compute).

Contracts under audit (read from the sources, not assumed):

* ``--sensitivity-sweep`` -> ``experiments.prompt_sensitivity_sweep``
  (experiments.py:789). Re-runs the search stage once natively and once per
  drop depth 1..``max_drop_bytes`` under ``TruncatedPromptBackend``
  (backends.py:678), which truncates every prompt mid-token. Per depth it
  records decision/selected winner and ``winner_changed = winner !=
  native_winner OR decision != native_decision``; ``flips_at`` lists the
  depths that changed. Diagnostic only; feeds ``reporting._sweep_section``
  (reporting.py:801).
* ``--logprob-diagnostic`` -> ``experiments.logprob_diagnostic``
  (experiments.py:703). One forward pass per candidate x fixture
  teacher-forcing the fixture's *expected answer* after the recorded prompt
  (``scoring.score_target_logprobs``). Aggregates only ``ok`` engine scores
  (token-count-weighted mean); interpretation bands are declared constants:
  mean >= -2.0 reads "greedy derailed", mean < -5.0 reads "weights don't
  know", between the bands is indeterminate. Diagnostic only; feeds
  ``reporting._logprob_section`` (reporting.py:920).
* (no flag; always on) -> ``experiments.budget_frontier``
  (experiments.py:275). Post-hoc and exact, zero model calls: from the
  recorded search observations alone it recomputes, for every smaller token
  budget, each candidate's pass counts and the selection decision.
  Per-observation minimal passing budget = first-boundary-cut +
  len(boundary); ``decision_at(b)`` is unique-all-pass -> selected, none ->
  no_candidate, several -> ambiguous; ``flip_ranges`` are the exact budget
  intervals where the decision differs from native; ``budget_robust`` is
  their absence. Runs for the completed decisions (compatible_on_fixture_set,
  held_out_failed, ambiguous, no_candidate); degrades to an "unavailable"
  marker otherwise (cli.py:681-713). Feeds the report's token-budget-frontier
  section (reporting.py:1787).

Method:

* Deterministic scripted backends (no model): a ``SeedSpyBackend`` whose
  ``generate`` inverts the lab's request-seed derivation (mirroring
  ``core.evaluate``) to recover the fixture, reads the candidate from a
  format marker, and returns output dictated by a behavior table the lab
  never sees; a prompt-tail-aware variant for the sweep; and a torch-logit
  backend prescribing exact per-fixture mean log-probabilities.
* INDEPENDENT hand-rolled oracles (pure ``math``/``struct``, never the
  implementation's helpers) for ``budget_frontier`` (full payload equality)
  and ``logprob_diagnostic`` (per-fixture scores, aggregates, band
  interpretation), plus minimal-pair tests asserting exactly the expected
  output delta for a one-input change.
* Adversarial: sweep on a perturbation that changes nothing (no false
  sensitivity flag); sweep on a prompt-tail perturbation (real flip
  detected); sweep on a winner-swap with unchanged decision (the OR in
  ``winner_changed`` must still flag); sweep where truncation splits a UTF-8
  codepoint (must record the failed depth, not skip or crash); logprob with
  identical logprobs across candidates; logprob exactly on the band edges
  (-2.0 -> derailed, -5.0 -> indeterminate); budget frontier at
  max_new_tokens=1 (budget 0 never appears); diagnostics on
  ambiguous/no_candidate decisions (must run, must not crash on the missing
  winner); empty observations / empty fixtures / backend without
  forward_logits.
* Mutation gate: 5 semantic mutants of experiments.py, each applied to its
  own /tmp package copy (repo sources never touched). Every mutant must be
  caught by at least one check; each check also runs against the pristine
  copy as a control.

Findings: CLEAN -- every diagnostic matches its independent oracle exactly
on all scenarios; all adversarial edge cases behave per contract; all 5
mutants are caught (see TestMutationGate). No genuine bugs found.

Deliberately NOT verified here: live-model backends (the audit uses
deterministic protocol-conformant scripted backends; the CLI wiring of the
flags to these exact functions is asserted structurally in
TestCliWiring, but no subprocess ``mpl run`` is executed), the report-HTML
rendering of the diagnostic sections, and statistical power of the sweep
(a methodology question, not a code-correctness one).
"""

from __future__ import annotations

import argparse
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from model_protocol_lab import core as real_core
from model_protocol_lab.cli import build_parser
from model_protocol_lab.experiments import (
    LOGPROB_DERAILED_MEAN_THRESHOLD,
    LOGPROB_UNKNOWN_MEAN_THRESHOLD,
    _aggregate_fixture_scores,
    _interpret_logprob_candidate,
    budget_frontier,
    logprob_diagnostic,
    prompt_sensitivity_sweep,
    run_experiment,
)

DERAILED = LOGPROB_DERAILED_MEAN_THRESHOLD  # -2.0
UNKNOWN = LOGPROB_UNKNOWN_MEAN_THRESHOLD  # -5.0

BOUNDARY_TEXT = "<|C69BND|>"
FILLER_ID = 32  # ord(" "): padding that can never contain the boundary
MAX_NEW_TOKENS = 64
VOCAB = 256  # scripted logit backend vocabulary size

SEARCH_SPECS = [("S1", "search question one?", "alpha"), ("S2", "search question two?", "beta")]
HELD_SPECS = [("H1", "held-out question one?", "delta")]


def _f32(value: float) -> float:
    """Independent float32 rounding (struct-based; the engine uses torch)."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _request_seed(exp_seed: int, fixture_id: str) -> int:
    """Independent re-implementation of core.evaluate's request-seed derivation."""
    material = f"{exp_seed}\0{fixture_id}".encode("utf-8", errors="surrogatepass")
    return int.from_bytes(sha256(material).digest()[:4], "big")


def _boundary_ids() -> tuple[int, ...]:
    return tuple(ord(c) for c in BOUNDARY_TEXT)


# ---------------------------------------------------------------------------
# Scripted backends
# ---------------------------------------------------------------------------


class SeedSpyBackend:
    """Deterministic Backend-protocol spy; output is a pure function of inputs.

    ``generate`` inverts the lab's request-seed derivation to recover the
    fixture, identifies the candidate from its format marker in the rendered
    prompt, and returns output dictated by the behavior table -- a table the
    lab under test never sees.
    """

    fingerprint = "cycle69-spy"

    def __init__(
        self,
        exp_seed: int,
        fixtures: dict[str, Any],
        behavior: dict[tuple[str, str], str],
        candidates: tuple[str, ...],
    ) -> None:
        self._fixtures = dict(fixtures)
        self._behavior = dict(behavior)
        self._candidates = tuple(candidates)
        self.seed_to_fid: dict[int, str] = {}
        for fid in self._fixtures:
            seed = _request_seed(exp_seed, fid)
            if seed in self.seed_to_fid:
                raise AssertionError(f"request-seed collision for fixture {fid!r}")
            self.seed_to_fid[seed] = fid

    # -- Backend protocol -------------------------------------------------
    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        return tuple(ord(c) for c in text)

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        return "".join(chr(i) for i in token_ids)

    def generate(
        self,
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Any,
    ) -> Any:
        prompt = self.decode(input_ids)
        candidate = next((n for n in self._candidates if f"[{n}]" in prompt), None)
        fixture_id = self.seed_to_fid.get(seed)
        if candidate is None or fixture_id is None:
            return real_core.Generation((), "error", "unknown_request")
        return real_core.Generation(
            self.expected_output_ids(candidate, fixture_id, max_new_tokens), "limit"
        )

    # -- oracle surface ----------------------------------------------------
    def expected_output_ids(
        self, candidate: str, fixture_id: str, max_new_tokens: int
    ) -> tuple[int, ...]:
        """The exact output IDs this backend must return for (candidate, fixture)."""
        mode = self._behavior.get((candidate, fixture_id), "pass")
        expected = self._fixtures[fixture_id].expected
        if mode == "pass":
            text = expected + BOUNDARY_TEXT
        elif mode == "fail_answer":
            text = expected + "WRONG" + BOUNDARY_TEXT
        elif mode == "fail_boundary":
            text = expected  # boundary never emitted: exact text, no boundary_found
        else:
            raise AssertionError(f"unknown behavior mode {mode!r}")
        assert BOUNDARY_TEXT not in expected, "fixture answer must not contain the boundary"
        ids = self.encode(text)
        assert len(ids) <= max_new_tokens, "fixture too long for the token cap"
        return ids + (FILLER_ID,) * (max_new_tokens - len(ids))


class PromptTailBackend(SeedSpyBackend):
    """Pass/fail depends on whether the prompt still ends with the native tail.

    Native prompts end with the assistant prefix (``A:``); the sweep's
    truncation drops tail bytes, so truncated prompts fail. ``swap=True``
    additionally inverts the pass pattern per candidate, producing a
    winner-swap with an unchanged decision.
    """

    fingerprint = "cycle69-tail"

    def __init__(self, *args: Any, swap: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._swap = swap

    def generate(
        self,
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Any,
    ) -> Any:
        prompt = self.decode(input_ids)
        candidate = next((n for n in self._candidates if f"[{n}]" in prompt), None)
        fixture_id = self.seed_to_fid.get(seed)
        if candidate is None or fixture_id is None:
            return real_core.Generation((), "error", "unknown_request")
        tail_ok = prompt.endswith("A:")
        ok = (candidate == "cand-A") == tail_ok if self._swap else tail_ok
        expected = self._fixtures[fixture_id].expected
        text = expected + BOUNDARY_TEXT if ok else expected + "WRONG" + BOUNDARY_TEXT
        ids = self.encode(text)
        assert len(ids) <= max_new_tokens
        return real_core.Generation(ids + (FILLER_ID,) * (max_new_tokens - len(ids)), "limit")


class LogitBackend(SeedSpyBackend):
    """Spy backend with a scripted ``forward_logits`` prescribing logprobs.

    ``forward_logits(input_ids)`` returns a (T, V) float64 torch tensor where
    row r assigns the prescribed log-probability to token ``input_ids[r+1]``
    (the causal-shift target); rows are never gathered for prompt positions,
    so the backend needs no knowledge of the prompt/target split. The
    prescribed *mean* logprob per (candidate, fixture) comes from
    ``means``; every token of that fixture scores the same value.
    """

    fingerprint = "cycle69-logit"

    def __init__(self, *args: Any, means: dict[tuple[str, str], float], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._means = dict(means)

    def _mean_for(self, input_ids: tuple[int, ...]) -> float:
        prompt = self.decode(input_ids)
        candidate = next((n for n in self._candidates if f"[{n}]" in prompt), None)
        for (cand, fid), mean in self._means.items():
            if cand != candidate:
                continue
            target_ids = self.encode(self._fixtures[fid].expected)
            if target_ids and tuple(input_ids[-len(target_ids) :]) == target_ids:
                return mean
        raise AssertionError("forward_logits called on an unrecognized prompt/target")

    def forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        seq = tuple(input_ids)
        if not seq or any(type(t) is not int or t < 0 or t >= VOCAB for t in seq):
            raise ValueError("invalid_token_ids")
        mean = self._mean_for(seq)
        prob = math.exp(mean)
        peak = math.log(prob * (VOCAB - 1) / (1 - prob))
        logits = torch.zeros((len(seq), VOCAB), dtype=torch.float64)
        for row in range(len(seq) - 1):
            logits[row, seq[row + 1]] = peak
        return logits


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------


def make_scenario(
    *,
    exp_seed: int,
    candidates: tuple[str, ...] = ("cand-A", "cand-B"),
    behavior: dict[tuple[str, str], str] | None = None,
    backend_cls: Any = SeedSpyBackend,
    backend_kwargs: dict[str, Any] | None = None,
    assistant_prefix: str = "A:",
    turn_end: str = "\n",
) -> Any:
    """Build fixtures, split, formats, and a scripted backend for one scenario."""
    from types import SimpleNamespace

    fixtures: dict[str, Any] = {}

    def _fixture(fid: str, text: str, expected: str) -> Any:
        fx = real_core.Fixture(
            id=fid,
            messages=(real_core.Message(role="user", content=text),),
            expected=expected,
            group=f"group-{fid}",
        )
        fixtures[fid] = fx
        return fx

    search = tuple(_fixture(fid, text, exp) for fid, text, exp in SEARCH_SPECS)
    held_out = tuple(_fixture(fid, text, exp) for fid, text, exp in HELD_SPECS)
    split = real_core.Split(search=search, held_out=held_out)
    formats = tuple(
        real_core.Format(
            name=name,
            bos="",
            user_prefix=f"[{name}]U:",
            assistant_prefix=assistant_prefix,
            turn_end=turn_end,
        )
        for name in candidates
    )
    backend = backend_cls(exp_seed, fixtures, behavior or {}, candidates, **(backend_kwargs or {}))
    return SimpleNamespace(
        exp_seed=exp_seed,
        candidates=candidates,
        split=split,
        formats=formats,
        backend=backend,
        fixtures=fixtures,
    )


def run_search_experiment(sc: Any) -> dict[str, Any]:
    return run_experiment(
        sc.backend,
        sc.formats,
        sc.split,
        boundary=_boundary_ids(),
        max_new_tokens=MAX_NEW_TOKENS,
        seed=sc.exp_seed,
    )


# ---------------------------------------------------------------------------
# Independent oracle: budget_frontier
# ---------------------------------------------------------------------------


def _oracle_first_boundary(tokens: tuple[int, ...], stop: tuple[int, ...]) -> int | None:
    """Independent first-occurrence scan (no use of core.first_boundary)."""
    stop = tuple(stop)
    for index in range(len(tokens) - len(stop) + 1):
        if tuple(tokens[index : index + len(stop)]) == stop:
            return index
    return None


def oracle_budget_frontier(
    search: list[dict[str, Any]],
    *,
    boundary: tuple[int, ...],
    max_new_tokens: int,
    native_decision: str,
    native_winner: str | None,
) -> dict[str, Any]:
    """Hand-rolled re-derivation of budget_frontier from its docstring."""
    names: list[str] = []
    per_candidate_min: dict[str, list[int | None]] = {}
    fixture_ids: list[str] = []
    for entry in search:
        name = entry["format"]["name"]
        names.append(name)
        entry_boundary = tuple(entry.get("boundary_token_ids") or boundary)
        mins: list[int | None] = []
        ids: list[str] = []
        for obs in entry["observations"]:
            ids.append(str(obs["fixture"]))
            minimum: int | None = None
            if (
                obs["status"] == "completed"
                and obs["exact_match"] is True
                and obs["boundary_found"] is True
            ):
                raw = obs.get("output_ids")
                oids = tuple(raw) if isinstance(raw, (list, tuple)) else ()
                if oids and all(type(t) is int for t in oids):
                    cut = _oracle_first_boundary(oids, entry_boundary)
                    if cut is not None:
                        minimum = cut + len(entry_boundary)
            mins.append(minimum)
        fixture_ids = ids
        per_candidate_min[name] = mins
    n_fixtures = len(fixture_ids)
    if native_decision in ("compatible_on_fixture_set", "held_out_failed"):
        native_search = ("selected", native_winner)
    else:
        native_search = (
            ("ambiguous", None) if native_decision == "ambiguous" else ("no_candidate", None)
        )

    sorted_mins = {
        name: sorted(mb for mb in mins if mb is not None)
        for name, mins in per_candidate_min.items()
    }

    def pass_count(name: str, budget: int) -> int:
        return sum(1 for mb in sorted_mins[name] if mb <= budget)

    candidates: dict[str, dict[str, Any]] = {}
    change_budgets: set[int] = set()
    for name in names:
        mins = per_candidate_min[name]
        marks = sorted({mb for mb in mins if mb is not None and 1 <= mb <= max_new_tokens})
        steps: list[list[int]] = []
        previous = -1
        seen: set[int] = set()
        for budget in [1, *marks, max_new_tokens]:
            if budget in seen:
                continue
            seen.add(budget)
            count = pass_count(name, budget)
            if count != previous:
                steps.append([budget, count])
                previous = count
            change_budgets.add(budget)
        candidates[name] = {
            "fixture_ids": fixture_ids,
            "min_budget_per_fixture": mins,
            "pass_steps": steps,
            "full_pass_budget": next((b for b, c in steps if c == n_fixtures), None),
        }

    def decision_at(budget: int) -> tuple[str, str | None]:
        winners = [n for n in names if pass_count(n, budget) == n_fixtures]
        if len(winners) == 1:
            return ("selected", winners[0])
        if not winners:
            return ("no_candidate", None)
        return ("ambiguous", None)

    decision_steps: list[list[Any]] = []
    previous_decision: tuple[str, str | None] | None = None
    for budget in sorted(change_budgets):
        current = decision_at(budget)
        if current != previous_decision:
            decision_steps.append([budget, current[0], current[1]])
            previous_decision = current
    ordered = sorted(change_budgets)
    flip_ranges: list[list[int]] = []
    for index, start in enumerate(ordered):
        end = ordered[index + 1] - 1 if index + 1 < len(ordered) else max_new_tokens
        if start > end or decision_at(start) == native_search:
            continue
        if flip_ranges and flip_ranges[-1][1] == start - 1:
            flip_ranges[-1][1] = end
        else:
            flip_ranges.append([start, end])
    return {
        "max_new_tokens": max_new_tokens,
        "native_decision": native_decision,
        "native_search_decision": native_search[0],
        "native_winner": native_search[1],
        "candidates": candidates,
        "decision_steps": decision_steps,
        "flip_ranges": flip_ranges,
        "budget_robust": not flip_ranges,
        "safe_budget_floor": (
            1
            if not flip_ranges
            else (flip_ranges[-1][1] + 1 if flip_ranges[-1][1] < max_new_tokens else None)
        ),
        "flip_attribution": [
            {
                "span": [lo, hi],
                "crossings": [
                    {"candidate": name, "fixture": fid, "min_budget": mb}
                    for mb, name, fid in sorted(
                        (mb, name, fid)
                        for name in sorted(candidates)
                        for fid, mb in zip(
                            candidates[name]["fixture_ids"],
                            candidates[name]["min_budget_per_fixture"],
                            strict=True,
                        )
                        if mb is not None and lo <= mb <= hi
                    )
                ],
            }
            for lo, hi in flip_ranges
        ],
    }


def _synth_obs(
    fixture: str,
    output_text: str | None,
    *,
    exact: bool,
    bfound: bool,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "fixture": fixture,
        "status": status,
        "exact_match": exact,
        "boundary_found": bfound,
        "output_ids": [ord(c) for c in output_text] if output_text is not None else [],
        "input_ids": [1, 2, 3],
    }


def _synth_entry(
    name: str, obs: list[dict[str, Any]], boundary_ids: tuple[int, ...] | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {"format": {"name": name}, "observations": obs}
    if boundary_ids is not None:
        entry["boundary_token_ids"] = list(boundary_ids)
    return entry


def _mixed_frontier_search() -> list[dict[str, Any]]:
    """Two candidates, mixed pass/fail, per-candidate boundary on cand-X."""
    short = (60, 62)  # "<>"
    x_obs = [
        _synth_obs("S1", "alpha" + "<>", exact=True, bfound=True),
        _synth_obs("S2", "betaZZ" + "<>", exact=True, bfound=True),
    ]
    y_obs = [
        _synth_obs("S1", "alphaWRONG" + BOUNDARY_TEXT, exact=False, bfound=True),
        _synth_obs("S2", "beta" + BOUNDARY_TEXT, exact=True, bfound=True),
    ]
    return [_synth_entry("cand-X", x_obs, boundary_ids=short), _synth_entry("cand-Y", y_obs)]


class TestBudgetFrontierOracle(unittest.TestCase):
    """Full-equality checks of budget_frontier against the hand-rolled oracle."""

    @staticmethod
    def test_mixed_scenario_matches_oracle_exactly():
        search = _mixed_frontier_search()
        impl = budget_frontier(
            search,
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="compatible_on_fixture_set",
            native_winner="cand-X",
        )
        oracle = oracle_budget_frontier(
            search,
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="compatible_on_fixture_set",
            native_winner="cand-X",
        )
        assert impl == oracle
        # Spot-check the oracle's own arithmetic, not just agreement:
        # cand-X uses the 2-id "<>" boundary: cuts 5 and 6 -> mins 7 and 8.
        assert impl["candidates"]["cand-X"]["min_budget_per_fixture"] == [7, 8]
        # cand-Y uses the 10-id global boundary: "beta"+boundary -> cut 4 -> 14.
        assert impl["candidates"]["cand-Y"]["min_budget_per_fixture"] == [None, 14]

    @staticmethod
    def test_per_candidate_boundary_token_ids_honored():
        bnd = _boundary_ids()
        text = "alpha" + BOUNDARY_TEXT
        search = [
            _synth_entry("c1", [_synth_obs("S1", text, exact=True, bfound=True)]),
            _synth_entry(
                "c2", [_synth_obs("S1", text, exact=True, bfound=True)], boundary_ids=(ord("<"),)
            ),
        ]
        impl = budget_frontier(
            search,
            boundary=bnd,
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="ambiguous",
            native_winner=None,
        )
        assert impl == oracle_budget_frontier(
            search,
            boundary=bnd,
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="ambiguous",
            native_winner=None,
        )
        # c1: cut 5 + 10 = 15; c2: first "<" at index 5 + 1 = 6.
        assert impl["candidates"]["c1"]["min_budget_per_fixture"] == [15]
        assert impl["candidates"]["c2"]["min_budget_per_fixture"] == [6]

    @staticmethod
    def test_minimal_pair_boundary_shift_deltas():
        search = _mixed_frontier_search()
        before = budget_frontier(
            search,
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="compatible_on_fixture_set",
            native_winner="cand-X",
        )
        shifted = [
            _synth_entry(
                "cand-X",
                [
                    _synth_obs("S1", "alpha###" + "<>", exact=True, bfound=True),
                    _synth_obs("S2", "betaZZ" + "<>", exact=True, bfound=True),
                ],
                boundary_ids=(60, 62),
            ),
            _synth_entry(
                "cand-Y",
                [
                    _synth_obs("S1", "alphaWRONG" + BOUNDARY_TEXT, exact=False, bfound=True),
                    _synth_obs("S2", "beta" + BOUNDARY_TEXT, exact=True, bfound=True),
                ],
            ),
        ]
        after = budget_frontier(
            shifted,
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="compatible_on_fixture_set",
            native_winner="cand-X",
        )
        assert after == oracle_budget_frontier(
            shifted,
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="compatible_on_fixture_set",
            native_winner="cand-X",
        )
        assert after["candidates"]["cand-X"]["min_budget_per_fixture"][0] == (
            before["candidates"]["cand-X"]["min_budget_per_fixture"][0] + 3
        )
        assert (
            after["candidates"]["cand-X"]["min_budget_per_fixture"][1:]
            == (before["candidates"]["cand-X"]["min_budget_per_fixture"][1:])
        )
        assert after["candidates"]["cand-Y"] == before["candidates"]["cand-Y"]

    @staticmethod
    def test_ambiguous_native_decision():
        search = _mixed_frontier_search()
        impl = budget_frontier(
            search,
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="ambiguous",
            native_winner=None,
        )
        assert impl == oracle_budget_frontier(
            search,
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="ambiguous",
            native_winner=None,
        )
        assert impl["native_search_decision"] == "ambiguous"
        assert impl["native_winner"] is None
        assert impl["budget_robust"] is False
        # Hand-derived from the docstring: X all-passes at budget 8
        # (mins 7,8), Y never all-passes (mins None,13); native is ambiguous,
        # so the decision differs from native at every budget 1..64.
        assert impl["flip_ranges"] == [[1, 64]]
        assert impl["decision_steps"] == [[1, "no_candidate", None], [8, "selected", "cand-X"]]

    @staticmethod
    def test_no_candidate_native_decision_is_robust():
        search = [
            _synth_entry(
                "c1",
                [
                    _synth_obs("S1", "alphaWRONG" + BOUNDARY_TEXT, exact=False, bfound=True),
                    _synth_obs("S2", "beta", exact=True, bfound=False),
                ],
            )
        ]
        impl = budget_frontier(
            search,
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="no_candidate",
            native_winner=None,
        )
        assert impl == oracle_budget_frontier(
            search,
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="no_candidate",
            native_winner=None,
        )
        assert impl["flip_ranges"] == []
        assert impl["budget_robust"] is True
        assert impl["decision_steps"] == [[1, "no_candidate", None]]

    @staticmethod
    def test_held_out_failed_native_decision():
        search = _mixed_frontier_search()
        impl = budget_frontier(
            search,
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="held_out_failed",
            native_winner="cand-X",
        )
        assert impl == oracle_budget_frontier(
            search,
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="held_out_failed",
            native_winner="cand-X",
        )
        assert impl["native_search_decision"] == "selected"
        assert impl["native_winner"] == "cand-X"

    @staticmethod
    def test_budget_zero_and_one():
        search = [
            _synth_entry(
                "c1",
                [
                    _synth_obs("S1", chr(7), exact=True, bfound=True),
                    _synth_obs("S2", chr(7), exact=True, bfound=True),
                ],
                boundary_ids=(7,),
            )
        ]
        impl = budget_frontier(
            search,
            boundary=(9,),
            max_new_tokens=1,
            native_decision="compatible_on_fixture_set",
            native_winner="c1",
        )
        assert impl == oracle_budget_frontier(
            search,
            boundary=(9,),
            max_new_tokens=1,
            native_decision="compatible_on_fixture_set",
            native_winner="c1",
        )
        assert impl["candidates"]["c1"]["min_budget_per_fixture"] == [1, 1]
        assert impl["candidates"]["c1"]["full_pass_budget"] == 1
        assert impl["decision_steps"] == [[1, "selected", "c1"]]
        assert impl["flip_ranges"] == []
        assert impl["budget_robust"] is True

    @staticmethod
    def test_failed_and_incomplete_observations_contribute_nothing():
        search = [
            _synth_entry(
                "c1",
                [
                    _synth_obs("S1", "x", exact=True, bfound=True, status="failed"),
                    _synth_obs("S2", None, exact=False, bfound=False),
                ],
            )
        ]
        impl = budget_frontier(
            search,
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="no_candidate",
            native_winner=None,
        )
        assert impl["candidates"]["c1"]["min_budget_per_fixture"] == [None, None]
        assert impl["candidates"]["c1"]["pass_steps"] == [[1, 0]]

    def test_rejects_empty_inputs(self):
        for kwargs in [
            {"search": [], "why": "empty search"},
            {"search": [_synth_entry("c1", [])], "why": "empty observations"},
        ]:
            with self.subTest(kwargs=kwargs):
                why = kwargs.pop("why")
                with self.assertRaises(ValueError):
                    budget_frontier(
                        kwargs["search"],
                        boundary=_boundary_ids(),
                        max_new_tokens=MAX_NEW_TOKENS,
                        native_decision="no_candidate",
                        native_winner=None,
                    )
                assert why  # documents which case ran

    def test_rejects_duplicate_candidate_names(self):
        search = [
            _synth_entry("c1", [_synth_obs("S1", "a", exact=True, bfound=True)]),
            _synth_entry("c1", [_synth_obs("S1", "a", exact=True, bfound=True)]),
        ]
        with self.assertRaises(ValueError):
            budget_frontier(
                search,
                boundary=_boundary_ids(),
                max_new_tokens=MAX_NEW_TOKENS,
                native_decision="no_candidate",
                native_winner=None,
            )

    def test_rejects_fixture_mismatch_across_entries(self):
        search = [
            _synth_entry("c1", [_synth_obs("S1", "a", exact=True, bfound=True)]),
            _synth_entry("c2", [_synth_obs("S2", "a", exact=True, bfound=True)]),
        ]
        with self.assertRaises(ValueError):
            budget_frontier(
                search,
                boundary=_boundary_ids(),
                max_new_tokens=MAX_NEW_TOKENS,
                native_decision="no_candidate",
                native_winner=None,
            )

    def test_rejects_bad_boundary(self):
        for bad_boundary in [(), (1, -2), ("x",)]:
            with self.subTest(bad_boundary=bad_boundary):
                search = [_synth_entry("c1", [_synth_obs("S1", "a", exact=True, bfound=True)])]
                with self.assertRaises(ValueError):
                    budget_frontier(
                        search,
                        boundary=bad_boundary,
                        max_new_tokens=MAX_NEW_TOKENS,
                        native_decision="no_candidate",
                        native_winner=None,
                    )

    def test_rejects_bad_max_new_tokens(self):
        for bad_cap in [0, 4097, "64"]:
            with self.subTest(bad_cap=bad_cap):
                search = [_synth_entry("c1", [_synth_obs("S1", "a", exact=True, bfound=True)])]
                with self.assertRaises(ValueError):
                    budget_frontier(
                        search,
                        boundary=_boundary_ids(),
                        max_new_tokens=bad_cap,
                        native_decision="no_candidate",
                        native_winner=None,
                    )

    def test_rejects_incomplete_native_decision(self):
        search = [_synth_entry("c1", [_synth_obs("S1", "a", exact=True, bfound=True)])]
        with self.assertRaises(ValueError):
            budget_frontier(
                search,
                boundary=_boundary_ids(),
                max_new_tokens=MAX_NEW_TOKENS,
                native_decision="cancelled",
                native_winner=None,
            )

    def test_rejects_winner_decision_mismatch(self):
        search = [_synth_entry("c1", [_synth_obs("S1", "a", exact=True, bfound=True)])]
        with self.assertRaises(ValueError):  # winner must be None for ambiguous
            budget_frontier(
                search,
                boundary=_boundary_ids(),
                max_new_tokens=MAX_NEW_TOKENS,
                native_decision="ambiguous",
                native_winner="c1",
            )
        with self.assertRaises(ValueError):  # winner required for selected decisions
            budget_frontier(
                search,
                boundary=_boundary_ids(),
                max_new_tokens=MAX_NEW_TOKENS,
                native_decision="compatible_on_fixture_set",
                native_winner=None,
            )
        with self.assertRaises(ValueError):  # winner must name a search candidate
            budget_frontier(
                search,
                boundary=_boundary_ids(),
                max_new_tokens=MAX_NEW_TOKENS,
                native_decision="compatible_on_fixture_set",
                native_winner="nope",
            )

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


class TestBudgetFrontierFromRealRun(unittest.TestCase):
    """budget_frontier over a real run_experiment result (end of the CLI path)."""

    @staticmethod
    def test_real_search_result_matches_oracle():
        sc = make_scenario(
            exp_seed=11,
            behavior={("cand-A", "S2"): "fail_answer", ("cand-B", "S1"): "fail_answer"},
        )
        result = run_search_experiment(sc)
        assert result["decision"] == "no_candidate"
        frontier = budget_frontier(
            result["search"],
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision=result["decision"],
            native_winner=result["selected_candidate"],
        )
        assert frontier == oracle_budget_frontier(
            result["search"],
            boundary=_boundary_ids(),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision=result["decision"],
            native_winner=result["selected_candidate"],
        )
        # cand-A passes S1 only (cut 5, 10-id boundary -> 15); cand-B passes S2 only.
        assert frontier["candidates"]["cand-A"]["min_budget_per_fixture"] == [15, None]
        assert frontier["candidates"]["cand-B"]["min_budget_per_fixture"] == [None, 14]
        assert frontier["budget_robust"] is True  # no_candidate at every budget

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


# ---------------------------------------------------------------------------
# Independent oracle: logprob_diagnostic
# ---------------------------------------------------------------------------


def _oracle_lp64(mean: float) -> float:
    """Independent log-softmax for the scripted backend's logit construction.

    Row r has peak logit x at the target token and 0 elsewhere, with
    x = log(p*(V-1)/(1-p)), p = exp(mean). The gathered logprob is
    -(log(1 + (V-1)*exp(-x))) in float64; the engine computes the same value
    via torch and float32-rounds it.
    """
    prob = math.exp(mean)
    peak = math.log(prob * (VOCAB - 1) / (1 - prob))
    return -math.log1p((VOCAB - 1) * math.exp(-peak))


def _oracle_target_score(mean: float, target_ids: tuple[int, ...], target_text: str) -> dict:
    n = len(target_ids)
    total64 = _oracle_lp64(mean) * n
    return {
        "status": "ok",
        "reason": None,
        "total_logprob": _f32(total64),
        "mean_logprob": _f32(total64 / n),
        "scored_token_count": n,
        "target_byte_length": len(target_text.encode("utf-8")),
    }


def _oracle_aggregate(per_fixture: list[dict]) -> dict:
    ok = [s for s in per_fixture if s["status"] == "ok"]
    total = sum(s["total_logprob"] for s in ok)
    tokens = sum(s["scored_token_count"] for s in ok)
    return {
        "scored_fixtures": len(ok),
        "total_logprob": total if ok and tokens else None,
        "mean_logprob": (total / tokens) if ok and tokens else None,
        "scored_token_count": tokens,
        "target_byte_length": sum(s["target_byte_length"] for s in ok),
    }


def _oracle_interpretation(fixture_scores: list[dict]) -> dict:
    """Band interpretation re-derived from the documented thresholds."""
    failures = [s for s in fixture_scores if not s["native_pass"]]
    scored = [s for s in failures if s["status"] == "ok" and type(s["mean_logprob"]) is float]
    if not failures:
        return {"kind": "no_failures"}
    if not scored:
        return {"kind": "unavailable"}
    highs = [s for s in scored if s["mean_logprob"] >= DERAILED]
    lows = [s for s in scored if s["mean_logprob"] < UNKNOWN]
    if highs and not lows:
        return {"kind": "greedy_derailed"}
    if lows and not highs:
        return {"kind": "weights_dont_know"}
    if highs and lows:
        return {"kind": "mixed"}
    return {"kind": "indeterminate"}


def _oracle_diagnostic_candidates(
    result: dict[str, Any], fixtures: tuple, means: dict[tuple[str, str], float]
) -> list[dict[str, Any]]:
    expected_by_id = {fx.id: fx.expected for fx in fixtures}
    out = []
    for entry in result["search"]:
        name = entry["format"]["name"]
        fixture_scores = []
        for obs in entry["observations"]:
            fid = obs["fixture"]
            native_pass = bool(obs["exact_match"] and obs["boundary_found"])
            target_text = expected_by_id[fid]
            target_ids = tuple(ord(c) for c in target_text)
            score = _oracle_target_score(means[(name, fid)], target_ids, target_text)
            fixture_scores.append(
                {
                    "fixture": fid,
                    "native_pass": native_pass,
                    **score,
                }
            )
        aggregate = _oracle_aggregate(fixture_scores)
        aggregate["native_passes"] = sum(1 for s in fixture_scores if s["native_pass"])
        aggregate["native_failures"] = sum(1 for s in fixture_scores if not s["native_pass"])
        aggregate["unavailable_fixtures"] = sum(
            1 for s in fixture_scores if s["status"] in ("unavailable", "not_scored")
        )
        out.append(
            {
                "name": name,
                "fixture_scores": fixture_scores,
                "aggregate": aggregate,
                "interpretation": _oracle_interpretation(fixture_scores),
            }
        )
    return out


def _logprob_scenario(
    means: dict[tuple[str, str], float],
    behavior: dict[tuple[str, str], str],
    exp_seed: int = 21,
) -> Any:
    return make_scenario(
        exp_seed=exp_seed,
        behavior=behavior,
        backend_cls=LogitBackend,
        backend_kwargs={"means": means},
    )


class TestLogprobDiagnosticOracle(unittest.TestCase):
    MEANS = {
        ("cand-A", "S1"): -1.0,
        ("cand-A", "S2"): -1.0,
        ("cand-B", "S1"): -6.0,
        ("cand-B", "S2"): -6.0,
    }
    BEHAVIOR = {("cand-A", "S2"): "fail_answer", ("cand-B", "S1"): "fail_answer"}

    def _run(self, means=None, behavior=None):
        means = self.MEANS if means is None else means
        behavior = self.BEHAVIOR if behavior is None else behavior
        sc = _logprob_scenario(means, behavior)
        result = run_search_experiment(sc)
        diag = logprob_diagnostic(sc.backend, result, sc.split.search)
        return sc, result, diag

    def test_full_payload_matches_independent_oracle(self):
        sc, result, diag = self._run()
        assert diag["available"] is True
        assert diag["detail"] is None
        assert diag["thresholds"] == {
            "derailed_mean_logprob_at_least": DERAILED,
            "unknown_mean_logprob_below": UNKNOWN,
        }
        assert diag["method"]  # documented non-empty method string
        expected = _oracle_diagnostic_candidates(result, sc.split.search, self.MEANS)
        assert len(diag["candidates"]) == len(expected) == 2
        for got, want in zip(diag["candidates"], expected, strict=True):
            assert got["name"] == want["name"]
            assert got["fixture_scores"] == want["fixture_scores"]
            assert got["aggregate"] == want["aggregate"]
            assert got["interpretation"]["kind"] == want["interpretation"]["kind"]
        # Spot-check the band readings: A fails S2 at -1.0 -> derailed;
        # B fails S1 at -6.0 -> weights don't know.
        kinds = {c["name"]: c["interpretation"]["kind"] for c in diag["candidates"]}
        assert kinds == {"cand-A": "greedy_derailed", "cand-B": "weights_dont_know"}

    def test_identical_logprobs_across_candidates(self):
        means = {(c, f): -3.0 for c in ("cand-A", "cand-B") for f in ("S1", "S2")}
        behavior = {("cand-A", "S1"): "fail_answer", ("cand-B", "S1"): "fail_answer"}
        _sc, _result, diag = self._run(means=means, behavior=behavior)
        assert diag["available"] is True
        agg_a, agg_b = (c["aggregate"] for c in diag["candidates"])
        assert agg_a["mean_logprob"] == agg_b["mean_logprob"]
        assert agg_a["total_logprob"] == agg_b["total_logprob"]
        kinds = {c["name"]: c["interpretation"]["kind"] for c in diag["candidates"]}
        # -3.0 sits between the bands -> both withhold judgment identically.
        assert kinds == {"cand-A": "indeterminate", "cand-B": "indeterminate"}

    def test_exact_band_edges(self):
        means = {
            ("cand-A", "S1"): -2.0,
            ("cand-A", "S2"): -1.0,
            ("cand-B", "S1"): -5.0,
            ("cand-B", "S2"): -1.0,
        }
        behavior = {("cand-A", "S1"): "fail_answer", ("cand-B", "S1"): "fail_answer"}
        _sc, _result, diag = self._run(means=means, behavior=behavior)
        kinds = {c["name"]: c["interpretation"]["kind"] for c in diag["candidates"]}
        assert kinds == {"cand-A": "greedy_derailed", "cand-B": "indeterminate"}

    def test_mixed_bands(self):
        means = {
            ("cand-A", "S1"): -1.0,
            ("cand-A", "S2"): -6.0,
            ("cand-B", "S1"): -6.0,
            ("cand-B", "S2"): -6.0,
        }
        behavior = {
            ("cand-A", "S1"): "fail_answer",
            ("cand-A", "S2"): "fail_answer",
            ("cand-B", "S1"): "fail_answer",
        }
        _sc, _result, diag = self._run(means=means, behavior=behavior)
        kinds = {c["name"]: c["interpretation"]["kind"] for c in diag["candidates"]}
        assert kinds["cand-A"] == "mixed"
        assert kinds["cand-B"] == "weights_dont_know"

    def test_minimal_pair_mean_shift_exact_delta(self):
        _sc1, _result1, diag1 = self._run()
        means2 = dict(self.MEANS)
        means2[("cand-A", "S2")] = -6.0
        sc2 = _logprob_scenario(means2, self.BEHAVIOR, exp_seed=21)
        result2 = run_search_experiment(sc2)
        diag2 = logprob_diagnostic(sc2.backend, result2, sc2.split.search)
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        cand1 = next(c for c in diag1["candidates"] if c["name"] == "cand-A")
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        cand2 = next(c for c in diag2["candidates"] if c["name"] == "cand-A")
        fs1 = {s["fixture"]: s for s in cand1["fixture_scores"]}
        fs2 = {s["fixture"]: s for s in cand2["fixture_scores"]}
        assert fs1["S1"] == fs2["S1"]  # untouched cell identical
        assert fs1["S2"]["mean_logprob"] != fs2["S2"]["mean_logprob"]
        self.assertAlmostEqual(fs2["S2"]["mean_logprob"], -6.0, delta=1e-6)
        delta = fs2["S2"]["total_logprob"] - fs1["S2"]["total_logprob"]
        got_delta = cand2["aggregate"]["total_logprob"] - cand1["aggregate"]["total_logprob"]
        self.assertLessEqual(abs(got_delta - delta), 1e-6 * max(1.0, abs(delta)))
        assert cand1["interpretation"]["kind"] == "greedy_derailed"
        assert cand2["interpretation"]["kind"] == "weights_dont_know"
        # cand-B untouched end to end.
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        b1 = next(c for c in diag1["candidates"] if c["name"] == "cand-B")
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        b2 = next(c for c in diag2["candidates"] if c["name"] == "cand-B")
        assert b1 == b2

    def test_diagnostic_runs_on_ambiguous_decision(self):
        sc = _logprob_scenario(self.MEANS, behavior={})
        result = run_search_experiment(sc)
        assert result["decision"] == "ambiguous"
        assert result["selected_candidate"] is None
        diag = logprob_diagnostic(sc.backend, result, sc.split.search)
        assert diag["available"] is True
        assert len(diag["candidates"]) == 2
        kinds = {c["name"]: c["interpretation"]["kind"] for c in diag["candidates"]}
        assert kinds == {"cand-A": "no_failures", "cand-B": "no_failures"}

    def test_diagnostic_runs_on_no_candidate_decision(self):
        behavior = {
            ("cand-A", "S1"): "fail_answer",
            ("cand-A", "S2"): "fail_answer",
            ("cand-B", "S1"): "fail_answer",
            ("cand-B", "S2"): "fail_answer",
        }
        sc = _logprob_scenario(self.MEANS, behavior)
        result = run_search_experiment(sc)
        assert result["decision"] == "no_candidate"
        diag = logprob_diagnostic(sc.backend, result, sc.split.search)
        assert diag["available"] is True
        assert len(diag["candidates"]) == 2

    def test_empty_fixtures_never_crash(self):
        sc, result, _ = self._run()
        diag = logprob_diagnostic(sc.backend, result, ())
        assert diag["available"] is True
        for cand in diag["candidates"]:
            assert all(s["status"] == "not_scored" for s in cand["fixture_scores"])
            assert cand["aggregate"]["scored_fixtures"] == 0
            assert cand["aggregate"]["total_logprob"] is None

    @staticmethod
    def test_backend_without_forward_logits_degrades():
        sc = make_scenario(exp_seed=21, behavior={("cand-A", "S2"): "fail_answer"})
        result = run_search_experiment(sc)
        assert not callable(getattr(sc.backend, "forward_logits", None))
        diag = logprob_diagnostic(sc.backend, result, sc.split.search)
        assert diag["available"] is False
        assert diag["detail"] == "logits_not_supported"
        assert diag["candidates"] == []

    def test_rejects_malformed_inputs(self):
        sc = make_scenario(exp_seed=21)
        result = run_search_experiment(sc)
        with self.assertRaises(ValueError):
            logprob_diagnostic(sc.backend, {"no": "search"}, sc.split.search)
        with self.assertRaises(ValueError):
            logprob_diagnostic(sc.backend, result, ("not", "fixtures"))

    @staticmethod
    def test_aggregate_nan_defense():
        from model_protocol_lab.scoring import TargetScore

        bad = TargetScore(
            status="ok",
            reason=None,
            per_token_logprobs=(),
            total_logprob=float("inf"),
            mean_logprob=float("inf"),
            scored_token_count=3,
            target_byte_length=9,
            prompt_token_count=4,
        )
        agg = _aggregate_fixture_scores([bad])
        assert agg["total_logprob"] is None
        assert agg["mean_logprob"] is None

    @staticmethod
    def test_interpretation_ignores_passing_fixtures():
        scores = [
            {
                "native_pass": True,
                "status": "ok",
                "mean_logprob": -0.5,
                "total_logprob": -1.0,
                "scored_token_count": 2,
            },
            {
                "native_pass": False,
                "status": "ok",
                "mean_logprob": -6.0,
                "total_logprob": -12.0,
                "scored_token_count": 2,
            },
        ]
        assert _interpret_logprob_candidate(scores)["kind"] == "weights_dont_know"

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


# ---------------------------------------------------------------------------
# prompt_sensitivity_sweep
# ---------------------------------------------------------------------------


class TestSensitivitySweep(unittest.TestCase):
    @staticmethod
    def test_noop_perturbation_flags_nothing():
        sc = make_scenario(exp_seed=31, candidates=("cand-A",))
        result = run_search_experiment(sc)
        assert result["decision"] == "compatible_on_fixture_set"
        sweep = prompt_sensitivity_sweep(
            sc.backend,
            sc.formats,
            sc.split,
            boundary=_boundary_ids(),
            max_drop_bytes=3,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=sc.exp_seed,
        )
        assert sweep["native_decision"] == "compatible_on_fixture_set"
        assert sweep["native_winner"] == "cand-A"
        assert sweep["max_drop_bytes"] == 3
        assert [d["drop_bytes"] for d in sweep["depths"]] == [1, 2, 3]
        assert all(d["winner_changed"] is False for d in sweep["depths"])
        assert all(d["decision"] == "compatible_on_fixture_set" for d in sweep["depths"])
        assert all(d["selected_candidate"] == "cand-A" for d in sweep["depths"])
        assert sweep["flips_at"] == []

    @staticmethod
    def test_real_prompt_tail_change_is_detected():
        sc = make_scenario(
            exp_seed=32,
            candidates=("cand-A",),
            backend_cls=PromptTailBackend,
        )
        sweep = prompt_sensitivity_sweep(
            sc.backend,
            sc.formats,
            sc.split,
            boundary=_boundary_ids(),
            max_drop_bytes=3,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=sc.exp_seed,
        )
        assert sweep["native_decision"] == "compatible_on_fixture_set"
        assert sweep["native_winner"] == "cand-A"
        assert sweep["flips_at"] == [1, 2, 3]
        for depth in sweep["depths"]:
            assert depth["winner_changed"] is True
            assert depth["decision"] == "no_candidate"
            assert depth["selected_candidate"] is None

    @staticmethod
    def test_winner_swap_with_same_decision_still_flagged():
        sc = make_scenario(
            exp_seed=33,
            candidates=("cand-A", "cand-B"),
            backend_cls=PromptTailBackend,
            backend_kwargs={"swap": True},
        )
        sweep = prompt_sensitivity_sweep(
            sc.backend,
            sc.formats,
            sc.split,
            boundary=_boundary_ids(),
            max_drop_bytes=3,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=sc.exp_seed,
        )
        assert sweep["native_winner"] == "cand-A"
        assert sweep["native_decision"] == "compatible_on_fixture_set"
        assert sweep["flips_at"] == [1, 2, 3]
        for depth in sweep["depths"]:
            assert depth["winner_changed"] is True
            assert depth["decision"] == "compatible_on_fixture_set"
            assert depth["selected_candidate"] == "cand-B"

    @staticmethod
    def test_codepoint_split_depth_records_failed_decision():
        sc = make_scenario(
            exp_seed=34,
            candidates=("cand-A",),
            assistant_prefix="",
            turn_end="é",  # prompt tail is a 2-byte UTF-8 char
        )
        native = run_search_experiment(sc)
        assert native["decision"] == "compatible_on_fixture_set"
        sweep = prompt_sensitivity_sweep(
            sc.backend,
            sc.formats,
            sc.split,
            boundary=_boundary_ids(),
            max_drop_bytes=2,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=sc.exp_seed,
        )
        by_drop = {d["drop_bytes"]: d for d in sweep["depths"]}
        # drop=1 splits the "é" codepoint -> encode fails -> depth records it.
        assert by_drop[1]["decision"] == "failed"
        assert by_drop[1]["selected_candidate"] is None
        assert by_drop[1]["winner_changed"] is True
        # drop=2 removes "é" whole -> the run completes; nothing crashes.
        assert by_drop[2]["decision"] == "compatible_on_fixture_set"
        assert sweep["flips_at"] == [1]

    def test_rejects_bad_max_drop_bytes(self):
        sc = make_scenario(exp_seed=35, candidates=("cand-A",))
        for bad in (0, 9, -1, "3"):
            with self.assertRaises(ValueError):
                prompt_sensitivity_sweep(
                    sc.backend,
                    sc.formats,
                    sc.split,
                    boundary=_boundary_ids(),
                    max_drop_bytes=bad,
                    max_new_tokens=MAX_NEW_TOKENS,
                    seed=sc.exp_seed,
                )

    @staticmethod
    def test_sweep_result_is_json_serializable():
        import json

        sc = make_scenario(exp_seed=36, candidates=("cand-A",))
        sweep = prompt_sensitivity_sweep(
            sc.backend,
            sc.formats,
            sc.split,
            boundary=_boundary_ids(),
            max_drop_bytes=1,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=sc.exp_seed,
        )
        json.dumps(sweep, allow_nan=False)

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


# ---------------------------------------------------------------------------
# CLI wiring contract
# ---------------------------------------------------------------------------


class TestCliWiring(unittest.TestCase):
    """Structural contract: which flags exist and what they default to."""

    @staticmethod
    def _run_options() -> dict[str, argparse.Action]:
        parser = build_parser()
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        run_parser = sub.choices["run"]
        return {name: action for action in run_parser._actions for name in action.option_strings}

    def test_sensitivity_sweep_flag(self):
        opts = self._run_options()
        assert "--sensitivity-sweep" in opts
        assert isinstance(opts["--sensitivity-sweep"], argparse._StoreTrueAction)
        assert opts["--sensitivity-sweep"].default is False
        assert "--max-drop-bytes" in opts
        assert opts["--max-drop-bytes"].default == 3

    def test_logprob_diagnostic_flag(self):
        opts = self._run_options()
        assert "--logprob-diagnostic" in opts
        assert isinstance(opts["--logprob-diagnostic"], argparse._StoreTrueAction)
        assert opts["--logprob-diagnostic"].default is False

    def test_budget_frontier_has_no_flag_it_is_always_on(self):
        opts = self._run_options()
        assert "--budget-frontier" not in opts
        assert "--budget_frontier" not in opts

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


# ---------------------------------------------------------------------------
# Mutation gate (on /tmp package copies only; repo sources never touched)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "model_protocol_lab"


def _read_experiments_source() -> str:
    return (PKG_DIR / "experiments.py").read_text(encoding="utf-8")


_MUTANT_PREAMBLE = """import sys
sys.path.insert(0, "__PKGDIR__")
from model_protocol_lab import core as _core
from model_protocol_lab import experiments as _exp
from model_protocol_lab.scoring import TargetScore as _TargetScore

def _obs(fixture, output_ids, *, exact, bfound, status="completed"):
    return {
        "fixture": fixture,
        "status": status,
        "exact_match": exact,
        "boundary_found": bfound,
        "output_ids": list(output_ids),
        "input_ids": [1, 2, 3],
    }

def _entry(name, obs, boundary_ids=None):
    entry = {"format": {"name": name}, "observations": obs}
    if boundary_ids is not None:
        entry["boundary_token_ids"] = list(boundary_ids)
    return entry

__CHECK__
"""

# Each mutant: (name, unique anchor in experiments.py, replacement, check code).
# The check must PASS on pristine sources (control) and FAIL on the mutant.
_MUTANTS: list[tuple[str, str, str, str]] = [
    (
        "M1 frontier min-budget off-by-one",
        "return cut + len(boundary)",
        "return cut + 1",
        """
B = (60, 62)
obs = [_obs("S1", (1, 2, 3, 4, 5, 60, 62, 9), exact=True, bfound=True)]
search = [_entry("c", obs, boundary_ids=B)]
out = _exp.budget_frontier(
    search, boundary=(7, 8, 9), max_new_tokens=64,
    native_decision="compatible_on_fixture_set", native_winner="c",
)
# cut=5, len(boundary)=2 -> minimal passing budget is 7, not 6.
assert out["candidates"]["c"]["min_budget_per_fixture"] == [7], out
""",
    ),
    (
        "M2 logprob derailed band edge >= turned into >",
        's["mean_logprob"] >= LOGPROB_DERAILED_MEAN_THRESHOLD',
        's["mean_logprob"] > LOGPROB_DERAILED_MEAN_THRESHOLD',
        """
scores = [{
    "native_pass": False, "status": "ok", "mean_logprob": -2.0,
    "total_logprob": -4.0, "scored_token_count": 2,
}]
interp = _exp._interpret_logprob_candidate(scores)
# Exactly -2.0 satisfies ">=": must read greedy_derailed, not indeterminate.
assert interp["kind"] == "greedy_derailed", interp
""",
    ),
    (
        "M3 sweep winner_changed OR turned into AND",
        'changed = winner != native_winner or mutated["decision"] != native["decision"]',
        'changed = winner != native_winner and mutated["decision"] != native["decision"]',
        """
from hashlib import sha256

def _rseed(seed, fid):
    material = f"{seed}\\0{fid}".encode("utf-8", errors="surrogatepass")
    return int.from_bytes(sha256(material).digest()[:4], "big")

class _B:
    fingerprint = "m3"

    def __init__(self):
        self._s2f = {}

    def encode(self, text):
        return tuple(ord(c) for c in text)

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        cand = "cand-A" if "[cand-A]" in prompt else "cand-B"
        fid = self._s2f[seed]
        tail_ok = prompt.endswith("A:")
        ok = (cand == "cand-A") == tail_ok
        expected = {"S1": "alpha", "S2": "beta", "H1": "delta"}[fid]
        text = expected + "<|B|>" if ok else expected + "WRONG<|B|>"
        ids = self.encode(text)
        return _core.Generation(ids + (32,) * (max_new_tokens - len(ids)), "limit")

def _fx(fid, exp, text):
    return _core.Fixture(
        id=fid,
        messages=(_core.Message(role="user", content=text),),
        expected=exp,
        group="g-" + fid,
    )

split = _core.Split(
    search=(_fx("S1", "alpha", "q1?"), _fx("S2", "beta", "q2?")),
    held_out=(_fx("H1", "delta", "q3?"),),
)
fmts = tuple(
    _core.Format(name=n, bos="", user_prefix="[" + n + "]U:",
                 assistant_prefix="A:", turn_end="\\n")
    for n in ("cand-A", "cand-B")
)
backend = _B()
for fid in ("S1", "S2", "H1"):
    backend._s2f[_rseed(11, fid)] = fid
out = _exp.prompt_sensitivity_sweep(
    backend, fmts, split, boundary=tuple(ord(c) for c in "<|B|>"),
    max_drop_bytes=3, max_new_tokens=64, seed=11,
)
assert out["native_winner"] == "cand-A", out
# Winner swaps A->B while the decision stays compatible: OR semantics flags it.
assert out["flips_at"] == [1, 2, 3], out
""",
    ),
    (
        "M4 frontier flip comparison == turned into !=",
        "if start > end or decision_at(start) == native_search:",
        "if start > end or decision_at(start) != native_search:",
        """
B = (7,)
search = [_entry("c", [
    _obs("S1", (7, 9), exact=True, bfound=True),
    _obs("S2", (7,), exact=True, bfound=True),
], boundary_ids=B)]
out = _exp.budget_frontier(
    search, boundary=(1, 2, 3), max_new_tokens=64,
    native_decision="compatible_on_fixture_set", native_winner="c",
)
# Both fixtures pass at budget 1, so the decision equals native everywhere.
assert out["budget_robust"] is True and out["flip_ranges"] == [], out
""",
    ),
    (
        "M5 aggregate mean divided by fixtures instead of tokens",
        "mean_out = total / tokens",
        "mean_out = total / scored_fixtures",
        """
s1 = _TargetScore(status="ok", reason=None, per_token_logprobs=(-1.0, -1.0),
                  total_logprob=-2.0, mean_logprob=-1.0,
                  scored_token_count=2, target_byte_length=5, prompt_token_count=3)
s2 = _TargetScore(status="ok", reason=None, per_token_logprobs=(-1.0,) * 6,
                  total_logprob=-6.0, mean_logprob=-1.0,
                  scored_token_count=6, target_byte_length=4, prompt_token_count=3)
agg = _exp._aggregate_fixture_scores([s1, s2])
# Token-weighted mean: -8/8 = -1.0 (mutant computes -8/2 = -4.0).
assert agg["total_logprob"] == -8.0, agg
assert agg["mean_logprob"] == -1.0, agg
""",
    ),
]


def _write_package_copy(dest: Path, mutate: tuple[str, str] | None) -> Path:
    """Copy the package to dest; optionally apply one (anchor, replacement)."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(PKG_DIR, dest / "model_protocol_lab")
    if mutate is not None:
        anchor, replacement = mutate
        target = dest / "model_protocol_lab" / "experiments.py"
        source = target.read_text(encoding="utf-8")
        assert source.count(anchor) == 1, f"anchor not unique/found: {anchor!r}"
        target.write_text(source.replace(anchor, replacement), encoding="utf-8")
    return dest / "model_protocol_lab"


def _run_check(pkgdir: Path, check: str) -> subprocess.CompletedProcess[str]:
    script = _MUTANT_PREAMBLE.replace("__PKGDIR__", str(pkgdir)).replace("__CHECK__", check)
    script_path = pkgdir.parent / "check.py"
    script_path.write_text(script, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script_path)],
        cwd=pkgdir.parent,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


class TestMutationGate(unittest.TestCase):
    def test_mutant_is_caught(self):
        tmp_path = self.tmp_path
        for spec in _MUTANTS:
            with self.subTest(spec=spec):
                name, anchor, replacement, check = spec
                assert tmp_path.is_absolute()
                mutant_pkg = _write_package_copy(tmp_path / "mutant", (anchor, replacement))
                control_pkg = _write_package_copy(tmp_path / "control", None)
                # Control first: the check must pass on pristine sources, proving it
                # is not vacuously failing.
                control = _run_check(control_pkg, check)
                assert control.returncode == 0, (
                    f"control failed for {name} -- the check itself is broken:\n{control.stderr}"
                )
                mutant = _run_check(mutant_pkg, check)
                assert mutant.returncode != 0, (
                    f"MUTANT SURVIVED: {name} was not caught by its check.\n{mutant.stdout}"
                )
                # Non-vacuous: the mutant must die on the check's assertion, not on
                # an unrelated crash (e.g. an import error in the copied package).
                assert "AssertionError" in mutant.stderr, (
                    f"{name}: mutant failed for an unexpected reason, not the check:\n"
                    f"{mutant.stderr[-2000:]}"
                )

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


if __name__ == "__main__":
    unittest.main()
