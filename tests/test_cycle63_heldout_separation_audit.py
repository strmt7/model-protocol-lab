"""Cycle 63: adversarial audit of the search/held-out separation.

HARD CONSTRAINT UNDER TEST
--------------------------
Fixtures used for search/candidate selection must never leak into held-out
evaluation, and vice versa. The separation contract, read from the sources:

1. ``Split`` (model_protocol_lab/core.py:136-188) -- the only dataset-split
   constructor. ``__post_init__`` enforces, NFC-normalized: both partitions
   are non-empty tuples of ``Fixture``; fixture IDs are globally unique; no
   ``group`` crosses the split; no identical message inputs cross the split.
   The CSV/JSON loaders (model_protocol_lab/fixtures.py:150-208) assign each
   row's partition from an explicit ``partition`` column and let the ``Split``
   constructor raise on any violation, before any file is written.
2. ``run_experiment`` (model_protocol_lab/experiments.py:942-1060):
   * search stage -- ``evaluate(backend, fmt, split.search, ...)`` once per
     candidate (line 1008). The unique-winner rule admits a candidate only
     when its search report is aggregate-eligible and
     ``exact_and_boundary_passes == planned_count`` (every search fixture).
   * selection gate (lines 1032-1034) -- ``len(winners) != 1`` returns
     ``ambiguous``/``no_candidate`` BEFORE the held-out ``evaluate`` call, so
     held-out fixtures are never measured unless selection already produced
     a unique all-search-pass winner.
   * held-out stage -- ``evaluate(backend, selected, split.held_out, ...)``
     (line 1042), exactly once, for the selected candidate only; the
     decision is ``compatible_on_fixture_set`` iff every held-out
     observation passes, else ``held_out_failed``. Held-out observations
     never flow back into selection (``selected_candidate`` is fixed first).
3. Request-seed separation (``evaluate``, model_protocol_lab/core.py): each
   backend call uses ``request_seed = sha256(f"{seed}\\0{case.id}")[:4]``.
   Fixture IDs are globally unique across the split, so search-stage and
   held-out-stage calls always carry distinct seeds -- a seed-keyed backend
   cache cannot confuse the stages.
4. Reporting keeps the stages apart: ``failure_taxonomy``
   (experiments.py:72-118) classifies held-out observations only when the
   decision is ``held_out_failed``; the CLI/reporting/diffreports surfaces
   read ``result["search"]`` and ``result["held_out"]`` separately. Post-hoc
   diagnostics (stop matrix, budget frontier, logprob diagnostic,
   sensitivity sweep) run after selection is final and, per cli.py:714,
   "never change the selection rule".

METHOD
------
* Independent oracle: a deterministic ``SeedSpyBackend`` whose ``generate``
  output is a pure function of (candidate, fixture) taken from a behavior
  table the lab never sees. The audit predicts the decision, the winner,
  the exact backend call sequence, and every per-observation verdict from
  the behavior table alone (differential testing), then asserts FULL
  EQUALITY against what ``run_experiment`` actually did -- exact ordered
  call sequences and exact multisets, never subset checks, so a passing
  check cannot be vacuous.
* Canary: a held-out fixture with a distinctive expected answer. In the
  ambiguous-selection scenario the lab must abstain WITHOUT measuring any
  held-out fixture; the spy proves the canary's request seed was never used
  and its answer never appears in search-stage records.
* Fuzz: randomized behaviors, split sizes, and fixture orderings (seeded),
  every trial checked against the independent oracle, including stage
  ordering (all search-stage calls precede all held-out-stage calls).
* Non-tautology: the mutation gate (throwaway script in /tmp -- repo
  sources are never edited) proves the canary and stage checks FAIL when a
  leak is deliberately introduced into a copy of experiments.py.
* Mutation gate: 4 semantic mutants of experiments.py, each in its own
  /tmp package copy -- (M1) search loop measures ``split.held_out``,
  (M2) unique-winner guard dropped, (M3) held-out output reuses search
  observations, (M4) held-out stage measures ``split.search``. Every mutant
  must be caught by at least one audit check.

FINDINGS
--------
CLEAN -- no real leaks in the automated pipeline. Every audit check passes
against the unmodified sources; all four mutants are caught. Specifically:

* Winner selection provably uses only search-stage data: with tied search
  winners the lab abstains (``ambiguous``) and the backend is never asked
  about any held-out fixture -- the exact ordered call sequence is pinned.
* The held-out stage measures only held-out fixtures, once, for the
  selected candidate; its observations are fresh measurements (fixture IDs
  and full output-ID sequences match the held-out fixtures exactly), never
  reused search observations.
* ``Split`` rejects every cross-split contamination attempt at
  construction: duplicate IDs (including NFC-variant duplicates), shared
  groups, identical message inputs, and empty partitions.
* Request seeds are disjoint across stages by construction (unique fixture
  IDs feed the derivation); reruns are byte-identical; no shared mutable
  state exists between runs (frozen dataclasses, fresh result dicts).
* By design (not a leak): ``mpl debug``'s REPL merges both partitions for
  interactive inspection of the operator's own dataset; it is outside the
  automated selection pipeline and never feeds selection.

Deliberately NOT verified here: live-model backends (the audit uses a
deterministic protocol-conformant spy; ``LocalHFBackend`` honors the same
request-local-seed contract per backends.py:732-733, 853-859), and the
statistical strength of the fixture split itself (a dataset-design
question, not a code-separation question).
"""

from __future__ import annotations

import json
import random
import unittest
from dataclasses import FrozenInstanceError
from hashlib import sha256
from types import SimpleNamespace
from typing import Any

from model_protocol_lab import core as real_core
from model_protocol_lab import experiments as real_exp

BOUNDARY_TEXT = "<|C63BND|>"
FILLER_ID = 32  # ord(" "): padding that can never contain the boundary
MAX_NEW_TOKENS = 64
CANDIDATES = ("cand-A", "cand-B")

# Shared spec literals (make_scenario never mutates them).
_STD_SEARCH = [("S1", "search question one?", "alpha"), ("S2", "search question two?", "beta")]
_STD_HELD = [("H1", "held-out question one?", "delta"), ("H2", "held-out question two?", "eps")]
_CANARY_HELD = [
    ("H1", "held-out question one?", "CANARY-EXPECTED-9f8e"),
    ("H2", "held-out question two?", "gamma"),
]


def _request_seed(exp_seed: int, fixture_id: str) -> int:
    """Independent re-implementation of evaluate()'s request-seed derivation.

    Mirrors model_protocol_lab/core.py ``evaluate``:
    ``material = f"{seed}\\0{case.id}".encode("utf-8", errors="surrogatepass")``,
    ``request_seed = int.from_bytes(sha256(material).digest()[:4], "big")``.
    If core.py ever changed its derivation, the spy's seed->fixture inversion
    would break loudly (unknown seeds) instead of passing silently.
    """
    material = f"{exp_seed}\0{fixture_id}".encode("utf-8", errors="surrogatepass")
    return int.from_bytes(sha256(material).digest()[:4], "big")


def _boundary_ids() -> tuple[int, ...]:
    return tuple(ord(c) for c in BOUNDARY_TEXT)


class SeedSpyBackend:
    """Deterministic Backend-protocol spy; output is a pure function of inputs.

    ``generate`` looks the fixture up by inverting the lab's request-seed
    derivation, identifies the candidate from its format marker in the
    rendered prompt, and returns output dictated by the behavior table --
    a table the lab under test never sees. Every call is logged as
    ``(request_seed, candidate, fixture)`` in order, so the audit can prove
    exactly which stage measured which fixture, and when.
    """

    fingerprint = "cycle63-spy"

    def __init__(
        self,
        core_mod: Any,
        exp_seed: int,
        fixtures: dict[str, Any],
        behavior: dict[tuple[str, str], str],
        candidates: tuple[str, ...],
    ) -> None:
        self._core = core_mod
        self._fixtures = dict(fixtures)
        self._behavior = dict(behavior)
        self._candidates = tuple(candidates)
        self.seed_to_fid: dict[int, str] = {}
        for fid in self._fixtures:
            seed = _request_seed(exp_seed, fid)
            if seed in self.seed_to_fid:
                raise AssertionError(f"request-seed collision for fixture {fid!r}")
            self.seed_to_fid[seed] = fid
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "request_seed": seed,
                "candidate": candidate,
                "fixture": fixture_id,
                "max_new_tokens": max_new_tokens,
            }
        )
        if candidate is None or fixture_id is None:
            return self._core.Generation((), "error", "unknown_request")
        return self._core.Generation(
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


# ---------------------------------------------------------------------------
# Scenario construction. Everything is parameterized by the ``core`` module so
# the mutation gate can rebuild identical scenarios against a mutated package
# copy (whose Split/Format/Fixture/Generation are distinct classes).
# ---------------------------------------------------------------------------


def make_scenario(
    core_mod: Any,
    *,
    exp_seed: int,
    candidates: tuple[str, ...] = CANDIDATES,
    search_specs: list[tuple[str, str, str]],
    held_specs: list[tuple[str, str, str]],
    behavior: dict[tuple[str, str], str] | None = None,
) -> SimpleNamespace:
    """Build fixtures, split, formats, and spy backend for one audit scenario.

    Specs are (fixture_id, user_text, expected_answer); each fixture gets a
    unique group so group-overlap rejection never triggers spuriously.
    """
    fixtures: dict[str, Any] = {}

    def _fixture(fid: str, text: str, expected: str) -> Any:
        fx = core_mod.Fixture(
            id=fid,
            messages=(core_mod.Message(role="user", content=text),),
            expected=expected,
            group=f"group-{fid}",
        )
        fixtures[fid] = fx
        return fx

    search = tuple(_fixture(fid, text, exp) for fid, text, exp in search_specs)
    held_out = tuple(_fixture(fid, text, exp) for fid, text, exp in held_specs)
    split = core_mod.Split(search=search, held_out=held_out)
    formats = tuple(
        core_mod.Format(
            name=name,
            bos="",
            user_prefix=f"[{name}]U:",
            assistant_prefix="A:",
            turn_end="\n",
        )
        for name in candidates
    )
    backend = SeedSpyBackend(core_mod, exp_seed, fixtures, behavior or {}, candidates)
    return SimpleNamespace(
        exp_seed=exp_seed,
        candidates=candidates,
        split=split,
        formats=formats,
        backend=backend,
        search_ids=[fid for fid, _, _ in search_specs],
        held_ids=[fid for fid, _, _ in held_specs],
    )


def run_experiment_on(exp_mod: Any, sc: SimpleNamespace) -> dict[str, Any]:
    return exp_mod.run_experiment(
        sc.backend,
        sc.formats,
        sc.split,
        boundary=_boundary_ids(),
        max_new_tokens=MAX_NEW_TOKENS,
        seed=sc.exp_seed,
    )


def _call_sequence(sc: SimpleNamespace) -> list[tuple[str | None, str | None]]:
    return [(c["candidate"], c["fixture"]) for c in sc.backend.calls]


def _search_verdicts_match_behavior(
    result: dict[str, Any], behavior: dict[tuple[str, str], str]
) -> None:
    """Differential check: every search observation's pass verdict equals the
    behavior table's intent for that (candidate, fixture).
    """
    for entry in result["search"]:
        candidate = entry["format"]["name"]
        for obs in entry["observations"]:
            want_pass = behavior.get((candidate, obs["fixture"]), "pass") == "pass"
            got_pass = bool(obs["exact_match"] and obs["boundary_found"])
            assert got_pass == want_pass, (
                f"candidate {candidate} fixture {obs['fixture']}: lab measured "
                f"pass={got_pass}, behavior table says pass={want_pass}"
            )


# ---------------------------------------------------------------------------
# Audit checks. Each raises AssertionError with a precise message on any
# separation violation, and is importable by the /tmp mutation gate, which
# runs them against mutated copies of experiments.py.
# ---------------------------------------------------------------------------


def audit_ambiguous_abstains(exp_mod: Any, core_mod: Any) -> dict[str, Any]:
    """Tied search winners => abstain WITHOUT ever measuring held-out.

    This is the core canary test: both candidates pass every search fixture,
    so selection cannot complete; the held-out canary fixture (distinctive
    expected answer) must never be touched by the backend, and its data must
    never appear in search-stage records.
    """
    behavior: dict[tuple[str, str], str] = {}
    sc = make_scenario(
        core_mod,
        exp_seed=7,
        search_specs=_STD_SEARCH,
        held_specs=_CANARY_HELD,
        behavior=behavior,
    )
    result = run_experiment_on(exp_mod, sc)

    assert result["decision"] == "ambiguous", f"expected abstention, got {result['decision']!r}"
    assert result["selected_candidate"] is None, "no winner may be selected on tied search"
    assert result["held_out"] is None, "held-out stage must not run without a unique winner"

    # Full-equality on the backend call sequence: each candidate measured on
    # each search fixture exactly once, in order -- nothing else, ever.
    expected_seq = [("cand-A", "S1"), ("cand-A", "S2"), ("cand-B", "S1"), ("cand-B", "S2")]
    assert _call_sequence(sc) == expected_seq, (
        f"search stage touched unexpected fixtures: {_call_sequence(sc)}"
    )

    # Canary: the held-out fixture's request seed was never used...
    canary_seed = _request_seed(sc.exp_seed, "H1")
    used_seeds = {c["request_seed"] for c in sc.backend.calls}
    assert canary_seed not in used_seeds, "LEAK: canary held-out fixture measured during search"
    assert all(_request_seed(sc.exp_seed, h) not in used_seeds for h in sc.held_ids), (
        "LEAK: a held-out fixture was measured before selection completed"
    )
    # ...and its answer never appears in any search-stage record.
    for entry in result["search"]:
        assert entry["format"]["name"] in sc.candidates, (
            f"search entry has unexpected candidate {entry['format']['name']!r}"
        )
        assert [o["fixture"] for o in entry["observations"]] == sc.search_ids, (
            "search observations must cover exactly the search fixtures in order"
        )
        assert all("CANARY" not in o["answer"] for o in entry["observations"]), (
            "LEAK: canary answer present in search-stage records"
        )
    _search_verdicts_match_behavior(result, behavior)
    return result


def audit_unique_winner_flow(exp_mod: Any, core_mod: Any) -> dict[str, Any]:
    """Unique search winner => held-out measured once, fresh, then verdict.

    cand-A passes all search; cand-B fails one search fixture; cand-A then
    fails one held-out fixture => decision must be held_out_failed, and the
    held-out observations must be fresh measurements of the held-out
    fixtures (not reused search observations).
    """
    behavior = {("cand-B", "S2"): "fail_answer", ("cand-A", "H2"): "fail_boundary"}
    sc = make_scenario(
        core_mod,
        exp_seed=11,
        search_specs=_STD_SEARCH,
        held_specs=_STD_HELD,
        behavior=behavior,
    )
    result = run_experiment_on(exp_mod, sc)

    assert result["decision"] == "held_out_failed", f"got {result['decision']!r}"
    assert result["selected_candidate"] == "cand-A"

    # Exact ordered call sequence: all search-stage calls precede held-out calls.
    expected_seq = [
        ("cand-A", "S1"),
        ("cand-A", "S2"),
        ("cand-B", "S1"),
        ("cand-B", "S2"),
        ("cand-A", "H1"),
        ("cand-A", "H2"),
    ]
    assert _call_sequence(sc) == expected_seq, (
        f"stage interleaving or wrong fixtures measured: {_call_sequence(sc)}"
    )

    # Held-out observations are fresh: right fixtures, right candidate, and
    # the full output-ID sequence is exactly what the backend must produce
    # for (selected candidate, held-out fixture) -- any reuse of search
    # observations or cache confusion fails this full-equality check.
    held_obs = result["held_out"]["observations"]
    assert [o["fixture"] for o in held_obs] == sc.held_ids, (
        f"held-out observations cover {[o['fixture'] for o in held_obs]}, "
        f"not the held-out fixtures {sc.held_ids}: stale or reused observations"
    )
    for obs in held_obs:
        assert obs["candidate"] == "cand-A", "held-out measured for the wrong candidate"
        assert tuple(obs["output_ids"]) == sc.backend.expected_output_ids(
            "cand-A", obs["fixture"], MAX_NEW_TOKENS
        ), f"held-out observation for {obs['fixture']} is not a fresh measurement"
        want_pass = behavior.get(("cand-A", obs["fixture"]), "pass") == "pass"
        assert bool(obs["exact_match"] and obs["boundary_found"]) == want_pass, (
            f"held-out verdict for {obs['fixture']} disagrees with the behavior table"
        )

    # Stage segregation inside the result document: no fixture id in both.
    search_ids = {o["fixture"] for e in result["search"] for o in e["observations"]}
    held_ids = {o["fixture"] for o in held_obs}
    assert search_ids == set(sc.search_ids) and held_ids == set(sc.held_ids), (
        f"stage fixture sets wrong: search={sorted(search_ids)}, held={sorted(held_ids)}"
    )
    assert not search_ids & held_ids, "a fixture id appears in both stages' records"
    _search_verdicts_match_behavior(result, behavior)
    return result


def audit_compatible_flow(exp_mod: Any, core_mod: Any) -> dict[str, Any]:
    """Unique search winner + all held-out pass => compatible_on_fixture_set."""
    behavior = {("cand-B", "S1"): "fail_boundary"}
    sc = make_scenario(
        core_mod,
        exp_seed=13,
        search_specs=_STD_SEARCH,
        held_specs=_STD_HELD,
        behavior=behavior,
    )
    result = run_experiment_on(exp_mod, sc)
    assert result["decision"] == "compatible_on_fixture_set", f"got {result['decision']!r}"
    assert result["selected_candidate"] == "cand-A"
    assert _call_sequence(sc) == [
        ("cand-A", "S1"),
        ("cand-A", "S2"),
        ("cand-B", "S1"),
        ("cand-B", "S2"),
        ("cand-A", "H1"),
        ("cand-A", "H2"),
    ], f"backend call sequence wrong: {_call_sequence(sc)}"
    held_obs = result["held_out"]["observations"]
    assert [o["fixture"] for o in held_obs] == sc.held_ids, (
        f"held-out observations cover {[o['fixture'] for o in held_obs]}, "
        f"not the held-out fixtures {sc.held_ids}: stale or reused observations"
    )
    for obs in held_obs:
        assert tuple(obs["output_ids"]) == sc.backend.expected_output_ids(
            "cand-A", obs["fixture"], MAX_NEW_TOKENS
        )
        assert obs["exact_match"] and obs["boundary_found"]
    # The failure taxonomy must not invent held-out findings on success.
    taxonomy = result["failure_taxonomy"]
    assert taxonomy["held_out"] is None, "held_out taxonomy must be None unless held_out_failed"
    _search_verdicts_match_behavior(result, behavior)
    return result


def audit_no_candidate_flow(exp_mod: Any, core_mod: Any) -> dict[str, Any]:
    """No search winner => no_candidate, held-out never measured."""
    behavior = {("cand-A", "S1"): "fail_answer", ("cand-B", "S2"): "fail_answer"}
    sc = make_scenario(
        core_mod,
        exp_seed=17,
        search_specs=_STD_SEARCH,
        held_specs=_STD_HELD,
        behavior=behavior,
    )
    result = run_experiment_on(exp_mod, sc)
    assert result["decision"] == "no_candidate", f"got {result['decision']!r}"
    assert result["selected_candidate"] is None, "no winner may be selected"
    assert result["held_out"] is None, "held-out stage must not run without a winner"
    assert _call_sequence(sc) == [
        ("cand-A", "S1"),
        ("cand-A", "S2"),
        ("cand-B", "S1"),
        ("cand-B", "S2"),
    ], "backend must never see held-out fixtures when selection fails"
    _search_verdicts_match_behavior(result, behavior)
    return result


def audit_seed_separation(exp_mod: Any, core_mod: Any) -> dict[str, Any]:
    """Request seeds are disjoint across stages; runs are deterministic."""
    sc = make_scenario(
        core_mod,
        exp_seed=7,
        search_specs=_STD_SEARCH,
        held_specs=_STD_HELD,
    )
    result = run_experiment_on(exp_mod, sc)
    search_seeds = {_request_seed(sc.exp_seed, fid) for fid in sc.search_ids}
    held_seeds = {_request_seed(sc.exp_seed, fid) for fid in sc.held_ids}
    assert not search_seeds & held_seeds, (
        "search and held-out stages share a request seed: a seed-keyed "
        "backend cache could confuse the stages"
    )
    # The spy's seed->fixture inversion is injective: every backend call is
    # attributable to exactly one fixture.
    assert len(sc.backend.seed_to_fid) == len(set(sc.backend.seed_to_fid.values())), (
        "request-seed collision: two fixtures share one backend seed"
    )
    # Reseeding actually re-derives per-fixture seeds.
    assert _request_seed(7, "S1") != _request_seed(8, "S1"), (
        "reseeding did not change the per-fixture request seed"
    )
    # Determinism: an identical rerun is byte-identical (no shared mutable
    # RNG or cross-run state).
    sc2 = make_scenario(
        core_mod,
        exp_seed=7,
        search_specs=_STD_SEARCH,
        held_specs=_STD_HELD,
    )
    result2 = run_experiment_on(exp_mod, sc2)
    assert json.dumps(result, sort_keys=True) == json.dumps(result2, sort_keys=True), (
        "identical rerun diverged: shared mutable state or RNG leak"
    )
    return result


# ---------------------------------------------------------------------------
# Pytest entry points against the real (unmodified) sources.
# ---------------------------------------------------------------------------


class HeldoutSeparationAuditTests(unittest.TestCase):
    """unittest entry point; pytest runs TestCase classes natively."""

    @staticmethod
    def test_ambiguous_selection_abstains_without_touching_held_out():
        audit_ambiguous_abstains(real_exp, real_core)

    @staticmethod
    def test_unique_winner_flow_measures_held_out_once_fresh():
        audit_unique_winner_flow(real_exp, real_core)

    @staticmethod
    def test_compatible_flow_when_held_out_passes():
        audit_compatible_flow(real_exp, real_core)

    @staticmethod
    def test_no_candidate_flow_never_measures_held_out():
        audit_no_candidate_flow(real_exp, real_core)

    @staticmethod
    def test_request_seeds_disjoint_and_runs_deterministic():
        audit_seed_separation(real_exp, real_core)

    def test_split_rejects_empty_search(self):
        with self.assertRaises(ValueError):
            real_core.Split(search=(), held_out=(_fx("H1"),))

    def test_split_rejects_empty_held_out(self):
        with self.assertRaises(ValueError):
            real_core.Split(search=(_fx("S1"),), held_out=())

    def test_split_rejects_duplicate_id_across_splits(self):
        with self.assertRaises(ValueError):
            real_core.Split(search=(_fx("DUP"),), held_out=(_fx("DUP", text="other?"),))

    def test_split_rejects_nfc_variant_duplicate_id(self):
        with self.assertRaises(ValueError):
            real_core.Split(search=(_fx("caf\u00e9"),), held_out=(_fx("cafe\u0301"),))

    def test_split_rejects_shared_group_across_splits(self):
        with self.assertRaises(ValueError):
            real_core.Split(
                search=(_fx("S1", group="shared-group"),),
                held_out=(_fx("H1", group="shared-group"),),
            )

    def test_split_rejects_identical_messages_across_splits(self):
        with self.assertRaises(ValueError):
            real_core.Split(
                search=(_fx("S1", text="same prompt?"),),
                held_out=(_fx("H1", text="same prompt?"),),
            )

    def test_run_experiment_rejects_unvalidated_split(self):
        sc = make_scenario(
            real_core,
            exp_seed=7,
            search_specs=[("S1", "sq?", "a")],
            held_specs=[("H1", "hq?", "a")],
        )
        with self.assertRaises(ValueError):
            real_exp.run_experiment(
                sc.backend,
                sc.formats,
                ("not", "a", "split"),  # type: ignore[arg-type]
                boundary=_boundary_ids(),
                max_new_tokens=MAX_NEW_TOKENS,
                seed=sc.exp_seed,
            )

    def test_run_experiment_rejects_empty_or_duplicate_candidates(self):
        sc = make_scenario(
            real_core,
            exp_seed=7,
            search_specs=[("S1", "sq?", "a")],
            held_specs=[("H1", "hq?", "a")],
        )
        with self.assertRaises(ValueError):
            real_exp.run_experiment(
                sc.backend, (), sc.split, boundary=_boundary_ids(), seed=sc.exp_seed
            )
        with self.assertRaises(ValueError):
            real_exp.run_experiment(
                sc.backend,
                (sc.formats[0], sc.formats[0]),
                sc.split,
                boundary=_boundary_ids(),
                seed=sc.exp_seed,
            )

    def test_evaluate_rejects_non_tuple_cases(self):
        sc = make_scenario(
            real_core,
            exp_seed=7,
            search_specs=[("S1", "sq?", "a")],
            held_specs=[("H1", "hq?", "a")],
        )
        with self.assertRaises(ValueError):
            real_core.evaluate(
                sc.backend,
                sc.formats[0],
                [sc.split.search[0]],  # type: ignore[arg-type] -- must be a tuple
                boundary=_boundary_ids(),
                max_new_tokens=MAX_NEW_TOKENS,
                seed=sc.exp_seed,
            )

    def test_split_and_fixtures_are_immutable(self):
        sc = make_scenario(
            real_core,
            exp_seed=7,
            search_specs=[("S1", "sq?", "a")],
            held_specs=[("H1", "hq?", "a")],
        )
        with self.assertRaises(FrozenInstanceError):
            sc.split.search = ()  # type: ignore[assignment]
        with self.assertRaises(FrozenInstanceError):
            sc.split.search[0].id = "mutated"  # type: ignore[assignment]
        assert type(sc.split.search) is tuple and type(sc.split.held_out) is tuple

    @staticmethod
    def test_no_shared_mutable_state_between_runs():
        """Mutating one run's result document cannot affect any later run."""
        kwargs = {
            "exp_seed": 7,
            "search_specs": [("S1", "sq?", "a")],
            "held_specs": [("H1", "hq?", "a")],
            # Unique search winner (cand-A) so the held-out stage actually runs.
            "behavior": {("cand-B", "S1"): "fail_answer"},
        }
        result1 = run_experiment_on(real_exp, make_scenario(real_core, **kwargs))
        snapshot = json.dumps(result1, sort_keys=True)
        # Hostile in-place mutation of the returned document...
        result1["search"][0]["observations"][0]["answer"] = "MUTATED"
        result1["held_out"]["observations"][0]["output_ids"] = [0]
        # ...must not leak into a fresh run: every run builds fresh records.
        result3 = run_experiment_on(real_exp, make_scenario(real_core, **kwargs))
        assert json.dumps(result3, sort_keys=True) == snapshot

    @staticmethod
    def test_result_documents_are_fresh_objects_per_run():
        behavior = {("cand-B", "S1"): "fail_answer"}
        sc1 = make_scenario(
            real_core,
            exp_seed=7,
            search_specs=[("S1", "sq?", "a")],
            held_specs=[("H1", "hq?", "a")],
            behavior=behavior,
        )
        sc2 = make_scenario(
            real_core,
            exp_seed=7,
            search_specs=[("S1", "sq?", "a")],
            held_specs=[("H1", "hq?", "a")],
            behavior=behavior,
        )
        r1 = run_experiment_on(real_exp, sc1)
        r2 = run_experiment_on(real_exp, sc2)
        assert r1["search"][0]["observations"][0] is not r2["search"][0]["observations"][0]
        assert r1["held_out"] is not r2["held_out"]
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)

    def test_fuzz_separation_matches_independent_oracle(self):
        for trial in range(24):
            with self.subTest(trial=trial):
                rng = random.Random(0xC63 + trial)  # noqa: S311 -- seeded fuzz PRNG, not cryptographic
                n_search = rng.choice([2, 3, 4])
                n_held = rng.choice([1, 2, 3])
                pool = [f"F{i}" for i in range(n_search + n_held)]
                rng.shuffle(pool)
                search_ids, held_ids = pool[:n_search], pool[n_search:]

                # Randomize fixture order inside each partition: the audit must be
                # order-agnostic, and observations must follow the split's order.
                search_specs = [(fid, f"question {fid}?", f"ans-{fid}") for fid in search_ids]
                held_specs = [(fid, f"question {fid}?", f"ans-{fid}") for fid in held_ids]
                rng.shuffle(search_specs)
                rng.shuffle(held_specs)

                behavior: dict[tuple[str, str], str] = {}
                for cand in CANDIDATES:
                    for fid in search_ids + held_ids:
                        roll = rng.random()
                        if roll >= 0.55:
                            behavior[(cand, fid)] = (
                                "fail_answer" if rng.random() < 0.5 else "fail_boundary"
                            )

                sc = make_scenario(
                    real_core,
                    exp_seed=1000 + trial,
                    search_specs=search_specs,
                    held_specs=held_specs,
                    behavior=behavior,
                )
                result = run_experiment_on(real_exp, sc)

                want_decision, want_winner = _oracle_decision(
                    behavior, CANDIDATES, sc.search_ids, sc.held_ids
                )
                assert result["decision"] == want_decision, (
                    f"trial {trial}: lab decided {result['decision']!r}, "
                    f"oracle says {want_decision!r}"
                )
                assert result["selected_candidate"] == want_winner

                # Backend call audit: every call attributable; each (candidate, search
                # fixture) measured exactly once; held-out measured iff there is a
                # unique winner, exactly once per held-out fixture, for the winner, and
                # strictly after every search-stage call.
                seq = _call_sequence(sc)
                search_set, held_set = set(sc.search_ids), set(sc.held_ids)
                assert all(
                    cand in CANDIDATES and fid in search_set | held_set for cand, fid in seq
                ), f"trial {trial}: backend measured an unknown fixture: {seq}"
                search_calls = [(c, f) for c, f in seq if f in search_set]
                held_calls = [(c, f) for c, f in seq if f in held_set]
                assert sorted(search_calls) == sorted(
                    (c, f) for c in CANDIDATES for f in sc.search_ids
                ), f"trial {trial}: search-stage coverage wrong: {search_calls}"
                if want_winner is None:
                    assert held_calls == [], f"trial {trial}: held-out measured without a winner"
                    assert result["held_out"] is None
                else:
                    assert held_calls == [(want_winner, h) for h in sc.held_ids], (
                        f"trial {trial}: held-out stage wrong: {held_calls}"
                    )
                    assert result["held_out"] is not None
                    last_search = max(i for i, s in enumerate(seq) if s in search_calls)
                    first_held = min(i for i, s in enumerate(seq) if s in held_calls)
                    assert last_search < first_held, f"trial {trial}: stages interleaved"
                    held_obs = result["held_out"]["observations"]
                    assert [o["fixture"] for o in held_obs] == sc.held_ids
                    for obs in held_obs:
                        assert obs["candidate"] == want_winner
                        assert tuple(obs["output_ids"]) == sc.backend.expected_output_ids(
                            want_winner, obs["fixture"], MAX_NEW_TOKENS
                        ), (
                            f"trial {trial}: held-out observation "
                            f"for {obs['fixture']} is stale/reused"
                        )

                # Search records: one entry per candidate, observations follow the
                # split's (shuffled) order, verdicts match the behavior table.
                assert [e["format"]["name"] for e in result["search"]] == list(CANDIDATES)
                for entry in result["search"]:
                    assert [o["fixture"] for o in entry["observations"]] == sc.search_ids, (
                        f"trial {trial}: search observations do not follow split order"
                    )
                _search_verdicts_match_behavior(result, behavior)


# ---------------------------------------------------------------------------
# Split-construction contract: every cross-split contamination attempt must
# be rejected at construction time, before any measurement can happen.
# ---------------------------------------------------------------------------


def _fx(fid: str, text: str = "q?", expected: str = "a", group: str | None = None) -> Any:
    return real_core.Fixture(
        id=fid,
        messages=(real_core.Message(role="user", content=text),),
        expected=expected,
        group=group or f"group-{fid}",
    )


# ---------------------------------------------------------------------------
# Red team: shared mutable state, in-place mutation, result freshness.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fuzz: randomized behaviors, split sizes, and fixture orderings, each trial
# checked against an independent oracle derived from the behavior table alone
# (the oracle never calls run_experiment and never sees its internals).
# ---------------------------------------------------------------------------


def _oracle_decision(
    behavior: dict[tuple[str, str], str],
    candidates: tuple[str, ...],
    search_ids: list[str],
    held_ids: list[str],
) -> tuple[str, str | None]:
    winners = [
        c for c in candidates if all(behavior.get((c, f), "pass") == "pass" for f in search_ids)
    ]
    if len(winners) != 1:
        return ("ambiguous" if winners else "no_candidate"), None
    winner = winners[0]
    held_ok = all(behavior.get((winner, h), "pass") == "pass" for h in held_ids)
    return ("compatible_on_fixture_set" if held_ok else "held_out_failed"), winner


if __name__ == "__main__":
    unittest.main()
