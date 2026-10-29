"""Cycle-45 re-audit of the ``run_experiment`` core loop (experiments.py).

Scope: the heart of the tool -- ``run_experiment`` in
``model_protocol_lab/experiments.py`` -- last differentially audited in
cycle 21. This file re-verifies the REAL behavior with a fresh,
hand-rolled, independent reference oracle plus directed property tests.

Method
------
* A scripted deterministic world (``World`` + ``TableBackend``) makes
  ``generate()`` a pure function of (candidate, fixture): scripted
  outcomes are pass / wrong-answer / no-boundary / both, padded to
  exactly the token cap. The backend also logs every generate call
  (candidate, fixture, cap, request seed) for ordering/discipline checks.
* ``oracle_run_experiment`` reimplements the documented algorithm from
  scratch -- search every candidate, the conservative unique-winner rule
  (exactly one candidate passing every search fixture wins; ties and
  total failures abstain), held-out evaluation of the winner only,
  decision strings, result-dict assembly, per-candidate boundary
  resolution, the failure taxonomy -- using independently written code
  (own subsequence search, own summary math, own classification).
  Only logic-free leaves are shared with the implementation
  (``Observation``/``Generation`` constructors, ``asdict``, ``sha256``,
  the fixed ``FAILURE_MODES`` vocabulary).
* ``differential_fuzz`` runs both against each other over randomized
  fixtures, token budgets, format families, boundaries (including
  per-candidate ``boundary_map`` overrides), eos ids and seeds, and
  diffs the full result dicts field by field. Any divergence is a real
  bug and would be documented here (none found -- see Findings).
* Directed tests additionally pin: (a) search/held-out separation under
  adversarial fixture layouts, (b) exact token-budget accounting,
  (c) byte-identical determinism for a fixed seed, (d) the conservative
  unique-winner rule, plus boundary edge cases, backend failure paths,
  and result-schema echoes.

Findings (2026-10-15)
---------------------
No real bugs found. Every differential fuzz iteration (200 seeds) and
every directed property test agrees with the independent oracle:
search/held-out discipline holds (held-out runs only for a unique
winner, strictly after search, and flipping held-out outcomes never
changes the selection), budget accounting is exact (every generate call
receives exactly the configured cap; every limit observation carries
exactly cap tokens), same-seed runs are byte-identical, and ties
abstain (``ambiguous`` with no held-out run). Backend failure paths
(``generate`` raising, ``reason="error"`` with allowlisted vs.
non-allowlisted codes) also match the oracle.

Mutation gate (2026-10-15): six temporary mutants were applied to a COPY
of experiments.py (off-by-one search budget, ties-select-first, held-out
gating removed, winner held-out evaluated with the default boundary,
lax >=1 winner rule, inverted held-out verdict). The differential fuzz
(24 iterations, fixed master seed) caught all six; the unmutated control
run passed. experiments.py was never modified (sha256 verified
before/after).
"""

import json
import random
import re
import unittest
from dataclasses import asdict
from hashlib import sha256

from model_protocol_lab.core import (
    Fixture,
    Format,
    Generation,
    Message,
    Observation,
    Split,
)
from model_protocol_lab.experiments import FAILURE_MODES, classify_observation, run_experiment

# Fixed-vocabulary backend error codes, mirroring the documented
# sanitization boundary in core.py: only these codes pass through
# verbatim; anything else collapses to "backend_reported_error".
_ERROR_CODES = frozenset(
    {
        "invalid_or_empty_input",
        "invalid_output_cap",
        "invalid_seed",
        "invalid_cancellation_event",
        "context_limit_exceeded",
    }
)

_DECISIONS = (
    "compatible_on_fixture_set",
    "held_out_failed",
    "ambiguous",
    "no_candidate",
    "cancelled",
    "failed",
)

# chr()-decodable pad range; boundary ids are huge ints that can never
# collide with encoded text, so scripted boundaries never appear by
# accident inside an answer region.
_PAD_BASE = 65
_PAD_MOD = 26


def _request_seed(seed, case_id):
    """Mirror of core.evaluate's per-request seed derivation (backend
    contract mechanics, not audited logic): request-local and stable.
    """
    material = f"{seed}\0{case_id}".encode("utf-8", errors="surrogatepass")
    return int.from_bytes(sha256(material).digest()[:4], "big")


def _first_occurrence(haystack, needle):
    """Independent first-subsequence search (cf. core.first_boundary)."""
    if not needle:
        raise ValueError("A nonempty boundary is required")
    width = len(needle)
    for index in range(len(haystack) - width + 1):
        if tuple(haystack[index : index + width]) == tuple(needle):
            return index
    return None


# ---------------------------------------------------------------------------
# Scripted world: deterministic backend fixture.
# ---------------------------------------------------------------------------


def _random_format(rng, index):
    """A random format *family*: distinct rendering knobs per candidate."""
    bos = rng.choice(["", "<bos>", "### "])
    return Format(
        name=f"cand-{index}",
        bos=bos,
        user_prefix=f"<c{index}>USR:",
        assistant_prefix=rng.choice(["AST:", "AST:\n", " A:"]),
        turn_end=rng.choice(["</t>", "\n", ""]),
        repeat_bos=rng.choice([True, False]),
        user_turn_end=rng.choice([None, "|u>", ""]),
        assistant_turn_end=rng.choice([None, "|a>"]),
        open_suffix=rng.choice([None, "<open>"]),
    )


class World:
    """Everything both the backend and the oracle read: fixtures, formats,
    per-candidate boundaries, and the scripted (candidate, fixture) plan.
    """

    def __init__(
        self,
        rng,
        n_cand,
        n_search,
        n_held,
        boundary,
        boundary_map,
        seed,
        eos_id,
        n_winners,
        held_ok,
        *,
        expected_choices=None,
        cap_override=None,
        force_winner_map=False,
    ):
        self.seed = seed
        self.eos_id = eos_id
        self.boundary = boundary
        self.expected_choices = expected_choices or ["a", "bb", "ccc", "yes", "no", "42", ""]
        self.boundaries = {
            f"cand-{i}": (boundary_map or {}).get(f"cand-{i}", boundary) for i in range(n_cand)
        }
        self.formats = tuple(_random_format(rng, i) for i in range(n_cand))
        self.search = tuple(
            self._make_fixture(rng, "s", j, rng.randrange(0xFFFF)) for j in range(n_search)
        )
        self.held = tuple(
            self._make_fixture(rng, "h", j, rng.randrange(0xFFFF)) for j in range(n_held)
        )
        self.split = Split(search=self.search, held_out=self.held)
        self.fixtures = {f.id: f for f in self.search + self.held}
        self.winners = rng.sample(range(n_cand), min(n_winners, n_cand))
        self.script = {}
        self._plant_search(rng, n_cand)
        self._plant_held(rng, n_cand, held_ok)
        self.effective_boundary_map = dict(boundary_map or {})
        if force_winner_map and len(self.winners) == 1:
            fresh = tuple(sorted(rng.sample(range(4_000_000, 4_001_000), len(boundary))))
            if fresh == self.boundaries[f"cand-{self.winners[0]}"]:
                fresh = tuple(t + 7 for t in fresh)
            self.boundaries[f"cand-{self.winners[0]}"] = fresh
            self.effective_boundary_map[f"cand-{self.winners[0]}"] = fresh
        self._finalize_cap(rng, cap_override)

    def _make_fixture(self, rng, tag, j, nonce):
        """Adversarial-friendly fixtures: ids carry a nonce, expected answers
        may collide across the split, messages stay unique per fixture.
        """
        fid = f"fx-{tag}-{j}-{nonce:04x}"
        question = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(1, 8)))
        return Fixture(
            id=fid,
            messages=(Message("user", f"<{fid}> {question}?"),),
            expected=rng.choice(self.expected_choices),
            group=f"{tag}-group-{rng.randint(0, 2)}",
        )

    def inject_fail(self, cand_name, fid, fail):
        """Directed failure injection: 'raise' or ('error', code)."""
        self.script[(cand_name, fid)]["fail"] = fail

    def _plant_search(self, rng, n_cand):
        for i in range(n_cand):
            for fixture in self.search:
                if i in self.winners:
                    kind = "pass"
                else:
                    kind = rng.choice(["wrong", "noboundary", "both", "pass"])
                self.script[(f"cand-{i}", fixture.id)] = {"kind": kind, "fail": None}
        # Every non-winner must fail at least one search fixture.
        for i in range(n_cand):
            if i not in self.winners:
                fixture = rng.choice(self.search)
                self.script[(f"cand-{i}", fixture.id)]["kind"] = rng.choice(
                    ["wrong", "noboundary", "both"]
                )

    def _plant_held(self, rng, n_cand, held_ok):
        for i in range(n_cand):
            for fixture in self.held:
                self.script[(f"cand-{i}", fixture.id)] = {
                    "kind": rng.choice(["pass", "wrong", "noboundary", "both"]),
                    "fail": None,
                }
        if len(self.winners) == 1:
            winner = self.winners[0]
            bad_fixture = None if held_ok else rng.choice(self.held)
            for fixture in self.held:
                if fixture is bad_fixture:
                    kind = rng.choice(["wrong", "noboundary", "both"])
                else:
                    kind = "pass"
                self.script[(f"cand-{winner}", fixture.id)] = {"kind": kind, "fail": None}

    def _scripted_ids(self, cand_name, fixture, request_seed):
        """Token ids the scripted model emits for (candidate, fixture).

        Returns (body, pad_token); the error path uses the body unpadded.
        Shared fixture logic (the backend IS the fixture here); the
        audited orchestration logic lives in the oracle below.
        """
        spec = self.script[(cand_name, fixture.id)]
        kind = spec["kind"]
        text = fixture.expected if kind in ("pass", "noboundary") else "WRONG"
        ids = tuple(ord(c) for c in text)
        if kind in ("pass", "wrong"):
            ids = ids + self.boundaries[cand_name]
        pad = _PAD_BASE + (request_seed % _PAD_MOD)
        return ids, pad

    def output_for(self, cand_name, fixture, cap, request_seed):
        body, pad = self._scripted_ids(cand_name, fixture, request_seed)
        if len(body) > cap:
            raise AssertionError("scripted world inconsistent: output exceeds cap")
        return body + (pad,) * (cap - len(body))

    def error_output_for(self, cand_name, fixture, request_seed):
        body, _ = self._scripted_ids(cand_name, fixture, request_seed)
        return body

    def _finalize_cap(self, rng, cap_override):
        need = 0
        for (cand_name, fid), spec in self.script.items():
            if spec["fail"] == "raise":
                continue
            body, _ = self._scripted_ids(cand_name, self.fixtures[fid], 0)
            need = max(need, len(body))
        self.need = need
        if cap_override is not None:
            if cap_override < need:
                raise AssertionError("cap_override below scripted need")
            self.cap = cap_override
        else:
            self.cap = max(1, need + rng.choice([0, 1, 2, 5]))

    @staticmethod
    def tag_of(fid):
        return "search" if "-s-" in fid else "held"


class TableBackend:
    """Deterministic scripted backend; logs every generate call."""

    fingerprint = "cycle45-audit-backend"

    def __init__(self, world):
        self.world = world
        self.calls = []  # (cand_name, fixture_id, cap, request_seed)

    @staticmethod
    def encode(text):
        return tuple(ord(c) for c in text)

    @staticmethod
    def decode(token_ids):
        return "".join(chr(t) for t in token_ids)

    def _locate(self, input_ids):
        text = self.decode(input_ids)
        cand = int(re.search(r"<c(\d+)>", text).group(1))
        fid = re.search(r"fx-[sh]-\d+-[0-9a-f]{4}", text).group(0)
        return f"cand-{cand}", fid

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        cand_name, fid = self._locate(input_ids)
        self.calls.append((cand_name, fid, max_new_tokens, seed))
        fixture = self.world.fixtures[fid]
        fail = self.world.script[(cand_name, fid)]["fail"]
        if fail == "raise":
            raise RuntimeError("scripted generate failure")
        if isinstance(fail, tuple):
            _, code = fail
            return Generation(
                token_ids=self.world.error_output_for(cand_name, fixture, seed),
                reason="error",
                error=code,
            )
        return Generation(
            token_ids=self.world.output_for(cand_name, fixture, max_new_tokens, seed),
            reason="limit",
        )


# ---------------------------------------------------------------------------
# Independent oracle: reimplemented from the documented algorithm.
# ---------------------------------------------------------------------------


def oracle_resolve_boundaries(formats, boundary, boundary_map):
    """Per-candidate boundaries: default for all, map overrides named ones."""
    if boundary_map is None:
        return {fmt.name: boundary for fmt in formats}
    if type(boundary_map) is not dict:
        raise ValueError("boundary_map must be a dict or None")
    names = {fmt.name for fmt in formats}
    resolved = {}
    for name, ids in boundary_map.items():
        if type(name) is not str or name not in names:
            raise ValueError(f"boundary_map names an unknown candidate: {name!r}")
        if not (type(ids) is tuple and ids and all(type(t) is int and t >= 0 for t in ids)):
            raise ValueError(f"boundary_map boundary for {name!r} must be a nonempty tuple")
        resolved[name] = ids
    for fmt in formats:
        resolved.setdefault(fmt.name, boundary)
    return resolved


def oracle_text_match(backend, output_ids, cut, boundary):
    """Diagnostic boundary-text check; backend trouble degrades to None."""
    try:
        boundary_text = backend.decode(boundary)
    except Exception:  # noqa: BLE001 -- diagnostic must not fail the run
        return None
    if type(boundary_text) is not str or not boundary_text:
        return None
    retained = output_ids if cut is None else output_ids[: cut + len(boundary)]
    try:
        retained_text = backend.decode(retained)
    except Exception:  # noqa: BLE001 -- diagnostic must not fail the run
        return None
    if type(retained_text) is not str:
        return None
    return retained_text.endswith(boundary_text)


def _oracle_completed_obs(
    world, backend, fmt, fixture, prompt, input_ids, output_ids, cut, boundary
):
    answer_ids = output_ids if cut is None else output_ids[:cut]
    answer = backend.decode(answer_ids)
    return Observation(
        candidate=fmt.name,
        fixture=fixture.id,
        input_ids=input_ids,
        output_ids=output_ids,
        answer=answer,
        exact_match=answer == fixture.expected,
        boundary_found=cut is not None,
        boundary_text_match=oracle_text_match(backend, output_ids, cut, boundary),
        raw_reason="limit",
        error=None,
        status="completed",
        input_text=prompt,
    )


def _oracle_failed_obs(fmt, fixture, prompt, input_ids, output_ids, cut, raw_reason, error):
    return Observation(
        candidate=fmt.name,
        fixture=fixture.id,
        input_ids=input_ids,
        output_ids=output_ids,
        answer="",
        exact_match=False,
        boundary_found=cut is not None,
        boundary_text_match=None,
        raw_reason=raw_reason,
        error=error,
        status="failed",
        input_text=prompt,
    )


def oracle_evaluate(world, backend, fmt, cases, cand_boundary, cap, seed):
    """Replicate evaluate's per-fixture loop from its documented contract:
    render -> encode -> generate -> boundary cut -> decode, retaining
    incomplete-run evidence with fixed-vocabulary error codes.

    Note: backend decode failures ("decode_failed") are deliberately not
    replicated. The scripted backend's decode is total over every id the
    correct implementation ever decodes: a candidate is always evaluated
    with the boundary it emits, so huge boundary ids never leak into a
    decoded answer region (cut=None implies nothing huge was emitted).
    A wrong-boundary mutant (M4 in the gate) does leak them and the real
    implementation records "decode_failed" -- surfacing as an
    implementation/oracle divergence, which is exactly what the gate
    checks for.
    """
    records = []
    for case in cases:
        prompt = fmt.render(case.messages)
        input_ids = backend.encode(prompt)
        request_seed = _request_seed(seed, case.id)
        fail = world.script[(fmt.name, case.id)]["fail"]
        if fail == "raise":
            records.append(
                _oracle_failed_obs(fmt, case, prompt, input_ids, (), None, None, "generate_failed")
            )
            return records, "failed"
        if isinstance(fail, tuple):
            _, code = fail
            output_ids = world.error_output_for(fmt.name, case, request_seed)
            cut = _first_occurrence(output_ids, cand_boundary)
            clean = code if code in _ERROR_CODES else "backend_reported_error"
            records.append(
                _oracle_failed_obs(fmt, case, prompt, input_ids, output_ids, cut, "error", clean)
            )
            return records, "failed"
        output_ids = world.output_for(fmt.name, case, cap, request_seed)
        cut = _first_occurrence(output_ids, cand_boundary)
        records.append(
            _oracle_completed_obs(
                world, backend, fmt, case, prompt, input_ids, output_ids, cut, cand_boundary
            )
        )
    return records, "completed"


def oracle_failure_code(records):
    for obs in reversed(records):
        if type(obs.error) is str and obs.error:
            return obs.error
    return None


def oracle_summary(records, planned, status):
    eligible = status == "completed" and len(records) == planned
    passed = sum(1 for r in records if r.exact_match and r.boundary_found)
    return {
        "status": status,
        "planned": planned,
        "attempted": len(records),
        "aggregate_eligible": eligible,
        "exact_and_boundary_passes": passed,
        "rate": passed / planned if eligible else None,
        "encoded_input_tokens": sum(len(r.input_ids) for r in records),
        "retained_output_tokens": sum(len(r.output_ids) for r in records),
        "failure_code": oracle_failure_code(records),
    }


def oracle_fingerprint(fmt):
    payload = asdict(fmt)
    if not payload["bos"]:
        payload["repeat_bos"] = False
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + sha256(text.encode("utf-8")).hexdigest()


def oracle_configuration_digest(
    backend, formats, split, *, boundary, boundary_map, max_new_tokens, seed, eos_id
):
    """Per _configuration_digest docstring: sha256 over the canonical JSON
    of the backend fingerprint, the candidate fingerprints in order, the
    full fixture content of both splits (id, messages, expected answer,
    group), the boundary and per-format boundary map, max_new_tokens,
    seed, and eos_id. Written from the docstring's field list, not
    imported from the implementation, so the oracle independently pins
    the field (cycle 207: the digest is a new run_experiment output
    key, and the oracle diff must carry it or every oracle case
    fails on the extra key).
    """

    def fixture_payload(case):
        return {
            "id": case.id,
            "messages": [asdict(message) for message in case.messages],
            "expected": case.expected,
            "group": case.group,
        }

    payload = {
        "backend_fingerprint": backend.fingerprint,
        "candidate_fingerprints": [oracle_fingerprint(fmt) for fmt in formats],
        "search": [fixture_payload(case) for case in split.search],
        "held_out": [fixture_payload(case) for case in split.held_out],
        "boundary_token_ids": list(boundary),
        "boundary_map": (
            {name: list(ids) for name, ids in sorted(boundary_map.items())}
            if boundary_map is not None
            else None
        ),
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "eos_id": eos_id,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + sha256(text.encode("utf-8")).hexdigest()


def oracle_classify(obs, eos_id):
    if type(obs) is not dict:
        raise ValueError("observation must be a dict")
    if obs.get("exact_match") and obs.get("boundary_found"):
        return None
    if obs.get("status") == "failed" or obs.get("raw_reason") == "error":
        return "backend-error"
    if obs.get("status") == "cancelled":
        return "backend-error"
    if not obs.get("boundary_found"):
        if obs.get("boundary_text_match") is True:
            return "boundary-text-mismatch"
        if eos_id is not None and eos_id in tuple(obs.get("output_ids") or ()):
            return "eos-without-boundary"
        return "no-boundary-in-N"
    return "no-boundary-in-N"


def oracle_taxonomy(result, eos_id):
    per_candidate = {}
    for entry in result.get("search") or []:
        if type(entry) is not dict:
            continue
        fmt = entry.get("format")
        name = fmt.get("name") if type(fmt) is dict else None
        if type(name) is not str:
            continue
        counts = dict.fromkeys(FAILURE_MODES, 0)
        observations = entry.get("observations")
        if type(observations) is list:
            for obs in observations:
                mode = oracle_classify(obs, eos_id)
                if mode is not None:
                    counts[mode] += 1
        per_candidate[name] = counts
    held_out_counts = None
    if result.get("decision") == "held_out_failed":
        held = result.get("held_out")
        observations = held.get("observations") if type(held) is dict else None
        held_out_counts = dict.fromkeys(FAILURE_MODES, 0)
        if type(observations) is list:
            for obs in observations:
                if type(obs) is dict and not (obs.get("exact_match") and obs.get("boundary_found")):
                    held_out_counts["held-out-divergence"] += 1
    return {
        "modes": list(FAILURE_MODES),
        "per_candidate": per_candidate,
        "held_out": held_out_counts,
        "eos_id_considered": eos_id is not None,
    }


def oracle_run_experiment(
    world, backend, formats, split, *, boundary, boundary_map, cap, seed, eos_id
):
    """Independent reimplementation of run_experiment's orchestration."""
    boundaries = oracle_resolve_boundaries(formats, boundary, boundary_map)
    output = {
        "schema_version": 1,
        "selection_rule": "unique_all_search_pass",
        "measurement": "exact_answer_and_boundary",
        "seed": seed,
        "max_new_tokens": cap,
        "boundary_token_ids": list(boundary),
        "eos_token_id": eos_id,
        "backend_fingerprint": backend.fingerprint,
        "configuration_digest": oracle_configuration_digest(
            backend,
            formats,
            split,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=cap,
            seed=seed,
            eos_id=eos_id,
        ),
        "statistical_confidence": "not_estimated",
        "compute_comparison": "not_equal_compute_certified",
        "decision": "not_started",
        "search": [],
        "held_out": None,
        "selected_candidate": None,
    }

    def finish():
        output["failure_taxonomy"] = oracle_taxonomy(output, eos_id)
        return output

    winners = []
    for fmt in formats:
        cand_boundary = boundaries[fmt.name]
        records, status = oracle_evaluate(
            world, backend, fmt, split.search, cand_boundary, cap, seed
        )
        eligible = status == "completed" and len(records) == len(split.search)
        summary = oracle_summary(records, len(split.search), status)
        output["search"].append(
            {
                "format": asdict(fmt),
                "candidate_fingerprint": oracle_fingerprint(fmt),
                "boundary_token_ids": list(cand_boundary),
                "summary": summary,
                "observations": [asdict(r) for r in records],
            }
        )
        if not eligible:
            output["decision"] = status
            return finish()
        if summary["exact_and_boundary_passes"] == len(split.search):
            winners.append(fmt)
    if len(winners) != 1:
        output["decision"] = "ambiguous" if winners else "no_candidate"
        return finish()
    selected = winners[0]
    output["selected_candidate"] = selected.name
    selected_boundary = boundaries[selected.name]
    output["selected_boundary_token_ids"] = list(selected_boundary)
    held_records, held_status = oracle_evaluate(
        world, backend, selected, split.held_out, selected_boundary, cap, seed
    )
    output["held_out"] = {
        "summary": oracle_summary(held_records, len(split.held_out), held_status),
        "observations": [asdict(r) for r in held_records],
    }
    held_eligible = held_status == "completed" and len(held_records) == len(split.held_out)
    if not held_eligible:
        output["decision"] = held_status
    else:
        ok = all(r.exact_match and r.boundary_found for r in held_records)
        output["decision"] = "compatible_on_fixture_set" if ok else "held_out_failed"
    return finish()


# ---------------------------------------------------------------------------
# Diff engine + direct property checks + fuzz driver.
# ---------------------------------------------------------------------------


def diff_results(actual, expected, path="result", _out=None):
    """Field-by-field recursive diff; [] means identical."""
    if _out is None:
        _out = []

    def rec(a, e, p):
        if isinstance(a, dict) and isinstance(e, dict):
            for key in sorted(set(a) | set(e), key=repr):
                if key not in a:
                    _out.append(f"{p}[{key!r}]: <missing> != {e[key]!r:.120}")
                elif key not in e:
                    _out.append(f"{p}[{key!r}]: {a[key]!r:.120} != <missing>")
                else:
                    rec(a[key], e[key], f"{p}[{key!r}]")
        elif isinstance(a, (list, tuple)) and isinstance(e, (list, tuple)):
            if len(a) != len(e):
                _out.append(f"{p}: len {len(a)} != len {len(e)}")
            else:
                for i, (x, y) in enumerate(zip(a, e, strict=True)):
                    rec(x, y, f"{p}[{i}]")
        elif type(a) is not type(e) or a != e:
            _out.append(f"{p}: {a!r:.120} != {e!r:.120}")

    rec(actual, expected, path)
    return _out


def check_properties(actual, backend, world, cap, seed, eos_id):
    """Direct audits of the real result that do not go through the oracle:

    budget accounting, generate-call ordering, search/held-out discipline,
    token sums, and config echoes. Returns a list of problem strings.
    """
    problems = []

    def problem(text):
        problems.append(text)

    # -- budget accounting: every generate call saw exactly the configured cap
    for cand_name, fid, call_cap, _ in backend.calls:
        if call_cap != cap:
            problem(f"generate cap {call_cap} != configured {cap} for {cand_name}/{fid}")
    if actual.get("max_new_tokens") != cap:
        problem(f"result max_new_tokens {actual.get('max_new_tokens')} != {cap}")

    # -- call ordering: search for every candidate in order, then held-out
    #    for the unique winner only
    expected_calls = [
        (fmt.name, f.id, cap, _request_seed(seed, f.id))
        for fmt in world.formats
        for f in world.search
    ]
    unique_winner = world.winners[0] if len(world.winners) == 1 else None
    if unique_winner is not None:
        wname = f"cand-{unique_winner}"
        expected_calls += [(wname, f.id, cap, _request_seed(seed, f.id)) for f in world.held]
    if backend.calls != expected_calls:
        problem(
            "generate call sequence diverged: "
            f"got {len(backend.calls)} calls, expected {len(expected_calls)}"
        )

    # -- held-out discipline
    search_ids = {f.id for f in world.search}
    held_ids = {f.id for f in world.held}
    held_calls = [c for c in backend.calls if c[1] in held_ids]
    search_calls = [c for c in backend.calls if c[1] in search_ids]
    if held_calls:
        first_held = backend.calls.index(held_calls[0])
        last_search = backend.calls.index(search_calls[-1])
        if first_held < last_search:
            problem("held-out generate calls interleave with search calls")
    decision = actual.get("decision")
    selected = actual.get("selected_candidate")
    if unique_winner is None:
        if held_calls:
            problem(f"held-out ran with no unique winner (decision={decision})")
        if actual.get("held_out") is not None:
            problem("held_out present with no unique winner")
        if selected is not None:
            problem(f"selected_candidate set with no unique winner: {selected}")
        if decision not in ("ambiguous", "no_candidate", "failed", "cancelled"):
            problem(f"unexpected decision without unique winner: {decision}")
    else:
        wname = f"cand-{unique_winner}"
        if selected != wname:
            problem(f"selected {selected} != planted unique winner {wname}")
        if {c[1] for c in held_calls} != held_ids:
            problem("held-out did not cover exactly the held-out fixtures")
        if any(c[0] != wname for c in held_calls):
            problem("held-out evaluated a non-winner candidate")
        if actual.get("held_out") is None:
            problem("held_out missing for unique winner")
        if decision not in ("compatible_on_fixture_set", "held_out_failed", "failed", "cancelled"):
            problem(f"unexpected decision with unique winner: {decision}")

    # -- token sums and per-observation budget
    for entry in actual.get("search", []):
        name = entry["format"]["name"]
        obs = entry["observations"]
        summary = entry["summary"]
        if summary["encoded_input_tokens"] != sum(len(o["input_ids"]) for o in obs):
            problem(f"{name}: encoded_input_tokens mismatch")
        if summary["retained_output_tokens"] != sum(len(o["output_ids"]) for o in obs):
            problem(f"{name}: retained_output_tokens mismatch")
        if entry["boundary_token_ids"] != list(world.boundaries[name]):
            problem(f"{name}: boundary_token_ids != resolved per-candidate boundary")
        for o in obs:
            if o["status"] == "completed" and len(o["output_ids"]) != cap:
                problem(f"{name}/{o['fixture']}: output length != cap")
            if o["fixture"] not in search_ids:
                problem(f"{name}: search observation on non-search fixture {o['fixture']}")
    held = actual.get("held_out")
    if held is not None:
        for o in held["observations"]:
            if o["fixture"] not in held_ids:
                problem(f"held-out observation on non-held-out fixture {o['fixture']}")
            if o["status"] == "completed" and len(o["output_ids"]) != cap:
                problem(f"held-out {o['fixture']}: output length != cap")

    # -- config echoes
    for key, want in (
        ("seed", seed),
        ("boundary_token_ids", list(world.boundary)),
        ("eos_token_id", eos_id),
        ("backend_fingerprint", backend.fingerprint),
        ("schema_version", 1),
        ("selection_rule", "unique_all_search_pass"),
        ("measurement", "exact_answer_and_boundary"),
    ):
        if actual.get(key) != want:
            problem(f"echo {key}: {actual.get(key)!r} != {want!r}")
    if decision not in _DECISIONS:
        problem(f"decision {decision!r} outside fixed vocabulary")
    has_selected_boundary = "selected_boundary_token_ids" in actual
    if (selected is None) == has_selected_boundary:
        problem("selected_boundary_token_ids presence inconsistent with selection")
    return problems


def random_config(rng):
    n_cand = rng.randint(1, 5)
    n_search = rng.randint(1, 6)
    n_held = rng.randint(1, 4)
    boundary = tuple(sorted(rng.sample(range(2_000_000, 2_001_000), rng.randint(1, 3))))
    boundary_map = None
    if rng.random() < 0.5:
        boundary_map = {}
        for i in rng.sample(range(n_cand), rng.randint(1, n_cand)):
            override = tuple(sorted(rng.sample(range(3_000_000, 3_001_000), rng.randint(1, 3))))
            boundary_map[f"cand-{i}"] = override
    seed = rng.randrange(0, 2**64)
    eos_id = rng.choice([None, None, 66, 90])
    n_winners = rng.choice([0, 0, 1, 1, 1, 2, 2, 3])
    held_ok = rng.random() < 0.6
    return {
        "n_cand": n_cand,
        "n_search": n_search,
        "n_held": n_held,
        "boundary": boundary,
        "boundary_map": boundary_map,
        "seed": seed,
        "eos_id": eos_id,
        "n_winners": n_winners,
        "held_ok": held_ok,
    }


def run_one_case(rng, force_winner_map=False):
    """Build a random world, run implementation vs oracle, return findings."""
    cfg = random_config(rng)
    world = World(
        rng,
        cfg["n_cand"],
        cfg["n_search"],
        cfg["n_held"],
        cfg["boundary"],
        cfg["boundary_map"],
        cfg["seed"],
        cfg["eos_id"],
        cfg["n_winners"],
        cfg["held_ok"],
        force_winner_map=force_winner_map,
    )
    backend = TableBackend(world)
    actual = run_experiment(
        backend,
        world.formats,
        world.split,
        boundary=world.boundary,
        boundary_map=world.effective_boundary_map or None,
        max_new_tokens=world.cap,
        seed=world.seed,
        eos_id=world.eos_id,
    )
    expected = oracle_run_experiment(
        world,
        backend,
        world.formats,
        world.split,
        boundary=world.boundary,
        boundary_map=world.effective_boundary_map or None,
        cap=world.cap,
        seed=world.seed,
        eos_id=world.eos_id,
    )
    diffs = diff_results(actual, expected)
    problems = check_properties(actual, backend, world, world.cap, world.seed, world.eos_id)
    coverage = {
        "decision": actual["decision"],
        "boundary_map": bool(world.effective_boundary_map),
        "eos_none": world.eos_id is None,
        "n_winners": len(world.winners),
    }
    return diffs, problems, coverage, cfg


def differential_fuzz(n_iter=48, master_seed=20261015):
    """Fuzz implementation vs oracle; returns (mismatches, problems, coverage).

    Pure function of (n_iter, master_seed); raises nothing -- findings are
    returned so the mutation-gate driver can run this without pytest.
    """
    rng = random.Random(master_seed)  # noqa: S311 -- deterministic seeded fuzz, not crypto
    mismatches = []
    problems = []
    decisions = set()
    map_used = 0
    eos_seen = set()
    for it in range(n_iter):
        diffs, props, coverage, cfg = run_one_case(rng, force_winner_map=(it % 6 == 5))
        decisions.add(coverage["decision"])
        map_used += coverage["boundary_map"]
        eos_seen.add(coverage["eos_none"])
        if diffs:
            mismatches.append((it, cfg, diffs[:8]))
        if props:
            problems.append((it, cfg, props[:8]))
    coverage = {
        "decisions": sorted(decisions),
        "boundary_map_iterations": map_used,
        "eos_variants": sorted(eos_seen),
        "iterations": n_iter,
    }
    return mismatches, problems, coverage


# ---------------------------------------------------------------------------
# Test scaffolding.
# ---------------------------------------------------------------------------


def make_world(
    master,
    n_cand=3,
    n_search=3,
    n_held=2,
    n_winners=1,
    held_ok=True,
    seed=1234,
    eos_id=None,
    boundary=(2000001, 2000002),
    boundary_map=None,
    expected_choices=None,
    cap_override=None,
    force_winner_map=False,
):
    rng = random.Random(master)  # noqa: S311 -- deterministic test seed
    return World(
        rng,
        n_cand,
        n_search,
        n_held,
        boundary,
        boundary_map,
        seed,
        eos_id,
        n_winners,
        held_ok,
        expected_choices=expected_choices,
        cap_override=cap_override,
        force_winner_map=force_winner_map,
    )


def run_actual(world):
    backend = TableBackend(world)
    result = run_experiment(
        backend,
        world.formats,
        world.split,
        boundary=world.boundary,
        boundary_map=world.effective_boundary_map or None,
        max_new_tokens=world.cap,
        seed=world.seed,
        eos_id=world.eos_id,
    )
    return result, backend


def oracle_of(world, backend):
    return oracle_run_experiment(
        world,
        backend,
        world.formats,
        world.split,
        boundary=world.boundary,
        boundary_map=world.effective_boundary_map or None,
        cap=world.cap,
        seed=world.seed,
        eos_id=world.eos_id,
    )


def assert_oracle_diff(case, world, result, backend, label):
    diffs = diff_results(result, oracle_of(world, backend))
    case.assertEqual(
        diffs,
        [],
        f"oracle/implementation mismatch ({label}):\n" + "\n".join(diffs[:12]),
    )


def assert_oracle_full(case, world, result, backend, label):
    assert_oracle_diff(case, world, result, backend, label)
    problems = check_properties(result, backend, world, world.cap, world.seed, world.eos_id)
    case.assertEqual(problems, [], f"property violations ({label}):\n" + "\n".join(problems[:12]))


class RunExperimentDifferentialFuzz(unittest.TestCase):
    def test_differential_fuzz_200(self):
        mismatches, problems, coverage = differential_fuzz(n_iter=200, master_seed=20261015)
        self.assertEqual(
            mismatches,
            [],
            "implementation/oracle divergences:\n"
            + "\n".join(f"iter {it} cfg={cfg}: {d}" for it, cfg, d in mismatches[:4]),
        )
        self.assertEqual(
            problems,
            [],
            "direct property violations:\n"
            + "\n".join(f"iter {it} cfg={cfg}: {p}" for it, cfg, p in problems[:4]),
        )
        for decision in (
            "compatible_on_fixture_set",
            "held_out_failed",
            "ambiguous",
            "no_candidate",
        ):
            self.assertIn(
                decision,
                coverage["decisions"],
                f"fuzz never produced decision {decision}; widen the plant",
            )
        self.assertGreaterEqual(
            coverage["boundary_map_iterations"],
            8,
            "too few boundary_map iterations to trust per-candidate coverage",
        )
        self.assertEqual(
            coverage["eos_variants"],
            [False, True],
            "fuzz must exercise both eos_id=None and eos_id set",
        )


class RunExperimentDirectedAudit(unittest.TestCase):
    def test_unique_winner_compatible(self):
        world = make_world(7, n_cand=3, n_search=4, n_held=2, n_winners=1, held_ok=True)
        result, backend = run_actual(world)
        assert_oracle_full(self, world, result, backend, "unique winner")
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        self.assertEqual(result["selected_candidate"], f"cand-{world.winners[0]}")
        self.assertEqual(
            result["selected_boundary_token_ids"],
            list(world.boundaries[f"cand-{world.winners[0]}"]),
        )
        self.assertEqual(len(result["held_out"]["observations"]), 2)

    def test_selection_invariance_under_heldout_flip(self):
        """Held-out outcomes must never change the selection: the search
        stage alone decides the winner.
        """
        world = make_world(8, n_cand=3, n_search=3, n_held=2, n_winners=1, held_ok=True)
        result1, backend1 = run_actual(world)
        assert_oracle_full(self, world, result1, backend1, "pre-flip")
        self.assertEqual(result1["decision"], "compatible_on_fixture_set")
        winner = result1["selected_candidate"]
        # Flip a held-out fixture for the winner to a boundary-less failure.
        # "noboundary" keeps the body short so the finalized cap still holds.
        world.script[(winner, world.held[0].id)]["kind"] = "noboundary"
        result2, backend2 = run_actual(world)
        assert_oracle_full(self, world, result2, backend2, "post-flip")
        self.assertEqual(result2["selected_candidate"], winner)
        self.assertEqual(result2["decision"], "held_out_failed")
        self.assertEqual(result1["search"], result2["search"])

    def test_ties_abstain(self):
        for n_winners, master in ((2, 11), (3, 12)):
            world = make_world(
                master, n_cand=3, n_search=3, n_held=2, n_winners=n_winners, held_ok=True
            )
            result, backend = run_actual(world)
            assert_oracle_full(self, world, result, backend, f"{n_winners}-way tie")
            self.assertEqual(result["decision"], "ambiguous")
            self.assertIsNone(result["selected_candidate"])
            self.assertIsNone(result["held_out"])
            self.assertNotIn("selected_boundary_token_ids", result)
            held_calls = [c for c in backend.calls if world.tag_of(c[1]) == "held"]
            self.assertEqual(held_calls, [])

    def test_all_pass_tie_abstains(self):
        world = make_world(13, n_cand=2, n_search=2, n_held=1, n_winners=2, held_ok=True)
        result, backend = run_actual(world)
        assert_oracle_full(self, world, result, backend, "all-pass tie")
        self.assertEqual(result["decision"], "ambiguous")
        self.assertIsNone(result["held_out"])

    def test_no_candidate(self):
        world = make_world(14, n_cand=3, n_search=3, n_held=2, n_winners=0)
        result, backend = run_actual(world)
        assert_oracle_full(self, world, result, backend, "no candidate")
        self.assertEqual(result["decision"], "no_candidate")
        self.assertIsNone(result["selected_candidate"])
        self.assertIsNone(result["held_out"])

    def test_budget_edge_cap_one(self):
        """Token budget of exactly 1: boundary of length 1 still fits, and
        every generate call and observation honors the cap exactly.
        """
        world = make_world(
            31,
            n_cand=1,
            n_search=2,
            n_held=1,
            n_winners=1,
            held_ok=True,
            boundary=(9999999,),
            expected_choices=[""],
            cap_override=1,
        )
        self.assertEqual(world.need, 1)
        result, backend = run_actual(world)
        assert_oracle_full(self, world, result, backend, "cap=1")
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        for entry in result["search"]:
            for obs in entry["observations"]:
                self.assertEqual(len(obs["output_ids"]), 1)
                self.assertEqual(tuple(obs["output_ids"]), (9999999,))

    def test_small_boundary_diagnostic_paths(self):
        """boundary_text_match True/False via a chr-decodable boundary, and
        the classify_observation taxonomy unit contract.
        """
        world = make_world(
            41,
            n_cand=1,
            n_search=2,
            n_held=1,
            n_winners=1,
            held_ok=True,
            boundary=(7,),
            expected_choices=["hi"],
            cap_override=6,
        )
        world.script[("cand-0", world.search[0].id)]["kind"] = "pass"
        world.script[("cand-0", world.search[1].id)]["kind"] = "noboundary"
        result, backend = run_actual(world)
        assert_oracle_diff(self, world, result, backend, "small boundary")
        obs = result["search"][0]["observations"]
        self.assertTrue(obs[0]["boundary_text_match"])
        self.assertFalse(obs[1]["boundary_text_match"])
        self.assertEqual(result["decision"], "no_candidate")
        # classify_observation unit contract against the real implementation.
        self.assertIsNone(classify_observation({"exact_match": True, "boundary_found": True}, None))
        self.assertEqual(
            classify_observation(
                {"exact_match": False, "boundary_found": False, "status": "failed"}, None
            ),
            "backend-error",
        )
        self.assertEqual(classify_observation({"status": "cancelled"}, None), "backend-error")
        self.assertEqual(
            classify_observation(
                {
                    "exact_match": False,
                    "boundary_found": False,
                    "boundary_text_match": True,
                },
                None,
            ),
            "boundary-text-mismatch",
        )
        self.assertEqual(
            classify_observation(
                {"exact_match": False, "boundary_found": False, "output_ids": (1, 66, 2)},
                66,
            ),
            "eos-without-boundary",
        )
        self.assertEqual(
            classify_observation(
                {"exact_match": False, "boundary_found": False, "output_ids": (1, 2)},
                66,
            ),
            "no-boundary-in-N",
        )
        self.assertEqual(
            classify_observation({"exact_match": False, "boundary_found": True}, None),
            "no-boundary-in-N",
        )

    def test_backend_failure_generate_raise(self):
        world = make_world(51, n_cand=3, n_search=3, n_held=2, n_winners=1, held_ok=True)
        world.inject_fail("cand-1", world.search[1].id, "raise")
        result, backend = run_actual(world)
        assert_oracle_diff(self, world, result, backend, "generate raise")
        self.assertEqual(result["decision"], "failed")
        self.assertEqual(len(result["search"]), 2)
        entry = result["search"][1]
        self.assertEqual(entry["summary"]["status"], "failed")
        self.assertEqual(entry["summary"]["attempted"], 2)
        self.assertEqual(entry["summary"]["failure_code"], "generate_failed")
        self.assertFalse(entry["summary"]["aggregate_eligible"])
        self.assertIsNone(entry["summary"]["rate"])
        self.assertIsNone(result["selected_candidate"])
        self.assertIsNone(result["held_out"])
        tax = result["failure_taxonomy"]["per_candidate"]["cand-1"]
        self.assertEqual(tax["backend-error"], 1)
        # The run stops at the failure: no later candidates, no held-out.
        self.assertEqual(backend.calls[-1][:2], ("cand-1", world.search[1].id))
        self.assertEqual([c for c in backend.calls if world.tag_of(c[1]) == "held"], [])

    def test_backend_failure_error_codes(self):
        world = make_world(52, n_cand=2, n_search=2, n_held=1, n_winners=1, held_ok=True)
        world.inject_fail("cand-0", world.search[0].id, ("error", "context_limit_exceeded"))
        result, backend = run_actual(world)
        assert_oracle_diff(self, world, result, backend, "allowlisted error")
        obs = result["search"][0]["observations"][0]
        self.assertEqual(obs["status"], "failed")
        self.assertEqual(obs["raw_reason"], "error")
        self.assertEqual(obs["error"], "context_limit_exceeded")
        self.assertEqual(result["decision"], "failed")

        world2 = make_world(53, n_cand=2, n_search=2, n_held=1, n_winners=1, held_ok=True)
        world2.inject_fail("cand-0", world2.search[0].id, ("error", "weird_custom_code"))
        result2, backend2 = run_actual(world2)
        assert_oracle_diff(self, world2, result2, backend2, "non-allowlisted error")
        self.assertEqual(result2["search"][0]["observations"][0]["error"], "backend_reported_error")

    def test_determinism_byte_identical(self):
        world = make_world(5, n_cand=3, n_search=4, n_held=3, n_winners=1, held_ok=False, seed=777)
        result1, backend1 = run_actual(world)
        result2, backend2 = run_actual(world)
        blob1 = json.dumps(result1, sort_keys=True)
        blob2 = json.dumps(result2, sort_keys=True)
        self.assertEqual(blob1, blob2)
        self.assertEqual(backend1.calls, backend2.calls)
        # A different seed keeps the verdicts but changes the request-local
        # padding, so the bytes differ.
        for other_seed in range(778, 842):
            alt = make_world(
                5,
                n_cand=3,
                n_search=4,
                n_held=3,
                n_winners=1,
                held_ok=False,
                seed=other_seed,
            )
            result3, _ = run_actual(alt)
            if json.dumps(result3, sort_keys=True) != blob1:
                break
        else:
            self.fail("request-seed padding never varied across 64 seeds")
        self.assertEqual(result3["decision"], result1["decision"])

    def test_adversarial_split_layouts(self):
        """Every fixture expects the same answer text and messages are
        near-identical: the split must still hold and the oracle must agree.
        """
        world = make_world(
            21,
            n_cand=2,
            n_search=3,
            n_held=2,
            n_winners=1,
            held_ok=False,
            expected_choices=["same"],
        )
        result, backend = run_actual(world)
        assert_oracle_full(self, world, result, backend, "adversarial layout")
        self.assertEqual(result["decision"], "held_out_failed")
        search_fixtures = {o["fixture"] for e in result["search"] for o in e["observations"]}
        held_fixtures = {o["fixture"] for o in result["held_out"]["observations"]}
        self.assertTrue(search_fixtures.isdisjoint(held_fixtures))

    def test_result_schema_echoes(self):
        world = make_world(61, eos_id=66)
        result, backend = run_actual(world)
        assert_oracle_full(self, world, result, backend, "schema echoes")
        for key, want in (
            ("schema_version", 1),
            ("selection_rule", "unique_all_search_pass"),
            ("measurement", "exact_answer_and_boundary"),
            ("seed", world.seed),
            ("max_new_tokens", world.cap),
            ("boundary_token_ids", list(world.boundary)),
            ("eos_token_id", 66),
            ("backend_fingerprint", "cycle45-audit-backend"),
            ("statistical_confidence", "not_estimated"),
            ("compute_comparison", "not_equal_compute_certified"),
        ):
            self.assertEqual(result[key], want, key)
        self.assertEqual(result["failure_taxonomy"]["modes"], list(FAILURE_MODES))
        self.assertTrue(result["failure_taxonomy"]["eos_id_considered"])
