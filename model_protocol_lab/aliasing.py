"""Candidate aliasing audit: did the run test distinct templates?

Candidates are names for renderings. On a given tokenizer and
fixture set, two differently named formats can render byte-identical
``input_ids`` on every search fixture -- a BOS mutation that is a
no-op for a family whose framing already starts with that token, two
families whose framing coincides on short single-turn messages, a
field the fixture set's turn shapes never exercise. A leaderboard
cannot see the collapse (tied rows look normal); this lab can,
because it renders every candidate itself and abstains on ties --
and a tie between two names for one token sequence is a different
event from a tie between two genuinely different prompts.

The audit is a pure post-hoc profile over the recorded search
observations (zero extra model calls). It groups candidates into
equivalence classes inside cohorts of candidates with identical
completed fixture sets, and reports two class notions that must
never be conflated:

- **Render classes** -- identical recorded ``input_ids`` on every
  cohort fixture. Support exactly one claim: these names produced
  the same prompt token sequences on this fixture set.
- **Measurement classes** -- render classes refined by the
  candidate's own response boundary (``boundary_token_ids``).
  Boundaries are part of the measurement in this lab: under
  ``--boundary-map``, two candidates with identical prompts and
  different boundaries are different experiments that happen to
  share a prompt. Effective counts are reported for both notions;
  the redundant-observation count and the alias-suspect flag cite
  measurement classes only.

Pre-registered claims (wave-7 research section 2, as corrected by
the red-team A/B cross-check):

- A1': classes are computed from recorded ``input_ids`` only --
  never from ``candidate_fingerprint`` (it hashes the format's
  name, so fingerprint classes would be singletons by
  construction), the ``format`` dict, or ``input_text``. A cell
  counts only if it is evaluable: completed, with nonempty,
  well-typed ids, a fixture unique within its entry, and a
  ``candidate`` field equal to the entry's name.
- A2': within one measurement class, the recorded ``output_ids``
  are identical per fixture, and the outcome fields
  (``exact_match``, ``boundary_found``, ``boundary_text_match``,
  ``status``) are identical -- checked as independent corollaries,
  not consequences (the outcome fields pass through the backend's
  decode). The premise is checkable and checked: the per-request
  seed derives from the run seed and the fixture ID only, and the
  generation cap is run-global, so identical measurement
  configurations present identical ``generate()`` inputs. A
  violation is reported per cell as a determinism-*suspect* cell:
  recorded cells a deterministic backend could not have produced
  from one measurement configuration, attributable to backend
  nondeterminism (stateful or order-dependent backends included)
  or artifact corruption. The audit asserts neither cause;
  ``verify --determinism`` replay is the discriminating follow-up.
- A3': the alias-suspect abstention flag fires iff the recorded
  decision is ``ambiguous``, the passing set recomputed from the
  ground records has at least two members, the passing set is
  contained in a single measurement class of the primary cohort,
  and that class contains no outcome-suspect cell. The flag is
  advisory: it never modifies the recorded decision or the
  selected candidate, and it asserts the structural containment
  only -- not that any candidate would have survived held-out,
  which was never run for any of them.

The payload carries no raw token IDs and no text: per cell it
carries SHA-256 digests of the canonical ID lists under a frozen
digest definition (``sha256-json-intlist-v1``). Digests are a
compactness and consistency device, not a proof: the diff-reports
validator recomputes every aggregate from the ground records and
so detects aggregates inconsistent with their records, but a
producer that fabricated the records themselves is out of scope --
the standing limitation of every ground layer in this lab. One
privacy consequence is stated plainly: an unsalted digest of a
short, structured token sequence permits offline confirmation of
a guessed prompt. That is strictly weaker than carrying raw IDs
(which reconstruct the prompt exactly), and it is the same trade
the surface-noise probe already makes with ``prompt_sha256``.

Scope, stated once and carried in every payload: search
observations only; completed cells only; classes are relative to
this run's fixture set and backend tokenizer; held-out is never
audited. Back-porting works over ``--include-evidence`` artifacts
and in-memory results only -- a public ``results.json`` has its
observations stripped, and the profile says so
(``observations_stripped``) instead of manufacturing a null result
from absent input.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

AUDIT_ID = "candidate-aliasing-audit-v1"

# The one canonical digest definition for token-ID lists, frozen
# as protocol data: SHA-256 of the compact JSON encoding of the
# list. Digest equality is the payload-level proxy for ID-list
# equality; a SHA-256 collision would be required to alias two
# distinct sequences falsely.
DIGEST_VERSION = "sha256-json-intlist-v1"

# A genuine run records at most 24 candidates x 100 search
# fixtures (run_experiment's own caps) = 2400 cells -- the same
# bound determinism.MAX_AUDIT_CELLS derives. The diff validator
# rejects payloads beyond it rather than recomputing quadratic
# pair structures over hostile input.
MAX_AUDIT_CELLS = 2400
MAX_CANDIDATES = 24

SCOPE = (
    "search observations only; completed cells only; classes are "
    "relative to this run's fixture set and backend tokenizer; "
    "held-out is never audited"
)


def ids_digest(ids: Any) -> str:
    """Canonical digest of a token-ID list under DIGEST_VERSION."""
    canonical = json.dumps(list(ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _valid_ids(value: Any, *, nonempty: bool) -> bool:
    """Whether ``value`` is a list/tuple of exact ints >= 0.

    Bools are excluded by the exact-type check (``isinstance(True,
    int)`` is True, and a bool is never a token ID).
    """
    if type(value) not in (list, tuple):
        return False
    if nonempty and not value:
        return False
    return all(type(item) is int and item >= 0 for item in value)


def is_evaluable_ids(value: Any) -> bool:
    """Whether ``value`` could be a completed cell's input IDs.

    Nonempty is load-bearing: ``core.evaluate``'s abort path
    records ``input_ids == ()`` for render/encode failures, so the
    empty tuple is a failure sentinel that collides with itself --
    two failed candidates would "coincide" on it. Completed cells
    with empty or malformed IDs are artifact defects (the
    determinism audit's ``recorded_ids_invalid``), never class
    members.
    """
    return _valid_ids(value, nonempty=True)


def analyze_candidate(
    observations: Any,
    *,
    candidate_name: str,
    boundary_token_ids: Any,
    eligible: bool,
    planned: Any,
) -> dict[str, Any]:
    """Ground records for one candidate's recorded observations.

    The join discipline is fail-closed: a duplicate fixture, an
    observation whose ``candidate`` field disagrees with the
    entry's name, or a completed cell with invalid recorded IDs
    makes the whole candidate unavailable with a named detail --
    a partial class computation in silence would manufacture the
    evidence this audit exists to check. Digests are computed
    only for evaluable cells; a non-evaluable cell carries
    ``None`` digests, never a digest of the empty list (the abort
    sentinel must not re-enter through the digest layer). Never
    raises on data.
    """
    if observations is None:
        return {"available": False, "detail": "observations_missing"}
    if type(observations) is not list:
        return {"available": False, "detail": "observations_not_a_list"}
    if not _valid_ids(boundary_token_ids, nonempty=True):
        return {"available": False, "detail": "boundary_invalid"}
    records: list[dict[str, Any]] = []
    seen_fixtures: set[str] = set()
    n_completed = 0
    for observation in observations:
        if type(observation) is not dict:
            return {"available": False, "detail": "observation_malformed"}
        fixture = observation.get("fixture")
        status = observation.get("status")
        if (
            type(fixture) is not str
            or not fixture
            or type(status) is not str
            or type(observation.get("exact_match")) is not bool
            or type(observation.get("boundary_found")) is not bool
            or not (
                observation.get("boundary_text_match") is None
                or type(observation.get("boundary_text_match")) is bool
            )
        ):
            return {"available": False, "detail": "observation_malformed"}
        if fixture in seen_fixtures:
            return {"available": False, "detail": "duplicate_observation"}
        seen_fixtures.add(fixture)
        if observation.get("candidate") != candidate_name:
            return {"available": False, "detail": "observation_candidate_mismatch"}
        evaluable = False
        input_sha256: str | None = None
        output_sha256: str | None = None
        if status == "completed":
            n_completed += 1
            if not is_evaluable_ids(observation.get("input_ids")) or not _valid_ids(
                observation.get("output_ids"), nonempty=False
            ):
                return {"available": False, "detail": "recorded_ids_invalid"}
            evaluable = True
            input_sha256 = ids_digest(observation["input_ids"])
            output_sha256 = ids_digest(observation["output_ids"])
        records.append(
            {
                "fixture": fixture,
                "status": status,
                "evaluable": evaluable,
                "input_sha256": input_sha256,
                "output_sha256": output_sha256,
                "exact_match": observation["exact_match"],
                "boundary_found": observation["boundary_found"],
                "boundary_text_match": observation.get("boundary_text_match"),
            }
        )
    return {
        "available": True,
        "eligible": eligible,
        "planned": planned if type(planned) is int and planned >= 0 else None,
        "boundary_token_ids": list(boundary_token_ids),
        "n_observations": len(records),
        "n_completed": n_completed,
        "n_not_completed": len(records) - n_completed,
        "n_evaluable": sum(1 for r in records if r["evaluable"]),
        "records": records,
    }


def evaluable_fixtures(record: dict[str, Any]) -> frozenset[str]:
    """The candidate's completed fixture set, from its records."""
    return frozenset(r["fixture"] for r in record["records"] if r["evaluable"])


def primary_cohort(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The primary cohort: one completed-set group, chosen totally.

    Cohorts are groups of available candidates with identical
    completed fixture sets; classes exist only inside a cohort
    (pairwise identity over differing sets is not transitive, so
    union "classes" would not be classes). On genuine runs the
    eligible candidates always share the full search set and at
    most one ineligible interrupter exists, so the primary cohort
    is the eligible set; the deterministic order below -- most
    candidates, then most fixtures, then lexicographic -- exists
    so foreign payloads get the same answer from the producer and
    the validator.
    """
    groups: dict[frozenset[str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(evaluable_fixtures(record), []).append(record)
    if not groups:
        return []

    def _rank(item: tuple[frozenset[str], list[dict[str, Any]]]) -> tuple[Any, ...]:
        """Return the rank."""
        fixtures, members = item
        return (
            -len(members),
            -len(fixtures),
            tuple(sorted(fixtures)),
            tuple(sorted(str(m.get("candidate")) for m in members)),
        )

    return sorted(groups.items(), key=_rank)[0][1]


def class_partitions(
    members: list[dict[str, Any]],
) -> tuple[list[list[str]], list[list[str]]]:
    """(render, measurement) class member lists for a cohort.

    A render signature is the tuple of input digests over the
    cohort's sorted fixture set; a measurement signature adds the
    candidate's own boundary tuple. Member lists are sorted, and
    the class lists are sorted by first member, so the producer
    and the validator derive identical structures.
    """
    fixtures = sorted(evaluable_fixtures(members[0])) if members else []
    render_groups: dict[tuple[Any, ...], list[str]] = {}
    measurement_groups: dict[tuple[Any, ...], list[str]] = {}
    for record in members:
        by_fixture = {r["fixture"]: r for r in record["records"] if r["evaluable"]}
        render_sig = tuple(by_fixture[f]["input_sha256"] for f in fixtures)
        measurement_sig = render_sig + (tuple(record["boundary_token_ids"]),)
        name = str(record.get("candidate"))
        render_groups.setdefault(render_sig, []).append(name)
        measurement_groups.setdefault(measurement_sig, []).append(name)

    def _sorted(groups: dict[tuple[Any, ...], list[str]]) -> list[list[str]]:
        """Return the sorted."""
        return sorted((sorted(names) for names in groups.values()), key=lambda names: names[0])

    return _sorted(render_groups), _sorted(measurement_groups)


def redundant_counts(
    members: list[dict[str, Any]],
) -> tuple[int, int]:
    """(measurement, render) redundant completed observations.

    One formula per notion, per fixture: candidates completed on
    the fixture minus distinct measurement (resp. render) tuples
    on it, summed over the cohort's fixtures. Each redundant
    observation is counted once; pair-matrix quantities are never
    summed into either number (pairs overlap by construction).
    These observations are redundant as prompt evaluations -- and
    they are also the replicates the A2' cross-check consumes,
    which is why the payload calls them redundant, never wasted.
    """
    if not members:
        return 0, 0
    fixtures = sorted(evaluable_fixtures(members[0]))
    measurement_total = 0
    render_total = 0
    for fixture in fixtures:
        measurement_tuples: set[tuple[Any, ...]] = set()
        render_tuples: set[Any] = set()
        for record in members:
            # skipcq: PTC-W0063 - cohort members share one evaluable fixture set by construction
            cell = next(r for r in record["records"] if r["evaluable"] and r["fixture"] == fixture)
            render_tuples.add(cell["input_sha256"])
            measurement_tuples.add((cell["input_sha256"], tuple(record["boundary_token_ids"])))
        measurement_total += len(members) - len(measurement_tuples)
        render_total += len(members) - len(render_tuples)
    return measurement_total, render_total


def suspect_cells(
    members: list[dict[str, Any]],
    measurement_classes: list[list[str]],
) -> list[dict[str, Any]]:
    """A2' violations: per (measurement class, fixture) suspect cells.

    A cell is suspect when class members recorded different output
    digests (kind ``output``), different outcome fields (kind
    ``outcome``), or both, on the same fixture. Outcome-suspect
    cells are the ones that bear on selection; a post-boundary-only
    output divergence is reported but does not by itself impeach
    the aliasing explanation of a tie (see the flag assembly).
    """
    by_name = {str(r.get("candidate")): r for r in members}
    fixtures = sorted(evaluable_fixtures(members[0])) if members else []
    cells: list[dict[str, Any]] = []
    for names in measurement_classes:
        if len(names) < 2:
            continue
        for fixture in fixtures:
            outputs: set[Any] = set()
            outcomes: set[tuple[Any, ...]] = set()
            for name in names:
                # skipcq: PTC-W0063 - cohort members share one evaluable fixture set by construction
                cell = next(
                    r
                    for r in by_name[name]["records"]
                    if r["evaluable"] and r["fixture"] == fixture
                )
                outputs.add(cell["output_sha256"])
                outcomes.add(
                    (
                        cell["exact_match"],
                        cell["boundary_found"],
                        cell["boundary_text_match"],
                        cell["status"],
                    )
                )
            output_bad = len(outputs) > 1
            outcome_bad = len(outcomes) > 1
            if output_bad or outcome_bad:
                if output_bad and outcome_bad:
                    kind = "both"
                elif output_bad:
                    kind = "output"
                else:
                    kind = "outcome"
                cells.append({"fixture": fixture, "candidates": list(names), "kind": kind})
    cells.sort(key=lambda c: (c["fixture"], c["candidates"]))
    return cells


def pair_matrix(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-pair render coincidence over completed-set intersections.

    Pairwise data, not classes: a pair entry reports how many of
    the fixtures both candidates completed they coincided on,
    with the denominator explicit. Entries over differing
    completed sets are legitimate here (the intersection is the
    denominator), which is exactly why the matrix must never be
    clustered into "partial classes" -- connected components of a
    coincidence graph are not equivalence classes.
    """
    available = [r for r in records if r.get("available") is True]
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(available):
        left_cells = {r["fixture"]: r["input_sha256"] for r in left["records"] if r["evaluable"]}
        for right in available[i + 1 :]:
            right_cells = {
                r["fixture"]: r["input_sha256"] for r in right["records"] if r["evaluable"]
            }
            common = sorted(set(left_cells) & set(right_cells))
            if not common:
                continue
            shared = [f for f in common if left_cells[f] == right_cells[f]]
            pairs.append(
                {
                    "candidates": sorted([str(left.get("candidate")), str(right.get("candidate"))]),
                    "n_common_completed": len(common),
                    "n_shared": len(shared),
                    "shared_fixture_ids": shared,
                }
            )
    pairs.sort(key=lambda p: p["candidates"])
    return pairs
