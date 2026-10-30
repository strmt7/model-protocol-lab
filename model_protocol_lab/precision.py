"""Load-precision x template probe (cycle 198): pure payload computation.

The producer-side half of ``run --precision`` (docs/PRECISION_DESIGN.md).
A second backend instance loads the *same* checkpoint in bfloat16 and
re-evaluates every search candidate on the search split; this module
owns everything derivable from the two arms' ground records:

* the protocol object and its fingerprint (seed, budget, per-candidate
  boundaries, the fixture-set digest, both backend fingerprints, the
  declared config dtype, and both verified arm dtypes);
* the paired flip matrix per candidate -- the four paired cells with
  ``n_paired`` beside them, the one-arm-only counts, and the derived
  unpaired list (a fixture an arm never reached is unpaired, never an
  arm failure);
* the output-divergence aggregates (the sensitive leading indicator;
  verdict flips are the lagging one) and the teacher-forced
  expected-answer logprob deltas, computed from the recorded reported
  (float32-rounded) values only;
* the paired-population selection re-application for both arms and
  the winner-survival readout, withheld on mismatched completed sets,
  on a non-winner-bearing primary, on bf16 unavailability, and at
  zero paired fixtures;
* the falsification readout, ``None`` over zero paired cells.

Ground records carry digests only -- never answer text, never raw
token ids beyond the first-divergence index. A cell's verdict is bound
by ``verdict == (answer_digest == expected_digest) and
boundary_found``; the diff validator recomputes every derived value
from the records with its own mirror and compares type-strictly.
This module never imports the backend or runs a model: the stage in
experiments.py drives evaluation and assembles the records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

#: Protocol version for the precision payload.
PROTOCOL_VERSION = 1

#: Records caps, enforced before any recompute loop (legal max is
#: 24 candidates x 100 fixtures -- the choice-audit precedent).
MAX_RECORDS = 2400
MAX_RECORDS_BYTES = 12 * 1024 * 1024

#: The two load dtypes the probe compares. The fp32 arm is always the
#: recorded native run; the second arm is always bfloat16. fp16 is
#: never substituted (docs/PRECISION_DESIGN.md, dtype integrity).
NATIVE_DTYPE = "float32"
ARM_DTYPE = "bfloat16"

#: Arm keys used in records and selections.
ARMS: tuple[str, str] = ("fp32", "bf16")

#: Selection decisions for an arm over a population (the tree's
#: existing per-condition vocabulary; no new verdict words).
SELECTION_DECISIONS: tuple[str, ...] = ("selected", "ambiguous", "no_candidate", "unavailable")

#: Withholding reasons for the survival readout.
WITHHELD_MISMATCHED_SETS = "mismatched_sets"
WITHHELD_PRIMARY_NOT_WINNER_BEARING = "primary_not_winner_bearing"
WITHHELD_NO_PAIRED_FIXTURES = "no_paired_fixtures"
WITHHELD_BF16_UNAVAILABLE = "bf16_unavailable"

#: Winner-bearing primary decisions (mirrors diffreports'
#: _PRIMARY_WINNER_DECISIONS; the validator binds the pair through
#: _valid_primary_pair, this set drives the survival definition).
_WINNER_BEARING = frozenset({"compatible_on_fixture_set", "held_out_failed"})

_HEX64 = frozenset("0123456789abcdef")


def text_digest(text: str) -> str:
    """SHA-256 of a text (records carry digests, not texts).

    ``surrogatepass`` keeps a pathological expected (a lone surrogate
    can survive the CSV loader) from crashing the probe; for every
    well-formed string the digest is the plain UTF-8 digest. Same
    construction as choiceaudit.text_digest.
    """
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def ids_digest(token_ids: Sequence[int]) -> str:
    """SHA-256 over a token-id sequence's canonical JSON form."""
    canonical = json.dumps(list(token_ids), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fixture_set_sha256(fixture_ids: Sequence[str]) -> str:
    """SHA-256 over the sorted fixture id set (leading's form)."""
    return hashlib.sha256("\n".join(sorted(fixture_ids)).encode("utf-8")).hexdigest()


def is_digest(value: Any) -> bool:
    """True for a lowercase hex SHA-256 digest string."""
    return type(value) is str and len(value) == 64 and all(c in _HEX64 for c in value)


def protocol_object(
    *,
    seed: int,
    max_new_tokens: int,
    fixture_ids: Sequence[str],
    candidates: Sequence[str],
    boundaries: dict[str, Sequence[int]],
    fp32_fingerprint: str,
    bf16_fingerprint: str,
    declared_dtype: str | None,
    native_dtype: str,
    arm_dtype: str,
) -> dict[str, Any]:
    """The frozen protocol as data; the fingerprint covers it."""
    return {
        "version": PROTOCOL_VERSION,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "max_records": MAX_RECORDS,
        "max_records_bytes": MAX_RECORDS_BYTES,
        "fixture_set_sha256": fixture_set_sha256(fixture_ids),
        "candidates": list(candidates),
        "boundaries": {name: list(boundaries[name]) for name in candidates},
        "fp32_fingerprint": fp32_fingerprint,
        "bf16_fingerprint": bf16_fingerprint,
        "declared_dtype": declared_dtype,
        "native_dtype": native_dtype,
        "arm_dtype": arm_dtype,
    }


def protocol_fingerprint(protocol: dict[str, Any]) -> str:
    """SHA-256 over the canonical protocol object (fingerprint field excluded)."""
    body = {key: value for key, value in protocol.items() if key != "fingerprint"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cell_verdict(cell: dict[str, Any], expected_digest: str) -> bool:
    """The verdict binding: digest equality plus the boundary."""
    return bool(cell["answer_digest"] == expected_digest and cell["boundary_found"])


def _arm_cells(
    records: Sequence[dict[str, Any]], candidate: str, arm: str
) -> dict[str, dict[str, Any]]:
    """fixture -> cell for one candidate/arm, completed cells only."""
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["candidate"] != candidate:
            continue
        cell = record[arm]
        if type(cell) is dict:
            out[str(record["fixture"])] = cell
    return out


def selection_over(
    records: Sequence[dict[str, Any]],
    candidates: Sequence[str],
    arm: str,
    fixtures: frozenset[str],
) -> dict[str, Any] | None:
    """Re-apply the selection rule to one arm over a shared population.

    Returns {"decision", "winner"} or None when the construction
    fails: any candidate with no completed fixture in the population,
    or candidates whose completed sets (within the population) differ.
    A candidate passes the population iff every one of its completed
    fixtures in it passed (the lab's unique-all-pass rule).
    """
    completed_sets: list[frozenset[str]] = []
    all_pass: dict[str, bool] = {}
    for name in candidates:
        cells = _arm_cells(records, name, arm)
        completed = frozenset(fid for fid in cells if fid in fixtures)
        completed_sets.append(completed)
        if not completed:
            return None
        all_pass[name] = all(
            cell_verdict(cells[fid], _expected_of(records, name, fid)) for fid in completed
        )
    if len(set(completed_sets)) > 1:
        return None
    passers = [name for name, ok in all_pass.items() if ok]
    if len(passers) == 1:
        return {"decision": "selected", "winner": passers[0]}
    if len(passers) > 1:
        return {"decision": "ambiguous", "winner": None}
    return {"decision": "no_candidate", "winner": None}


def _expected_of(records: Sequence[dict[str, Any]], candidate: str, fixture: str) -> str:
    """Return the expected digest recorded for a candidate and fixture."""
    for record in records:
        if record["candidate"] == candidate and record["fixture"] == fixture:
            return str(record["expected_digest"])
    raise KeyError((candidate, fixture))


def _candidate_block(
    records: Sequence[dict[str, Any]],
    candidate: str,
    fp32_passed: bool | None,
    bf16_passed: bool | None,
) -> dict[str, Any]:
    """Every per-candidate derived value, from the records alone."""
    own = [r for r in records if r["candidate"] == candidate]
    paired = [r for r in own if type(r["fp32"]) is dict and type(r["bf16"]) is dict]
    cells = {"pass_pass": 0, "pass_fail": 0, "fail_pass": 0, "fail_fail": 0}  # nosec B105 -- 'pass' keys are verdict counts in the frozen payload schema, not credentials
    flips: list[dict[str, Any]] = []
    n_diverged = 0
    deltas: list[float] = []
    for record in paired:
        fp32_v = cell_verdict(record["fp32"], str(record["expected_digest"]))
        bf16_v = cell_verdict(record["bf16"], str(record["expected_digest"]))
        if fp32_v and bf16_v:
            cells["pass_pass"] += 1
        elif fp32_v:
            cells["pass_fail"] += 1
            flips.append({"fixture": record["fixture"], "direction": "pass_to_fail"})
        elif bf16_v:
            cells["fail_pass"] += 1
            flips.append({"fixture": record["fixture"], "direction": "fail_to_pass"})
        else:
            cells["fail_fail"] += 1
        if record["diverged"] is True:
            n_diverged += 1
        fp32_lp = record["fp32"]["logprob"]
        bf16_lp = record["bf16"]["logprob"]
        if type(fp32_lp) is float and type(bf16_lp) is float:
            deltas.append(bf16_lp - fp32_lp)
    n_paired = len(paired)
    return {
        "fp32_passed": fp32_passed,
        "bf16_passed": bf16_passed,
        "n_paired": n_paired,
        "cells": cells,
        "flips": flips,
        "n_fp32_only": sum(1 for r in own if type(r["fp32"]) is dict and r["bf16"] is None),
        "n_bf16_only": sum(1 for r in own if type(r["bf16"]) is dict and r["fp32"] is None),
        "n_diverged": n_diverged,
        "divergence_rate": (n_diverged / n_paired) if n_paired else None,
        "logprob": {
            "n": len(deltas),
            "mean_delta": (sum(deltas) / len(deltas)) if deltas else None,
            "min_delta": min(deltas) if deltas else None,
            "max_delta": max(deltas) if deltas else None,
        },
    }


def compute_derived(
    records: Sequence[dict[str, Any]],
    candidates: Sequence[str],
    arm_passed: dict[str, dict[str, bool | None]],
    primary_decision: str,
    _primary_winner: str | None,
) -> dict[str, Any]:
    """All derived payload fields from the ground records.

    ``arm_passed`` carries the stage-attested per-arm all-pass
    verdicts (bool | None per arm per candidate): eligibility is a
    property of the arm's evaluation report, not of the cells, so it
    is attested here and consistency-checked by the validator
    (passed=True requires every fixture's cell present and passing).
    Everything else -- cells, flips, unpaired, divergence, logprob
    summaries, both paired-population selections, survival, and the
    falsification readout -- is computed here and recomputed by the
    diff validator's independent mirror.
    """
    ordered = sorted(records, key=lambda r: (str(r["candidate"]), str(r["fixture"])))
    blocks: dict[str, Any] = {}
    for name in candidates:
        blocks[name] = _candidate_block(
            ordered, name, arm_passed[name]["fp32"], arm_passed[name]["bf16"]
        )
    unpaired: list[dict[str, Any]] = []
    for record in ordered:
        if type(record["fp32"]) is dict and record["bf16"] is None:
            unpaired.append(
                {"candidate": record["candidate"], "fixture": record["fixture"], "arm": "fp32"}
            )
        elif type(record["bf16"]) is dict and record["fp32"] is None:
            unpaired.append(
                {"candidate": record["candidate"], "fixture": record["fixture"], "arm": "bf16"}
            )
    flips_all: list[dict[str, Any]] = []
    for name in candidates:
        for flip in blocks[name]["flips"]:
            flips_all.append(
                {"candidate": name, "fixture": flip["fixture"], "direction": flip["direction"]}
            )
    flips_all.sort(key=lambda f: (str(f["candidate"]), str(f["fixture"])))
    total_paired = sum(blocks[name]["n_paired"] for name in candidates)
    total_cells = {"pass_pass": 0, "pass_fail": 0, "fail_pass": 0, "fail_fail": 0}  # nosec B105 -- 'pass' keys are verdict counts in the frozen payload schema, not credentials
    for name in candidates:
        for key in total_cells:
            total_cells[key] += blocks[name]["cells"][key]
    total_diverged = sum(blocks[name]["n_diverged"] for name in candidates)
    totals = {
        "n_paired": total_paired,
        "cells": total_cells,
        "n_flips": len(flips_all),
        "n_diverged": total_diverged,
        "divergence_rate": (total_diverged / total_paired) if total_paired else None,
    }
    falsified: bool | None = None if total_paired == 0 else len(flips_all) == 0
    # Winner survival over the shared paired population.
    shared: frozenset[str] | None = None
    mismatched = False
    for name in candidates:
        fp32_done = frozenset(_arm_cells(ordered, name, "fp32"))
        bf16_done = frozenset(_arm_cells(ordered, name, "bf16"))
        if fp32_done != bf16_done:
            mismatched = True
        pair = fp32_done & bf16_done
        shared = pair if shared is None else (shared & pair)
    if shared is None:
        shared = frozenset()
    selection: dict[str, Any]
    if primary_decision not in _WINNER_BEARING:
        selection = {
            "paired_population": sorted(shared),
            "fp32": None,
            "bf16": None,
            "survived": None,
            "withheld": WITHHELD_PRIMARY_NOT_WINNER_BEARING,
        }
    elif mismatched:
        selection = {
            "paired_population": sorted(shared),
            "fp32": None,
            "bf16": None,
            "survived": None,
            "withheld": WITHHELD_MISMATCHED_SETS,
        }
    elif not shared:
        selection = {
            "paired_population": [],
            "fp32": None,
            "bf16": None,
            "survived": None,
            "withheld": WITHHELD_NO_PAIRED_FIXTURES,
        }
    else:
        fp32_sel = selection_over(ordered, candidates, "fp32", shared)
        bf16_sel = selection_over(ordered, candidates, "bf16", shared)
        if fp32_sel is None or bf16_sel is None:
            selection = {
                "paired_population": sorted(shared),
                "fp32": fp32_sel,
                "bf16": bf16_sel,
                "survived": None,
                "withheld": WITHHELD_MISMATCHED_SETS,
            }
        else:
            selection = {
                "paired_population": sorted(shared),
                "fp32": fp32_sel,
                "bf16": bf16_sel,
                "survived": fp32_sel == bf16_sel,
                "withheld": None,
            }
    return {
        "candidates": blocks,
        "unpaired": unpaired,
        "flips": flips_all,
        "totals": totals,
        "selection": selection,
        "precision_effect_falsified": falsified,
    }


__all__ = [
    "ARM_DTYPE",
    "ARMS",
    "MAX_RECORDS",
    "MAX_RECORDS_BYTES",
    "NATIVE_DTYPE",
    "PROTOCOL_VERSION",
    "SELECTION_DECISIONS",
    "WITHHELD_BF16_UNAVAILABLE",
    "WITHHELD_MISMATCHED_SETS",
    "WITHHELD_NO_PAIRED_FIXTURES",
    "WITHHELD_PRIMARY_NOT_WINNER_BEARING",
    "cell_verdict",
    "compute_derived",
    "fixture_set_sha256",
    "ids_digest",
    "is_digest",
    "protocol_fingerprint",
    "protocol_object",
    "selection_over",
    "text_digest",
]
