"""Multi-turn drift probe (cycle 189, research wave-8 rank 2).

Every other probe in the lab is effectively single-exchange,
yet the lab's problem statement is that conversations are
formatted incorrectly. This probe asks whether the winner
chosen on single-exchange fixtures still wins when the same
information arrives sharded across turns and the model's own
earlier replies become the history -- per template family, per
depth, and whether the active ingredient is the self-written
history or the multi-turn framing bundle around it.

The incorporation policy is frozen in docs/MULTITURN_DESIGN.md
before any code (cycle 179 recorded the policy, not the fixture
schema, as the blocker):

- P1: a turn is a frozen user shard; the model's boundary-cut
  reply text is the next turn's assistant message, verbatim,
  re-rendered as history under the candidate's own Format.
- P2: only the final turn is scored, against the fixture's
  single ``expected``, under the unchanged exact-match +
  boundary rule. Intermediate replies are never scored.
- P3: the probe is informational and never touches native
  selection; per-depth selection readouts re-apply the
  conservative rule and are withheld when completed sets
  differ.

Fixtures are generated from the run seed (SHA-256 derivation,
never ``random.Random``): two task families (ledger-sum,
shelf-argmax) at depths 2 and 3, 12 fixtures per family per
depth. Three arms per fixture: ``full`` (all shards in one
user turn), ``sharded`` (one shard per turn, self-generated
history), ``scripted`` (one shard per turn, the frozen reply
``Noted.`` as every intermediate). The sharded-vs-scripted
contrast is a bundle -- authorship, content, and history
length move together, because content and length are the
mechanism of self-history -- so the headline is the discordant
pair, never a net-only difference (the cycle-185 settlement:
marginal subtraction was disproved as an acquittal rule).

Generated text re-enters prompts as history, so every
intermediate reply is screened for special-token literals
before incorporation (a forged turn boundary inside history
is token-real, not inert text); a screen hit ends the cell as
``arm_screened``. Records carry the final answer's digest
only; the cell verdict is bound by the identity
``verdict == (answer_digest == sha256(expected)) and
boundary_found``, recomputed by the diff validator from a
mirror-regenerated fixture set. The one stage-attested link
is the correspondence between the recorded intermediate
digests and the history actually rendered -- disclosed in the
design's Honesty section, not papered over.

The generated tasks are a probe instrument, not a benchmark:
they measure whether verdicts survive sharded delivery on
tasks whose entire information content fits in three short
shards. Nothing here estimates performance on real user
conversations.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .confidentiality import mcnemar_exact_p

PROTOCOL_VERSION = 1

#: Task families, in canonical generation order.
FAMILIES: tuple[str, ...] = ("ledger", "shelf")
#: Shard counts per fixture.
DEPTHS: tuple[int, ...] = (2, 3)
FIXTURES_PER_FAMILY_DEPTH = 12
LABELS: tuple[str, ...] = ("A", "B", "C")
GROUP = "multiturn"

LEDGER_QUESTION = "What is the ledger total?"
SHELF_QUESTION = "Which shelf holds the most units? Answer with the shelf label."
#: The full arm joins shards with this literal (dateline JOINER
#: precedent).
JOINER = "\n\n"
#: The scripted arm's frozen intermediate reply, identical for
#: every candidate, fixture, and turn.
SCRIPTED_REPLY = "Noted."

#: The shard text shapes, fingerprinted in the protocol object.
#: ``{v}`` renders via ``str(int)`` -- entry style and expected
#: style coincide, so entry formatting cannot become a form
#: trap (the numform lesson).
SHARD_TEMPLATES: tuple[str, ...] = (
    "Ledger entry: {v}.",
    "Ledger entry: {v}. " + LEDGER_QUESTION,
    "Shelf {label} holds {v} units.",
    "Shelf {label} holds {v} units. " + SHELF_QUESTION,
)

ARMS: tuple[str, ...] = ("full", "sharded", "scripted")
#: The per-arm detail vocabulary (house style: dateline's
#: arm_* details).
ARM_DETAILS: tuple[str, ...] = (
    "arm_context_exceeded",
    "arm_evaluation_failed",
    "arm_screened",
)

#: Floors gate the classification only, never a reported rate
#: or count (house constants, shared with numform/dateline).
MIN_PAIRED_EVALUABLE = 6
MIN_DISCORDANT = 2

#: 24 candidates (run_experiment's cap) x 216 generations per
#: candidate (24 depth-2 fixtures x 4 + 24 depth-3 fixtures
#: x 5). Generations and records are different quantities in
#: this probe and carry separate constants: one record per
#: (candidate, fixture) = 24 x 48.
GENERATION_BUDGET = 5184
MAX_GROUND_RECORDS = 1152
MAX_RECORDS_BYTES = 12 * 1024 * 1024

GENERATIONS_PER_CANDIDATE = 216


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text (the records' digest form)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


EMPTY_DIGEST = digest_text("")


def _derive(seed: int, family: str, depth: int, index: int, slot: int, counter: int) -> int:
    """One deterministic draw (the numform ``_derive``
    construction; depth is in the material -- depths are
    independent task sets and no nesting is claimed).
    """
    material = f"multiturn-v1|{seed}|{family}|{depth}|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _ledger_values(seed: int, depth: int, index: int) -> list[int]:
    """Nonzero addends in +/-1..+/-999 whose sum equals no
    single stated entry (the entry-equality screen: a sum that
    restates one entry would be copyable from the shard text).
    A failing fixture is redrawn whole at the next counter.
    """
    counter = 0
    while True:
        values: list[int] = []
        for slot in range(depth):
            derived = _derive(seed, "ledger", depth, index, slot, counter)
            magnitude = 1 + derived % 999
            values.append(magnitude if (derived >> 32) % 2 == 0 else -magnitude)
        if str(sum(values)) not in {str(value) for value in values}:
            return values
        counter += 1


def _shelf_values(seed: int, depth: int, index: int) -> list[int]:
    """Pairwise-distinct holdings in 1..99, so the argmax is
    unique by construction. A colliding slot is redrawn at the
    next counter; earlier slots never move.
    """
    values: list[int] = []
    for slot in range(depth):
        counter = 0
        while True:
            value = 1 + _derive(seed, "shelf", depth, index, slot, counter) % 99
            if value not in values:
                values.append(value)
                break
            counter += 1
    return values


def _shards(family: str, values: list[int]) -> list[str]:
    """The frozen shard texts for one fixture; the question
    rides the final shard, so depth is the entry count.
    """
    depth = len(values)
    shards: list[str] = []
    for slot, value in enumerate(values):
        final = slot == depth - 1
        if family == "ledger":
            text = f"Ledger entry: {value}."
            if final:
                text += f" {LEDGER_QUESTION}"
        else:
            text = f"Shelf {LABELS[slot]} holds {value} units."
            if final:
                text += f" {SHELF_QUESTION}"
        shards.append(text)
    return shards


def _expected(family: str, values: list[int]) -> str:
    """Return the expected."""
    if family == "ledger":
        return str(sum(values))
    return LABELS[max(range(len(values)), key=lambda slot: values[slot])]


def generate_fixtures(seed: int) -> list[dict[str, Any]]:
    """The frozen generated fixture set for a run seed, in
    canonical (family, depth, index) order. Pure and
    deterministic -- the same seed regenerates the identical
    set across runs and PYTHONHASHSEED values.
    """
    if type(seed) is not int:
        raise ValueError("multiturn seed must be an int")
    fixtures: list[dict[str, Any]] = []
    for family in FAMILIES:
        for depth in DEPTHS:
            for index in range(FIXTURES_PER_FAMILY_DEPTH):
                values = (
                    _ledger_values(seed, depth, index)
                    if family == "ledger"
                    else _shelf_values(seed, depth, index)
                )
                fixtures.append(
                    {
                        "fixture_id": f"mt-{family}-{depth}-{index:02d}",
                        "family": family,
                        "depth": depth,
                        "shards": _shards(family, values),
                        "values": values,
                        "expected": _expected(family, values),
                    }
                )
    return fixtures


def canonical_generated_rows(seed: int) -> list[list[Any]]:
    """The canonical generated set the ground digest binds."""
    return [
        [
            fixture["fixture_id"],
            fixture["family"],
            fixture["depth"],
            fixture["shards"],
            fixture["values"],
            fixture["expected"],
        ]
        for fixture in generate_fixtures(seed)
    ]


def ground_sha256(seed: int) -> str:
    """Digest of the canonical generated rows for a seed: two
    runs share an instrument only when this matches, because
    the generated set is the treatment.
    """
    canonical = json.dumps(canonical_generated_rows(seed), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def template_digest() -> str:
    """Digest of the frozen shard templates, questions, joiner,
    and scripted reply -- the protocol text a payload is
    checked against.
    """
    canonical = json.dumps(
        [list(SHARD_TEMPLATES), JOINER, SCRIPTED_REPLY],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The frozen instrument description a payload is checked
    against: families, depths, templates, floors, budgets, and
    the seed/backend pair that generated the set.
    """
    return {
        "version": PROTOCOL_VERSION,
        "families": list(FAMILIES),
        "depths": list(DEPTHS),
        "fixtures_per_family_depth": FIXTURES_PER_FAMILY_DEPTH,
        "arms": list(ARMS),
        "template_digest": template_digest(),
        "ground_sha256": ground_sha256(seed),
        "min_paired_evaluable": MIN_PAIRED_EVALUABLE,
        "min_discordant": MIN_DISCORDANT,
        "generation_budget": GENERATION_BUDGET,
        "max_ground_records": MAX_GROUND_RECORDS,
        "max_records_bytes": MAX_RECORDS_BYTES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }


def protocol_fingerprint(seed: int, backend_fingerprint: str) -> str:
    """SHA-256 over the canonical protocol object."""
    canonical = json.dumps(
        protocol_object(seed, backend_fingerprint),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cell(record: dict[str, Any], arm: str) -> dict[str, Any] | None:
    """Return the cell."""
    cells = record.get("cells")
    if type(cells) is not dict:
        return None
    cell = cells.get(arm)
    return cell if type(cell) is dict else None


def _verdict(record: dict[str, Any], arm: str) -> bool:
    """The arm's verdict, False when the cell never completed.

    Call sites work over pre-filtered paired records, where
    the cell always exists; the None arm exists so mypy can
    narrow without a cast and so a stray unpaired record
    degrades to a fail rather than an exception.
    """
    cell = _cell(record, arm)
    return bool(cell["verdict"]) if cell is not None else False


def _pair_counts(records: list[dict[str, Any]], arm_a: str, arm_b: str) -> dict[str, int]:
    """Discordant counts for one arm pair over the records
    where both arms completed (a = reference arm).
    """
    # skipcq: FLK-E501 - ruff's E501 exemptions cover this frozen, test-pinned literal
    counts = {"n_paired": 0, "a_only": 0, "b_only": 0, "both_pass": 0, "both_fail": 0}  # nosec B105 -- 'pass' keys are verdict counts in the frozen payload schema, not credentials
    for record in records:
        cell_a = _cell(record, arm_a)
        cell_b = _cell(record, arm_b)
        if cell_a is None or cell_b is None:
            continue
        counts["n_paired"] += 1
        verdict_a = bool(cell_a["verdict"])
        verdict_b = bool(cell_b["verdict"])
        if verdict_a and verdict_b:
            counts["both_pass"] += 1
        elif verdict_a:
            counts["a_only"] += 1
        elif verdict_b:
            counts["b_only"] += 1
        else:
            counts["both_fail"] += 1
    return counts


def aggregate_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-candidate readouts recomputed from the records
    alone: per-depth paired blocks (delivery and scripted
    2x2s, the self-history discordant pair, per-family counts,
    completion/attrition) and the pooled classification. The
    diff validator recomputes this exact function over the
    payload's records and rejects on any disagreement.
    """
    depths_out: dict[str, Any] = {}
    for depth in DEPTHS:
        at_depth = [record for record in records if record["depth"] == depth]
        paired_records = [
            record for record in at_depth if all(_cell(record, arm) is not None for arm in ARMS)
        ]
        n_paired = len(paired_records)
        completion: dict[str, Any] = {}
        for arm in ARMS:
            details: dict[str, int] = {}
            completed = 0
            for record in at_depth:
                if _cell(record, arm) is not None:
                    completed += 1
                else:
                    detail = record["cell_details"][arm]
                    details[detail] = details.get(detail, 0) + 1
            completion[arm] = {
                "completed": completed,
                "defined": len(at_depth),
                "details": details,
            }
        passed: dict[str, int] = {}
        rates: dict[str, Any] = {}
        for arm in ARMS:
            n_pass = sum(1 for record in paired_records if _verdict(record, arm))
            passed[arm] = n_pass
            rates[arm] = (n_pass / n_paired) if n_paired else None
        delivery = _pair_counts(paired_records, "full", "sharded")
        scripted_pair = _pair_counts(paired_records, "full", "scripted")
        # The self-history pair, over the paired full-passers:
        # sharded and scripted disagree about a fixture the
        # single-turn arm passed.
        self_history = {
            # skipcq: FLK-E501 - ruff's E501 exemptions cover this frozen, test-pinned literal
            "n_full_pass": 0,  # nosec B105 -- 'pass' keys are verdict counts in the frozen payload schema, not credentials
            "sh_only": 0,
            "sc_only": 0,
            # skipcq: FLK-E501 - ruff's E501 exemptions cover this frozen, test-pinned literal
            "both_pass": 0,  # nosec B105 -- 'pass' keys are verdict counts in the frozen payload schema, not credentials
            "both_fail": 0,
        }
        for record in paired_records:
            if not _verdict(record, "full"):
                continue
            self_history["n_full_pass"] += 1
            sharded = _verdict(record, "sharded")
            scripted = _verdict(record, "scripted")
            if sharded and scripted:
                self_history["both_pass"] += 1
            elif scripted:
                self_history["sh_only"] += 1
            elif sharded:
                self_history["sc_only"] += 1
            else:
                self_history["both_fail"] += 1
        per_family: dict[str, Any] = {}
        for family in FAMILIES:
            family_records = [record for record in paired_records if record["family"] == family]
            fam_delivery = _pair_counts(family_records, "full", "sharded")
            fam_scripted = _pair_counts(family_records, "full", "scripted")
            fam_self = {"sh_only": 0, "sc_only": 0}
            for record in family_records:
                if not _verdict(record, "full"):
                    continue
                sharded = _verdict(record, "sharded")
                scripted = _verdict(record, "scripted")
                if scripted and not sharded:
                    fam_self["sh_only"] += 1
                elif sharded and not scripted:
                    fam_self["sc_only"] += 1
            per_family[family] = {
                "n_paired": len(family_records),
                "drift": fam_delivery["a_only"],
                "gains": fam_delivery["b_only"],
                "drift_scripted": fam_scripted["a_only"],
                "sh_only": fam_self["sh_only"],
                "sc_only": fam_self["sc_only"],
            }
        depths_out[str(depth)] = {
            "n_defined": len(at_depth),
            "n_paired": n_paired,
            "completion": completion,
            "passed": passed,
            "pass_rates": rates,
            "delivery": {
                "drift": delivery["a_only"],
                "gains": delivery["b_only"],
                "both_pass": delivery["both_pass"],
                "both_fail": delivery["both_fail"],
                "mcnemar_p": mcnemar_exact_p(delivery["a_only"], delivery["b_only"]),
            },
            "scripted_contrast": {
                "drift": scripted_pair["a_only"],
                "gains": scripted_pair["b_only"],
                "both_pass": scripted_pair["both_pass"],
                "both_fail": scripted_pair["both_fail"],
                "mcnemar_p": mcnemar_exact_p(scripted_pair["a_only"], scripted_pair["b_only"]),
            },
            "self_history": {
                **self_history,
                "net_excess": self_history["sh_only"] - self_history["sc_only"],
                "mcnemar_p": mcnemar_exact_p(self_history["sh_only"], self_history["sc_only"]),
            },
            "per_family": per_family,
        }
    return {
        "depths": depths_out,
        "classification": classify_candidate(depths_out),
    }


def classify_candidate(depths: dict[str, Any]) -> str:
    """The candidate classification: a total function of the
    per-depth counts (docs/MULTITURN_DESIGN.md). A depth is
    evaluable when its paired denominator reaches the floor;
    the strong label requires at-floor paired self-history
    evidence at a single depth; discordance in either
    direction counts (a one-directional null would acquit a
    surface that only improves verdicts).
    """
    evaluable = [block for block in depths.values() if block["n_paired"] >= MIN_PAIRED_EVALUABLE]
    if not evaluable:
        return "insufficient_evidence"
    for block in evaluable:
        self_history = block["self_history"]
        if (
            self_history["sh_only"] >= MIN_DISCORDANT
            and self_history["sh_only"] > self_history["sc_only"]
        ):
            return "self_history_drift"
    for block in evaluable:
        if (
            max(
                block["delivery"]["drift"],
                block["delivery"]["gains"],
                block["scripted_contrast"]["drift"],
                block["scripted_contrast"]["gains"],
            )
            >= MIN_DISCORDANT
        ):
            return "framing_only"
    for block in evaluable:
        if (
            block["delivery"]["drift"]
            + block["delivery"]["gains"]
            + block["scripted_contrast"]["drift"]
            + block["scripted_contrast"]["gains"]
            + block["self_history"]["sh_only"]
            + block["self_history"]["sc_only"]
            > 0
        ):
            return "below_threshold"
    return "no_drift"
