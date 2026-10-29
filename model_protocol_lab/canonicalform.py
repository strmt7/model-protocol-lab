"""Canonical-form ladder (cycle 209, wave-10 slate rank 1).

Generated fact-recall cases rendered under two canonical forms, NFC
and NFD, per candidate. The measurement is whether the NFC/NFD
distinction survives the checkpoint's full text-to-ids pipeline and,
if it does, whether it moves verdicts.

Binding design pins (docs/CANONICALFORM_DESIGN.md):

* Two arms only: ``nfc``, ``nfd``. No NFKC arm in v1, no
  as-authored control (degenerate == NFC for NFC-authored text).
* THE SCORER PIN. Verdicts are ``core.evaluate``'s byte-exact
  ``answer == case.expected``. A model echoing the prompt's form
  flips its verdict with zero behavioural defect if the expected
  value itself carries accents, so decomposable characters live
  in carrier prose, topic, and question ONLY; every expected
  value is pure ASCII. The generator enforces this; the cycle
  test pins it with a real-``evaluate`` echo execution.
* Generated cases only: no user-fixture arm (shipped fixtures are
  pure ASCII, and Split's NFC keys make form-twins the same
  fixture). Cases are stage-local and never enter the user's
  Split.
* The headline population is ID-DIVERGENT paired cases only
  (both cells completed AND the full render id sequences
  differ). Ids-identical cases are the instrument-inapplicable
  share, never robustness; same-ids verdict flips are a
  determinism finding, never fragility (the ws-scan precedent).
* Floors transplanted from placement: ``MIN_PAIRED_EVALUABLE`` =
  6, ``MIN_DISCORDANT`` = 2. They gate the label only, never a
  reported count or rate.
* Records commit both arms' message-content digests, so the diff
  validator's independent mirror can verify the treatment itself
  (field-order pattern), not just token counts.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any

from .confidentiality import mcnemar_exact_p

PROTOCOL_VERSION = 1

GROUP = "canonicalform"

ARM_KEYS: tuple[str, ...] = ("nfc", "nfd")
BASELINE_ARM = "nfc"

CASE_COUNT = 24

#: Classification floors, frozen at the placement values. They
#: gate the classification label and the falsification boolean
#: only, never a reported count or rate.
MIN_PAIRED_EVALUABLE = 6
MIN_DISCORDANT = 2

#: 24 candidates (run_experiment's cap) x 24 cases; two
#: generations per (candidate, case). Records carry one
#: constant each because the record count is the generation
#: count divided by the arm count.
MAX_GROUND_RECORDS = 24 * CASE_COUNT
GENERATION_BUDGET = len(ARM_KEYS) * MAX_GROUND_RECORDS
#: Records carry two cells with answer digests plus two content
#: digests; the bound is the house 12 MiB for uniformity with
#: the sibling probes.
MAX_RECORDS_BYTES = 12 * 1024 * 1024

#: The classification vocabulary (a total function of the paired
#: counts -- see classify_canonical_form).
CLASSIFICATIONS: tuple[str, ...] = (
    "insufficient",
    "no_discordance_observed",
    "discordant_balanced",
    "discordant_asymmetric",
    "below_threshold",
)

#: The winner-headline vocabulary (a total function of the two
#: arms' selections -- see winner_headline).
WINNER_HEADLINES: tuple[str, ...] = (
    "winner_changed",
    "winner_stable",
    "winner_abstention_asymmetric",
    "winner_undefined",
)

CELL_DETAILS: tuple[str, ...] = (
    "",
    "arm_context_exceeded",
    "arm_evaluation_failed",
)

FACT_TEMPLATE = "The {topic} inspection code is {value}."
QUESTION_TEMPLATE = "What is the {topic} inspection code?"

#: Accented topics, authored in NFC. Every topic contains at
#: least one canonically decomposable character, no singleton
#: (a character whose NFD is a single different codepoint),
#: and no Hangul syllable (deferred stress variant: NFD renders
#: run ~3x the code points, confounding form with length).
#: The alphabet is defined by the mapping predicate in
#: _validate_case_text, not by this list's block; the list is
#: the frozen material the ground digest binds.
TOPICS: tuple[str, ...] = (
    "café",
    "naïve",
    "Zürich",
    "crème",
    "señor",
    "façade",
    "résumé",
    "piñata",
    "jalapeño",
    "soufflé",
    "touché",
    "protégé",
    "fiancé",
    "blasé",
    "cliché",
    "exposé",
    "saké",
    "pâté",
    "brûlée",
    "décor",
    "élan",
    "über",
    "schön",
    "tête",
)

#: The code-word pool for generated values. ASCII only, by the
#: scorer pin: an accented value would let the byte-exact scorer
#: manufacture discordance with zero template effect.
WORDS: tuple[str, ...] = (
    "alder",
    "basalt",
    "cedar",
    "drift",
    "elm",
    "fern",
    "glade",
    "heath",
    "iris",
    "jade",
    "kelp",
    "loom",
    "moss",
    "nova",
    "ochre",
    "pine",
    "quill",
    "reed",
    "sage",
    "tide",
    "urn",
    "vale",
    "wren",
    "yew",
)

#: The neutral carrier sentences. Each carries at least one
#: accented word that is NOT a topic, no digits, no topic
#: tokens, no code words, and no part of the fact template's
#: carrier phrase, so the payload's only answer-bearing
#: sentence is the fact sentence. Accents in the carrier prose
#: are part of the treatment: the topic alone would make the
#: divergence depend on a single word per case.
NEUTRAL_POOL: tuple[str, ...] = (
    "The hôtel lobby smells faintly of coffee and old paper.",
    "A forêt trail winds past the lake before sunrise.",
    "The côte road curves above the harbour all morning.",
    "The fête banners hang above the market stalls at dusk.",
    "A silver locket with a fête ribbon rests on the wooden dresser.",
    "Letters from his mère fill the shallow drawer.",
    "The théâtre posters cover the wall beside the ticket desk.",
    "A matinée crowd gathers outside the cinéma doors.",
    "The musée catalogue lists every painting by number.",
    "Rain taps the lycée windows above the courtyard all day.",
    "The opéra programme rests on top of the piano.",
    "A séance of quiet reading fills the long afternoon.",
    "The Noël market smells of pine and roasted chestnuts.",
    "Dust gathers on the crèche figures in the attic box.",
)


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text (the records' digest form)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_singleton(char: str) -> bool:
    """Whether a character is a canonical singleton: its NFD is
    a single codepoint different from itself (e.g. U+2126 OHM
    SIGN, U+212B ANGSTROM SIGN). Singletons decompose to a
    different letter and are excluded from the v1 alphabet.
    """
    decomposed = unicodedata.normalize("NFD", char)
    return len(decomposed) == 1 and decomposed != char


def _is_hangul_syllable(char: str) -> bool:
    """Whether a character is a Hangul syllable (deferred stress
    variant: NFD renders run ~3x the code points).
    """
    return 0xAC00 <= ord(char) <= 0xD7A3


def _validate_case_text(content: str, value: str) -> None:
    """Enforce the generator's alphabet and scorer-pin rules on
    one authored case. Raises ValueError on any violation, so a
    future table edit cannot silently admit an out-of-alphabet
    character or an accented value into the instrument.
    """
    if not isinstance(content, str) or not content:
        raise ValueError("canonical-form case content must be nonempty text")
    # Singletons first: a singleton is never NFC-stable (its
    # NFD is a single different codepoint, and NFC of that is
    # the codepoint itself), so the NFC check below would
    # shadow this guard entirely if it ran first -- a
    # single-character substitution is a different authoring
    # error from non-NFC text and deserves its own verdict.
    for char in content:
        if _is_singleton(char):
            raise ValueError("canonical-form case content must not contain singletons")
        if _is_hangul_syllable(char):
            raise ValueError("canonical-form case content must not contain Hangul")
    if unicodedata.normalize("NFC", content) != content:
        raise ValueError("canonical-form case content must be authored in NFC")
    if unicodedata.normalize("NFC", content) == unicodedata.normalize("NFD", content):
        raise ValueError("canonical-form case content must be NFC/NFD-divergent")
    if not isinstance(value, str) or not value.isascii():
        raise ValueError("canonical-form expected value must be pure ASCII")


def _derive(seed: int, index: int, slot: int, counter: int) -> int:
    """One deterministic draw. The material string, the slot
    meanings (0 = code word, 1 = number, 2 = first neutral
    sentence, 3 = second neutral sentence), and the redraw
    rules in :func:`generate_cases` are frozen together -- the
    diff validator's mirror reproduces them exactly.
    """
    material = f"{seed}|canonicalform|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def generate_cases(seed: int) -> list[dict[str, Any]]:
    """The frozen generated case set for a run seed, in
    canonical index order. Pure and deterministic -- the same
    seed regenerates the identical set across runs and
    PYTHONHASHSEED values. A value colliding with an earlier
    case's value redraws both value slots at the next counter,
    so the 24 values are pairwise distinct strings by
    construction. The two neutral sentences are distinct by a
    frozen adjustment: when the second draw lands on the first
    sentence's index, it advances by one (mod the pool size).
    The authored content is NFC; the NFD arm is derived from it
    by normalization, never independently authored.
    """
    if type(seed) is not int:
        raise ValueError("canonical-form seed must be an int")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(CASE_COUNT):
        topic = TOPICS[index]
        counter = 0
        while True:
            word = WORDS[_derive(seed, index, 0, counter) % len(WORDS)]
            number = 1000 + _derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
        seen.add(value)
        first = _derive(seed, index, 2, 0) % len(NEUTRAL_POOL)
        second = _derive(seed, index, 3, 0) % len(NEUTRAL_POOL)
        if second == first:
            second = (first + 1) % len(NEUTRAL_POOL)
        fact = FACT_TEMPLATE.format(topic=topic, value=value)
        payload = f"{NEUTRAL_POOL[first]} {fact} {NEUTRAL_POOL[second]}"
        content = f"{payload} {QUESTION_TEMPLATE.format(topic=topic)}"
        _validate_case_text(content, value)
        cases.append(
            {
                "fixture_id": f"canonicalform-{index:02d}",
                "topic": topic,
                "value": value,
                "fact": fact,
                "payload": payload,
                "content": content,
                "question": QUESTION_TEMPLATE.format(topic=topic),
                "expected": value,
                "sentence_indices": [first, second],
            }
        )
    return cases


def form_content(case: dict[str, Any], arm: str) -> str:
    """One arm's user-turn content for a case: the authored NFC
    content, or its NFD derivation. The template is never
    touched; only generated message text is transformed.
    """
    if arm not in ARM_KEYS:
        raise ValueError("unknown canonical-form arm")
    content = str(case["content"])
    if arm == "nfc":
        return unicodedata.normalize("NFC", content)
    return unicodedata.normalize("NFD", content)


def canonical_generated_rows(seed: int) -> list[list[Any]]:
    """The canonical generated set the ground digest binds,
    including both arms' content digests so the ground digest
    binds the treatment, not just the case skeleton.
    """
    rows: list[list[Any]] = []
    for case in generate_cases(seed):
        rows.append(
            [
                case["fixture_id"],
                case["topic"],
                case["value"],
                list(case["sentence_indices"]),
                digest_text(form_content(case, "nfc")),
                digest_text(form_content(case, "nfd")),
            ]
        )
    return rows


def ground_sha256(seed: int) -> str:
    """Digest of the canonical generated set: two runs share
    an instrument only when this matches, because the generated
    set is the treatment.
    """
    canonical = json.dumps(canonical_generated_rows(seed), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def case_material_sha256() -> str:
    """Digest of the frozen case material (templates, topics,
    words, pool): the lists are bound by digest in the protocol,
    so a payload cannot quietly carry a different instrument
    under the same version number.
    """
    canonical = json.dumps(
        [FACT_TEMPLATE, QUESTION_TEMPLATE, list(TOPICS), list(WORDS), list(NEUTRAL_POOL)],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The frozen instrument description a payload is checked
    against: arms, case material, floors, budgets, and the
    seed/backend pair that generated the set.
    """
    return {
        "version": PROTOCOL_VERSION,
        "arm_keys": list(ARM_KEYS),
        "baseline_arm": BASELINE_ARM,
        "case_count": CASE_COUNT,
        "case_material_sha256": case_material_sha256(),
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


# ---------------------------------------------------------------------
# Aggregation (the validator recomputes this with its own mirror)
# ---------------------------------------------------------------------


def _cell(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the cell."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _verdict(record: dict[str, Any], key: str) -> bool:
    """Return the verdict."""
    cell = _cell(record, key)
    return bool(cell is not None and cell["verdict"])


def _pair_block_divergent(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Discordant counts over the headline population: both
    cells completed AND the arms' render id sequences diverged.
    Ids-identical records never enter this block; they are the
    instrument-inapplicable share, counted separately.
    """
    n_paired = 0
    n_nfc_only = 0
    n_nfd_only = 0
    n_both = 0
    n_pass_nfc = 0
    n_pass_nfd = 0
    for record in records:
        if record.get("ids_identical") is not False:
            continue
        cell_nfc = _cell(record, "nfc")
        cell_nfd = _cell(record, "nfd")
        if cell_nfc is None or cell_nfd is None:
            continue
        n_paired += 1
        verdict_nfc = bool(cell_nfc["verdict"])
        verdict_nfd = bool(cell_nfd["verdict"])
        if verdict_nfc:
            n_pass_nfc += 1
        if verdict_nfd:
            n_pass_nfd += 1
        if verdict_nfc and verdict_nfd:
            n_both += 1
        elif verdict_nfc:
            n_nfc_only += 1
        elif verdict_nfd:
            n_nfd_only += 1
    return {
        "n_paired": n_paired,
        "n_nfc_only": n_nfc_only,
        "n_nfd_only": n_nfd_only,
        "n_both": n_both,
        "n_neither": n_paired - n_nfc_only - n_nfd_only - n_both,
        "rate_nfc": (n_pass_nfc / n_paired) if n_paired else None,
        "rate_nfd": (n_pass_nfd / n_paired) if n_paired else None,
        "mcnemar_p": mcnemar_exact_p(n_nfc_only, n_nfd_only),
    }


def classify_canonical_form(
    n_paired: int,
    nfc_only: int,
    nfd_only: int,
) -> str:
    """The headline classification (total function).

    Decision list, first match wins (docs/CANONICALFORM_DESIGN.md).
    The floors gate this label only. A single discordant case at
    a sufficient population is ``below_threshold``: observed, but
    below the floor -- NOT falsified. That is the transplant
    that keeps one noisy case from closing the question.
    """
    d = nfc_only + nfd_only
    if n_paired < MIN_PAIRED_EVALUABLE:
        return "insufficient"
    if d == 0:
        return "no_discordance_observed"
    if nfc_only == nfd_only and nfc_only >= MIN_DISCORDANT:
        return "discordant_balanced"
    if max(nfc_only, nfd_only) >= MIN_DISCORDANT:
        return "discordant_asymmetric"
    return "below_threshold"


def invariance_falsified(classification: str) -> bool:
    """The pre-registered falsification boolean, direction
    stated explicitly: True means the INVARIANCE claim fell
    (discordance at or above the floor), not that an effect was
    confirmed absent.
    """
    return classification in ("discordant_balanced", "discordant_asymmetric")


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def aggregate_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-candidate aggregates over its ground records.

    Every count carries its denominator; a rate is None only
    when its denominator is 0. The id-divergence magnitudes are
    reported per case (in case order) so a reader can separate
    "diverged by 1 id" from "diverged into byte ids" at a glance.
    """
    arms: dict[str, Any] = {}
    for key in ARM_KEYS:
        n_completed = 0
        n_pass = 0
        for record in records:
            cell = _cell(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": (n_pass / n_completed) if n_completed else None,
        }
    n_ids_identical = 0
    n_ids_divergent = 0
    deltas: list[int] = []
    for record in records:
        if record.get("ids_identical") is True:
            n_ids_identical += 1
        elif record.get("ids_identical") is False:
            n_ids_divergent += 1
        delta = record.get("id_length_delta")
        deltas.append(delta if type(delta) is int else 0)
    n_same_ids_flips = 0
    for record in records:
        if record.get("ids_identical") is not True:
            continue
        cell_nfc = _cell(record, "nfc")
        cell_nfd = _cell(record, "nfd")
        if cell_nfc is None or cell_nfd is None:
            continue
        if bool(cell_nfc["verdict"]) != bool(cell_nfd["verdict"]):
            n_same_ids_flips += 1
    headline_raw = _pair_block_divergent(records)
    classification = classify_canonical_form(
        headline_raw["n_paired"], headline_raw["n_nfc_only"], headline_raw["n_nfd_only"]
    )
    headline = {
        "n_paired": headline_raw["n_paired"],
        "n_nfc_only": headline_raw["n_nfc_only"],
        "n_nfd_only": headline_raw["n_nfd_only"],
        "n_both": headline_raw["n_both"],
        "n_neither": headline_raw["n_neither"],
        "rate_nfc": headline_raw["rate_nfc"],
        "rate_nfd": headline_raw["rate_nfd"],
        "mcnemar_p": headline_raw["mcnemar_p"],
        "classification": classification,
    }
    return {
        "arms": arms,
        "ids": {
            "n_ids_identical": n_ids_identical,
            "n_ids_divergent": n_ids_divergent,
            "n_same_ids_verdict_flips": n_same_ids_flips,
            "id_length_delta_by_case": deltas,
            "max_abs_id_delta": max((abs(d) for d in deltas), default=0),
        },
        "headline": headline,
        "invariance_falsified": invariance_falsified(classification),
        "n_cases": len(records),
        "n_incomplete": sum(
            1 for record in records if _cell(record, "nfc") is None or _cell(record, "nfd") is None
        ),
    }


def shared_fixtures(
    records_by_candidate: dict[str, list[dict[str, Any]]],
) -> frozenset[str]:
    """Fixture ids recorded for every candidate: the conditions
    population. There is no closed-book screen in this probe, so
    unlike placement the shared set is not screened by union.
    """
    populations: list[frozenset[str]] = []
    for records in records_by_candidate.values():
        populations.append(frozenset(str(r.get("fixture")) for r in records))
    if not populations:
        return frozenset()
    return frozenset.intersection(*populations)


def _selection_over(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    key: str,
    fixtures: frozenset[str],
) -> dict[str, Any] | None:
    """The conservative selection for one arm restricted to
    ``fixtures``: None when any candidate's completed set on
    that restriction is empty or the restricted completed sets
    differ across candidates.
    """
    completed_sets: list[frozenset[str]] = []
    all_pass: dict[str, bool | None] = {}
    for name, records in records_by_candidate.items():
        completed = frozenset(
            str(record["fixture"])
            for record in records
            if record["fixture"] in fixtures and _cell(record, key) is not None
        )
        completed_sets.append(completed)
        if not completed:
            return None
        all_pass[name] = all(
            _verdict(record, key) for record in records if record["fixture"] in completed
        )
    if len(set(completed_sets)) > 1:
        return None
    passers = [name for name, ok in all_pass.items() if ok]
    if len(passers) == 1:
        return {"decision": "selected", "winner": passers[0]}
    if len(passers) > 1:
        return {"decision": "ambiguous", "winner": None}
    return {"decision": "no_candidate", "winner": None}


def arm_selection(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    arm: str,
) -> dict[str, Any]:
    """Per-arm selection readout over the shared population,
    with the flip against the NFC baseline recomputed on the
    same restricted completed set (comparing against the
    full-set baseline would mix set differences into the flip).
    Withheld when the restricted completed sets differ across
    candidates or are empty; the baseline arm's own flip is
    False when not withheld, and a flip is None when either
    side is withheld.
    """
    shared = shared_fixtures(records_by_candidate)
    own = _selection_over(records_by_candidate, arm, shared)
    if own is None:
        return {
            "decision": "unavailable",
            "detail": "selection_withheld_mismatched_sets",
            "winner": None,
            "withheld": True,
            "flip": None,
        }
    if arm == BASELINE_ARM:
        flip: bool | None = False
    else:
        completed = frozenset(
            str(record["fixture"])
            for records in records_by_candidate.values()
            for record in records
            if record["fixture"] in shared and _cell(record, arm) is not None
        )
        baseline = _selection_over(records_by_candidate, BASELINE_ARM, completed)
        flip = None if baseline is None else own != baseline
    return {
        "decision": own["decision"],
        "detail": "",
        "winner": own["winner"],
        "withheld": False,
        "flip": flip,
    }


def winner_headline(conditions: dict[str, Any]) -> dict[str, Any]:
    """The winner half of the claim as a total function of the
    two arms' selections. Selection-vs-abstention is NEVER a
    winner change: exactly one side selecting vacates the winner
    half to ``winner_abstention_asymmetric``. ``winner_changed``
    is True only for a genuine winner change, False for a
    decided non-change, and None when either selection was
    withheld (the comparison is then undefined, not stable).
    """
    nfc = conditions.get("nfc") if isinstance(conditions, dict) else None
    nfd = conditions.get("nfd") if isinstance(conditions, dict) else None
    if not isinstance(nfc, dict) or not isinstance(nfd, dict):
        return {
            "headline": "winner_undefined",
            "detail": "selection_missing",
            "winner_changed": None,
        }
    if nfc.get("withheld") is True or nfd.get("withheld") is True:
        return {
            "headline": "winner_undefined",
            "detail": "selection_withheld_mismatched_sets",
            "winner_changed": None,
        }
    nfc_selected = nfc.get("decision") == "selected"
    nfd_selected = nfd.get("decision") == "selected"
    if nfc_selected and nfd_selected:
        if nfc.get("winner") != nfd.get("winner"):
            return {"headline": "winner_changed", "detail": "", "winner_changed": True}
        return {"headline": "winner_stable", "detail": "", "winner_changed": False}
    if nfc_selected != nfd_selected:
        return {
            "headline": "winner_abstention_asymmetric",
            "detail": "",
            "winner_changed": False,
        }
    return {"headline": "winner_undefined", "detail": "", "winner_changed": False}
