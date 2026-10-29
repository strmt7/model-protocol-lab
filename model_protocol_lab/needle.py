"""Needle-depth probe (cycle 191, research wave-8 rank 3).

Every other filler probe in the lab varies how much irrelevant
text surrounds a fixture's own question. This one varies where
an answer-bearing fact sits inside a fixed amount of filler:
one generated fact sentence carrying a unique synthetic code
is planted at five depths of a neutral bulk at two doses, and
the model is asked for the code. The independent variable is
the chat template -- same checkpoint, same generated cases,
same filler, only the hand-rendered format changes -- so the
readout is a recall-vs-depth curve per template family, with
the depth offset verified on each candidate's own rendered
token sequence.

The measurement settlements were earned by two red teams
before any code (docs/NEEDLE_DESIGN.md carries the full
adjudication): dose is a rendered-token delta over the case's
own dose-0 render, never a property of the filler text alone;
the offset is the prefix-anchored LCP difference ``D``, which
cancels the template head's merge behaviour present in both
renders (a marker-derived head anchor was executed failing
outright under merging tokenizers); both equalities are
verified jointly on the accepted construction's final ids
before the cell is generated. Cases are generated from the
run seed by a frozen SHA-256 derivation the diff validator
mirrors exactly; records carry digests only, and the cell
verdict is bound by ``verdict == (answer_digest ==
sha256(expected)) and boundary_found``.

The depth phenomenon is published prior art (Lost in the
Middle, NoLiMa, RULER, BABILong -- see docs/RESEARCH_LOG.md
cycle 191); the deficit readout below is Lost in the Middle's
edge-mean-minus-middle functional form, credited there and in
the design doc, not renamed. The filler is topically
dissimilar prose, so the curves are a lower bound on position
effects under realistic retrieval bulk, and each cell is one
greedy generation: the curve is a five-point sample of
position with printed denominators, not a smooth function.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .distraction import NEUTRAL_CORPUS

__all__ = [
    "CASE_COUNT",
    "CELL_DETAILS",
    "CELL_KEYS",
    "DEPTHS",
    "DOSES",
    "FACT_TEMPLATE",
    "GENERATION_BUDGET",
    "GROUP",
    "MAX_GROUND_RECORDS",
    "MAX_RECORDS_BYTES",
    "MIN_HEADLINE_POPULATION",
    "PROTOCOL_VERSION",
    "QUESTION_TEMPLATE",
    "SCREENING_RULE_TEXT",
    "SIZING_ENCODE_BUDGET",
    "SIZING_RULE_TEXT",
    "STAGE_DETAILS",
    "STREAM_CYCLES",
    "TOPICS",
    "WORDS",
    "aggregate_candidate",
    "canonical_generated_rows",
    "case_material_sha256",
    "cell_key_absent",
    "cell_key_duplicate",
    "cell_key_treated",
    "cell_selection",
    "digest_text",
    "dose0_control",
    "duplicate_readout",
    "false_recall_readout",
    "filler_screened",
    "filler_stream",
    "generate_cases",
    "ground_sha256",
    "headline_population",
    "headline_readouts",
    "lcp_len",
    "offset_measurement",
    "protocol_fingerprint",
    "protocol_object",
    "screening_rule_sha256",
    "sizing_rule_sha256",
    "target_offset",
]

PROTOCOL_VERSION = 1
GROUP = "needle"

#: Filler doses in rendered tokens (the brief's pair; the 4096
#: dose needs --max-context above the default 2048 cap, stated
#: in the CLI help before the run).
DOSES: tuple[int, ...] = (1024, 4096)
#: Needle depths as percent of the dose span. Both doses make
#: every target offset an exact integer.
DEPTHS: tuple[int, ...] = (0, 25, 50, 75, 100)
CASE_COUNT = 20

FACT_TEMPLATE = "The {topic} code is {value}. "
QUESTION_TEMPLATE = "What is the {topic} code?"

#: Case topics, frozen. Every topic is absent from the neutral
#: corpus as a case-folded substring (a test pins this), so the
#: question's topic phrase occurs in a prompt only via the
#: needle and the question itself.
TOPICS: tuple[str, ...] = (
    "astronomy society",
    "beekeeping circle",
    "chess club",
    "rowing team",
    "radio club",
    "pottery class",
    "fencing team",
    "archive office",
    "textile guild",
    "mountain rescue",
    "vineyard crew",
    "observatory staff",
    "printing workshop",
    "cycling club",
    "dance academy",
    "karate school",
    "model railway club",
    "stamp society",
    "hiking group",
    "choir society",
)

#: Code words, frozen. Screened against the corpus the same
#: way (four first-draft words collided with corpus sentences
#: and were replaced before the freeze: quench, lantern,
#: timber, hazel).
WORDS: tuple[str, ...] = (
    "ochre",
    "velvet",
    "cobalt",
    "harbor",
    "saffron",
    "tundra",
    "zephyr",
    "marble",
    "onyx",
    "pepper",
    "quartz",
    "riddle",
    "saddle",
    "temple",
    "umbrella",
    "violet",
    "walnut",
    "yonder",
    "zipper",
    "anchor",
    "beacon",
    "cinder",
    "dagger",
    "ember",
    "falcon",
    "garnet",
    "helmet",
    "inkwell",
    "jubilee",
    "kestrel",
    "basalt",
    "magnet",
    "nectar",
    "obelisk",
    "pumice",
    "quiver",
    "rampart",
    "scepter",
    "sienna",
    "utensil",
    "vortex",
    "willow",
    "yonks",
    "zenith",
    "almond",
    "birch",
    "copper",
    "drizzle",
    "elmwood",
    "fjord",
    "glacier",
    "mauve",
    "ivory",
    "juniper",
    "kraken",
    "linden",
    "meteor",
    "nickel",
    "opal",
    "pyrite",
    "quill",
    "russet",
    "sorrel",
    "topaz",
)


def cell_key_treated(dose: int, depth: int) -> str:
    """The payload key for one treated (dose, depth) cell."""
    return f"dose_{dose}_depth_{depth}"


def cell_key_absent(dose: int) -> str:
    """The payload key for one needle-absent cell."""
    return f"absent_{dose}"


def cell_key_duplicate(dose: int) -> str:
    """The payload key for one duplicate-needle cell."""
    return f"duplicate_{dose}"


CELL_KEYS: tuple[str, ...] = (
    ("dose_0",)
    + tuple(cell_key_treated(dose, depth) for dose in DOSES for depth in DEPTHS)
    + tuple(cell_key_absent(dose) for dose in DOSES)
    + tuple(cell_key_duplicate(dose) for dose in DOSES)
)

#: The per-cell detail vocabulary. A cell is present exactly
#: when its detail is "".
CELL_DETAILS: tuple[str, ...] = (
    "arm_context_exceeded",
    "arm_evaluation_failed",
    "arm_screened",
    "arm_unmatchable",
)
#: The stage-level detail vocabulary.
STAGE_DETAILS: tuple[str, ...] = (
    "cancelled",
    "needle_record_budget_exceeded",
    "needle_screen_failed",
)

#: The sizing rule, frozen as text: the protocol carries the
#: SHA-256 of this string, so a bare version number never has
#: to stand in for the rule itself (the multiturn
#: template_digest settlement). The stage implements exactly
#: this text; the diff validator checks offsets and deltas,
#: not the search, so the rule binds the stage's honesty, not
#: the payload's shape.
SIZING_RULE_TEXT = (
    "needle-phased-exact-v1: spans are whole sentences plus one "
    "truncated suffix (span_text) over the cycled neutral corpus "
    "stream, consumed in stream order; a truncated sentence "
    "counts as consumed and the next span continues after it. "
    "Treated cells size in two phases: prefix candidates in "
    "span_candidates closeness order, probed as (prefix + needle "
    "+ question) against the comparator (prefix + question), "
    "accepting the first whose D equals the target offset (the "
    "empty prefix at depth 0, no search); then suffix candidates "
    "continuing the stream, probed as the full arm render, "
    "accepting the first whose rendered delta over the dose-0 "
    "render equals the dose with D re-verified on the same ids "
    "(the empty suffix at depth 100, verified on a single "
    "render). Duplicate cells insert a middle phase between "
    "them: mid candidates continuing the stream, probed as "
    "(prefix + needle + mid + needle + question) against "
    "(prefix + needle + mid + question), accepting the first "
    "whose D equals the second target offset; the suffix phase "
    "then targets the duplicate delta (dose + needle_len) and "
    "re-verifies both offsets on the final ids. Absent cells "
    "size a single span against the question-only render. The "
    "encode budget counts every probe encode in the phases "
    "(two per prefix or middle probe, one per suffix or absent "
    "probe); the cached question-only, dose-0, and accepted-"
    "comparator renders are infrastructure and the joint final "
    "verification reuses the accepted renders, adding none."
)
SIZING_ENCODE_BUDGET = 512

#: The screening rule, frozen as text on the same terms.
SCREENING_RULE_TEXT = (
    "needle-screen-v1: per constructed filler span, case-folded: "
    "the span is screened when it contains the case value, or "
    "when the candidate's decoded boundary text occurs in it. "
    "The needle sentence itself is the plant and is never "
    "screened. Stage-level, the frozen texts (fact template "
    "with slots emptied, topics, words, neutral corpus) are "
    "audited for special-token literals against every candidate "
    "format, and the generated texts (fact template with slots "
    "emptied, topics, words) are checked for containment of "
    "any candidate's decoded boundary text; a hit fails the "
    "whole stage closed before any generation."
)

#: The filler stream is the neutral corpus cycled this many
#: times. One pass is ~1,278 tokens under a typical tokenizer,
#: so a 4096-token bulk wraps the stream about 3.2 times; the
#: stream must be long enough that the largest construction
#: (prefix + needle + suffix at dose 4096) never exhausts it.
STREAM_CYCLES = 6

#: The deficit and first-failure readouts are withheld below
#: this headline population (a five-point curve over fewer
#: cases is anecdote with a formula attached).
MIN_HEADLINE_POPULATION = 6

#: 24 candidates (run_experiment's cap) x 20 cases; fifteen
#: cells per case. Generations and records carry one constant
#: each here because they coincide: one record per
#: (candidate, case), fifteen generations per record.
MAX_GROUND_RECORDS = 24 * CASE_COUNT
GENERATION_BUDGET = len(CELL_KEYS) * MAX_GROUND_RECORDS
#: Records carry fifteen cells with answer digests; a maximal
#: record measures ~2.6 KB, so a maximal payload's records
#: sections stay under ~1.3 MB. The bound is the house 12 MiB
#: for uniformity with the sibling probes.
MAX_RECORDS_BYTES = 12 * 1024 * 1024


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text (the records' digest form)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _derive(seed: int, index: int, slot: int, counter: int) -> int:
    """One deterministic draw. The material string, the slot
    meanings (0 = code word, 1 = number), and the redraw rule
    in :func:`generate_cases` are frozen together -- the diff
    validator's mirror reproduces them exactly.
    """
    material = f"{seed}|needle|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def generate_cases(seed: int) -> list[dict[str, Any]]:
    """The frozen generated case set for a run seed, in
    canonical index order. Pure and deterministic -- the same
    seed regenerates the identical set across runs and
    PYTHONHASHSEED values. A value colliding with an earlier
    case's value redraws both slots at the next counter, so the
    twenty values are pairwise distinct by construction.
    """
    if type(seed) is not int:
        raise ValueError("needle seed must be an int")
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
        cases.append(
            {
                "fixture_id": f"needle-{index:02d}",
                "topic": topic,
                "value": value,
                "fact": FACT_TEMPLATE.format(topic=topic, value=value),
                "question": QUESTION_TEMPLATE.format(topic=topic),
                "expected": value,
            }
        )
    return cases


def canonical_generated_rows(seed: int) -> list[list[Any]]:
    """The canonical generated set the ground digest binds."""
    return [[case["fixture_id"], case["topic"], case["value"]] for case in generate_cases(seed)]


def ground_sha256(seed: int) -> str:
    """Digest of the canonical generated rows for a seed: two
    runs share an instrument only when this matches, because
    the generated set is the treatment.
    """
    canonical = json.dumps(canonical_generated_rows(seed), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def case_material_sha256() -> str:
    """Digest of the frozen case material (fact template,
    topics, words): the lists are bound by digest in the
    protocol, so a payload cannot quietly carry a different
    instrument under the same version number.
    """
    canonical = json.dumps(
        [FACT_TEMPLATE, QUESTION_TEMPLATE, list(TOPICS), list(WORDS)],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sizing_rule_sha256() -> str:
    """Digest of the frozen sizing-rule text."""
    return digest_text(SIZING_RULE_TEXT)


def screening_rule_sha256() -> str:
    """Digest of the frozen screening-rule text."""
    return digest_text(SCREENING_RULE_TEXT)


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The frozen instrument description a payload is checked
    against: doses, depths, case material, rule digests,
    floors, budgets, and the seed/backend pair.
    """
    return {
        "version": PROTOCOL_VERSION,
        "doses": list(DOSES),
        "depths": list(DEPTHS),
        "case_count": CASE_COUNT,
        "cell_keys": list(CELL_KEYS),
        "case_material_sha256": case_material_sha256(),
        "ground_sha256": ground_sha256(seed),
        "sizing_rule_sha256": sizing_rule_sha256(),
        "sizing_encode_budget": SIZING_ENCODE_BUDGET,
        "screening_rule_sha256": screening_rule_sha256(),
        "stream_cycles": STREAM_CYCLES,
        "min_headline_population": MIN_HEADLINE_POPULATION,
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


def filler_stream() -> tuple[str, ...]:
    """The filler sentence stream: the neutral corpus cycled
    ``STREAM_CYCLES`` times, in order.
    """
    return tuple(
        NEUTRAL_CORPUS[i % len(NEUTRAL_CORPUS)] for i in range(STREAM_CYCLES * len(NEUTRAL_CORPUS))
    )


def filler_screened(span: str, value: str, boundary_text: str) -> bool:
    """Per-span screen (screening rule needle-screen-v1).

    True when the case value could be copied from this span
    (case-folded containment) or when the candidate's decoded
    boundary text occurs in it (the filler would carry the
    terminator). The value carries digits and the corpus
    carries none, so the value screen is a guard against a
    future corpus edit, not a live filter -- stated so nobody
    mistakes it for one.
    """
    if value and value.casefold() in span.casefold():
        return True
    return bool(boundary_text) and boundary_text in span


def target_offset(dose: int, depth: int) -> int:
    """The frozen target offset for a (dose, depth) cell."""
    return dose * depth // 100


def lcp_len(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Length of the longest common prefix of two id tuples."""
    n = 0
    limit = min(len(a), len(b))
    while n < limit and a[n] == b[n]:
        n += 1
    return n


def offset_measurement(
    arm_ids: tuple[int, ...],
    comparator_ids: tuple[int, ...],
    dose0_ids: tuple[int, ...],
    question_ids: tuple[int, ...],
) -> int:
    """The offset anchor ``D``: LCP(arm, comparator) minus
    LCP(dose-0, question-only).

    The comparator is the arm's own text up to just before the
    measured needle, with the question appended; the dose-0
    render is the anchor because its needle sits at offset 0
    by construction. A template head that merges with the
    following token moves both LCPs together, so the merge
    cancels in the difference -- the property the executed
    red-team evidence selected this anchor for.
    """
    return lcp_len(arm_ids, comparator_ids) - lcp_len(dose0_ids, question_ids)


# ---------------------------------------------------------------------
# Per-candidate aggregation (the validator recomputes this)
# ---------------------------------------------------------------------


def _cell(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the cell."""
    cells = record.get("cells")
    if type(cells) is not dict:
        return None
    cell = cells.get(key)
    return cell if type(cell) is dict else None


def _verdict(record: dict[str, Any], key: str) -> bool:
    """Return the verdict."""
    cell = _cell(record, key)
    return bool(cell["verdict"]) if cell is not None else False


def dose0_control(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The candidate's own control level: dose-0 recorded and
    passed counts over the generated cases.
    """
    n_recorded = 0
    n_pass = 0
    for record in records:
        cell = _cell(record, "dose_0")
        if cell is None:
            continue
        n_recorded += 1
        if cell["verdict"]:
            n_pass += 1
    return {
        "n_recorded": n_recorded,
        "n_pass": n_pass,
        "rate": (n_pass / n_recorded) if n_recorded else None,
    }


def false_recall_readout(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-dose false-recall floor from the absent cells: a
    pass with no needle planted is fabrication or guessing,
    and the rate is reported beside that dose's curve.
    """
    out: dict[str, Any] = {}
    for dose in DOSES:
        key = cell_key_absent(dose)
        n_evaluable = 0
        n_false_pass = 0
        for record in records:
            cell = _cell(record, key)
            if cell is None:
                continue
            n_evaluable += 1
            if cell["verdict"]:
                n_false_pass += 1
        out[str(dose)] = {
            "n_evaluable": n_evaluable,
            "n_false_pass": n_false_pass,
            "rate": (n_false_pass / n_evaluable) if n_evaluable else None,
        }
    return out


def duplicate_readout(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-dose duplicate-needle readout over the candidate's
    own population (dose-0 pass with the duplicate and the 25 /
    50 / 75% cells at that dose recorded).

    Two needles could help merely by giving the model two
    independent chances, so the raw pass rate is printed beside
    the independence baseline computed from the same
    population's single-needle rates at the duplicate's depths
    -- a descriptive contrast under a stated modelling
    assumption, not a test. The rescue counts (mid-depth fail,
    duplicate pass) are the within-candidate contrast the arm
    exists for.
    """
    out: dict[str, Any] = {}
    for dose in DOSES:
        population = [
            record
            for record in records
            if _verdict(record, "dose_0")
            and _cell(record, cell_key_duplicate(dose)) is not None
            and _cell(record, cell_key_treated(dose, 25)) is not None
            and _cell(record, cell_key_treated(dose, 50)) is not None
            and _cell(record, cell_key_treated(dose, 75)) is not None
        ]
        n_population = len(population)
        if not n_population:
            out[str(dose)] = {
                "n_population": 0,
                "n_pass": 0,  # nosec B105 -- verdict count key, not a credential
                "pass_rate": None,  # nosec B105 -- verdict rate key, not a credential
                "rate_25": None,
                "rate_75": None,
                "independence_expected": None,
                "excess_over_independence": None,
                "n_mid_fail": 0,
                "n_rescued": 0,
                "rescue_rate": None,
            }
            continue
        n_pass = sum(1 for r in population if _verdict(r, cell_key_duplicate(dose)))
        rate_25 = (
            sum(1 for r in population if _verdict(r, cell_key_treated(dose, 25))) / n_population
        )
        rate_75 = (
            sum(1 for r in population if _verdict(r, cell_key_treated(dose, 75))) / n_population
        )
        expected = 1.0 - (1.0 - rate_25) * (1.0 - rate_75)
        pass_rate = n_pass / n_population
        mid_fails = [r for r in population if not _verdict(r, cell_key_treated(dose, 50))]
        n_rescued = sum(1 for r in mid_fails if _verdict(r, cell_key_duplicate(dose)))
        out[str(dose)] = {
            "n_population": n_population,
            "n_pass": n_pass,
            "pass_rate": pass_rate,
            "rate_25": rate_25,
            "rate_75": rate_75,
            "independence_expected": expected,
            "excess_over_independence": pass_rate - expected,
            "n_mid_fail": len(mid_fails),
            "n_rescued": n_rescued,
            "rescue_rate": (n_rescued / len(mid_fails)) if mid_fails else None,
        }
    return out


def aggregate_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-candidate readouts recomputed from the records
    alone: the analysis-side definition. The diff validator
    carries its own mirror arithmetic and rejects on any
    disagreement (the cycle-187 settlement).
    """
    return {
        "dose0_control": dose0_control(records),
        "false_recall": false_recall_readout(records),
        "duplicate": duplicate_readout(records),
    }


# ---------------------------------------------------------------------
# Cross-candidate readouts: headline population, curves, selection
# ---------------------------------------------------------------------


def headline_population(
    records_by_candidate: dict[str, list[dict[str, Any]]], dose: int
) -> list[str]:
    """The headline population for one dose, in canonical case
    order: cases where every candidate passed dose-0 with all
    five depth cells at this dose recorded.

    One shared population per dose keeps the curves, the
    deficit, and first-failure like-for-like across depths and
    candidates; per-candidate populations were rejected on
    executed red-team evidence (a flat curve could coexist
    with a large deficit over a differently-conditioned
    subset).
    """
    keys = [cell_key_treated(dose, depth) for depth in DEPTHS]
    names = list(records_by_candidate)
    if not names:
        return []
    by_fixture = {
        name: {record["fixture"]: record for record in records}
        for name, records in records_by_candidate.items()
    }
    population: list[str] = []
    for record in records_by_candidate[names[0]]:
        fixture = record["fixture"]
        ok = True
        for name in names:
            match = by_fixture[name].get(fixture)
            if match is None or not _verdict(match, "dose_0"):
                ok = False
                break
            if any(_cell(match, key) is None for key in keys):
                ok = False
                break
        if ok:
            population.append(fixture)
    return population


def headline_readouts(
    records_by_candidate: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Curves, mid-depth deficit, and first-failure per dose
    over the headline population.

    The deficit is Lost in the Middle's functional form --
    (rate at 0% + rate at 100%) / 2 - rate at 50% -- computed
    on the same population as the curves, so a flat curve
    makes it exactly zero. Deficit and first-failure are
    withheld below MIN_HEADLINE_POPULATION; the curves print
    at any population because they are counts with their
    denominator attached.
    """
    out: dict[str, Any] = {}
    by_fixture: dict[str, dict[str, dict[str, Any]]] = {
        name: {record["fixture"]: record for record in records}
        for name, records in records_by_candidate.items()
    }
    for dose in DOSES:
        population = headline_population(records_by_candidate, dose)
        n_population = len(population)
        curves: dict[str, Any] = {}
        deficit: dict[str, Any] = {}
        first_failure: dict[str, Any] = {}
        for name in records_by_candidate:
            points: dict[str, Any] = {}
            rates: dict[int, float | None] = {}
            for depth in DEPTHS:
                key = cell_key_treated(dose, depth)
                n_pass = sum(
                    1 for fixture in population if _verdict(by_fixture[name][fixture], key)
                )
                rate = (n_pass / n_population) if n_population else None
                rates[depth] = rate
                points[str(depth)] = {"n_pass": n_pass, "rate": rate}
            curves[name] = points
            rate_0 = rates[0]
            rate_50 = rates[50]
            rate_100 = rates[100]
            if (
                n_population < MIN_HEADLINE_POPULATION
                or rate_0 is None
                or rate_50 is None
                or rate_100 is None
            ):
                deficit[name] = None
                first_failure[name] = None
                continue
            deficit[name] = (rate_0 + rate_100) / 2 - rate_50
            counts = {str(depth): 0 for depth in DEPTHS}
            n_no_failure = 0
            for fixture in population:
                record = by_fixture[name][fixture]
                first: int | None = None
                for depth in DEPTHS:
                    if not _verdict(record, cell_key_treated(dose, depth)):
                        first = depth
                        break
                if first is None:
                    n_no_failure += 1
                else:
                    counts[str(first)] += 1
            first_failure[name] = {"counts": counts, "n_no_failure": n_no_failure}
        out[str(dose)] = {
            "population": population,
            "n_population": n_population,
            "curves": curves,
            "deficit": deficit,
            "first_failure": first_failure,
        }
    return out


def _selection_over(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    key: str,
    fixtures: frozenset[str],
) -> dict[str, Any] | None:
    """The conservative selection for one cell key restricted
    to ``fixtures``: None when any candidate's completed set
    on that restriction is empty or the restricted completed
    sets differ across candidates.
    """
    completed_sets: list[frozenset[str]] = []
    all_pass: dict[str, bool | None] = {}
    for name, records in records_by_candidate.items():
        completed = frozenset(
            record["fixture"]
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


def cell_selection(
    records_by_candidate: dict[str, list[dict[str, Any]]], key: str
) -> dict[str, Any]:
    """Per-cell selection readout with the paired-restricted
    flip baseline.

    The selection re-applies the conservative rule over the
    cell's completed case set and is withheld when a completed
    set is empty or the sets differ across candidates. The
    flip compares the outcome to the dose-0 selection
    recomputed on the same completed set -- comparing against
    the full-set dose-0 outcome would mix set differences into
    the flip -- and is None when either side is withheld. The
    dose-0 entry itself reports its own selection with flip
    False when not withheld.
    """
    all_fixtures = frozenset(
        record["fixture"] for records in records_by_candidate.values() for record in records
    )
    own = _selection_over(records_by_candidate, key, all_fixtures)
    if own is None:
        return {
            "decision": "unavailable",
            "detail": "selection_withheld_mismatched_sets",
            "winner": None,
            "withheld": True,
            "flip": None,
        }
    if key == "dose_0":
        flip: bool | None = False
    else:
        # The cell's completed set is common to all candidates
        # here (the selection above was not withheld), so the
        # union over candidates is that shared set.
        completed = frozenset(
            record["fixture"]
            for records in records_by_candidate.values()
            for record in records
            if _cell(record, key) is not None
        )
        baseline = _selection_over(records_by_candidate, "dose_0", completed)
        flip = None if baseline is None else own != baseline
    return {
        "decision": own["decision"],
        "detail": "",
        "winner": own["winner"],
        "withheld": False,
        "flip": flip,
    }
