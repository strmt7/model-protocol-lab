"""Prompt distractor dilution: pure protocol and arithmetic.

Cycle 184 (wave-7 rank 5). The stage in
:mod:`model_protocol_lab.experiments` inserts deterministic
irrelevant filler into the last user message at exact
rendered-token doses and re-measures verdicts per condition.
This module owns everything that must be frozen for the numbers
to mean anything: the condition set, the neutral corpus, the
near-miss sentence templates, the planted-answer rule, the span
sizing search order, the span screen, the construction
classification, and the per-candidate aggregation the diff
validator independently recomputes.

Design settlements earned by the two red teams (see
docs/REPETITION_DESIGN.md's sibling, docs/DISTRACTION_DESIGN.md):
the plant is produced by ASCII digit-string surgery, never
Decimal arithmetic (Decimal is non-total on loader-legal
expecteds, context-dependent past 28 digits, and script-lax);
near-miss filler is a frozen sentence template with one plant
slot, never a bare repeated answer string; capture is an upper
bound, shipped beside a neutral-control excess readout; and
the factorial is cut to 10 treated conditions — near-miss at
1024 tokens would measure repetition pressure on the plant,
not dilution, so near-miss curves terminate at 256 and say so.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

from .core import Fixture, Format
from .repetition import target_index

__all__ = [
    "CONDITIONS",
    "CONDITION_KEYS",
    "GENERATION_BUDGET",
    "JOINER",
    "MAX_GROUND_RECORDS",
    "MAX_RECORDS_BYTES",
    "NEARMISS_LEVELS",
    "NEARMISS_TEMPLATES",
    "NEARMISS_TEMPLATES_SHA256",
    "NEUTRAL_CORPUS",
    "NEUTRAL_CORPUS_SHA256",
    "NEUTRAL_LEVELS",
    "PLACEMENTS",
    "PROTOCOL_VERSION",
    "SCREENING_RULE_VERSION",
    "SIZING_ENCODE_BUDGET",
    "SIZING_RULE",
    "aggregate_candidate",
    "build_cell_fixture",
    "cell_condition",
    "classify_cell_construction",
    "curves_for",
    "nearmiss_sentences",
    "planted_answer",
    "planted_occurrences",
    "protocol_fingerprint",
    "protocol_object",
    "span_candidates",
    "span_screened",
    "span_text",
    "suffix_cuts_plant",
]

PROTOCOL_VERSION = 1
JOINER = "\n"
NEUTRAL_LEVELS: tuple[int, ...] = (64, 256, 1024)
NEARMISS_LEVELS: tuple[int, ...] = (64, 256)
PLACEMENTS: tuple[str, ...] = ("before", "after")
# The shipped condition set is a protocol fact (fingerprint
# covers it): neutral at three doses, near-miss at two, each in
# both placements — 10 treated conditions, 24,000 generations
# at the run caps. Near-miss at 1024 is cut deliberately: the
# plant would repeat dozens of times, and the cell would
# measure repetition pressure, not dilution.
CONDITIONS: tuple[tuple[str, int, str], ...] = tuple(
    ("neutral", level, placement) for level in NEUTRAL_LEVELS for placement in PLACEMENTS
) + tuple(("near-miss", level, placement) for level in NEARMISS_LEVELS for placement in PLACEMENTS)
MAX_GROUND_RECORDS = 2400
GENERATION_BUDGET = len(CONDITIONS) * MAX_GROUND_RECORDS
# Ground records carry ten cells with answer digests; a maximal
# synthetic record measures ~3.5 KB, so the records sections of
# a maximal payload stay under ~9 MiB. The validator rejects a
# records section over this bound before recomputing anything
# (the whole-file loader cap is 32 MiB and is shared with the
# native search section).
MAX_RECORDS_BYTES = 12 * 1024 * 1024
SIZING_RULE = "whole-sentences-plus-one-truncated-suffix-v1"
SIZING_ENCODE_BUDGET = 512
SCREENING_RULE_VERSION = 1
NEARMISS_RULE = "ascii-digit-surgery-v2"

_MARKER = "\x00mpldst\x00"


def cell_condition(kind: str, level: int, placement: str) -> str:
    """The payload key for one treated condition."""
    return f"{kind}_{level}_{placement}"


CONDITION_KEYS: tuple[str, ...] = tuple(
    cell_condition(kind, level, placement) for kind, level, placement in CONDITIONS
)


# ---------------------------------------------------------------------
# The planted answer: ASCII digit-string surgery (rule v2)
# ---------------------------------------------------------------------

# Admitted shapes: optional sign; a mantissa that is plain
# digits, grouped digits (1-3 then comma triplets), or a bare
# fraction (.5); an optional fraction; an optional scientific
# suffix whose exponent is never touched. [0-9] rather than \d:
# \d admits non-ASCII decimal digits, and the plant must never
# switch script.
_PLANT_SHAPE = re.compile(
    r"^(?P<sign>[+-]?)"
    r"(?P<mantissa>[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+|\.[0-9]+)"
    r"(?P<fraction>\.[0-9]+)?"
    r"(?P<exponent>[eE][+-]?[0-9]+)?$"
)


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def planted_answer(expected: str) -> str | None:
    """The planted wrong answer for ``expected``, or None.

    Rule v2: increment the last digit of the mantissa digit
    sequence by one, carry propagating left through digits
    only. Sign, grouping, fraction width, and the exponent are
    preserved by construction, so the plant is format-matched:
    ``"9,999" -> "10,000"``, ``"42.0" -> "42.1"``,
    ``"-0" -> "-1"`` (magnitude reading, pinned deliberately),
    ``"1e3" -> "2e3"``, ``".5" -> ".6"``. Anything outside the
    frozen shape — whitespace-padded, underscored, non-ASCII
    digits, dates, words — returns None (near-miss undefined),
    never an exception and never a guess.
    """
    if type(expected) is not str:
        return None
    match = _PLANT_SHAPE.match(expected)
    if match is None:
        return None
    mantissa = match.group("mantissa")
    fraction = match.group("fraction")
    if mantissa.startswith("."):
        if fraction is not None:
            return None  # ".5.5" is not a number shape we admit
        int_digits = ""
        frac_digits = mantissa[1:]
        grouped = False
    else:
        grouped = "," in mantissa
        int_digits = mantissa.replace(",", "")
        frac_digits = fraction[1:] if fraction is not None else ""
    # The admitted shapes all carry at least one mantissa digit,
    # so the sequence is never empty here.
    sequence = list(int_digits + frac_digits)
    carry = 1
    index = len(sequence) - 1
    while carry and index >= 0:
        digit = ord(sequence[index]) - 48 + carry
        carry = digit // 10
        sequence[index] = chr(48 + digit % 10)
        index -= 1
    int_len = len(int_digits)
    if carry:
        sequence.insert(0, "1")
        int_len += 1
    new_int = "".join(sequence[:int_len])
    new_frac = "".join(sequence[int_len:])
    if grouped and new_int:
        groups: list[str] = []
        rest = new_int
        while rest:
            groups.append(rest[-3:])
            rest = rest[:-3]
        new_int = ",".join(reversed(groups))
    had_dot = "." in mantissa or fraction is not None
    rendered = (new_int + ("." + new_frac if had_dot else "")) if new_int else ("." + new_frac)
    return match.group("sign") + rendered + (match.group("exponent") or "")


# ---------------------------------------------------------------------
# Frozen texts: the neutral corpus and the near-miss templates
# ---------------------------------------------------------------------

# The corpus is a sequence of sentences, no sentence repeated,
# consumed as nested prefixes: every level's span starts from
# the same first sentence, so levels compare doses of one text,
# not different texts. Sentences carry a trailing space so a
# span is a plain concatenation. No digits anywhere: a number
# in the neutral corpus would be an unplanted lure.
NEUTRAL_CORPUS: tuple[str, ...] = (
    "The harbour bell rang at six while the ferries were still tied along the eastern pier. ",
    "A light rain had polished the cobbles outside the customs house during the night. ",
    "The gardener trimmed the hedge along the lane and swept the cuttings into a barrow. ",
    "Two cyclists stopped at the fountain to refill their bottles before the climb. ",
    "The library van called at the village green every second Thursday of the month. ",
    "Bakers in the old town still prove their loaves in baskets lined with linen. ",
    "The tide tables were pinned beside the lifeboat station door for the winter season. ",
    "A kettle sang on the stove while Marta shelled peas into a chipped enamel bowl. ",
    "The orchard wall had lost a few stones where the ivy pushed between them. ",
    "Shepherds moved the flock down from the high pasture ahead of the first snow. ",
    "The clockmaker oiled the tower movement and set the hands by the railway clock. ",
    "Crates of apples waited under the awning, sorted by size and ripeness. ",
    "The ferry crew hosed the deck and coiled the ropes in neat flat spirals. ",
    "A heron stood motionless in the shallows below the mill weir at dusk. ",
    "The postman left the parcels under the porch roof to keep them out of the rain. ",
    "Children flew a red kite from the hill above the reservoir all afternoon. ",
    "The joiner planed the door until it swung without catching on the frame. ",
    "Swallows nested under the eaves of the barn and darted over the yard. ",
    "The soup wanted salt, a bay leaf, and another hour over a low flame. ",
    "Fishing boats came in with the tide and unloaded their catch on the quay. ",
    "The map in the station hall showed the branch line closing for repairs. ",
    "An old dog slept in the sun outside the bakery, undisturbed by the queue. ",
    "The choir rehearsed in the hall while the caretaker stacked the chairs. ",
    "Brass bands played on the bandstand through the warm part of the afternoon. ",
    "The lighthouse keeper recorded the wind direction in a leather bound log. ",
    "Hay carts crossed the bridge slowly, piled high and smelling of summer. ",
    "The tailor chalked the pattern onto the cloth before lifting the shears. ",
    "A narrow boat waited at the lock while the gates drained and settled. ",
    "The market square filled early with stalls of cheese, honey, and flowers. ",
    "Pigeons scattered from the cathedral steps whenever the bells began. ",
    "The blacksmith quenched the shoe and watched the steam curl upward. ",
    "Walkers followed the waymarked path along the cliff toward the next bay. ",
    "The miller checked the sacks of flour stacked against the cool north wall. ",
    "A violin maker varnished the back of an instrument by the window light. ",
    "The harbourmaster raised the flag that warned of a change in the weather. ",
    "Grandmother knitted by the fire and told stories about the winter fair. ",
    "The bees worked the lavender rows in the heat of the early afternoon. ",
    "Stone masons repaired the churchyard wall with lime mortar and patience. ",
    "The evening train crossed the viaduct as the valley lights came on. ",
    "A basket of mushrooms sat by the door, gathered from the pine woods. ",
    "The cooper bound the barrels with hoops heated over the forge fire. ",
    "Swifts screamed around the tower and vanished into the gathering dusk. ",
    "The allotment holders shared seed potatoes over the fence in spring. ",
    "A cat watched the sparrows from the top of the woodpile, perfectly still. ",
    "The ferryman mended his nets on the bench outside the boathouse. ",
    "Clouds built over the ridge and the air smelled of coming rain. ",
    "The schoolchildren planted beans in rows along the sunny side of the plot. ",
    "An accordion player entertained the queue outside the ticket office. ",
    "The weaver dyed the wool with madder and hung the skeins to dry. ",
    "Gulls followed the plough across the field behind the farmhouse. ",
    "The night watchman made his rounds with a lantern and a bunch of keys. ",
    "A hedgehog shuffled through the leaf litter at the bottom of the garden. ",
    "The potter centred the clay and raised the walls of a bowl in silence. ",
    "Ice formed on the puddles along the towpath during the clear night. ",
    "The shepherd whistled and the dog circled wide behind the flock. ",
    "Barges loaded with timber waited for the morning tide at the wharf. ",
    "The organist practised the voluntary while the verger swept the aisle. ",
    "A squirrel buried beechnuts under the hedge at the edge of the wood. ",
    "The laundry steamed on the line above the yard despite the cold wind. ",
    "Fishermen mended the lobster pots stacked along the harbour wall. ",
    "The lane narrowed between high banks crowned with hazel and bramble. ",
    "A kettle of broth simmered beside the bread oven all morning long. ",
    "The lighthouse beam swept the bay twice before the fog closed in. ",
    "Children gathered conkers beneath the chestnut trees in the park. ",
    "The farrier arrived with his apron, his hammer, and his travelling anvil. ",
    "Mist lay in the hollows of the meadow long after sunrise that day. ",
    "The bookbinder stitched the gatherings and pressed them under weights. ",
    "A robin followed the gardener, watching for worms in the turned soil. ",
    "The canal froze hard enough for skaters by the end of the cold week. ",
    "Sheep bells sounded faintly from the pasture across the river bend. ",
    "The watchmaker bent over the movement with his loupe screwed in place. ",
)

NEUTRAL_CORPUS_SHA256 = hashlib.sha256(
    json.dumps(list(NEUTRAL_CORPUS), ensure_ascii=True).encode("utf-8")
).hexdigest()

# Near-miss filler is a frozen template cycle: four sentences,
# each an irrelevant factual aside with exactly one plant slot.
# The bare planted string is never inserted alone — an
# off-by-one number inside a plausible sentence is the GSM-IC
# lure; a repeated bare answer string would be cycle 183's
# repetition treatment wearing this probe's label.
NEARMISS_TEMPLATES: tuple[str, ...] = (
    "The depot ledger for that week recorded {planted} crates in the "
    "third bay, and the entry was initialled by two clerks. ",
    "A notice pinned beside the timetable said the hall would seat "
    "{planted} people once the extra chairs arrived. ",
    "The ferry manifest listed {planted} sacks of grain loaded before "
    "dawn, none of them marked for the island run. ",
    "According to the workshop log, the kiln had fired {planted} tiles "
    "that month, a little above the usual count. ",
)

NEARMISS_TEMPLATES_SHA256 = hashlib.sha256(
    json.dumps(list(NEARMISS_TEMPLATES), ensure_ascii=True).encode("utf-8")
).hexdigest()

# How many template sentences the sizing sequence offers. The
# largest near-miss dose is 256 tokens; the cycle must be able
# to overshoot it under any plausible tokenizer, and per-run
# adequacy is decided by the sizing search itself (a sequence
# that cannot reach the dose yields arm_unmatchable cells).
NEARMISS_SEQUENCE_LENGTH = 24


def nearmiss_sentences(planted: str) -> tuple[str, ...]:
    """The frozen template cycle with ``planted`` substituted."""
    return tuple(
        NEARMISS_TEMPLATES[i % len(NEARMISS_TEMPLATES)].replace("{planted}", planted)
        for i in range(NEARMISS_SEQUENCE_LENGTH)
    )


def planted_occurrences(span: str, planted: str) -> int:
    """Occurrences of the plant in a constructed span (recount)."""
    if not planted:
        return 0
    return span.count(planted)


def span_screened(span: str, expected: str, boundary_texts: Sequence[str]) -> bool:
    """Per-span screen (screening rule v1).

    True when the fixture's expected answer could be copied
    from this span (either string contains the other,
    case-folded — the screen_filler predicate) or when a
    decoded boundary text occurs in the span (the treatment
    would carry the terminator). Screening is per constructed
    span, never per corpus: a collision word sitting at corpus
    position 900 does not contaminate a 64-token span.
    """
    folded = span.casefold()
    answer = expected.strip().casefold()
    if answer and (answer in folded or folded.strip() in answer):
        return True
    return any(text and text in span for text in boundary_texts)


def suffix_cuts_plant(suffix: str, rest: str, planted: str) -> bool:
    """True when ending a span at ``suffix`` splits a plant.

    ``suffix`` is the truncated tail of a template sentence and
    ``rest`` is what follows it in that sentence. The cut
    splits a planted occurrence iff the suffix ends with a
    proper prefix of the plant and the rest continues it.
    """
    if not planted or not rest:
        return False
    for k in range(1, len(planted)):
        if suffix.endswith(planted[:k]) and rest.startswith(planted[k:]):
            return True
    return False


# ---------------------------------------------------------------------
# Span construction and the sizing search order
# ---------------------------------------------------------------------


def span_text(sentences: Sequence[str], whole: int, suffix_chars: int) -> str:
    """The span of ``whole`` sentences plus a truncated suffix."""
    if whole < 0 or whole > len(sentences):
        raise ValueError("whole out of range")
    text = "".join(sentences[:whole])
    if suffix_chars:
        if whole >= len(sentences):
            raise ValueError("no sentence to truncate")
        sentence = sentences[whole]
        if suffix_chars < 0 or suffix_chars > len(sentence):
            raise ValueError("suffix out of range")
        text += sentence[:suffix_chars]
    return text


def span_candidates(
    sentences: Sequence[str], *, target: int, chars_per_token: float
) -> list[tuple[int, int, int]]:
    """Deterministic probe order for the sizing search.

    Returns ``(whole, suffix_chars, span_chars)`` triples ordered
    by closeness of the estimated dose to ``target`` (span
    characters scaled by the run's measured chars-per-token),
    ties broken by the triple itself, so every run probes in the
    same order. The stage probes at most SIZING_ENCODE_BUDGET of
    these per cell and takes the first whose *rendered* delta
    is exact — the estimate only orders the search, it never
    decides it.
    """
    triples: list[tuple[int, int, int]] = []
    prefix_chars = 0
    for whole in range(len(sentences) + 1):
        if whole:
            prefix_chars += len(sentences[whole - 1])
        triples.append((whole, 0, prefix_chars))
        if whole < len(sentences):
            sentence = sentences[whole]
            for suffix in range(1, len(sentence) + 1):
                triples.append((whole, suffix, prefix_chars + suffix))

    def _closeness(triple: tuple[int, int, int]) -> tuple[float, int, int, int]:
        """Return the closeness."""
        estimate = triple[2] / chars_per_token if chars_per_token > 0 else float(triple[2])
        return (abs(estimate - target), triple[0], triple[1], triple[2])

    triples.sort(key=_closeness)
    return triples


def build_cell_fixture(fixture: Fixture, span: str, placement: str) -> Fixture:
    """The fixture with the filler span inserted in its target.

    Surgery on the last user message's content, pre-render:
    ``span + JOINER + content`` (before) or ``content + JOINER
    + span`` (after). The fixture keeps its original id, so the
    per-fixture seed is identical across conditions.
    """
    index = target_index(fixture)
    content = fixture.messages[index].content
    if placement == "before":
        new_content = span + JOINER + content
    elif placement == "after":
        new_content = content + JOINER + span
    else:
        raise ValueError(f"unknown placement: {placement!r}")
    messages = list(fixture.messages)
    original = messages[index]
    messages[index] = type(original)(role=original.role, content=new_content)
    return Fixture(
        id=fixture.id,
        messages=tuple(messages),
        expected=fixture.expected,
        group=fixture.group,
    )


# ---------------------------------------------------------------------
# Construction classification (measured, never assumed)
# ---------------------------------------------------------------------


def _marker_region(
    fmt: Format, fixture: Fixture, encode: Any
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    """(prefix ids, content ids, suffix ids) of a marker render.

    The fixture's target content is replaced by a marker string;
    Format.render is pure concatenation, so the encoded render
    splits around the marker's own token ids when they survive.
    Returns None when the marker does not survive as a unique
    contiguous run — the caller then reports text_level rather
    than guessing a region.
    """
    index = target_index(fixture)
    messages = list(fixture.messages)
    original = messages[index]
    messages[index] = type(original)(role=original.role, content=_MARKER)
    probe = Fixture(
        id=fixture.id,
        messages=tuple(messages),
        expected=fixture.expected,
        group=fixture.group,
    )
    rendered = fmt.render(probe.messages)
    ids = tuple(encode(rendered))
    marker_ids = tuple(encode(_MARKER))
    if not marker_ids:
        return None
    hits = [
        i
        for i in range(len(ids) - len(marker_ids) + 1)
        if ids[i : i + len(marker_ids)] == marker_ids
    ]
    if len(hits) != 1:
        return None
    start = hits[0]
    return ids[:start], ids[start : start + len(marker_ids)], ids[start + len(marker_ids) :]


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def classify_cell_construction(
    fmt: Format,
    fixture: Fixture,
    span: str,
    placement: str,
    native_ids: Sequence[int],
    arm_ids: Sequence[int],
    encode: Any,
) -> tuple[str, list[int]]:
    """Classify one cell's construction as id_exact or text_level.

    id_exact requires every check: the rendered delta equals
    the span's standalone dose (the caller has already accepted
    the span only when the delta equals the arm level), the
    native prefix and suffix survive bit-identically, the
    content's own tokens are untouched, and the inserted region
    equals the standalone encoding of span+joiner (before) or
    joiner+span (after). Any merge at either seam breaks the
    last check and yields text_level with the inserted region's
    boundary ids as the seam evidence.
    """
    native = _marker_region(fmt, fixture, encode)
    if native is None:
        return "text_level", []
    n_prefix, _n_marker, n_suffix = native
    if tuple(native_ids[: len(n_prefix)]) != n_prefix:
        return "text_level", []
    content_ids = tuple(native_ids[len(n_prefix) : len(native_ids) - len(n_suffix)])
    arm = tuple(arm_ids)
    if placement == "before":
        if arm[: len(n_prefix)] != n_prefix:
            return "text_level", []
        if n_suffix and arm[len(arm) - len(n_suffix) :] != n_suffix:
            return "text_level", []
        content_start = len(arm) - len(n_suffix) - len(content_ids)
        if content_start < len(n_prefix):
            return "text_level", []
        if arm[content_start : content_start + len(content_ids)] != content_ids:
            return "text_level", []
        inserted = arm[len(n_prefix) : content_start]
        standalone = tuple(encode(span + JOINER))
    else:
        if arm[: len(n_prefix)] != n_prefix:
            return "text_level", []
        if n_suffix and arm[len(arm) - len(n_suffix) :] != n_suffix:
            return "text_level", []
        if arm[len(n_prefix) : len(n_prefix) + len(content_ids)] != content_ids:
            return "text_level", []
        inserted_start = len(n_prefix) + len(content_ids)
        inserted_end = len(arm) - len(n_suffix)
        if inserted_end < inserted_start:
            return "text_level", []
        inserted = arm[inserted_start:inserted_end]
        standalone = tuple(encode(JOINER + span))
    if inserted == standalone:
        return "id_exact", []
    seam = [inserted[0], inserted[-1]] if inserted else []
    return "text_level", seam


# ---------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------


def protocol_object(fixture_ids: Sequence[str], backend_fingerprint: str) -> dict[str, Any]:
    """The frozen protocol as data; the fingerprint covers it."""
    return {
        "version": PROTOCOL_VERSION,
        "joiner": JOINER,
        "conditions": [list(c) for c in CONDITIONS],
        "neutral_corpus_sha256": NEUTRAL_CORPUS_SHA256,
        "nearmiss_templates_sha256": NEARMISS_TEMPLATES_SHA256,
        "nearmiss_rule": NEARMISS_RULE,
        "sizing_rule": SIZING_RULE,
        "sizing_encode_budget": SIZING_ENCODE_BUDGET,
        "screening_rule_version": SCREENING_RULE_VERSION,
        "generation_budget": GENERATION_BUDGET,
        "max_ground_records": MAX_GROUND_RECORDS,
        "fixture_set_sha256": hashlib.sha256(
            "\n".join(sorted(fixture_ids)).encode("utf-8")
        ).hexdigest(),
        "backend_fingerprint": backend_fingerprint,
    }


def protocol_fingerprint(fixture_ids: Sequence[str], backend_fingerprint: str) -> str:
    """SHA-256 over the canonical protocol object."""
    canonical = json.dumps(
        protocol_object(fixture_ids, backend_fingerprint),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# Per-candidate aggregation (the validator recomputes this)
# ---------------------------------------------------------------------

_CURVES: tuple[tuple[str, str, tuple[int, ...]], ...] = tuple(
    (kind, placement, NEUTRAL_LEVELS if kind == "neutral" else NEARMISS_LEVELS)
    for kind in ("neutral", "near-miss")
    for placement in PLACEMENTS
)


def curves_for() -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """The (kind, placement, levels) curves, in payload order."""
    return _CURVES


def _cell(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the cell."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def aggregate_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-candidate aggregates over its ground records.

    Every rate is paired: its denominator counts fixtures whose
    native verdict and cell verdict are both recorded and whose
    fixture is not vacuous. First-flip follows the frozen state
    machine over fixtures that pass natively with every curve
    cell evaluable; fail-at-0 fixtures with every cell
    evaluable are the curve's gain population. Capture is
    counted over native-passing fixtures whose near-miss is
    defined for this candidate, and the excess readout
    additionally requires the same-dose neutral cell.
    """
    live = [r for r in records if not r.get("vacuous")]
    curves: dict[str, Any] = {}
    first_flip: dict[str, Any] = {}
    for kind, placement, levels in _CURVES:
        curve_key = f"{kind}_{placement}"
        points: dict[str, Any] = {}
        flip_counts = {str(level): 0 for level in levels}
        n_flip_population = 0
        n_no_flip = 0
        n_returns = 0
        n_gain_population = 0
        n_gains = 0
        for level in levels:
            key = cell_condition(kind, level, placement)
            n_paired = 0
            n_pass = 0
            for record in live:
                native = record.get("native")
                cell = _cell(record, key)
                if not isinstance(native, dict) or cell is None:
                    continue
                if native.get("verdict") is None or cell.get("verdict") is None:
                    continue
                n_paired += 1
                if cell["verdict"]:
                    n_pass += 1
            points[str(level)] = {
                "n_paired": n_paired,
                "n_pass": n_pass,
                "rate": (n_pass / n_paired) if n_paired else None,
            }
        for record in live:
            native = record.get("native")
            if not isinstance(native, dict) or native.get("verdict") is None:
                continue
            verdicts: list[bool | None] = []
            for level in levels:
                cell = _cell(record, cell_condition(kind, level, placement))
                verdicts.append(cell.get("verdict") if cell is not None else None)
            if any(v is None for v in verdicts):
                continue
            if native["verdict"]:
                n_flip_population += 1
                flipped_at: int | None = None
                for level, verdict in zip(levels, verdicts, strict=True):
                    if not verdict:
                        flipped_at = level
                        break
                if flipped_at is None:
                    n_no_flip += 1
                else:
                    flip_counts[str(flipped_at)] += 1
                    for level, verdict in zip(levels, verdicts, strict=True):
                        if level > flipped_at and verdict:
                            n_returns += 1
            else:
                n_gain_population += 1
                if any(verdicts):
                    n_gains += 1
        curves[curve_key] = points
        first_flip[curve_key] = {
            "n_population": n_flip_population,
            "flip_counts": flip_counts,
            "n_no_flip": n_no_flip,
            "n_returns": n_returns,
            "n_gain_population": n_gain_population,
            "n_gains": n_gains,
        }
    capture: dict[str, Any] = {}
    for placement in PLACEMENTS:
        for level in NEARMISS_LEVELS:
            key = cell_condition("near-miss", level, placement)
            neutral_key = cell_condition("neutral", level, placement)
            n_population = 0
            n_captured = 0
            n_excess_population = 0
            n_excess = 0
            for record in live:
                native = record.get("native")
                cell = _cell(record, key)
                if not record.get("near_miss_defined"):
                    continue
                if not isinstance(native, dict) or not native.get("verdict"):
                    continue
                if cell is None or cell.get("verdict") is None:
                    continue
                n_population += 1
                planted_digest = record.get("planted_digest")
                captured = (
                    planted_digest is not None and cell.get("answer_digest") == planted_digest
                )
                if captured:
                    n_captured += 1
                neutral_cell = _cell(record, neutral_key)
                if neutral_cell is not None and neutral_cell.get("verdict") is not None:
                    n_excess_population += 1
                    if captured and neutral_cell.get("answer_digest") != planted_digest:
                        n_excess += 1
            capture[f"{placement}_{level}"] = {
                "n_population": n_population,
                "n_captured": n_captured,
                "capture_rate": (n_captured / n_population) if n_population else None,
                "n_excess_population": n_excess_population,
                "n_excess": n_excess,
                "capture_excess_over_neutral": (
                    (n_excess / n_excess_population) if n_excess_population else None
                ),
            }
    n_nearmiss_defined = sum(1 for r in live if r.get("near_miss_defined"))
    return {
        "curves": curves,
        "first_flip": first_flip,
        "capture": capture,
        "n_fixtures": len(records),
        "n_vacuous": len(records) - len(live),
        "n_echo": sum(1 for r in live if r.get("echo_fixture")),
        "n_nearmiss_defined": n_nearmiss_defined,
        "nearmiss_defined_share": (n_nearmiss_defined / len(live)) if live else None,
    }
