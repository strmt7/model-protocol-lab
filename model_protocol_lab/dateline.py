"""Date-line sensitivity probe protocol: frozen dates, screen, aggregation.

Why this exists: serving stacks prepend a "Today is ..." line the
user never wrote, and Sanz-Guerrero et al. 2026 (arXiv:2609.36931,
*Dating the Model*) show that changing only that injected date
moves scores by up to 6% on MCQA, 14% on math reasoning, and 7%
on code generation across 9 LLMs and 6 datasets, reorders
rankings, and is not removed by few-shot prompting (chain-of-
thought amplifies it). Their instrument varies the date inside
each vendor's own system-prompt rendering. This probe crosses a
frozen date-line treatment with the lab's hand-rendered template
families on one local checkpoint, under exact-match greedy
scoring with the abstention rule, against a length-matched
no-date control -- the combination no surveyed harness ships
(docs/RESEARCH_LOG.md, cycle 188). The design contract, the
red-team record behind it, and the falsification statement live
in docs/DATELINE_DESIGN.md.

Definitions that carry the design (red-team cross-checked,
cycle 188):

* Every date is a protocol constant, never the wall clock, and
  every weekday name is frozen protocol data computed from its
  own calendar date at freeze time. Nothing here reads the
  clock or the locale; month and weekday names come from the
  frozen tables below.
* The headline is per-pair discordant counts between date arms,
  partitioned by pair type (pure-date, pure-format, cross-both).
  A union "any-pair flip rate" is not a shipped statistic: over
  24 date arms it saturates under generic per-arm fragility and
  would report a 5% wobble as a majority "date-flip rate".
* The date-loss subset is a joint per-fixture property (passes
  the no-date control, fails a date value), defined per date
  value with formats collapsed, with its asymmetric counterpart
  reported beside it. It is not leading's "caving": nothing is
  planted here for the model to adopt.
* Records carry answer digests only. The verdict is recomputed
  by the diff validator as answer_digest == expected_digest
  conjoined with the boundary flag, so there is no stored
  derived answer field to forge (the numform F3 lesson, applied
  by construction rather than repaired later).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import date as _date
from itertools import combinations
from typing import Any

from .confidentiality import mcnemar_exact_p
from .core import Fixture
from .repetition import target_index
from .scorerlens import rung_match

PROTOCOL_VERSION = 1

#: The frozen block envelopes. The date sentence is the
#: treatment; the control keeps the envelope and replaces the
#: sentence with sized neutral text.
BLOCK_TEMPLATE = "[Context] Today's date is {DATE}. [/Context]"
CONTROL_TEMPLATE = "[Context] {NEUTRAL}[/Context]"
JOINER = "\n\n"

#: Frozen neutral unit for the no-date control. Written for this
#: probe and temporally inert: repetition's filler contains the
#: temporal phrase "that morning", which would put temporal
#: priming in the no-date control of a date probe. The trailing
#: space is part of the unit (the repetition filler convention),
#: so whole units concatenate cleanly and the control block's
#: single space before the closing bracket comes from the unit.
NEUTRAL_UNIT = "The reference panel is mounted on the north wall of the room. "
SIZING_RULE = "whole-units-plus-one-truncated-suffix-v1"
#: Bound on rendered-length probes per (candidate, fixture,
#: format) while searching for an exactly-sized control text.
SIZING_SEARCH_BUDGET = 512

SCREEN_RULE = "dateline-components-r4-v1"

MONTH_NAMES: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
#: Index 0 is Monday (datetime.date.weekday()'s convention).
WEEKDAY_NAMES: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
MONTH_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "January": ("Jan",),
    "February": ("Feb",),
    "March": ("Mar",),
    "April": ("Apr",),
    "May": (),
    "June": ("Jun",),
    "July": ("Jul",),
    "August": ("Aug",),
    "September": ("Sep", "Sept"),
    "October": ("Oct",),
    "November": ("Nov",),
    "December": ("Dec",),
}
WEEKDAY_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "Monday": ("Mon",),
    "Tuesday": ("Tue", "Tues"),
    "Wednesday": ("Wed",),
    "Thursday": ("Thu", "Thur", "Thurs"),
    "Friday": ("Fri",),
    "Saturday": ("Sat",),
    "Sunday": ("Sun",),
}

#: The frozen date values: (key, (year, month, day)). Weekdays
#: at freeze time, each computed from its own date and pinned by
#: test: anchor Wednesday, minus_day Tuesday, plus_day Thursday,
#: minus_year Tuesday, plus_year Thursday, leap Tuesday,
#: boundary_eve Thursday, boundary_day Friday.
DATE_VALUES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("anchor", (2026, 10, 21)),
    ("minus_day", (2026, 10, 20)),
    ("plus_day", (2026, 10, 22)),
    ("minus_year", (2025, 10, 21)),
    ("plus_year", (2027, 10, 21)),
    ("leap", (2028, 2, 29)),
    ("boundary_eve", (2026, 12, 31)),
    ("boundary_day", (2027, 1, 1)),
)
VALUE_KEYS: tuple[str, ...] = tuple(key for key, _ in DATE_VALUES)
_DATE_BY_KEY: dict[str, tuple[int, int, int]] = dict(DATE_VALUES)

FORMATS: tuple[str, ...] = ("iso", "long", "weekday_iso")

#: Treated arm keys, in payload order: 24 date arms (value-major),
#: 3 per-format no-date controls, 1 weekday-lie arm.
DATE_ARM_KEYS: tuple[str, ...] = tuple(f"{value}_{fmt}" for value in VALUE_KEYS for fmt in FORMATS)
CONTROL_KEYS: tuple[str, ...] = tuple(f"nodate_{fmt}" for fmt in FORMATS)
LIE_KEY = "lie"
TREATED_KEYS: tuple[str, ...] = DATE_ARM_KEYS + CONTROL_KEYS + (LIE_KEY,)

#: The lie arm: the anchor date in the weekday-bearing format
#: with the weekday name shifted +1 day (Thursday for the
#: anchor's Wednesday -- the smaller length step). Defined only
#: inside weekday_iso; its literal date string is frozen data,
#: and the diff validator checks it differs from the recomputed
#: calendar weekday rather than deriving it.
LIE_VALUE = "anchor"
LIE_FORMAT = "weekday_iso"
LIE_WEEKDAY_SHIFT = 1

#: Discordance floors, frozen at leading's values. They gate
#: classifications only, never a reported count or rate.
MIN_PAIRED_EVALUABLE = 6
MIN_DISCORDANT = 2

#: 28 treated arms x MAX_GROUND_RECORDS.
GENERATION_BUDGET = 67200
MAX_GROUND_RECORDS = 2400
#: Records-bytes cap. The driver is cells per record (the record
#: cap already folds in the candidate multiplication): 29 cells
#: of the leading minimal shape measure ~5.2 KB per record as
#: canonical JSON on the cycle-188 test schema, so 2,400 records
#: at the legal maximum measure ~12.6 MiB; the cap leaves
#: headroom above that measurement for longer fixture ids and
#: seam lists, and stays under the 32 MiB whole-file loader
#: limit.
MAX_RECORDS_BYTES = 24 * 1024 * 1024

#: The pair-classification vocabulary (a total function of the
#: paired counts -- see classify_pair).
CLASSIFICATIONS: tuple[str, ...] = (
    "insufficient",
    "no_discordance_observed",
    "discordant_asymmetric",
    "discordant_balanced",
    "below_threshold",
)

CELL_DETAILS: tuple[str, ...] = (
    "",
    "vacuous",
    "expected_blank",
    "screened",
    "message_date_bearing",
    "arm_screened",
    "arm_context_exceeded",
    "arm_evaluation_failed",
    "control_unmatchable",
    "control_screened",
)


def digest_text(text: str) -> str:
    """SHA-256 hexdigest of a text (the payload's only view of it)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _calendar_weekday(year: int, month: int, day: int) -> int:
    """Weekday index (Monday=0) by pure calendar arithmetic.

    ``datetime.date.weekday`` is locale-free integer arithmetic
    on a frozen constant -- no clock is read.
    """
    return _date(year, month, day).weekday()


def format_date(value_key: str, fmt: str) -> str:
    """The frozen rendering of one date value in one format."""
    year, month, day = _DATE_BY_KEY[value_key]
    if fmt == "iso":
        return f"{year:04d}-{month:02d}-{day:02d}"
    if fmt == "long":
        return f"{day} {MONTH_NAMES[month - 1]} {year}"
    if fmt == "weekday_iso":
        weekday = WEEKDAY_NAMES[_calendar_weekday(year, month, day)]
        return f"{weekday}, {year:04d}-{month:02d}-{day:02d}"
    raise ValueError(f"unknown dateline format: {fmt!r}")


def lie_date_string() -> str:
    """The lie arm's frozen date string (weekday shifted +1)."""
    year, month, day = _DATE_BY_KEY[LIE_VALUE]
    weekday = WEEKDAY_NAMES[(_calendar_weekday(year, month, day) + LIE_WEEKDAY_SHIFT) % 7]
    return f"{weekday}, {year:04d}-{month:02d}-{day:02d}"


def arm_date_string(arm_key: str) -> str:
    """The date string carried by a date arm (or the lie arm)."""
    if arm_key == LIE_KEY:
        return lie_date_string()
    # Value keys contain underscores, so split on the format side:
    # the longest format suffix that leaves a known value key.
    for fmt in ("weekday_iso", "long", "iso"):
        suffix = f"_{fmt}"
        if arm_key.endswith(suffix) and arm_key[: -len(suffix)] in _DATE_BY_KEY:
            return format_date(arm_key[: -len(suffix)], fmt)
    raise ValueError(f"unknown dateline arm: {arm_key!r}")


def date_arm_table() -> dict[str, str]:
    """Every treated date-bearing arm key -> its frozen date string."""
    table = {key: arm_date_string(key) for key in DATE_ARM_KEYS}
    table[LIE_KEY] = lie_date_string()
    return table


def block_text(date_string: str) -> str:
    """The treatment block for one date string."""
    return BLOCK_TEMPLATE.replace("{DATE}", date_string)


# skipcq: PYL-W0621 - the name mirrors the frozen payload field it consumes
def control_block(neutral_text: str) -> str:
    """The control block: same envelope, sized neutral text."""
    return CONTROL_TEMPLATE.replace("{NEUTRAL}", neutral_text)


def neutral_text(reps: int, cut: int) -> str:
    """Whole neutral units plus at most one truncated suffix."""
    return NEUTRAL_UNIT * reps + NEUTRAL_UNIT[:cut]


def neutral_search_order(unit_token_estimate: int, delta: int) -> list[tuple[int, int]]:
    """Deterministic (reps, cut) order for the control sizing search.

    Mirrors repetition.filler_search_order over this probe's own
    unit: ordered by closeness of the estimated token count to
    ``delta`` (ties by fewer reps, then shorter cut), capped at
    SIZING_SEARCH_BUDGET; the caller verifies each candidate on
    the actual rendered ids.
    """
    max_reps = max(2, delta // max(1, unit_token_estimate) + 3)
    pairs: list[tuple[int, int, int]] = []
    for reps in range(0, max_reps + 1):
        for cut in range(0, len(NEUTRAL_UNIT) + 1):
            if reps == 0 and cut == 0:
                continue
            estimate = reps * unit_token_estimate + round(
                unit_token_estimate * cut / len(NEUTRAL_UNIT)
            )
            pairs.append((abs(estimate - delta), reps, cut))
    pairs.sort()
    return [(reps, cut) for _dist, reps, cut in pairs[:SIZING_SEARCH_BUDGET]]


def build_arm_fixture(fixture: Fixture, arm_key: str, neutral: str = "") -> Fixture:
    """The fixture with the arm's block prepended to its last user message.

    Surgery on the target content, pre-render:
    ``block + JOINER + content``. The fixture keeps its original
    id, group, and expected, so the per-fixture seed is identical
    across arms. Control arms take the sized neutral text.
    """
    if arm_key in CONTROL_KEYS:
        block = control_block(neutral)
    else:
        block = block_text(arm_date_string(arm_key))
    index = target_index(fixture)
    content = fixture.messages[index].content
    messages = list(fixture.messages)
    original = messages[index]
    messages[index] = type(original)(role=original.role, content=block + JOINER + content)
    return Fixture(
        id=fixture.id,
        messages=tuple(messages),
        expected=fixture.expected,
        group=fixture.group,
    )


# ---------------------------------------------------------------------
# Eligibility screen (docs/DATELINE_DESIGN.md, rule
# dateline-components-r4-v1). A fixture whose expected answer is
# date-derived measures copying, not sensitivity. Both naive
# poles are unsound here (executed pre-build): rung-4 equality
# against the full arm strings catches none of "21" / "2026" /
# "October", and substring containment voids nearly every
# single-digit numeric fixture ("1" is contained in the anchor's
# ISO string). The screen is component-wise rung-4 equivalence
# plus a closed-format date-parse leg.
# ---------------------------------------------------------------------

_COMPONENT_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def _components(text: str) -> list[str]:
    """Alphanumeric components of a text (the screen's tokens)."""
    return [token for token in _COMPONENT_SPLIT.split(text) if token]


def date_bearing_components() -> frozenset[str]:
    """Date-bearing tokens of every arm string, plus name aliases.

    Years, month numbers and names, day numbers, weekday names --
    and the frozen abbreviation aliases for every month/weekday
    name present. Carrier words from the block envelope
    ("Today's", "date", "is", "Context") are deliberately absent:
    screening on envelope words would void common-word expecteds
    for a property of the envelope, not of the date.
    """
    components: set[str] = set()
    for date_string in date_arm_table().values():
        components.update(_components(date_string))
    for name, aliases in MONTH_ABBREVIATIONS.items():
        if name in components:
            components.update(aliases)
    for name, aliases in WEEKDAY_ABBREVIATIONS.items():
        if name in components:
            components.update(aliases)
    return frozenset(components)


_MONTH_LOOKUP: dict[str, int] = {}
for _index, _name in enumerate(MONTH_NAMES, start=1):
    _MONTH_LOOKUP[_name.casefold()] = _index
    for _alias in MONTH_ABBREVIATIONS[_name]:
        _MONTH_LOOKUP[_alias.casefold()] = _index

_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_D_MONTH_Y_RE = re.compile(r"^(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})$")
_MONTH_D_Y_RE = re.compile(r"^([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$")
_SLASH_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_DOTTED_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")
_Y_SLASH_RE = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$")
_D_MONTH_RE = re.compile(r"^(\d{1,2})\s+([A-Za-z]+)\.?$")
_MONTH_D_RE = re.compile(r"^([A-Za-z]+)\.?\s+(\d{1,2})$")


def _valid_date(year: int, month: int, day: int) -> bool:
    """Return whether date is valid."""
    try:
        _date(year, month, day)
    except ValueError:
        return False
    return True


def parse_date_candidates(text: str) -> set[tuple[int | None, int, int]]:
    """(year | None, month, day) readings of a whole string.

    The closed format list of the design doc, matched against the
    protocol's frozen English month table -- no strptime, no
    locale, no general parser. Slash dates yield BOTH readings
    (3/4/2026 is 2026-03-04 or 2026-04-03); a year-less month/day
    yields year None. Readings that are not real calendar dates
    are dropped.
    """
    text = text.strip()
    out: set[tuple[int | None, int, int]] = set()
    match = _ISO_RE.match(text) or _Y_SLASH_RE.match(text)
    if match:
        year, month, day = (int(match.group(i)) for i in (1, 2, 3))
        if _valid_date(year, month, day):
            out.add((year, month, day))
        return out
    match = _D_MONTH_Y_RE.match(text)
    if match:
        month_num = _MONTH_LOOKUP.get(match.group(2).casefold())
        if month_num is not None:
            day, year = int(match.group(1)), int(match.group(3))
            if _valid_date(year, month_num, day):
                out.add((year, month_num, day))
        return out
    match = _MONTH_D_Y_RE.match(text)
    if match:
        month_num = _MONTH_LOOKUP.get(match.group(1).casefold())
        if month_num is not None:
            day, year = int(match.group(2)), int(match.group(3))
            if _valid_date(year, month_num, day):
                out.add((year, month_num, day))
        return out
    match = _SLASH_RE.match(text)
    if match:
        first, second, year = (int(match.group(i)) for i in (1, 2, 3))
        for month, day in ((first, second), (second, first)):
            if _valid_date(year, month, day):
                out.add((year, month, day))
        return out
    match = _DOTTED_RE.match(text)
    if match:
        day, month, year = (int(match.group(i)) for i in (1, 2, 3))
        if _valid_date(year, month, day):
            out.add((year, month, day))
        return out
    match = _D_MONTH_RE.match(text)
    if match:
        month_num = _MONTH_LOOKUP.get(match.group(2).casefold())
        if month_num is not None:
            day = int(match.group(1))
            if _valid_date(2024, month_num, day):  # leap year: Feb 29 is a real month/day
                out.add((None, month_num, day))
        return out
    match = _MONTH_D_RE.match(text)
    if match:
        month_num = _MONTH_LOOKUP.get(match.group(1).casefold())
        if month_num is not None:
            day = int(match.group(2))
            if _valid_date(2024, month_num, day):
                out.add((None, month_num, day))
        return out
    return out


def _arm_date_tuples() -> frozenset[tuple[int, int, int]]:
    """Return the arm date tuples."""
    return frozenset(_DATE_BY_KEY.values())


def _parse_hits_arm(text: str) -> bool:
    """Whether a whole-string date reading lands on an arm date."""
    arm_dates = _arm_date_tuples()
    arm_month_days = frozenset((month, day) for _, month, day in arm_dates)
    for year, month, day in parse_date_candidates(text):
        if year is None:
            if (month, day) in arm_month_days:
                return True
        elif (year, month, day) in arm_dates:
            return True
    return False


_MESSAGE_DATE_RE = re.compile(
    r"\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}/\d{1,2}/\d{4}"
    r"|\d{4}/\d{1,2}/\d{1,2}"
    r"|\d{1,2}\.\d{1,2}\.\d{4}"
    r"|\d{1,2}\s+(?:" + "|".join(MONTH_NAMES) + r")\.?(?:\s*,?\s+\d{4})?"
    r"|(?:" + "|".join(MONTH_NAMES) + r")\.?\s+\d{1,2}(?:\s*,\s*\d{4})?"
)


def message_states_date(text: str) -> bool:
    """Whether a message text contains a recognisable date.

    Substring search over the same closed format family (month
    names in their full frozen form). A fixture whose question
    states a date turns the arm into a hierarchy-style conflict,
    not a sensitivity measurement, so the stage flags it as its
    own record class.
    """
    return _MESSAGE_DATE_RE.search(text) is not None


def screen_expected(expected: str) -> bool:
    """Whether an expected answer is date-derived (screen rule).

    Three legs: rung-4 equality against any full arm string;
    rung-4 equality of any expected component against any
    date-bearing arm component; the closed-format parse leg.
    Rung 4 collapses leading-zero and decimal variants ("021"
    matches the day component "21"): a deliberate over-exclusion,
    disclosed in the design doc -- the screen is over answer
    values, and the scorer-lens ladder already treats rung-4-equal
    answers as the same answer.
    """
    if not expected.strip():
        return False
    table = date_arm_table()
    if any(rung_match(expected, arm_string, 4) for arm_string in table.values()):
        return True
    components = date_bearing_components()
    for component in _components(expected):
        if any(rung_match(component, arm_component, 4) for arm_component in components):
            return True
    return _parse_hits_arm(expected)


def fixture_screen(fixture: Fixture) -> tuple[bool, bool]:
    """(screened, message_date_bearing) for one fixture."""
    screened = screen_expected(fixture.expected)
    bearing = any(message_states_date(message.content) for message in fixture.messages)
    return screened, bearing


def screen_neutral(
    expected_by_fixture: dict[str, str],
    boundary_text_by_candidate: dict[str, str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Screen the neutral unit against answers and boundaries.

    The repetition.screen_filler discipline over this probe's own
    unit: a fixture is control-screened when its expected could
    be copied from the neutral text; a candidate when its decoded
    boundary text occurs in it.
    """
    unit = NEUTRAL_UNIT.casefold()
    screened_fixtures = set()
    for fixture_id, expected in expected_by_fixture.items():
        answer = expected.strip().casefold()
        if answer and (answer in unit or unit.strip() in answer):
            screened_fixtures.add(fixture_id)
    screened_candidates = set()
    for name, boundary_text in boundary_text_by_candidate.items():
        if boundary_text and boundary_text in NEUTRAL_UNIT:
            screened_candidates.add(name)
    return frozenset(screened_fixtures), frozenset(screened_candidates)


def fixture_set_sha256(fixture_ids: Sequence[str]) -> str:
    """SHA-256 over the sorted fixture id set."""
    return hashlib.sha256("\n".join(sorted(fixture_ids)).encode("utf-8")).hexdigest()


def protocol_object(
    fixture_ids: Sequence[str],
    backend_fingerprint: str,
) -> dict[str, Any]:
    """The frozen protocol as data; the fingerprint covers it."""
    return {
        "version": PROTOCOL_VERSION,
        "block_template": BLOCK_TEMPLATE,
        "control_template": CONTROL_TEMPLATE,
        "joiner": JOINER,
        "neutral_unit": NEUTRAL_UNIT,
        "sizing_rule": SIZING_RULE,
        "screen_rule": SCREEN_RULE,
        "arm_dates": date_arm_table(),
        "treated_arms": list(TREATED_KEYS),
        "min_paired_evaluable": MIN_PAIRED_EVALUABLE,
        "min_discordant": MIN_DISCORDANT,
        "generation_budget": GENERATION_BUDGET,
        "max_ground_records": MAX_GROUND_RECORDS,
        "max_records_bytes": MAX_RECORDS_BYTES,
        "fixture_set_sha256": fixture_set_sha256(fixture_ids),
        "backend_fingerprint": backend_fingerprint,
    }


def protocol_fingerprint(
    fixture_ids: Sequence[str],
    backend_fingerprint: str,
) -> str:
    """SHA-256 over the canonical protocol object."""
    canonical = json.dumps(
        protocol_object(fixture_ids, backend_fingerprint),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def classify_pair(n_paired: int, a_only: int, b_only: int) -> str:
    """The per-pair classification (total function).

    Over the paired population (both cells completed),
    ``a_only`` counts pass-A/fail-B and ``b_only`` the mirror.
    The floors gate this label only:

    - ``insufficient``: paired population below the floor;
    - ``no_discordance_observed``: no discordance either way;
    - ``discordant_balanced``: equal discordance at/above the
      floor in both directions (generic fragility, acquitted);
    - ``discordant_asymmetric``: discordance at/above the floor
      concentrated on one side;
    - ``below_threshold``: the remaining small-discordance cases.
    """
    if n_paired < MIN_PAIRED_EVALUABLE:
        return "insufficient"
    if a_only == 0 and b_only == 0:
        return "no_discordance_observed"
    if a_only == b_only and a_only >= MIN_DISCORDANT:
        return "discordant_balanced"
    if max(a_only, b_only) >= MIN_DISCORDANT:
        return "discordant_asymmetric"
    return "below_threshold"


# ---------------------------------------------------------------------
# Per-candidate aggregation (the validator recomputes this)
# ---------------------------------------------------------------------


def _cell(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the cell."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def live_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records that enter populations: not vacuous, blank,
    screened, or message-date-bearing.
    """
    return [
        r
        for r in records
        if not r.get("vacuous")
        and not r.get("expected_blank")
        and not r.get("screened")
        and not r.get("message_date_bearing")
    ]


_live = live_records


def _pair_block(records: list[dict[str, Any]], key_a: str, key_b: str) -> dict[str, Any]:
    """Discordant counts for one arm pair over the paired population."""
    n_paired = 0
    n_a_only = 0
    n_b_only = 0
    n_both = 0
    n_pass_a = 0
    n_pass_b = 0
    for record in records:
        cell_a = _cell(record, key_a)
        cell_b = _cell(record, key_b)
        if cell_a is None or cell_b is None:
            continue
        n_paired += 1
        verdict_a = bool(cell_a["verdict"])
        verdict_b = bool(cell_b["verdict"])
        if verdict_a:
            n_pass_a += 1
        if verdict_b:
            n_pass_b += 1
        if verdict_a and verdict_b:
            n_both += 1
        elif verdict_a:
            n_a_only += 1
        elif verdict_b:
            n_b_only += 1
    return {
        "n_paired": n_paired,
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "n_both": n_both,
        "n_neither": n_paired - n_a_only - n_b_only - n_both,
        "rate_a": (n_pass_a / n_paired) if n_paired else None,
        "rate_b": (n_pass_b / n_paired) if n_paired else None,
        "mcnemar_p": mcnemar_exact_p(n_a_only, n_b_only),
        "classification": classify_pair(n_paired, n_a_only, n_b_only),
    }


def _loss_block(live: list[dict[str, Any]], value: str) -> dict[str, Any]:
    """Date-attributable loss for one date value, formats collapsed.

    A fixture is evaluable for the value when, in at least one
    format, both the format's control cell and the value's cell
    completed. It counts as lost when in at least one such format
    it passes the control and fails the value; the counterpart
    counts fixtures that fail the control and pass the value.
    Denominators are per value by construction.
    """
    evaluable: set[str] = set()
    lost: set[str] = set()
    counterpart: set[str] = set()
    per_format: dict[str, Any] = {}
    for fmt in FORMATS:
        control_key = f"nodate_{fmt}"
        value_key = f"{value}_{fmt}"
        n_evaluable = 0
        n_loss = 0
        for record in live:
            control = _cell(record, control_key)
            cell = _cell(record, value_key)
            if control is None or cell is None:
                continue
            n_evaluable += 1
            evaluable.add(str(record.get("fixture")))
            if control["verdict"] and not cell["verdict"]:
                n_loss += 1
                lost.add(str(record.get("fixture")))
            elif cell["verdict"] and not control["verdict"]:
                counterpart.add(str(record.get("fixture")))
        per_format[fmt] = {"n_evaluable": n_evaluable, "n_loss": n_loss}
    block: dict[str, Any] = {
        "n_evaluable": len(evaluable),
        "n_loss": len(lost),
        "loss_rate": (len(lost) / len(evaluable)) if evaluable else None,
        "n_counterpart": len(counterpart),
        "per_format": per_format,
    }
    if value == LIE_VALUE:
        # The lie split, defined only for the anchor in the
        # weekday-bearing format: failing the lie too is
        # consistent with "any date text perturbs"; passing it
        # while failing the honest anchor is content-specific.
        control_key = f"nodate_{LIE_FORMAT}"
        value_key = f"{LIE_VALUE}_{LIE_FORMAT}"
        n_three = 0
        n_any_text = 0
        n_content_specific = 0
        for record in live:
            control = _cell(record, control_key)
            cell = _cell(record, value_key)
            lie = _cell(record, LIE_KEY)
            if control is None or cell is None or lie is None:
                continue
            n_three += 1
            if control["verdict"] and not cell["verdict"]:
                if not lie["verdict"]:
                    n_any_text += 1
                else:
                    n_content_specific += 1
        block["lie_split"] = {
            "n_evaluable": n_three,
            "n_loss_any_text": n_any_text,
            "n_loss_content_specific": n_content_specific,
        }
    else:
        block["lie_split"] = None
    return block


def aggregate_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-candidate aggregates over its ground records.

    Every count carries its denominator; a rate is None only when
    its denominator is 0. The falsification flag is the design
    doc's restated condition: every pure-date pair's discordant
    count is zero and every date value's loss count is zero.
    """
    live = _live(records)
    arms: dict[str, Any] = {}
    for key in TREATED_KEYS:
        n_completed = 0
        n_pass = 0
        for record in live:
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
    pairs_by_format: dict[str, Any] = {}
    total_pure_date_discordant = 0
    for fmt in FORMATS:
        table: dict[str, Any] = {}
        for value_a, value_b in combinations(VALUE_KEYS, 2):
            block = _pair_block(live, f"{value_a}_{fmt}", f"{value_b}_{fmt}")
            total_pure_date_discordant += block["n_a_only"] + block["n_b_only"]
            table[f"{value_a}|{value_b}"] = block
        pairs_by_format[fmt] = table
    format_pairs: dict[str, Any] = {}
    for value in VALUE_KEYS:
        table = {}
        for fmt_a, fmt_b in combinations(FORMATS, 2):
            table[f"{fmt_a}|{fmt_b}"] = _pair_block(live, f"{value}_{fmt_a}", f"{value}_{fmt_b}")
        format_pairs[value] = table
    date_loss = {value: _loss_block(live, value) for value in VALUE_KEYS}
    control_evaluable: dict[str, Any] = {}
    for fmt in FORMATS:
        flags = [
            record.get("control_evaluable", {}).get(fmt)
            for record in live
            if isinstance(record.get("control_evaluable"), dict)
        ]
        n_evaluable = sum(1 for flag in flags if flag)
        control_evaluable[fmt] = {
            "n_evaluable": n_evaluable,
            "n_live": len(flags),
            "share": (n_evaluable / len(flags)) if flags else None,
        }
    return {
        "arms": arms,
        "pairs_by_format": pairs_by_format,
        "format_pairs": format_pairs,
        "date_loss": date_loss,
        "control_evaluable": control_evaluable,
        "date_content_falsified": bool(
            total_pure_date_discordant == 0
            and all(block["n_loss"] == 0 for block in date_loss.values())
        ),
        "n_fixtures": len(records),
        "n_vacuous": sum(1 for r in records if r.get("vacuous")),
        "n_expected_blank": sum(1 for r in records if r.get("expected_blank")),
        "n_echo": sum(1 for r in live if r.get("echo_fixture")),
        "n_screened": sum(1 for r in records if r.get("screened")),
        "n_message_date_bearing": sum(1 for r in records if r.get("message_date_bearing")),
    }


def treated_population_ids(records: list[dict[str, Any]]) -> frozenset[str]:
    """Fixture ids with every treated cell completed (one candidate)."""
    ids: set[str] = set()
    for record in _live(records):
        if all(_cell(record, key) is not None for key in TREATED_KEYS):
            ids.add(str(record.get("fixture")))
    return frozenset(ids)


def common_population(
    records_by_candidate: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """The common-population readout (the ranked comparison).

    Fixtures in *every* available candidate's treated population.
    Per candidate over that set: summed pure-date discordants and
    summed date-loss counts. Emptiness is reported with a fixed
    detail and no rates, never silently (the leading pattern).
    """
    populations = {
        name: treated_population_ids(records) for name, records in records_by_candidate.items()
    }
    if not populations:
        return {"n_common": 0, "detail": "common_population_empty", "per_candidate": {}}
    common = frozenset.intersection(*populations.values())
    if not common:
        return {"n_common": 0, "detail": "common_population_empty", "per_candidate": {}}
    per_candidate: dict[str, Any] = {}
    for name, records in records_by_candidate.items():
        subset = [r for r in _live(records) if r.get("fixture") in common]
        discordant = 0
        for fmt in FORMATS:
            for value_a, value_b in combinations(VALUE_KEYS, 2):
                block = _pair_block(subset, f"{value_a}_{fmt}", f"{value_b}_{fmt}")
                discordant += block["n_a_only"] + block["n_b_only"]
        loss = sum(_loss_block(subset, value)["n_loss"] for value in VALUE_KEYS)
        per_candidate[name] = {
            "n_pure_date_discordant": discordant,
            "n_loss_total": loss,
        }
    detail = "" if len(common) >= MIN_PAIRED_EVALUABLE else "common_population_below_floor"
    return {"n_common": len(common), "detail": detail, "per_candidate": per_candidate}


def classify_construction(
    region_native: tuple[int, ...] | None,
    region_treated: tuple[int, ...] | None,
    block_region: tuple[int, ...] | None,
    joiner_region: tuple[int, ...] | None,
) -> tuple[str, tuple[int, ...]]:
    """Classify the token-level construction at the prepend seam.

    ``id_exact`` iff the treated content region is exactly the
    block region, the joiner region, and the native region -- no
    cross-seam merge anywhere. Otherwise ``text_level``, with the
    seam ids (the tokens around the block/joiner/content
    junctions) recorded when the regions are locatable. A
    candidate whose prefix merges into "[Context]" is thereby
    distinguishable from a clean render in the payload.
    """
    if (
        region_native is None
        or region_treated is None
        or block_region is None
        or joiner_region is None
    ):
        return "text_level", ()
    if region_treated == block_region + joiner_region + region_native:
        return "id_exact", ()
    junction = len(block_region)
    seam = region_treated[max(0, junction - 1) : junction + len(joiner_region) + 2]
    return "text_level", seam
