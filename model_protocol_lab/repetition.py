"""Prompt-repetition probe protocol: arm construction and aggregation.

Why this exists: Leviathan, Kalman & Matias (arXiv:2512.14982) show
that repeating a prompt improves non-reasoning LLMs — on API models,
each rendered with its provider's own chat template, with a padding
ablation showing random-token padding does not reproduce the effect.
Whether repetition changes verdicts when the *chat template* is the
variable — the same checkpoint, foreign hand-rendered formats, an
abstention rule — is a different question, and the one this probe
answers. The paper's "no latency increase" claim is not imported:
this lab's decoder recomputes the full prefix with no KV cache, so a
doubled prompt costs real compute (see docs/REPETITION_DESIGN.md).

The module is pure. It freezes the protocol (joiner, filler unit,
pre-repetition detector), builds arm fixtures by surgery on the last
user message's content *before* render (the copy lives inside the
same user turn — a post-render append would land in the assistant's
mouth), classifies the token-level construction, and aggregates
per-fixture ground records into the reported contrasts. Generation
lives in ``experiments.repetition_stage``.

Definitions that carry the design (red-team cross-checked, cycle
183):

* The construction invariant is string-level: an arm render equals
  the x1 render with the target content replaced — verified by
  *rebuilding* the render from the arm fixture, never by textual
  search-and-replace (identical first/last user contents make
  replace-based oracles certify a wrongly-doubled first turn).
  Token-level exactness is a *measured* property (`id_exact` vs
  `text_level` with seam ids), not an invariant with a fallback.
* The padding control is matched per (candidate, fixture) on
  rendered token length, by a bounded search over whole filler units
  plus at most one truncated suffix, verified on the rendered ids.
  Pairs that cannot be matched exactly — delta <= 0 with no
  positive-filler match, overshoot, search exhausted — are
  `pad_evaluable: False`, never shipped mismatched.
* Gains are paired per contrast: each rate uses only fixtures
  completed in both compared arms, with `n_paired` reported and
  rates `None` (never 0.0) when nothing is paired.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .core import Fixture, Message

PROTOCOL_VERSION = 1
JOINER = "\n"
# Frozen filler unit: declarative, non-instruction prose with no
# digits and no answer-shaped tokens we can foresee; it is still
# screened against the run's own expected answers, boundary texts,
# and the special-token inventory before any pad arm is built,
# because neutrality is a property of the run, not of the text.
FILLER_UNIT = "the lane past the orchard was quiet that morning and the air smelled of rain. "
DETECTOR_VERSION = 1
SIZING_RULE = "whole-units-plus-one-truncated-suffix-v1"
MAX_GROUND_RECORDS = 2400
# Bound on rendered-length probes per (candidate, fixture) while
# searching for an exactly-sized filler.
SIZING_SEARCH_BUDGET = 512

ARMS: tuple[str, ...] = ("x1", "x2", "pad")
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("x2", "x1", "raw_gain"),
    ("pad", "x1", "length_effect"),
    ("x2", "pad", "repetition_effect"),
)


def fixture_set_sha256(fixture_ids: list[str] | tuple[str, ...]) -> str:
    """The scorerlens precedent: SHA-256 over sorted fixture ids."""
    return hashlib.sha256("\n".join(sorted(fixture_ids)).encode("utf-8")).hexdigest()


def protocol_fingerprint(fixture_ids: list[str] | tuple[str, ...], backend_fingerprint: str) -> str:
    """SHA-256 over the canonical protocol table (drift is a version bump)."""
    table = {
        "version": PROTOCOL_VERSION,
        "joiner": JOINER,
        "filler_unit_sha256": hashlib.sha256(FILLER_UNIT.encode("utf-8")).hexdigest(),
        "detector_version": DETECTOR_VERSION,
        "sizing_rule": SIZING_RULE,
        "fixture_set_sha256": fixture_set_sha256(fixture_ids),
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = json.dumps(table, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def protocol_block(
    fixture_ids: list[str] | tuple[str, ...], backend_fingerprint: str
) -> dict[str, Any]:
    """The payload's protocol record."""
    return {
        "version": PROTOCOL_VERSION,
        "fingerprint": protocol_fingerprint(fixture_ids, backend_fingerprint),
        "joiner": JOINER,
        "filler_unit_sha256": hashlib.sha256(FILLER_UNIT.encode("utf-8")).hexdigest(),
        "detector_version": DETECTOR_VERSION,
        "sizing_rule": SIZING_RULE,
        "fixture_set_sha256": fixture_set_sha256(fixture_ids),
        "backend_fingerprint": backend_fingerprint,
    }


def target_index(fixture: Fixture) -> int:
    """Index of the last user message (the deliforge precedent)."""
    for index in range(len(fixture.messages) - 1, -1, -1):
        if fixture.messages[index].role == "user":
            return index
    raise ValueError("fixture has no user message")


def target_content(fixture: Fixture) -> str:
    """The last user message's content — the surgery target."""
    return fixture.messages[target_index(fixture)].content


def is_vacuous(content: str) -> bool:
    """Vacuous is a content predicate, not render equality: under the
    frozen joiner an empty content's x2 render differs from x1 by the
    joiner alone, so render comparison would miss it.
    """
    return content.strip() == ""


def detect_pre_repeated(content: str) -> bool:
    """Detector v1 (exact, mechanical).

    Let c be the content after whitespace-collapse (runs of
    whitespace become one space, ends stripped; case-sensitive).
    Flag iff (i) c = t + sep + t for some nonempty t and sep in
    {"", " "}, or (ii) c contains two immediately adjacent copies of
    a substring of length >= 8 (contiguous, no separator).

    This is an exact-repetition screen: one-word-changed near
    repeats are not flagged, by definition, and that limitation is
    named wherever the pooled/excluded split is reported.
    """
    c = " ".join(content.split())
    n = len(c)
    for sep in ("", " "):
        rest = n - len(sep)
        if rest > 0 and rest % 2 == 0:
            half = rest // 2
            if c[:half] == c[half + len(sep) :]:
                return True
    for size in range(8, n // 2 + 1):
        for start in range(0, n - 2 * size + 1):
            if c[start : start + size] == c[start + size : start + 2 * size]:
                return True
    return False


def is_echo_fixture(fixture: Fixture) -> bool:
    """The expected answer equals, or is contained in, the target content.

    Under x2 such a fixture hands the model a second copy of the
    answer span; the flag keeps echo-completion separable from
    re-reading in the readout.
    """
    expected = fixture.expected.strip()
    return bool(expected) and expected in target_content(fixture)


# skipcq: PYL-W0621 - the name mirrors the frozen payload field it consumes
def build_arm_fixture(fixture: Fixture, arm: str, filler_text: str = "") -> Fixture:
    """A copy of ``fixture`` with the target content surgically rewritten.

    Same id (request seeds stay paired across arms), same group and
    expected; only the last user message's content differs.
    """
    index = target_index(fixture)
    content = fixture.messages[index].content
    if arm == "x2":
        new_content = content + JOINER + content
    elif arm == "pad":
        new_content = content + JOINER + filler_text
    elif arm == "x1":
        new_content = content
    else:
        raise ValueError(f"unknown repetition arm {arm!r}")
    messages = tuple(
        Message(role=m.role, content=new_content if i == index else m.content)
        for i, m in enumerate(fixture.messages)
    )
    return Fixture(id=fixture.id, messages=messages, expected=fixture.expected, group=fixture.group)


def filler_text(reps: int, cut: int) -> str:
    """Whole filler units plus at most one truncated suffix."""
    return FILLER_UNIT * reps + FILLER_UNIT[:cut]


def filler_search_order(unit_token_estimate: int, delta: int) -> list[tuple[int, int]]:
    """Deterministic (reps, cut) order for the rendered-length search.

    Ordered by closeness of the estimated token count to ``delta``
    (ties by fewer reps, then shorter cut), so the first exact
    rendered-length match is well-defined. Capped at
    SIZING_SEARCH_BUDGET entries; the caller verifies each candidate
    on the actual rendered ids — the estimate only orders the search.
    """
    max_reps = max(2, delta // max(1, unit_token_estimate) + 3)
    pairs: list[tuple[int, int, int]] = []
    for reps in range(0, max_reps + 1):
        for cut in range(0, len(FILLER_UNIT) + 1):
            if reps == 0 and cut == 0:
                continue
            estimate = reps * unit_token_estimate + round(
                unit_token_estimate * cut / len(FILLER_UNIT)
            )
            pairs.append((abs(estimate - delta), reps, cut))
    pairs.sort()
    return [(reps, cut) for _dist, reps, cut in pairs[:SIZING_SEARCH_BUDGET]]


def screen_filler(
    expected_by_fixture: dict[str, str],
    boundary_text_by_candidate: dict[str, str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Screen the filler unit against this run's answers and boundaries.

    Returns (screened fixture ids, screened candidate names). A
    fixture is screened when its expected answer could be copied from
    the filler (either string contains the other, case-folded); a
    candidate is screened when its decoded boundary text occurs in
    the filler (the treatment would carry the terminator).
    """
    unit = FILLER_UNIT.casefold()
    screened_fixtures = set()
    for fixture_id, expected in expected_by_fixture.items():
        answer = expected.strip().casefold()
        if answer and (answer in unit or unit.strip() in answer):
            screened_fixtures.add(fixture_id)
    screened_candidates = set()
    for name, boundary_text in boundary_text_by_candidate.items():
        if boundary_text and boundary_text in FILLER_UNIT:
            screened_candidates.add(name)
    return frozenset(screened_fixtures), frozenset(screened_candidates)


def content_region(
    ids: tuple[int, ...], prefix_ids: tuple[int, ...], suffix_ids: tuple[int, ...]
) -> tuple[int, ...] | None:
    """The ids attributable to the target content, or None.

    Render-relative: the ids between the encoded render prefix
    (everything before the target content) and suffix (everything
    after it). Seam merges that eat an affix make the region
    unlocatable — None, never a guess.
    """
    if len(ids) < len(prefix_ids) + len(suffix_ids):
        return None
    if ids[: len(prefix_ids)] != prefix_ids or ids[len(ids) - len(suffix_ids) :] != suffix_ids:
        return None
    return ids[len(prefix_ids) : len(ids) - len(suffix_ids)]


def classify_construction(
    region_x1: tuple[int, ...] | None,
    region_x2: tuple[int, ...] | None,
    joiner_region: tuple[int, ...] | None,
) -> tuple[str, tuple[int, ...]]:
    """Classify the token-level construction of one (candidate, fixture).

    `id_exact` iff the x2 content region is exactly the x1 region,
    the joiner region, and the x1 region again — no cross-seam merge
    anywhere. Otherwise `text_level`, with the seam ids (the tokens
    around the copy junction) recorded when the region is locatable.
    """
    if region_x1 is None or region_x2 is None or joiner_region is None:
        return "text_level", ()
    if region_x2 == region_x1 + joiner_region + region_x1:
        return "id_exact", ()
    junction = len(region_x1)
    seam = region_x2[max(0, junction - 1) : junction + len(joiner_region) + 1]
    return "text_level", seam


def _arm_verdict(record: dict[str, Any], arm: str) -> bool | None:
    """Return the arm verdict."""
    cell = record["arms"][arm]
    if cell is None:
        return None
    return bool(cell["verdict"])


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def aggregate_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-candidate aggregates over ground records (paired contrasts).

    Each contrast pairs only fixtures completed in both arms and not
    vacuous. Rates are None when a contrast has no paired fixture.
    Also computed excluding pre-repeated fixtures (a subpopulation
    estimate, labelled as such by the caller).
    """

    def _contrast(
        recs: list[dict[str, Any]], arm_a: str, arm_b: str, effect_name: str
    ) -> dict[str, Any]:
        """Return the contrast."""
        paired = [
            r
            for r in recs
            if not r["vacuous"]
            and _arm_verdict(r, arm_a) is not None
            and _arm_verdict(r, arm_b) is not None
        ]
        n = len(paired)
        if n == 0:
            return {
                "n_paired": 0,
                f"rate_{arm_a}": None,
                f"rate_{arm_b}": None,
                effect_name: None,
            }
        rate_a = sum(1 for r in paired if _arm_verdict(r, arm_a)) / n
        rate_b = sum(1 for r in paired if _arm_verdict(r, arm_b)) / n
        return {
            "n_paired": n,
            f"rate_{arm_a}": rate_a,
            f"rate_{arm_b}": rate_b,
            effect_name: rate_a - rate_b,
        }

    contrasts: dict[str, Any] = {}
    mismatches: dict[str, bool] = {}
    for arm_a, arm_b, effect_name in CONTRASTS:
        key = f"{arm_a}_vs_{arm_b}"
        contrasts[key] = _contrast(records, arm_a, arm_b, effect_name)
        sets_a = {
            r["fixture"] for r in records if not r["vacuous"] and _arm_verdict(r, arm_a) is not None
        }
        sets_b = {
            r["fixture"] for r in records if not r["vacuous"] and _arm_verdict(r, arm_b) is not None
        }
        mismatches[key] = sets_a != sets_b
    clean_records = [r for r in records if not r["pre_repeated"]]
    contrasts_excluding: dict[str, Any] = {}
    for arm_a, arm_b, effect_name in CONTRASTS:
        key = f"{arm_a}_vs_{arm_b}"
        contrasts_excluding[key] = _contrast(clean_records, arm_a, arm_b, effect_name)
    boundary_rates: dict[str, Any] = {}
    for arm in ARMS:
        cells = [r["arms"][arm] for r in records if r["arms"][arm] is not None]
        boundary_rates[arm] = (
            sum(1 for c in cells if c["boundary_found"]) / len(cells) if cells else None
        )
    delta_signs = {"negative": 0, "zero": 0, "positive": 0}
    for r in records:
        delta = r["delta_tokens"]
        if delta is None:
            continue
        delta_signs["negative" if delta < 0 else "zero" if delta == 0 else "positive"] += 1
    n_pad_evaluable = sum(1 for r in records if r["pad_evaluable"])
    return {
        "n_fixtures": len(records),
        "n_vacuous": sum(1 for r in records if r["vacuous"]),
        "n_pre_repeated": sum(1 for r in records if r["pre_repeated"]),
        "n_echo": sum(1 for r in records if r["echo_fixture"]),
        "n_pad_evaluable": n_pad_evaluable,
        "pad_evaluable_share": (n_pad_evaluable / len(records)) if records else None,
        "delta_sign_counts": delta_signs,
        "contrasts": contrasts,
        "contrasts_excluding_pre_repeated": contrasts_excluding,
        "evaluable_mismatch": mismatches,
        "boundary_found_rates": boundary_rates,
    }


def arm_all_pass(records: list[dict[str, Any]], arm: str) -> bool | None:
    """All-pass over the arm's completed, non-vacuous fixtures.

    None when the arm completed nothing — the caller treats that as
    arm-unavailable, never as a pass or a fail. Selection sets and
    gain denominators are the same population by construction.
    """
    verdicts = [
        _arm_verdict(r, arm)
        for r in records
        if not r["vacuous"] and _arm_verdict(r, arm) is not None
    ]
    if not verdicts:
        return None
    return all(verdicts)
