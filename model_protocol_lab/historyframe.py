"""History-frame probe (cycle 219, research wave-11 rank 3;
fresh review BUILD-as-amended, coordinator #61).

Every search fixture of the run is preceded by the frozen
synthetic history (synthhistory ``history_6``: six scripted
turns, three exchanges). The probe reframes ONLY that
prepended segment: the history turns are rendered once in the
candidate's own framing (the ``own`` arm) and once in each of
two frozen foreign framings (``foreign_chatml``,
``foreign_llama3``), while the fixture's own messages -- the
tail -- plus the prompt-initial bos and the open suffix are
always the candidate's own. A fourth arm, ``verbatim_chatml``,
differs from ``foreign_chatml`` only in the leading bos
string and is descriptive only: its discordance against its
headline twin is the BOS-substitution share.

Binding design pins (docs/HISTORYFRAME_DESIGN.md, the
candidate text as amended by the fresh review's amendments
A1--A10):

* Segment semantics (section 6.3, frozen): a segment function
  over Format fields with the ``is not None`` fallback
  semantics of ``core.Format.render``; repeat_bos follows the
  segment's family at global message indices; the history
  family's bos appears in no headline prompt; the history
  family's open suffix never appears; no separator is
  inserted at the seam. Under A == B the hybrid reproduces
  ``Format.render`` byte-identically.
* Construction pins (amendment A5): the stage takes
  ``build_history_fixtures``' messages and re-wraps them
  under each fixture's original id, original ``expected``,
  and this module's frozen group label; the built fixtures'
  namespaced ids are never stage fixture ids (they would
  change request seeds). The history is free of answer
  vocabulary and task content (synthhistory's discipline);
  its conversational phrasing is common-mode across arms.
* Screens (amendments A9): S1 is the per-arm slice-swap
  identity, checked pre-generation for every arm of every
  fixture -- each hybrid must equal the candidate's native
  render with only the history-segment slice exchanged, and
  the own hybrid must equal the native render byte for byte;
  S2 audits the frozen history texts and the donor/candidate
  framing literals for special-token literals; S3 is the
  budget check. Any screen failure fails the whole stage
  closed before any generation.
* Own-arm claim (amendment A10): the own arm is
  byte-identical to synthhistory's ``history_6`` render of
  the same messages. Verdict-identity with the synthhistory
  STAGE is not claimed: that stage runs namespaced ids, hence
  different request seeds. This stage's baseline is its own
  ``own`` arm.
* Verbatim accounting (amendment A1): when the chatml bos
  equals the candidate's bos, ``verbatim_chatml`` is
  byte-identical to ``foreign_chatml``; the arm is then
  inapplicable for that candidate -- counted and printed,
  never pooled, never read as a BOS share of zero. The
  verbatim construction removes only the LEADING bos of its
  headline twin. Applicability is derived from the recorded
  bos digests, never carried as a bare flag. The
  single-donor asymmetry (there is no ``verbatim_llama3``)
  is printed scope.
* Records (amendment A4): per-arm token-id digests ride in
  every record (negation's ``digest_ids`` pattern); the
  ids-identical gate is derived from digest equality, and
  ``id_length_delta`` is recomputed from the recorded arm
  token counts, never trusted as carried.
* Scorer: the cell verdict is byte-exact conjoined with
  boundary presence (``core.py:797`` via the carrier
  evaluation), no stripping or parsing anywhere on the
  headline path. Discordant pairs decompose as
  ``boundary_only`` (answer digests equal), ``exact_only``
  (digests differ, both boundaries found), or ``mixed``; all
  three counts print beside every classification.
* Classification (per candidate, per headline contrast),
  over paired completed render-divergent ids-divergent
  cases: n < 10 -> ``insufficient``; d = 0 ->
  ``no_discordance_observed``; d < 6 -> ``below_threshold``;
  a == b -> ``discordant_balanced``; else two-sided exact
  McNemar p < 0.05 -> ``discordant_asymmetric``, else
  ``below_threshold``. The n floor of 10 stands as a frozen
  conservative constant (amendment A3: the smallest
  interior-attainable n is 8; the printed operating
  characteristics, derived with the floor in place, are its
  justification). The falsification boolean is
  ``discordant_asymmetric`` alone. Discordance is never
  pooled across candidates or contrasts.
* Family-wise (amendment A2): the expectation is computed
  from the exact at-n random-labeller fire rate of this
  rule, summed over the run's emitted classifications --
  never a frozen constant. At n = 100 over 48 decisions that
  is 1.76 expected false labels, P(at least one) 0.83.
* Operating characteristics: the correlated-null column of
  the round-1 PoC is dropped (amendment A7 -- no reading of
  it is reproducible); the printed bounds stand under every
  executed reading (random fire at most 3.6 percent,
  competent at most 3.1 percent, at every tabulated N). The
  symmetric-jitter grid (amendment A6) is printed in the
  section: fire 0.0004 / 0.0048 / 0.0213 at epsilon 0.01 /
  0.02 / 0.05, D-invariant under the symmetric model;
  one-sided variants are arm-asymmetric perturbations, i.e.
  inside the alternative, and are named as such.
* Residuals (amendment A8): the section names the arm-level
  directional channels -- batch composition and numerics
  correlated with the foreign prompts' length -- beside the
  length/encoding consequences. The label certifies
  string-level discordance on this run, not a mechanism.
* Selection: per headline arm, the conservative selection
  recomputed on the common completed set (canonical-form
  discipline, withheld on mismatched or empty sets); winner
  headlines per contrast in canonical-form's vocabulary.
  Flips against the primary run are printed separately and
  labelled as pooling history presence with frame -- never
  the headline.
* Counts: 4 arms x at most 100 search fixtures x at most 24
  candidates = 9,600 generations legal maximum (headline
  arms 7,200); 2,400 ground records; the house 12 MiB
  records cap.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import Format

__all__ = [
    "ARM_KEYS",
    "CELL_DETAILS",
    "CLASSIFICATIONS",
    "CONTRASTS",
    "DECOMPOSITION_CLASSES",
    "DONOR_FAMILIES",
    "GENERATION_BUDGET",
    "GROUP",
    "HEADLINE_ARMS",
    "HISTORY_CONDITION",
    "HISTORY_TURNS",
    "JITTER_GRID",
    "MAX_CANDIDATES",
    "MAX_GROUND_RECORDS",
    "MAX_RECORDS_BYTES",
    "MAX_SEARCH_FIXTURES",
    "MIN_D",
    "MIN_N",
    "OPERATING_CHARACTERISTICS",
    "PROTOCOL_VERSION",
    "SEGMENT_RULE_TEXT",
    "STAGE_DETAILS",
    "STRATA",
    "WINNER_HEADLINES",
    "aggregate_candidate",
    "arm_selection",
    "carrier_format",
    "classify_history_frame",
    "decompose_pair",
    "digest_ids",
    "digest_text",
    "donor_format",
    "encoding_stratum",
    "familywise_expectation",
    "framing_literals",
    "history_segment",
    "history_segment_digest",
    "hybrid_text",
    "primary_flips",
    "protocol_fingerprint",
    "protocol_object",
    "random_labeller_fire_rate",
    "segment_rule_sha256",
    "shared_fixtures",
    "slice_swap_identity",
    "stage_fixtures",
    "verbatim_slice_identity",
    "verbatim_applicable",
    "verbatim_text",
    "winner_headline",
]

PROTOCOL_VERSION = 1

GROUP = "history_frame"

#: The frozen prepend: synthhistory's history_6 condition
#: (six scripted turns, three exchanges), reused, never
#: re-authored.
HISTORY_CONDITION = "history_6"
HISTORY_TURNS = 6

ARM_KEYS: tuple[str, ...] = (
    "own",
    "foreign_chatml",
    "foreign_llama3",
    "verbatim_chatml",
)
BASELINE_ARM = "own"
#: Arms that carry a headline contrast and a selection
#: condition. ``verbatim_chatml`` is descriptive only.
HEADLINE_ARMS: tuple[str, ...] = ("own", "foreign_chatml", "foreign_llama3")
#: Headline contrasts, each against the ``own`` baseline.
CONTRASTS: tuple[str, ...] = ("foreign_chatml", "foreign_llama3")
#: The frozen foreign donor pair, by catalog family name.
DONOR_FAMILIES: dict[str, str] = {
    "foreign_chatml": "chatml",
    "foreign_llama3": "llama3",
}
VERBATIM_ARM = "verbatim_chatml"
VERBATIM_TWIN = "foreign_chatml"

#: Classification floors, frozen by the design (section 6.6).
#: The d floor is forced by alpha = 0.05 (a unanimous 5-0
#: split gives two-sided p = 0.0625; 6-0 gives 0.03125). The
#: n floor stands as a frozen conservative constant
#: (amendment A3).
MIN_N = 10
MIN_D = 6

#: 24 candidates (run_experiment's operative cap) x 100
#: search fixtures (the experiment fixture limit); four
#: generations per (candidate, fixture).
MAX_CANDIDATES = 24
MAX_SEARCH_FIXTURES = 100
MAX_GROUND_RECORDS = MAX_CANDIDATES * MAX_SEARCH_FIXTURES
GENERATION_BUDGET = len(ARM_KEYS) * MAX_GROUND_RECORDS
#: The house 12 MiB records bound, for uniformity with the
#: sibling probes.
MAX_RECORDS_BYTES = 12 * 1024 * 1024

#: The section classification vocabulary (exactly
#: canonical-form's; section 6.8). The falsification boolean
#: is ``discordant_asymmetric`` alone.
CLASSIFICATIONS: tuple[str, ...] = (
    "insufficient",
    "no_discordance_observed",
    "discordant_balanced",
    "discordant_asymmetric",
    "below_threshold",
)

#: Winner headlines (exactly canonical-form's vocabulary).
WINNER_HEADLINES: tuple[str, ...] = (
    "winner_changed",
    "winner_stable",
    "winner_abstention_asymmetric",
    "winner_undefined",
)

#: Decomposition of a discordant pair (section 6.6).
DECOMPOSITION_CLASSES: tuple[str, ...] = ("boundary_only", "exact_only", "mixed")

CELL_DETAILS: tuple[str, ...] = ("", "arm_context_exceeded", "arm_evaluation_failed")

STAGE_DETAILS: tuple[str, ...] = (
    "cancelled",
    "historyframe_record_budget_exceeded",
    "historyframe_screen_failed",
)

#: Encoding-stratum labels (section 6.5), run-level per
#: foreign family.
STRATA: tuple[str, ...] = ("all_single", "all_fragmented", "mixed")

#: The segment rule, frozen as text: the protocol carries the
#: SHA-256 of this string, so a bare version number never has
#: to stand in for the rule itself.
SEGMENT_RULE_TEXT = (
    "historyframe-segment-v1: seg(F, m, i) = (F.bos if i > 0 "
    "and F.repeat_bos else '') + (F.user_prefix if m.role is "
    "user else F.assistant_prefix) + m.content + (per-role end "
    "of F for m.role, the per-role field when it is not None "
    "else F.turn_end); open(B) = (B.bos if B.repeat_bos else "
    "'') + (B.open_suffix if B.open_suffix is not None else "
    "B.assistant_prefix); hybrid(A, B) = B.bos + seg(A) over "
    "the first 6 messages + seg(B) over the rest + open(B); "
    "verbatim(A, B) = A.bos + hybrid(A, B) with only the "
    "leading B.bos removed. Screens: S1 per-arm slice-swap "
    "identity pre-generation (each hybrid equals the native "
    "B render with only the history-segment slice exchanged; "
    "the verbatim arm exchanges the same slice under its "
    "defining leading-bos substitution; hybrid(B, B) equals "
    "the native render byte for byte); S2 "
    "special-token-literal audit over the frozen history texts "
    "and the donor and candidate framing literals; S3 budget "
    "caps. User fixture contents are never screened: they are "
    "byte-identical across arms, so they cannot manufacture a "
    "frame contrast."
)

#: Operating characteristics of the frozen rule, from the
#: design's PoC5 table as confirmed by the fresh review's
#: exact enumeration (three parties agree). Printed beside
#: every classification; never re-derived at run time. The
#: exact at-n random-labeller fire rate is computed by
#: :func:`random_labeller_fire_rate` and reproduces the
#: tabulated rows (16: 0.0196, 24: 0.0243, 30: 0.0278,
#: 50: 0.0296, 100: 0.0367); the correlated-null column is
#: dropped (amendment A7) -- the bounds below are what print.
OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "random_labeller_fire_bound": 0.036,
    "competent_fire_bound": 0.031,
    "power_25_point_at_30": 0.52,
    "power_25_point_at_100": 0.988,
    "power_50_point_at_30": 0.983,
    "power_50_point_at_100": 1.0,
}

#: The agenda-7 jitter grid (amendment A6): symmetric two-arm
#: jitter at N = 100, exact trinomial / Monte Carlo agreement
#: across three parties. D-invariant under the symmetric
#: model -- the model matching this stage, where both arms
#: are freshly generated in one run.
JITTER_GRID: tuple[dict[str, Any], ...] = (
    {"epsilon": 0.01, "fire_rate": 0.0004},
    {"epsilon": 0.02, "fire_rate": 0.0048},
    {"epsilon": 0.05, "fire_rate": 0.0213},
)


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text (the records' digest form)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_ids(ids: tuple[int, ...]) -> str:
    """SHA-256 hex digest of a render id sequence, so records
    carry the sequence's identity without its contents.
    """
    canonical = ",".join(str(i) for i in ids)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def donor_format(arm: str) -> Format:
    """The frozen donor Format for a foreign arm, from the
    catalog. Raises ValueError on an unknown arm.
    """
    from .catalog import CATALOG

    family = DONOR_FAMILIES.get(arm)
    if family is None:
        raise ValueError("unknown history-frame foreign arm")
    return CATALOG[family]


def _turn_end(fmt: Format, role: str) -> str:
    """The per-role terminator with render's fallback: the
    per-role field when it is not None, else ``turn_end``.
    """
    per_role = fmt.user_turn_end if role == "user" else fmt.assistant_turn_end
    return per_role if per_role is not None else fmt.turn_end


def _segment(fmt: Format, messages: tuple[Any, ...], indices: range) -> str:
    """Return the segment."""
    parts: list[str] = []
    for i in indices:
        message = messages[i]
        if i > 0 and fmt.repeat_bos:
            parts.append(fmt.bos)
        prefix = fmt.user_prefix if message.role == "user" else fmt.assistant_prefix
        parts.extend((prefix, message.content, _turn_end(fmt, message.role)))
    return "".join(parts)


def history_segment(fmt: Format, messages: tuple[Any, ...]) -> str:
    """The history segment of ``messages`` framed by ``fmt``:
    the first ``HISTORY_TURNS`` messages under the frozen
    segment semantics (no leading bos -- the prompt-initial
    bos is the final format's, added by :func:`hybrid_text`).
    """
    return _segment(fmt, messages, range(min(HISTORY_TURNS, len(messages))))


def _open_suffix(final_fmt: Format) -> str:
    """Return the open suffix."""
    suffix = (
        final_fmt.open_suffix if final_fmt.open_suffix is not None else final_fmt.assistant_prefix
    )
    return (final_fmt.bos if final_fmt.repeat_bos else "") + suffix


def hybrid_text(hist_fmt: Format, final_fmt: Format, messages: tuple[Any, ...]) -> str:
    """The hybrid render: the final format's bos, the history
    segment in the history format's framing, the tail in the
    final format's framing, the final format's open suffix.
    Under hist == final this reproduces ``Format.render``
    byte-identically (screen S1).
    """
    k = min(HISTORY_TURNS, len(messages))
    return (
        final_fmt.bos
        + _segment(hist_fmt, messages, range(k))
        + _segment(final_fmt, messages, range(k, len(messages)))
        + _open_suffix(final_fmt)
    )


def verbatim_text(hist_fmt: Format, final_fmt: Format, messages: tuple[Any, ...]) -> str:
    """The descriptive verbatim arm: the history format's bos
    followed by the hybrid with ONLY the leading final-format
    bos removed (amendment A1 -- internal bos occurrences of
    repeat_bos formats are never touched).
    """
    hybrid = hybrid_text(hist_fmt, final_fmt, messages)
    if final_fmt.bos and hybrid.startswith(final_fmt.bos):
        hybrid = hybrid[len(final_fmt.bos) :]
    return hist_fmt.bos + hybrid


def history_segment_digest(fmt: Format, messages: tuple[Any, ...]) -> str:
    """Digest of one format's history segment over
    ``messages`` -- the render-divergence primitive the
    records carry.
    """
    return digest_text(history_segment(fmt, messages))


def slice_swap_identity(
    native_text: str,
    hybrid: str,
    native_segment: str,
    arm_segment: str,
    bos: str,
) -> bool:
    """Amendment A9's strengthened S1: the hybrid equals the
    native render with only the history-segment slice
    exchanged. Assembly-level seam corruption (a separator
    inserted or dropped at the seam, a tail byte altered)
    fails this identity for the affected arm even when the
    diagonal hybrid(B, B) is intact.
    """
    prefix = bos + native_segment
    if not native_text.startswith(prefix):
        return False
    return hybrid == bos + arm_segment + native_text[len(prefix) :]


def verbatim_slice_identity(
    native_text: str,
    verbatim: str,
    native_segment: str,
    arm_segment: str,
    bos: str,
    arm_bos: str,
) -> bool:
    """The verbatim arm's slice identity: the native render
    with only the history-segment slice exchanged and the
    leading bos substituted by the history format's bos --
    the arm's defining difference. Seam corruption fails it
    exactly as it fails the headline identity.
    """
    prefix = bos + native_segment
    if not native_text.startswith(prefix):
        return False
    return verbatim == arm_bos + arm_segment + native_text[len(prefix) :]


def carrier_format(final_fmt: Format) -> Format:
    """The carrier Format holding a hybrid text as a single
    user message: empty framing fields render the message
    verbatim (verified against ``Format.render``).
    """
    from .core import Format

    return Format(final_fmt.name, "", "", "", "")


def verbatim_applicable(own_bos_digest: Any, donor_bos_digest: Any) -> bool:
    """Amendment A1's applicability, derived from digests:
    the verbatim arm is applicable iff the candidate's bos
    differs from the chatml donor's bos (equal bos strings
    make the arm byte-identical to its headline twin).
    """
    return (
        type(own_bos_digest) is str
        and type(donor_bos_digest) is str
        and own_bos_digest != donor_bos_digest
    )


def stage_fixtures(fixtures: tuple[Any, ...]) -> tuple[Any, ...]:
    """Amendment A5's construction: take
    ``build_history_fixtures``' prepended messages and
    re-wrap them under each fixture's ORIGINAL id, original
    ``expected``, and this module's frozen group label. The
    built fixtures' namespaced ids are never used (they
    would change the request seed of every cell).
    """
    from .core import Fixture
    from .synthhistory import build_history_fixtures

    built, _k = build_history_fixtures(fixtures, condition=HISTORY_CONDITION)
    return tuple(
        Fixture(id=original.id, messages=b.messages, expected=original.expected, group=GROUP)
        for original, b in zip(fixtures, built, strict=True)
    )


def framing_literals(fmt: Format) -> tuple[str, ...]:
    """The distinct nonempty framing literals a history
    segment in ``fmt``'s framing is built from: bos, per-role
    prefixes, and the resolved per-role terminators. The open
    suffix is excluded -- the history family's open suffix
    never appears in any prompt.
    """
    values = (
        fmt.bos,
        fmt.user_prefix,
        fmt.assistant_prefix,
        _turn_end(fmt, "user"),
        _turn_end(fmt, "assistant"),
    )
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


def encoding_stratum(backend: Any, fmt: Format) -> dict[str, Any]:
    """The run-level encoding stratum for one foreign family
    (section 6.5): per framing literal, ``token_real`` iff
    the backend encodes it to exactly one id; the stratum is
    ``all_single`` when every literal is token-real,
    ``all_fragmented`` when none is, else ``mixed``. A
    run-time measurement of the checkpoint's tokenizer,
    never a design-time assertion.
    """
    literals: dict[str, bool] = {}
    for literal in framing_literals(fmt):
        literals[literal] = len(backend.encode(literal)) == 1
    if literals and all(literals.values()):
        label = "all_single"
    elif literals and not any(literals.values()):
        label = "all_fragmented"
    elif literals:
        label = "mixed"
    else:
        label = "all_single"
    return {"label": label, "literals": literals}


def segment_rule_sha256() -> str:
    """Digest of the frozen segment-rule text."""
    return digest_text(SEGMENT_RULE_TEXT)


def _history_material_sha256() -> str:
    """Digest of the frozen history transcript (the
    synthhistory turns' roles and contents): the transcript
    is part of the treatment, so the protocol binds it.
    """
    from .synthhistory import SYNTHETIC_TURNS

    canonical = json.dumps(
        [[turn.role, turn.content] for turn in SYNTHETIC_TURNS],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def protocol_object(seed: int, backend_fingerprint: str, fixture_ids: list[str]) -> dict[str, Any]:
    """The frozen instrument description a payload is checked
    against (amendment A4's sibling-pattern protocol): arms,
    the donor pair, the prepend, floors, budgets, the history
    material digest, and the seed / backend / fixture-set
    identity that generated the records.
    """
    return {
        "version": PROTOCOL_VERSION,
        "arm_keys": list(ARM_KEYS),
        "headline_arms": list(HEADLINE_ARMS),
        "contrasts": list(CONTRASTS),
        "donor_families": dict(DONOR_FAMILIES),
        "verbatim_arm": VERBATIM_ARM,
        "verbatim_twin": VERBATIM_TWIN,
        "history_condition": HISTORY_CONDITION,
        "history_turns": HISTORY_TURNS,
        "history_material_sha256": _history_material_sha256(),
        "segment_rule_sha256": segment_rule_sha256(),
        "min_n": MIN_N,
        "min_d": MIN_D,
        "generation_budget": GENERATION_BUDGET,
        "max_ground_records": MAX_GROUND_RECORDS,
        "max_records_bytes": MAX_RECORDS_BYTES,
        "max_search_fixtures": MAX_SEARCH_FIXTURES,
        "max_candidates": MAX_CANDIDATES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
        "fixture_ids": list(fixture_ids),
    }


def protocol_fingerprint(seed: int, backend_fingerprint: str, fixture_ids: list[str]) -> str:
    """SHA-256 over the canonical protocol object."""
    canonical = json.dumps(
        protocol_object(seed, backend_fingerprint, fixture_ids),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _mcnemar_p(a: int, b: int) -> float | None:
    """Two-sided exact McNemar p for discordant counts a, b
    (the tree's ``confidentiality.mcnemar_exact_p``).
    """
    from .confidentiality import mcnemar_exact_p

    return mcnemar_exact_p(a, b)


def classify_history_frame(n: int, own_only: int, foreign_only: int) -> str:
    """The section classification (total function of the
    paired count and the discordant split; section 6.6).

    n < 10 -> ``insufficient`` (withheld, never acquittal);
    d = 0 -> ``no_discordance_observed``; d < 6 ->
    ``below_threshold``; a == b -> ``discordant_balanced``
    (descriptive only); else exact McNemar p < 0.05 ->
    ``discordant_asymmetric`` (the falsification boolean),
    else ``below_threshold`` (an observed null at this
    panel, weak evidence, never invariance).
    """
    if n < MIN_N:
        return "insufficient"
    d = own_only + foreign_only
    if d == 0:
        return "no_discordance_observed"
    if d < MIN_D:
        return "below_threshold"
    if own_only == foreign_only:
        return "discordant_balanced"
    p = _mcnemar_p(own_only, foreign_only)
    if p is not None and p < 0.05:
        return "discordant_asymmetric"
    return "below_threshold"


def decompose_pair(own_cell: dict[str, Any], foreign_cell: dict[str, Any]) -> str:
    """The decomposition class of one discordant pair:
    ``boundary_only`` when the answer digests are equal (the
    discordance is a boundary event), ``exact_only`` when the
    digests differ and both boundaries were found, else
    ``mixed``.
    """
    if own_cell.get("answer_digest") == foreign_cell.get("answer_digest"):
        return "boundary_only"
    if own_cell.get("boundary_found") and foreign_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def random_labeller_fire_rate(n: int) -> float:
    """Exact asymmetric-fire probability of the frozen rule
    under the random labeller (each arm passes independently
    with probability 1/2), by exact enumeration over the
    discordant count and its split. Reproduces the design's
    tabulated rows (n=100 -> 0.0367). Below the n floor the
    classification is always ``insufficient``, so the fire
    rate is 0.
    """
    if type(n) is not int or n < MIN_N:
        return 0.0
    total = 0.0
    for d in range(MIN_D, n + 1):
        # P(exactly d discordant) under independent fair arms:
        # each case is discordant with probability 1/2 (the
        # two arms disagree), concordant otherwise.
        p_d = math.comb(n, d) * (0.5**n)
        for a in range(d + 1):
            b = d - a
            if a == b:
                continue
            p = _mcnemar_p(a, b)
            if p is not None and p < 0.05:
                total += p_d * math.comb(d, a) * (0.5**d)
    return total


def familywise_expectation(rates: list[float]) -> dict[str, Any]:
    """Amendment A2's family-wise accounting over the run's
    emitted classifications: the expected false-label count
    is the sum of the exact at-n random-labeller fire rates,
    and P(at least one) assumes independence across
    decisions. Never a frozen constant.
    """
    expected = sum(rates)
    survival = 1.0
    for rate in rates:
        survival *= 1.0 - rate
    return {
        "decisions": len(rates),
        "expected_false_labels": expected,
        "p_at_least_one": 1.0 - survival,
    }


def _cell(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the cell."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _detail(record: dict[str, Any], key: str) -> str:
    """Return the detail."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(key, "")
    return detail if type(detail) is str else ""


def _rate(numerator: int, denominator: int) -> float | None:
    """Return the rate."""
    return (numerator / denominator) if denominator else None


def _contrast_aggregate(records: list[dict[str, Any]], contrast: str) -> dict[str, Any]:
    """One headline contrast (own vs ``contrast``) over one
    candidate's records: eligibility accounting, the paired
    classification inputs, the decomposition, the ungated
    sensitivity, and the id-divergence readout.
    """
    n_render_identical = 0
    n_ids_identical = 0
    n_same_ids_flips = 0
    n_paired = 0
    own_only = 0
    foreign_only = 0
    own_passes = 0
    foreign_passes = 0
    ungated_n = 0
    ungated_own_only = 0
    ungated_foreign_only = 0
    decomposition: dict[str, int] = dict.fromkeys(DECOMPOSITION_CLASSES, 0)
    deltas: list[int] = []
    for record in records:
        segments = record.get("segment_digests")
        segments = segments if isinstance(segments, dict) else {}
        render_divergent = segments.get(contrast) != segments.get(BASELINE_ARM)
        own_tokens = (record.get("arm_tokens") or {}).get(BASELINE_ARM)
        foreign_tokens = (record.get("arm_tokens") or {}).get(contrast)
        if type(own_tokens) is int and type(foreign_tokens) is int:
            deltas.append(own_tokens - foreign_tokens)
        own_cell = _cell(record, BASELINE_ARM)
        foreign_cell = _cell(record, contrast)
        if own_cell is not None and foreign_cell is not None:
            ungated_n += 1
            if own_cell["verdict"] and not foreign_cell["verdict"]:
                ungated_own_only += 1
            elif foreign_cell["verdict"] and not own_cell["verdict"]:
                ungated_foreign_only += 1
        if not render_divergent:
            n_render_identical += 1
            continue
        id_digests = record.get("id_digests")
        id_digests = id_digests if isinstance(id_digests, dict) else {}
        own_digest = id_digests.get(BASELINE_ARM)
        foreign_digest = id_digests.get(contrast)
        ids_identical = (
            type(own_digest) is str and own_digest != "" and own_digest == foreign_digest
        )
        if ids_identical:
            n_ids_identical += 1
            if (
                own_cell is not None
                and foreign_cell is not None
                and own_cell["verdict"] != foreign_cell["verdict"]
            ):
                # A same-ids verdict flip is a determinism
                # finding, never frame fragility.
                n_same_ids_flips += 1
            continue
        if own_cell is None or foreign_cell is None:
            continue
        n_paired += 1
        if own_cell["verdict"]:
            own_passes += 1
        if foreign_cell["verdict"]:
            foreign_passes += 1
        if own_cell["verdict"] and not foreign_cell["verdict"]:
            own_only += 1
            cls = decompose_pair(own_cell, foreign_cell)
            decomposition[cls] = decomposition.get(cls, 0) + 1
        elif foreign_cell["verdict"] and not own_cell["verdict"]:
            foreign_only += 1
            cls = decompose_pair(own_cell, foreign_cell)
            decomposition[cls] = decomposition.get(cls, 0) + 1
    applicable = n_render_identical < len(records)
    d = own_only + foreign_only
    return {
        "applicable": applicable,
        "n_render_identical": n_render_identical,
        "n_ids_identical": n_ids_identical,
        "n_same_ids_flips": n_same_ids_flips,
        "n_paired": n_paired,
        "own_only": own_only,
        "foreign_only": foreign_only,
        "discordant": d,
        "own_passes_paired": own_passes,
        "foreign_passes_paired": foreign_passes,
        "paired_rate_own": _rate(own_passes, n_paired),
        "paired_rate_foreign": _rate(foreign_passes, n_paired),
        "ungated_n": ungated_n,
        "ungated_own_only": ungated_own_only,
        "ungated_foreign_only": ungated_foreign_only,
        "decomposition": decomposition,
        "classification": (
            classify_history_frame(n_paired, own_only, foreign_only) if applicable else None
        ),
        "falsification": (
            classify_history_frame(n_paired, own_only, foreign_only) == "discordant_asymmetric"
            if applicable
            else False
        ),
        "mcnemar_p": _mcnemar_p(own_only, foreign_only) if d else None,
        "random_fire_rate": random_labeller_fire_rate(n_paired) if applicable else None,
        "id_length_delta": deltas,
        "max_abs_id_delta": max((abs(x) for x in deltas), default=0),
    }


def _verbatim_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The descriptive verbatim readout (amendment A1): when
    the candidate's bos equals the chatml bos the arm is
    byte-identical to its headline twin and the readout is
    inapplicable -- counted, never pooled, never a BOS share
    of zero. When applicable, the discordance between
    ``foreign_chatml`` and ``verbatim_chatml`` over paired
    completed fixtures is the BOS-substitution share.
    """
    applicable = all(
        verbatim_applicable(record.get("own_bos_digest"), record.get("donor_bos_digest"))
        for record in records
    )
    n_paired = 0
    twin_only = 0
    verbatim_only = 0
    if applicable:
        for record in records:
            twin = _cell(record, VERBATIM_TWIN)
            verbatim = _cell(record, VERBATIM_ARM)
            if twin is None or verbatim is None:
                continue
            n_paired += 1
            if twin["verdict"] and not verbatim["verdict"]:
                twin_only += 1
            elif verbatim["verdict"] and not twin["verdict"]:
                verbatim_only += 1
    return {
        "applicable": applicable,
        "n_paired": n_paired,
        "twin_only": twin_only,
        "verbatim_only": verbatim_only,
        "discordant": twin_only + verbatim_only,
    }


def aggregate_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-candidate aggregates over its ground records:
    per-arm completion and attrition, the two headline
    contrasts, and the descriptive verbatim readout. Every
    count carries its denominator; discordance is never
    pooled across candidates -- this function sees one
    candidate's records only.
    """
    arms: dict[str, Any] = {}
    for key in ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        for record in records:
            detail = _detail(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _cell(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": _rate(n_pass, n_completed),
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
        }
    return {
        "arms": arms,
        "contrasts": {contrast: _contrast_aggregate(records, contrast) for contrast in CONTRASTS},
        "verbatim": _verbatim_aggregate(records),
        "operating_characteristics": dict(OPERATING_CHARACTERISTICS),
        "jitter_grid": [dict(row) for row in JITTER_GRID],
        "n_fixtures": len(records),
    }


def shared_fixtures(records_by_candidate: dict[str, list[dict[str, Any]]]) -> frozenset[str]:
    """Fixture ids recorded for every candidate: the
    selection conditions' population (canonical-form
    discipline).
    """
    populations: list[frozenset[str]] = []
    for records in records_by_candidate.values():
        populations.append(frozenset(str(r.get("fixture")) for r in records))
    if not populations:
        return frozenset()
    return frozenset.intersection(*populations)


def _verdict(record: dict[str, Any], key: str) -> bool:
    """One record's verdict for one arm; False when the cell
    did not complete. (On a completed-restricted set this is
    exactly the cell's verdict.)
    """
    cell = _cell(record, key)
    return bool(cell is not None and cell["verdict"])


def _selection_over(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    key: str,
    fixtures: frozenset[str],
) -> dict[str, Any] | None:
    """The conservative selection for one arm restricted to
    ``fixtures``: None when any candidate's completed set on
    that restriction is empty or the restricted completed
    sets differ across candidates (canonical-form's
    ``_selection_over``, mirrored).
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
    """Per-arm selection readout over the shared population
    (canonical-form discipline): withheld when the restricted
    completed sets differ across candidates or are empty.
    """
    shared = shared_fixtures(records_by_candidate)
    own = _selection_over(records_by_candidate, arm, shared) if shared else None
    if own is None:
        return {
            "decision": "unavailable",
            "detail": "selection_withheld_mismatched_sets",
            "winner": None,
            "withheld": True,
        }
    return {"decision": own["decision"], "detail": "", "winner": own["winner"], "withheld": False}


def winner_headline(baseline: Any, other: Any) -> dict[str, Any]:
    """The winner headline for one contrast, as a total
    function of the two arms' selections (canonical-form
    semantics): selection-vs-abstention is NEVER a winner
    change; ``winner_changed`` is True only for a genuine
    winner change, and None when either selection was
    withheld (the comparison is then undefined, not stable).
    """
    if not isinstance(baseline, dict) or not isinstance(other, dict):
        return {
            "headline": "winner_undefined",
            "detail": "selection_missing",
            "winner_changed": None,
        }
    if baseline.get("withheld") is True or other.get("withheld") is True:
        return {
            "headline": "winner_undefined",
            "detail": "selection_withheld_mismatched_sets",
            "winner_changed": None,
        }
    baseline_selected = baseline.get("decision") == "selected"
    other_selected = other.get("decision") == "selected"
    if baseline_selected and other_selected:
        if baseline.get("winner") != other.get("winner"):
            return {"headline": "winner_changed", "detail": "", "winner_changed": True}
        return {"headline": "winner_stable", "detail": "", "winner_changed": False}
    if baseline_selected != other_selected:
        return {"headline": "winner_abstention_asymmetric", "detail": "", "winner_changed": False}
    return {"headline": "winner_undefined", "detail": "", "winner_changed": False}


def primary_flips(
    conditions: dict[str, Any], primary: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Flips of the per-arm selections against the primary
    run (synthhistory's verdict mapping): the run's success
    decisions map onto the selection vocabulary, other
    outcomes compare verbatim; withheld arms never flip.
    These flips pool history presence with frame and are
    printed separately, never as the headline.
    """
    if not isinstance(primary, dict):
        return []
    decision = primary.get("decision")
    winner = primary.get("winner")
    if decision in ("compatible_on_fixture_set", "held_out_failed"):
        verdict = ("selected", winner)
    elif type(decision) is str and decision:
        verdict = (decision, winner)
    else:
        return []
    flips: list[dict[str, Any]] = []
    for arm in HEADLINE_ARMS:
        selection = conditions.get(arm) if isinstance(conditions, dict) else None
        if not isinstance(selection, dict) or selection.get("withheld") is True:
            continue
        if (selection.get("decision"), selection.get("winner")) != verdict:
            flips.append(
                {
                    "arm": arm,
                    "decision_from": verdict[0],
                    "decision_to": selection.get("decision"),
                    "winner_from": verdict[1],
                    "winner_to": selection.get("winner"),
                }
            )
    return flips
