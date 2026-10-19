"""Fake-turn history-tampering robustness per template family.

The question: a forged prior turn (fake assistant message asserting a
wrong answer, or fake system instruction contradicting the task)
injected into user content flips task verdicts differently depending
on *whose* delimiters it wears. "Hidden in Plain Sight"
(arXiv:2405.20234) shows chat-history tampering success is
template-token-set-dependent: Vicuna/OpenAI tokens transfer at >=85%
across models while InternLM's own tuning resists others at 82%
(their Table 4). That paper varies the *model* with a fixed attacker
token set; the lab fixes the model and varies the *template* -- the
exact complement -- and additionally measures cross-family delimiter
transfer, which neither the paper nor any harness measures.

Design: for each search fixture, a forged turn is appended to the
last user message's content (exactly the deliforge payload builders:
:func:`deliforge.build_forged_payload` over the same fixed
``MARKERS`` list), but here the independent variable is the forged
turn's delimiter *provenance*, with encoding held at the family's
native form:

* ``same`` -- the forged marker string also occurs as a genuine
  delimiter in the rendering template's own benign render, i.e. the
  forged turn wears the family's own delimiters;
* ``cross`` -- the forged marker is one of the lab's fixed
  chat-template control tokens (each real in at least one catalog
  family) but does not occur as a genuine delimiter in this
  template's render, i.e. the forged turn wears another family's
  delimiters.

The verdict flip rate of the forged arm against the native search
verdicts is measured per (candidate, payload, marker) cell, and
``transfer_gap = flip_rate(cross) - flip_rate(same)`` (in [-1, 1])
is reported per candidate: a positive gap means foreign delimiters
flip verdicts more often than the family's own.

Boundary with deliforge (wave-4 idea 1): deliforge varies the
*encoding* of fixed forged text (reserved single-id vs subword of
identical decoded text); faketurn varies the *delimiter provenance*
(same-family vs cross-family marker text) at the family's native
encoding. Different falsifiable claims, shared payload builders.

Pre-registered claims (falsifiable):

1. A forged prior turn flips the task verdict at the same rate across
   all template families, and cross-family delimiter transfer shows
   no structure -- i.e. ``transfer_gap`` is ~0 and flip rates do not
   cluster by template token set. *Proved wrong if* flip rates
   cluster by token set (the paper's Table 4 pattern) or the gap is
   systematically non-zero.
2. Provenance is operational and checkable: ``same`` iff the marker
   string occurs in the candidate's benign render.
3. Every forged payload carries its marker exactly twice and its
   unique ref exactly once in the rendered prompt; a violation fails
   the cell closed, never a phantom measurement.

Everything here is pure: fixture building, provenance, and gap
aggregation need no model.
"""

from __future__ import annotations

from collections.abc import Sequence

from .core import Fixture, Message
from .deliforge import (
    MARKERS,
    PAYLOADS,
    _decoy_answer,
    build_forged_payload,
    forged_target_index,
    validate_marker_index,
    validate_payload,
)

#: Token-provenance labels, in canonical order. ``same``: the forged
#: marker is also a genuine delimiter of the rendering template;
#: ``cross``: it is a real chat-template control token from another
#: family's token set.
SOURCES = ("same", "cross")

#: Detail codes for unsupported measurement cells (fail-closed, never
#: a phantom flip rate).
DETAIL_NO_PROVENANCE = "no_provenance"
DETAIL_MARKER_COUNT_MISMATCH = "marker_count_mismatch"
DETAIL_REF_MISSING = "ref_missing"
DETAIL_CELL_FAILED = "cell_failed"


def validate_source(source: object) -> str:
    """Validate a token-provenance label; returns it. Raises ``ValueError``."""
    if source not in SOURCES:
        raise ValueError(f"unknown faketurn source {source!r}")
    return str(source)


def build_faketurn_fixtures(
    cases: Sequence[Fixture],
    *,
    payload: str,
    marker_index: int,
) -> tuple[Fixture, ...]:
    """Build forged fixtures from search fixtures for one payload x marker.

    Mirrors :func:`deliforge.build_forged_fixtures` but tags the
    ``faketurn`` experiment: the forged turn is appended to the last
    user message's content (falling back to the last message), the
    fixture's ``expected`` is unchanged so the forged run is graded
    against the true answer, and each fixture carries a unique
    ``(ref: faketurn/...)`` marker so scripted backends can key their
    policy deterministically (inert for real models). The forged
    payload contains the marker string exactly twice. Raises
    ``ValueError`` on bad input.
    """
    validate_payload(payload)
    validate_marker_index(marker_index)
    if type(cases) is not tuple or not cases or any(type(c) is not Fixture for c in cases):
        raise ValueError("cases must be a nonempty tuple of Fixture")
    marker = MARKERS[marker_index]
    decoy = _decoy_answer(cases)
    forged_text = build_forged_payload(payload, marker, decoy)
    fixtures: list[Fixture] = []
    for case in cases:
        messages = case.messages
        target = forged_target_index(messages)
        new_messages = tuple(
            Message(
                role=message.role,
                content=(
                    f"{message.content} {forged_text}\n"
                    f"(ref: faketurn/{payload}/{marker_index}/{case.id})"
                    if i == target
                    else message.content
                ),
            )
            for i, message in enumerate(messages)
        )
        fixtures.append(
            Fixture(
                id=f"faketurn/{payload}/{marker_index}/{case.id}",
                messages=new_messages,
                expected=case.expected,
                group="faketurn",
            )
        )
    return tuple(fixtures)


def marker_provenance(benign_rendered: object, marker: object) -> str | None:
    """Classify a forged marker's token provenance for one candidate.

    ``"same"`` when ``marker`` occurs in the candidate's benign
    rendered prompt (it is also a genuine delimiter of this
    template); ``"cross"`` when it does not (the marker is a fixed
    known chat-template control token from another family's token
    set). Returns None (fail-closed) on non-string input. Never
    raises.
    """
    if type(benign_rendered) is not str or type(marker) is not str or not marker:
        return None
    return "same" if marker in benign_rendered else "cross"


def payload_ref(fixture_id: object, payload: object, marker_index: object) -> str | None:
    """The unique ref string a forged fixture must carry exactly once.

    Returns ``"(ref: faketurn/<payload>/<marker_index>/<fixture_id>)"``
    for valid inputs, else None (fail-closed). Never raises.
    """
    if type(fixture_id) is not str or not fixture_id:
        return None
    if type(payload) is not str or payload not in PAYLOADS:
        return None
    if (
        type(marker_index) is not int
        or isinstance(marker_index, bool)
        or not 0 <= marker_index < len(MARKERS)
    ):
        return None
    return f"(ref: faketurn/{payload}/{marker_index}/{fixture_id})"


def payload_intact(rendered_text: object, ref: object, marker: object) -> bool:
    """True iff the forged payload survived rendering intact.

    Requires the unique ``ref`` to occur exactly once in
    ``rendered_text`` (the render neither dropped nor duplicated the
    forged payload) and the forged marker to occur at least twice
    (the builders pin exactly two forged occurrences; genuine
    template markers can only add more). Fail-closed: False on any
    malformed input. Never raises.
    """
    if type(rendered_text) is not str or type(ref) is not str or not ref:
        return False
    if type(marker) is not str or not marker:
        return False
    # str.find is literal (never regex) and cannot raise on exact strs.
    first = rendered_text.find(ref)
    if first == -1:
        return False
    if rendered_text.find(ref, first + 1) != -1:
        return False
    return rendered_text.count(marker) >= 2


def transfer_gap(
    flip_rate_cross: float | None,
    flip_rate_same: float | None,
) -> float | None:
    """Cross-minus-same flip-rate gap, in [-1, 1].

    None when either rate is unavailable. Raises ``ValueError`` on
    non-rate inputs.
    """
    for rate in (flip_rate_cross, flip_rate_same):
        if rate is not None and (type(rate) not in (int, float) or not 0.0 <= rate <= 1.0):
            raise ValueError(f"flip rates must be in [0, 1], got {rate!r}")
    if flip_rate_cross is None or flip_rate_same is None:
        return None
    return float(flip_rate_cross) - float(flip_rate_same)
