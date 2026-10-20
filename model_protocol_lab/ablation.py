"""Template field ablation: which structural field drives the verdict.

Monday-morning utility: a platform engineer asks not "which template won"
but "what *about* it won -- the BOS token, the role labels, or the turn
terminator?" -- because only the causal field transfers to the next template
they write. This probe answers it: starting from a candidate template, it
replaces one ``Format`` structural field at a time with its neutral value
and re-applies the conservative selection bar per ablation arm, reporting
per-field verdict-flip attribution.

Design (falsifiable): each arm is a single-field swap to neutral --
``bos``/``user_prefix``/``assistant_prefix``/``turn_end`` to ``""``,
``repeat_bos`` to ``False``, ``user_turn_end``/``assistant_turn_end``/
``open_suffix`` to ``None`` (fall back to the shared value). If a field
already holds its neutral value the arm is ``not_applicable`` (a no-op
swap proves nothing). Single-field (main-effects) only: pairwise
interactions (8 fields -> 28 pairs x candidates x fixtures) are explicitly
out of scope and reported as such.

Honest scope: an arm that flips the verdict shows the field is
*load-bearing for that verdict on those fixtures with those weights* --
causal within the measured triple (template, fixtures, backend), not a
universal claim about the field. An arm with ~zero marginal attribution
across all candidates is a publishable negative ("no single field explains
the winner"); the probe reports ``any_attribution`` so the negative result
is explicit, not silent.

Everything here is pure (no backend, no I/O); the stage that runs arms
lives in ``experiments.ablation_stage``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from .core import Format

#: Structural fields of ``Format`` that ablation may swap. ``name`` is an
#: identifier, not a structural field, and is never ablated.
ABLATION_FIELDS: tuple[str, ...] = (
    "bos",
    "user_prefix",
    "assistant_prefix",
    "turn_end",
    "repeat_bos",
    "user_turn_end",
    "assistant_turn_end",
    "open_suffix",
)

#: Neutral value per field: the value the field takes when it "contributes
#: nothing". Text fields -> "", repeat_bos -> False, optional overrides ->
#: None (fall back to the shared turn_end / assistant_prefix).
NEUTRAL_VALUES: dict[str, Any] = {
    "bos": "",
    "user_prefix": "",
    "assistant_prefix": "",
    "turn_end": "",
    "repeat_bos": False,
    "user_turn_end": None,
    "assistant_turn_end": None,
    "open_suffix": None,
}


@dataclass(frozen=True)
class AblationArm:
    """One single-field ablation arm: ``field`` replaced by its neutral value."""

    field: str
    original: Any
    ablated: Format


def neutral_value(field: str) -> Any:
    """Neutral value for an ablatable field. Raises on unknown fields."""
    if type(field) is not str or field not in NEUTRAL_VALUES:
        raise ValueError(f"unknown ablation field: {field!r}")
    return NEUTRAL_VALUES[field]


def ablate_format(base: Format, field: str) -> Format:
    """Return ``base`` with ``field`` replaced by its neutral value.

    Raises ``ValueError`` on unknown fields or non-``Format`` bases.
    """
    if type(base) is not Format:
        raise ValueError("base must be a Format")
    neutral = neutral_value(field)  # validates the field name
    return dataclasses.replace(base, **{field: neutral})


def ablation_arms(base: Format) -> tuple[AblationArm, ...]:
    """Applicable ablation arms for ``base``, in ``ABLATION_FIELDS`` order.

    A field whose current value already equals the neutral value yields no
    arm (the swap would be a no-op); such fields are reported separately by
    :func:`inapplicable_fields`.
    """
    if type(base) is not Format:
        raise ValueError("base must be a Format")
    arms: list[AblationArm] = []
    for field in ABLATION_FIELDS:
        original = getattr(base, field)
        if original != NEUTRAL_VALUES[field]:
            arms.append(
                AblationArm(
                    field=field,
                    original=original,
                    ablated=ablate_format(base, field),
                )
            )
    return tuple(arms)


def inapplicable_fields(base: Format) -> tuple[str, ...]:
    """Fields already at their neutral value (ablation would be a no-op)."""
    if type(base) is not Format:
        raise ValueError("base must be a Format")
    return tuple(
        field for field in ABLATION_FIELDS if getattr(base, field) == NEUTRAL_VALUES[field]
    )


def _check_verdicts(verdicts: tuple[bool, ...], name: str) -> tuple[bool, ...]:
    if type(verdicts) is not tuple or any(type(v) is not bool for v in verdicts):
        raise ValueError(f"{name} must be a tuple of bool")
    return verdicts


def summarize_arm(
    base_verdicts: tuple[bool, ...],
    arm_verdicts: tuple[bool, ...],
    fixture_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Compare one arm's per-fixture verdict vector against the base.

    ``base_verdicts``/``arm_verdicts`` are parallel bool tuples (True =
    pass, i.e. exact_match and boundary_found) in ``fixture_ids`` order.
    Returns flip counts in both directions, the flipped fixture ids, and
    whether the arm still clears the conservative selection bar (passes
    every fixture).
    """
    base_verdicts = _check_verdicts(base_verdicts, "base_verdicts")
    arm_verdicts = _check_verdicts(arm_verdicts, "arm_verdicts")
    if type(fixture_ids) is not tuple or any(type(i) is not str for i in fixture_ids):
        raise ValueError("fixture_ids must be a tuple of str")
    if not (len(base_verdicts) == len(arm_verdicts) == len(fixture_ids)):
        raise ValueError("verdict vectors and fixture_ids must be parallel")
    flipped: list[str] = []
    pass_to_fail = 0
    fail_to_pass = 0
    for fid, base_v, arm_v in zip(fixture_ids, base_verdicts, arm_verdicts, strict=True):
        if base_v != arm_v:
            flipped.append(fid)
            if base_v and not arm_v:
                pass_to_fail += 1
            else:
                fail_to_pass += 1
    return {
        "n_fixtures": len(fixture_ids),
        "n_flips": len(flipped),
        "pass_to_fail": pass_to_fail,
        "fail_to_pass": fail_to_pass,
        "flipped_fixture_ids": tuple(flipped),
        # The conservative selection bar: the arm passes every fixture.
        "selection_bar_held": all(arm_verdicts),
    }
