"""Position sensitivity: primacy/recency as a template-family property (research wave-3 idea 1).

The question: when the correct answer of a multiple-choice fixture is
moved between the first, middle, and last option positions, does the
task verdict change -- and does the rate vary by chat template family?

Source: arXiv:2507.13949 ("Exploiting Primacy Effect To Improve Large
Language Models", Jul 2025) -- fine-tuned LLMs pick the first listed
option far more often than chance in MCQA; corroborated by
arXiv:2512.02665 (primacy effect in multi-document summarization). Those
studies vary prompt *text*; the lab varies the *chat template* as the
independent variable while holding model, task, and decoder fixed --
an interaction surface no other harness measures.

Design: synthetic multiple-choice fixtures. Each spec has a question, a
small set of option strings, and the correct option's text. For one
condition the correct option is placed first/middle/last among the
options (the distractor order is a seeded shuffle, so layouts are
deterministic for a given seed); the fixture's ``expected`` is always
the correct option's *text*, so the pipeline's ``exact_match`` grades
the run unchanged. A model that always emits the first-listed option
scores 1.0 under ``correct_first`` and 0.0 under ``correct_last`` --
a primacy index of +1.0.

Three falsifiable claims, pre-registered here:

1. For a fixed task, permuting which position holds the correct answer
   changes the task verdict at a measurable rate (the selection rule
   re-applied per condition picks a different winner), or it does not
   (flip rate ~= 0 across all conditions).
2. The per-candidate primacy index (P(correct | first) - P(correct |
   last)) varies across template families rather than being constant.
3. The condition effect is not an artifact of distractor ordering:
   for a fixed seed the correct placement is invariant under
   distractor shuffling (pinned by tests).

Everything here is pure: spec validation, seeded layout, fixture
construction, and the index arithmetic need no model.
"""

from __future__ import annotations

import random
from typing import cast

from .core import Fixture, Message

#: Conditions: which position holds the correct option in the permuted
#: fixture layout.
CONDITIONS = ("correct_first", "correct_middle", "correct_last")

#: Option labels used in fixture rendering; a fixed vocabulary.
LABELS = ("A", "B", "C", "D")

#: Synthetic multiple-choice fixture specifications. Each spec is a
#: benign, fully deterministic question: the options are distinct
#: nonblank strings, exactly one of which is the correct answer, and
#: the prompt asks for the correct option's text verbatim so grading is
#: the pipeline's exact string comparison. Questions are chosen so the
#: correct option is unambiguous.
POSITION_SPECS: tuple[dict[str, tuple[str, ...] | str], ...] = (
    {
        "id": "capital",
        "question": "Which of these is a European capital?",
        "options": ("Lisbon", "Oslo", "Lima", "Quito"),
        "correct": "Oslo",
    },
    {
        "id": "citrus",
        "question": "Which of these is a citrus fruit?",
        "options": ("kiwi", "mango", "lemon", "plum"),
        "correct": "lemon",
    },
    {
        "id": "bird",
        "question": "Which of these is a bird?",
        "options": ("falcon", "hound", "salmon", "lizard"),
        "correct": "falcon",
    },
    {
        "id": "primary-color",
        "question": "Which of these is a primary color?",
        "options": ("azure", "red", "emerald", "mauve"),
        "correct": "red",
    },
    {
        "id": "prime",
        "question": "Which of these is a prime number?",
        "options": ("4", "6", "7", "9"),
        "correct": "7",
    },
    {
        "id": "precious-metal",
        "question": "Which of these is a precious metal?",
        "options": ("iron", "copper", "gold", "tin"),
        "correct": "gold",
    },
    {
        "id": "triangle",
        "question": "Which of these shapes has exactly three sides?",
        "options": ("square", "hexagon", "triangle", "circle"),
        "correct": "triangle",
    },
    {
        "id": "island-capital",
        "question": "Which of these is the capital of an island nation?",
        "options": ("Oslo", "Reykjavik", "Lima", "Madrid"),
        "correct": "Reykjavik",
    },
)

_SPEC_KEYS = ("id", "question", "options", "correct")


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def validate_position_specs(specs: object) -> tuple[dict[str, object], ...]:
    """Validate synthetic multiple-choice fixture specifications.

    Returns the specs as a tuple of plain dicts with ``options`` as a
    tuple of strings. Raises ``ValueError`` on any malformed spec: not a
    dict, missing/blank fields, fewer than two or more than four
    options, duplicate/blank option strings, duplicate ids, or a
    ``correct`` value that is not exactly one of the options.
    """
    if type(specs) is not tuple or not specs:
        raise ValueError("specs must be a nonempty tuple")
    seen: set[str] = set()
    clean: list[dict[str, object]] = []
    for spec in specs:
        if type(spec) is not dict:
            raise ValueError("each spec must be a dict")
        spec_id = spec.get("id")
        question = spec.get("question")
        if type(spec_id) is not str or not spec_id.strip():
            raise ValueError("spec field 'id' must be a nonblank string")
        if type(question) is not str or not question.strip():
            raise ValueError("spec field 'question' must be a nonblank string")
        options = spec.get("options")
        if type(options) is not tuple or not 2 <= len(options) <= len(LABELS):
            raise ValueError("spec field 'options' must be a tuple of 2-4 strings")
        if any(type(opt) is not str or not opt.strip() for opt in options):
            raise ValueError("every option must be a nonblank string")
        if len(set(options)) != len(options):
            raise ValueError(f"spec {spec_id!r}: options must be distinct")
        correct = spec.get("correct")
        if type(correct) is not str or correct not in options:
            raise ValueError(f"spec {spec_id!r}: 'correct' must be exactly one of the options")
        if spec_id in seen:
            raise ValueError(f"duplicate spec id {spec_id!r}")
        seen.add(spec_id)
        clean.append(
            {
                "id": spec_id,
                "question": question,
                "options": tuple(options),
                "correct": correct,
            }
        )
    return tuple(clean)


def layout_options(
    options: tuple[str, ...], correct: str, *, condition: str, seed: int
) -> tuple[str, ...]:
    """Permute option layout for one condition (pure, seeded).

    The distractors (options minus the correct one) are shuffled with
    the seeded RNG; the correct option is then placed at index 0
    (``correct_first``), ``len // 2`` (``correct_middle``), or
    ``len - 1`` (``correct_last``). The correct placement is invariant
    under the distractor shuffle -- the shuffle only reorders the other
    options. Options must be distinct strings: duplicates would silently
    shorten the layout (the distractor filter removes every copy), so
    they are rejected fail-closed. Raises ``ValueError`` on bad input.
    """
    if type(options) is not tuple or not options:
        raise ValueError("options must be a nonempty tuple")
    if any(type(opt) is not str for opt in options):
        raise ValueError("options must contain only strings")
    if len(set(options)) != len(options):
        raise ValueError("options must be distinct strings")
    if type(correct) is not str or correct not in options:
        raise ValueError("correct must be one of the options")
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    rng = random.Random(seed)  # noqa: S311 # nosec B311
    distractors = [opt for opt in options if opt != correct]
    rng.shuffle(distractors)
    if condition == "correct_first":
        target = 0
    elif condition == "correct_middle":
        target = len(options) // 2
    else:
        target = len(options) - 1
    layout = distractors[:target] + [correct] + distractors[target:]
    return tuple(layout)


def render_options(question: str, layout: tuple[str, ...]) -> str:
    """Render the question plus the option layout as fixture text."""
    if type(question) is not str or not question.strip():
        raise ValueError("question must be a nonblank string")
    if type(layout) is not tuple or not 2 <= len(layout) <= len(LABELS):
        raise ValueError("layout must be a tuple of 2-4 option strings")
    if any(type(opt) is not str for opt in layout):
        raise ValueError("layout must contain only strings")
    lines = [question, "", "Options:"]
    # Lengths are validated above (2-4); zip stops at the shorter side,
    # which is always the layout here (LABELS has 4).
    for label, option in zip(LABELS, layout, strict=False):
        lines.append(f"{label}. {option}")
    lines.extend(("", "Reply with exactly the correct option text."))
    return "\n".join(lines)


def build_position_fixtures(
    specs: tuple[dict[str, object], ...], *, condition: str, seed: int
) -> tuple[Fixture, ...]:
    """Build permuted multiple-choice fixtures for one condition.

    Every fixture's ``expected`` is the correct option's *text*, so the
    pipeline's ``exact_match`` grades the run unchanged across
    conditions. Raises ``ValueError`` on invalid specs/condition/seed.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    clean = validate_position_specs(specs)
    fixtures: list[Fixture] = []
    for spec in clean:
        options = cast("tuple[str, ...]", spec["options"])  # validated above
        layout = layout_options(options, str(spec["correct"]), condition=condition, seed=seed)
        text = render_options(str(spec["question"]), layout)
        fixtures.append(
            Fixture(
                id=f"position/{condition}/{spec['id']}",
                messages=(Message("user", text),),
                expected=str(spec["correct"]),
                group=f"position-{condition}",
            )
        )
    return tuple(fixtures)


def primacy_index(n_first: int, correct_first: int, n_last: int, correct_last: int) -> float | None:
    """Per-candidate primacy index: P(correct | first) - P(correct | last).

    Each rate is correct/total over the fixtures of that condition.
    Returns ``None`` when either condition has no fixtures or the
    counts are malformed (never raises on bad input; the caller decides
    the fail-closed detail code). The index is bounded in [-1, 1].
    """
    for value in (n_first, correct_first, n_last, correct_last):
        if type(value) is not int or value < 0:
            return None
    if correct_first > n_first or correct_last > n_last:
        return None
    if n_first == 0 or n_last == 0:
        return None
    return correct_first / n_first - correct_last / n_last
