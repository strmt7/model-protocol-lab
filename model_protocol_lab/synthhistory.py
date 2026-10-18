"""Synthetic-history selection stability (research wave-2 idea 5).

The question: does the winning protocol survive when fixtures are asked
after synthetic prior conversation turns -- i.e., is protocol selection
robust to conversational context, or does history flip the winner?

The mechanism is well-established (Anil et al., Anthropic, *Many-shot
Jailbreaking*, Apr 2024: fabricated prior assistant turns steer subsequent
behavior predictably). The open question is whether it moves *protocol
selection* specifically -- which is precisely what this experiment
measures. No prompt-format harness re-runs selection under synthetic
history today.

Design: K fixed, benign, fully deterministic synthetic prior turns are
prepended to every fixture's message list before family rendering --
conditions ``history_2`` (one exchange) and ``history_6`` (three
exchanges). The fixture's ``expected`` answer is unchanged, so the
standard pipeline ``exact_match`` already grades the history-conditioned
run (the same prefix-cancellation argument as the prefill probe: the
synthetic turns are context, not part of the answer). Each candidate is
re-evaluated with ``core.evaluate()`` unchanged, and the conservative
selection rule is re-applied per condition.

Three falsifiable claims, pre-registered here:

1. Synthetic history changes per-candidate pass outcomes for at least
   some (candidate, condition) cells (measured, not asserted).
2. The selected winner flips under synthetic history for at least one
   condition in runs where history matters at the selection level
   (measured per condition against the primary run).
3. Families whose templates use unambiguous per-turn open/close markers
   degrade less under history than families with loose delimiting.
   The labels below are derived *mechanically* from the lab's own
   ``Format`` hypotheses -- a family is "explicit" iff every turn type
   (user and assistant) has a non-blank open marker and a non-blank
   close marker in the catalog entry; otherwise it is "loose". Families
   absent from the catalog are "unknown". This is a correlation readout
   over the lab's own catalog, reported as such -- not a causal claim
   about the real templates.

Everything here is pure: history construction and the delimiter-label
rule need no model.
"""

from __future__ import annotations

from .catalog import CATALOG
from .core import Fixture, Format, Message
from .hierarchy import family_of

#: Synthetic prior conversation turns: three benign, fully deterministic
#: exchanges. Kept deliberately free of any fixture answer vocabulary
#: ("BLUE"/"GREEN"/"RED"/"7"/"NORTH"/"SOUTH") and of instruction-like
#: phrasing, so the turns are pure conversational context, never a
#: second task. Fixed order; conditions take a prefix of this tuple.
SYNTHETIC_TURNS: tuple[Message, ...] = (
    Message("user", "Hello. I have a few short questions for you."),
    Message("assistant", "Hello! I am ready. Please ask your questions one at a time."),
    Message("user", "Thank you. I will keep each question short."),
    Message("assistant", "Understood. I will keep my answers short as well."),
    Message("user", "Let us begin whenever you are ready."),
    Message("assistant", "I am ready. Please go ahead with your first question."),
)

#: History conditions: the number of synthetic turns prepended.
CONDITIONS = ("history_2", "history_6")

#: Synthetic-turn count per condition.
TURN_COUNTS: dict[str, int] = {"history_2": 2, "history_6": 6}

#: Delimiter-explicitness labels.
DELIMITER_LABELS = ("explicit", "loose", "unknown")


def _blank(text: object) -> bool:
    """True for non-strings and for strings with no visible characters."""
    return type(text) is not str or not text.strip()


def validate_history_fixtures(fixtures: object) -> tuple[Fixture, ...]:
    """Validate the base fixtures a history condition is built from.

    Returns the fixtures as a tuple. Raises ``ValueError`` on a
    non-tuple, an empty tuple, or any non-``Fixture`` element.
    """
    if type(fixtures) is not tuple or not fixtures:
        raise ValueError("fixtures must be a nonempty tuple")
    if any(type(case) is not Fixture for case in fixtures):
        raise ValueError("fixtures must contain Fixture values")
    return fixtures


def prepend_history(messages: tuple[Message, ...], k: int) -> tuple[Message, ...]:
    """Prepend the first ``k`` synthetic turns to a message list.

    ``k`` must be a positive even number not exceeding
    ``len(SYNTHETIC_TURNS)`` (whole exchanges only). Raises
    ``ValueError`` otherwise.
    """
    if type(messages) is not tuple or any(type(m) is not Message for m in messages):
        raise ValueError("messages must be a tuple of Message values")
    if type(k) is not int or k <= 0 or k % 2 or k > len(SYNTHETIC_TURNS):
        raise ValueError(f"k must be a positive even number <= {len(SYNTHETIC_TURNS)}")
    return SYNTHETIC_TURNS[:k] + messages


def build_history_fixtures(
    fixtures: tuple[Fixture, ...], *, condition: str
) -> tuple[tuple[Fixture, ...], int]:
    """Build history-conditioned fixtures for one condition.

    Prepends the condition's synthetic turns to every fixture's message
    list. Fixture ids and groups are namespaced
    (``synthhistory/<condition>/<id>``) so history runs never collide
    with primary fixture ids; ``expected`` is unchanged.

    Returns ``(fixtures, n_turns)``. Raises ``ValueError`` on a bad
    condition or invalid fixtures.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    clean = validate_history_fixtures(fixtures)
    k = TURN_COUNTS[condition]
    built: list[Fixture] = []
    for case in clean:
        built.append(
            Fixture(
                id=f"synthhistory/{condition}/{case.id}",
                messages=prepend_history(case.messages, k),
                expected=case.expected,
                group=f"synthhistory-{condition}",
            )
        )
    return tuple(built), k


def delimiter_explicitness(fmt: Format) -> str:
    """Structural delimiter-explicitness label for one catalog ``Format``.

    A family is ``"explicit"`` iff every turn type has a non-blank open
    marker *and* a non-blank close marker in the lab's own ``Format``
    hypothesis (user close falls back from ``user_turn_end`` to
    ``turn_end``; likewise for the assistant). Otherwise ``"loose"``.
    A bare newline is not an unambiguous delimiter, so whitespace-only
    markers count as absent. Raises ``ValueError`` on a non-``Format``.
    """
    if type(fmt) is not Format:
        raise ValueError("fmt must be a Format")
    user_close = fmt.user_turn_end if fmt.user_turn_end is not None else fmt.turn_end
    assistant_close = fmt.assistant_turn_end if fmt.assistant_turn_end is not None else fmt.turn_end
    if (
        _blank(fmt.user_prefix)
        or _blank(user_close)
        or _blank(fmt.assistant_prefix)
        or _blank(assistant_close)
    ):
        return "loose"
    return "explicit"


def family_delimiter_explicitness(family: object) -> str:
    """Delimiter-explicitness label for a catalog family name.

    Applies :func:`delimiter_explicitness` to the family's catalog
    ``Format``. Returns ``"unknown"`` for non-string inputs and for
    families absent from the catalog. Never raises.
    """
    if type(family) is not str or not family:
        return "unknown"
    fmt = CATALOG.get(family)
    if fmt is None:
        return "unknown"
    return delimiter_explicitness(fmt)


def candidate_delimiter_label(candidate_name: str) -> str:
    """Delimiter label for a candidate (``family`` or ``family/variant``).

    The family is the part before the first ``/`` (see
    :func:`hierarchy.family_of`). Raises ``ValueError`` on a non-string
    or blank name.
    """
    return family_delimiter_explicitness(family_of(candidate_name))
