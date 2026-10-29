"""Stop-cause termination audit (research wave-3 idea 3).

The question: the lab's boundary audit (cycle 134) measures *where* a
response ends, but treats stop-cause as one channel. Termination is a
learned behavior, not a guaranteed mechanism (arXiv:2407.20042,
"When to Stop?"): an LLM may emit its end-of-turn token, hit a
server-side stop string, or run to the token cap -- and the serving
stack conflates the first two into one finish reason (llama-cpp
"finish reason 'stop'" covers both the EOS token and the stop-string
substring match). The falsifiable claim: termination *cause*
distributions are identical across template families. Proved wrong if
some families systematically run to max-length or bleed past their own
end-of-turn token while others terminate cleanly -- i.e. the template's
own stop tokens leak into measured verdicts (an answer judged "wrong"
because generation never stopped, not because the model answered
wrong).

Honesty of measurement: the lab's backend contract deliberately
generates *through* EOS to the token cap ("Never stop at EOS or
candidate boundaries" in ``core.Backend``). Every cause below is
therefore post-hoc classification of what a stop-aware engine *would
have* done given the recorded token ids -- the same honest emulation
stance as cycle 137's ``eos_appended`` cell. The stage never claims an
engine stopped early; it reports which mechanism would have fired
first, where, and how many tokens the lab's non-stopping decoder
emitted past that point ("over-generation bleed" no real engine
would emit).

Pre-registered claims (falsifiable):

1. ``classify_cause`` precedence is fixed: the earliest-firing
   mechanism wins; a stop-string firing in the same step as an EOS
   token classifies ``stop_string`` (mirrors the vLLM port notes in
   ``backends.py``: stop string beats EOS in the same step).
2. ``stop_unknown`` is fail-closed: when neither the family's EOT id
   nor a stop text is available, the observation records
   ``stop_unknown`` rather than guessing a cause.
3. ``cause_rates`` always sums to 1.0 over a nonempty cause list
   (pinned by Hypothesis).

Everything here is pure: id-list classification and rate aggregation
need no model and no tokenizer.
"""

from __future__ import annotations

#: Termination-cause vocabulary for one generation.
CAUSES = ("eos_id", "stop_string", "max_length", "stop_unknown")

#: Detail recorded when no stop mechanism could be evaluated.
STOP_UNKNOWN = "stop_unknown"


def first_position(output_ids: object, token_id: int) -> int | None:
    """First index of ``token_id`` in ``output_ids``, or ``None``.

    Raises ``ValueError`` when ``output_ids`` is not a tuple of
    nonnegative ints or ``token_id`` is not a nonnegative int.
    """
    if type(token_id) is not int or token_id < 0:
        raise ValueError("token_id must be a nonnegative int")
    if type(output_ids) is not tuple or any(
        type(token) is not int or token < 0 for token in output_ids
    ):
        raise ValueError("output_ids must be a tuple of nonnegative ints")
    for index, token in enumerate(output_ids):
        if token == token_id:
            return index
    return None


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def classify_cause(
    *,
    output_len: int,
    max_new_tokens: int,
    eos_first: int | None,
    stop_first: int | None,
    eos_known: bool,
    stop_known: bool,
) -> str:
    """Classify one generation's termination cause.

    ``eos_first`` / ``stop_first`` are the 0-based indices of the first
    EOT token / stop-string firing token (``None`` when that mechanism
    did not fire). ``eos_known`` / ``stop_known`` say whether the
    mechanism could be evaluated at all. Returns one of
    :data:`CAUSES`:

    - ``"stop_unknown"`` when neither mechanism was evaluable
      (fail-closed: no guessing);
    - the earliest-firing mechanism otherwise, with a same-step tie
      going to ``"stop_string"`` (stop string beats EOS in the same
      step, per the vLLM port notes in ``backends.py``);
    - ``"max_length"`` when nothing fired and the cap was reached;
    - ``"stop_unknown"`` defensively when nothing fired and the cap
      was not reached (unreachable for completed lab observations).

    Raises ``ValueError`` on malformed inputs.
    """
    if type(output_len) is not int or output_len < 0:
        raise ValueError("output_len must be a nonnegative int")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive int")
    for name, value in (("eos_first", eos_first), ("stop_first", stop_first)):
        if value is not None and (type(value) is not int or value < 0 or value >= output_len):
            raise ValueError(f"{name} must be None or a valid index into the output")
    if type(eos_known) is not bool or type(stop_known) is not bool:
        raise ValueError("eos_known and stop_known must be bools")
    if not eos_known and not stop_known:
        return STOP_UNKNOWN
    fired: list[tuple[int, str]] = []
    if eos_known and eos_first is not None:
        fired.append((eos_first, "eos_id"))
    if stop_known and stop_first is not None:
        fired.append((stop_first, "stop_string"))
    if fired:
        # Earliest firing step wins; on a same-step tie the stop
        # string wins (stop string beats EOS in the same step, per the
        # vLLM port notes in backends.py).
        best = min(fired, key=lambda item: (item[0], 0 if item[1] == "stop_string" else 1))
        return best[1]
    if output_len >= max_new_tokens:
        return "max_length"
    return STOP_UNKNOWN


def over_generated_tokens(output_len: int, eos_first: int | None) -> int:
    """Tokens emitted after the first EOT token (the bleed).

    A stop-aware engine truncates at the first EOT token; every token
    after it is output the lab's non-stopping decoder emits that no
    real engine would. Returns 0 when no EOT token was emitted.
    Raises ``ValueError`` on malformed inputs.
    """
    if type(output_len) is not int or output_len < 0:
        raise ValueError("output_len must be a nonnegative int")
    if eos_first is not None and (
        type(eos_first) is not int or eos_first < 0 or eos_first >= output_len
    ):
        raise ValueError("eos_first must be None or a valid index into the output")
    if eos_first is None:
        return 0
    return output_len - eos_first - 1


def cause_rates(causes: list[str]) -> dict[str, float]:
    """Rate per cause over the given observations.

    Returns ``{cause: rate}`` for every cause in :data:`CAUSES`; rates
    sum to 1.0 for a nonempty list and are all 0.0 for an empty one.
    Raises ``ValueError`` on a non-list or an unknown cause value.
    """
    if type(causes) is not list:
        raise ValueError("causes must be a list")
    rates = dict.fromkeys(CAUSES, 0.0)
    for cause in causes:
        if cause not in rates:
            raise ValueError(f"unknown cause {cause!r}")
        rates[cause] += 1.0
    if causes:
        rates = {cause: count / len(causes) for cause, count in rates.items()}
    return rates


def aggregate_cause_records(records: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate validated per-observation stop-cause records.

    Each record is ``{"cause": str, "over_generated": int}``. Returns
    ``{"n", "cause_counts", "cause_rates", "over_generation_rate",
    "would_stop_early_rate", "max_length_rate"}`` where
    ``would_stop_early_rate`` is the fraction of observations where a
    stop-aware engine would have terminated before the cap (an EOS or
    stop-string mechanism fired) and ``over_generation_rate`` is the
    fraction with bleed tokens past the first EOT. Raises
    ``ValueError`` on malformed records.
    """
    if type(records) is not list:
        raise ValueError("records must be a list")
    causes: list[str] = []
    n_bleed = 0
    n_early = 0
    for record in records:
        if type(record) is not dict:
            raise ValueError("records must be dicts")
        cause = record.get("cause")
        bleed = record.get("over_generated")
        if cause not in CAUSES:
            raise ValueError(f"record has unknown cause {cause!r}")
        if type(bleed) is not int or bleed < 0:
            raise ValueError("record over_generated must be a nonnegative int")
        causes.append(cause)
        if bleed > 0:
            n_bleed += 1
        if cause in ("eos_id", "stop_string"):
            n_early += 1
    counts = dict.fromkeys(CAUSES, 0)
    for cause in causes:
        counts[cause] += 1
    n = len(records)
    return {
        "n": n,
        "cause_counts": counts,
        "cause_rates": cause_rates(causes),
        "over_generation_rate": n_bleed / n if n else 0.0,
        "would_stop_early_rate": n_early / n if n else 0.0,
        "max_length_rate": counts["max_length"] / n if n else 0.0,
    }
