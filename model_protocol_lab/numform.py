"""Numeric surface-form probe protocol: generation, forms, aggregation.

Why this exists: production prompts carry figures -- invoices,
metrics, tables pasted as text -- and the operator choosing how
to write those figures (``1234567`` vs ``1,234,567`` vs
``1.234567e6``) currently chooses blind. Reddy et al. 2026
(arXiv:2601.15251) show the written form moves arithmetic
accuracy on API models under vendor renderings, including when
the answer must come back in the operands' format; Yang et al.
2025 (NUPA, arXiv:2411.03766) cross representations including
scientific notation with add/compare/digit tasks but never vary
grouping separators or the chat template. This probe varies the
*rendering*: one checkpoint, hand-rendered candidate templates,
generated integer fixtures whose values are drawn once and
written in four surface forms, so the form contrast is measured
per candidate with everything else held fixed. The design doc
(docs/NUMFORM_DESIGN.md) states the scope and the settlements
below; the uniqueness pass is logged in docs/RESEARCH_LOG.md.

Definitions that carry the design (red-team cross-checked,
cycle 186):

* Renderers are total functions of the integer value alone,
  never of an operand string -- that is what makes the expected
  answer unique (a sum is re-rendered from its value, so
  regrouping and scientific renormalisation have exactly one
  right string). Decimal appears nowhere: its default context
  rounds at 28 significant digits, its parser accepts
  underscores, whitespace, NaN and non-ASCII digits, and its
  string forms are not these forms (all executed pre-build).
  Arithmetic is Python ``int``; parsing is a frozen ASCII
  grammar with digit/exponent caps enforced before conversion,
  because the parser consumes model output and an answer like
  ``1e999999999`` must classify as a failure, never allocate.
* The design is paired: one value set per task, rendered into
  every form, all arms sharing one fixture id so
  ``core.request_seed_for`` pairs their request seeds. The
  headline is a paired discordant contrast per form pair over
  the values completed in both forms -- marginal-rate
  subtraction was disproved as an acquittal rule in cycle 185
  and is not repeated here.
* A pass is value-and-form-and-boundary, and the payload says
  which conjunct failed: every cell decomposes into exact pass,
  right value in the wrong form, or wrong value. Without the
  decomposition a low rate in one form cannot distinguish
  "computes fine, won't emit the form" from "cannot compute",
  and those worlds demand opposite operator actions.
* Generated values are not user data, so records store the
  expected side's canonical integers in the clear, and the
  diff validator regenerates the whole set from the seed and
  recomputes every expected-side fact from it. The answer
  side: a record stores the answer text verbatim beside
  the value the stage parsed from it, and the cell stores
  the text's digest; the validator binds the digest to the
  text and re-parses the text under its own mirrored
  grammars (cycle 190 -- before that, the decomposition
  rested on a stage-attested parse the validator could
  not re-derive, and a forged stored value validated).
  (A hash over a value of at most 12 digits would be
  invertible by enumeration anyway, so digest-only
  expected-side storage would buy no privacy either.)
* The tokenizer grouping label is measured from the attached
  tokenizer by a frozen count-based probe battery, never
  inferred from a family name (Llama 3 onward groups digits by
  three; the single-digit Llama is Llama 2). One run attaches
  one backend, so the label takes one value per run: it is a
  run-level descriptor and a cross-run diff variable, never a
  within-run correlation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from .confidentiality import mcnemar_exact_p
from .core import Fixture, Message

PROTOCOL_VERSION = 1

#: Task kinds, in payload order. Rates are never pooled across
#: tasks: copy is form fidelity, compare is verbalised
#: magnitude, add is arithmetic -- a pooled headline would
#: manufacture a claim none of them supports.
TASKS: tuple[str, ...] = ("copy", "compare", "add")

#: Surface forms, in payload order.
FORMS: tuple[str, ...] = ("plain", "comma", "space", "scientific")

#: Digit-length bands (inclusive bounds) and the frozen set
#: size: 4 values drawn per band per task, 16 per task. A
#: pair's recorded band is its larger operand's band, so
#: compare's recorded per-band counts vary around 4 -- read
#: denominators from a payload's own by_band counts, never
#: from this constant.
BANDS: tuple[tuple[int, int], ...] = ((1, 3), (4, 6), (7, 9), (10, 12))
VALUES_PER_BAND = 4
VALUES_PER_TASK = VALUES_PER_BAND * len(BANDS)

#: The space-grouped form uses U+202F (narrow no-break space,
#: the SI grouping separator). ASCII space would be ambiguous
#: against the prompt's own operand separation; U+00A0 is named
#: future work alongside the other codepoints.
SPACE_CHAR = "\u202f"
SPACE_CODEPOINT = "U+202F"

SCIENTIFIC_GRAMMAR = "mantissa-[1,10)-lowercase-e-v1"
RENDERER_SPEC = "value-only-renderers-v1"
GENERATOR_RULE = "sha256-derived-paired-v1"

#: Frozen prompt templates. They contain no digit and no worked
#: example in any form: a single example is a few-shot arm for
#: one form only, and few-shot recovery is the largest
#: mitigation Reddy et al. measure (20-60%). The add/copy
#: templates instruct the model to answer in the operands'
#: form; that requirement is held constant across arms, so it
#: is part of the task definition, not a varying confound.
PROMPT_TEMPLATES: dict[str, str] = {
    "copy": "Copy this number exactly, character for character:\n{operand}",
    "compare": (
        "Which of these two numbers is larger? Reply with only the word first or second.\n{a}\n{b}"
    ),
    "add": (
        "Add these two numbers and reply with only the sum, "
        "written in the same format as the numbers:\n{a}\n{b}"
    ),
}
PROMPT_TEMPLATES_SHA256 = hashlib.sha256(
    json.dumps(PROMPT_TEMPLATES, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
).hexdigest()

#: Discordance floors, frozen for this probe (the confidentiality
#: / leading discipline: floors gate the contrast classification
#: only, never a reported rate or count).
MIN_PAIRED_EVALUABLE = 6
MIN_DISCORDANT = 2

#: candidates x tasks x values x forms at the legal maxima
#: (24 candidates): the generation budget and the record cap
#: are the same number because every generation is one record.
GENERATION_BUDGET = 4608
MAX_GROUND_RECORDS = 4608
MAX_RECORDS_BYTES = 12 * 1024 * 1024

#: Parser caps, enforced before any conversion. Operand values
#: have at most 12 digits and sums 13; the caps leave headroom
#: for costume variants (leading zeros, padded mantissas) while
#: keeping a hostile answer string from forcing a huge int.
MAX_PARSE_DIGITS = 20
#: The exponent cap is enforced by the grammar itself: the
#: scientific pattern admits at most three exponent digits,
#: so no parsed exponent can exceed this bound and no
#: post-match comparison is needed.
MAX_PARSE_EXPONENT = 999

#: The contrast classification vocabulary for one form pair.
CONTRAST_CLASSIFICATIONS: tuple[str, ...] = (
    "insufficient",
    "no_difference_observed",
    "below_threshold",
    "mixed_discordance",
)

#: Tokenizer grouping classes from the probe battery.
GROUPING_CLASSES: tuple[str, ...] = (
    "single_digit",
    "chunked_up_to_3",
    "irregular_or_mixed",
    "unknown",
)

_PLAIN_RE = re.compile(r"[0-9]+")
_COMMA_RE = re.compile(r"[0-9]{1,3}(?:,[0-9]{3})*")
_SPACE_RE = re.compile(r"[0-9]{1,3}(?:\u202f[0-9]{3})*")
_SCI_RE = re.compile(r"([0-9]+)(?:\.([0-9]+))?[eE]\+?([0-9]{1,3})")


def digest_text(text: str) -> str:
    """The one digest used for answer identity in this probe:
    SHA-256 of the UTF-8 encoding, hex-encoded.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Renderers: total functions of the integer value alone.
# ---------------------------------------------------------------------------


def _grouped(digits: str, separator: str) -> str:
    """Return the grouped."""
    groups: list[str] = []
    while len(digits) > 3:
        groups.append(digits[-3:])
        digits = digits[:-3]
    groups.append(digits)
    return separator.join(reversed(groups))


def render_form(value: int, form: str) -> str:
    """Render a non-negative integer in one surface form.

    Total on the domain: grouping is recomputed from the
    value's own decimal string (never inherited from an
    operand's grouping, which is what makes a sum's rendering
    unique), and the scientific form is the frozen grammar --
    mantissa in [1, 10) with trailing zeros stripped, lowercase
    e, bare exponent; zero is the special case ``0e0``.
    """
    if type(value) is not int or value < 0:
        raise ValueError("numform values are non-negative ints")
    digits = str(value)
    if form == "plain":
        return digits
    if form == "comma":
        return _grouped(digits, ",")
    if form == "space":
        return _grouped(digits, SPACE_CHAR)
    if form == "scientific":
        if value == 0:
            return "0e0"
        exponent = len(digits) - 1
        mantissa = digits.rstrip("0")
        if len(mantissa) > 1:
            return f"{mantissa[0]}.{mantissa[1:]}e{exponent}"
        return f"{mantissa}e{exponent}"
    raise ValueError(f"unknown numform form: {form!r}")


# ---------------------------------------------------------------------------
# The value parser: a frozen ASCII grammar with pre-conversion caps.
# ---------------------------------------------------------------------------


def parse_value(text: str) -> int | None:
    """Parse a string under the numform grammars to an int.

    Accepts exactly: plain digit strings (leading zeros parse),
    strict 3-3-3 comma grouping, strict 3-3-3 U+202F grouping,
    and scientific notation (capital E and a signed exponent
    parse -- costume differences belong to the value class, not
    to failure). Anything else -- wrong-width grouping, decimal
    points outside scientific notation, signs, underscores,
    non-ASCII digits, a scientific value that is not an integer
    -- returns None. Digit and exponent caps are checked before
    any conversion, so a model answer like ``1e999999999``
    classifies as a failure instead of allocating a huge int.
    """
    if type(text) is not str or not text:
        return None
    if _PLAIN_RE.fullmatch(text):
        if len(text) > MAX_PARSE_DIGITS:
            return None
        return int(text)
    if _COMMA_RE.fullmatch(text):
        digits = text.replace(",", "")
        if len(digits) > MAX_PARSE_DIGITS:
            return None
        return int(digits)
    if _SPACE_RE.fullmatch(text):
        digits = text.replace(SPACE_CHAR, "")
        if len(digits) > MAX_PARSE_DIGITS:
            return None
        return int(digits)
    match = _SCI_RE.fullmatch(text)
    if match is not None:
        head, frac, exponent_text = match.group(1), match.group(2) or "", match.group(3)
        if len(head) + len(frac) > MAX_PARSE_DIGITS:
            return None
        exponent = int(exponent_text)
        mantissa = int(head + frac)
        shift = exponent - len(frac)
        if shift >= 0:
            result: int = mantissa * 10**shift
            return result
        divisor = 10 ** (-shift)
        if mantissa % divisor:
            return None
        quotient: int = mantissa // divisor
        return quotient
    return None


# ---------------------------------------------------------------------------
# Stratum classifiers: digit-wise, independent of the generator's
# construction (the generator accepts a drawn pair only when the
# classifier confirms the target class).
# ---------------------------------------------------------------------------


def carry_class(a: int, b: int) -> str:
    """Classify an addition pair by its carry pattern.

    Positions run from the units digit up; a position's
    carry-out feeds the next position's carry-in. ``none``: no
    position carries. ``full_chain``: every position up to the
    wider operand's length carries. ``some``: anything between.
    """
    digits_a = str(a)
    digits_b = str(b)
    width = max(len(digits_a), len(digits_b))
    carries: list[bool] = []
    carry_in = 0
    for pos in range(width):
        digit_a = int(digits_a[-1 - pos]) if pos < len(digits_a) else 0
        digit_b = int(digits_b[-1 - pos]) if pos < len(digits_b) else 0
        carry_out = 1 if digit_a + digit_b + carry_in >= 10 else 0
        carries.append(bool(carry_out))
        carry_in = carry_out
    if not any(carries):
        return "none"
    if all(carries):
        return "full_chain"
    return "some"


def margin_class(a: int, b: int) -> str:
    """Classify a compare pair by where the values first differ.

    ``different_length``: digit lengths differ (decidable by
    length alone). Otherwise the first differing digit from
    the left: the last position is ``same_length_last_digit``,
    a position in the leading half is ``same_length_early``,
    anything else is ``same_length_middle`` -- a real class the
    classifier reports honestly, but one the generator never
    emits (its pairs are constructed into the three named
    classes), so a record carrying it fails validation.
    """
    digits_a = str(a)
    digits_b = str(b)
    if len(digits_a) != len(digits_b):
        return "different_length"
    for index, (digit_a, digit_b) in enumerate(zip(digits_a, digits_b, strict=True)):
        if digit_a != digit_b:
            if index == len(digits_a) - 1:
                return "same_length_last_digit"
            if index * 2 < len(digits_a):
                return "same_length_early"
            return "same_length_middle"
    return "equal"


# ---------------------------------------------------------------------------
# Generation: SHA-256-derived, paired across forms, stratified.
# ---------------------------------------------------------------------------


def _derive(seed: int, task: str, band_index: int, slot: int, counter: int, stream: int) -> int:
    """Return the derive."""
    material = f"numform-v1|{seed}|{task}|{band_index}|{slot}|{counter}|{stream}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _value_in_band(derived: int, lo: int, hi: int, *, min_length: int = 1) -> int:
    """Return the value in band."""
    low = max(lo, min_length)
    length = low + derived % (hi - low + 1)
    rest = derived >> 16
    if length == 1:
        return rest % 10
    value: int = 10 ** (length - 1) + rest % (9 * 10 ** (length - 1))
    return value


def _round_value(band_index: int, slot: int) -> int:
    """The round stratum: a power of ten, then all-nines, at the
    band's top length. Real figures are disproportionately
    round; a uniform-random-only generator would measure a
    distribution no invoice ever produced.
    """
    _lo, hi = BANDS[band_index]
    if slot == 0:
        power: int = 10 ** (hi - 1)
        return power
    nines: int = 10**hi - 1
    return nines


def _band_of(value: int) -> int:
    """Return the frozen band index for a value's digit length."""
    length = len(str(value))
    for index, (lo, hi) in enumerate(BANDS):
        if lo <= length <= hi:
            return index
    raise ValueError("value outside the frozen bands")


def _compare_partner(a: int, target: str, derived: int) -> int:
    """Construct a compare partner for ``a`` in the target
    margin class (verified by the caller against margin_class;
    the construction is a proposal, the classifier decides).
    """
    digits = str(a)
    length = len(digits)
    if target == "different_length":
        new_length = length + 1 if derived % 2 == 0 else length - 1
        if new_length > 12:
            new_length = length - 1
        if new_length < 1:
            new_length = length + 1
        return _value_in_band(derived >> 8, new_length, new_length)
    if target == "same_length_last_digit":
        last = int(digits[-1])
        replacement = (last + 1 + (derived % 9)) % 10
        return int(digits[:-1] + str(replacement)) if length > 1 else replacement
    # same_length_early: change one digit in the leading half,
    # keeping the leading digit nonzero.
    position = derived % max(1, length // 2)
    current = int(digits[position])
    replacement = (current + 1 + (derived >> 4) % 9) % 10
    if position == 0 and replacement == 0:
        replacement = 1
    return int(digits[:position] + str(replacement) + digits[position + 1 :])


def _add_partner(a: int, target: str, derived: int) -> int:
    """Construct an add partner for ``a`` in the target carry
    class, digit by digit from the units position (verified by
    the caller against carry_class).
    """
    digits = [int(ch) for ch in reversed(str(a))]
    partner_digits: list[int] = []
    if target == "full_chain" and digits[0] == 0:
        digits[0] = 1
    stream = derived
    if target == "none":
        for digit in digits:
            partner_digits.append(stream % (10 - digit) if digit < 9 else 0)
            stream //= 10
    elif target == "full_chain":
        carry_in = 0
        for digit in digits:
            minimum = 10 - digit - carry_in
            partner_digits.append(minimum + stream % (10 - minimum))
            carry_in = 1
            stream //= 10
    else:  # some: force exactly one carry position, none below it
        pivot_candidates = [pos for pos, digit in enumerate(digits) if digit >= 1]
        if not pivot_candidates:
            # a == 0: no digit can carry, so no partner is in
            # class "some". Propose zero and let the caller
            # reject the pair -- carry_class(0, 0) is "none"
            # and a != b fails too -- and redraw with the next
            # counter. Rejecting here (instead of skipping the
            # draw in _gen_pairs) keeps every healthy seed's
            # generated set byte-identical: only the crashing
            # path changes.
            return 0
        pivot = pivot_candidates[stream % len(pivot_candidates)]
        carry_in = 0
        for pos, digit in enumerate(digits):
            if pos < pivot:
                partner_digits.append(stream % (10 - digit) if digit < 9 else 0)
            elif pos == pivot:
                minimum = 10 - digit - carry_in
                partner_digits.append(minimum + stream % (10 - minimum))
                carry_in = 1
            elif digit + carry_in <= 9:
                partner_digits.append(stream % (10 - digit - carry_in))
                carry_in = 0
            else:
                partner_digits.append(stream % 10)
                carry_in = 1
            stream //= 10
    partner = int("".join(str(d) for d in reversed(partner_digits)))
    if partner == 0:
        # A zero addend turns add into copy; nudge the least
        # significant position the class construction allows.
        partner = 1 if digits[0] <= 8 or target != "none" else 10
    return partner


def _is_round_value(value: int) -> bool:
    """Intrinsically round: a power of ten or all-nines. The
    copy stratum label is defined by this property, so random
    draws exclude these values -- the label is then a property
    of the value itself, recomputable by the diff validator
    from the record alone.
    """
    if value < 10:
        return False
    digits = str(value)
    result: bool = value in (10 ** (len(digits) - 1), 10 ** len(digits) - 1)
    return result


def _gen_copy(seed: int) -> list[dict[str, Any]]:
    """Return the gen copy."""
    entries: list[dict[str, Any]] = []
    index = 0
    for band_index, (lo, hi) in enumerate(BANDS):
        for slot in range(VALUES_PER_BAND):
            is_round = slot < 2
            if is_round:
                value = _round_value(band_index, slot)
            else:
                value = _value_in_band(_derive(seed, "copy", band_index, slot, 0, 0), lo, hi)
                counter = 0
                while _is_round_value(value):
                    counter += 1
                    value = _value_in_band(
                        _derive(seed, "copy", band_index, slot, counter, 0), lo, hi
                    )
            entries.append(
                {
                    "value_id": f"nf-copy-{index:02d}",
                    "task": "copy",
                    "band": band_index,
                    "round": is_round,
                    "stratum": "round" if is_round else "random",
                    "operands": [value],
                    "expected_value": value,
                }
            )
            index += 1
    return entries


def _gen_pairs(seed: int, task: str) -> list[dict[str, Any]]:
    """Return the gen pairs."""
    entries: list[dict[str, Any]] = []
    index = 0
    targets = (
        ("different_length", "same_length_early", "same_length_last_digit")
        if task == "compare"
        else ("none", "some", "full_chain")
    )
    classifier = margin_class if task == "compare" else carry_class
    for band_index, (lo, hi) in enumerate(BANDS):
        for slot in range(VALUES_PER_BAND):
            target = targets[slot % len(targets)]
            is_round = slot < 2
            pair: tuple[int, int] | None = None
            for counter in range(1000):
                if is_round and counter == 0:
                    a = _round_value(band_index, slot)
                else:
                    a = _value_in_band(
                        _derive(seed, task, band_index, slot, counter, 0),
                        lo,
                        hi,
                        min_length=2 if target == "same_length_early" else 1,
                    )
                derived = _derive(seed, task, band_index, slot, counter, 1)
                b = (
                    _compare_partner(a, target, derived)
                    if task == "compare"
                    else _add_partner(a, target, derived)
                )
                if b < 0 or len(str(b)) > 12:
                    continue
                if task == "add" and target == "full_chain" and str(a)[-1] == "0":
                    # _add_partner nudges its own copy of a; keep
                    # the recorded a in step with the construction.
                    a = a + 1
                if classifier(a, b) == target and a != b:
                    pair = (a, b)
                    break
            if pair is None:  # pragma: no cover - construction is verified below
                raise ValueError(f"numform generation failed for {task} slot {slot}")
            a, b = pair
            entry: dict[str, Any] = {
                "value_id": f"nf-{task}-{index:02d}",
                "task": task,
                "band": _band_of(max(a, b)),
                # Pairs carry no round label: compare reorders
                # its operands for presentation, so which operand
                # was drawn from the round table is not a property
                # a record can honestly claim. Round first
                # operands in slots 0-1 remain a property of the
                # generated set, bound by generator_sha256.
                "round": False,
                "stratum": target,
                "operands": [a, b],
                "expected_value": max(a, b) if task == "compare" else a + b,
            }
            if task == "compare":
                larger_first = slot % 2 == 0
                first, second = (max(a, b), min(a, b)) if larger_first else (min(a, b), max(a, b))
                entry["operands"] = [first, second]
                entry["larger_first"] = larger_first
                entry["expected_label"] = "first" if larger_first else "second"
            entries.append(entry)
            index += 1
    return entries


def generate_value_set(seed: int) -> dict[str, list[dict[str, Any]]]:
    """The frozen generated set for a run seed: per task, the 16
    value entries every form arm shares. Pure and deterministic
    -- the same seed regenerates the identical set across runs
    and PYTHONHASHSEED values (SHA-256 derivation, sorted
    iteration only).
    """
    if type(seed) is not int:
        raise ValueError("numform seed must be an int")
    return {
        "copy": _gen_copy(seed),
        "compare": _gen_pairs(seed, "compare"),
        "add": _gen_pairs(seed, "add"),
    }


def expected_answer(entry: dict[str, Any], form: str) -> str:
    """The expected answer string for one entry in one form:
    the rendered operand (copy), the rendered sum (add), or the
    form-invariant label (compare -- the one arm where the
    scorer cannot be a form confound, by construction).
    """
    if entry["task"] == "compare":
        return str(entry["expected_label"])
    return render_form(int(entry["expected_value"]), form)


def prompt_text(entry: dict[str, Any], form: str) -> str:
    """The frozen prompt for one entry in one form, operands
    rendered in that form, one per line.
    """
    operands = [render_form(int(value), form) for value in entry["operands"]]
    template = PROMPT_TEMPLATES[str(entry["task"])]
    if entry["task"] == "copy":
        return template.format(operand=operands[0])
    return template.format(a=operands[0], b=operands[1])


def build_fixtures(seed: int) -> dict[tuple[str, str], tuple[Fixture, ...]]:
    """All fixtures for a run, keyed by (task, form). Fixture
    ids are the value ids -- identical across forms, so request
    seeds pair the arms of one value.
    """
    value_set = generate_value_set(seed)
    fixtures: dict[tuple[str, str], tuple[Fixture, ...]] = {}
    for task in TASKS:
        for form in FORMS:
            fixtures[(task, form)] = tuple(
                Fixture(
                    id=str(entry["value_id"]),
                    group="numform",
                    messages=(Message("user", prompt_text(entry, form)),),
                    expected=expected_answer(entry, form),
                )
                for entry in value_set[task]
            )
    return fixtures


def canonical_generated_rows(seed: int) -> list[list[Any]]:
    """The canonical generated set the protocol fingerprint
    binds: per entry, the task, value id, operand values, and
    the expected answer in every form. A payload whose
    generator drifted is a different instrument, and diff says
    so instead of comparing across instruments.
    """
    value_set = generate_value_set(seed)
    rows: list[list[Any]] = []
    for task in TASKS:
        for entry in value_set[task]:
            rows.append(
                [
                    task,
                    entry["value_id"],
                    [int(value) for value in entry["operands"]],
                    int(entry["expected_value"]),
                    [expected_answer(entry, form) for form in FORMS],
                ]
            )
    return rows


def generator_sha256(seed: int) -> str:
    """Digest of the canonical generated rows for a seed: two
    runs share an instrument only when this matches, because
    the generated set is the treatment.
    """
    canonical = json.dumps(canonical_generated_rows(seed), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The frozen instrument description a payload is checked
    against: tasks, forms, bands, floors, budgets, renderer
    and generator rules, prompt templates with their digest,
    and the seed/backend pair that generated the set.
    """
    return {
        "version": PROTOCOL_VERSION,
        "tasks": list(TASKS),
        "forms": list(FORMS),
        "bands": [list(band) for band in BANDS],
        "values_per_task": VALUES_PER_TASK,
        "min_paired_evaluable": MIN_PAIRED_EVALUABLE,
        "min_discordant": MIN_DISCORDANT,
        "generation_budget": GENERATION_BUDGET,
        "max_ground_records": MAX_GROUND_RECORDS,
        "max_records_bytes": MAX_RECORDS_BYTES,
        "space_codepoint": SPACE_CODEPOINT,
        "scientific_grammar": SCIENTIFIC_GRAMMAR,
        "renderer_spec": RENDERER_SPEC,
        "generator_rule": GENERATOR_RULE,
        "prompt_templates": dict(PROMPT_TEMPLATES),
        "prompt_templates_sha256": PROMPT_TEMPLATES_SHA256,
        "seed": seed,
        "generator_sha256": generator_sha256(seed),
        "backend_fingerprint": backend_fingerprint,
    }


def protocol_fingerprint(seed: int, backend_fingerprint: str) -> str:
    """SHA-256 over the canonical protocol object, so any
    drift in a frozen scalar is a different fingerprint.
    """
    canonical = json.dumps(
        protocol_object(seed, backend_fingerprint),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Tokenizer grouping: a frozen count-based probe battery.
# ---------------------------------------------------------------------------

#: Probe strings: plain digit strings of every length 1-12 in
#: three digit patterns (sequential, all-same, zero-heavy), plus
#: every form of one frozen probe value. The classification
#: reads token counts only -- the Backend interface exposes
#: encode/decode and nothing else, so per-token text recovery
#: is not available to build on.
PROBE_VALUE = 123456789


def probe_strings() -> tuple[str, ...]:
    """The frozen grouping-battery strings: digit strings of
    every length 1-12 in sequential, all-ones, and zero-heavy
    patterns, plus every form of the probe value.
    """
    probes: list[str] = []
    sequential = "123456789012"
    ones = "1" * 12
    zero_heavy = "1" + "0" * 11
    for length in range(1, 13):
        probes.append(sequential[:length])
        probes.append(ones[:length])
        probes.append(zero_heavy[:length])
    for form in FORMS:
        probes.append(render_form(PROBE_VALUE, form))
    return tuple(probes)


def _digit_probes() -> list[str]:
    """Return the digit probes."""
    return [probe for probe in probe_strings() if probe.isascii() and probe.isdigit()]


def classify_grouping(evidence: dict[str, int]) -> str:
    """The frozen decision rule over probe token counts.

    ``single_digit`` iff every all-digit probe's token count
    equals its digit count; ``chunked_up_to_3`` iff every
    all-digit probe of length >= 2 counts exactly ceil(len/3)
    and length-1 probes count 1; ``irregular_or_mixed`` iff
    every count lies between those two predictions without
    matching either rule; otherwise ``unknown`` -- including
    missing or empty evidence, because a label the evidence
    cannot support is worse than no label.
    """
    probes = _digit_probes()
    if not evidence or any(probe not in evidence for probe in probes):
        return "unknown"
    counts = [(len(probe), evidence[probe]) for probe in probes]
    if any(count < 1 for _length, count in counts):
        return "unknown"
    if all(count == length for length, count in counts):
        return "single_digit"
    if all(count == (length + 2) // 3 for length, count in counts):
        return "chunked_up_to_3"
    if all((length + 2) // 3 <= count <= length for length, count in counts):
        return "irregular_or_mixed"
    return "unknown"


def measure_grouping(
    encode: Callable[[str], Sequence[int]],
    value_set: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Measure the grouping evidence and the per-form operand
    token table with the attached backend's encoder.

    Evidence and table are measured facts; the classification
    is a summary that can be ``unknown`` without invalidating
    the counts. An encoder that raises yields empty evidence
    and ``unknown`` -- the per-form rates never depend on the
    label, so a tokenizer the battery cannot read must not
    take the probe down with it.
    """
    evidence: dict[str, int] = {}
    try:
        for probe in probe_strings():
            evidence[probe] = len(encode(probe))
    except Exception:  # noqa: BLE001 -- an unreadable tokenizer is "unknown"
        evidence = {}
    form_tokens: dict[str, list[int]] = {form: [] for form in FORMS}
    plain_tokens: dict[int, int] = {}
    try:
        for task in TASKS:
            for entry in value_set[task]:
                for value in entry["operands"]:
                    rendered_plain = render_form(int(value), "plain")
                    if int(value) not in plain_tokens:
                        plain_tokens[int(value)] = len(encode(rendered_plain))
                    for form in FORMS:
                        form_tokens[form].append(len(encode(render_form(int(value), form))))
    except Exception:  # noqa: BLE001 -- the table is diagnostic, never fatal
        form_tokens = {form: [] for form in FORMS}
        plain_tokens = {}
    means: dict[str, float | None] = {}
    deltas: dict[str, float | None] = {}
    for form in FORMS:
        counts = form_tokens[form]
        means[form] = (sum(counts) / len(counts)) if counts else None
    paired_plain: list[int] = []
    paired_form: dict[str, list[int]] = {form: [] for form in FORMS}
    try:
        for task in TASKS:
            for entry in value_set[task]:
                for value in entry["operands"]:
                    paired_plain.append(plain_tokens[int(value)])
                    for form in FORMS:
                        paired_form[form].append(len(encode(render_form(int(value), form))))
    except Exception:  # noqa: BLE001 -- diagnostic only
        paired_plain = []
        paired_form = {form: [] for form in FORMS}
    for form in FORMS:
        if paired_plain and len(paired_form[form]) == len(paired_plain):
            diffs = [
                other - plain for other, plain in zip(paired_form[form], paired_plain, strict=True)
            ]
            deltas[form] = sum(diffs) / len(diffs)
        else:
            deltas[form] = None
    return {
        "classification": classify_grouping(evidence),
        "evidence": evidence,
        "form_token_means": means,
        "token_deltas_vs_plain": deltas,
    }


# ---------------------------------------------------------------------------
# Aggregation: records -> per-form aggregates and paired contrasts.
# ---------------------------------------------------------------------------


def cell_decomposition(record: dict[str, Any]) -> str:
    """The decomposition class of one completed cell.

    Copy/add: exact pass, right value in the wrong form, or
    wrong value. Compare: exact pass, the other label, or a
    non-label answer -- groundable from digests alone because
    the label set has two known strings. Callers must only
    invoke this on completed cells (cell present).
    """
    cell = record["cell"]
    if cell is None:
        raise ValueError("decomposition requires a completed cell")
    if cell["verdict"]:
        return "exact_pass"
    if record["task"] == "compare":
        other = "second" if record["expected_answer"] == "first" else "first"
        if cell["answer_digest"] == digest_text(other):
            return "wrong_label"
        return "non_label"
    if record["answer_value"] is not None and str(record["answer_value"]) == str(
        record["expected_value"]
    ):
        return "value_pass_form_fail"
    return "value_fail"


def _rate(numerator: int, denominator: int) -> float | None:
    """Return the rate."""
    return (numerator / denominator) if denominator else None


def aggregate_records(records: list[dict[str, Any]], task: str, form: str) -> dict[str, Any]:
    """Aggregates for one candidate x task x form over its
    ground records. Rates are plain ratios with their
    denominators beside them; nothing is pooled anywhere.
    """
    selected = [r for r in records if r["task"] == task and r["form"] == form]
    completed = [r for r in selected if r["cell"] is not None]
    n_exact = sum(1 for r in completed if r["cell"]["verdict"])
    decomposition: dict[str, int] = {}
    for record in completed:
        cls = cell_decomposition(record)
        decomposition[cls] = decomposition.get(cls, 0) + 1
    if task != "compare":
        boundary_split = sum(
            1
            for r in completed
            if cell_decomposition(r) == "value_pass_form_fail" and r["cell"]["boundary_found"]
        )
        decomposition["value_pass_form_fail_boundary_found"] = boundary_split
    by_band: dict[str, dict[str, Any]] = {}
    for band_index in range(len(BANDS)):
        band_records = [r for r in selected if r["band"] == band_index]
        band_completed = [r for r in band_records if r["cell"] is not None]
        by_band[str(band_index)] = {
            "n_values": len(band_records),
            "n_completed": len(band_completed),
            "n_exact": sum(1 for r in band_completed if r["cell"]["verdict"]),
            "n_form_identical": sum(1 for r in band_records if r["form_identical"]),
        }
    input_tokens = [int(r["cell"]["input_tokens"]) for r in completed]
    answer_tokens = [int(r["expected_answer_tokens"]) for r in completed]
    return {
        "n_values": len(selected),
        "n_completed": len(completed),
        "n_exact": n_exact,
        "exact_rate": _rate(n_exact, len(completed)),
        "decomposition": decomposition,
        "by_band": by_band,
        "mean_input_tokens": (sum(input_tokens) / len(input_tokens)) if input_tokens else None,
        "mean_expected_answer_tokens": (
            (sum(answer_tokens) / len(answer_tokens)) if answer_tokens else None
        ),
    }


def classify_contrast(n_paired: int, n_a_only: int, n_b_only: int) -> str:
    """The total classification of one paired form contrast.

    A function of the discordant counts alone (the cycle-185
    settlement): below the paired floor the contrast is
    ``insufficient``; no discordants is ``no_difference_observed``;
    a one-sided discordance at or above the discordant floor is
    an advantage for the side whose passes they are; symmetric
    discordance is ``mixed_discordance``; anything else is
    ``below_threshold``.
    """
    if n_paired < MIN_PAIRED_EVALUABLE:
        return "insufficient"
    if n_a_only == 0 and n_b_only == 0:
        return "no_difference_observed"
    if n_a_only >= MIN_DISCORDANT and n_a_only > n_b_only:
        return "a_advantage"
    if n_b_only >= MIN_DISCORDANT and n_b_only > n_a_only:
        return "b_advantage"
    if n_a_only == n_b_only and n_a_only >= MIN_DISCORDANT:
        return "mixed_discordance"
    return "below_threshold"


def paired_contrast(
    records: list[dict[str, Any]], task: str, form_a: str, form_b: str
) -> dict[str, Any]:
    """The paired contrast between two forms for one candidate
    and task, over the values completed in both forms.
    """
    by_value: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["task"] != task or record["form"] not in (form_a, form_b):
            continue
        by_value.setdefault(str(record["value_id"]), {})[str(record["form"])] = record
    n_paired = n_a_only = n_b_only = n_both_pass = n_both_fail = 0
    for pair in by_value.values():
        rec_a = pair.get(form_a)
        rec_b = pair.get(form_b)
        if rec_a is None or rec_b is None:
            continue
        if rec_a["cell"] is None or rec_b["cell"] is None:
            continue
        n_paired += 1
        pass_a = bool(rec_a["cell"]["verdict"])
        pass_b = bool(rec_b["cell"]["verdict"])
        if pass_a and pass_b:
            n_both_pass += 1
        elif pass_a:
            n_a_only += 1
        elif pass_b:
            n_b_only += 1
        else:
            n_both_fail += 1
    label = classify_contrast(n_paired, n_a_only, n_b_only)
    if label == "a_advantage":
        label = f"{form_a}_advantage"
    elif label == "b_advantage":
        label = f"{form_b}_advantage"
    return {
        "n_paired": n_paired,
        "n_pass_a_only": n_a_only,
        "n_pass_b_only": n_b_only,
        "n_both_pass": n_both_pass,
        "n_both_fail": n_both_fail,
        "mcnemar_p": mcnemar_exact_p(n_b_only, n_a_only),
        "classification": label,
    }


def form_pairs() -> tuple[tuple[str, str], ...]:
    """The six unordered form pairs in FORMS order -- the
    fixed contrast set every candidate reports.
    """
    pairs: list[tuple[str, str]] = []
    for index, form_a in enumerate(FORMS):
        for form_b in FORMS[index + 1 :]:
            pairs.append((form_a, form_b))
    return tuple(pairs)


def records_bytes(records_by_candidate: dict[str, list[dict[str, Any]]]) -> int:
    """Canonical JSON size of the ground records, the quantity
    the payload byte cap is enforced against.
    """
    return len(
        json.dumps(records_by_candidate, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )
