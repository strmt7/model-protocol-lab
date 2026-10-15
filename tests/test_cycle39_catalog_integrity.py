"""Cycle 39: whole-catalog integrity cross-check for ``catalog.py``.

Unlike cycles 10/11/13/14/15 (per-family byte-exact template pins), this
sweep verifies WHOLE-CATALOG invariants across all 28 families:

* key-set pin (exact 28 keys; sorted order is stable across runs),
* every family renders deterministically for 1-turn and multi-turn
  conversations (byte-identical across runs),
* independent oracle: a from-first-principles re-implementation of the
  ``Format.render`` contract agrees with ``Format.render`` for every
  family on fuzzed conversations (deterministic string seeds),
* framing fields are exact ``str`` (no subclasses), names non-blank,
  ``repeat_bos`` an exact bool, entries frozen,
* ``candidates(family, mutate=True)`` yields valid ``Format`` objects
  with documented relationships to their bases (branch pins for the
  8 generation-prompt branch families),
* no two families are accidentally identical: the only framing-identical
  group is ``{chatml, qwen3, kolibri}``, which is deliberate (see below),
* per-role turn-terminator fallback semantics hold for every family,
* hostile catalog lookups behave as documented.

Doc/code discrepancies found (source behavior NOT changed -- pinned
as-is, see the tests marked DISCREPANCY):

1. (Fixed in cycle 39; was an observation, not a bug in the catalog
   data.) ``candidates()`` documents ``ValueError("Unknown format
   family")`` for unknown families, but an *unhashable* lookup key (e.g.
   a list) used to raise ``TypeError: unhashable type`` from the ``in``
   check before the ``ValueError`` path was reached. ``candidates()`` now
   type-gates first (``type(family) is not str``), so the documented
   ValueError holds for every input. Pinned in
   ``test_unhashable_lookup_raises_valueerror``.

2. (Observation.) The framing-identical trio ``chatml`` / ``qwen3`` /
   ``kolibri`` is deliberate, not an accident: the catalog comments
   record that Qwen3's text-only no-BOS render is ChatML-shaped and
   that Kolibri-1's user/assistant turns are standard ChatML (its
   documented deltas -- the always-emitted system block, the thinking
   branches -- are not modeled in ``Format``). Pinned in
   ``test_only_documented_duplicate_group``.

3. (Observation.) ``deepseek41`` differs from ``deepseek`` *only* in
   ``open_suffix`` (``<|Assistant|><think>`` vs the assistant-prefix
   fallback); ``glm53`` vs ``glm4`` differ in whitespace framing only.
   These near-duplicates are the documented version deltas, pinned by
   the group test so any future accidental convergence is caught.

Sensitivity: the file was validated against 9 hand-built mutants on a
copy of the package in /tmp (never the repo); every mutant is caught by
at least one test. Mutant list: drop ``nanbeige42`` from ``CATALOG``;
add a ``chatml`` framing clone under a new key; swap the ``chatml`` and
``llama3`` entries; make ``candidates("nope")`` return ``()`` instead of
raising; drop the ``kolibri/thinking-off`` branch; set Mistral's
``user_turn_end`` to ``None``; set ``lfm25``'s ``open_suffix`` to
``None``; blank the ``gemma4`` name; make ``phi4`` framing-identical to
``chatml``.
"""

from __future__ import annotations

import random
import unittest
from dataclasses import FrozenInstanceError, replace

from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.core import Format, Message

# Exact 28-family key pin, sorted (the declaration order in catalog.py is
# NOT sorted; the sorted order must be stable across runs).
EXPECTED_KEYS_SORTED = [
    "chatml",
    "deepseek",
    "deepseek41",
    "gemma",
    "gemma3",
    "gemma4",
    "glm4",
    "glm45",
    "glm53",
    "hy4",
    "k2horizon",
    "kimi2",
    "kimi3",
    "kolibri",
    "laguna",
    "lfm25",
    "ling",
    "llama3",
    "llama4",
    "minimax3",
    "mistral",
    "mistral_large",
    "muse_glimmer",
    "muse_spark",
    "nanbeige42",
    "nemotron35",
    "phi4",
    "qwen3",
]

# Families with an explicit open_suffix (everyone else falls back to
# assistant_prefix per the Format.render contract).
EXPLICIT_OPEN_SUFFIX = {
    "deepseek41",
    "glm4",
    "glm45",
    "glm53",
    "hy4",
    "k2horizon",
    "kimi3",
    "laguna",
    "lfm25",
    "ling",
    "minimax3",
    "mistral_large",
    "muse_glimmer",
    "muse_spark",
    "nanbeige42",
    "nemotron35",
}

# Families with explicit per-role turn terminators (both roles set).
EXPLICIT_PER_ROLE_ENDS = {
    "deepseek",
    "deepseek41",
    "kimi3",
    "laguna",
    "mistral",
    "mistral_large",
}

# Documented generation-prompt branch suffixes riding the mutate path,
# per family (the kolibri/thinking-off convention).
EXPECTED_BRANCHES = {
    "kolibri": {"thinking-off"},
    "nemotron35": {"thinking-off"},
    "ling": {"thinking-off"},
    "laguna": {"thinking-on"},
    "k2horizon": {"effort-medium", "effort-low"},
    "hy4": {"no-think"},
    "minimax3": {"thinking-disabled", "thinking-enabled"},
    "nanbeige42": {"thinking-off"},
}


def _oracle_render(fmt: Format, messages: tuple[Message, ...]) -> str:
    """Independent re-derivation of the Format.render contract.

    Written from the docstring alone ("repeat_bos means BOS before every
    message, including the open reply; per-role turn terminators fall back
    to turn_end when unset; the open-reply suffix falls back to
    assistant_prefix when unset"), never by copying the implementation.
    """
    parts = [fmt.bos]
    for index, message in enumerate(messages):
        if index and fmt.repeat_bos:
            parts.append(fmt.bos)
        if message.role == "user":
            prefix, end = fmt.user_prefix, fmt.user_turn_end
        else:
            prefix, end = fmt.assistant_prefix, fmt.assistant_turn_end
        if end is None:
            end = fmt.turn_end
        parts.extend((prefix, message.content, end))
    if fmt.repeat_bos:
        parts.append(fmt.bos)
    suffix = fmt.open_suffix if fmt.open_suffix is not None else fmt.assistant_prefix
    parts.append(suffix)
    return "".join(parts)


def _fuzz_conversations(seed: str, count: int):
    """Deterministic fuzzed conversations (string seed -> random.Random)."""
    rng = random.Random(seed)  # noqa: S311 -- deterministic string seed for test fuzz
    pool = [
        "Hello",
        "",
        " ",
        "\n",
        "How are you?\nFine.\tThanks",
        "<|im_start|>user\n",
        "<|im_end|>",
        "南北阁",
        "é\tü",
        "reasoning_effort:{effort}",
        "<｜hy_start:opensource｜>",
        "]~b]ai\n</mm:think>",
        "x" * 500,
        "\x00",
        "[INST] [/INST]",
        "{content}",
        "<role>HUMAN</role>",
    ]
    conversations = []
    for _ in range(count):
        turns = rng.randint(1, 5)  # user turns; total messages = 2*turns - 1
        messages = []
        for turn in range(turns):
            messages.append(Message("user", rng.choice(pool)))
            if turn < turns - 1:
                messages.append(Message("assistant", rng.choice(pool)))
        conversations.append(tuple(messages))
    return conversations


def _framing_key(fmt: Format) -> tuple:
    """Every framing field except the name (and repeat_bos normalized)."""
    return (
        fmt.bos,
        fmt.user_prefix,
        fmt.assistant_prefix,
        fmt.turn_end,
        fmt.repeat_bos if fmt.bos else False,
        fmt.user_turn_end,
        fmt.assistant_turn_end,
        fmt.open_suffix,
    )


class KeySetTests(unittest.TestCase):
    def test_exact_28_keys(self):
        self.assertEqual(set(CATALOG.keys()), set(EXPECTED_KEYS_SORTED))
        self.assertEqual(len(CATALOG), 28)

    def test_sorted_order_stable(self):
        self.assertEqual(sorted(CATALOG.keys()), EXPECTED_KEYS_SORTED)

    def test_keys_are_exact_nonblank_str(self):
        for key in CATALOG:
            with self.subTest(key=key):
                self.assertIs(type(key), str)
                self.assertTrue(key.strip())
                self.assertEqual(CATALOG[key].name, key)

    def test_catalog_values_are_formats(self):
        for key, fmt in CATALOG.items():
            with self.subTest(key=key):
                self.assertIsInstance(fmt, Format)


class DeterminismTests(unittest.TestCase):
    def test_render_byte_identical_across_runs(self):
        one_turn = (Message("user", "Hello"),)
        three_turn = (
            Message("user", "Hello"),
            Message("assistant", "Hi there"),
            Message("user", "Bye"),
        )
        five_turn = (
            Message("user", "a"),
            Message("assistant", "b"),
            Message("user", "c"),
            Message("assistant", "d"),
            Message("user", "e"),
        )
        for key, fmt in CATALOG.items():
            with self.subTest(family=key):
                for messages in (one_turn, three_turn, five_turn):
                    first = fmt.render(messages)
                    self.assertEqual(fmt.render(messages), first)
                    self.assertEqual(fmt.render(messages), first)

    def test_fuzzed_conversations_deterministic(self):
        for messages in _fuzz_conversations("cycle39-determinism", 60):
            for key, fmt in CATALOG.items():
                with self.subTest(family=key, turns=len(messages)):
                    self.assertEqual(fmt.render(messages), fmt.render(messages))

    def test_render_result_is_exact_str(self):
        for key, fmt in CATALOG.items():
            with self.subTest(family=key):
                rendered = fmt.render((Message("user", "x"),))
                self.assertIs(type(rendered), str)


class OracleAgreementTests(unittest.TestCase):
    def test_oracle_agrees_on_fixed_conversations(self):
        conversations = [
            (Message("user", "Hello"),),
            (
                Message("user", "Hello"),
                Message("assistant", "Hi there"),
                Message("user", "Bye"),
            ),
            (
                Message("user", ""),
                Message("assistant", ""),
                Message("user", ""),
            ),
        ]
        for key, fmt in CATALOG.items():
            for messages in conversations:
                with self.subTest(family=key, turns=len(messages)):
                    self.assertEqual(fmt.render(messages), _oracle_render(fmt, messages))

    def test_oracle_agrees_on_fuzzed_conversations(self):
        for messages in _fuzz_conversations("cycle39-oracle", 80):
            for key, fmt in CATALOG.items():
                with self.subTest(family=key, turns=len(messages)):
                    self.assertEqual(fmt.render(messages), _oracle_render(fmt, messages))

    def test_oracle_agrees_with_repeat_bos_enabled(self):
        # repeat_bos is False for every catalog entry, so exercise the
        # True path on copies (independent of the catalog data).
        for key, fmt in CATALOG.items():
            with self.subTest(family=key):
                toggled = replace(fmt, name=fmt.name + "/t", repeat_bos=True)
                messages = (
                    Message("user", "Hello"),
                    Message("assistant", "Hi"),
                    Message("user", "Bye"),
                )
                self.assertEqual(toggled.render(messages), _oracle_render(toggled, messages))

    def test_oracle_agrees_for_every_mutate_variant(self):
        messages = (
            Message("user", "Hello"),
            Message("assistant", "Hi there"),
            Message("user", "Bye"),
        )
        for key in CATALOG:
            for variant in candidates(key, mutate=True):
                with self.subTest(variant=variant.name):
                    self.assertEqual(variant.render(messages), _oracle_render(variant, messages))


class FramingFieldTests(unittest.TestCase):
    def test_all_text_fields_are_exact_str(self):
        fields = (
            "bos",
            "user_prefix",
            "assistant_prefix",
            "turn_end",
            "user_turn_end",
            "assistant_turn_end",
            "open_suffix",
        )
        for key, fmt in CATALOG.items():
            with self.subTest(family=key):
                self.assertIs(type(fmt.name), str)
                self.assertTrue(fmt.name.strip())
                self.assertIs(type(fmt.repeat_bos), bool)
                for field in fields:
                    value = getattr(fmt, field)
                    self.assertTrue(
                        value is None or type(value) is str,
                        f"{field} must be exact str or None, got {type(value)}",
                    )

    def test_entries_are_frozen(self):
        for key, fmt in CATALOG.items():
            with self.subTest(family=key):
                with self.assertRaises(FrozenInstanceError):
                    fmt.bos = "<tampered>"  # type: ignore[misc]
                self.assertEqual(CATALOG[key].bos, fmt.bos)

    def test_explicit_open_suffix_pin(self):
        actual = {k for k, f in CATALOG.items() if f.open_suffix is not None}
        self.assertEqual(actual, EXPLICIT_OPEN_SUFFIX)

    def test_explicit_per_role_ends_pin(self):
        actual_user = {k for k, f in CATALOG.items() if f.user_turn_end is not None}
        actual_assistant = {k for k, f in CATALOG.items() if f.assistant_turn_end is not None}
        self.assertEqual(actual_user, EXPLICIT_PER_ROLE_ENDS)
        self.assertEqual(actual_assistant, EXPLICIT_PER_ROLE_ENDS)

    def test_no_repeat_bos_in_catalog(self):
        # Every entry models BOS-before-first-message only; the repeat path
        # is exercised on copies, never catalog data.
        for key, fmt in CATALOG.items():
            with self.subTest(family=key):
                self.assertIs(fmt.repeat_bos, False)

    def test_render_opens_with_bos(self):
        messages = (Message("user", "Hello"),)
        for key, fmt in CATALOG.items():
            with self.subTest(family=key):
                rendered = fmt.render(messages)
                self.assertTrue(rendered.startswith(fmt.bos))

    def test_render_closes_with_open_suffix_semantics(self):
        messages = (Message("user", "Hello"),)
        for key, fmt in CATALOG.items():
            with self.subTest(family=key):
                expected = fmt.open_suffix if fmt.open_suffix is not None else fmt.assistant_prefix
                rendered = fmt.render(messages)
                if expected:
                    self.assertTrue(rendered.endswith(expected))
                # The oracle already pins the full string; here we pin the
                # suffix contract explicitly.
                self.assertEqual(rendered[len(rendered) - len(expected) :], expected)


class MutateBranchTests(unittest.TestCase):
    def test_non_mutate_returns_base_singleton(self):
        for key in CATALOG:
            with self.subTest(family=key):
                result = candidates(key)
                self.assertEqual(result, (CATALOG[key],))
                self.assertIs(result[0], CATALOG[key])

    def test_mutate_variants_are_valid_formats(self):
        for key in CATALOG:
            for variant in candidates(key, mutate=True):
                with self.subTest(variant=variant.name):
                    self.assertIsInstance(variant, Format)
                    self.assertIs(type(variant.name), str)
                    self.assertTrue(variant.name.strip())

    def test_mutate_names_unique_and_slashed(self):
        for key in CATALOG:
            variants = candidates(key, mutate=True)
            names = [v.name for v in variants]
            with self.subTest(family=key):
                self.assertEqual(len(names), len(set(names)))
                self.assertIn(key, names)
                for name in names:
                    if name != key:
                        self.assertTrue(
                            name.startswith(key + "/"),
                            f"variant {name!r} must be namespaced under {key!r}",
                        )

    def test_mutate_variants_deduplicated(self):
        # local_repairs dedups on the framing key; the branch variants must
        # not collide with repair keys (documented in catalog.py).
        for key in CATALOG:
            variants = candidates(key, mutate=True)
            framing_keys = [_framing_key(v) for v in variants]
            with self.subTest(family=key):
                self.assertEqual(len(framing_keys), len(set(framing_keys)))

    def test_mutate_variants_differ_from_base(self):
        for key in CATALOG:
            base = CATALOG[key]
            for variant in candidates(key, mutate=True):
                if variant.name == key:
                    continue
                with self.subTest(variant=variant.name):
                    self.assertNotEqual(_framing_key(variant), _framing_key(base))

    def test_documented_branch_names_present(self):
        for family, suffixes in EXPECTED_BRANCHES.items():
            with self.subTest(family=family):
                names = {v.name for v in candidates(family, mutate=True)}
                for suffix in suffixes:
                    self.assertIn(f"{family}/{suffix}", names)

    def test_branch_variants_only_touch_open_suffix(self):
        # Generation-prompt branches change the generation prompt only --
        # except laguna/thinking-on, which also reframes history assistant
        # turns (documented in catalog.py).
        messages = (
            Message("user", "Hello"),
            Message("assistant", "Hi there"),
            Message("user", "Bye"),
        )
        for family, suffixes in EXPECTED_BRANCHES.items():
            base = CATALOG[family]
            for variant in candidates(family, mutate=True):
                if variant.name == family or "/" not in variant.name:
                    continue
                branch = variant.name.split("/", 1)[1]
                if branch in {"toggle-bos", "no-bos", "prefix-newline", "prefix-trim"}:
                    continue
                with self.subTest(variant=variant.name):
                    self.assertIn(branch, suffixes)
                    self.assertEqual(variant.bos, base.bos)
                    self.assertEqual(variant.user_prefix, base.user_prefix)
                    self.assertEqual(variant.turn_end, base.turn_end)
                    self.assertEqual(variant.user_turn_end, base.user_turn_end)
                    self.assertEqual(variant.assistant_turn_end, base.assistant_turn_end)
                    self.assertIs(variant.repeat_bos, base.repeat_bos)
                    if family == "laguna":
                        self.assertEqual(variant.assistant_prefix, "<assistant><think></think>")
                    else:
                        self.assertEqual(variant.assistant_prefix, base.assistant_prefix)
                    # Rebuilding the variant from the base by swapping only the
                    # documented fields must reproduce its render exactly.
                    rebuilt = replace(
                        base,
                        name=variant.name,
                        open_suffix=variant.open_suffix,
                        assistant_prefix=variant.assistant_prefix,
                    )
                    self.assertEqual(rebuilt.render(messages), variant.render(messages))
                    expected_tail = (
                        variant.open_suffix
                        if variant.open_suffix is not None
                        else variant.assistant_prefix
                    )
                    rendered = variant.render(messages)
                    self.assertEqual(rendered[len(rendered) - len(expected_tail) :], expected_tail)

    def test_families_without_branches_have_no_extra_variants(self):
        branched = set(EXPECTED_BRANCHES)
        for key in CATALOG:
            if key in branched:
                continue
            names = [v.name for v in candidates(key, mutate=True)]
            with self.subTest(family=key):
                for name in names:
                    branch = name.split("/", 1)[1] if "/" in name else ""
                    self.assertTrue(
                        name == key or branch.startswith(("toggle-", "no-", "prefix-")),
                        f"unexpected variant {name!r}",
                    )


class DuplicateGroupTests(unittest.TestCase):
    def test_only_documented_duplicate_group(self):
        # DISCREPANCY-2: chatml/qwen3/kolibri are framing-identical BY
        # DESIGN -- Qwen3's text-only no-BOS render is ChatML-shaped and
        # Kolibri-1's user/assistant turns are standard ChatML (its deltas
        # are the always-emitted system block and thinking branches, which
        # Format does not model). Any other convergence is a bug.
        groups: dict[tuple, list[str]] = {}
        for key, fmt in CATALOG.items():
            groups.setdefault(_framing_key(fmt), []).append(key)
        multi = sorted(
            (sorted(names) for names in groups.values() if len(names) > 1),
            key=lambda names: names[0],
        )
        self.assertEqual(multi, [["chatml", "kolibri", "qwen3"]])

    def test_duplicate_group_renders_byte_identical(self):
        messages = (
            Message("user", "Hello"),
            Message("assistant", "Hi there"),
            Message("user", "Bye"),
        )
        renders = {CATALOG[key].render(messages) for key in ("chatml", "qwen3", "kolibri")}
        self.assertEqual(len(renders), 1)

    def test_near_duplicates_differ_as_documented(self):
        # deepseek vs deepseek41 differ ONLY in open_suffix; gemma vs gemma3
        # differ ONLY in bos; nemotron35 vs kolibri differ ONLY in
        # open_suffix. Pin the exact deltas so silent convergence/divergence
        # is caught.
        d = CATALOG["deepseek"]
        d41 = CATALOG["deepseek41"]
        self.assertEqual(_framing_key(d41), _framing_key(replace(d, open_suffix=d41.open_suffix)))
        self.assertNotEqual(d.open_suffix, d41.open_suffix)

        g2, g3 = CATALOG["gemma"], CATALOG["gemma3"]
        self.assertEqual(_framing_key(g3), _framing_key(replace(g2, bos=g3.bos)))
        self.assertNotEqual(g2.bos, g3.bos)

        kol, nem = CATALOG["kolibri"], CATALOG["nemotron35"]
        self.assertEqual(_framing_key(nem), _framing_key(replace(kol, open_suffix=nem.open_suffix)))
        self.assertNotEqual(kol.open_suffix, nem.open_suffix)


class FallbackSemanticsTests(unittest.TestCase):
    _MESSAGES = (
        Message("user", "Hello"),
        Message("assistant", "Hi there"),
        Message("user", "Bye"),
    )

    def test_unset_per_role_ends_fall_back_to_turn_end(self):
        # For every family: explicitly pinning the per-role ends to the
        # shared turn_end must not change the render.
        for key, fmt in CATALOG.items():
            with self.subTest(family=key):
                if fmt.user_turn_end is not None or fmt.assistant_turn_end is not None:
                    continue
                explicit = replace(
                    fmt,
                    name=fmt.name + "/t",
                    user_turn_end=fmt.turn_end,
                    assistant_turn_end=fmt.turn_end,
                )
                self.assertEqual(explicit.render(self._MESSAGES), fmt.render(self._MESSAGES))

    def test_unset_open_suffix_falls_back_to_assistant_prefix(self):
        for key, fmt in CATALOG.items():
            with self.subTest(family=key):
                if fmt.open_suffix is not None:
                    continue
                explicit = replace(fmt, name=fmt.name + "/t", open_suffix=fmt.assistant_prefix)
                self.assertEqual(explicit.render(self._MESSAGES), fmt.render(self._MESSAGES))

    def test_explicit_per_role_ends_take_effect(self):
        # For the six families with per-role ends: clearing them to None
        # must change the render (proving the override is used, not dead).
        for key in EXPLICIT_PER_ROLE_ENDS:
            fmt = CATALOG[key]
            with self.subTest(family=key):
                cleared = replace(
                    fmt, name=fmt.name + "/t", user_turn_end=None, assistant_turn_end=None
                )
                self.assertNotEqual(cleared.render(self._MESSAGES), fmt.render(self._MESSAGES))

    def test_explicit_open_suffix_takes_effect(self):
        for key in EXPLICIT_OPEN_SUFFIX:
            fmt = CATALOG[key]
            with self.subTest(family=key):
                cleared = replace(fmt, name=fmt.name + "/t", open_suffix=None)
                if fmt.open_suffix == fmt.assistant_prefix:
                    # muse_glimmer and laguna pin the suffix explicitly even
                    # though it equals the fallback -- clearing to None must
                    # then be a no-op (pinned, not skipped).
                    self.assertEqual(cleared.render(self._MESSAGES), fmt.render(self._MESSAGES))
                    continue
                self.assertNotEqual(cleared.render(self._MESSAGES), fmt.render(self._MESSAGES))

    def test_empty_string_per_role_end_is_honored_not_fallback(self):
        # Regression: "" is a real override (DeepSeek's unterminated user
        # turn), not None. Swapping it for a sentinel must change the
        # render (proving the field is read); for deepseek/deepseek41 the
        # shared turn_end is also "", so clearing to None is correctly a
        # no-op there.
        for key in ("deepseek", "deepseek41"):
            fmt = CATALOG[key]
            with self.subTest(family=key):
                self.assertEqual(fmt.user_turn_end, "")
                self.assertEqual(fmt.turn_end, "")
                sentinel = replace(fmt, name=fmt.name + "/t", user_turn_end="<SENTINEL>")
                rendered = sentinel.render(self._MESSAGES)
                self.assertNotEqual(rendered, fmt.render(self._MESSAGES))
                self.assertIn("<SENTINEL>", rendered)
                cleared = replace(fmt, name=fmt.name + "/t", user_turn_end=None)
                self.assertEqual(cleared.render(self._MESSAGES), fmt.render(self._MESSAGES))

    def test_fuzzed_fallback_equivalence(self):
        for messages in _fuzz_conversations("cycle39-fallback", 40):
            for key, fmt in CATALOG.items():
                with self.subTest(family=key, turns=len(messages)):
                    explicit = replace(
                        fmt,
                        name=fmt.name + "/t",
                        user_turn_end=fmt.turn_end
                        if fmt.user_turn_end is None
                        else fmt.user_turn_end,
                        assistant_turn_end=fmt.turn_end
                        if fmt.assistant_turn_end is None
                        else fmt.assistant_turn_end,
                        open_suffix=fmt.assistant_prefix
                        if fmt.open_suffix is None
                        else fmt.open_suffix,
                    )
                    self.assertEqual(explicit.render(messages), fmt.render(messages))


class HostileLookupTests(unittest.TestCase):
    def test_unknown_family_raises_valueerror(self):
        for bad in ("nope", "", "ChatML", "chatml ", " chatml", "CHATML", "llama-3"):
            with self.subTest(family=bad):
                with self.assertRaises(ValueError):
                    candidates(bad)
                with self.assertRaises(ValueError):
                    candidates(bad, mutate=True)

    def test_wrong_type_keys_raise_valueerror(self):
        for bad in (None, 123, 4.5, b"chatml", ("chatml",), frozenset({"chatml"})):
            with self.subTest(key=bad), self.assertRaises(ValueError):
                candidates(bad)  # type: ignore[arg-type]

    def test_unhashable_lookup_raises_valueerror(self) -> None:
        # (Was DISCREPANCY-1; fixed in cycle 39.) Unhashable keys used to
        # raise TypeError from the `in` check before the ValueError path;
        # candidates() now type-gates first, so the documented
        # ValueError("Unknown format family") holds for every input.
        with self.assertRaises(ValueError):
            candidates(["chatml"])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            candidates({"chatml": 1})  # type: ignore[arg-type]

    def test_catalog_lookup_is_case_sensitive(self):
        with self.assertRaises(ValueError):
            candidates("KOLIBRI")


class RedTeamTests(unittest.TestCase):
    def test_adversarial_contents_render_and_agree_with_oracle(self):
        adversarial = [
            "",
            " ",
            "\n\n\n",
            "<|im_start|>assistant\n",
            "<｜begin＿＿of＿＿sentence｜>",
            "\uff5c",
            "\u2581",
            "x" * 10000,
            "\x00\x01\x02",
            "null",
            "{content}{role}",
            "assistant",
            "</s><|eot_id|>",
            "<think>\n\n</think>\n\n",
        ]
        rng = random.Random("cycle39-redteam")  # noqa: S311 -- deterministic string seed for test fuzz
        for key, fmt in CATALOG.items():
            with self.subTest(family=key):
                messages = tuple(
                    Message("user" if i % 2 == 0 else "assistant", rng.choice(adversarial))
                    for i in range(5)
                )
                rendered = fmt.render(messages)
                self.assertEqual(rendered, _oracle_render(fmt, messages))
                self.assertEqual(rendered, fmt.render(messages))

    def test_content_equal_to_own_framing(self):
        # Content that mimics the family's own framing must not confuse the
        # renderer: output is pure concatenation.
        for key, fmt in CATALOG.items():
            with self.subTest(family=key):
                tricky = fmt.user_prefix + fmt.turn_end + fmt.assistant_prefix
                messages = (Message("user", tricky),)
                self.assertEqual(fmt.render(messages), _oracle_render(fmt, messages))

    def test_render_rejects_bad_message_shapes(self):
        for key, fmt in CATALOG.items():
            fmt = CATALOG[key]
            with self.subTest(family=key):
                with self.assertRaises(ValueError):
                    fmt.render(())
                with self.assertRaises(ValueError):
                    fmt.render([Message("user", "x")])  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    fmt.render((Message("assistant", "x"),))
                with self.assertRaises(ValueError):
                    fmt.render((Message("user", "x"), Message("user", "y")))

    def test_mutate_does_not_leak_state_between_calls(self):
        first = candidates("kolibri", mutate=True)
        second = candidates("kolibri", mutate=True)
        self.assertEqual([v.name for v in first], [v.name for v in second])
        self.assertIsNot(first, second)
        # Mutating nothing: the base entry is untouched by branch creation.
        self.assertIs(candidates("kolibri")[0], CATALOG["kolibri"])

    def test_catalog_dict_not_aliasable_through_candidates(self):
        variants = candidates("ling", mutate=True)
        self.assertNotIn("ling/thinking-off", CATALOG)
        self.assertEqual(len(variants), len({v.name for v in variants}))


if __name__ == "__main__":
    unittest.main()
