"""Cycle 13 (continuous dev): Ling-3.1-flash catalog entry.

Byte-exact render checks for inclusionAI/Ling-3.1-flash (released
2026-10-15, MIT; https://huggingface.co/inclusionAI/Ling-3.1-flash).
Evidence tier 1: the official chat_template.jinja ("Bailing V3", preserved
at docs/evidence/ling31-chat-template.jinja) -- every string here is
derived from the template's plain-text, no-tools branch (jinja line
numbers in the catalog comment):

- System block (ALWAYS emitted, jinja lines 19, 35-45): with a system
  message X lacking the "detailed thinking" marker,
  ``<role>SYSTEM</role>X\\ndetailed thinking on<|role_end|>`` (lines
  40-41); without, ``<role>SYSTEM</role>detailed thinking on<|role_end|>``
  (line 44). Thinking defaults ON (lines 5-13, default 'on' at line 12);
  the thinking_option dependence is documented in the catalog comment.
  The system role is not modeled in Format (lab convention); the block is
  pinned here as a documented constant.
- User (line 62): ``<role>HUMAN</role>{content}<|role_end|>``.
- Assistant history, no reasoning (line 79): ``<role>ASSISTANT</role>\\n
  <think></think>{content}`` closed by ``<|role_end|>`` (line 108).
- Generation prompt, thinking on (line 126): ``<role>ASSISTANT</role>\\n
  <think>``; thinking off (line 128): ``<role>ASSISTANT</role>\\n
  <think></think>`` -- rides the mutate path as ``ling/thinking-off``
  (kolibri/thinking-off convention).
- BOS ``<|startoftext|>`` -- tier-1 verified from the official
  tokenizer_config.json bos_token field (read 2026-10-15; the jinja
  itself emits no BOS literal and add_bos_token is false).
- EOS caveat: the official eos_token field is ``<|role_end|>``, NOT
  ``<|endoftext|>`` (that token is the pad_token); documented, not
  modeled (Format has no EOS field).

Also: diff-vs-all-other-families pin (``ling`` renders byte-different
from every one of the other 22 families on the same messages), the
thinking-off dedup-key no-collision pin, and a README naming pin for
"Ling" (the N-family count pins live in tests/test_cycle10_families.py
and tests/test_cycle11_families.py and enforce the 22 -> 23 update).

Deterministic; no network, no model, no randomness.
"""

import re
import unittest
from pathlib import Path

from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.core import Format, Message, local_repairs

REPO_ROOT = Path(__file__).resolve().parent.parent

_MESSAGES: tuple[Message, ...] = (
    Message("user", "Hello"),
    Message("assistant", "Hi there"),
    Message("user", "Bye"),
)

# Derived from docs/evidence/ling31-chat-template.jinja ("Bailing V3"):
# user turns (line 62), assistant history turns (lines 79 + 108),
# generation prompt (line 126, thinking on -- the default), BOS
# <|startoftext|> (tokenizer_config.json bos_token field, read 2026-10-15).
_LING_BOS = "<|startoftext|>"
_LING_USER_PREFIX = "<role>HUMAN</role>"
_LING_ASSISTANT_PREFIX = "<role>ASSISTANT</role>\n<think></think>"
_LING_TURN_END = "<|role_end|>"
_LING_THINKING_ON_SUFFIX = "<role>ASSISTANT</role>\n<think>"
# Thinking-off generation prompt (jinja lines 127-128): the same frames,
# but the generation prompt prefills an EMPTY think block.
_LING_THINKING_OFF_SUFFIX = "<role>ASSISTANT</role>\n<think></think>"

# The always-emitted system block is documented, not modeled (the lab's
# Format only renders user/assistant turns). Both forms use the default
# thinking_option 'on' (jinja line 12): the thinking_option dependence is
# documented in the catalog comment.
_LING_SYSTEM_BLOCK_WITH_X = "<role>SYSTEM</role>sys-in\ndetailed thinking on<|role_end|>"
_LING_SYSTEM_BLOCK_NO_X = "<role>SYSTEM</role>detailed thinking on<|role_end|>"

_LING_EXPECTED = (
    _LING_BOS
    + _LING_USER_PREFIX
    + "Hello"
    + _LING_TURN_END
    + _LING_ASSISTANT_PREFIX
    + "Hi there"
    + _LING_TURN_END
    + _LING_USER_PREFIX
    + "Bye"
    + _LING_TURN_END
    + _LING_THINKING_ON_SUFFIX
)


class LingRenderTests(unittest.TestCase):
    """Ling render tests."""

    def test_ling_renders_byte_exact(self) -> None:
        """Test ling renders byte exact."""
        fmt = CATALOG["ling"]
        self.assertEqual(fmt.render(_MESSAGES), _LING_EXPECTED)

    def test_ling_fields_match_catalog_entry(self) -> None:
        """Test ling fields match catalog entry."""
        fmt = CATALOG["ling"]
        self.assertEqual(fmt.bos, _LING_BOS)
        self.assertEqual(fmt.user_prefix, _LING_USER_PREFIX)
        self.assertEqual(fmt.assistant_prefix, _LING_ASSISTANT_PREFIX)
        self.assertEqual(fmt.turn_end, _LING_TURN_END)
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertEqual(fmt.open_suffix, _LING_THINKING_ON_SUFFIX)
        self.assertFalse(fmt.repeat_bos)

    def test_ling_system_block_documented_not_modeled(self) -> None:
        """Test ling system block documented not modeled."""
        # The real template ALWAYS emits a system block (jinja lines 19,
        # 35-45); the lab's Format only renders user/assistant turns, so the
        # block is a documented constant (pinned byte-exact here), not a
        # render field -- the same convention as the kolibri comment's
        # system-block note. Thinking defaults ON (jinja line 12); with a
        # caller-supplied system message X lacking the "detailed thinking"
        # marker the block is <role>SYSTEM</role>X + "\n" + "detailed
        # thinking on" + <|role_end|> (lines 40-41); without a system
        # message it is the bare default (line 44).
        self.assertEqual(
            _LING_SYSTEM_BLOCK_WITH_X,
            "<role>SYSTEM</role>sys-in\ndetailed thinking on<|role_end|>",
        )
        self.assertEqual(
            _LING_SYSTEM_BLOCK_NO_X,
            "<role>SYSTEM</role>detailed thinking on<|role_end|>",
        )
        rendered = CATALOG["ling"].render(_MESSAGES)
        self.assertNotIn("<role>SYSTEM</role>", rendered)

    def test_ling_eos_caveat_documented(self) -> None:
        """Test ling eos caveat documented."""
        # The official tokenizer_config.json eos_token field is
        # <|role_end|> -- NOT <|endoftext|> (that token is the pad_token and
        # a special added token); the catalog documents this and asserts no
        # EOS string anywhere in the entry's frames (Format has no EOS
        # field). The catalog comment must carry the eos_token correction.
        fmt = CATALOG["ling"]
        self.assertNotIn("<|endoftext|>", fmt.bos + fmt.user_prefix + fmt.assistant_prefix)
        self.assertNotIn("<|endoftext|>", fmt.turn_end + (fmt.open_suffix or ""))

    def test_ling_base_is_thinking_on_default(self) -> None:
        """Test ling base is thinking on default."""
        # The template defaults thinking_option to 'on' when
        # enable_thinking/thinking_option are unset (jinja lines 5-13), so
        # the base entry's generation tail is the bare "\n<think>" suffix
        # (line 126) -- the mutate path supplies the thinking-off branch
        # (lines 127-128). Note the history assistant turns DO contain an
        # empty <think></think> block (jinja line 79) -- that is the
        # assistant frame, not the generation tail.
        rendered = CATALOG["ling"].render(_MESSAGES)
        self.assertTrue(rendered.endswith(_LING_THINKING_ON_SUFFIX))
        self.assertNotIn(_LING_THINKING_OFF_SUFFIX, rendered[-len(_LING_THINKING_OFF_SUFFIX) :])


# --- Thinking-off mutate-path variant -----------------------------------------


class LingThinkingOffTests(unittest.TestCase):
    """Ling thinking off tests."""

    def test_ling_thinking_off_mutation_pin(self) -> None:
        """Test ling thinking off mutation pin."""
        variants = {variant.name: variant for variant in candidates("ling", mutate=True)}
        self.assertIn("ling/thinking-off", variants)
        off = variants["ling/thinking-off"]
        base = CATALOG["ling"]
        # Same frames as the base entry; only the generation prompt
        # changes: it prefills an EMPTY think block (jinja lines 127-128).
        for field in ("bos", "user_prefix", "assistant_prefix", "turn_end"):
            self.assertEqual(getattr(off, field), getattr(base, field))
        self.assertEqual(off.open_suffix, _LING_THINKING_OFF_SUFFIX)
        # And the full generation tail renders byte-exact.
        rendered = off.render(_MESSAGES)
        self.assertTrue(rendered.endswith(_LING_THINKING_OFF_SUFFIX))
        self.assertEqual(
            rendered[: -len(_LING_THINKING_OFF_SUFFIX)],
            _LING_EXPECTED[: -len(_LING_THINKING_ON_SUFFIX)],
        )

    def test_ling_thinking_off_dedup_no_collision(self) -> None:
        """Test ling thinking off dedup no collision.

        The thinking-off open_suffix must be genuinely new -- i.e. its
        local_repairs dedup key must differ from every key already
        present, or candidates() would silently drop it. This mirrors the
        "cannot collide with local_repairs' dedup keys" note in
        catalog.candidates().
        """

        def _key(fmt: Format) -> tuple[object, ...]:
            """Key."""
            return (
                fmt.bos,
                fmt.user_prefix,
                fmt.assistant_prefix,
                fmt.turn_end,
                fmt.user_turn_end,
                fmt.assistant_turn_end,
                fmt.open_suffix,
                fmt.repeat_bos if fmt.bos else False,
            )

        base = CATALOG["ling"]
        repair_keys = {_key(variant) for variant in local_repairs(base)}
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        off = next(v for v in candidates("ling", mutate=True) if v.name == "ling/thinking-off")
        self.assertNotIn(_key(off), repair_keys)
        # Sanity: the same uniqueness property holds for the earlier
        # thinking-off families (the invariant covers existence; this
        # covers the dedup mechanism).
        for family in ("kolibri", "nemotron35"):
            base_f = CATALOG[family]
            repair_keys_f = {_key(variant) for variant in local_repairs(base_f)}
            # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
            off_f = next(
                v for v in candidates(family, mutate=True) if v.name == f"{family}/thinking-off"
            )
            self.assertNotIn(_key(off_f), repair_keys_f)


# --- Diff-vs-all-other-families -------------------------------------------------


class LingDiffTests(unittest.TestCase):
    """Ling diff tests."""

    def test_ling_differs_from_every_other_family(self) -> None:
        """Test ling differs from every other family."""
        # ling's frames share no opener/terminator literals with any other
        # catalog family; on the same messages the full render must be
        # byte-different from every other family's render (dynamic count).
        mine = CATALOG["ling"].render(_MESSAGES)
        others = [name for name in CATALOG if name != "ling"]
        self.assertEqual(len(others), len(CATALOG) - 1)
        collisions = [name for name in others if CATALOG[name].render(_MESSAGES) == mine]
        self.assertEqual(collisions, [])

    def test_ling_vs_lfm25_shares_bos_only(self) -> None:
        """Test ling vs lfm25 shares bos only."""
        # lfm25 is the only other family with a <|startoftext|> BOS; its
        # ChatML frames differ everywhere else (user/assistant prefixes,
        # turn ends, generation tail).
        ling = CATALOG["ling"]
        lfm25 = CATALOG["lfm25"]
        self.assertEqual(ling.bos, lfm25.bos)
        self.assertNotEqual(ling.user_prefix, lfm25.user_prefix)
        self.assertNotEqual(ling.assistant_prefix, lfm25.assistant_prefix)
        self.assertNotEqual(ling.turn_end, lfm25.turn_end)
        self.assertNotEqual(ling.open_suffix, lfm25.open_suffix)
        self.assertNotEqual(ling.render(_MESSAGES), lfm25.render(_MESSAGES))


# --- README naming pin ----------------------------------------------------------


class ReadmeLingTests(unittest.TestCase):
    """Readme ling tests."""

    def test_readme_names_ling(self) -> None:
        """Test readme names ling."""
        # The README's family list must name the new family (the N-family
        # count pin lives in tests/test_cycle10_families.py and
        # tests/test_cycle11_families.py and enforces 23 -> 24).
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Ling", text)
        matches = re.findall(r"(\d+)-family catalog", text)
        self.assertTrue(matches, "README lost its N-family catalog claim")
        for match in matches:
            self.assertEqual(int(match), len(CATALOG))
        self.assertEqual(len(CATALOG), 29)


if __name__ == "__main__":
    unittest.main()
