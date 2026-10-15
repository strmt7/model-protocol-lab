"""Cycle 9 (continuous dev): Gemma 4 catalog entry.

Byte-exact render checks for google/gemma-4-31B-it
(https://huggingface.co/google/gemma-4-31B-it) -- full multi-turn render,
the open-reply suffix (exactly "<|turn>model\\n"), the single prepended
<bos>, the assistant->model role mapping, zero Gemma 3 turn-channel token
overlap on the same messages, and pins for the documented evidence tiers
(including the bos_token-field discrepancy) and the unmodeled system role.

render() requires a final user turn and only user/assistant roles, so the
fixture is user/assistant/user. The real template's system turn
(<|turn>system\\n{content}<turn|>\\n) is DOCUMENTED but not renderable here
-- the lab's Format does not model the system role (same convention as the
mistral_large [SYSTEM_PROMPT] frame); this file pins that boundary too.

Deterministic; no network, no model, no randomness.
"""

import unittest

from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.core import Message

_MESSAGES: tuple[Message, ...] = (
    Message("user", "Hello"),
    Message("assistant", "Hi there"),
    Message("user", "Bye"),
)

# Gemma 4 per the agreeing tier-2 implementations:
# <bos><|turn>user\n{content}<turn|>\n<|turn>model\n{content}<turn|>\n ...
# with a trailing open-reply suffix of "<|turn>model\n" (the first response
# template start anchor) -- note the trailing newline, and NO <|turn>\n vs
# model distinction for the generation prompt (unlike GLM-4's bare suffix).
_GEMMA4_BOS: str = "<bos>"
_GEMMA4_USER_OPEN: str = "<|turn>user\n"
_GEMMA4_MODEL_OPEN: str = "<|turn>model\n"
_GEMMA4_TURN_END: str = "<turn|>\n"

# Byte-exact render of _MESSAGES plus the trailing open-reply suffix.
GEMMA4_EXPECTED: str = (
    _GEMMA4_BOS
    + _GEMMA4_USER_OPEN
    + "Hello"
    + _GEMMA4_TURN_END
    + _GEMMA4_MODEL_OPEN
    + "Hi there"
    + _GEMMA4_TURN_END
    + _GEMMA4_USER_OPEN
    + "Bye"
    + _GEMMA4_TURN_END
    + _GEMMA4_MODEL_OPEN
)

# --- Evidence-tier pins (documented in the catalog entry's source comment)
# Tier 1 (primary, tokenizer_config.json read 2026-10-14):
_START_ANCHORS: tuple[str, str] = ("<|turn>model\n", "<tool_response|>")
_THINKING_OPEN: str = "<|channel>thought\n"
_THINKING_CLOSE: str = "<channel|>"
# Tier 2 (three independent byte-level implementations agreeing): the real
# template frames a system message as its own turn -- unlike Gemma 3, where
# the system folds into the first user turn. Not modelable by Format (only
# user/assistant roles exist), pinned here as a string constant so the
# documented behavior stays greppable and checkable.
_REAL_SYSTEM_FRAME_PREFIX: str = "<|turn>system\n"
_REAL_SYSTEM_FRAME_SUFFIX: str = "<turn|>\n"


class Gemma4RenderTests(unittest.TestCase):
    def test_gemma4_renders_byte_exact(self) -> None:
        self.assertEqual(CATALOG["gemma4"].render(_MESSAGES), GEMMA4_EXPECTED)

    def test_gemma4_open_suffix_is_exactly_model_turn(self) -> None:
        # The generation prompt ends with "<|turn>model\n" -- trailing
        # newline included. open_suffix is None so the renderer falls back to
        # assistant_prefix; the test pins both the fallback wiring and the
        # byte-exact tail of the render.
        fmt = CATALOG["gemma4"]
        self.assertIsNone(fmt.open_suffix)
        self.assertEqual(fmt.assistant_prefix, "<|turn>model\n")
        rendered = fmt.render(_MESSAGES)
        self.assertTrue(rendered.endswith("<|turn>model\n"))
        self.assertFalse(rendered.endswith("<|turn>model\n\n"))
        # The full multi-turn render ends with the system-free sequence:
        # user turn close, then the open model turn.
        self.assertTrue(rendered.endswith("<turn|>\n<|turn>model\n"))

    def test_gemma4_bos_prepended_exactly_once(self) -> None:
        # BOS discrepancy pin (see catalog comment): the config's
        # bos_token field is EMPTY (tier 1), but all three tier-2
        # implementations prepend the literal <bos> -- so does the catalog.
        fmt = CATALOG["gemma4"]
        self.assertEqual(fmt.bos, "<bos>")
        rendered = fmt.render(_MESSAGES)
        self.assertTrue(rendered.startswith("<bos>"))
        self.assertEqual(rendered.count("<bos>"), 1)
        # ...and no newline is glued to the BOS (Gemma 3 emits "<bos>\n").
        self.assertFalse(rendered.startswith("<bos>\n"))

    def test_gemma4_assistant_role_maps_to_model(self) -> None:
        fmt = CATALOG["gemma4"]
        self.assertEqual(fmt.user_prefix, "<|turn>user\n")
        self.assertEqual(fmt.assistant_prefix, "<|turn>model\n")
        rendered = fmt.render(_MESSAGES)
        self.assertIn("<|turn>model\nHi there<turn|>\n", rendered)
        # The word "assistant" appears nowhere in the rendered prompt.
        self.assertNotIn("assistant", rendered)

    def test_gemma4_shares_no_turn_tokens_with_gemma3(self) -> None:
        # Zero turn-channel token overlap on the same messages: the renders
        # differ, gemma4's uses the <|turn>/<turn|> channel, and gemma3's
        # render contains neither. (The "<bos>" glyph itself is shared --
        # gemma3 emits it as "<bos>\\n", gemma4 as bare "<bos>" -- which the
        # BOS-count test above distinguishes.)
        gemma4 = CATALOG["gemma4"].render(_MESSAGES)
        gemma3 = CATALOG["gemma3"].render(_MESSAGES)
        self.assertNotEqual(gemma4, gemma3)
        self.assertIn("<|turn>", gemma4)
        self.assertNotIn("<|turn>", gemma3)
        self.assertNotIn("<turn|>", gemma3)
        self.assertNotIn("<start_of_turn>", gemma4)

    def test_gemma4_pins_documented_evidence_tiers(self) -> None:
        # Tier 1 pin: the rendered generation prompt must end at the
        # documented response-template start anchor -- the open model turn.
        self.assertEqual(_START_ANCHORS[0], "<|turn>model\n")
        rendered = CATALOG["gemma4"].render(_MESSAGES)
        self.assertTrue(rendered.endswith(_START_ANCHORS[0]))
        # The tier-1 thinking channel and the tier-2 system frame are out of
        # the no-tools, no-system core: none of their tokens may leak into
        # the render, but their documented spellings are pinned above.
        for token in (
            _START_ANCHORS[1],
            _THINKING_OPEN,
            _THINKING_CLOSE,
            _REAL_SYSTEM_FRAME_PREFIX,
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, rendered)
        self.assertEqual(_REAL_SYSTEM_FRAME_SUFFIX, "<turn|>\n")

    def test_gemma4_system_role_is_out_of_model_scope(self) -> None:
        # The lab's Message only supports user/assistant roles, so a system
        # message cannot be constructed at all -- this pins the documented
        # boundary: the real template's <|turn>system\\n{content}<turn|>\\n
        # turn (tier 2) is not representable in Format.
        with self.assertRaises(ValueError):
            Message("system", "Be helpful")  # type: ignore[arg-type]

    def test_gemma4_fields_match_catalog_entry(self) -> None:
        fmt = CATALOG["gemma4"]
        self.assertEqual(fmt.name, "gemma4")
        self.assertEqual(fmt.turn_end, "<turn|>\n")
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertEqual(candidates("gemma4"), (fmt,))


class MutatePathTests(unittest.TestCase):
    def test_candidates_mutate_preserves_gemma4_frames(self) -> None:
        # Local repairs must not lose the Gemma 4 turn-channel frames, the
        # assistant-prefix fallback wiring for the open reply, or the
        # explicit BOS the tier-2 evidence requires. (assistant_prefix is
        # intentionally excluded: local_repairs deliberately creates
        # prefix-newline/prefix-trim variants of it.)
        for variant in candidates("gemma4", mutate=True):
            self.assertEqual(variant.user_prefix, "<|turn>user\n")
            self.assertEqual(variant.turn_end, "<turn|>\n")
            self.assertIsNone(variant.open_suffix)
        self.assertGreater(len(candidates("gemma4", mutate=True)), 1)


if __name__ == "__main__":
    unittest.main()
