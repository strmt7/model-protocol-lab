"""Cycle 6: catalog expansion to 14 families (Round 4 research R1).

Byte-exact render checks for the three remaining R1 families -- Kimi K3
(XTML, https://huggingface.co/moonshotai/Kimi-K3), DeepSeek V4.1-Flash
(https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), and GLM-4.5/4.6
(https://huggingface.co/zai-org/GLM-4.6) -- plus the open-reply suffix each
template emits, full-width codepoint guards for DeepSeek V4.1-Flash,
catalog-listing coverage (14 families), and a local-repairs (mutate)
regression check that the new per-role fields survive repair variants.

render() requires a final user turn, so the fixture is user/assistant/user:
the smallest conversation that exercises both per-role terminators and the
open suffix.

Deterministic; no network, no model, no randomness.
"""

import unittest

from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.core import Message

# The DeepSeek template's full-width bars/blocks, spelled as escapes so the
# expectations are exact regardless of editor encoding: U+FF5C FULLWIDTH
# VERTICAL BAR, U+2581 LOWER ONE EIGHTH BLOCK.
_B: str = "\uff5c"
_BLK: str = "\u2581"

_MESSAGES: tuple[Message, ...] = (
    Message("user", "Hello"),
    Message("assistant", "Hi there"),
    Message("user", "Bye"),
)


def _deepseek_bos() -> str:
    return f"<{_B}begin{_BLK}of{_BLK}sentence{_B}>"


def _deepseek_user() -> str:
    return f"<{_B}User{_B}>"


def _deepseek_assistant() -> str:
    return f"<{_B}Assistant{_B}>"


def _deepseek_eos() -> str:
    return f"<{_B}end{_BLK}of{_BLK}sentence{_B}>"


_KIMI3_USER_OPEN: str = '<|open|>message role="user"<|sep|>'
_KIMI3_USER_END: str = "<|close|>message<|sep|><|end_of_msg|>"
_KIMI3_ASSISTANT_OPEN: str = (
    '<|open|>message role="assistant"<|sep|>'
    "<|open|>think<|sep|><|close|>think<|sep|>"
    "<|open|>response<|sep|>"
)
_KIMI3_ASSISTANT_END: str = "<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>"
_KIMI3_OPEN_SUFFIX: str = '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>'


# Expected byte-exact renders of _MESSAGES plus the trailing open-reply
# suffix, keyed by catalog name.
EXPECTED: dict[str, tuple[str, str]] = {
    "kimi3": (
        _KIMI3_USER_OPEN
        + "Hello"
        + _KIMI3_USER_END
        + _KIMI3_ASSISTANT_OPEN
        + "Hi there"
        + _KIMI3_ASSISTANT_END
        + _KIMI3_USER_OPEN
        + "Bye"
        + _KIMI3_USER_END
        + _KIMI3_OPEN_SUFFIX,
        _KIMI3_OPEN_SUFFIX,
    ),
    "deepseek41": (
        _deepseek_bos()
        + _deepseek_user()
        + "Hello"
        + _deepseek_assistant()
        + "Hi there"
        + _deepseek_eos()
        + _deepseek_user()
        + "Bye"
        + _deepseek_assistant()
        + "<think>",
        _deepseek_assistant() + "<think>",
    ),
    "glm45": (
        "[gMASK]<sop>\n<|user|>\nHello\n"
        "<|assistant|>\n<think></think>\nHi there\n"
        "<|user|>\nBye\n<|assistant|>",
        "<|assistant|>",
    ),
}


class NewFamilyRenderTests(unittest.TestCase):
    def test_kimi3_renders_byte_exact(self) -> None:
        rendered, suffix = EXPECTED["kimi3"]
        self.assertEqual(CATALOG["kimi3"].render(_MESSAGES), rendered)
        self.assertTrue(CATALOG["kimi3"].render(_MESSAGES).endswith(suffix))

    def test_kimi3_uses_empty_think_block_and_ends_inside_think(self) -> None:
        rendered = CATALOG["kimi3"].render(_MESSAGES)
        # The text-only no-reasoning hypothesis carries an EMPTY think block
        # in the assistant turn ...
        self.assertIn("<|open|>think<|sep|><|close|>think<|sep|><|open|>response<|sep|>", rendered)
        # ... and the open suffix starts generation inside the think block.
        self.assertTrue(rendered.endswith(_KIMI3_OPEN_SUFFIX))
        self.assertNotIn("<|close|>think", _KIMI3_OPEN_SUFFIX)
        # No BOS.
        self.assertTrue(rendered.startswith(_KIMI3_USER_OPEN))

    def test_deepseek41_renders_byte_exact(self) -> None:
        rendered, suffix = EXPECTED["deepseek41"]
        self.assertEqual(CATALOG["deepseek41"].render(_MESSAGES), rendered)
        self.assertTrue(CATALOG["deepseek41"].render(_MESSAGES).endswith(suffix))

    def test_deepseek41_uses_fullwidth_bars_and_block(self) -> None:
        # The DeepSeek framing characters are U+FF5C (not ASCII '|') and
        # U+2581 (not ASCII '_'): an ASCII-bar lookalike would render the
        # wrong token IDs. Guard the exact codepoints, not just the string.
        rendered = CATALOG["deepseek41"].render(_MESSAGES)
        self.assertIn("\uff5c", rendered)
        self.assertIn("\u2581", rendered)
        self.assertNotIn("|", rendered)  # no ASCII pipe lookalikes
        self.assertIn("\uff5cbegin\u2581of\u2581sentence\uff5c", rendered)

    def test_deepseek41_differs_from_deepseek_only_in_open_suffix(self) -> None:
        # V4.1-Flash keeps the V3 frames; only the open suffix changes (chat
        # mode appends <think> to the bare <Assistant> transition token). The
        # V3 entry leaves open_suffix unset, so its effective suffix is the
        # assistant prefix (per Format.render's fallback rule).
        v3 = CATALOG["deepseek"]
        v41 = CATALOG["deepseek41"]
        v3_suffix = v3.open_suffix if v3.open_suffix is not None else v3.assistant_prefix
        self.assertEqual(v41.open_suffix, v3_suffix + "<think>")
        self.assertEqual(v41.render(_MESSAGES), v3.render(_MESSAGES) + "<think>")

    def test_glm45_renders_byte_exact(self) -> None:
        rendered, suffix = EXPECTED["glm45"]
        self.assertEqual(CATALOG["glm45"].render(_MESSAGES), rendered)
        self.assertTrue(CATALOG["glm45"].render(_MESSAGES).endswith(suffix))

    def test_glm45_think_block_and_newline_prefix(self) -> None:
        rendered = CATALOG["glm45"].render(_MESSAGES)
        # The mandatory think block is present and empty (text-only,
        # no-reasoning hypothesis).
        self.assertIn("<|assistant|>\n<think></think>\nHi there\n", rendered)
        # The [gMASK]<sop> prefix carries a newline that GLM-4-9B omits.
        self.assertTrue(rendered.startswith("[gMASK]<sop>\n"))
        # Per-message trailing newline, and the open suffix is the bare frame
        # (the generation prompt drops the frame newline, as in GLM-4).
        self.assertIn("<|user|>\nBye\n<|assistant|>", rendered)
        self.assertTrue(rendered.endswith("<|assistant|>"))
        self.assertFalse(rendered.endswith("<|assistant|>\n"))

    def test_all_new_families_open_suffix_matches_render_tail(self) -> None:
        for family, (_, suffix) in EXPECTED.items():
            with self.subTest(family=family):
                self.assertTrue(CATALOG[family].render(_MESSAGES).endswith(suffix))


class CatalogListingTests(unittest.TestCase):
    def test_catalog_holds_twenty_two_families(self) -> None:
        self.assertEqual(
            sorted(CATALOG),
            [
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
            ],
        )
        for family, fmt in CATALOG.items():
            with self.subTest(family=family):
                self.assertEqual(fmt.name, family)

    def test_candidates_returns_base_for_each_new_family(self) -> None:
        for family in EXPECTED:
            with self.subTest(family=family):
                self.assertEqual(candidates(family), (CATALOG[family],))


class MutatePathTests(unittest.TestCase):
    def test_candidates_mutate_returns_base_first_with_unique_variants(self) -> None:
        for family in EXPECTED:
            with self.subTest(family=family):
                variants = candidates(family, mutate=True)
                self.assertEqual(variants[0], CATALOG[family])
                names = [variant.name for variant in variants]
                self.assertEqual(len(names), len(set(names)))  # dedup left no clones
                for variant in variants:
                    variant.render(_MESSAGES)  # the mutate path must stay renderable

    def test_candidates_mutate_preserves_new_per_role_fields(self) -> None:
        # Local repairs must not lose the asymmetric Kimi K3 terminators, the
        # DeepSeek V4.1-Flash chat-mode open suffix, or the GLM-4.5/4.6 bare
        # open suffix.
        base_fields: dict[str, dict[str, str | None]] = {
            "kimi3": {
                "user_turn_end": _KIMI3_USER_END,
                "assistant_turn_end": _KIMI3_ASSISTANT_END,
                "open_suffix": _KIMI3_OPEN_SUFFIX,
            },
            "deepseek41": {
                "user_turn_end": "",
                "assistant_turn_end": _deepseek_eos(),
                "open_suffix": _deepseek_assistant() + "<think>",
            },
            "glm45": {
                "user_turn_end": None,
                "assistant_turn_end": None,
                "open_suffix": "<|assistant|>",
            },
        }
        for family, fields in base_fields.items():
            for variant in candidates(family, mutate=True):
                with self.subTest(family=family, variant=variant.name):
                    self.assertEqual(variant.user_turn_end, fields["user_turn_end"])
                    self.assertEqual(variant.assistant_turn_end, fields["assistant_turn_end"])
                    self.assertEqual(variant.open_suffix, fields["open_suffix"])


if __name__ == "__main__":
    unittest.main()
