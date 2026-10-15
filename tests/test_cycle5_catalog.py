"""Cycle 5: catalog expansion to 11 families (Round 4 research R1).

Byte-exact render checks for the 8 new families from
docs/RESEARCH_LOG.md ("Round 4 — Cycle 1 research / R1"), plus the
open-reply suffix each template emits, plus catalog-listing coverage and
regression coverage for the optional per-role Format fields added to
support them.
"""

import unittest

from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.core import Format, Message

# The DeepSeek template's full-width bars/blocks, spelled as escapes so the
# expectations are exact regardless of editor encoding: U+FF5C FULLWIDTH
# VERTICAL BAR, U+2581 LOWER ONE EIGHTH BLOCK.
_B = "\uff5c"
_BLK = "\u2581"

_MESSAGES = (Message("user", "Hello"), Message("assistant", "Hi there"), Message("user", "Bye"))


def _deepseek_bos() -> str:
    return f"<{_B}begin{_BLK}of{_BLK}sentence{_B}>"


def _deepseek_user() -> str:
    return f"<{_B}User{_B}>"


def _deepseek_assistant() -> str:
    return f"<{_B}Assistant{_B}>"


def _deepseek_eos() -> str:
    return f"<{_B}end{_BLK}of{_BLK}sentence{_B}>"


# Expected byte-exact renders of _MESSAGES plus the trailing open-reply
# suffix, keyed by catalog name. render() requires a final user turn, so the
# fixture is user/assistant/user: the smallest conversation that exercises
# both per-role terminators and the open suffix.
EXPECTED = {
    "qwen3": (
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\nHi there<|im_end|>\n"
        "<|im_start|>user\nBye<|im_end|>\n"
        "<|im_start|>assistant\n",
        "<|im_start|>assistant\n",
    ),
    "deepseek": (
        _deepseek_bos()
        + _deepseek_user()
        + "Hello"
        + _deepseek_assistant()
        + "Hi there"
        + _deepseek_eos()
        + _deepseek_user()
        + "Bye"
        + _deepseek_assistant(),
        _deepseek_assistant(),
    ),
    "mistral": (
        "<s>[INST] Hello [/INST]Hi there</s>[INST] Bye [/INST]",
        "",
    ),
    "phi4": (
        "<|im_start|>user<|im_sep|>Hello<|im_end|>"
        "<|im_start|>assistant<|im_sep|>Hi there<|im_end|>"
        "<|im_start|>user<|im_sep|>Bye<|im_end|>"
        "<|im_start|>assistant<|im_sep|>",
        "<|im_start|>assistant<|im_sep|>",
    ),
    "gemma3": (
        "<bos>\n<start_of_turn>user\nHello<end_of_turn>\n"
        "<start_of_turn>model\nHi there<end_of_turn>\n"
        "<start_of_turn>user\nBye<end_of_turn>\n"
        "<start_of_turn>model\n",
        "<start_of_turn>model\n",
    ),
    "llama4": (
        "<|begin_of_text|><|header_start|>user<|header_end|>\n\nHello<|eot|>"
        "<|header_start|>assistant<|header_end|>\n\nHi there<|eot|>"
        "<|header_start|>user<|header_end|>\n\nBye<|eot|>"
        "<|header_start|>assistant<|header_end|>\n\n",
        "<|header_start|>assistant<|header_end|>\n\n",
    ),
    "glm4": (
        "[gMASK]<sop><|user|>\nHello<|assistant|>\nHi there<|user|>\nBye<|assistant|>",
        "<|assistant|>",
    ),
    "kimi2": (
        "<|im_user|>user<|im_middle|>Hello<|im_end|>"
        "<|im_assistant|>assistant<|im_middle|>Hi there<|im_end|>"
        "<|im_user|>user<|im_middle|>Bye<|im_end|>"
        "<|im_assistant|>assistant<|im_middle|>",
        "<|im_assistant|>assistant<|im_middle|>",
    ),
}


class NewFamilyRenderTests(unittest.TestCase):
    def test_qwen3_renders_byte_exact(self):
        rendered, suffix = EXPECTED["qwen3"]
        self.assertEqual(CATALOG["qwen3"].render(_MESSAGES), rendered)
        self.assertTrue(CATALOG["qwen3"].render(_MESSAGES).endswith(suffix))

    def test_deepseek_renders_byte_exact(self):
        rendered, suffix = EXPECTED["deepseek"]
        self.assertEqual(CATALOG["deepseek"].render(_MESSAGES), rendered)
        self.assertTrue(CATALOG["deepseek"].render(_MESSAGES).endswith(suffix))

    def test_deepseek_uses_fullwidth_bars_and_block(self):
        # The DeepSeek framing characters are U+FF5C (not ASCII '|') and
        # U+2581 (not ASCII '_'): an ASCII-bar lookalike would render the
        # wrong token IDs. Guard the exact codepoints, not just the string.
        rendered = CATALOG["deepseek"].render(_MESSAGES)
        self.assertIn("\uff5c", rendered)
        self.assertIn("\u2581", rendered)
        self.assertNotIn("|", rendered)  # no ASCII pipe lookalikes
        self.assertIn("\uff5cbegin\u2581of\u2581sentence\uff5c", rendered)

    def test_mistral_renders_byte_exact(self):
        rendered, suffix = EXPECTED["mistral"]
        self.assertEqual(CATALOG["mistral"].render(_MESSAGES), rendered)
        self.assertTrue(CATALOG["mistral"].render(_MESSAGES).endswith(suffix))
        # The v0.3 release template has a space before [/INST] and no
        # leading space on the assistant turn.
        self.assertIn("[INST] Hello [/INST]", CATALOG["mistral"].render(_MESSAGES))
        self.assertIn("[/INST]Hi there</s>", CATALOG["mistral"].render(_MESSAGES))

    def test_phi4_renders_byte_exact(self):
        rendered, suffix = EXPECTED["phi4"]
        self.assertEqual(CATALOG["phi4"].render(_MESSAGES), rendered)
        self.assertTrue(CATALOG["phi4"].render(_MESSAGES).endswith(suffix))

    def test_gemma3_renders_byte_exact(self):
        rendered, suffix = EXPECTED["gemma3"]
        self.assertEqual(CATALOG["gemma3"].render(_MESSAGES), rendered)
        self.assertTrue(CATALOG["gemma3"].render(_MESSAGES).endswith(suffix))
        # Assistant role is "model", not "assistant".
        self.assertNotIn("assistant", CATALOG["gemma3"].render(_MESSAGES))

    def test_llama4_renders_byte_exact(self):
        rendered, suffix = EXPECTED["llama4"]
        self.assertEqual(CATALOG["llama4"].render(_MESSAGES), rendered)
        self.assertTrue(CATALOG["llama4"].render(_MESSAGES).endswith(suffix))

    def test_glm4_renders_byte_exact(self):
        rendered, suffix = EXPECTED["glm4"]
        self.assertEqual(CATALOG["glm4"].render(_MESSAGES), rendered)
        self.assertTrue(CATALOG["glm4"].render(_MESSAGES).endswith(suffix))
        # Frames are concatenated with no separators; the open suffix drops
        # the frame newline.
        self.assertIn("Hello<|assistant|>\nHi there", CATALOG["glm4"].render(_MESSAGES))

    def test_kimi2_renders_byte_exact(self):
        rendered, suffix = EXPECTED["kimi2"]
        self.assertEqual(CATALOG["kimi2"].render(_MESSAGES), rendered)
        self.assertTrue(CATALOG["kimi2"].render(_MESSAGES).endswith(suffix))
        # The role name appears twice per frame.
        self.assertIn("<|im_user|>user<|im_middle|>", CATALOG["kimi2"].render(_MESSAGES))

    def test_all_new_families_open_suffix_matches_render_tail(self):
        for family, (_, suffix) in EXPECTED.items():
            with self.subTest(family=family):
                self.assertTrue(CATALOG[family].render(_MESSAGES).endswith(suffix))


class CatalogListingTests(unittest.TestCase):
    def test_catalog_holds_nineteen_families(self):
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
                "kimi2",
                "kimi3",
                "kolibri",
                "lfm25",
                "llama3",
                "llama4",
                "mistral",
                "mistral_large",
                "phi4",
                "qwen3",
            ],
        )
        for family, fmt in CATALOG.items():
            with self.subTest(family=family):
                self.assertEqual(fmt.name, family)

    def test_candidates_returns_base_for_each_new_family(self):
        for family in EXPECTED:
            with self.subTest(family=family):
                self.assertEqual(candidates(family), (CATALOG[family],))

    def test_candidates_mutate_preserves_per_role_fields(self):
        # Local repairs must not lose the asymmetric terminators or the
        # GLM-4 open-suffix override.
        for variant in candidates("deepseek", mutate=True):
            self.assertEqual(variant.user_turn_end, "")
            self.assertEqual(variant.assistant_turn_end, _deepseek_eos())
        for variant in candidates("glm4", mutate=True):
            self.assertEqual(variant.open_suffix, "<|assistant|>")

    def test_candidates_unknown_family(self):
        with self.assertRaises(ValueError):
            candidates("nope")


class PerRoleFieldTests(unittest.TestCase):
    def test_unset_per_role_fields_fall_back(self):
        fmt = Format("fallback", "<B>", "<U>", "<A>", "<E>")
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertIsNone(fmt.open_suffix)
        self.assertEqual(
            fmt.render(_MESSAGES),
            "<B><U>Hello<E><A>Hi there<E><U>Bye<E><A>",
        )

    def test_empty_string_override_is_honored_not_fallback(self):
        # None means "use the default"; "" is a real override (the DeepSeek
        # user turn is genuinely unterminated).
        fmt = Format("empty-user-end", "", "<U>", "<A>", "<E>", user_turn_end="")
        self.assertEqual(fmt.render(_MESSAGES), "<U>Hello<A>Hi there<E><U>Bye<A>")

    def test_optional_fields_reject_non_strings(self):
        for kwargs in (
            {"user_turn_end": 5},
            {"assistant_turn_end": b"</s>"},
            {"open_suffix": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Format("bad", "", "<U>", "<A>", "<E>", **kwargs)


if __name__ == "__main__":
    unittest.main()
