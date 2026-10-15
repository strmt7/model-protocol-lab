"""Cycle 7: catalog expansion to 16 families (Round 4 research, Cycle 3).

Byte-exact render checks for the two families re-verified 2026-10-14 --
Mistral-Large-Instruct-2411 ("Basic Instruct Template (V7)",
https://huggingface.co/mistralai/Mistral-Large-Instruct-2411) and
GLM-5.3-Flash (https://huggingface.co/zai-org/GLM-5.3-Flash) -- plus the
open-reply suffix each template emits, the two GLM-5.3-Flash whitespace
deltas vs the R1 record, the Mistral-Large vs Mistral v0.3 frame
differences, catalog-listing coverage (16 families), and a local-repairs
(mutate) regression check that the new per-role fields and open-suffix
overrides survive repair variants.

render() requires a final user turn, so the fixture is user/assistant/user:
the smallest conversation that exercises both per-role terminators and the
open suffix.

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

# Mistral-Large-Instruct-2411 "Basic Instruct Template (V7)" (model card):
# <s>[SYSTEM_PROMPT] {sys}[/SYSTEM_PROMPT][INST] {u}[/INST] {a}</s>
# [INST] {u2}[/INST] -- space after the [SYSTEM_PROMPT] opening tag, NO
# space before [/INST], LEADING space on the assistant turn, and the
# generation prompt ends right after [/INST] with NO trailing space.
_ML_USER_OPEN: str = "[INST] "
_ML_USER_END: str = "[/INST]"
_ML_ASSISTANT_OPEN: str = " "
_ML_ASSISTANT_END: str = "</s>"
_ML_OPEN_SUFFIX: str = ""

# GLM-5.3-Flash: BOS prefix [gMASK]<sop> with NO trailing newline (the first
# whitespace delta vs GLM-4.5/4.6), user <|user|>{content} with no trailing
# newline, assistant <|assistant|><think></think>{content} with NO newline
# after <|assistant|> (the second whitespace delta), open suffix
# <|assistant|><think> (generation starts inside the think block).
_GLM53_PREFIX: str = "[gMASK]<sop>"
_GLM53_USER_OPEN: str = "<|user|>"
_GLM53_ASSISTANT_OPEN: str = "<|assistant|><think></think>"
_GLM53_OPEN_SUFFIX: str = "<|assistant|><think>"

# Expected byte-exact renders of _MESSAGES plus the trailing open-reply
# suffix, keyed by catalog name.
EXPECTED: dict[str, tuple[str, str]] = {
    "mistral_large": (
        "<s>"
        + _ML_USER_OPEN
        + "Hello"
        + _ML_USER_END
        + _ML_ASSISTANT_OPEN
        + "Hi there"
        + _ML_ASSISTANT_END
        + _ML_USER_OPEN
        + "Bye"
        + _ML_USER_END
        + _ML_OPEN_SUFFIX,
        _ML_OPEN_SUFFIX,
    ),
    "glm53": (
        _GLM53_PREFIX
        + _GLM53_USER_OPEN
        + "Hello"
        + _GLM53_ASSISTANT_OPEN
        + "Hi there"
        + _GLM53_USER_OPEN
        + "Bye"
        + _GLM53_OPEN_SUFFIX,
        _GLM53_OPEN_SUFFIX,
    ),
}


class NewFamilyRenderTests(unittest.TestCase):
    def test_mistral_large_renders_byte_exact(self) -> None:
        rendered, _ = EXPECTED["mistral_large"]
        self.assertEqual(CATALOG["mistral_large"].render(_MESSAGES), rendered)

    def test_mistral_large_generation_prompt_has_no_trailing_space(self) -> None:
        # The model card warns: "Be careful with subtle missing or trailing
        # white spaces!" The generation prompt ends right after [/INST] --
        # the space precedes the assistant response, so the open suffix must
        # be the EXPLICIT empty string, not the assistant-prefix fallback
        # (" "), which would append a spurious trailing space.
        fmt = CATALOG["mistral_large"]
        self.assertEqual(fmt.open_suffix, "")
        self.assertIsNotNone(fmt.open_suffix)
        rendered = fmt.render(_MESSAGES)
        self.assertTrue(rendered.endswith("[INST] Bye[/INST]"))
        self.assertFalse(rendered.endswith(" "))

    def test_mistral_large_differs_from_mistral_v03_frames(self) -> None:
        # Mistral v0.3 (release) vs Mistral-Large-Instruct-2411 (V7): v0.3 has
        # a space BEFORE [/INST] and NO leading space on the assistant turn;
        # V7 has no space before [/INST] and a LEADING-space assistant turn.
        # Both end the generation prompt after [/INST] with no trailing
        # space, but v0.3 gets there via the assistant-prefix fallback (its
        # assistant prefix is "") while V7 needs the explicit "" override
        # (its assistant prefix is " ").
        v03 = CATALOG["mistral"]
        large = CATALOG["mistral_large"]
        self.assertEqual(v03.user_turn_end, " [/INST]")
        self.assertEqual(large.user_turn_end, "[/INST]")
        self.assertEqual(v03.assistant_prefix, "")
        self.assertEqual(large.assistant_prefix, " ")
        self.assertIsNone(v03.open_suffix)
        self.assertEqual(large.open_suffix, "")
        self.assertFalse(v03.render(_MESSAGES).endswith(" "))
        self.assertFalse(large.render(_MESSAGES).endswith(" "))

    def test_glm53_renders_byte_exact(self) -> None:
        rendered, suffix = EXPECTED["glm53"]
        self.assertEqual(CATALOG["glm53"].render(_MESSAGES), rendered)
        self.assertTrue(CATALOG["glm53"].render(_MESSAGES).endswith(suffix))

    def test_glm53_whitespace_deltas_vs_glm45(self) -> None:
        rendered = CATALOG["glm53"].render(_MESSAGES)
        # Delta 1: NO newline after the [gMASK]<sop> prefix (GLM-4.5/4.6
        # emit [gMASK]<sop>\n).
        self.assertTrue(rendered.startswith("[gMASK]<sop><|user|>"))
        self.assertFalse(rendered.startswith("[gMASK]<sop>\n"))
        # Delta 2: NO newline after <|assistant|> (GLM-4.5/4.6 frame
        # <|assistant|>\n{content}\n); the think block is mandatory and empty
        # in the text-only no-reasoning hypothesis.
        self.assertIn("<|assistant|><think></think>Hi there", rendered)
        self.assertNotIn("<|assistant|>\n", rendered)
        # The released template auto-injects a
        # <|system|>Reasoning Effort: Max line; the base hypothesis
        # deliberately omits it (deterministic base, like DeepSeek
        # V4.1-Flash's effort-preamble omission).
        self.assertNotIn("Reasoning Effort", rendered)
        self.assertNotIn("<|system|>", rendered)
        # Generation starts inside the think block.
        self.assertTrue(rendered.endswith("<|assistant|><think>"))

    def test_glm53_differs_from_glm45_only_in_whitespace_and_think(self) -> None:
        old = CATALOG["glm45"]
        new = CATALOG["glm53"]
        self.assertEqual(new.bos, old.bos.rstrip("\n"))
        self.assertEqual(new.user_prefix, old.user_prefix.rstrip("\n"))
        self.assertEqual(new.open_suffix, "<|assistant|><think>")
        self.assertEqual(old.open_suffix, "<|assistant|>")

    def test_all_new_families_open_suffix_matches_render_tail(self) -> None:
        for family, (_, suffix) in EXPECTED.items():
            with self.subTest(family=family):
                rendered = CATALOG[family].render(_MESSAGES)
                # endswith("") is vacuous, so slice the tail explicitly: the
                # empty-string open suffix is pinned byte-exact too.
                self.assertEqual(rendered[len(rendered) - len(suffix) :], suffix)


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
                "kimi2",
                "kimi3",
                "kolibri",
                "laguna",
                "lfm25",
                "llama3",
                "llama4",
                "mistral",
                "mistral_large",
                "muse_glimmer",
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
    def test_candidates_mutate_preserves_new_family_fields(self) -> None:
        # Local repairs must not lose the Mistral-Large explicit empty open
        # suffix or its asymmetric terminators, nor the GLM-5.3-Flash open
        # suffix that starts generation inside the think block.
        for variant in candidates("mistral_large", mutate=True):
            self.assertEqual(variant.open_suffix, "")
            self.assertEqual(variant.user_turn_end, "[/INST]")
            self.assertEqual(variant.assistant_turn_end, "</s>")
        for variant in candidates("glm53", mutate=True):
            self.assertEqual(variant.open_suffix, "<|assistant|><think>")
        self.assertGreater(len(candidates("mistral_large", mutate=True)), 1)
        self.assertGreater(len(candidates("glm53", mutate=True)), 1)


if __name__ == "__main__":
    unittest.main()
