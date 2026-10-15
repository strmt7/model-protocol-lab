"""Cycle 10 (continuous dev): Kolibri-1 and LFM2.5-24B-A2B catalog entries.

Byte-exact render checks for two October-2026 families, following the
conventions of tests/test_cycle9_families.py:

- ``kolibri`` (Aleph-Alpha Kolibri-1, released 2026-10-03, Apache 2.0;
  https://huggingface.co/Aleph-Alpha/Kolibri-1). Evidence tier 1: the
  official tokenizer_config.json chat_template, byte-verified by an
  independent fork. The text-only no-system core is byte-identical to
  chatml (the always-emitted system block, the effort-dependent sentence,
  and the thinking on/off branches are documented in the catalog comment;
  only the byte-verified system-block prefix is pinned here).
- ``lfm25`` (LiquidAI LFM2.5-24B-A2B, released 2026-10-13;
  https://huggingface.co/LiquidAI/LFM2.5-24B-A2B). Evidence tier 2: the
  official model card plus the HF blog rendered example.

Also: the deepseek41 system-framing re-verification (the catalog claims
V4.1 frames system messages as <System>{content}, while third-party docs
of the sibling V4 encoder render system text raw -- re-checked against the
primary source 2026-10-15 and confirmed), and a README family-count
consistency pin (the README's "N-family catalog" claim must match
len(CATALOG)).

Deterministic; no network, no model, no randomness.
"""

import re
import unittest
from pathlib import Path

from model_protocol_lab.catalog import _DS_BAR, CATALOG, candidates
from model_protocol_lab.core import Message

REPO_ROOT = Path(__file__).resolve().parent.parent

_MESSAGES: tuple[Message, ...] = (
    Message("user", "Hello"),
    Message("assistant", "Hi there"),
    Message("user", "Bye"),
)

# --- Kolibri-1 pins ---------------------------------------------------------
# Official template (tier 1): standard ChatML user/assistant frames, no BOS.
# Thinking ON (default): the generation prompt ends at the bare assistant
# frame -- the model opens <think> itself. Thinking OFF: the generation
# prompt prefills an empty think block.
_KOLIBRI_USER_OPEN = "<|im_start|>user\n"
_KOLIBRI_ASSISTANT_OPEN = "<|im_start|>assistant\n"
_KOLIBRI_TURN_END = "<|im_end|>\n"
_KOLIBRI_THINKING_ON_SUFFIX = "<|im_start|>assistant\n"
_KOLIBRI_THINKING_OFF_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

# The real template ALWAYS emits a system block, even with no system
# message: <|im_start|>system\n# Reasoning effort\n\n<SENTENCE>[+ user
# system content]<|im_end|>\n. SENTENCE is effort-dependent; the lab has no
# effort knob, so the catalog documents the DEFAULT-effort (unset -> "high")
# behavior. Only the byte-verified sentence prefix is pinned here -- the
# wording beyond it is deliberately not asserted, and the system role is not
# modelable by Format (same convention as mistral_large/gemma4).
_KOLIBRI_SYSTEM_BLOCK_VERIFIED_PREFIX = (
    "<|im_start|>system\n# Reasoning effort\n\n"
    "Reasoning effort is set to high. Think carefully through"
)

KOLIBRI_EXPECTED = (
    _KOLIBRI_USER_OPEN
    + "Hello"
    + _KOLIBRI_TURN_END
    + _KOLIBRI_ASSISTANT_OPEN
    + "Hi there"
    + _KOLIBRI_TURN_END
    + _KOLIBRI_USER_OPEN
    + "Bye"
    + _KOLIBRI_TURN_END
    + _KOLIBRI_THINKING_ON_SUFFIX
)

KOLIBRI_THINKING_OFF_EXPECTED = (
    _KOLIBRI_USER_OPEN
    + "Hello"
    + _KOLIBRI_TURN_END
    + _KOLIBRI_ASSISTANT_OPEN
    + "Hi there"
    + _KOLIBRI_TURN_END
    + _KOLIBRI_USER_OPEN
    + "Bye"
    + _KOLIBRI_TURN_END
    + _KOLIBRI_THINKING_OFF_SUFFIX
)

# --- LFM2.5-24B-A2B pins ----------------------------------------------------
# Official card render (tier 2): <|startoftext|> BOS, ChatML-like frames, and
# -- unlike LFM2 -- the generation prompt appends <think>\n (the 24B variant
# always thinks).
_LFM25_BOS = "<|startoftext|>"
_LFM25_USER_OPEN = "<|im_start|>user\n"
_LFM25_ASSISTANT_OPEN = "<|im_start|>assistant\n"
_LFM25_TURN_END = "<|im_end|>\n"
_LFM25_OPEN_SUFFIX = "<|im_start|>assistant\n<think>\n"

# Documented caveats (tier 2): whether <|startoftext|> comes from the
# bos_token field or a template literal is UNVERIFIED (byte-identical either
# way); whether the system block is auto-injected or merely illustrative is
# UNVERIFIED -- and the system role is not modeled in Format regardless.
# Pinned here as documented constants, exactly like the gemma4 system frame.
_LFM25_SYSTEM_BLOCK_EXAMPLE = (
    "<|im_start|>system\nYou are a helpful assistant trained by Liquid AI.<|im_end|>\n"
)

LFM25_EXPECTED = (
    _LFM25_BOS
    + _LFM25_USER_OPEN
    + "Hello"
    + _LFM25_TURN_END
    + _LFM25_ASSISTANT_OPEN
    + "Hi there"
    + _LFM25_TURN_END
    + _LFM25_USER_OPEN
    + "Bye"
    + _LFM25_TURN_END
    + _LFM25_OPEN_SUFFIX
)


class KolibriRenderTests(unittest.TestCase):
    def test_kolibri_renders_byte_exact(self) -> None:
        self.assertEqual(CATALOG["kolibri"].render(_MESSAGES), KOLIBRI_EXPECTED)

    def test_kolibri_base_is_thinking_on(self) -> None:
        # The base entry is the DEFAULT (thinking-on) branch: the generation
        # prompt ends at the bare assistant frame, and the model opens
        # <think> itself. open_suffix is None so the renderer falls back to
        # assistant_prefix; the test pins the wiring and the byte-exact tail.
        fmt = CATALOG["kolibri"]
        self.assertIsNone(fmt.open_suffix)
        rendered = fmt.render(_MESSAGES)
        self.assertTrue(rendered.endswith(_KOLIBRI_THINKING_ON_SUFFIX))
        self.assertNotIn("<think>", rendered)

    def test_kolibri_has_no_bos(self) -> None:
        fmt = CATALOG["kolibri"]
        self.assertEqual(fmt.bos, "")
        rendered = fmt.render(_MESSAGES)
        self.assertTrue(rendered.startswith("<|im_start|>user\n"))

    def test_kolibri_fields_match_catalog_entry(self) -> None:
        fmt = CATALOG["kolibri"]
        self.assertEqual(fmt.name, "kolibri")
        self.assertEqual(fmt.user_prefix, "<|im_start|>user\n")
        self.assertEqual(fmt.assistant_prefix, "<|im_start|>assistant\n")
        self.assertEqual(fmt.turn_end, "<|im_end|>\n")
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertEqual(candidates("kolibri"), (fmt,))

    def test_kolibri_system_block_is_documented_not_rendered(self) -> None:
        # The always-emitted system block (and its effort-dependent
        # sentence) is documented in the catalog comment but not modeled:
        # the system role is unmodelable, and only the byte-verified prefix
        # is asserted -- never the sentence wording beyond it.
        rendered = CATALOG["kolibri"].render(_MESSAGES)
        self.assertNotIn("<|im_start|>system", rendered)
        self.assertNotIn("# Reasoning effort", rendered)
        self.assertEqual(
            _KOLIBRI_SYSTEM_BLOCK_VERIFIED_PREFIX,
            "<|im_start|>system\n# Reasoning effort\n\n"
            "Reasoning effort is set to high. Think carefully through",
        )
        # The lab's Message cannot carry a system role at all -- this pins
        # the boundary, exactly like the gemma4 test.
        with self.assertRaises(ValueError):
            Message("system", "Be helpful")  # type: ignore[arg-type]

    def test_kolibri_thinking_off_mutation_pin(self) -> None:
        # Thinking OFF is a generation-prompt branch of the same template, so
        # it rides the mutate path (not a second catalog entry): the
        # generation prompt prefills an EMPTY think block.
        variants = {variant.name: variant for variant in candidates("kolibri", mutate=True)}
        self.assertIn("kolibri/thinking-off", variants)
        thinking_off = variants["kolibri/thinking-off"]
        self.assertEqual(thinking_off.open_suffix, _KOLIBRI_THINKING_OFF_SUFFIX)
        self.assertEqual(thinking_off.render(_MESSAGES), KOLIBRI_THINKING_OFF_EXPECTED)
        # Same frames as the base entry -- only the open-reply suffix
        # differs.
        base = CATALOG["kolibri"]
        for field in ("bos", "user_prefix", "assistant_prefix", "turn_end"):
            self.assertEqual(getattr(thinking_off, field), getattr(base, field))

    def test_kolibri_mutate_yields_thinking_off_in_addition_to_repairs(self) -> None:
        # The mutate path is the union of the generic BOS/prefix local
        # repairs and the family-specific thinking-off branch.
        names = {variant.name for variant in candidates("kolibri", mutate=True)}
        self.assertIn("kolibri/thinking-off", names)
        # kolibri has no BOS marker, so the BOS repairs collapse (documented
        # local_repairs dedup); the surviving generic repairs are the prefix
        # pair.
        self.assertTrue(any(name.startswith("kolibri/prefix-newline") for name in names))
        self.assertTrue(any(name.startswith("kolibri/prefix-trim") for name in names))
        # No other family gains a thinking-off branch.
        for family in CATALOG:
            if family == "kolibri":
                continue
            with self.subTest(family=family):
                self.assertNotIn(
                    f"{family}/thinking-off",
                    {variant.name for variant in candidates(family, mutate=True)},
                )


class Lfm25RenderTests(unittest.TestCase):
    def test_lfm25_renders_byte_exact(self) -> None:
        self.assertEqual(CATALOG["lfm25"].render(_MESSAGES), LFM25_EXPECTED)

    def test_lfm25_bos_and_think_suffix(self) -> None:
        fmt = CATALOG["lfm25"]
        self.assertEqual(fmt.bos, "<|startoftext|>")
        self.assertEqual(fmt.open_suffix, "<|im_start|>assistant\n<think>\n")
        rendered = fmt.render(_MESSAGES)
        self.assertTrue(rendered.startswith("<|startoftext|>"))
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n<think>\n"))
        # BOS emitted exactly once, never repeated.
        self.assertEqual(rendered.count("<|startoftext|>"), 1)

    def test_lfm25_fields_match_catalog_entry(self) -> None:
        fmt = CATALOG["lfm25"]
        self.assertEqual(fmt.name, "lfm25")
        self.assertEqual(fmt.user_prefix, "<|im_start|>user\n")
        self.assertEqual(fmt.assistant_prefix, "<|im_start|>assistant\n")
        self.assertEqual(fmt.turn_end, "<|im_end|>\n")
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertEqual(candidates("lfm25"), (fmt,))

    def test_lfm25_tier2_caveats_are_documented_not_asserted(self) -> None:
        # The bos_token-vs-template-literal origin and the
        # auto-injected-vs-illustrative status of the system block are
        # UNVERIFIED by design (byte-identical either way / unmodelable) --
        # the example block is pinned as a documented constant and must not
        # leak into the render.
        self.assertEqual(
            _LFM25_SYSTEM_BLOCK_EXAMPLE,
            "<|im_start|>system\nYou are a helpful assistant trained by Liquid AI.<|im_end|>\n",
        )
        rendered = CATALOG["lfm25"].render(_MESSAGES)
        self.assertNotIn("You are a helpful assistant trained by Liquid AI.", rendered)

    def test_lfm25_mutate_path_has_no_thinking_branch(self) -> None:
        # The 24B variant always thinks: there is no thinking-off branch, so
        # the mutate path carries only the generic local repairs.
        names = {variant.name for variant in candidates("lfm25", mutate=True)}
        self.assertNotIn("lfm25/thinking-off", names)
        self.assertGreater(len(names), 1)


class ClosestFamilyDiffTests(unittest.TestCase):
    def test_kolibri_vs_chatml_identical_text_only_core(self) -> None:
        # Kolibri-1's text-only no-system core is byte-identical to chatml:
        # same frames, no BOS, thinking-on tail via the same fallback. The
        # families differ ONLY in the unmodeled system block (always emitted
        # by Kolibri-1, absent from chatml) and the thinking-off branch.
        self.assertEqual(CATALOG["kolibri"].render(_MESSAGES), CATALOG["chatml"].render(_MESSAGES))
        # ...and the mutate sets differ exactly by the thinking-off branch.
        kolibri_names = {v.name for v in candidates("kolibri", mutate=True)}
        chatml_names = {v.name for v in candidates("chatml", mutate=True)}
        self.assertEqual(
            kolibri_names - {"kolibri/thinking-off"},
            {name.replace("chatml", "kolibri") for name in chatml_names},
        )

    def test_lfm25_vs_chatml_differs_in_bos_and_think_suffix(self) -> None:
        # Closest family is chatml (same per-turn ChatML frames), but lfm25
        # prepends <|startoftext|> and always-thinks: two byte-level deltas.
        lfm25 = CATALOG["lfm25"].render(_MESSAGES)
        chatml = CATALOG["chatml"].render(_MESSAGES)
        self.assertNotEqual(lfm25, chatml)
        self.assertTrue(lfm25.startswith("<|startoftext|>"))
        self.assertFalse(chatml.startswith("<|startoftext|>"))
        self.assertTrue(lfm25.endswith("<|im_start|>assistant\n<think>\n"))
        self.assertTrue(chatml.endswith("<|im_start|>assistant\n"))
        # The per-turn frames are shared: strip lfm25's BOS and the
        # generation-prompt tails and the message bodies coincide.
        self.assertIn("<|im_start|>user\nHello<|im_end|>\n", lfm25)
        self.assertIn("<|im_start|>assistant\nHi there<|im_end|>\n", lfm25)
        # chatml's turn body frames appear verbatim inside lfm25's render.
        body = chatml[: -len("<|im_start|>assistant\n")]
        self.assertIn(body, lfm25)


class EvidenceTierDocumentationTests(unittest.TestCase):
    def test_evidence_tiers_pinned_per_family(self) -> None:
        # Tier 1 (kolibri): the official tokenizer_config.json chat_template
        # is byte-verified, so the base render asserts the exact official
        # frames with no BOS and the thinking-on tail -- and the system block
        # prefix pin above is byte-verified, not illustrative.
        kolibri = CATALOG["kolibri"]
        self.assertEqual(kolibri.bos, "")
        self.assertTrue(KOLIBRI_EXPECTED.endswith("<|im_start|>assistant\n"))
        # Tier 2 (lfm25): the render follows the official card/HF blog
        # example exactly (BOS + <think> generation suffix), while the two
        # documented caveats stay as test constants, never asserted in code.
        lfm25 = CATALOG["lfm25"]
        self.assertEqual(lfm25.bos, "<|startoftext|>")
        self.assertEqual(lfm25.open_suffix, "<|im_start|>assistant\n<think>\n")


# --- DeepSeek V4.1 system-framing re-verification (2026-10-15) ---------------
# The catalog claims V4.1 frames system messages as <System>{content}
# (unlike V3, which concatenates them raw), while third-party docs of the
# sibling V4 official encoder (encoding_dsv4.py) render system text raw.
# Re-checked against the primary source 2026-10-15 and CONFIRMED:
# - Official model card (deepseek-ai/DeepSeek-V4.1-Flash, "Prompt Encoding"):
#   "This release does not include a Jinja-format chat template. The
#   `encoding` folder contains a self-contained Python reference
#   implementation (`encoding.py`)..." --
#   https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
# - A direct quote of that official encoding's render shows the system
#   block opened with the fullwidth System frame:
#   "<System>Reasoning Effort: 75 (range 1-100, ...)" --
#   https://github.com/corvous/hi-blue/blob/HEAD/docs/prompting/deepseek-v4-1-flash-guide.md
#   ("DeepSeek publishes the exact prompt encoding in the model repo
#   (`encoding/encoding.py`)")
# - The official deepseek-recipe repo (Rust, same prompt format) defines a
#   dedicated V4.1 System role token (id 128799) -- no such frame exists for
#   V4 -- https://github.com/deepseek-ai/deepseek-recipe
# - Contrast (sibling V4, NOT V4.1): the official V4 encoder renders system
#   RAW -- "<begin of sentence>{system}<User>{question}<Assistant>...",
#   "System content is bare text with no wrapper" --
#   https://github.com/waybarrios/vllm-mlx/pull/676
# Verdict: the catalog's V4.1 claim stands; the V3/V4-vs-V4.1 difference is
# real. The system role itself is unmodelable (Format has no system role),
# so the claim is pinned here as a documented constant -- exactly like the
# kolibri/lfm25/gemma4 system-block pins above -- and the render is pinned
# to carry no system frame and no effort preamble (the latter deliberately
# omitted per the catalog comment).
#
# Fullwidth bars, spelled as escapes exactly like the catalog entry
# (U+FF5C FULLWIDTH VERTICAL BAR): the constant is checked against the
# catalog's own escape-composed frame, so the pin is not vacuous.
_DS41_SYSTEM_FRAME = "\uff5cSystem\uff5c"


class Deepseek41SystemFramingTests(unittest.TestCase):
    def test_deepseek41_system_frame_claim_is_pinned(self) -> None:
        # The documented V4.1 system frame: fullwidth bars around "System",
        # byte-identical to the catalog entry's _DS_BAR/_DS_BLK spelling.
        self.assertEqual(_DS41_SYSTEM_FRAME, f"{_DS_BAR}System{_DS_BAR}")

    def test_deepseek41_render_carries_no_system_frame(self) -> None:
        # The system role is not modeled in Format: the frame must never leak
        # into a text-only render, and the effort preamble stays deliberately
        # omitted (per the catalog comment).
        rendered = CATALOG["deepseek41"].render(_MESSAGES)
        self.assertNotIn("System", rendered)
        self.assertNotIn("Reasoning Effort", rendered)

    def test_deepseek41_text_only_core_matches_v3_plus_think(self) -> None:
        # The modeled difference between V4.1 and V3 is exactly the chat-mode
        # <Assistant><think> open suffix -- same BOS, same user/assistant
        # prefixes, same assistant terminator.
        v41 = CATALOG["deepseek41"]
        v3 = CATALOG["deepseek"]
        for field in ("bos", "user_prefix", "assistant_prefix", "turn_end"):
            self.assertEqual(getattr(v41, field), getattr(v3, field))
        self.assertEqual(v41.assistant_turn_end, v3.assistant_turn_end)
        self.assertEqual(v41.open_suffix, v3.assistant_prefix + "<think>")


class ReadmeFamilyCountTests(unittest.TestCase):
    def test_readme_family_count_matches_catalog(self) -> None:
        # The README's "N-family catalog" claim must track len(CATALOG) --
        # adding a family without updating the README fails here (same
        # discipline as test_readme_test_count_matches_discovered_suite).
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        matches = re.findall(r"(\d+)-family catalog", text)
        self.assertTrue(matches, "README lost its N-family catalog claim")
        for match in matches:
            self.assertEqual(int(match), len(CATALOG))

    def test_readme_names_new_families(self) -> None:
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Kolibri-1", text)
        self.assertIn("LFM2.5", text)


if __name__ == "__main__":
    unittest.main()
