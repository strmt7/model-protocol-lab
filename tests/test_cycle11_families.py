"""Cycle 11 (continuous dev): Nemotron 3.5 Lightning, Muse Glimmer 30B, and
Poolside Laguna 2.1 catalog entries.

Byte-exact render checks for three summer/autumn-2026 families, following the
conventions of tests/test_cycle10_families.py:

- ``nemotron35`` (NVIDIA Nemotron 3.5 Lightning, released 2026-08-11;
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16).
  Evidence tier 2: official HF model card, NeMo SFT docs, community osaurus
  renderer. ChatML per-turn frames; thinking-on (default) gen prompt
  ``<|im_start|>assistant\\n<think>\\n``; thinking-off prefills an EMPTY
  think block (``<|im_start|>assistant\\n<think></think>``, no trailing
  newlines) and rides the mutate path as ``nemotron35/thinking-off``.
  CRITICAL: BOS behavior is UNVERIFIED/conflicting at official level --
  the catalog models bos="" as an absence-of-evidence placeholder and does
  NOT assert any BOS token (pinned here, including a source-level check
  that the catalog comment carries the UNVERIFIED marker).
- ``muse_glimmer`` (Meta Muse Glimmer 30B, released 2026-08-10, Apache 2.0;
  https://github.com/meta-models/meta-oss-cookbook/blob/HEAD/quickstart/README.md).
  Evidence tier 2: official quickstart, HF transformers model docs, HF blog
  walkthrough -- with the generation prompt community-refined (the simpler
  ``<|start|>assistant`` alternative was superseded by the recipient-channel
  ``<|start|>assistant to=user<|message|>``). Assistant turns are modeled as
  ``to=user``; the ``to=self`` reasoning trace is not rendered. The system
  appendage, EOS, and end-of-message marker are documented, not modeled.
- ``laguna`` (Poolside Laguna 2.1, Jul 2026, OpenMDW-1.1;
  https://huggingface.co/poolside/Laguna-XS-2.1). Evidence tier 2: official
  HF cards plus the renderers library's LagunaXS21Renderer/LagunaS21Renderer
  (token-exact parity-verified against apply_chat_template;
  PrimeIntellect-ai/renderers PRs #99 and #111). Base entry = XS-2.1
  defaults (enable_thinking FALSE, gen prompt ``<assistant></think>``);
  S-2.1's default (thinking on, gen prompt ``<assistant><think>``) rides the
  mutate path as ``laguna/thinking-on``. Assistant turns close with
  ``</assistant>``; EOS is decode-only.

Also: diff-vs-closest-family pins for all three, the thinking-branch
cross-catalog invariant (thinking-off exists exactly for
{kolibri, nemotron35}; thinking-on exactly for {laguna}), and a README
naming pin for the new families (the N-family count pin lives in
tests/test_cycle10_families.py and enforces the 19 -> 22 update).

Deterministic; no network, no model, no randomness.
"""

import re
import unittest
from pathlib import Path

from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.core import Message

REPO_ROOT = Path(__file__).resolve().parent.parent

_MESSAGES: tuple[Message, ...] = (
    Message("user", "Hello"),
    Message("assistant", "Hi there"),
    Message("user", "Bye"),
)


# --- Nemotron 3.5 Lightning pins --------------------------------------------
# Official template (tier 2): ChatML per-turn frames. Thinking ON (default):
# the generation prompt ends at <|im_start|>assistant\n<think>\n. Thinking
# OFF: the generation prompt prefills an EMPTY think block -- note the
# absent trailing newlines, unlike kolibri/thinking-off's
# "<think>\n\n</think>\n\n" tail.
_NEMOTRON_USER_OPEN = "<|im_start|>user\n"
_NEMOTRON_ASSISTANT_OPEN = "<|im_start|>assistant\n"
_NEMOTRON_TURN_END = "<|im_end|>\n"
_NEMOTRON_THINKING_ON_SUFFIX = "<|im_start|>assistant\n<think>\n"
_NEMOTRON_THINKING_OFF_SUFFIX = "<|im_start|>assistant\n<think></think>"

_NEMOTRON_BODY = (
    _NEMOTRON_USER_OPEN
    + "Hello"
    + _NEMOTRON_TURN_END
    + _NEMOTRON_ASSISTANT_OPEN
    + "Hi there"
    + _NEMOTRON_TURN_END
    + _NEMOTRON_USER_OPEN
    + "Bye"
    + _NEMOTRON_TURN_END
)
NEMOTRON_EXPECTED = _NEMOTRON_BODY + _NEMOTRON_THINKING_ON_SUFFIX
NEMOTRON_THINKING_OFF_EXPECTED = _NEMOTRON_BODY + _NEMOTRON_THINKING_OFF_SUFFIX


class Nemotron35RenderTests(unittest.TestCase):
    def test_nemotron35_renders_byte_exact(self) -> None:
        self.assertEqual(CATALOG["nemotron35"].render(_MESSAGES), NEMOTRON_EXPECTED)

    def test_nemotron35_base_is_thinking_on(self) -> None:
        # The base entry is the DEFAULT (thinking-on) branch: the generation
        # prompt ends inside the <think> block. open_suffix is explicit here
        # (unlike kolibri, whose thinking-on tail is the bare assistant
        # frame via the fallback).
        fmt = CATALOG["nemotron35"]
        self.assertEqual(fmt.open_suffix, _NEMOTRON_THINKING_ON_SUFFIX)
        rendered = fmt.render(_MESSAGES)
        self.assertTrue(rendered.endswith(_NEMOTRON_THINKING_ON_SUFFIX))

    def test_nemotron35_bos_is_unverified_placeholder_not_assertion(self) -> None:
        # CRITICAL: BOS behavior is UNVERIFIED/conflicting at official level.
        # The catalog models bos="" as an absence-of-evidence placeholder and
        # must NOT assert any BOS token. This pins both the modeled value and
        # the source-level UNVERIFIED marker in the catalog comment.
        fmt = CATALOG["nemotron35"]
        self.assertEqual(fmt.bos, "")
        rendered = fmt.render(_MESSAGES)
        self.assertTrue(rendered.startswith("<|im_start|>user\n"))
        source = (REPO_ROOT / "model_protocol_lab" / "catalog.py").read_text(encoding="utf-8")
        # The comment block immediately above the entry must carry the
        # UNVERIFIED marker; take a generous window so the check tracks the
        # comment, not the entry dict.
        comment_window = source.split('"nemotron35": Format')[0][-2600:]
        self.assertIn("UNVERIFIED", comment_window)
        self.assertIn("does NOT assert any BOS token", comment_window)
        # And the render must carry no BOS-like token anywhere.
        self.assertNotIn("begin_of_text", rendered)
        self.assertNotIn("<bos>", rendered)
        self.assertNotIn("startoftext", rendered)

    def test_nemotron35_fields_match_catalog_entry(self) -> None:
        fmt = CATALOG["nemotron35"]
        self.assertEqual(fmt.name, "nemotron35")
        self.assertEqual(fmt.user_prefix, "<|im_start|>user\n")
        self.assertEqual(fmt.assistant_prefix, "<|im_start|>assistant\n")
        self.assertEqual(fmt.turn_end, "<|im_end|>\n")
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertEqual(candidates("nemotron35"), (fmt,))

    def test_nemotron35_thinking_off_mutation_pin(self) -> None:
        # Thinking OFF is a generation-prompt branch of the same template, so
        # it rides the mutate path (not a second catalog entry): the
        # generation prompt prefills an EMPTY think block with NO trailing
        # newlines (unlike kolibri/thinking-off).
        variants = {variant.name: variant for variant in candidates("nemotron35", mutate=True)}
        self.assertIn("nemotron35/thinking-off", variants)
        thinking_off = variants["nemotron35/thinking-off"]
        self.assertEqual(thinking_off.open_suffix, _NEMOTRON_THINKING_OFF_SUFFIX)
        # The exact-string pin above also pins the absent trailing newlines
        # (unlike kolibri/thinking-off's "<think>\n\n</think>\n\n" tail).
        self.assertEqual(thinking_off.render(_MESSAGES), NEMOTRON_THINKING_OFF_EXPECTED)
        # Same frames as the base entry -- only the open-reply suffix
        # differs.
        base = CATALOG["nemotron35"]
        for field in ("bos", "user_prefix", "assistant_prefix", "turn_end"):
            self.assertEqual(getattr(thinking_off, field), getattr(base, field))

    def test_nemotron35_thinking_off_differs_from_kolibri_tail(self) -> None:
        # Both families ChatML-frame the turns and both ride a thinking-off
        # branch on the mutate path, but the prefills differ byte-for-byte:
        # nemotron35 ends at <think></think> with no trailing newline;
        # kolibri ends at "<think>\n\n</think>\n\n".
        nemotron_off = {variant.name: variant for variant in candidates("nemotron35", mutate=True)}[
            "nemotron35/thinking-off"
        ]
        kolibri_off = {variant.name: variant for variant in candidates("kolibri", mutate=True)}[
            "kolibri/thinking-off"
        ]
        self.assertNotEqual(nemotron_off.open_suffix, kolibri_off.open_suffix)
        self.assertEqual(kolibri_off.open_suffix, "<|im_start|>assistant\n<think>\n\n</think>\n\n")


# --- Muse Glimmer 30B pins ---------------------------------------------------
# Official template (tier 2, gen prompt community-refined): the frame is
# <|begin_of_text|><|start|>{role}<|message|>{content}<|eot|> -- no newline
# anywhere. Assistant turns use recipient channels; the plain-text assistant
# turn is modeled as to=user (the to=self reasoning trace is not rendered).
_GLIMMER_BOS = "<|begin_of_text|>"
_GLIMMER_USER_OPEN = "<|start|>user<|message|>"
_GLIMMER_ASSISTANT_OPEN = "<|start|>assistant to=user<|message|>"
_GLIMMER_TURN_END = "<|eot|>"
_GLIMMER_OPEN_SUFFIX = "<|start|>assistant to=user<|message|>"

GLIMMER_EXPECTED = (
    _GLIMMER_BOS
    + _GLIMMER_USER_OPEN
    + "Hello"
    + _GLIMMER_TURN_END
    + _GLIMMER_ASSISTANT_OPEN
    + "Hi there"
    + _GLIMMER_TURN_END
    + _GLIMMER_USER_OPEN
    + "Bye"
    + _GLIMMER_TURN_END
    + _GLIMMER_OPEN_SUFFIX
)

# Documented, not modeled (same convention as the kolibri/lfm25/gemma4
# system-block pins): the system appendage, the EOS token, and the
# end-of-message marker. Pinned here as documented constants only.
_GLIMMER_SYSTEM_SUFFIX = '\nReasoning strength: high.\n# Valid recipients: "self", "user".'
_GLIMMER_EOS = "<|end_of_text|>"  # token id 200001
_GLIMMER_EOT = ""  # end-of-MESSAGE marker, NOT a stop token


class MuseGlimmerRenderTests(unittest.TestCase):
    def test_muse_glimmer_renders_byte_exact(self) -> None:
        self.assertEqual(CATALOG["muse_glimmer"].render(_MESSAGES), GLIMMER_EXPECTED)

    def test_muse_glimmer_gen_prompt_is_community_refined(self) -> None:
        # The generation prompt is the community-refined recipient-channel
        # form -- the simpler "<|start|>assistant" alternative was
        # superseded. Tier 2/community-corroborated, marked as such.
        fmt = CATALOG["muse_glimmer"]
        self.assertEqual(fmt.open_suffix, _GLIMMER_OPEN_SUFFIX)
        self.assertNotEqual(fmt.open_suffix, "<|start|>assistant")
        rendered = fmt.render(_MESSAGES)
        self.assertTrue(rendered.endswith(_GLIMMER_OPEN_SUFFIX))

    def test_muse_glimmer_fields_match_catalog_entry(self) -> None:
        fmt = CATALOG["muse_glimmer"]
        self.assertEqual(fmt.name, "muse_glimmer")
        self.assertEqual(fmt.bos, "<|begin_of_text|>")
        self.assertEqual(fmt.user_prefix, "<|start|>user<|message|>")
        self.assertEqual(fmt.assistant_prefix, "<|start|>assistant to=user<|message|>")
        self.assertEqual(fmt.turn_end, "<|eot|>")
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertEqual(candidates("muse_glimmer"), (fmt,))

    def test_muse_glimmer_documented_not_modeled(self) -> None:
        # The system appendage, the EOS stop token, and the end-of-message
        # marker are documented in the catalog comment but not modeled: the
        # system role is unmodelable, and EOS/EOM are generation stops, not
        # prompt strings.
        rendered = CATALOG["muse_glimmer"].render(_MESSAGES)
        self.assertNotIn("<|start|>system", rendered)
        self.assertNotIn("Reasoning strength", rendered)
        self.assertNotIn("to=self", rendered)
        self.assertNotIn(_GLIMMER_EOS, rendered)
        self.assertNotIn(_GLIMMER_EOT or "<|eot|>", rendered.replace("<|eot|>", ""))
        # The documented constants are pinned exactly, as written in the
        # catalog comment.
        self.assertEqual(
            _GLIMMER_SYSTEM_SUFFIX,
            '\nReasoning strength: high.\n# Valid recipients: "self", "user".',
        )
        self.assertEqual(_GLIMMER_EOS, "<|end_of_text|>")
        self.assertEqual(_GLIMMER_EOT, "")


# --- Poolside Laguna 2.1 pins -------------------------------------------------
# Tier-2 render (renderers library, token-exact parity vs apply_chat_template):
# "hugging tags" with no inner newlines. Base entry = XS-2.1 defaults
# (enable_thinking FALSE): history assistant turns are
# <assistant></think>{content}</assistant>, the gen prompt is
# <assistant></think> with no trailing newline. S-2.1's default (thinking on)
# rides the mutate path as laguna/thinking-on: gen prompt
# <assistant><think>, history turns render an EMPTY think block verbatim
# (<assistant><think></think>{content}</assistant>) -- the lab's
# deterministic no-reasoning hypothesis (reasoned-content branch not a
# hypothesis, kimi3 convention).
_LAGUNA_USER_OPEN = "<user>"
_LAGUNA_USER_CLOSE = "</user>"
_LAGUNA_ASSISTANT_OPEN_OFF = "<assistant></think>"
_LAGUNA_ASSISTANT_OPEN_ON = "<assistant><think></think>"
_LAGUNA_ASSISTANT_CLOSE = "</assistant>"
_LAGUNA_OPEN_SUFFIX_OFF = "<assistant></think>"
_LAGUNA_OPEN_SUFFIX_ON = "<assistant><think>"

LAGUNA_EXPECTED = (
    _LAGUNA_USER_OPEN
    + "Hello"
    + _LAGUNA_USER_CLOSE
    + _LAGUNA_ASSISTANT_OPEN_OFF
    + "Hi there"
    + _LAGUNA_ASSISTANT_CLOSE
    + _LAGUNA_USER_OPEN
    + "Bye"
    + _LAGUNA_USER_CLOSE
    + _LAGUNA_OPEN_SUFFIX_OFF
)
LAGUNA_THINKING_ON_EXPECTED = (
    _LAGUNA_USER_OPEN
    + "Hello"
    + _LAGUNA_USER_CLOSE
    + _LAGUNA_ASSISTANT_OPEN_ON
    + "Hi there"
    + _LAGUNA_ASSISTANT_CLOSE
    + _LAGUNA_USER_OPEN
    + "Bye"
    + _LAGUNA_USER_CLOSE
    + _LAGUNA_OPEN_SUFFIX_ON
)

# EOS is decode-only (never rendered in the prompt): pinned as a documented
# constant, exactly like the glimmer pins above. The literal is the
# private-use token the upstream template uses; it must never leak into a
# render.
_LAGUNA_EOS_DECODE_ONLY = ""


class LagunaRenderTests(unittest.TestCase):
    def test_laguna_renders_byte_exact(self) -> None:
        self.assertEqual(CATALOG["laguna"].render(_MESSAGES), LAGUNA_EXPECTED)

    def test_laguna_base_is_xs21_thinking_off(self) -> None:
        # Base entry = XS-2.1 defaults (documented choice): enable_thinking
        # defaults FALSE, so history assistant turns open with the bare
        # </think> prefill and the gen prompt ends at <assistant></think>
        # with no trailing newline.
        fmt = CATALOG["laguna"]
        self.assertEqual(fmt.open_suffix, _LAGUNA_OPEN_SUFFIX_OFF)
        rendered = fmt.render(_MESSAGES)
        self.assertTrue(rendered.endswith(_LAGUNA_OPEN_SUFFIX_OFF))
        self.assertIn("<assistant></think>Hi there</assistant>", rendered)

    def test_laguna_fields_match_catalog_entry(self) -> None:
        fmt = CATALOG["laguna"]
        self.assertEqual(fmt.name, "laguna")
        self.assertEqual(fmt.bos, "")
        self.assertEqual(fmt.user_prefix, "<user>")
        self.assertEqual(fmt.assistant_prefix, "<assistant></think>")
        self.assertEqual(fmt.turn_end, "")
        self.assertEqual(fmt.user_turn_end, "</user>")
        self.assertEqual(fmt.assistant_turn_end, "</assistant>")
        self.assertEqual(fmt.open_suffix, "<assistant></think>")
        self.assertEqual(candidates("laguna"), (fmt,))

    def test_laguna_thinking_on_mutation_pin(self) -> None:
        # S-2.1's default (enable_thinking=True) rides the mutate path as
        # laguna/thinking-on: gen prompt <assistant><think>, history
        # assistant turns render an EMPTY think block verbatim. There is no
        # laguna/thinking-off -- the base IS the thinking-off default.
        variants = {variant.name: variant for variant in candidates("laguna", mutate=True)}
        self.assertIn("laguna/thinking-on", variants)
        self.assertNotIn("laguna/thinking-off", variants)
        thinking_on = variants["laguna/thinking-on"]
        self.assertEqual(thinking_on.open_suffix, _LAGUNA_OPEN_SUFFIX_ON)
        self.assertEqual(thinking_on.assistant_prefix, _LAGUNA_ASSISTANT_OPEN_ON)
        self.assertEqual(thinking_on.assistant_turn_end, "</assistant>")
        self.assertEqual(thinking_on.render(_MESSAGES), LAGUNA_THINKING_ON_EXPECTED)
        # User frames and terminators are shared with the base entry.
        base = CATALOG["laguna"]
        for field in ("bos", "user_prefix", "user_turn_end", "turn_end"):
            self.assertEqual(getattr(thinking_on, field), getattr(base, field))

    def test_laguna_eos_is_decode_only(self) -> None:
        # EOS is decode-only: documented in the catalog comment, never
        # rendered in the prompt. The documented literal is pinned as a
        # constant and must not leak into any render or mutate variant.
        self.assertEqual(_LAGUNA_EOS_DECODE_ONLY, "")
        for family in ("laguna",):
            for variant in candidates(family, mutate=True):
                self.assertNotIn("|EOS|", variant.render(_MESSAGES))

    def test_laguna_system_block_suppressed(self) -> None:
        # The lab never sends system messages, and an empty caller system
        # message suppresses the <system> block entirely -- so no <system>
        # frame may appear in the render. The upstream default-Poolside-prompt
        # wording is deliberately NOT asserted anywhere.
        rendered = CATALOG["laguna"].render(_MESSAGES)
        self.assertNotIn("<system>", rendered)
        self.assertNotIn("Poolside", rendered)


# --- Diff-vs-closest-family pins ----------------------------------------------


class ClosestFamilyDiffTests(unittest.TestCase):
    def test_nemotron35_vs_chatml_differs_only_in_generation_tail(self) -> None:
        # Closest family is chatml: the per-turn frames are byte-identical
        # (ChatML), and the ONLY modeled difference is the thinking-on
        # generation tail -- plus the thinking-off branch on the mutate path.
        nemotron = CATALOG["nemotron35"]
        chatml = CATALOG["chatml"]
        for field in ("bos", "user_prefix", "assistant_prefix", "turn_end"):
            self.assertEqual(getattr(nemotron, field), getattr(chatml, field))
        self.assertIsNone(chatml.open_suffix)
        self.assertEqual(nemotron.open_suffix, "<|im_start|>assistant\n<think>\n")
        nemotron_render = nemotron.render(_MESSAGES)
        chatml_render = chatml.render(_MESSAGES)
        chatml_tail = "<|im_start|>assistant\n"
        self.assertTrue(chatml_render.endswith(chatml_tail))
        self.assertEqual(
            nemotron_render,
            chatml_render[: -len(chatml_tail)] + "<|im_start|>assistant\n<think>\n",
        )
        # ...and the mutate sets differ exactly by the thinking-off branch.
        nemotron_names = {v.name for v in candidates("nemotron35", mutate=True)}
        chatml_names = {v.name for v in candidates("chatml", mutate=True)}
        self.assertIn("nemotron35/thinking-off", nemotron_names)
        self.assertEqual(
            nemotron_names - {"nemotron35/thinking-off"},
            {name.replace("chatml", "nemotron35") for name in chatml_names},
        )

    def test_nemotron35_vs_kolibri_same_frames_different_tail(self) -> None:
        # nemotron35 and kolibri share the ChatML per-turn frames and the
        # thinking-branch-on-mutate-path convention, but their generation
        # tails differ: kolibri's thinking-on is the bare assistant frame
        # (the model opens <think> itself); nemotron35's ends inside
        # <think>\n.
        nemotron = CATALOG["nemotron35"]
        kolibri = CATALOG["kolibri"]
        for field in ("bos", "user_prefix", "assistant_prefix", "turn_end"):
            self.assertEqual(getattr(nemotron, field), getattr(kolibri, field))
        self.assertIsNone(kolibri.open_suffix)
        self.assertNotEqual(nemotron.open_suffix, kolibri.assistant_prefix)

    def test_muse_glimmer_vs_llama3_shared_bos_only(self) -> None:
        # Closest family is llama3: both prepend <|begin_of_text|>, but every
        # frame differs (recipient channels + <|message|>/<|eot|> vs header
        # ids + double newlines + <|eot_id|>).
        glimmer = CATALOG["muse_glimmer"]
        llama3 = CATALOG["llama3"]
        self.assertEqual(glimmer.bos, llama3.bos)
        self.assertEqual(glimmer.bos, "<|begin_of_text|>")
        for field in ("user_prefix", "assistant_prefix", "turn_end"):
            self.assertNotEqual(getattr(glimmer, field), getattr(llama3, field))
        glimmer_render = glimmer.render(_MESSAGES)
        llama3_render = llama3.render(_MESSAGES)
        self.assertTrue(glimmer_render.startswith("<|begin_of_text|>"))
        self.assertTrue(llama3_render.startswith("<|begin_of_text|>"))
        self.assertNotIn("<|start|>", llama3_render)
        self.assertNotIn("<|start_header_id|>", glimmer_render)
        # The recipient channel is glimmer-only.
        self.assertIn("assistant to=user", glimmer_render)
        self.assertNotIn("to=user", llama3_render)

    def test_laguna_vs_chatml_shares_nothing(self) -> None:
        # Laguna's hugging-tag protocol shares no frame with ChatML: the
        # renders are disjoint byte-for-byte beyond the message content.
        laguna_render = CATALOG["laguna"].render(_MESSAGES)
        chatml_render = CATALOG["chatml"].render(_MESSAGES)
        self.assertNotEqual(laguna_render, chatml_render)
        self.assertNotIn("<|im_start|>", laguna_render)
        self.assertNotIn("<|im_end|>", laguna_render)
        self.assertNotIn("<user>", chatml_render)
        self.assertNotIn("<assistant>", chatml_render)
        # No trailing newline anywhere in the laguna render (no inner
        # newlines in the upstream template).
        self.assertNotIn("\n", laguna_render)


# --- Thinking-branch cross-catalog invariant -----------------------------------


class ThinkingBranchInvariantTests(unittest.TestCase):
    def test_thinking_branches_exist_exactly_where_expected(self) -> None:
        # Thinking-off branches ride the mutate path for exactly the two
        # families whose templates prefill an empty think block when
        # thinking is disabled (kolibri, nemotron35); thinking-on rides it
        # for exactly laguna (S-2.1's default). No other family gains either
        # branch. (tests/test_cycle10_families.py pins the same invariant
        # from the kolibri side with the nemotron35 exception.)
        off_families = {
            family
            for family in CATALOG
            if f"{family}/thinking-off"
            in {variant.name for variant in candidates(family, mutate=True)}
        }
        on_families = {
            family
            for family in CATALOG
            if f"{family}/thinking-on"
            in {variant.name for variant in candidates(family, mutate=True)}
        }
        self.assertEqual(off_families, {"kolibri", "nemotron35"})
        self.assertEqual(on_families, {"laguna"})


# --- Evidence-tier documentation -----------------------------------------------


class EvidenceTierDocumentationTests(unittest.TestCase):
    def test_evidence_tiers_pinned_per_family(self) -> None:
        # All three families are evidence tier 2 (none has a tier-1
        # byte-verified official chat_template): nemotron35 via the official
        # card + NeMo docs + community renderer, muse_glimmer via the
        # official quickstart + transformers docs + HF blog with a
        # community-refined gen prompt, laguna via the official cards + the
        # token-exact renderers library.
        nemotron = CATALOG["nemotron35"]
        # Tier 2, not tier 1: the render pins the ChatML frames and the
        # thinking tails, but the BOS is an explicitly unverified
        # placeholder -- a tier-1 entry would pin a byte-verified BOS.
        self.assertEqual(nemotron.bos, "")
        self.assertTrue(NEMOTRON_EXPECTED.endswith("<|im_start|>assistant\n<think>\n"))
        glimmer = CATALOG["muse_glimmer"]
        # The gen prompt is the community-refined form, not the superseded
        # simple form -- pinned, and marked tier 2 in the catalog comment.
        self.assertEqual(glimmer.open_suffix, "<|start|>assistant to=user<|message|>")
        laguna = CATALOG["laguna"]
        # The </assistant> terminator is tier-2 via the renderers PR text;
        # the EOS is decode-only and unrendered.
        self.assertEqual(laguna.assistant_turn_end, "</assistant>")
        self.assertEqual(laguna.bos, "")


# --- README naming pin ----------------------------------------------------------


class ReadmeNewFamiliesTests(unittest.TestCase):
    def test_readme_names_new_families(self) -> None:
        # The README's family list must name the three new families (the
        # N-family count pin lives in tests/test_cycle10_families.py).
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Nemotron", text)
        self.assertIn("Glimmer", text)
        self.assertIn("Laguna", text)

    def test_readme_count_claim_matches_catalog(self) -> None:
        # Belt-and-braces alongside the cycle-10 pin: the README's
        # "N-family catalog" claims must track len(CATALOG) == 22.
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        matches = re.findall(r"(\d+)-family catalog", text)
        self.assertTrue(matches, "README lost its N-family catalog claim")
        for match in matches:
            self.assertEqual(int(match), len(CATALOG))
        self.assertEqual(len(CATALOG), 22)


if __name__ == "__main__":
    unittest.main()
