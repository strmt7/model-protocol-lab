"""Cycle 54 (continuous dev): Qwen3.8 catalog entry.

Byte-exact render checks for the Qwen3.8 family (``qwen38``):

- ``qwen38`` (Qwen/Qwen3.8-27B, released Aug 2026;
  https://huggingface.co/Qwen/Qwen3.8-27B). Evidence tier 1: the official
  chat_template.jinja at main (verified commit 412f8b6). Plain-text
  no-tools core: per-turn frames are ChatML, byte-identical to ``qwen3``
  (``<|im_start|>{role}\\n{content}<|im_end|>\\n``, no BOS). The deltas vs
  qwen3 -- the reason this is a separate family, not a qwen3 alias -- are:
  thinking ON is the default and the generation prompt OPENS the think
  block (``<|im_start|>assistant\\n<think>\\n``, qwen3's ends at
  ``<|im_start|>assistant\\n``); thinking OFF (enable_thinking=false)
  prefills an EMPTY think block
  (``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``) and rides the
  mutate path as ``qwen38/thinking-off`` (kolibri/thinking-off convention).
  Thinking enabled also auto-injects a reasoning-effort system block and
  prefixes assistant history turns with think blocks -- both DOCUMENTED,
  not modeled (system role unmodelable; effort-steering belongs in a
  separate hypothesis; the reasoned-content branch is not a hypothesis --
  the glm53/kimi3/laguna conventions). The verified xhigh reasoning
  sentence is pinned here as a documented constant only.

Deterministic; no network, no model, no randomness.
"""

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

# --- Qwen3.8 pins ------------------------------------------------------------
# Official chat_template.jinja (tier 1, verified commit 412f8b6):
# per-message frames are ChatML -- byte-identical to qwen3. Thinking ON
# (the default): the generation prompt ends inside the <think> block.
# Thinking OFF: the generation prompt prefills an EMPTY think block with
# the "<think>\n\n</think>\n\n" tail (same tail as kolibri and
# nanbeige42, unlike nemotron35's "<think></think>").
_QWEN38_USER_OPEN = "<|im_start|>user\n"
_QWEN38_ASSISTANT_OPEN = "<|im_start|>assistant\n"
_QWEN38_TURN_END = "<|im_end|>\n"
_QWEN38_THINKING_ON_SUFFIX = "<|im_start|>assistant\n<think>\n"
_QWEN38_THINKING_OFF_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

_QWEN38_BODY = (
    _QWEN38_USER_OPEN
    + "Hello"
    + _QWEN38_TURN_END
    + _QWEN38_ASSISTANT_OPEN
    + "Hi there"
    + _QWEN38_TURN_END
    + _QWEN38_USER_OPEN
    + "Bye"
    + _QWEN38_TURN_END
)
QWEN38_EXPECTED = _QWEN38_BODY + _QWEN38_THINKING_ON_SUFFIX
QWEN38_THINKING_OFF_EXPECTED = _QWEN38_BODY + _QWEN38_THINKING_OFF_SUFFIX

# Documented, NOT modeled (glm53/kolibri/kimi3/laguna conventions): when
# thinking is enabled (the default), the official template auto-injects a
# system block carrying a reasoning-effort sentence if the caller sends no
# system message (jinja lines 183-186), or prepends it to a caller system
# message (jinja line 175); reasoning_effort defaults to 'xhigh' (jinja
# line 102). The sentence below is transcribed from the official
# chat_template.jinja (verified commit 412f8b6); it is pinned as a
# documented constant only and must never leak into a render.
_QWEN38_XHIGH_REASONING_SENTENCE = (
    "Reasoning effort is set to xhigh. Please think carefully through the "
    "task, validate key assumptions, consider plausible alternatives, and "
    "prioritize correctness, consistency, and clarity in the final answer."
)

# The primary source, pinned so the citation cannot drift.
_QWEN38_SOURCE_URL = "https://huggingface.co/Qwen/Qwen3.8-27B"


class Qwen38RenderTests(unittest.TestCase):
    def test_qwen38_multiturn_renders_byte_exact(self) -> None:
        self.assertEqual(CATALOG["qwen38"].render(_MESSAGES), QWEN38_EXPECTED)

    def test_qwen38_single_turn_renders_byte_exact(self) -> None:
        messages = (Message("user", "What is 2+2?"),)
        expected = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n"
        self.assertEqual(CATALOG["qwen38"].render(messages), expected)

    def test_qwen38_empty_messages_render_byte_exact(self) -> None:
        # Empty content renders as an empty span between the frame and the
        # turn terminator -- no placeholder injection (qwen3 convention).
        messages = (
            Message("user", ""),
            Message("assistant", ""),
            Message("user", " "),
        )
        expected = (
            "<|im_start|>user\n<|im_end|>\n"
            "<|im_start|>assistant\n<|im_end|>\n"
            "<|im_start|>user\n <|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        self.assertEqual(CATALOG["qwen38"].render(messages), expected)

    def test_qwen38_base_is_thinking_on(self) -> None:
        # The base entry is the DEFAULT (thinking-on) branch: the
        # generation prompt ends inside the <think> block. open_suffix is
        # explicit here (unlike kolibri, whose thinking-on tail is the bare
        # assistant frame via the fallback).
        fmt = CATALOG["qwen38"]
        self.assertEqual(fmt.open_suffix, _QWEN38_THINKING_ON_SUFFIX)
        rendered = fmt.render(_MESSAGES)
        self.assertTrue(rendered.endswith(_QWEN38_THINKING_ON_SUFFIX))

    def test_qwen38_bos_is_empty(self) -> None:
        # The official template emits no BOS literal (qwen3 convention:
        # modeled as the empty string, not as an asserted token).
        fmt = CATALOG["qwen38"]
        self.assertEqual(fmt.bos, "")
        rendered = fmt.render(_MESSAGES)
        self.assertTrue(rendered.startswith("<|im_start|>user\n"))
        self.assertNotIn("<|begin_of_text|>", rendered)
        self.assertNotIn("<bos>", rendered)
        self.assertNotIn("<|startoftext|>", rendered)

    def test_qwen38_fields_match_catalog_entry(self) -> None:
        fmt = CATALOG["qwen38"]
        self.assertEqual(fmt.name, "qwen38")
        self.assertEqual(fmt.user_prefix, "<|im_start|>user\n")
        self.assertEqual(fmt.assistant_prefix, "<|im_start|>assistant\n")
        self.assertEqual(fmt.turn_end, "<|im_end|>\n")
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertFalse(fmt.repeat_bos)
        self.assertEqual(candidates("qwen38"), (fmt,))

    def test_qwen38_no_auto_injected_system_block_in_render(self) -> None:
        # The effort-sentence system auto-injection is documented, not
        # modeled (the system role is not modeled in Format): the no-system
        # render must carry no <|im_start|>system block and no trace of the
        # reasoning sentence.
        rendered = CATALOG["qwen38"].render(_MESSAGES)
        self.assertNotIn("<|im_start|>system", rendered)
        self.assertNotIn("Reasoning effort is set to", rendered)
        self.assertNotIn("reasoning_effort", rendered)


class Qwen38ThinkingBranchTests(unittest.TestCase):
    def test_qwen38_thinking_off_mutation_pin(self) -> None:
        # Thinking OFF is a generation-prompt branch of the same template,
        # so it rides the mutate path (not a second catalog entry): the
        # generation prompt prefills an EMPTY think block with the
        # "<think>\n\n</think>\n\n" tail (same tail as kolibri and
        # nanbeige42, unlike nemotron35's "<think></think>").
        variants = {variant.name: variant for variant in candidates("qwen38", mutate=True)}
        self.assertIn("qwen38/thinking-off", variants)
        thinking_off = variants["qwen38/thinking-off"]
        self.assertEqual(thinking_off.open_suffix, _QWEN38_THINKING_OFF_SUFFIX)
        # The exact-string pin also pins the "<think>\n\n</think>\n\n"
        # tail byte-for-byte (the trailing newlines are load-bearing).
        self.assertEqual(thinking_off.render(_MESSAGES), QWEN38_THINKING_OFF_EXPECTED)
        # Same frames as the base entry -- only the open-reply suffix
        # differs.
        base = CATALOG["qwen38"]
        for field in ("bos", "user_prefix", "assistant_prefix", "turn_end"):
            self.assertEqual(getattr(thinking_off, field), getattr(base, field))

    def test_qwen38_thinking_off_tail_matches_kolibri_not_nemotron35(self) -> None:
        # The thinking-off prefills of qwen38, kolibri and nanbeige42 are
        # byte-identical (the official qwen3.8 template uses the same tail
        # as kolibri); nemotron35's ends at "<think></think>" with no
        # trailing newlines.
        variants = {variant.name: variant for variant in candidates("qwen38", mutate=True)}
        qwen38_off = variants["qwen38/thinking-off"]
        kolibri_off = {v.name: v for v in candidates("kolibri", mutate=True)}[
            "kolibri/thinking-off"
        ]
        nemotron35_off = {v.name: v for v in candidates("nemotron35", mutate=True)}[
            "nemotron35/thinking-off"
        ]
        self.assertEqual(qwen38_off.open_suffix, kolibri_off.open_suffix)
        self.assertNotEqual(qwen38_off.open_suffix, nemotron35_off.open_suffix)
        self.assertEqual(nemotron35_off.open_suffix, "<|im_start|>assistant\n<think></think>")

    def test_qwen38_has_no_thinking_on_variant(self) -> None:
        # The base entry IS the thinking-on default (the template's
        # enable_thinking defaults true), so there is no
        # qwen38/thinking-on -- symmetric to kolibri, which likewise has
        # only a thinking-off branch.
        variant_names = {variant.name for variant in candidates("qwen38", mutate=True)}
        self.assertNotIn("qwen38/thinking-on", variant_names)
        self.assertIn("qwen38/thinking-off", variant_names)


class Qwen38VsQwen3Tests(unittest.TestCase):
    def test_qwen38_vs_qwen3_shares_frames_differs_in_open_tail(self) -> None:
        # Closest family is qwen3: per-turn frames and BOS are
        # byte-identical; the ONLY modeled difference is the generation
        # tail -- qwen3.8's opens the <think> block, qwen3's does not
        # (qwen3's open_suffix falls back to the bare assistant frame).
        qwen38 = CATALOG["qwen38"]
        qwen3 = CATALOG["qwen3"]
        for field in ("bos", "user_prefix", "assistant_prefix", "turn_end"):
            self.assertEqual(getattr(qwen38, field), getattr(qwen3, field))
        self.assertIsNone(qwen3.open_suffix)
        self.assertEqual(qwen38.open_suffix, "<|im_start|>assistant\n<think>\n")

    def test_qwen38_render_equals_qwen3_render_with_tail_swapped(self) -> None:
        # The qwen38 render is the qwen3 render with exactly the open-reply
        # tail swapped: qwen3 ends at "<|im_start|>assistant\n", qwen3.8 at
        # "<|im_start|>assistant\n<think>\n".
        qwen38_render = CATALOG["qwen38"].render(_MESSAGES)
        qwen3_render = CATALOG["qwen3"].render(_MESSAGES)
        qwen3_tail = "<|im_start|>assistant\n"
        self.assertTrue(qwen3_render.endswith(qwen3_tail))
        self.assertEqual(
            qwen38_render,
            qwen3_render[: -len(qwen3_tail)] + "<|im_start|>assistant\n<think>\n",
        )

    def test_qwen38_not_a_qwen3_alias(self) -> None:
        # The family must be a distinct catalog entry -- byte-distinct from
        # qwen3 on the same fixtures.
        self.assertNotEqual(
            CATALOG["qwen38"].render(_MESSAGES),
            CATALOG["qwen3"].render(_MESSAGES),
        )


class Qwen38DiscoverabilityTests(unittest.TestCase):
    def test_qwen38_registered_in_catalog(self) -> None:
        self.assertIn("qwen38", CATALOG)
        self.assertEqual(CATALOG["qwen38"].name, "qwen38")

    def test_qwen38_candidates_behavior(self) -> None:
        # Without mutation: exactly the base entry. With mutation: the base
        # plus local repairs plus the qwen38/thinking-off branch -- all
        # names unique and qwen38-prefixed.
        self.assertEqual(candidates("qwen38"), (CATALOG["qwen38"],))
        variants = candidates("qwen38", mutate=True)
        names = [variant.name for variant in variants]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertTrue(name == "qwen38" or name.startswith("qwen38/"))
        self.assertIn("qwen38/thinking-off", names)


class Qwen38EvidenceTests(unittest.TestCase):
    def test_qwen38_source_cited_in_catalog(self) -> None:
        # The catalog comment above the qwen38 entry must cite the primary
        # source (the official Qwen/Qwen3.8-27B repo).
        source = (REPO_ROOT / "model_protocol_lab" / "catalog.py").read_text(encoding="utf-8")
        comment_window = source.split('"qwen38": Format')[0][-3200:]
        self.assertIn(_QWEN38_SOURCE_URL, comment_window)
        self.assertIn("chat_template.jinja", comment_window)

    def test_qwen38_xhigh_sentence_documented_exactly(self) -> None:
        # The auto-injected xhigh reasoning sentence is documented, not
        # modeled: pinned byte-exact here as transcribed from the official
        # chat_template.jinja (verified commit 412f8b6). It must never leak
        # into any qwen38 render or mutate variant.
        self.assertEqual(
            _QWEN38_XHIGH_REASONING_SENTENCE,
            "Reasoning effort is set to xhigh. Please think carefully "
            "through the task, validate key assumptions, consider plausible "
            "alternatives, and prioritize correctness, consistency, and "
            "clarity in the final answer.",
        )
        for variant in candidates("qwen38", mutate=True):
            self.assertNotIn(_QWEN38_XHIGH_REASONING_SENTENCE, variant.render(_MESSAGES))


if __name__ == "__main__":
    unittest.main()
