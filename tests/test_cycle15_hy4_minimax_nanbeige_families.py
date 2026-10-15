"""Cycle 15 (continuous dev): Tencent Hy4, MiniMax-M3, Nanbeige 4.2-3B families.

Byte-exact render checks for the three families triaged as pinnable in
the cycle-14 research log and re-verified against the primary sources
this round (2026-10-15):

1. hy4 -- Tencent Hy4 preview (tencent/Hy4-preview, released 2026-08-28,
   Apache 2.0; https://huggingface.co/tencent/Hy4-preview).
   Evidence tier 1: the official chat_template.jinja (preserved
   byte-identical at docs/evidence/hy4-chat-template.jinja, 309 lines,
   md5 a0b7e28386487d804009340155e7be2b) -- every string here is derived
   from the template's plain-text, no-tools branch (jinja line numbers in
   the catalog comment), and cross-checked by rendering the official
   template byte-exact. The template compiles under stock Jinja2
   (raise_exception, lines 36/38, is render-time only).
   - Frames: <｜hy_start:opensource｜>{role}<｜hy_middle:opensource｜>
     {content}<｜hy_end:opensource｜> (｜ = U+FF5C FULLWIDTH VERTICAL BAR,
     spelled as \\uff5c escapes here so the file is exact regardless of
     editor encoding -- the same convention as _DS_BAR in the catalog).
   - Assistant history, no reasoning (line 243): <think:opensource>
     </think:opensource> (empty think tags) + content -- the lab's
     deterministic no-reasoning hypothesis; the reasoned-content branch
     (line 241) is not a hypothesis.
   - Generation prompt: reasoning_effort defaults to 'high' (line 33):
     ...<think:opensource> (line 306, base entry open_suffix);
     'no_think' closes the think block (line 304) and rides the mutate
     path as hy4/no-think (kolibri/thinking-off convention).
   - System synthesis (lines 196-204): with no leading system message the
     template ALWAYS emits a reasoning_mode system block -- documented,
     not modeled (the lab's Format only renders user/assistant turns).
   - BOS: the template emits no BOS literal, so the entry's bos is ""
     (phi4/qwen3 convention); the tokenizer_config.json bos_token is
     <｜hy_start:opensource｜> and eos_token <｜hy_end:opensource｜>
     (tier-1 verified 2026-10-15).

2. minimax3 -- MiniMax-M3 (MiniMaxAI/MiniMax-M3, released 2026-06,
   MiniMax Community License; https://huggingface.co/MiniMaxAI/MiniMax-M3).
   Evidence tier 1: the official chat_template.jinja (preserved
   byte-identical at docs/evidence/minimax-chat-template.jinja, 247
   lines, md5 455403f9cba16ea73388aeb8898da752) -- every string here is
   derived from the template's plain-text, no-tools branch and
   cross-checked by rendering the official template byte-exact.
   - Template-local tokens: bod ]~!b[, bos ]~b[, eos [e~[; think
     <mm:think>...</mm:think> (lines 2-11).
   - System super-prompt (lines 113-117): ALWAYS emitted, even with no
     messages: ]~!b[]~b]system\\n + default identity + the
     <thinking_instructions> block + [e~[\\n -- documented, not modeled.
   - Developer super-prompt (lines 120-121): ]~b]developer\\n"You are a
     helpful assistant."[e~[\\n -- documented, not modeled.
   - User (line 228): ]~b]user\\n{content}[e~[\\n.
   - Assistant history, no reasoning (lines 167-168): ]~b]ai\\n
     </mm:think>{content}[e~[\\n -- the think_end-prefix branch is the
     lab's deterministic no-reasoning hypothesis; the reasoned-content
     branch (line 164) is not a hypothesis.
   - Generation prompt (lines 234-245): thinking_mode undefined (the
     default) or 'adaptive' -> ]~b]ai\\n with NO prefix (base entry
     open_suffix); 'disabled' -> ]~b]ai\\n</mm:think> (line 238);
     'enabled' -> ]~b]ai\\n<mm:think> (line 242) -- the disabled/enabled
     branches ride the mutate path as minimax3/thinking-disabled and
     minimax3/thinking-enabled (kolibri/thinking-off convention).
   - BOS: the template emits no BOS literal (the ]~b[ frame prefixes are
     message prefixes, not a BOS), so the entry's bos is "" (phi4/qwen3
     convention); tokenizer_config.json: bos_token ]~b], eos_token [e~[,
     pad_token ]!p~[ (tier-1 verified 2026-10-15).

3. nanbeige42 -- Nanbeige 4.2-3B (Nanbeige/Nanbeige4.2-3B, Nanbeige LLM
   Lab / Boss Zhipin, paper 2026-07-27;
   https://huggingface.co/Nanbeige/Nanbeige4.2-3B).
   Evidence tier 1: the full chat_template embedded in the official
   tokenizer_config.json (no standalone chat_template.jinja -- verified
   404 this round; the embedded template is preserved at
   docs/evidence/nanbeige-chat-template.jinja) -- every string here is
   derived from the template's plain-text, no-tools branch and
   cross-checked by rendering the embedded template byte-exact. Config:
   bos_token <|im_start|>, eos_token <|im_end|>, tokenizer_class
   LlamaTokenizer.
   - Default system prompt (lines 59-62, no tools): ALWAYS emitted:
     <|im_start|>system\\n + Chinese Nanbeige identity line +
     <|im_end|>\\n -- documented, not modeled.
   - User (line 150): <|im_start|>user\\n{content}<|im_end|>\\n.
   - Assistant history, no reasoning (line 99): <|im_start|>assistant
     \\n<think>\\n\\n</think>\\n\\n{content}<|im_end|>\\n -- the empty
     think block is the lab's deterministic no-reasoning hypothesis; the
     reasoned-content branch (line 101) is not a hypothesis.
   - Generation prompt (lines 181-187): default (thinking on) ->
     <|im_start|>assistant\\n<think>\\n (base entry open_suffix);
     enable_thinking=false -> <|im_start|>assistant\\n<think>\\n\\n
     </think>\\n\\n (line 185) -- rides the mutate path as
     nanbeige42/thinking-off (kolibri convention).
   - BOS: the template emits no BOS literal, so the entry's bos is ""
     (phi4/qwen3 convention).

Also: the mutate-variant pins, the dedup-key no-collision pins, and a
per-family diff-vs-all-other-families pin (dynamic: len(CATALOG) - 1).
There is intentionally NO README naming pin in this file: the README
counts/naming are owned by the coordinator this cycle, and a pin here
would fail before the README update lands.

Deterministic; no network, no model, no randomness.
"""

import re
import unittest
from pathlib import Path

from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.core import Format, Message, local_repairs

REPO_ROOT = Path(__file__).resolve().parent.parent

_MESSAGES_1: tuple[Message, ...] = (Message("user", "Hello"),)

_MESSAGES_2: tuple[Message, ...] = (
    Message("user", "Hello"),
    Message("assistant", "Hi there"),
    Message("user", "Bye"),
)


# --- Tencent Hy4 ---------------------------------------------------------------

# Derived from docs/evidence/hy4-chat-template.jinja (official Tencent
# template, 309 lines; verified by rendering the official template
# byte-exact, minus its ALWAYS-synthesized system block). ｜ is U+FF5C,
# spelled as an escape so the file is exact regardless of editor encoding.
_HY4_USER_PREFIX = "<\uff5chy_start:opensource\uff5c>user<\uff5chy_middle:opensource\uff5c>"
_HY4_ASSISTANT_PREFIX = (
    "<\uff5chy_start:opensource\uff5c>assistant<\uff5chy_middle:opensource\uff5c>"
    "<think:opensource></think:opensource>"
)
_HY4_TURN_END = "<\uff5chy_end:opensource\uff5c>"
# Generation prompt, reasoning_effort 'high' (the default, jinja line 33):
# ...assistant<｜hy_middle:opensource｜><think:opensource> (jinja line 306).
_HY4_OPEN_SUFFIX = (
    "<\uff5chy_start:opensource\uff5c>assistant<\uff5chy_middle:opensource\uff5c><think:opensource>"
)
# Generation-prompt branch reasoning_effort='no_think' (jinja line 304):
# the think block closes immediately.
_HY4_NO_THINK_SUFFIX = (
    "<\uff5chy_start:opensource\uff5c>assistant<\uff5chy_middle:opensource\uff5c>"
    "<think:opensource></think:opensource>"
)

# The ALWAYS-synthesized system block (jinja lines 196-204) is documented,
# not modeled (the lab's Format only renders user/assistant turns).
# reasoning_effort defaults to 'high'; the block is byte-identical to the
# official template's render.
_HY4_SYSTEM_BLOCK = (
    "<\uff5chy_start:opensource\uff5c>system<\uff5chy_middle:opensource\uff5c>"
    "<\uff5creasoning_mode:opensource\uff5c>reasoning_effort:high<\uff5chy_end:opensource\uff5c>"
)

_HY4_EXPECTED_1 = _HY4_USER_PREFIX + "Hello" + _HY4_TURN_END + _HY4_OPEN_SUFFIX

_HY4_EXPECTED_2 = (
    _HY4_USER_PREFIX
    + "Hello"
    + _HY4_TURN_END
    + _HY4_ASSISTANT_PREFIX
    + "Hi there"
    + _HY4_TURN_END
    + _HY4_USER_PREFIX
    + "Bye"
    + _HY4_TURN_END
    + _HY4_OPEN_SUFFIX
)


class Hy4RenderTests(unittest.TestCase):
    def test_hy4_renders_byte_exact_1_turn(self) -> None:
        self.assertEqual(CATALOG["hy4"].render(_MESSAGES_1), _HY4_EXPECTED_1)

    def test_hy4_renders_byte_exact_2_turn(self) -> None:
        self.assertEqual(CATALOG["hy4"].render(_MESSAGES_2), _HY4_EXPECTED_2)

    def test_hy4_fields_match_catalog_entry(self) -> None:
        fmt = CATALOG["hy4"]
        self.assertEqual(fmt.bos, "")
        self.assertEqual(fmt.user_prefix, _HY4_USER_PREFIX)
        self.assertEqual(fmt.assistant_prefix, _HY4_ASSISTANT_PREFIX)
        self.assertEqual(fmt.turn_end, _HY4_TURN_END)
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertEqual(fmt.open_suffix, _HY4_OPEN_SUFFIX)
        self.assertFalse(fmt.repeat_bos)

    def test_hy4_markers_character_exact(self) -> None:
        # The fullwidth bar is U+FF5C, not an ASCII pipe -- a plain '|'
        # would silently collide with ASCII-pipe families in dedup.
        self.assertEqual(ord("｜"), 0xFF5C)
        self.assertIn("｜", _HY4_USER_PREFIX)
        self.assertNotIn("|", _HY4_USER_PREFIX)
        self.assertTrue(_HY4_USER_PREFIX.endswith("user<\uff5chy_middle:opensource\uff5c>"))
        self.assertEqual(_HY4_TURN_END, "<\uff5chy_end:opensource\uff5c>")
        self.assertNotIn("\n", _HY4_TURN_END)
        self.assertEqual(
            _HY4_ASSISTANT_PREFIX,
            "<\uff5chy_start:opensource\uff5c>assistant<\uff5chy_middle:opensource\uff5c>"
            "<think:opensource></think:opensource>",
        )
        self.assertEqual(
            _HY4_OPEN_SUFFIX,
            "<\uff5chy_start:opensource\uff5c>assistant<\uff5chy_middle:opensource\uff5c>"
            "<think:opensource>",
        )

    def test_hy4_system_block_documented_not_modeled(self) -> None:
        # With no leading system message the real template ALWAYS emits a
        # reasoning_mode system block (jinja lines 196-204); the lab's
        # Format only renders user/assistant turns, so the block is a
        # documented constant (pinned byte-exact here), not a render field.
        self.assertEqual(
            _HY4_SYSTEM_BLOCK,
            "<\uff5chy_start:opensource\uff5c>system<\uff5chy_middle:opensource\uff5c>"
            "<\uff5creasoning_mode:opensource\uff5c>reasoning_effort:high"
            "<\uff5chy_end:opensource\uff5c>",
        )
        rendered = CATALOG["hy4"].render(_MESSAGES_2)
        self.assertNotIn("<\uff5chy_start:opensource\uff5c>system", rendered)

    def test_hy4_eos_caveat_documented(self) -> None:
        # The official tokenizer_config.json eos_token field is
        # <｜hy_end:opensource｜>; the catalog documents this and asserts no
        # stray EOS string beyond the turn end (Format has no EOS field).
        fmt = CATALOG["hy4"]
        frames = fmt.bos + fmt.user_prefix + fmt.assistant_prefix + fmt.turn_end
        self.assertNotIn("\n", fmt.turn_end)
        self.assertEqual(frames.count("<\uff5chy_end:opensource\uff5c>"), 1)


# --- MiniMax-M3 ----------------------------------------------------------------

# Derived from docs/evidence/minimax-chat-template.jinja (official MiniMax
# template, 247 lines; verified by rendering the official template
# byte-exact, minus the ALWAYS-emitted system + developer super-prompts).
_MM3_USER_PREFIX = "]~b]user\n"
_MM3_ASSISTANT_PREFIX = "]~b]ai\n</mm:think>"
_MM3_TURN_END = "[e~[\n"
# Generation prompt, thinking_mode undefined/adaptive (the default): no
# prefix after the ai frame (jinja lines 234-237, 239-241).
_MM3_OPEN_SUFFIX = "]~b]ai\n"
# Generation-prompt branches (jinja lines 238/242): 'disabled' closes the
# think block; 'enabled' opens it.
_MM3_DISABLED_SUFFIX = "]~b]ai\n</mm:think>"
_MM3_ENABLED_SUFFIX = "]~b]ai\n<mm:think>"

# The ALWAYS-emitted system super-prompt (jinja lines 113-117) and
# developer super-prompt (jinja lines 120-121) are documented, not modeled
# (the lab's Format only renders user/assistant turns). Both pinned
# byte-exact against the official template's render.
_MM3_SYSTEM_SUPERPROMPT = (
    "]~!b[]~b]system\n"
    "Your model version is MiniMax-M3, developed by MiniMax. Knowledge "
    "cutoff: January 2026. Founded in early 2022, MiniMax is a global AI "
    "foundation model company committed to advancing the frontiers of AI "
    "towards AGI."
    "\n\n<thinking_instructions>\n"
    "You have a thinking capability that allows you to reason step by step "
    "before responding. When thinking is enabled, wrap your reasoning in "
    "<mm:think></mm:think> tags before your response. When thinking is "
    "disabled, begin your response directly after the </mm:think> prefix. "
    "When thinking is adaptive, decide on your own whether to think for the "
    "current turn.\n"
    "Current thinking mode: adaptive. You are encouraged to think for complex "
    "decision-making, multi-step reasoning, or when analyzing function/tool "
    "results.\n"
    "</thinking_instructions>"
    "[e~[\n"
)
_MM3_DEVELOPER_SUPERPROMPT = "]~b]developer\nYou are a helpful assistant.[e~[\n"

_MM3_EXPECTED_1 = _MM3_USER_PREFIX + "Hello" + _MM3_TURN_END + _MM3_OPEN_SUFFIX

_MM3_EXPECTED_2 = (
    _MM3_USER_PREFIX
    + "Hello"
    + _MM3_TURN_END
    + _MM3_ASSISTANT_PREFIX
    + "Hi there"
    + _MM3_TURN_END
    + _MM3_USER_PREFIX
    + "Bye"
    + _MM3_TURN_END
    + _MM3_OPEN_SUFFIX
)


class MiniMax3RenderTests(unittest.TestCase):
    def test_minimax3_renders_byte_exact_1_turn(self) -> None:
        self.assertEqual(CATALOG["minimax3"].render(_MESSAGES_1), _MM3_EXPECTED_1)

    def test_minimax3_renders_byte_exact_2_turn(self) -> None:
        self.assertEqual(CATALOG["minimax3"].render(_MESSAGES_2), _MM3_EXPECTED_2)

    def test_minimax3_fields_match_catalog_entry(self) -> None:
        fmt = CATALOG["minimax3"]
        self.assertEqual(fmt.bos, "")
        self.assertEqual(fmt.user_prefix, _MM3_USER_PREFIX)
        self.assertEqual(fmt.assistant_prefix, _MM3_ASSISTANT_PREFIX)
        self.assertEqual(fmt.turn_end, _MM3_TURN_END)
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertEqual(fmt.open_suffix, _MM3_OPEN_SUFFIX)
        self.assertFalse(fmt.repeat_bos)

    def test_minimax3_markers_character_exact(self) -> None:
        # The bracket-tilde dialect: frames open with ]~b], turns end with
        # [e~[ and a newline; the no-reasoning assistant history frame
        # carries the </mm:think> prefix (jinja line 167), while the default
        # generation prompt is bare (jinja lines 234-241).
        self.assertEqual(_MM3_USER_PREFIX, "]~b]user\n")
        self.assertEqual(_MM3_TURN_END, "[e~[\n")
        self.assertEqual(_MM3_ASSISTANT_PREFIX, "]~b]ai\n</mm:think>")
        self.assertEqual(_MM3_OPEN_SUFFIX, "]~b]ai\n")
        self.assertNotIn("]~!b[", _MM3_USER_PREFIX + _MM3_ASSISTANT_PREFIX + _MM3_TURN_END)

    def test_minimax3_superprompts_documented_not_modeled(self) -> None:
        # The real template ALWAYS emits a system super-prompt (the bod
        # ]~!b[ + identity + <thinking_instructions>, jinja lines 113-117)
        # and a developer super-prompt ("]~b]developer\\nYou are a helpful
        # assistant.", jinja lines 120-121); both are documented constants
        # (pinned byte-exact here), not render fields.
        self.assertTrue(_MM3_SYSTEM_SUPERPROMPT.startswith("]~!b[]~b]system\n"))
        self.assertIn("MiniMax-M3", _MM3_SYSTEM_SUPERPROMPT)
        self.assertTrue(_MM3_SYSTEM_SUPERPROMPT.endswith("[e~[\n"))
        self.assertEqual(
            _MM3_DEVELOPER_SUPERPROMPT, "]~b]developer\nYou are a helpful assistant.[e~[\n"
        )
        rendered = CATALOG["minimax3"].render(_MESSAGES_2)
        self.assertNotIn("]~!b[]~b]system", rendered)
        self.assertNotIn("]~b]developer", rendered)

    def test_minimax3_eos_caveat_documented(self) -> None:
        # The official tokenizer_config.json eos_token field is [e~[; the
        # catalog documents this and asserts the frames carry no stray EOS
        # (Format has no EOS field).
        fmt = CATALOG["minimax3"]
        self.assertEqual(fmt.turn_end.count("[e~["), 1)
        self.assertNotIn("[e~[", fmt.user_prefix + fmt.assistant_prefix)


# --- Nanbeige 4.2-3B ------------------------------------------------------------

# Derived from docs/evidence/nanbeige-chat-template.jinja (the chat_template
# embedded in the official tokenizer_config.json -- no standalone jinja
# file, verified 404 2026-10-15; verified by rendering the embedded
# template byte-exact, minus the ALWAYS-emitted default system prompt).
_NBG_USER_PREFIX = "<|im_start|>user\n"
_NBG_ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
_NBG_TURN_END = "<|im_end|>\n"
# Generation prompt, default thinking-on (jinja line 187):
# <|im_start|>assistant\n<think>\n.
_NBG_OPEN_SUFFIX = "<|im_start|>assistant\n<think>\n"
# Generation-prompt branch enable_thinking=false (jinja line 185):
# the think block is prefilled empty.
_NBG_THINKING_OFF_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

# The ALWAYS-emitted default system prompt (jinja lines 59-62, no tools)
# is documented, not modeled (the lab's Format only renders
# user/assistant turns). Pinned byte-exact against the embedded template.
_NBG_SYSTEM_PROMPT = (
    "<|im_start|>system\n你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。<|im_end|>\n"
)

_NBG_EXPECTED_1 = _NBG_USER_PREFIX + "Hello" + _NBG_TURN_END + _NBG_OPEN_SUFFIX

_NBG_EXPECTED_2 = (
    _NBG_USER_PREFIX
    + "Hello"
    + _NBG_TURN_END
    + _NBG_ASSISTANT_PREFIX
    + "Hi there"
    + _NBG_TURN_END
    + _NBG_USER_PREFIX
    + "Bye"
    + _NBG_TURN_END
    + _NBG_OPEN_SUFFIX
)


class Nanbeige42RenderTests(unittest.TestCase):
    def test_nanbeige42_renders_byte_exact_1_turn(self) -> None:
        self.assertEqual(CATALOG["nanbeige42"].render(_MESSAGES_1), _NBG_EXPECTED_1)

    def test_nanbeige42_renders_byte_exact_2_turn(self) -> None:
        self.assertEqual(CATALOG["nanbeige42"].render(_MESSAGES_2), _NBG_EXPECTED_2)

    def test_nanbeige42_fields_match_catalog_entry(self) -> None:
        fmt = CATALOG["nanbeige42"]
        self.assertEqual(fmt.bos, "")
        self.assertEqual(fmt.user_prefix, _NBG_USER_PREFIX)
        self.assertEqual(fmt.assistant_prefix, _NBG_ASSISTANT_PREFIX)
        self.assertEqual(fmt.turn_end, _NBG_TURN_END)
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertEqual(fmt.open_suffix, _NBG_OPEN_SUFFIX)
        self.assertFalse(fmt.repeat_bos)

    def test_nanbeige42_markers_character_exact(self) -> None:
        # ChatML-like with mandatory think blocks: the history assistant
        # frame carries an EMPTY <think>\n\n</think>\n\n block (jinja line
        # 99), the generation prompt opens <think>\n (jinja line 187), and
        # every turn closes with <|im_end|>\n.
        self.assertEqual(_NBG_USER_PREFIX, "<|im_start|>user\n")
        self.assertEqual(_NBG_ASSISTANT_PREFIX, "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.assertEqual(_NBG_TURN_END, "<|im_end|>\n")
        self.assertEqual(_NBG_OPEN_SUFFIX, "<|im_start|>assistant\n<think>\n")

    def test_nanbeige42_system_prompt_documented_not_modeled(self) -> None:
        # The real template ALWAYS emits a default system prompt (the
        # Chinese Nanbeige identity line, jinja lines 59-62) when the
        # conversation has no leading system message; the lab's Format only
        # renders user/assistant turns, so the block is a documented
        # constant (pinned byte-exact here), not a render field.
        self.assertEqual(
            _NBG_SYSTEM_PROMPT,
            "<|im_start|>system\n你是南北阁，一款由BOSS直聘自主研发并训练的"
            "专业大语言模型。<|im_end|>\n",
        )
        rendered = CATALOG["nanbeige42"].render(_MESSAGES_2)
        self.assertNotIn("<|im_start|>system", rendered)

    def test_nanbeige42_eos_caveat_documented(self) -> None:
        # The official tokenizer_config.json eos_token field is <|im_end|>;
        # the catalog documents this and asserts the frames carry no stray
        # EOS beyond the turn end (Format has no EOS field).
        fmt = CATALOG["nanbeige42"]
        self.assertEqual(fmt.turn_end, "<|im_end|>\n")
        self.assertNotIn("<|im_end|>", fmt.user_prefix + fmt.assistant_prefix)


# --- Mutate-path variants --------------------------------------------------------


class Cycle15MutateTests(unittest.TestCase):
    def _variants(self, family: str) -> dict[str, Format]:
        return {variant.name: variant for variant in candidates(family, mutate=True)}

    def _check_branch(
        self,
        family: str,
        branch_name: str,
        expected_suffix: str,
        expected_2: str,
        base_open_suffix: str,
    ) -> None:
        variants = self._variants(family)
        self.assertIn(branch_name, variants)
        branch = variants[branch_name]
        base = CATALOG[family]
        # Same frames as the base entry; only the generation prompt
        # changes.
        for field in ("bos", "user_prefix", "assistant_prefix", "turn_end"):
            self.assertEqual(getattr(branch, field), getattr(base, field))
        self.assertEqual(branch.open_suffix, expected_suffix)
        rendered = branch.render(_MESSAGES_2)
        self.assertTrue(rendered.endswith(expected_suffix))
        self.assertEqual(
            rendered[: -len(expected_suffix)],
            expected_2[: -len(base_open_suffix)],
        )

    def test_hy4_no_think_mutation_pin(self) -> None:
        # reasoning_effort='no_think' (jinja line 304): the generation
        # prompt closes the think block immediately.
        self._check_branch(
            "hy4", "hy4/no-think", _HY4_NO_THINK_SUFFIX, _HY4_EXPECTED_2, _HY4_OPEN_SUFFIX
        )

    def test_minimax3_thinking_disabled_mutation_pin(self) -> None:
        # thinking_mode='disabled' (jinja line 238): the generation prompt
        # closes the think block.
        self._check_branch(
            "minimax3",
            "minimax3/thinking-disabled",
            _MM3_DISABLED_SUFFIX,
            _MM3_EXPECTED_2,
            _MM3_OPEN_SUFFIX,
        )

    def test_minimax3_thinking_enabled_mutation_pin(self) -> None:
        # thinking_mode='enabled' (jinja line 242): the generation prompt
        # opens the think block.
        self._check_branch(
            "minimax3",
            "minimax3/thinking-enabled",
            _MM3_ENABLED_SUFFIX,
            _MM3_EXPECTED_2,
            _MM3_OPEN_SUFFIX,
        )

    def test_nanbeige42_thinking_off_mutation_pin(self) -> None:
        # enable_thinking=false (jinja line 185): the generation prompt
        # prefills an empty <think> block (kolibri/thinking-off convention).
        self._check_branch(
            "nanbeige42",
            "nanbeige42/thinking-off",
            _NBG_THINKING_OFF_SUFFIX,
            _NBG_EXPECTED_2,
            _NBG_OPEN_SUFFIX,
        )

    def test_cycle15_branch_dedup_no_collision(self) -> None:
        # Each branch open_suffix must be genuinely new -- i.e. its
        # local_repairs dedup key must differ from every key already
        # present, or candidates() would silently drop it. This mirrors the
        # "cannot collide with local_repairs' dedup keys" notes in
        # catalog.candidates().
        def _key(fmt: Format) -> tuple[object, ...]:
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

        branches = (
            ("hy4", "hy4/no-think"),
            ("minimax3", "minimax3/thinking-disabled"),
            ("minimax3", "minimax3/thinking-enabled"),
            ("nanbeige42", "nanbeige42/thinking-off"),
        )
        for family, branch_name in branches:
            base = CATALOG[family]
            repair_keys = {_key(variant) for variant in local_repairs(base)}
            variants = self._variants(family)
            self.assertNotIn(_key(variants[branch_name]), repair_keys)


# --- Diff-vs-all-other-families ---------------------------------------------------


class Cycle15DiffTests(unittest.TestCase):
    def _assert_differs(self, family: str) -> None:
        mine = CATALOG[family].render(_MESSAGES_2)
        others = [name for name in CATALOG if name != family]
        self.assertEqual(len(others), len(CATALOG) - 1)
        collisions = [name for name in others if CATALOG[name].render(_MESSAGES_2) == mine]
        self.assertEqual(collisions, [])

    def test_hy4_differs_from_every_other_family(self) -> None:
        self._assert_differs("hy4")

    def test_minimax3_differs_from_every_other_family(self) -> None:
        self._assert_differs("minimax3")

    def test_nanbeige42_differs_from_every_other_family(self) -> None:
        self._assert_differs("nanbeige42")


class ReadmeCycle15NamesTests(unittest.TestCase):
    def test_readme_names_new_families(self) -> None:
        # The README's family list must name the new families, and its
        # "N-family catalog" claims must track len(CATALOG) (the absolute
        # count pins live in tests/test_cycle10_families.py and
        # tests/test_cycle11_families.py).
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Hy4", text)
        self.assertIn("MiniMax-M3", text)
        self.assertIn("Nanbeige", text)
        matches = re.findall(r"(\d+)-family catalog", text)
        self.assertTrue(matches, "README lost its N-family catalog claim")
        for match in matches:
            self.assertEqual(int(match), len(CATALOG))


if __name__ == "__main__":
    unittest.main()
