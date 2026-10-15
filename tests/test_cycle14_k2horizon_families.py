"""Cycle 14 (continuous dev): IFM K2-Horizon fleet catalog entry.

Byte-exact render checks for the IFM K2-Horizon fleet (Institute of
Foundation Models, launched by MBZUAI; released 2026-09-03, Apache 2.0;
six models sharing one chat template: K2-Horizon-375B-A23B,
K2-Horizon-MoVA-36B-A4B, K2-Horizon-32B, K2-Horizon-7B, K2-Horizon-3.7B,
K2-Horizon-0.9B; https://huggingface.co/IFM/K2-Horizon-375B-A23B).
Evidence tier 1: the official chat_template.jinja (preserved at
docs/evidence/k2horizon-chat-template.jinja) -- every string here is
derived from the template's plain-text, no-tools branch (jinja line
numbers in the catalog comment), and cross-checked by rendering the
official template with a no-op Jinja extension for its custom
{% generation %} tag:

- BOS <|ifm|begin_of_text|> (jinja line 1, {{- bos_token }}; tier-1
  verified from the official tokenizer_config.json bos_token field, read
  2026-10-15 on the 375B-A23B and MoVA-36B-A4B repos; the two configs are
  identical).
- User (line 959): <|ifm|im_start|>user\\n{content}<|ifm|im_end|> (no
  trailing newline).
- Assistant history, no reasoning (lines 983-997): the template REQUIRES
  a thinking field on assistant messages (message.think / think_fast /
  think_faster / reasoning_content / reasoning as a string); the lab's
  deterministic no-reasoning hypothesis is an EMPTY think field (line
  989): <|ifm|im_start|>assistant\\n<ifm|think>\\n</ifm|think>{content}
  <|ifm|im_end|>. The \\n after "assistant" is the source line break
  inside the {% generation %} block under standard Jinja block-extension
  semantics (trim_blocks/lstrip_blocks, the transformers chat-template
  env) -- the only render consistent with the template's own explicit
  generation prompt (line 1006).
- Generation prompt, effort high (default, line 1006):
  <|ifm|im_start|>assistant\\n<ifm|think>\\n. Effort medium/low prefill
  the effort-matched think tag (lines 1008/1010) and ride the mutate path
  as k2horizon/effort-medium and k2horizon/effort-low
  (kolibri/thinking-off convention).
- System block (line 948, no tools): <|ifm|im_start|>system\\n{content}
  <|ifm|im_end|> -- documented, not modeled (the lab's Format only renders
  user/assistant turns).
- EOS caveat: the official eos_token field is <|ifm|endoftext|>, and
  generation_config.json lists eos_token_id [1, 250019] (250019 =
  <|ifm|im_end|>); documented, not modeled (Format has no EOS field).
- {% generation %} caveat: the official template's {% generation %} /
  {% endgeneration %} tags are not valid stock Jinja2 (verified
  TemplateSyntaxError with jinja2 3.1.6 and transformers 5.19.0's
  sandboxed env); they emit no bytes. Community renderers that strip the
  tags textually emit the assistant frame WITHOUT the \\n after
  "assistant" -- a known ecosystem divergence; the catalog models the
  proper-extension render.

Also: the effort-variant mutation pins, the dedup-key no-collision pin, a
diff-vs-all-other-families pin (k2horizon renders byte-different from
every one of the other 23 families on the same messages), and a README
naming pin for "K2-Horizon" (the N-family count pins live in
tests/test_cycle10_families.py, tests/test_cycle11_families.py and
tests/test_cycle13_families.py).

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

# Derived from docs/evidence/k2horizon-chat-template.jinja (official IFM
# template, 1014 lines; BOS <|ifm|begin_of_text|> from the official
# tokenizer_config.json bos_token field, read 2026-10-15). Verified by
# rendering the official template with a no-op {% generation %} Jinja
# extension: these strings are byte-identical to the official render for
# plain messages (assistant history with an empty think field).
_K2H_BOS = "<|ifm|begin_of_text|>"
_K2H_USER_PREFIX = "<|ifm|im_start|>user\n"
_K2H_ASSISTANT_PREFIX = "<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>"
_K2H_TURN_END = "<|ifm|im_end|>"
# Generation prompt, reasoning_effort 'high' (the default, jinja line
# 1005): <|ifm|im_start|>assistant\n<ifm|think>\n (jinja line 1006).
_K2H_OPEN_SUFFIX = "<|ifm|im_start|>assistant\n<ifm|think>\n"
# Generation-prompt branches (jinja lines 1008/1010): the effort-matched
# think tag opens in place of <ifm|think>.
_K2H_EFFORT_MEDIUM_SUFFIX = "<|ifm|im_start|>assistant\n<ifm|think_fast>\n"
_K2H_EFFORT_LOW_SUFFIX = "<|ifm|im_start|>assistant\n<ifm|think_faster>\n"

# The no-tools system block (jinja line 948) is documented, not modeled
# (the lab's Format only renders user/assistant turns).
_K2H_SYSTEM_BLOCK = "<|ifm|im_start|>system\nsys-in<|ifm|im_end|>"

_K2H_EXPECTED_1 = _K2H_BOS + _K2H_USER_PREFIX + "Hello" + _K2H_TURN_END + _K2H_OPEN_SUFFIX

_K2H_EXPECTED_2 = (
    _K2H_BOS
    + _K2H_USER_PREFIX
    + "Hello"
    + _K2H_TURN_END
    + _K2H_ASSISTANT_PREFIX
    + "Hi there"
    + _K2H_TURN_END
    + _K2H_USER_PREFIX
    + "Bye"
    + _K2H_TURN_END
    + _K2H_OPEN_SUFFIX
)


class K2HorizonRenderTests(unittest.TestCase):
    def test_k2horizon_renders_byte_exact_1_turn(self) -> None:
        self.assertEqual(CATALOG["k2horizon"].render(_MESSAGES_1), _K2H_EXPECTED_1)

    def test_k2horizon_renders_byte_exact_2_turn(self) -> None:
        self.assertEqual(CATALOG["k2horizon"].render(_MESSAGES_2), _K2H_EXPECTED_2)

    def test_k2horizon_fields_match_catalog_entry(self) -> None:
        fmt = CATALOG["k2horizon"]
        self.assertEqual(fmt.bos, _K2H_BOS)
        self.assertEqual(fmt.user_prefix, _K2H_USER_PREFIX)
        self.assertEqual(fmt.assistant_prefix, _K2H_ASSISTANT_PREFIX)
        self.assertEqual(fmt.turn_end, _K2H_TURN_END)
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertEqual(fmt.open_suffix, _K2H_OPEN_SUFFIX)
        self.assertFalse(fmt.repeat_bos)

    def test_k2horizon_markers_character_exact(self) -> None:
        # Every structural marker asserted character-for-character: BOS has
        # no trailing newline; the user frame opens with a newline after
        # the role and closes with no trailing newline; the assistant frame
        # has the \n after "assistant" (the {% generation %} line break),
        # then the empty think block; the turn end is bare.
        self.assertEqual(_K2H_BOS, "<|ifm|begin_of_text|>")
        self.assertTrue(_K2H_USER_PREFIX.endswith("user\n"))
        self.assertTrue(_K2H_USER_PREFIX.startswith("<|ifm|im_start|>"))
        self.assertEqual(_K2H_TURN_END, "<|ifm|im_end|>")
        self.assertNotIn("\n", _K2H_TURN_END)
        self.assertEqual(
            _K2H_ASSISTANT_PREFIX,
            "<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>",
        )
        self.assertEqual(_K2H_OPEN_SUFFIX, "<|ifm|im_start|>assistant\n<ifm|think>\n")

    def test_k2horizon_system_block_documented_not_modeled(self) -> None:
        # The real template emits a system block for a leading system
        # message (jinja line 948, no tools); the lab's Format only renders
        # user/assistant turns, so the block is a documented constant
        # (pinned byte-exact here), not a render field -- the same
        # convention as the kolibri/ling system-block notes.
        self.assertEqual(
            _K2H_SYSTEM_BLOCK,
            "<|ifm|im_start|>system\nsys-in<|ifm|im_end|>",
        )
        rendered = CATALOG["k2horizon"].render(_MESSAGES_2)
        self.assertNotIn("<|ifm|im_start|>system", rendered)

    def test_k2horizon_eos_caveat_documented(self) -> None:
        # The official tokenizer_config.json eos_token field is
        # <|ifm|endoftext|>, and generation_config.json lists eos_token_id
        # [1, 250019] (250019 = <|ifm|im_end|>); the catalog documents this
        # and asserts no EOS string anywhere in the entry's frames (Format
        # has no EOS field).
        fmt = CATALOG["k2horizon"]
        frames = fmt.bos + fmt.user_prefix + fmt.assistant_prefix
        self.assertNotIn("<|ifm|endoftext|>", frames)
        self.assertNotIn("<|ifm|endoftext|>", fmt.turn_end + (fmt.open_suffix or ""))

    def test_k2horizon_base_is_effort_high_default(self) -> None:
        # reasoning_effort defaults to 'high' (jinja line 1005), so the base
        # entry's generation tail opens <ifm|think> (line 1006) -- the
        # mutate path supplies the medium/low branches (lines 1008/1010).
        rendered = CATALOG["k2horizon"].render(_MESSAGES_2)
        self.assertTrue(rendered.endswith(_K2H_OPEN_SUFFIX))
        self.assertIn("<ifm|think>\n", rendered[-len(_K2H_OPEN_SUFFIX) :])


# --- Effort-branch mutate-path variants ---------------------------------------


class K2HorizonEffortTests(unittest.TestCase):
    def test_k2horizon_effort_medium_mutation_pin(self) -> None:
        variants = {variant.name: variant for variant in candidates("k2horizon", mutate=True)}
        self.assertIn("k2horizon/effort-medium", variants)
        medium = variants["k2horizon/effort-medium"]
        base = CATALOG["k2horizon"]
        # Same frames as the base entry; only the generation prompt
        # changes: the effort-matched think tag opens (jinja line 1008).
        for field in ("bos", "user_prefix", "assistant_prefix", "turn_end"):
            self.assertEqual(getattr(medium, field), getattr(base, field))
        self.assertEqual(medium.open_suffix, _K2H_EFFORT_MEDIUM_SUFFIX)
        rendered = medium.render(_MESSAGES_2)
        self.assertTrue(rendered.endswith(_K2H_EFFORT_MEDIUM_SUFFIX))
        self.assertEqual(
            rendered[: -len(_K2H_EFFORT_MEDIUM_SUFFIX)],
            _K2H_EXPECTED_2[: -len(_K2H_OPEN_SUFFIX)],
        )

    def test_k2horizon_effort_low_mutation_pin(self) -> None:
        variants = {variant.name: variant for variant in candidates("k2horizon", mutate=True)}
        self.assertIn("k2horizon/effort-low", variants)
        low = variants["k2horizon/effort-low"]
        base = CATALOG["k2horizon"]
        for field in ("bos", "user_prefix", "assistant_prefix", "turn_end"):
            self.assertEqual(getattr(low, field), getattr(base, field))
        self.assertEqual(low.open_suffix, _K2H_EFFORT_LOW_SUFFIX)
        rendered = low.render(_MESSAGES_2)
        self.assertTrue(rendered.endswith(_K2H_EFFORT_LOW_SUFFIX))
        self.assertEqual(
            rendered[: -len(_K2H_EFFORT_LOW_SUFFIX)],
            _K2H_EXPECTED_2[: -len(_K2H_OPEN_SUFFIX)],
        )

    def test_k2horizon_effort_dedup_no_collision(self) -> None:
        # The effort open_suffixes must be genuinely new -- i.e. their
        # local_repairs dedup keys must differ from every key already
        # present, or candidates() would silently drop them. This mirrors
        # the "cannot collide with local_repairs' dedup keys" note in
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

        base = CATALOG["k2horizon"]
        repair_keys = {_key(variant) for variant in local_repairs(base)}
        variants = {variant.name: variant for variant in candidates("k2horizon", mutate=True)}
        for name in ("k2horizon/effort-medium", "k2horizon/effort-low"):
            self.assertNotIn(_key(variants[name]), repair_keys)


# --- Diff-vs-all-other-families -------------------------------------------------


class K2HorizonDiffTests(unittest.TestCase):
    def test_k2horizon_differs_from_every_other_family(self) -> None:
        # k2horizon's <|ifm|...> frames share no opener/terminator literals
        # with any other catalog family; on the same messages the full
        # render must be byte-different from all 23 other families'
        # renders.
        mine = CATALOG["k2horizon"].render(_MESSAGES_2)
        others = [name for name in CATALOG if name != "k2horizon"]
        self.assertEqual(len(others), 23)
        collisions = [name for name in others if CATALOG[name].render(_MESSAGES_2) == mine]
        self.assertEqual(collisions, [])


# --- README naming pin ----------------------------------------------------------


class ReadmeK2HorizonTests(unittest.TestCase):
    def test_readme_names_k2horizon(self) -> None:
        # The README's family list must name the new family, and its
        # "N-family catalog" claims must track len(CATALOG) (the absolute
        # count pins live in tests/test_cycle10_families.py,
        # tests/test_cycle11_families.py and tests/test_cycle13_families.py).
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("K2-Horizon", text)
        matches = re.findall(r"(\d+)-family catalog", text)
        self.assertTrue(matches, "README lost its N-family catalog claim")
        for match in matches:
            self.assertEqual(int(match), len(CATALOG))


if __name__ == "__main__":
    unittest.main()
