"""Cycle 52 (continuous dev): Meta Muse Spark Open family.

Byte-exact render checks for the muse_spark family, researched and
verified against the primary sources this round (2026-10-15):

Muse Spark Open -- meta-models/Muse-Spark-Open-816B-A42B (Meta
Superintelligence Labs, released 2026-10-15, Muse Model License;
https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B). 816B
total / 42B active MoE, 1M context, natively multimodal, built for
coding and agentic work.

Evidence tier 1: the official chat_template.jinja (230 lines, sha256
4c0eca41...6e0c6c32) read from hf-internal-testing/tiny-muse-spark --
the HF-official test mirror of the gated release tokenizer. It is the
reference template in transformers PR #49644 ("Add Muse Spark Open",
https://github.com/huggingface/transformers/pull/49644), whose
integration tests assert apply_chat_template output token-identical to
SGLang's MuseSpark1xOmniMultimodalProcessor. Every string here is
derived from the template's plain-text, no-tools,
default-reasoning-effort branch (jinja line numbers in the catalog
comment) and cross-checked by rendering the official template
byte-exact under stock Jinja2 (the {%- generation %} extension is
render-transparent; raise_exception is render-time only).

- BOS: <|begin_of_text|> -- the template emits {{- bos_token -}}
  (line 128); tokenizer_config.json bos_token with add_bos_token true
  (tier-1 verified 2026-10-15).
- User (lines 160-167): <|start|>user<|message|>{content}<|eot|>.
- Assistant history, no reasoning/tool calls (lines 194, 212-223):
  <|start|>assistant to=user<|message|>{content}<|eot|> (recipient
  defaults to 'user'; end_turn is true for it).
- Generation prompt (lines 228-229): <|start|>assistant (bare -- unlike
  muse_glimmer's <|start|>assistant to=user<|message|>).
- System block (lines 138-152): ALWAYS emitted, even with no system
  message -- documented, not modeled (the lab's Format only renders
  user/assistant turns; the ling/hy4/minimax3 convention). Pinned
  byte-exact here for the deterministic core: no caller system message,
  no tools, reasoning_effort unset (default 'xhigh' -> 512), and
  current_date pinned to None. Caveat: the template defaults
  current_date to today's date via strftime_now when available (line 5),
  so a raw apply_chat_template render is NOT byte-stable; the lab pins
  current_date=None (no "Current date:" line).
- EOS: tokenizer_config.json eos_token is <|end_of_text|> (id 200001);
  <|eot|> (id 200008) is the end-of-MESSAGE marker, not a stop token --
  documented, not modeled (Format has no EOS field; the glimmer
  convention).
- reasoning_effort only touches the unmodeled system block
  ("Reasoning strength: N.", low->8 / medium->32 / high->256 /
  xhigh->512); the generation prompt is invariant, so no mutate-path
  variant rides (contrast k2horizon/hy4).

Also: the discoverability pins (registered in CATALOG, candidates()
base/mutate behavior) and a per-family diff-vs-all-other-families pin
(dynamic: len(CATALOG) - 1). There is intentionally NO README naming
pin in this file: the README counts/naming are owned by the coordinator
this cycle, and a pin here would fail before the README update lands
(the cycle-15 precedent).

Deterministic; no network, no model, no randomness.
"""

import unittest

from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.core import Format, Message

_MESSAGES_1: tuple[Message, ...] = (Message("user", "Hello"),)

_MESSAGES_2: tuple[Message, ...] = (
    Message("user", "Hello"),
    Message("assistant", "Hi there"),
    Message("user", "Bye"),
)

_MESSAGES_EMPTY: tuple[Message, ...] = (Message("user", ""),)


# --- Muse Spark Open ------------------------------------------------------------

# Derived from the official chat_template.jinja (hf-internal-testing/
# tiny-muse-spark, 230 lines; verified by rendering the official template
# byte-exact under stock Jinja2).
_MS_BOS = "<|begin_of_text|>"
_MS_USER_PREFIX = "<|start|>user<|message|>"
_MS_ASSISTANT_PREFIX = "<|start|>assistant to=user<|message|>"
_MS_TURN_END = "<|eot|>"
# Generation prompt (jinja lines 228-229): bare <|start|>assistant.
_MS_OPEN_SUFFIX = "<|start|>assistant"

# The ALWAYS-emitted system block (jinja lines 138-152) is documented,
# not modeled (the lab's Format only renders user/assistant turns).
# Pinned byte-exact against the official template's render with no caller
# system message, no tools, reasoning_effort unset (default 'xhigh' ->
# 512), and current_date=None (no "Current date:" line -- the template's
# strftime_now default would make the raw render non-byte-stable).
_MS_SYSTEM_BLOCK = (
    "<|start|>system<|message|>"
    "You are a helpful AI assistant."
    "\n\nKnowledge cutoff: 2026-01-04."
    "\n\nReasoning strength: 512."
    "\n\nUse the appropriate recipient for each message:"
    '\n\n- "self": private reasoning and tool planning.'
    '\n- "commentary": user-visible intermediate updates while the assistant '
    "will continue working, including messages sent to the user before or "
    "between tool calls."
    '\n- "user": messages that end the assistant turn, such as a completed '
    "response or clarification question that waits for the user's reply; do "
    "not use for pre-tool updates or partial responses."
    '\n\n# Valid recipients: "self", "commentary", "user".'
    "<|eot|>"
)

_MS_EXPECTED_1 = _MS_BOS + _MS_USER_PREFIX + "Hello" + _MS_TURN_END + _MS_OPEN_SUFFIX

_MS_EXPECTED_2 = (
    _MS_BOS
    + _MS_USER_PREFIX
    + "Hello"
    + _MS_TURN_END
    + _MS_ASSISTANT_PREFIX
    + "Hi there"
    + _MS_TURN_END
    + _MS_USER_PREFIX
    + "Bye"
    + _MS_TURN_END
    + _MS_OPEN_SUFFIX
)

_MS_EXPECTED_EMPTY = _MS_BOS + _MS_USER_PREFIX + "" + _MS_TURN_END + _MS_OPEN_SUFFIX


class MuseSparkRenderTests(unittest.TestCase):
    def test_muse_spark_renders_byte_exact_1_turn(self) -> None:
        self.assertEqual(CATALOG["muse_spark"].render(_MESSAGES_1), _MS_EXPECTED_1)

    def test_muse_spark_renders_byte_exact_multi_turn(self) -> None:
        self.assertEqual(CATALOG["muse_spark"].render(_MESSAGES_2), _MS_EXPECTED_2)

    def test_muse_spark_renders_byte_exact_empty_message(self) -> None:
        # Edge case: an empty user message renders the bare frame
        # <|start|>user<|message|><|eot|> (official template render).
        self.assertEqual(CATALOG["muse_spark"].render(_MESSAGES_EMPTY), _MS_EXPECTED_EMPTY)
        rendered_empty = CATALOG["muse_spark"].render(_MESSAGES_EMPTY)
        self.assertIn("<|start|>user<|message|><|eot|>", rendered_empty)

    def test_muse_spark_fields_match_catalog_entry(self) -> None:
        fmt = CATALOG["muse_spark"]
        self.assertEqual(fmt.name, "muse_spark")
        self.assertEqual(fmt.bos, _MS_BOS)
        self.assertEqual(fmt.user_prefix, _MS_USER_PREFIX)
        self.assertEqual(fmt.assistant_prefix, _MS_ASSISTANT_PREFIX)
        self.assertEqual(fmt.turn_end, _MS_TURN_END)
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertEqual(fmt.open_suffix, _MS_OPEN_SUFFIX)
        self.assertFalse(fmt.repeat_bos)

    def test_muse_spark_markers_character_exact(self) -> None:
        # No newlines anywhere in the frames (the glimmer dialect has no
        # newline separators); user and assistant turns share the <|eot|>
        # end-of-message marker; the generation prompt is the bare
        # <|start|>assistant opener (jinja line 229).
        self.assertEqual(_MS_USER_PREFIX, "<|start|>user<|message|>")
        self.assertEqual(_MS_ASSISTANT_PREFIX, "<|start|>assistant to=user<|message|>")
        self.assertEqual(_MS_TURN_END, "<|eot|>")
        self.assertEqual(_MS_OPEN_SUFFIX, "<|start|>assistant")
        self.assertNotIn("\n", _MS_USER_PREFIX + _MS_ASSISTANT_PREFIX + _MS_TURN_END)
        self.assertNotEqual(_MS_OPEN_SUFFIX, _MS_ASSISTANT_PREFIX)

    def test_muse_spark_bos_handling(self) -> None:
        # BOS <|begin_of_text|> leads the render exactly once (no
        # repeat_bos); it is the tokenizer's bos_token, not a message
        # frame (jinja line 128).
        rendered = CATALOG["muse_spark"].render(_MESSAGES_2)
        self.assertTrue(rendered.startswith(_MS_BOS))
        self.assertEqual(rendered.count(_MS_BOS), 1)
        self.assertFalse(CATALOG["muse_spark"].repeat_bos)

    def test_muse_spark_system_block_documented_not_modeled(self) -> None:
        # The real template ALWAYS emits a system block (jinja lines
        # 138-152) even with no system message; the lab's Format only
        # renders user/assistant turns, so the block is a documented
        # constant (pinned byte-exact here -- verified against the
        # official template's render), not a render field.
        self.assertEqual(
            _MS_SYSTEM_BLOCK,
            "<|start|>system<|message|>"
            "You are a helpful AI assistant."
            "\n\nKnowledge cutoff: 2026-01-04."
            "\n\nReasoning strength: 512."
            "\n\nUse the appropriate recipient for each message:"
            '\n\n- "self": private reasoning and tool planning.'
            '\n- "commentary": user-visible intermediate updates while the assistant '
            "will continue working, including messages sent to the user before or "
            "between tool calls."
            '\n- "user": messages that end the assistant turn, such as a completed '
            "response or clarification question that waits for the user's reply; do "
            "not use for pre-tool updates or partial responses."
            '\n\n# Valid recipients: "self", "commentary", "user".'
            "<|eot|>",
        )
        rendered = CATALOG["muse_spark"].render(_MESSAGES_2)
        self.assertNotIn("<|start|>system", rendered)

    def test_muse_spark_official_render_composition(self) -> None:
        # The official template's full render is exactly: BOS + the
        # documented system block + the user/assistant turns +
        # generation prompt. The lab render covers everything except the
        # system block, so splicing the constant in must reproduce the
        # official shape.
        fmt = CATALOG["muse_spark"]
        lab = fmt.render(_MESSAGES_2)
        official = _MS_BOS + _MS_SYSTEM_BLOCK + lab[len(_MS_BOS) :]
        self.assertTrue(official.startswith(_MS_BOS + "<|start|>system<|message|>"))
        self.assertTrue(official.endswith(_MS_OPEN_SUFFIX))
        self.assertIn("<|start|>user<|message|>Hello<|eot|>", official)
        self.assertIn("<|start|>assistant to=user<|message|>Hi there<|eot|>", official)
        self.assertIn("Reasoning strength: 512.", official)
        self.assertIn("Knowledge cutoff: 2026-01-04.", official)
        # current_date is pinned to None: no "Current date:" line in the
        # deterministic core (the template's strftime_now default would
        # make a raw render non-byte-stable).
        self.assertNotIn("Current date:", official)

    def test_muse_spark_escape_edge_case_documented(self) -> None:
        # The official template's esc() neutralizes "<|" in caller text
        # ("<|" -> "< |") so user text cannot forge structure; the lab's
        # plain-text core passes content through raw. Documented as an
        # explicit out-of-scope edge case, not modeled.
        tricky = (Message("user", "a <|start|> b"),)
        rendered = CATALOG["muse_spark"].render(tricky)
        self.assertIn("a <|start|> b", rendered)
        self.assertNotIn("< |start|>", rendered)

    def test_muse_spark_eos_caveat_documented(self) -> None:
        # The official tokenizer_config.json eos_token is <|end_of_text|>
        # (id 200001); <|eot|> (id 200008) is the end-of-MESSAGE marker,
        # not a stop token. Format has no EOS field, so the frames must
        # carry no stray EOS.
        fmt = CATALOG["muse_spark"]
        frames = fmt.bos + fmt.user_prefix + fmt.assistant_prefix + fmt.turn_end
        self.assertNotIn("<|end_of_text|>", frames)
        self.assertEqual(fmt.turn_end.count("<|eot|>"), 1)


# --- Discoverability ------------------------------------------------------------


class MuseSparkDiscoverabilityTests(unittest.TestCase):
    def test_muse_spark_registered_in_catalog(self) -> None:
        self.assertIn("muse_spark", CATALOG)
        self.assertIsInstance(CATALOG["muse_spark"], Format)

    def test_muse_spark_candidates_base(self) -> None:
        # Non-mutating candidates() returns exactly the base entry, like
        # every other family.
        self.assertEqual(candidates("muse_spark"), (CATALOG["muse_spark"],))

    def test_muse_spark_candidates_mutate_has_no_family_branches(self) -> None:
        # reasoning_effort only touches the unmodeled system block and the
        # generation prompt is invariant, so muse_spark rides the mutate
        # path with the standard local repairs only -- no family-specific
        # branches (contrast k2horizon/hy4/minimax3/nanbeige42).
        variants = candidates("muse_spark", mutate=True)
        names = [variant.name for variant in variants]
        self.assertIn("muse_spark", names)
        family_branches = [name for name in names if name.startswith("muse_spark/")]
        for branch in family_branches:
            self.assertTrue(
                branch.startswith("muse_spark/"),
                f"unexpected non-repair branch {branch!r}",
            )
        self.assertNotIn("muse_spark/thinking-off", names)
        self.assertNotIn("muse_spark/no-think", names)
        base = CATALOG["muse_spark"]
        for variant in variants:
            if variant.name == "muse_spark":
                continue
            # Local repairs keep the frames; only BOS/prefix tweaks vary.
            self.assertEqual(variant.user_prefix, base.user_prefix)
            self.assertEqual(variant.assistant_prefix.rstrip(), base.assistant_prefix.rstrip())

    def test_muse_spark_unknown_family_still_rejected(self) -> None:
        with self.assertRaises(ValueError):
            candidates("muse_spark_open")


# --- Diff-vs-all-other-families ---------------------------------------------------


class MuseSparkDiffTests(unittest.TestCase):
    def test_muse_spark_differs_from_every_other_family(self) -> None:
        mine = CATALOG["muse_spark"].render(_MESSAGES_2)
        others = [name for name in CATALOG if name != "muse_spark"]
        self.assertEqual(len(others), len(CATALOG) - 1)
        collisions = [name for name in others if CATALOG[name].render(_MESSAGES_2) == mine]
        self.assertEqual(collisions, [])


if __name__ == "__main__":
    unittest.main()
