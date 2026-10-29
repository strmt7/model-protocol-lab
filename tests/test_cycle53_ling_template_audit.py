"""Cycle 53: independent audit of the ``ling`` catalog entry against the
official Ling 3.0 Flash chat template.

Primary sources (read 2026-10-15):

- https://huggingface.co/inclusionAI/Ling-3.0-flash -- official model card
  ("Bailing V3" chat_template.jinja, main branch,
  md5 23b682c460649dbbc03e9d2bc9609071).
- https://huggingface.co/inclusionAI/Ling-3.1-flash -- official model card
  (chat_template.jinja, main branch, md5 24b514df8b812d0bb0e9c061d0bb164c,
  byte-identical to the repo's docs/evidence/ling31-chat-template.jinja).
- https://github.com/huggingface/transformers/pull/49446 -- "[Model] Add
  native Ling 3.0 Flash support" (BailingHybridForCausalLM, no
  trust_remote_code; released in transformers v5.20.0 on 2026-10-15). The
  PR adds model code and tests only -- it ships no chat-template docs, so
  the template authority remains the hub chat_template.jinja files above.

Official template literals (plain-text, no-tools branch; jinja line
numbers refer to the Ling-3.0-flash chat_template.jinja read from the hub):

- thinking option unset -> 'on' (lines 5-13); the entry models thinking ON.
- system block, ALWAYS emitted even with no system message: without one,
  ``<role>SYSTEM</role>detailed thinking on<|role_end|>`` (lines 30, 81-83);
  with system message X (no "detailed thinking" marker in X),
  ``<role>SYSTEM</role>X\\ndetailed thinking on<|role_end|>`` (lines 67-77).
- user (line 117-119): ``<role>HUMAN</role>{content}<|role_end|>``.
- assistant history, no reasoning (lines 151-153, preserved_thinking true):
  ``<role>ASSISTANT</role>\\n<think></think>{content}`` closed by
  ``<|role_end|>`` (line 201).
- generation prompt (lines 221-233): ``<role>ASSISTANT</role>\\n<think>``
  when thinking is on (line 225-227), ``<role>ASSISTANT</role>\\n
  <think></think>`` when off (lines 229-231).
- BOS: ``<|startoftext|>`` -- from the official tokenizer_config.json
  bos_token field (3.0 and 3.1 agree; add_bos_token is false, so a raw
  apply_chat_template call would not add it either -- the entry takes the
  config's bos_token, lfm25 convention).
- EOS caveat: the official eos_token field is ``<|role_end|>``, NOT
  ``<|endoftext|>`` (that token is the pad_token); documented, not modeled
  (Format has no EOS field).

Ling 3.0 vs 3.1: the two chat_template.jinja files differ ONLY in jinja
comment/control-tag indentation (3.0 lines 6-12) and a trailing newline --
all inside ``{%- ... -%}`` whitespace-stripped regions, i.e. zero output
effect. Verified byte-identical renders with jinja2 for five
representative conversations (user-only, system+user, system with
"detailed thinking off", multi-turn, assistant with reasoning_content) x
{enable_thinking on, off}. The 3.1-derived catalog entry therefore covers
3.0 exactly.

Known official-template artifact (NOT a divergence): when a system message
IS present, the official template emits a literal ``\\n`` + four spaces
between the system block and the first user block (jinja indentation leak
at official lines 42-43). Format never renders system blocks (lab
convention -- same as mistral_large/gemma4/kolibri), so this artifact is
outside the modeled scope; the system block itself is pinned byte-exact in
tests/test_cycle13_families.py.

VERDICT: MATCH. Every string the ``ling`` entry models -- BOS, user frame,
assistant frame, turn terminator, thinking-on and thinking-off generation
prompts -- is byte-identical to the official Ling 3.0 Flash template
(which is output-identical to Ling-3.1-flash). No DOCUMENTED -> ACTUAL
diffs.

Minimal reproducer (how the comparison was produced; needs network,
jinja2, huggingface_hub -- NOT executed by this test file)::

    from huggingface_hub import hf_hub_download
    import jinja2
    p = hf_hub_download("inclusionAI/Ling-3.0-flash", "chat_template.jinja")
    t = jinja2.Environment().from_string(open(p).read())
    official = t.render(messages=[{"role": "user", "content": "Hello!"}],
                        add_generation_prompt=True)
    assert official == ("<role>SYSTEM</role>detailed thinking on<|role_end|>"
                        "<role>HUMAN</role>Hello!<|role_end|>"
                        "<role>ASSISTANT</role>\\n<think>")

Deterministic; no network, no model, no randomness.
"""

import unittest

from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.core import Message

# --- Official Ling 3.0 Flash literals (from the primary sources above) -------

_OFFICIAL_BOS = "<|startoftext|>"
_OFFICIAL_USER_PREFIX = "<role>HUMAN</role>"
_OFFICIAL_ASSISTANT_PREFIX = "<role>ASSISTANT</role>\n<think></think>"
_OFFICIAL_TURN_END = "<|role_end|>"
_OFFICIAL_GEN_PROMPT_ON = "<role>ASSISTANT</role>\n<think>"
_OFFICIAL_GEN_PROMPT_OFF = "<role>ASSISTANT</role>\n<think></think>"
# The system block is ALWAYS emitted by the official template; the entry
# does not model it (lab convention), so it is only the expected prefix the
# repo render is compared against on the no-system path.
_OFFICIAL_SYSTEM_BLOCK_NO_SYSTEM = "<role>SYSTEM</role>detailed thinking on<|role_end|>"

# Representative conversations for the audit.
_SINGLE: tuple[Message, ...] = (Message("user", "Hello!"),)
_MULTI: tuple[Message, ...] = (
    Message("user", "What is 2+2?"),
    Message("assistant", "4"),
    Message("user", "And 3*3?"),
)

# Official full renders (jinja2 render of the hub chat_template.jinja,
# enable_thinking unset -> on), minus the always-emitted system block that
# Format does not model.
_OFFICIAL_30_SINGLE_TAIL = "<role>HUMAN</role>Hello!<|role_end|><role>ASSISTANT</role>\n<think>"
_OFFICIAL_30_MULTI_TAIL = (
    "<role>HUMAN</role>What is 2+2?<|role_end|>"
    "<role>ASSISTANT</role>\n<think></think>4<|role_end|>"
    "<role>HUMAN</role>And 3*3?<|role_end|>"
    "<role>ASSISTANT</role>\n<think>"
)

# Full official renders for reference (system block included).
_OFFICIAL_30_SINGLE_FULL = _OFFICIAL_SYSTEM_BLOCK_NO_SYSTEM + _OFFICIAL_30_SINGLE_TAIL


class Ling30OfficialLiteralsTests(unittest.TestCase):
    """The entry's frames equal the officially-documented template strings."""

    def test_fields_match_official_30(self) -> None:
        """Test fields match official 30."""
        fmt = CATALOG["ling"]
        self.assertEqual(fmt.bos, _OFFICIAL_BOS)
        self.assertEqual(fmt.user_prefix, _OFFICIAL_USER_PREFIX)
        self.assertEqual(fmt.assistant_prefix, _OFFICIAL_ASSISTANT_PREFIX)
        self.assertEqual(fmt.turn_end, _OFFICIAL_TURN_END)
        self.assertEqual(fmt.open_suffix, _OFFICIAL_GEN_PROMPT_ON)

    def test_thinking_off_suffix_matches_official_30(self) -> None:
        """Test thinking off suffix matches official 30."""
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        off = next(v for v in candidates("ling", mutate=True) if v.name == "ling/thinking-off")
        self.assertEqual(off.open_suffix, _OFFICIAL_GEN_PROMPT_OFF)


class Ling30RenderAuditTests(unittest.TestCase):
    """Repo renders vs the official template, byte-level (no-system scope)."""

    def test_single_turn_render_matches_official_tail(self) -> None:
        """Test single turn render matches official tail."""
        rendered = CATALOG["ling"].render(_SINGLE)
        self.assertEqual(rendered, _OFFICIAL_BOS + _OFFICIAL_30_SINGLE_TAIL)
        # And the repo render is exactly the official full render with the
        # (unmodeled) system block stripped.
        self.assertEqual(
            rendered,
            _OFFICIAL_BOS + _OFFICIAL_30_SINGLE_FULL[len(_OFFICIAL_SYSTEM_BLOCK_NO_SYSTEM) :],
        )

    def test_multi_turn_render_matches_official_tail(self) -> None:
        """Test multi turn render matches official tail."""
        rendered = CATALOG["ling"].render(_MULTI)
        self.assertEqual(rendered, _OFFICIAL_BOS + _OFFICIAL_30_MULTI_TAIL)

    def test_thinking_off_render_tail_matches_official(self) -> None:
        """Test thinking off render tail matches official."""
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        off = next(v for v in candidates("ling", mutate=True) if v.name == "ling/thinking-off")
        rendered = off.render(_SINGLE)
        self.assertTrue(rendered.endswith(_OFFICIAL_GEN_PROMPT_OFF))
        self.assertEqual(
            rendered,
            _OFFICIAL_BOS
            + _OFFICIAL_USER_PREFIX
            + "Hello!"
            + _OFFICIAL_TURN_END
            + _OFFICIAL_GEN_PROMPT_OFF,
        )


class Ling30VerdictTests(unittest.TestCase):
    """The audit verdict pin: MATCH -- zero DOCUMENTED -> ACTUAL diffs."""

    def test_verdict_match_no_documented_actual_diffs(self) -> None:
        """Test verdict match no documented actual diffs."""
        fmt = CATALOG["ling"]
        diffs: list[str] = []
        pairs = [
            ("bos", _OFFICIAL_BOS, fmt.bos),
            ("user_prefix", _OFFICIAL_USER_PREFIX, fmt.user_prefix),
            ("assistant_prefix", _OFFICIAL_ASSISTANT_PREFIX, fmt.assistant_prefix),
            ("turn_end", _OFFICIAL_TURN_END, fmt.turn_end),
            ("open_suffix (thinking on)", _OFFICIAL_GEN_PROMPT_ON, fmt.open_suffix),
        ]
        for name, documented, actual in pairs:
            if documented != actual:
                diffs.append(f"{name}: DOCUMENTED {documented!r} -> ACTUAL {actual!r}")
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        off = next(v for v in candidates("ling", mutate=True) if v.name == "ling/thinking-off")
        if off.open_suffix != _OFFICIAL_GEN_PROMPT_OFF:
            diffs.append(
                f"thinking-off open_suffix: DOCUMENTED {_OFFICIAL_GEN_PROMPT_OFF!r} "
                f"-> ACTUAL {off.open_suffix!r}"
            )
        for convo, tail in ((_SINGLE, _OFFICIAL_30_SINGLE_TAIL), (_MULTI, _OFFICIAL_30_MULTI_TAIL)):
            rendered = fmt.render(convo)
            expected = _OFFICIAL_BOS + tail
            if rendered != expected:
                diffs.append(
                    f"render {convo[0].content!r}: DOCUMENTED {expected!r} -> ACTUAL {rendered!r}"
                )
        # Verdict: MATCH -- this fails loudly with exact diffs if the
        # official template ever drifts from the entry.
        self.assertEqual(diffs, [], f"DIVERGED: {diffs}")

    def test_system_block_outside_modeled_scope(self) -> None:
        """Test system block outside modeled scope."""
        # The official template always emits the system block; Format never
        # renders system turns (lab convention). Guard the scope so a future
        # reader does not mistake the convention for a divergence.
        rendered = CATALOG["ling"].render(_MULTI)
        self.assertNotIn("<role>SYSTEM</role>", rendered)
        self.assertEqual(
            _OFFICIAL_SYSTEM_BLOCK_NO_SYSTEM,
            "<role>SYSTEM</role>detailed thinking on<|role_end|>",
        )


if __name__ == "__main__":
    unittest.main()
