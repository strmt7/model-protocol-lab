"""Cycle 74: adversarial render-correctness audit of the 29-family catalog.

Mission: verify the hand-rendered ``Format`` hypotheses in
``model_protocol_lab/catalog.py`` themselves -- no dedicated adversarial
audit had ever checked the renders, only their downstream use.

Three prongs:

1. STRUCTURAL INVARIANTS (all 29 families x all ``candidates(mutate=True)``
   variants = 118 renders): determinism, non-empty output, distinct
   user/assistant markers, no ``None``/placeholder leakage, exact render
   mechanics (BOS head, per-role terminators, open-suffix tail), and correct
   multi-turn role alternation on a 3-turn conversation. System-role
   handling follows the lab convention: ``Message`` only admits
   user/assistant, so a system turn cannot even be constructed.

2. PRIMARY-SOURCE CROSS-CHECK (6 families, fetched 2026-10-16 via
   read-only text fetch; each test pins the exact source URL and the
   template lines the expected strings were transcribed from):
   - llama3: https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/prompt_format.md
     (the original models/llama3/prompt_format.md was removed in a repo
     restructure; the 3.3 doc carries the unchanged no-tools plain-text
     core; Input Prompt Format lines 78-84).
   - gemma: https://ai.google.dev/gemma/docs/formatting (dialogue example
     lines 30-44; turn-separator/prompt-prefix statement line 48).
   - qwen3: https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json
     (chat_template field, lines 118-139: per-message frames, BOS null,
     generation prompt).
   - deepseek: https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/tokenizer_config.json
     (chat_template field, lines 6-20: raw-concatenated system, unterminated
     user turns, assistant terminator, generation prompt; bos/frames use
     U+FF5C and U+2581 as the catalog claims).
   - mistral (v0.3): version-pin coherence. The catalog pins the RELEASE
     template (2024-05-22, ``<s>[INST] {u} [/INST]{a}</s>``), not the
     current Hub file. The current file
     (https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/raw/main/tokenizer_config.json,
     chat_template lines 701-716, fetched 2026-10-16) exhibits exactly the
     rewrite the catalog comment describes (no space before ``[/INST]``,
     leading-space assistant turn, no generation-prompt block) --
     corroborating the pin. The test pins both the release render and the
     current-Hub transcription and their exact delta.
   - hy4: https://huggingface.co/tencent/Hy4-preview/raw/main/chat_template.jinja
     (live file, 568 lines: user branch lines 418-424, assistant empty-think
     branch lines 446-456, generation prompt lines 552-562, 556-562;
     reasoning_effort default 'high' lines 58-60) AND the locally preserved
     byte-identical template (docs/evidence/hy4-chat-template.jinja, 309
     lines, md5 a0b7e28386487d804009340155e7be2b, special-token variables
     lines 2-21), which was additionally executed under stock Jinja2 during
     this audit: template output minus the documented synthesized system
     block is byte-identical to the catalog render (1-turn and 2-turn), and
     the ``hy4/no-think`` mutate variant matches the ``reasoning_effort=
     'no_think'`` generation prompt.

   RESULT: CLEAN -- all six cross-checks match. No catalog fix required.

3. ADVERSARIAL EDGES: hostile ``candidates()`` lookups, empty/degenerate
   conversations, marker-looking and brace-heavy content passing through
   verbatim, unicode salad preservation across all families, and a
   501-turn conversation.

Mutation gate: 6 mutants planted in /tmp copies of the package (never the
repo tree); every one is caught by at least one test in this file:
  1. llama3 BOS blanked -> structural BOS-head invariant + llama3
     cross-check.
  2. gemma user/assistant prefixes swapped -> gemma cross-check.
  3. mistral user_turn_end " [/INST]" -> "[/INST]" (space dropped) ->
     mistral release-pin cross-check.
  4. deepseek U+2581 block replaced with "_" -> deepseek cross-check.
  5. hy4 open_suffix with the think-begin token removed -> hy4 cross-check.
  6. qwen38/thinking-off branch removed from candidates() -> variant
     inventory pin.
A clean (unmutated) copy passes the whole file: the failures are caused by
the mutants, not the harness.
"""

from __future__ import annotations

import time
import unittest
from typing import Literal

from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.core import Format, Message

Role = Literal["user", "assistant"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _msgs(*role_content: tuple[Role, str]) -> tuple[Message, ...]:
    """Msgs."""
    return tuple(Message(role=role, content=content) for role, content in role_content)


def _one_turn(content: str = "hello") -> tuple[Message, ...]:
    """One turn."""
    return _msgs(("user", content))


def _three_turn() -> tuple[Message, ...]:
    """Three turn."""
    return _msgs(("user", "U1"), ("assistant", "A1"), ("user", "U2"))


def _tail(fmt: Format) -> str:
    """Tail."""
    return fmt.open_suffix if fmt.open_suffix is not None else fmt.assistant_prefix


def _user_end(fmt: Format) -> str:
    """User end."""
    return fmt.user_turn_end if fmt.user_turn_end is not None else fmt.turn_end


def _expected_one_turn(fmt: Format, content: str) -> str:
    """The documented render contract for a single user turn, rebuilt here
    from the field values (pins the mechanics per variant).
    """
    out = fmt.bos + fmt.user_prefix + content + _user_end(fmt)
    if fmt.repeat_bos:
        out += fmt.bos
    return out + _tail(fmt)


_VARIANTS: list[tuple[str, Format]] = [
    (family, fmt) for family in CATALOG for fmt in candidates(family, mutate=True)
]
_VARIANT_IDS = [fmt.name for _, fmt in _VARIANTS]

# The 9 documented generation-prompt branch families and their mutate-path
# variant names (kolibri/thinking-off convention).
_BRANCH_VARIANTS: dict[str, tuple[str, ...]] = {
    "hy4": ("hy4/no-think",),
    "k2horizon": ("k2horizon/effort-medium", "k2horizon/effort-low"),
    "kolibri": ("kolibri/thinking-off",),
    "laguna": ("laguna/thinking-on",),
    "ling": ("ling/thinking-off",),
    "minimax3": ("minimax3/thinking-disabled", "minimax3/thinking-enabled"),
    "nanbeige42": ("nanbeige42/thinking-off",),
    "nemotron35": ("nemotron35/thinking-off",),
    "qwen38": ("qwen38/thinking-off",),
}


# ---------------------------------------------------------------------------
# 1. structural invariants: every family x every mutate variant
# ---------------------------------------------------------------------------


class TestStructuralInvariants(unittest.TestCase):
    """Test structural invariants."""

    def test_determinism(self):
        """Test determinism."""
        for family, fmt in _VARIANTS:
            with self.subTest(family=family, fmt=fmt):
                assert fmt.render(_one_turn()) == fmt.render(_one_turn())
                assert fmt.render(_three_turn()) == fmt.render(_three_turn())

    def test_nonempty(self):
        """Test nonempty."""
        for family, fmt in _VARIANTS:
            with self.subTest(family=family, fmt=fmt):
                assert fmt.render(_one_turn()) != ""

    def test_markers_distinct(self):
        """Test markers distinct."""
        for family, fmt in _VARIANTS:
            with self.subTest(family=family, fmt=fmt):
                assert fmt.user_prefix != fmt.assistant_prefix
                assert fmt.user_prefix != ""

    def test_no_none_or_placeholder_leakage(self):
        """Test no none or placeholder leakage."""
        for family, fmt in _VARIANTS:
            with self.subTest(family=family, fmt=fmt):
                out = fmt.render(_three_turn())
                assert "None" not in out
                assert "{{" not in out
                assert "}}" not in out

    def test_one_turn_mechanics(self):
        """Test one turn mechanics."""
        for family, fmt in _VARIANTS:
            with self.subTest(family=family, fmt=fmt):
                fmt_rendered = fmt.render(_one_turn("hello"))
                assert fmt_rendered == _expected_one_turn(fmt, "hello")

    def test_bos_head(self):
        """Test bos head."""
        for family, fmt in _VARIANTS:
            with self.subTest(family=family, fmt=fmt):
                if fmt.bos:
                    assert fmt.render(_three_turn()).startswith(fmt.bos)

    def test_open_suffix_tail(self):
        """Test open suffix tail."""
        for family, fmt in _VARIANTS:
            with self.subTest(family=family, fmt=fmt):
                assert fmt.render(_three_turn()).endswith(_tail(fmt))

    def test_three_turn_alternation(self):
        """Test three turn alternation."""
        for family, fmt in _VARIANTS:
            with self.subTest(family=family, fmt=fmt):
                out = fmt.render(_three_turn())
                i1, i2, i3 = out.index("U1"), out.index("A1"), out.index("U2")
                assert i1 < i2 < i3
                # each content is immediately preceded by its role's marker
                assert out[i1 - len(fmt.user_prefix) : i1] == fmt.user_prefix
                assert out[i3 - len(fmt.user_prefix) : i3] == fmt.user_prefix
                if fmt.assistant_prefix:
                    assert out[i2 - len(fmt.assistant_prefix) : i2] == fmt.assistant_prefix

    def test_unicode_content_preserved(self):
        """Test unicode content preserved."""
        for family, fmt in _VARIANTS:
            with self.subTest(family=family, fmt=fmt):
                salad = "h\u00e9llo \U0001f30d \u4f60\u597d \U0001d11e\u200b\uff71\uff72"
                out = fmt.render(_msgs(("user", salad)))
                assert salad in out


# ---------------------------------------------------------------------------
# variant inventory
# ---------------------------------------------------------------------------


class TestVariantInventory(unittest.TestCase):
    """Test variant inventory."""

    @staticmethod
    def test_total_variant_count():
        """Test total variant count."""
        assert len(_VARIANTS) == 118

    @staticmethod
    def test_branch_variant_names():
        """Test branch variant names."""
        for family, expected in _BRANCH_VARIANTS.items():
            names = [f.name for f in candidates(family, mutate=True)]
            for want in expected:
                assert want in names, f"{want} missing for {family}"

    @staticmethod
    def test_no_unexpected_branch_variants():
        """Test no unexpected branch variants."""
        for family in CATALOG:
            names = [f.name for f in candidates(family, mutate=True)]
            branch = [
                n
                for n in names
                if "/" in n and "toggle-bos" not in n and "no-bos" not in n and "prefix-" not in n
            ]
            assert tuple(branch) == _BRANCH_VARIANTS.get(family, ()), family

    @staticmethod
    def test_base_is_first_variant():
        """Test base is first variant."""
        for family in CATALOG:
            assert candidates(family, mutate=True)[0] is CATALOG[family]
            assert candidates(family) == (CATALOG[family],)

    @staticmethod
    def test_variant_names_scoped():
        """Test variant names scoped."""
        for family, fmt in _VARIANTS:
            assert fmt.name == family or fmt.name.startswith(family + "/")


# ---------------------------------------------------------------------------
# 2. primary-source cross-checks (transcribed 2026-10-16; see module docstring)
# ---------------------------------------------------------------------------

# --- llama3 ---
# Source: https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/prompt_format.md
# (models/llama3/prompt_format.md was removed in a repo restructure; the 3.3
# doc carries the unchanged no-tools plain-text core). Input Prompt Format,
# lines 78-84:
#   <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}
#   <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n  (gen prompt)
_LLAMA3_2TURN = (
    "<|begin_of_text|>"
    "<|start_header_id|>user<|end_header_id|>\n\nU1<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\nA1<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\nU2<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


class CatalogRenderAuditTests(unittest.TestCase):
    """unittest entry point; pytest runs TestCase classes natively."""

    @staticmethod
    def test_crosscheck_llama3():
        """Test crosscheck llama3."""
        assert CATALOG["llama3"].render(_three_turn()) == _LLAMA3_2TURN

    @staticmethod
    def test_crosscheck_gemma():
        """Test crosscheck gemma."""
        assert CATALOG["gemma"].render(_one_turn("U1")) == _GEMMA_1TURN

    @staticmethod
    def test_crosscheck_qwen3():
        """Test crosscheck qwen3."""
        assert CATALOG["qwen3"].render(_one_turn("U1")) == _QWEN3_1TURN
        assert CATALOG["qwen3"].render(_three_turn()) == _QWEN3_2TURN

    @staticmethod
    def test_crosscheck_deepseek():
        """Test crosscheck deepseek."""
        assert CATALOG["deepseek"].render(_three_turn()) == _DEEPSEEK_2TURN

    @staticmethod
    def test_crosscheck_mistral_release_pin():
        """Test crosscheck mistral release pin."""
        assert CATALOG["mistral"].render(_one_turn("U1")) == _MISTRAL_RELEASE_1TURN
        assert CATALOG["mistral"].render(_three_turn()) == _MISTRAL_RELEASE_2TURN

    @staticmethod
    def test_crosscheck_mistral_pin_delta_vs_current_hub():
        """Test crosscheck mistral pin delta vs current hub."""
        assert _MISTRAL_RELEASE_1TURN.replace(" ", "") == _MISTRAL_HUB_CURRENT_1TURN.replace(
            " ", ""
        )
        assert _MISTRAL_RELEASE_2TURN.replace(" ", "") == _MISTRAL_HUB_CURRENT_2TURN.replace(
            " ", ""
        )
        assert _MISTRAL_RELEASE_1TURN != _MISTRAL_HUB_CURRENT_1TURN
        assert "[INST] U1 [/INST]" in _MISTRAL_RELEASE_1TURN  # space kept (release)
        assert "[INST] U1[/INST]" in _MISTRAL_HUB_CURRENT_1TURN  # space gone (hub)
        assert "[/INST]A1</s>" in _MISTRAL_RELEASE_2TURN  # no leading space (release)
        assert "[/INST] A1</s>" in _MISTRAL_HUB_CURRENT_2TURN  # leading space (hub)

    @staticmethod
    def test_crosscheck_hy4():
        """Test crosscheck hy4."""
        assert CATALOG["hy4"].render(_one_turn("U1")) == _HY4_1TURN
        assert CATALOG["hy4"].render(_three_turn()) == _HY4_2TURN

    @staticmethod
    def test_crosscheck_hy4_no_think_variant():
        """Test crosscheck hy4 no think variant."""
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        variant = next(f for f in candidates("hy4", mutate=True) if f.name == "hy4/no-think")
        assert variant.render(_one_turn("U1")) == _HY4_USER + _HY4_NOTHINK_GEN


# --- gemma ---
# Source: https://ai.google.dev/gemma/docs/formatting -- dialogue example
# lines 30-44 (<start_of_turn>user\n{content}<end_of_turn>\n,
# <start_of_turn>model\n{content}<end_of_turn>\n) and line 48: 'The token
# "<end_of_turn>\n" is the turn separator, and the prompt prefix is
# "<start_of_turn>model\n"'. BOS <bos> is the config bos_token convention.
_GEMMA_1TURN = "<bos><start_of_turn>user\nU1<end_of_turn>\n<start_of_turn>model\n"


# --- qwen3 ---
# Source: https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json,
# chat_template lines 118-139. Per-message (no-tools plain text):
#   '<|im_start|>' + role + '\n' + content + '<|im_end|>\n'   (lines 126-132)
# generation prompt: '<|im_start|>assistant\n'                (lines 138-139)
# bos_token is null, add_bos_token false                      (lines 0, 116)
_QWEN3_1TURN = "<|im_start|>user\nU1<|im_end|>\n<|im_start|>assistant\n"
_QWEN3_2TURN = (
    "<|im_start|>user\nU1<|im_end|>\n"
    "<|im_start|>assistant\nA1<|im_end|>\n"
    "<|im_start|>user\nU2<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# --- deepseek ---
# Source: https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/tokenizer_config.json,
# chat_template lines 6-20 (plain-text, no-tools core):
#   {{bos_token}}{{ns.system_prompt}}            -- system concatenated raw
#   user:      '<｜User｜>' + content            -- NO closing token (line 10)
#   assistant: '<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>' (line 17)
#   generation prompt: '<｜Assistant｜>'         (line 20)
# bos_token content: '<｜begin▁of▁sentence｜>' (line 0), add_bos_token true.
# ｜ = U+FF5C, ▁ = U+2581, spelled as escapes like catalog.py.
_DS_BAR = "\uff5c"
_DS_BLK = "\u2581"
_DS_BOS = f"<{_DS_BAR}begin{_DS_BLK}of{_DS_BLK}sentence{_DS_BAR}>"
_DS_USER = f"<{_DS_BAR}User{_DS_BAR}>"
_DS_ASST = f"<{_DS_BAR}Assistant{_DS_BAR}>"
_DS_END = f"<{_DS_BAR}end{_DS_BLK}of{_DS_BLK}sentence{_DS_BAR}>"
_DEEPSEEK_2TURN = _DS_BOS + _DS_USER + "U1" + _DS_ASST + "A1" + _DS_END + _DS_USER + "U2" + _DS_ASST


# --- mistral (v0.3): version-pin coherence ---
# The catalog pins the RELEASE template (2024-05-22):
#   <s>[INST] {u} [/INST]{a}</s>   (space before [/INST], no leading space on
#   the assistant turn). The CURRENT Hub file --
# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/raw/main/tokenizer_config.json,
# chat_template lines 701-716, fetched 2026-10-16 -- is the rewritten
# post-release form the catalog comment describes:
#   user:      "[INST] " + content + "[/INST]"      (line 707, no space)
#   assistant: " " + content|trim + eos_token       (line 711, leading space)
#   no add_generation_prompt block (template ends at line 716).
_MISTRAL_RELEASE_1TURN = "<s>[INST] U1 [/INST]"
_MISTRAL_RELEASE_2TURN = "<s>[INST] U1 [/INST]A1</s>[INST] U2 [/INST]"
_MISTRAL_HUB_CURRENT_1TURN = "<s>[INST] U1[/INST]"
_MISTRAL_HUB_CURRENT_2TURN = "<s>[INST] U1[/INST] A1</s>[INST] U2[/INST]"


# --- hy4 ---
# Source: https://huggingface.co/tencent/Hy4-preview/raw/main/chat_template.jinja
# (live, 568 lines, fetched 2026-10-16): special-token variables lines 2-21
# (hy_start <｜hy_start:opensource｜>, hy_middle <｜hy_middle:opensource｜>,
# hy_end <｜hy_end:opensource｜>, think_begin <think:opensource>, think_end
# </think:opensource>, reasoning_mode <｜reasoning_mode:opensource｜>);
# reasoning_effort defaults 'high' (lines 58-60); user branch lines 418-424:
#   hy_start ~ 'user' ~ hy_middle ~ content ~ hy_end;
# assistant history, no reasoning: hy_start ~ 'assistant' ~ hy_middle ~
#   think_begin ~ think_end ~ content ~ hy_end (lines 446-456);
# generation prompt ('high'): hy_start ~ 'assistant' ~ hy_middle ~
#   think_begin (lines 556-562); 'no_think': ... ~ think_begin ~ think_end
#   (line 558). Token values independently confirmed from the preserved
# byte-identical template docs/evidence/hy4-chat-template.jinja (309 lines,
# md5 a0b7e28386487d804009340155e7be2b, lines 2-21), which was additionally
# executed under stock Jinja2 during this audit: template output minus the
# documented synthesized system block is byte-identical to the catalog
# render (1-turn and 2-turn), and the no-think generation prompt matches
# the hy4/no-think mutate variant.
_HY_BAR = "\uff5c"
_HS = f"<{_HY_BAR}hy_start:opensource{_HY_BAR}>"
_HM = f"<{_HY_BAR}hy_middle:opensource{_HY_BAR}>"
_HE = f"<{_HY_BAR}hy_end:opensource{_HY_BAR}>"
_HY4_USER = _HS + "user" + _HM + "U1" + _HE
_HY4_GEN = _HS + "assistant" + _HM + "<think:opensource>"
_HY4_1TURN = _HY4_USER + _HY4_GEN
_HY4_2TURN = (
    _HY4_USER
    + _HS
    + "assistant"
    + _HM
    + "<think:opensource></think:opensource>A1"
    + _HE
    + _HS
    + "user"
    + _HM
    + "U2"
    + _HE
    + _HY4_GEN
)
_HY4_NOTHINK_GEN = _HS + "assistant" + _HM + "<think:opensource></think:opensource>"


# ---------------------------------------------------------------------------
# 3. adversarial edges
# ---------------------------------------------------------------------------


class TestAdversarialEdges(unittest.TestCase):
    """Test adversarial edges."""

    def test_unknown_family_raises(self):
        """Test unknown family raises."""
        for bad in [
            "no-such-family",
            "",
            "CHATML",
            " chatml",
            "chatml ",
            None,
            123,
            ["chatml"],
            ("chatml",),
        ]:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "Unknown format family"):
                candidates(bad)  # type: ignore[arg-type]

    def test_empty_conversation_raises(self):
        """Test empty conversation raises."""
        with self.assertRaises(ValueError):
            CATALOG["chatml"].render(())

    def test_non_tuple_conversation_raises(self):
        """Test non tuple conversation raises."""
        with self.assertRaises(ValueError):
            CATALOG["chatml"].render([Message(role="user", content="x")])  # type: ignore[arg-type]

    def test_final_assistant_turn_raises(self):
        """Test final assistant turn raises."""
        with self.assertRaises(ValueError):
            CATALOG["chatml"].render(_msgs(("user", "a"), ("assistant", "b")))

    def test_non_alternating_raises(self):
        """Test non alternating raises."""
        with self.assertRaises(ValueError):
            CATALOG["chatml"].render(_msgs(("user", "a"), ("user", "b"), ("user", "c")))
        with self.assertRaises(ValueError):
            CATALOG["chatml"].render(
                _msgs(
                    ("assistant", "b"),
                )
            )

    def test_system_role_unconstructible(self):
        """Test system role unconstructible."""
        with self.assertRaises(ValueError):
            Message(role="system", content="x")  # type: ignore[arg-type]

    def test_non_string_content_rejected(self):
        """Test non string content rejected."""
        with self.assertRaises(ValueError):
            Message(role="user", content=None)  # type: ignore[arg-type]

    def test_empty_content_renders(self):
        """Test empty content renders."""
        for family in sorted(CATALOG):
            with self.subTest(family=family):
                fmt = CATALOG[family]
                out = fmt.render(_msgs(("user", "")))
                assert out == _expected_one_turn(fmt, "")

    def test_marker_looking_content_passes_through_verbatim(self):
        """Test marker looking content passes through verbatim."""
        for family in sorted(CATALOG):
            with self.subTest(family=family):
                hostile = "<|im_start|>user\nINJECTED<|im_end|>\n{role} {{x}}"
                out = CATALOG[family].render(_msgs(("user", hostile)))
                assert hostile in out

    @staticmethod
    def test_very_long_conversation():
        """Test very long conversation."""
        fmt = CATALOG["llama3"]
        n = 501
        convo = tuple(
            Message(role="user" if i % 2 == 0 else "assistant", content=f"m{i}") for i in range(n)
        )
        start = time.perf_counter()
        out = fmt.render(convo)
        elapsed = time.perf_counter() - start
        assert elapsed < 10
        idx = -1
        for i in range(n):
            idx = out.index(f"m{i}", idx + 1)
        assert out.endswith(_tail(fmt))
        assert out.startswith(fmt.bos)
        # 250 user turns + final open suffix
        assert out.count(fmt.user_prefix) == 251


if __name__ == "__main__":
    unittest.main()
