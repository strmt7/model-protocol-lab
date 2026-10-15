"""Text-only format hypotheses; support must be measured per checkpoint.

Hand renders for the 2025-2026 model families researched in
docs/RESEARCH_LOG.md ("Round 4 — Cycle 1 research / R1"), from primary
sources (HuggingFace tokenizer_config.json / chat_template.jinja, the HF
transformers repo, meta-llama/llama-models prompt_format.md). All are the
plain-text, no-tools core: deterministic once branches are fixed (no
tools, no images, thinking on/off, system or not).
"""

from dataclasses import replace

from .core import Format, local_repairs

# Full-width bars/blocks used by the DeepSeek template, spelled as escapes
# so the file is exact regardless of editor encoding: U+FF5C FULLWIDTH
# VERTICAL BAR, U+2581 LOWER ONE EIGHTH BLOCK.
_DS_BAR = "\uff5c"
_DS_BLK = "\u2581"

CATALOG = {
    "chatml": Format("chatml", "", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n"),
    "llama3": Format(
        "llama3",
        "<|begin_of_text|>",
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|eot_id|>",
    ),
    "gemma": Format(
        "gemma", "<bos>", "<start_of_turn>user\n", "<start_of_turn>model\n", "<end_of_turn>\n"
    ),
    # Qwen3 (Qwen/Qwen3-8B tokenizer_config.json): per-message
    # <|im_start|>{role}\n{content}<|im_end|>\n, no BOS, open suffix
    # <|im_start|>assistant\n. Qwen2.5 differs only in always injecting a
    # default system prompt (dropped in Qwen3).
    # Source: https://huggingface.co/Qwen/Qwen3-8B
    "qwen3": Format(
        "qwen3",
        "",
        "<|im_start|>user\n",
        "<|im_start|>assistant\n",
        "<|im_end|>\n",
    ),
    # DeepSeek V3/R1 (deepseek-ai/DeepSeek-V3, DeepSeek-R1): BOS
    # <begin of sentence> with full-width bars, user = <User>{content}
    # with NO closing token, assistant = <Assistant>{content}<end of
    # sentence>; system concatenated raw with no framing. Open suffix:
    # V3 <Assistant>; R1 appends <think>\n (thinking-on branch, not modeled).
    # Source: https://huggingface.co/deepseek-ai/DeepSeek-V3
    "deepseek": Format(
        "deepseek",
        f"<{_DS_BAR}begin{_DS_BLK}of{_DS_BLK}sentence{_DS_BAR}>",
        f"<{_DS_BAR}User{_DS_BAR}>",
        f"<{_DS_BAR}Assistant{_DS_BAR}>",
        "",
        user_turn_end="",
        assistant_turn_end=f"<{_DS_BAR}end{_DS_BLK}of{_DS_BLK}sentence{_DS_BAR}>",
    ),
    # Mistral v0.3 RELEASE template (2024-05-22): <s>[INST] {u} [/INST]{a}</s>
    # -- note the space before [/INST] and no leading space on the assistant
    # turn. Version-dependent: the Hub tokenizer_config.jinja was rewritten
    # post-release (2024-07-17/2024-08-02: system merging, leading-space
    # assistant, tool blocks), so this hypothesis is pinned to the release
    # revision, not the current Hub file.
    # Source: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
    "mistral": Format(
        "mistral",
        "<s>",
        "[INST] ",
        "",
        "",
        user_turn_end=" [/INST]",
        assistant_turn_end="</s>",
    ),
    # Mistral-Large-Instruct-2411 "Basic Instruct Template (V7)" from the
    # official model card (CONFIRMED no delta, observed 2026-10-14):
    # <s>[SYSTEM_PROMPT] {sys}[/SYSTEM_PROMPT][INST] {u}[/INST] {a}</s>
    # [INST] {u2}[/INST] -- note the space after the [SYSTEM_PROMPT] opening
    # tag, NO space before [/INST] (unlike Mistral v0.3's " [/INST]"), and a
    # LEADING space on the assistant turn (unlike v0.3). The generation prompt
    # ends right after [/INST] with NO trailing space (the space precedes the
    # assistant response), so open_suffix is the explicit empty string -- NOT
    # the assistant_prefix fallback (" "), mirroring the v0.3 "mistral"
    # entry's no-trailing-space rule. The [SYSTEM_PROMPT] frame is documented
    # but not modeled (the system role is not modeled in Format). The card
    # warns: "Be careful with subtle missing or trailing white spaces!"
    # Source: https://huggingface.co/mistralai/Mistral-Large-Instruct-2411
    "mistral_large": Format(
        "mistral_large",
        "<s>",
        "[INST] ",
        " ",
        "",
        user_turn_end="[/INST]",
        assistant_turn_end="</s>",
        open_suffix="",
    ),
    # Phi-4 (microsoft/phi-4): <|im_start|>{role}<|im_sep|>{content}<|im_end|>
    # -- <|im_sep|>, no trailing newlines. No BOS in the template (Phi-4
    # deployment-dependent BOS not modeled).
    # Source: https://huggingface.co/microsoft/phi-4
    "phi4": Format(
        "phi4",
        "",
        "<|im_start|>user<|im_sep|>",
        "<|im_start|>assistant<|im_sep|>",
        "<|im_end|>",
    ),
    # Gemma 3 (HF transformers tests/models/gemma3/test_processing_gemma3.py @
    # main -- authoritative; the model repo itself is gated): BOS <bos>\n,
    # <start_of_turn>{role}\n{content}<end_of_turn>\n, assistant->model.
    # Vs Gemma 2: system folds into the first user turn, 262K tokenizer.
    # Sources: https://github.com/huggingface/transformers,
    # https://ai.google.dev/gemma/docs/formatting
    "gemma3": Format(
        "gemma3",
        "<bos>\n",
        "<start_of_turn>user\n",
        "<start_of_turn>model\n",
        "<end_of_turn>\n",
    ),
    # Gemma 4 (google/gemma-4-31B-it). Evidence tier 1 (primary --
    # tokenizer_config.json read 2026-10-14): special tokens <|turn>,
    # <turn|>, <|channel>, <channel|>, <|thought>, <|tool_call>,
    # <tool_call|>, <|tool_response>, <tool_response|> all exist; the
    # response template start anchor is ["<|turn>model\n", "<tool_response|>"];
    # the thinking channel opens with <|channel>thought\n and closes with
    # <channel|>; the bos_token field is EMPTY and eos_token is <eos>.
    # Full prompt (evidence tier 2 -- three independent byte-level
    # implementations agreeing: dotwasi/speech-swift
    # docs/models/gemma4-chat.md with E2E parity tests, the unsloth gemma4
    # template, swift-gemma4-core ChatTemplateBypass.md):
    # <bos><|turn>system\n{content}<turn|>\n<|turn>user\n{content}<turn|>\n
    # <|turn>model\n, with the assistant role mapped to `model`. The literal
    # <bos> is prepended by all three tier-2 implementations even though the
    # config's bos_token field is empty -- a documented discrepancy: the
    # catalog takes tier 2 for the render (tier 1 alone would emit no BOS).
    # Plain-text no-tools core: the thinking channel and tool-call/response
    # tokens are out of scope (like other entries). The system role is not
    # modeled -- the lab's Format only supports user/assistant turns; the
    # real template frames a system message as
    # <|turn>system\n{content}<turn|>\n (unlike Gemma 3, where the system
    # folds into the first user turn).
    # Source: https://huggingface.co/google/gemma-4-31B-it
    "gemma4": Format(
        "gemma4",
        "<bos>",
        "<|turn>user\n",
        "<|turn>model\n",
        "<turn|>\n",
    ),
    # Llama 4 (meta-llama/llama-models models/llama4/prompt_format.md):
    # <|header_start|>/<|header_end|> replace <|start_header_id|>/
    # <|end_header_id|>; <|eot|> replaces <|eot_id|>; new <|end_of_text|> token.
    # Stop-token selection (<|eot|> vs <|end_of_text|>) not fully verifiable
    # (HF template gated).
    # Source: https://github.com/meta-llama/llama-models/blob/main/models/llama4/prompt_format.md
    "llama4": Format(
        "llama4",
        "<|begin_of_text|>",
        "<|header_start|>user<|header_end|>\n\n",
        "<|header_start|>assistant<|header_end|>\n\n",
        "<|eot|>",
    ),
    # GLM-4-9B (zai-org/glm-4-9b-chat): [gMASK]<sop> prefix, per-role frames
    # <|{role}|>\n{content} concatenated with no separators, open suffix
    # <|assistant|> (bare: the generation prompt omits the frame newline).
    # Version-dependent: GLM-4.5/4.6 add trailing \n per message and mandatory
    # empty <think></think> blocks; GLM-5.3-Flash auto-injects a system line.
    # Source: https://huggingface.co/zai-org/glm-4-9b-chat
    "glm4": Format(
        "glm4",
        "[gMASK]<sop>",
        "<|user|>\n",
        "<|assistant|>\n",
        "",
        open_suffix="<|assistant|>",
    ),
    # Kimi K2 (moonshotai/Kimi-K2-Instruct chat_template.jinja):
    # <|im_{role}|>{role_name}<|im_middle|>{content}<|im_end|> (role name
    # twice); default Kimi system injected if missing (not modeled); no BOS
    # emitted (declared-but-unused [BOS] in the tokenizer config).
    # Kimi K3 (Jul 2026) abandoned this format for Python-built XTML.
    # Source: https://huggingface.co/moonshotai/Kimi-K2-Instruct
    "kimi2": Format(
        "kimi2",
        "",
        "<|im_user|>user<|im_middle|>",
        "<|im_assistant|>assistant<|im_middle|>",
        "<|im_end|>",
    ),
    # Kimi K3 (moonshotai/Kimi-K3, encoding_k3.py): K3 abandoned the K2 Jinja
    # template for Python-built XTML. User message: <|open|>message
    # role="user"<|sep|>{content}<|close|>message<|sep|><|end_of_msg|>.
    # Assistant message: the same message frame with a think block injected
    # before the response: <|open|>message role="assistant"<|sep|>
    # <|open|>think<|sep|>{reasoning}<|close|>think<|sep|><|open|>response
    # <|sep|>{content}<|close|>response<|sep|><|close|>message<|sep|>
    # <|end_of_msg|>. The lab models the text-only no-reasoning hypothesis --
    # an EMPTY think block (<|open|>think<|sep|><|close|>think<|sep|>),
    # deterministic -- the reasoned-content branch is not a hypothesis here.
    # No BOS. Generation starts inside the think block, so the open suffix is
    # <|open|>message role="assistant"<|sep|><|open|>think<|sep|>.
    # Source: https://huggingface.co/moonshotai/Kimi-K3
    "kimi3": Format(
        "kimi3",
        "",
        '<|open|>message role="user"<|sep|>',
        '<|open|>message role="assistant"<|sep|>'
        "<|open|>think<|sep|><|close|>think<|sep|>"
        "<|open|>response<|sep|>",
        "",
        user_turn_end="<|close|>message<|sep|><|end_of_msg|>",
        assistant_turn_end="<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>",
        open_suffix='<|open|>message role="assistant"<|sep|><|open|>think<|sep|>',
    ),
    # DeepSeek V4.1-Flash (Oct 2026, MIT): same BOS (<begin of sentence>,
    # full-width bars) and assistant terminator as DeepSeek V3/R1. Differences
    # from V3 (all documented here, not all modeled): system messages are
    # framed as <｜System｜>{content} (V3 concatenated them raw with no
    # framing); the chat template wires in ｜DSML｜ tool markup (not modeled --
    # plain-text no-tools core); and the released template auto-injects a
    # "Reasoning Effort: 75 (range 1-100, the higher the value, the more
    # thorough the reasoning)" preamble line. That effort preamble is
    # DELIBERATELY OMITTED from this base hypothesis: it is
    # version/branch-specific (reasoning-effort steering), and the base must
    # stay deterministic and testable against any effort value; an
    # effort-preamble branch belongs in a separate hypothesis. So the
    # text-only renders are: user = <｜User｜>{content} (no closing token, as
    # in V3), assistant = <｜Assistant｜>{content}<｜end of sentence｜>. Open
    # suffix is the chat-mode <｜Assistant｜><think> (V3's was the bare
    # <｜Assistant｜>; the <｜Assistant｜> transition token is emitted as a
    # post-user marker).
    # Source: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
    "deepseek41": Format(
        "deepseek41",
        f"<{_DS_BAR}begin{_DS_BLK}of{_DS_BLK}sentence{_DS_BAR}>",
        f"<{_DS_BAR}User{_DS_BAR}>",
        f"<{_DS_BAR}Assistant{_DS_BAR}>",
        "",
        user_turn_end="",
        assistant_turn_end=f"<{_DS_BAR}end{_DS_BLK}of{_DS_BLK}sentence{_DS_BAR}>",
        open_suffix=f"<{_DS_BAR}Assistant{_DS_BAR}><think>",
    ),
    # GLM-4.5/4.6 (zai-org/GLM-4.6): [gMASK]<sop>\n prefix (note the newline,
    # GLM-4-9B omits it), per-message frames <|{role}|>\n{content}\n with
    # trailing newlines, and a mandatory <think></think> block injected
    # before the assistant content -- EMPTY in the text-only no-reasoning
    # hypothesis modeled here (the reasoning branch fills it; deterministic
    # once the branch is fixed). Open suffix <|assistant|> (bare, as in
    # GLM-4). GLM-5.3-Flash (Aug 2026, MIT) auto-injects a system line and is
    # not this hypothesis.
    # Source: https://huggingface.co/zai-org/GLM-4.6
    "glm45": Format(
        "glm45",
        "[gMASK]<sop>\n",
        "<|user|>\n",
        "<|assistant|>\n<think></think>\n",
        "\n",
        open_suffix="<|assistant|>",
    ),
    # GLM-5.3-Flash (zai-org/GLM-5.3-Flash, Aug 2026, MIT): TWO whitespace
    # deltas vs the R1 record -- NO newline after the [gMASK]<sop> prefix
    # (GLM-4.5/4.6 emit [gMASK]<sop>\n) and NO newline after <|assistant|>
    # (GLM-4.5/4.6 frame <|assistant|>\n{content}\n). Exact: BOS prefix
    # [gMASK]<sop>; user <|user|>{content} (no trailing newline); assistant
    # <|assistant|><think></think>{content} -- the think block is mandatory
    # and EMPTY in the text-only no-reasoning hypothesis modeled here
    # (deterministic; the reasoning branch fills it, like glm45). The released
    # template auto-injects a <|system|>Reasoning Effort: Max system line
    # (reasoning_effort defaults to `max` per the model card); it is
    # DELIBERATELY OMITTED from this base hypothesis, exactly like DeepSeek
    # V4.1-Flash's effort-preamble omission -- an effort-steering branch
    # belongs in a separate hypothesis. Open suffix <|assistant|><think>
    # (generation starts inside the think block, as in DeepSeek V4.1-Flash
    # chat mode).
    # Source: https://huggingface.co/zai-org/GLM-5.3-Flash
    "glm53": Format(
        "glm53",
        "[gMASK]<sop>",
        "<|user|>",
        "<|assistant|><think></think>",
        "",
        open_suffix="<|assistant|><think>",
    ),
    # Kolibri-1 (Aleph-Alpha, released 2026-10-03, Apache 2.0). Evidence
    # tier 1: the official tokenizer_config.json chat_template, byte-verified
    # by an independent fork. User/assistant turns are standard ChatML --
    # <|im_start|>user\n{content}<|im_end|>\n -- with no BOS, so the text-only
    # no-system render is byte-identical to chatml (pinned in
    # tests/test_cycle10_families.py). Everything that differs is documented
    # here, not modeled: the real template ALWAYS emits a system block, even
    # with no system message:
    # <|im_start|>system\n# Reasoning effort\n\n<SENTENCE>[+ user system
    # content]<|im_end|>\n. SENTENCE is effort-dependent; the lab renders
    # plain-text fixtures with no effort knob, so the catalog documents the
    # DEFAULT-effort (unset -> "high") behavior: the only byte-verified
    # wording is the sentence prefix "Reasoning effort is set to high. Think
    # carefully through..." -- the sentence wording beyond that verified
    # prefix is deliberately NOT asserted. The system role is not modeled in
    # Format (same convention as mistral_large's [SYSTEM_PROMPT] frame and
    # gemma4's system turn); the verified system-block prefix is pinned as a
    # documented constant in the test file. Thinking ON (default): the
    # generation prompt ends at <|im_start|>assistant\n (the model opens
    # <think> itself) -- this base entry's open-reply suffix via the
    # assistant_prefix fallback. Thinking OFF: the generation prompt
    # prefills an empty think block,
    # <|im_start|>assistant\n<think>\n\n</think>\n\n -- modeled as the
    # "kolibri/thinking-off" variant in the mutate path (see candidates()), a
    # generation-prompt branch of the same template, not a second base entry.
    # EOS is <|im_end|> (a generation stop, not a Format field).
    # Source: https://huggingface.co/Aleph-Alpha/Kolibri-1
    "kolibri": Format(
        "kolibri",
        "",
        "<|im_start|>user\n",
        "<|im_start|>assistant\n",
        "<|im_end|>\n",
    ),
    # LFM2.5-24B-A2B (LiquidAI, released 2026-10-13). Evidence tier 2: the
    # official model card plus the HF blog rendered example. ChatML-like:
    # <|startoftext|> BOS, per-message
    # <|im_start|>{role}\n{content}<|im_end|>\n, and -- unlike LFM2 -- the 24B
    # variant always thinks, so the generation prompt appends <think>\n
    # after the assistant frame:
    # <|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user
    # \n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n.
    # Documented caveats: whether <|startoftext|> comes from the bos_token
    # field or a template literal is UNVERIFIED (byte-identical either way);
    # whether the <|im_start|>system\n{system}<|im_end|>\n block is
    # auto-injected or merely illustrative in the card's example is UNVERIFIED
    # -- and the system role is not modeled in Format regardless (same
    # convention as the other entries). EOS is not asserted beyond the
    # per-turn <|im_end|>\n framing.
    # Source: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B
    "lfm25": Format(
        "lfm25",
        "<|startoftext|>",
        "<|im_start|>user\n",
        "<|im_start|>assistant\n",
        "<|im_end|>\n",
        open_suffix="<|im_start|>assistant\n<think>\n",
    ),
    # NVIDIA Nemotron 3.5 Lightning
    # (nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16, released 2026-08-11).
    # Evidence tier 2: the official HF model card, the NeMo SFT docs, and the
    # community osaurus renderer (a byte-level implementation of the official
    # template). Per-turn frames are ChatML:
    # <|im_start|>{role}\n{content}<|im_end|>\n. Thinking ON (default --
    # enable_thinking defaults true): the generation prompt ends at
    # <|im_start|>assistant\n<think>\n (the base entry's open_suffix).
    # Thinking OFF: the generation prompt prefills an EMPTY think block,
    # <|im_start|>assistant\n<think></think> (note: no trailing newlines --
    # unlike kolibri/thinking-off's "<think>\n\n</think>\n\n" tail) -- modeled
    # as the "nemotron35/thinking-off" variant in the mutate path (see
    # candidates()), a generation-prompt branch of the same template, not a
    # second base entry (kolibri/thinking-off convention).
    # CRITICAL BOS CAVEAT: BOS behavior is UNVERIFIED/conflicting at official
    # level -- the catalog models bos="" as an ABSENCE-OF-EVIDENCE
    # placeholder and does NOT assert any BOS token (contrast lfm25, whose
    # <|startoftext|> is a byte-verified tier-2 render). A verified official
    # reading may revise this entry.
    # Sources: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16,
    # https://github.com/nvidia-nemo/nemotron/blob/HEAD/docs/nemotron/lightning35/sft.md,
    # https://github.com/osaurus-ai/osaurus/pull/2360
    "nemotron35": Format(
        "nemotron35",
        "",
        "<|im_start|>user\n",
        "<|im_start|>assistant\n",
        "<|im_end|>\n",
        open_suffix="<|im_start|>assistant\n<think>\n",
    ),
    # Meta Muse Glimmer 30B (meta-models, released 2026-08-10, Apache 2.0).
    # Evidence tier 2: the official meta-oss-cookbook quickstart, the HF
    # transformers model docs, and the HF blog walkthrough -- with the
    # generation prompt COMMUNITY-REFINED (tier 2/community-corroborated):
    # the simpler "<|start|>assistant" alternative was superseded by the
    # recipient-channel form below.
    # Frame: <|begin_of_text|><|start|>{role}<|message|>{content}<|eot|>
    # (no newline anywhere). Assistant turns use recipient channels: the
    # reasoning trace is <|start|>assistant to=self<|message|>... and the
    # user-facing reply <|start|>assistant to=user<|message|>...<|eot|> --
    # the lab's plain-text no-reasoning hypothesis models the assistant turn
    # as to=user and does NOT render the to=self trace (same class of
    # omission as kimi3's empty think block and glm45/glm53's empty
    # <think></think>: the reasoned-content branch is not a hypothesis).
    # Generation prompt: <|start|>assistant to=user<|message|>.
    # Documented, NOT modeled: system messages render the caller text with
    # "\nReasoning strength: high.\n# Valid recipients: \"self\", \"user\"."
    # APPENDED after the caller content (single newlines -- not a standalone
    # preamble block); the system role is not modeled in Format (same
    # convention as mistral_large/gemma4/kolibri). EOS is <|end_of_text|>
    # (id 200001);  is the end-of-MESSAGE marker, NOT a stop token.
    # Sources: https://github.com/meta-models/meta-oss-cookbook/blob/HEAD/quickstart/README.md,
    # http://huggingface.co/docs/transformers/main/model_doc/muse_glimmer,
    # https://github.com/huggingface/blog/blob/main/muse-glimmer.md
    "muse_glimmer": Format(
        "muse_glimmer",
        "<|begin_of_text|>",
        "<|start|>user<|message|>",
        "<|start|>assistant to=user<|message|>",
        "<|eot|>",
        open_suffix="<|start|>assistant to=user<|message|>",
    ),
    # Poolside Laguna 2.1 (poolside/Laguna-XS-2.1, Jul 2026, OpenMDW-1.1).
    # Evidence tier 2: the official HF model cards plus the renderers
    # library's LagunaXS21Renderer/LagunaS21Renderer, which are token-exact
    # parity-verified against apply_chat_template.
    # Base entry = XS-2.1 defaults (documented choice): enable_thinking
    # defaults FALSE, so the generation prompt is <assistant></think> (bare
    # </think> prefill, reasoning dropped from history turns), with no
    # trailing newline. S-2.1 (enable_thinking defaults TRUE, gen prompt
    # <assistant><think>) is the "laguna/thinking-on" mutate-path variant --
    # the kolibri/thinking-off convention: a generation-prompt branch of the
    # same template, not a second catalog entry.
    # Frames ("hugging tags", no inner newlines): <user>{content}</user>,
    # <tool_response>{content}</tool_response>, <system>{content}</system>.
    # Assistant turns, XS-2.1/thinking-off: <assistant></think>{content}
    # </assistant>. Thinking-on renders reasoning VERBATIM (empty included),
    # so the lab's deterministic no-reasoning hypothesis uses an EMPTY think
    # block: <assistant><think></think>{content}</assistant> (the
    # reasoned-content branch is not a hypothesis here -- kimi3 convention).
    # The </assistant> terminator is tier-2 via the renderers PR text ("the
    # two opener tokens are unsampled scaffold; everything through the
    # sampled </assistant> is the model's emission"). EOS is <|EOS|> --
    # DECODE-ONLY, never rendered in the prompt (documented, not modeled).
    # BOS: no BOS token appears in any tier-2 render; modeled as empty with
    # the absence-of-evidence caveat (contrast nemotron35, whose BOS is
    # explicitly conflicting at official level).
    # System role: not modeled (lab convention). The upstream template ships
    # a default Poolside system prompt, but an empty caller system message
    # suppresses the <system> block entirely -- and the lab never sends
    # system messages, so the no-system render modeled here is byte-identical
    # either way; the default-prompt wording is deliberately NOT asserted.
    # Sources: https://huggingface.co/poolside/Laguna-XS-2.1,
    # https://github.com/PrimeIntellect-ai/renderers/pull/99,
    # https://github.com/PrimeIntellect-ai/renderers/pull/111
    "laguna": Format(
        "laguna",
        "",
        "<user>",
        "<assistant></think>",
        "",
        user_turn_end="</user>",
        assistant_turn_end="</assistant>",
        open_suffix="<assistant></think>",
    ),
    # Ling-3.1-flash (inclusionAI, released 2026-10-15, MIT). Evidence
    # tier 1: the official chat_template.jinja ("Bailing V3", preserved at
    # docs/evidence/ling31-chat-template.jinja). Every string below is
    # derived from the template's plain-text, no-tools branch, cited by
    # jinja line number:
    # - thinking option (lines 5-13): enable_thinking/thinking_option
    #   unset -> thinking_option defaults to 'on' (line 12), so the base
    #   entry models thinking ON (the lab's default render). Thinking OFF
    #   is a generation-prompt branch (lines 127-128) and rides the mutate
    #   path as "ling/thinking-off" (kolibri/thinking-off convention).
    # - system block (lines 19, 35-45): ALWAYS emitted even with no system
    #   message. With system message X (no "detailed thinking" marker in
    #   X): <role>SYSTEM</role>X\ndetailed thinking on<|role_end|>
    #   (lines 40-41); without: <role>SYSTEM</role>detailed thinking
    #   on<|role_end|> (line 44). The system role is not modeled in Format
    #   (same convention as mistral_large/gemma4/kolibri); the block is
    #   pinned as a documented constant in tests/test_cycle13_families.py.
    # - user (line 62): <role>HUMAN</role>{content}<|role_end|>.
    # - assistant history, no reasoning (lines 75-79): <role>ASSISTANT</role>
    #   \n<think></think>{content} (line 79: the empty-think branch;
    #   preserved_thinking is true, line 16), closed by <|role_end|>
    #   (line 108).
    # - generation prompt (lines 123-129): <role>ASSISTANT</role>\n<think>
    #   when thinking is on (line 126), <role>ASSISTANT</role>\n<think>
    #   </think> when off (line 128).
    # BOS: <|startoftext|> -- tier-1 verified from the official
    # tokenizer_config.json bos_token field (read 2026-10-15). Caveat: the
    # jinja template itself emits no BOS literal and add_bos_token is
    # false, so a raw apply_chat_template call would not add it either;
    # the entry takes the config's bos_token (lfm25 convention).
    # EOS caveat: the official eos_token field is <|role_end|> -- NOT
    # <|endoftext|> (that token is the pad_token and a special added
    # token); documented, not modeled (Format has no EOS field). Out of
    # scope for the plain-text core: the tool-call branches (lines 24-34,
    # 84-107), the reasoning-content branches (lines 67-77), and the
    # multi-step tool scan (lines 47-56).
    # Source: https://huggingface.co/inclusionAI/Ling-3.1-flash
    "ling": Format(
        "ling",
        "<|startoftext|>",
        "<role>HUMAN</role>",
        "<role>ASSISTANT</role>\n<think></think>",
        "<|role_end|>",
        open_suffix="<role>ASSISTANT</role>\n<think>",
    ),
}


# Kolibri-1 thinking-off branch: same frames as the base entry, but the
# generation prompt prefills an empty <think> block (thinking disabled).
# Deterministic string; pinned byte-exact in tests/test_cycle10_families.py.
_KOLIBRI_THINKING_OFF_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Nemotron 3.5 Lightning thinking-off branch: same frames as the base
# entry, but the generation prompt prefills an empty <think> block
# (thinking disabled). Note the absent trailing newlines -- unlike
# kolibri/thinking-off's "<think>\n\n</think>\n\n" tail. Deterministic
# string; pinned byte-exact in tests/test_cycle11_families.py.
_NEMOTRON35_THINKING_OFF_SUFFIX = "<|im_start|>assistant\n<think></think>"

# Ling-3.1-flash thinking-off branch: same frames as the base entry, but
# the generation prompt prefills an empty <think> block (thinking
# disabled, jinja lines 127-128). Deterministic string; pinned byte-exact
# in tests/test_cycle13_families.py.
_LING_THINKING_OFF_SUFFIX = "<role>ASSISTANT</role>\n<think></think>"


def candidates(family: str, *, mutate: bool = False) -> tuple[Format, ...]:
    if family not in CATALOG:
        raise ValueError("Unknown format family")
    base = CATALOG[family]
    if not mutate:
        return (base,)
    variants = local_repairs(base)
    if family == "kolibri":
        # Thinking-off is a generation-prompt branch of the same template,
        # so it rides the mutate path like the BOS/prefix local repairs
        # rather than becoming a second catalog entry. Its open_suffix is
        # new, so it cannot collide with local_repairs' dedup keys.
        variants += (
            replace(
                base,
                name=base.name + "/thinking-off",
                open_suffix=_KOLIBRI_THINKING_OFF_SUFFIX,
            ),
        )
    if family == "nemotron35":
        # Thinking-off is a generation-prompt branch of the same template
        # (the official default is thinking-on, which the base entry models
        # via its open_suffix), so it rides the mutate path like
        # kolibri/thinking-off rather than becoming a second catalog entry.
        # Its open_suffix is new, so it cannot collide with local_repairs'
        # dedup keys.
        variants += (
            replace(
                base,
                name=base.name + "/thinking-off",
                open_suffix=_NEMOTRON35_THINKING_OFF_SUFFIX,
            ),
        )
    if family == "ling":
        # Thinking-off is a generation-prompt branch of the same template
        # (jinja lines 127-128; the official default is thinking-on, which
        # the base entry models via its open_suffix), so it rides the mutate
        # path like kolibri/thinking-off rather than becoming a second
        # catalog entry. Its open_suffix is new, so it cannot collide with
        # local_repairs' dedup keys.
        variants += (
            replace(
                base,
                name=base.name + "/thinking-off",
                open_suffix=_LING_THINKING_OFF_SUFFIX,
            ),
        )
    if family == "laguna":
        # S-2.1's default (enable_thinking=True) is a generation-prompt
        # branch of the same template, so it rides the mutate path like
        # kolibri/thinking-off rather than becoming a second catalog entry:
        # the gen prompt is <assistant><think>, and history assistant turns
        # render an EMPTY think block verbatim (the lab's deterministic
        # no-reasoning hypothesis -- the reasoned-content branch is not a
        # hypothesis, kimi3 convention). The base entry stays XS-2.1
        # defaults (thinking off). The new open_suffix and assistant_prefix
        # cannot collide with local_repairs' dedup keys.
        variants += (
            replace(
                base,
                name=base.name + "/thinking-on",
                open_suffix="<assistant><think>",
                assistant_prefix="<assistant><think></think>",
            ),
        )
    return variants
