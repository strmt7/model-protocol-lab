"""Basic text-only format hypotheses; support must be measured per checkpoint."""
from .core import Format, local_repairs

CATALOG = {
    "chatml": Format("chatml", "", "<|im_start|>user\n",
                     "<|im_start|>assistant\n", "<|im_end|>\n"),
    "llama3": Format("llama3", "<|begin_of_text|>",
                     "<|start_header_id|>user<|end_header_id|>\n\n",
                     "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"),
    "gemma": Format("gemma", "<bos>", "<start_of_turn>user\n",
                    "<start_of_turn>model\n", "<end_of_turn>\n"),
}


def candidates(family: str, *, mutate: bool = False) -> tuple[Format, ...]:
    if family not in CATALOG:
        raise ValueError("Unknown format family")
    base = CATALOG[family]
    return local_repairs(base) if mutate else (base,)
