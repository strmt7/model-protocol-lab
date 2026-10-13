"""Local safetensors inference with greedy, raw-token, full-prefix decoding."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from threading import Event, RLock
from typing import Any

from .core import Generation


def _directory(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("invalid_local_directory")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        # Missing paths, permission errors and symlink loops all normalize
        # to the function's documented error vocabulary.
        raise ValueError("invalid_local_directory") from None
    if not path.is_dir():
        raise ValueError("invalid_local_directory")
    return path


def _member(root: Path, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("invalid_asset_name")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing_or_unsafe_asset")
    return path


def _read_json(path: Path) -> dict:
    with path.open("rb") as handle:
        content = handle.read(128 * 1024 * 1024 + 1)
    if len(content) > 128 * 1024 * 1024:
        raise ValueError("oversized_asset_json")
    value = json.loads(content.decode("utf-8"))
    if type(value) is not dict:
        raise ValueError("invalid_asset_json")
    return value


def _model_assets(root: Path) -> tuple[str, ...]:
    if (root / "adapter_config.json").exists():
        raise ValueError("adapters_not_supported")
    single, index = root / "model.safetensors", root / "model.safetensors.index.json"
    if single.exists() == index.exists():
        raise ValueError("require_one_standard_safetensors_layout")
    names = {"config.json"}
    if (root / "generation_config.json").exists():
        names.add("generation_config.json")
    if single.exists():
        names.add("model.safetensors")
    else:
        names.add("model.safetensors.index.json")
        mapping = _read_json(_member(root, index.name)).get("weight_map")
        if type(mapping) is not dict or not mapping:
            raise ValueError("invalid_weight_index")
        for name in mapping.values():
            if type(name) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", name):
                raise ValueError("invalid_weight_index")
            names.add(name)
    for name in names:
        _member(root, name)
    return tuple(sorted(names))


def _manifest(root: Path, names: tuple[str, ...]) -> list[dict]:
    records = []
    for name in names:
        path = _member(root, name)
        before = path.stat()
        digest = sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise ValueError("assets_changed")
        records.append({"name": name, "bytes": after.st_size, "sha256": digest.hexdigest()})
    return records


def _ids_valid(value: object, vocab_size: int) -> bool:
    return type(value) is tuple and all(
        type(token) is int and 0 <= token < vocab_size for token in value
    )


class _ContextWindowError(Exception):
    """max_context exceeds the checkpoint's declared context window.

    Converted to a distinct fixed RuntimeError code (not collapsed into the
    generic init failure) so the operator learns the actionable cause:
    stock GPT-2 checkpoints declare n_positions=1024, below the CLI's
    default --max-context of 2048.
    """


class LocalHFBackend:
    """Built-in GPT-2/Llama only; self-contained local assets; no download fallback.

    Greedy fp32 decoding recomputes the full prefix and ignores automatic stops.
    The seed is validated but does not change greedy selection. Constructor
    loading may affect global RNG state; requests do not seed global generators.
    Full-content fingerprints read selected assets twice during initialization.
    Cancellation is cooperative between forwards, not an interruption of kernels.
    """

    def __init__(
        self, model_path: str | Path, *, device: str = "cpu", max_context: int = 2048
    ) -> None:
        self._lock = RLock()
        self._model = self._tokenizer = None
        try:
            if type(max_context) is not int or not 1 <= max_context <= 32768:
                raise ValueError("invalid_context_limit")
            if device not in ("cpu", "cuda"):
                raise ValueError("invalid_explicit_device")
            effective_device = "cuda:0" if device == "cuda" else "cpu"
            root = _directory(model_path)
            model_names = _model_assets(root)
            before = {
                "model": _manifest(root, model_names),
                "tokenizer": _manifest(root, ("tokenizer.json",)),
            }
            raw_config = _read_json(_member(root, "config.json"))
            if raw_config.get("model_type") not in {"gpt2", "llama"}:
                raise ValueError("unsupported_model_type")
            if any(
                raw_config.get(key)
                for key in (
                    "auto_map",
                    "quantization_config",
                    "is_encoder_decoder",
                    "add_cross_attention",
                )
            ):
                raise ValueError("unsupported_model_configuration")
            token_model = _read_json(_member(root, "tokenizer.json")).get("model")
            if type(token_model) is not dict:
                raise ValueError("invalid_tokenizer")
            if token_model.get("dropout") not in (None, 0, 0.0):
                raise ValueError("stochastic_tokenizer_not_supported")

            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

            self._torch = torch
            self._device = torch.device(effective_device)
            if self._device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError("requested_device_unavailable")
            # downloading; trust_remote_code=False blocks custom model code.
            config = AutoConfig.from_pretrained(  # nosec B615 -- local_files_only, no remote code
                str(root), local_files_only=True, trust_remote_code=False, token=False
            )
            declared = (
                config.n_positions
                if config.model_type == "gpt2"
                else config.max_position_embeddings
            )
            if type(declared) is not int or not 1 <= max_context <= declared:
                raise _ContextWindowError(
                    f"max_context {max_context} exceeds declared window {declared}"
                )
            config.use_cache = False
            # _read_json capped this file at 128 MB, but HF re-reads it below;
            # enforce the same bound on the file HF will actually parse so a
            # swapped-in oversized tokenizer.json cannot exhaust memory.
            tokenizer_path = _member(root, "tokenizer.json")
            if tokenizer_path.stat().st_size > 128 * 1024 * 1024:
                raise ValueError("oversized_tokenizer_file")
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_path),
                clean_up_tokenization_spaces=False,
            )
            # transformers ships without type stubs; keep the untyped boundary
            # explicit so mypy does not have to infer through it.
            pretrained: Any = AutoModelForCausalLM.from_pretrained(  # nosec B615 -- local only
                str(root),
                config=config,
                local_files_only=True,
                trust_remote_code=False,
                token=False,
                use_safetensors=True,
                weights_only=True,
                dtype=torch.float32,
                device_map=None,
                attn_implementation="eager",
                output_loading_info=True,
            )
            model: Any
            loading: Any
            model, loading = pretrained
            if any(
                loading.get(key)
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            ):
                raise ValueError("checkpoint_loading_mismatch")
            model.eval()
            model.requires_grad_(False)
            model.to(self._device)
            vocab_size = model.get_input_embeddings().num_embeddings
            tokenizer_ids = frozenset(tokenizer.get_vocab().values())
            if not tokenizer_ids or any(
                type(t) is not int or not 0 <= t < vocab_size for t in tokenizer_ids
            ):
                raise ValueError("tokenizer_model_vocabulary_mismatch")
            after = {
                "model": _manifest(root, model_names),
                "tokenizer": _manifest(root, ("tokenizer.json",)),
            }
            if before != after:
                raise ValueError("assets_changed")
            identity = {
                "adapter": "local-hf-greedy-v1",
                "assets": after,
                "versions": {
                    p: version(p) for p in ("torch", "transformers", "tokenizers", "safetensors")
                },
                "device": effective_device,
                "dtype": "float32",
                "attention": "eager",
                "context_limit": max_context,
                "seed_used_for_selection": False,
                "tokenizer_policy": "raw-tokenizer-json-no-added-specials",
                "generation_policy": "greedy-full-prefix-no-stops-no-cache",
            }
            payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.fingerprint = "sha256:" + sha256(payload).hexdigest()
            self.seed_used_for_selection = False
            self._context_limit, self._vocab_size = max_context, vocab_size
            self._tokenizer_ids, self._tokenizer, self._model = tokenizer_ids, tokenizer, model
        except _ContextWindowError:
            self._model = self._tokenizer = None
            raise RuntimeError("local_context_limit_exceeds_declared_window") from None
        except Exception:  # noqa: BLE001 -- init must fail closed on any dependency error
            self._model = self._tokenizer = None
            raise RuntimeError("local_backend_initialization_failed") from None

    def _ready(self) -> tuple[Any, Any]:
        """Return initialized components, narrowing the None-able attributes.

        __init__ either completes (leaving both set) or raises, so reaching
        here with a None component means a programming error, not a state
        the caller can produce through the public API. The components are
        typed Any because they come from the optional, untyped HF stack.
        """
        tokenizer, model = self._tokenizer, self._model
        if tokenizer is None or model is None:
            raise RuntimeError("local_backend_not_initialized")
        return tokenizer, model

    def encode(self, text: str) -> tuple[int, ...]:
        if type(text) is not str:
            raise ValueError("invalid_text")
        tokenizer, _ = self._ready()
        try:
            with self._lock:
                values = tuple(
                    tokenizer.encode(
                        text, add_special_tokens=False, truncation=False, padding=False
                    )
                )
            if not _ids_valid(values, self._vocab_size):
                raise ValueError("invalid_encoded_ids")
            return values
        except Exception:  # noqa: BLE001 -- sanitization boundary, see pyproject
            raise RuntimeError("local_encode_failed") from None

    def decode(self, token_ids: tuple[int, ...]) -> str:
        if not _ids_valid(token_ids, self._vocab_size) or any(
            t not in self._tokenizer_ids for t in token_ids
        ):
            raise ValueError("invalid_decode_ids")
        tokenizer, _ = self._ready()
        try:
            with self._lock:
                text: str = tokenizer.decode(
                    list(token_ids),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                return text
        except Exception:  # noqa: BLE001 -- sanitization boundary, see pyproject
            raise RuntimeError("local_decode_failed") from None

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        if not _ids_valid(input_ids, self._vocab_size) or not input_ids:
            return Generation((), "error", "invalid_or_empty_input")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
            return Generation((), "error", "invalid_output_cap")
        if type(seed) is not int or not 0 <= seed < 2**64:
            return Generation((), "error", "invalid_seed")
        if not isinstance(cancel, Event):
            return Generation((), "error", "invalid_cancellation_event")
        if cancel.is_set():
            return Generation((), "cancelled")
        if len(input_ids) + max_new_tokens > self._context_limit:
            return Generation((), "error", "context_limit_exceeded")
        _, model = self._ready()
        acquired = False
        output: list[int] = []
        sequence = result = logits = None
        torch = self._torch
        try:
            while not acquired:
                if cancel.is_set():
                    return Generation((), "cancelled")
                acquired = self._lock.acquire(timeout=0.05)
            if cancel.is_set():
                return Generation((), "cancelled")
            with torch.inference_mode():
                sequence = torch.tensor([input_ids], dtype=torch.long, device=self._device)
                for step in range(max_new_tokens):
                    if cancel.is_set():
                        return Generation(tuple(output), "cancelled")
                    result = model(
                        input_ids=sequence,
                        attention_mask=torch.ones_like(sequence),
                        past_key_values=None,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    logits = result.logits[0, -1, :]
                    if (
                        logits.ndim != 1
                        or logits.numel() != self._vocab_size
                        or not bool(torch.isfinite(logits).all().item())
                    ):
                        return Generation(tuple(output), "error", "invalid_next_token_logits")
                    token = int(torch.argmax(logits).item())
                    if cancel.is_set():
                        return Generation(tuple(output), "cancelled")
                    output.append(token)
                    result = logits = None
                    if step + 1 < max_new_tokens:
                        sequence = torch.cat(
                            (
                                sequence,
                                torch.tensor([[token]], dtype=torch.long, device=self._device),
                            ),
                            dim=1,
                        )
            return Generation(tuple(output), "cancelled" if cancel.is_set() else "limit")
        except Exception:  # noqa: BLE001 -- inference errors become fixed codes
            return Generation(tuple(output), "error", "local_inference_failed")
        finally:
            sequence = result = logits = None
            output.clear()
            if acquired:
                self._lock.release()
