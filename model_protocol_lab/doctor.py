"""Pre-flight validation for local checkpoints and fixture files.

`doctor` inspects a model directory and (optionally) a fixture file without
loading model weights, executing model code, or touching the network. It is
meant to run before any experiment so configuration problems fail fast with
a clear report instead of mid-run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends import _directory, _manifest, _member, _model_assets, _read_json
from .datasets import load_dataset

_SUPPORTED_TYPES = ("gpt2", "llama")
_REJECTED_CONFIG_KEYS = (
    "auto_map",
    "quantization_config",
    "is_encoder_decoder",
    "add_cross_attention",
)


def _check(status: str, detail: str = "") -> dict[str, str]:
    return {"status": status, "detail": detail}


def _model_result(
    verdict: str,
    checks: dict[str, dict[str, str]],
    *,
    assets: list[dict] | None = None,
    special_tokens: dict[str, int] | None = None,
    added_tokens: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Every examine_model return carries the same keys.

    Early-failure returns previously omitted "assets"/"special_tokens"/
    "added_tokens", so consumers can key on them unconditionally; the
    collections are simply empty when the corresponding stage never ran.
    """
    return {
        "verdict": verdict,
        "model": {
            "checks": checks,
            "assets": assets if assets is not None else [],
            "special_tokens": special_tokens if special_tokens is not None else {},
            "added_tokens": added_tokens if added_tokens is not None else {},
        },
    }


def examine_model(
    model_path: str | Path, *, boundary_ids: tuple[int, ...] | None = None
) -> dict[str, Any]:
    """Validate a local checkpoint directory; never loads weights."""
    checks: dict[str, dict[str, str]] = {}
    try:
        root = _directory(model_path)
    except ValueError:
        checks["directory"] = _check("failed", "not a readable local directory")
        return _model_result("failed", checks)
    checks["directory"] = _check("ok", str(root))

    try:
        names = _model_assets(root)
    except ValueError as exc:
        checks["asset_layout"] = _check("failed", str(exc))
        return _model_result("failed", checks)
    try:
        manifest = _manifest(root, names)
    except ValueError as exc:
        checks["asset_manifest"] = _check("failed", str(exc))
        return _model_result("failed", checks)
    checks["asset_layout"] = _check("ok", f"{len(names)} assets")
    checks["asset_manifest"] = _check("ok", "sha256 recorded")

    try:
        config = _read_json(_member(root, "config.json"))
    except ValueError as exc:
        checks["config"] = _check("failed", str(exc))
        return {"verdict": "failed", "model": {"checks": checks}}
    model_type = config.get("model_type")
    rejected = [key for key in _REJECTED_CONFIG_KEYS if config.get(key)]
    if model_type not in _SUPPORTED_TYPES:
        checks["config"] = _check("failed", f"unsupported model_type: {model_type!r}")
    elif rejected:
        checks["config"] = _check("failed", f"rejected config keys: {', '.join(rejected)}")
    else:
        checks["config"] = _check("ok", f"model_type={model_type}")
    if checks["config"]["status"] == "failed":
        return _model_result("failed", checks, assets=manifest)

    try:
        tokenizer_doc = _read_json(_member(root, "tokenizer.json"))
    except ValueError as exc:
        checks["tokenizer_file"] = _check("failed", str(exc))
        return _model_result("failed", checks, assets=manifest)
    token_model = tokenizer_doc.get("model")
    if type(token_model) is not dict:
        checks["tokenizer_file"] = _check("failed", "tokenizer model section missing")
    elif token_model.get("dropout") not in (None, 0, 0.0):
        checks["tokenizer_file"] = _check("failed", "stochastic tokenizer not supported")
    else:
        checks["tokenizer_file"] = _check("ok", "standalone tokenizer.json")

    vocab_size: int | None = None
    specials: dict[str, int] = {}
    added_tokens: dict[str, int] = {}
    # Added tokens are read from the document itself: transformers does not
    # reliably surface them when loading from tokenizer_file alone, and the
    # file is the authority for what the checkpoint ships.
    added_section = tokenizer_doc.get("added_tokens")
    if type(added_section) is list:
        for entry in added_section:
            if (
                type(entry) is dict
                and type(entry.get("content")) is str
                and type(entry.get("id")) is int
            ):
                added_tokens[entry["content"]] = entry["id"]
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        checks["tokenizer_deep"] = _check(
            "skipped", "transformers not installed; install the 'hf' extra for deep checks"
        )
    else:
        try:
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(_member(root, "tokenizer.json")),
                clean_up_tokenization_spaces=False,
            )
            vocab = tokenizer.get_vocab()
            vocab_size = len(vocab)
            for attr in (
                "unk_token",
                "bos_token",
                "eos_token",
                "pad_token",
                "sep_token",
                "cls_token",
                "mask_token",
            ):
                token = getattr(tokenizer, attr, None)
                if token is not None:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    if type(token_id) is int:
                        specials[attr] = token_id
            checks["tokenizer_deep"] = _check(
                "ok", f"vocab_size={vocab_size}, {len(added_tokens)} added tokens"
            )
        except Exception:  # noqa: BLE001 -- deep check must fail closed
            # Sanitized vocabulary: never interpolate raw dependency exception
            # text (controlled by the inspected tokenizer.json) into output.
            checks["tokenizer_deep"] = _check("failed", "tokenizer_deep_check_failed")

    if boundary_ids is not None:
        if vocab_size is None:
            checks["boundary_ids"] = _check(
                "skipped", "vocabulary size unknown without the 'hf' extra"
            )
        else:
            # Strict element types (bool is not an int here, matching the
            # backend's _ids_valid): a non-iterable or mistyped element is a
            # failed check, never a bare TypeError.
            try:
                ids = tuple(boundary_ids)
            except TypeError:
                ids = ()
            if ids and all(type(token) is int and 0 <= token < vocab_size for token in ids):
                checks["boundary_ids"] = _check("ok", f"{len(ids)} ids within vocabulary")
            else:
                checks["boundary_ids"] = _check("failed", "invalid boundary ids")

    verdict = "failed" if any(c["status"] == "failed" for c in checks.values()) else "ok"
    return _model_result(
        verdict,
        checks,
        assets=manifest,
        special_tokens=specials,
        added_tokens=added_tokens,
    )


def examine_dataset(path: str | Path) -> dict[str, Any]:
    """Validate a fixture file; reports counts only."""
    try:
        # a bounded local JSON reader; not huggingface_hub.datasets.
        split = load_dataset(path)  # nosec B615 -- own bounded JSON reader, not HF datasets
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return {
            "verdict": "failed",
            "dataset": {"status": "failed", "detail": f"{type(exc).__name__}: {exc}"},
        }
    return {
        "verdict": "ok",
        "dataset": {
            "status": "ok",
            "search_cases": len(split.search),
            "held_out_cases": len(split.held_out),
        },
    }


def doctor(
    model_path: str | Path,
    dataset_path: str | Path | None = None,
    *,
    boundary_ids: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Combined pre-flight report for a model directory and fixture file."""
    report = examine_model(model_path, boundary_ids=boundary_ids)
    if dataset_path is not None:
        dataset_report = examine_dataset(dataset_path)
        report["dataset"] = dataset_report["dataset"]
        if dataset_report["verdict"] == "failed":
            report["verdict"] = "failed"
    return report
