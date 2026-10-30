"""Differential verification of the checkpoint-asset validation in doctor.py.

Scope: the functions that validate a local checkpoint directory without
loading weights -- ``examine_model`` (directory / asset-layout / manifest /
config / tokenizer-file / deep-tokenizer / boundary checks), the
``tokenizer_config.json`` handling inside ``template_conformance``, and the
context-window fitting inside ``_next_command`` (exercised via ``doctor``).

Method: synthetic checkpoint directories are built in tmp dirs only
(hand-written safetensors files with valid / truncated / bad-magic headers,
index JSON variants, tokenizer_config.json variants, generation_config.json
variants). An independent oracle reimplements the *documented* decision logic
from the docstrings -- its own ``struct``-based safetensors header parse, its
own JSON reader carrying the same fixed error vocabulary, its own
context-window arithmetic -- and every test compares the implementation's
verdicts against the oracle field-by-field: which checks pass/fail, the
failure codes/reasons, the asset manifest, the context window, the
special/added token maps, and the suggested next-command limits.

Documented-behavior discrepancies recorded while grounding the oracle (the
implementation matches its docstrings; the gaps below are between the task
brief's assumptions and the actual code -- no real bug was found):

* doctor.py performs byte-level safetensors header validation (the
  ``weight_headers`` check, added after this file's initial verification):
  for each ``*.safetensors`` asset only the 8-byte little-endian header
  length plus the JSON header are read (tensor data is never touched); the
  file must hold that many bytes and the header must parse as a JSON
  object, otherwise the check fails with the fixed detail
  ``invalid_safetensors_header: <name>`` and the verdict is ``failed``.
  The filename-level ``asset_layout`` check is unchanged: truncated
  headers and bad-magic bytes still pass *it*; they now fail
  ``weight_headers`` instead.
* There are NO generation_config.json content checks: the file is hashed
  into the asset manifest when present and otherwise ignored. The presence or
  absence of ``eos_token_id`` changes no verdict.
* The oracle does not reproduce ``_manifest``'s ``assets_changed`` TOCTOU
  guard: it cannot be triggered deterministically in a single-threaded test.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import shlex
import shutil
import struct
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_protocol_lab import doctor as doctor_mod
from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.core import Fixture, Format, Message
from model_protocol_lab.datasets import load_dataset

# ---------------------------------------------------------------------------
# Independent oracle: reimplements the documented decision logic of
# doctor.examine_model / template_conformance / _next_command without calling
# any of them (or the backends helpers they delegate to).
# ---------------------------------------------------------------------------

_JSON_CAP = 128 * 1024 * 1024  # documented 128 MB bound on asset JSON reads
_TEMPLATE_CAP = 1_000_000  # documented chat-template source cap
_DEFAULT_NEW_TOKENS = 64  # documented default suggestion budget
_SUPPORTED_TYPES = ("gpt2", "llama")
_REJECTED_CONFIG_KEYS = (
    "auto_map",
    "quantization_config",
    "is_encoder_decoder",
    "add_cross_attention",
)
_SPECIAL_ATTRS = (
    "unk_token",
    "bos_token",
    "eos_token",
    "pad_token",
    "sep_token",
    "cls_token",
    "mask_token",
)


def _o_read_json(path: Path) -> dict:
    """Oracle JSON reader with the documented fixed error vocabulary."""
    with open(path, "rb") as handle:
        content = handle.read(_JSON_CAP + 1)
    if len(content) > _JSON_CAP:
        raise ValueError("oversized_asset_json")
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("invalid_asset_json") from None
    if type(value) is not dict:
        raise ValueError("invalid_asset_json")
    return value


def _o_member(root: Path, name: str) -> Path:
    """Oracle asset member check: safe name, regular non-symlink file."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("invalid_asset_name")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing_or_unsafe_asset")
    return path


def _o_directory(value: str | Path) -> Path:
    """Oracle directory check with the documented error vocabulary."""
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("invalid_local_directory")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("invalid_local_directory") from None
    if not path.is_dir():
        raise ValueError("invalid_local_directory")
    return path


def _o_model_assets(root: Path) -> tuple[str, ...]:
    """Oracle asset-layout validation per the documented layout rules."""
    if (root / "adapter_config.json").exists():
        raise ValueError("adapters_not_supported")
    single = root / "model.safetensors"
    index = root / "model.safetensors.index.json"
    if single.exists() == index.exists():
        raise ValueError("require_one_standard_safetensors_layout")
    names = {"config.json"}
    if (root / "generation_config.json").exists():
        names.add("generation_config.json")
    if single.exists():
        names.add("model.safetensors")
    else:
        names.add("model.safetensors.index.json")
        mapping = _o_read_json(_o_member(root, "model.safetensors.index.json")).get("weight_map")
        if type(mapping) is not dict or not mapping:
            raise ValueError("invalid_weight_index")
        for name in mapping.values():
            if type(name) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", name):
                raise ValueError("invalid_weight_index")
            names.add(name)
    for name in names:
        _o_member(root, name)
    return tuple(sorted(names))


def _o_manifest(root: Path, names: tuple[str, ...]) -> list[dict]:
    """Oracle sha256 manifest (own hashing; TOCTOU guard not reproduced)."""
    records = []
    for name in names:
        data = (root / name).read_bytes()
        records.append(
            {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    return records


def _o_added_tokens(tokenizer_doc: dict) -> dict[str, int]:
    """Oracle added-token map, parsed from the tokenizer.json document."""
    added: dict[str, int] = {}
    section = tokenizer_doc.get("added_tokens")
    if type(section) is list:
        for entry in section:
            if (
                type(entry) is dict
                and type(entry.get("content")) is str
                and type(entry.get("id")) is int
            ):
                added[entry["content"]] = entry["id"]
    return added


def _o_load_tokenizer(root: Path) -> object:
    """Oracle's own tokenizer load (own import, own 128 MB pre-load cap)."""
    from transformers import PreTrainedTokenizerFast

    path = root / "tokenizer.json"
    if path.stat().st_size > _JSON_CAP:
        raise ValueError("oversized_tokenizer_file")
    return PreTrainedTokenizerFast(tokenizer_file=str(path), clean_up_tokenization_spaces=False)


def _o_specials(tokenizer: object) -> dict[str, int]:
    """Oracle special-token map from the loaded tokenizer's attributes."""
    specials: dict[str, int] = {}
    for attr in _SPECIAL_ATTRS:
        token = getattr(tokenizer, attr, None)
        if token is not None:
            token_id = tokenizer.convert_tokens_to_ids(token)  # type: ignore[union-attr]
            if type(token_id) is int:
                specials[attr] = token_id
    return specials


def _o_model_result(
    verdict: str,
    checks: dict[str, dict[str, str]],
    *,
    assets: list[dict] | None = None,
    special_tokens: dict[str, int] | None = None,
    added_tokens: dict[str, int] | None = None,
    context_window: int | None = None,
) -> dict:
    """Oracle report envelope: every return carries the same keys."""
    return {
        "verdict": verdict,
        "model": {
            "checks": checks,
            "assets": assets if assets is not None else [],
            "special_tokens": special_tokens if special_tokens is not None else {},
            "added_tokens": added_tokens if added_tokens is not None else {},
            "context_window": context_window,
        },
    }


def oracle_examine_model(
    model_path: str | Path,
    *,
    boundary_ids: tuple[int, ...] | None = None,
    boundary_map: dict[str, tuple[int, ...]] | None = None,
) -> dict:
    """Independent reimplementation of examine_model's documented checks."""
    checks: dict[str, dict[str, str]] = {}
    try:
        root = _o_directory(model_path)
    except ValueError:
        checks["directory"] = {"status": "failed", "detail": "not a readable local directory"}
        return _o_model_result("failed", checks)
    checks["directory"] = {"status": "ok", "detail": str(root)}

    try:
        names = _o_model_assets(root)
    except ValueError as exc:
        checks["asset_layout"] = {"status": "failed", "detail": str(exc)}
        return _o_model_result("failed", checks)
    manifest = _o_manifest(root, names)
    checks["asset_layout"] = {"status": "ok", "detail": f"{len(names)} assets"}
    checks["asset_manifest"] = {"status": "ok", "detail": "sha256 recorded"}
    checks["weight_headers"] = _o_weight_headers_check(root, names)
    if checks["weight_headers"]["status"] == "failed":
        return _o_model_result("failed", checks, assets=manifest)

    try:
        config = _o_read_json(_o_member(root, "config.json"))
    except ValueError as exc:
        checks["config"] = {"status": "failed", "detail": str(exc)}
        return _o_model_result("failed", checks, assets=manifest)
    model_type = config.get("model_type")
    rejected = [key for key in _REJECTED_CONFIG_KEYS if config.get(key)]
    if model_type not in _SUPPORTED_TYPES:
        checks["config"] = {
            "status": "failed",
            "detail": f"unsupported model_type: {model_type!r}",
        }
    elif rejected:
        checks["config"] = {
            "status": "failed",
            "detail": f"rejected config keys: {', '.join(rejected)}",
        }
    else:
        checks["config"] = {"status": "ok", "detail": f"model_type={model_type}"}
    if checks["config"]["status"] == "failed":
        return _o_model_result("failed", checks, assets=manifest)

    declared = (
        config.get("n_positions") if model_type == "gpt2" else config.get("max_position_embeddings")
    )
    context_window = declared if type(declared) is int and declared > 0 else None

    try:
        tokenizer_doc = _o_read_json(_o_member(root, "tokenizer.json"))
    except ValueError as exc:
        checks["tokenizer_file"] = {"status": "failed", "detail": str(exc)}
        return _o_model_result("failed", checks, assets=manifest, context_window=context_window)
    token_model = tokenizer_doc.get("model")
    if type(token_model) is not dict:
        checks["tokenizer_file"] = {
            "status": "failed",
            "detail": "tokenizer model section missing",
        }
    elif token_model.get("dropout") not in (None, 0, 0.0):
        checks["tokenizer_file"] = {
            "status": "failed",
            "detail": "stochastic tokenizer not supported",
        }
    else:
        checks["tokenizer_file"] = {"status": "ok", "detail": "standalone tokenizer.json"}

    vocab_size: int | None = None
    specials: dict[str, int] = {}
    added_tokens = _o_added_tokens(tokenizer_doc)
    try:
        tokenizer = _o_load_tokenizer(root)
        vocab_size = len(tokenizer.get_vocab())  # type: ignore[union-attr]
        specials = _o_specials(tokenizer)
        checks["tokenizer_deep"] = {
            "status": "ok",
            "detail": f"vocab_size={vocab_size}, {len(added_tokens)} added tokens",
        }
    except ImportError:
        checks["tokenizer_deep"] = {
            "status": "skipped",
            "detail": "transformers not installed; install the 'hf' extra for deep checks",
        }
    except Exception:  # noqa: BLE001 -- oracle mirrors the fail-closed deep check
        checks["tokenizer_deep"] = {"status": "failed", "detail": "tokenizer_deep_check_failed"}

    if boundary_ids is not None:
        if vocab_size is None:
            checks["boundary_ids"] = {
                "status": "skipped",
                "detail": "vocabulary size unknown without the 'hf' extra",
            }
        else:
            try:
                ids = tuple(boundary_ids)
            except TypeError:
                ids = ()
            if ids and all(type(token) is int and 0 <= token < vocab_size for token in ids):
                checks["boundary_ids"] = {
                    "status": "ok",
                    "detail": f"{len(ids)} ids within vocabulary",
                }
            else:
                checks["boundary_ids"] = {"status": "failed", "detail": "invalid boundary ids"}

    if boundary_map is not None:
        if type(boundary_map) is not dict:
            checks["boundary_map"] = {"status": "failed", "detail": "invalid boundary map"}
        elif vocab_size is None:
            for name in boundary_map:
                checks[f"boundary_ids:{name}"] = {
                    "status": "skipped",
                    "detail": "vocabulary size unknown without the 'hf' extra",
                }
        else:
            for name, mapped in boundary_map.items():
                try:
                    ids = tuple(mapped)
                except TypeError:
                    ids = ()
                label = f"boundary_ids:{name}"
                if (
                    type(name) is str
                    and name
                    and ids
                    and all(type(token) is int and 0 <= token < vocab_size for token in ids)
                ):
                    checks[label] = {
                        "status": "ok",
                        "detail": f"{len(ids)} ids within vocabulary",
                    }
                else:
                    checks[label] = {"status": "failed", "detail": "invalid boundary ids"}

    verdict = "failed" if any(c["status"] == "failed" for c in checks.values()) else "ok"
    return _o_model_result(
        verdict,
        checks,
        assets=manifest,
        special_tokens=specials,
        added_tokens=added_tokens,
        context_window=context_window,
    )


def oracle_template_conformance_status(model_path: str | Path) -> tuple[str, str]:
    """Oracle (status, detail) for template_conformance's documented outcomes.

    Per the docstring: a missing template reports "no_template_declared"; any
    other problem reports "unavailable" with a fixed-vocabulary detail.
    """
    try:
        root = _o_directory(model_path)
        try:
            config_doc = _o_read_json(_o_member(root, "tokenizer_config.json"))
        except ValueError:
            return ("no_template_declared", "tokenizer_config_unreadable")
        template = config_doc.get("chat_template")
        if type(template) is not str or not template.strip():
            return ("no_template_declared", "no_chat_template_in_config")
        if len(template) > _TEMPLATE_CAP:
            return ("unavailable", "chat_template_too_large")
        from transformers import PreTrainedTokenizerFast  # noqa: F401 -- presence probe
    except ImportError:
        return ("unavailable", "transformers_not_installed")
    except Exception:  # noqa: BLE001 -- oracle mirrors "any other problem -> unavailable"
        return ("unavailable", "tokenizer_load_failed")
    return ("loadable", "")


def oracle_parse_safetensors_header(path: str | Path) -> tuple[bool, str]:
    """Independent struct-based safetensors header parse: (ok, reason).

    This is the STRICT oracle (it also validates tensor entries and data
    ranges). The implementation under test performs format-level validation
    only (header length + JSON-object header); ``_o_weight_headers_check``
    below mirrors exactly that rule set for the differential comparison.
    """
    data = Path(path).read_bytes()
    if len(data) < 8:
        return (False, "too_short_for_header_size")
    (size,) = struct.unpack("<Q", data[:8])
    if 8 + size > len(data):
        return (False, "truncated_header")
    try:
        header = json.loads(data[8 : 8 + size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return (False, "bad_header_json")
    if type(header) is not dict:
        return (False, "bad_header_json")
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if type(entry) is not dict:
            return (False, "bad_tensor_entry")
        offsets = entry.get("data_offsets")
        if (
            type(offsets) is not list
            or len(offsets) != 2
            or any(type(v) is not int for v in offsets)
        ):
            return (False, "bad_tensor_entry")
        start, end = offsets
        if not 0 <= start <= end or 8 + size + end > len(data):
            return (False, "tensor_data_out_of_range")
        if type(entry.get("dtype")) is not str or type(entry.get("shape")) is not list:
            return (False, "bad_tensor_entry")
    return (True, "ok")


def _o_weight_headers_check(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    """Oracle for doctor's byte-level safetensors header check.

    Independent implementation of the documented rule set (mirrors
    ``doctor._check_weight_headers`` without calling it): for each
    ``*.safetensors`` asset, read the 8-byte little-endian header length
    plus the JSON header (never tensor data); the file must hold that many
    bytes, the declared size must be within the 64 MiB cap, and the header
    must parse as a JSON object. The first invalid file fails the check,
    exactly like the implementation's early return.
    """
    for name in names:
        if not name.endswith(".safetensors"):
            continue
        try:
            data = (root / name).read_bytes()
        except OSError:
            return {"status": "failed", "detail": f"invalid_safetensors_header: {name}"}
        if len(data) < 8:
            return {"status": "failed", "detail": f"invalid_safetensors_header: {name}"}
        (size,) = struct.unpack("<Q", data[:8])
        if size > 64 * 1024 * 1024 or 8 + size > len(data):
            return {"status": "failed", "detail": f"invalid_safetensors_header: {name}"}
        try:
            header = json.loads(data[8 : 8 + size].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"status": "failed", "detail": f"invalid_safetensors_header: {name}"}
        if type(header) is not dict:
            return {"status": "failed", "detail": f"invalid_safetensors_header: {name}"}
    return {"status": "ok", "detail": "safetensors headers valid"}


def oracle_longest_prompt_tokens(tokenizer: object, dataset_path: str | Path) -> int | None:
    """Oracle's own longest-prompt measurement over every catalog candidate."""
    try:
        split = load_dataset(Path(dataset_path))
        fixtures = (*split.search, *split.held_out)
        longest = 0
        for family in CATALOG:
            for mutate in (False, True):
                for fmt in candidates(family, mutate=mutate):
                    for case in fixtures:
                        try:
                            count = len(
                                tokenizer.encode(  # type: ignore[union-attr]
                                    fmt.render(tuple(case.messages)),
                                    add_special_tokens=False,
                                    truncation=False,
                                    padding=False,
                                )
                            )
                        except Exception:  # noqa: BLE001,S112 -- oracle mirrors per-pair degrade
                            continue
                        longest = max(longest, count)
        return longest if longest > 0 else None
    except Exception:  # noqa: BLE001 -- oracle mirrors the silent degradation
        return None


def oracle_suggestion(
    *,
    verdict_ok: bool,
    dataset_path: str | Path | None,
    boundary_ids: tuple[int, ...] | None,
    window: int | None,
    tokenizer: object | None,
    model_path: str | Path,
) -> dict[str, str] | None:
    """Independent reimplementation of _next_command's documented fitting."""
    if not verdict_ok or dataset_path is None:
        return None
    try:
        ids = tuple(boundary_ids or ())
    except TypeError:
        return None
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        return None
    if type(window) is not int:
        return None
    new_tokens = _DEFAULT_NEW_TOKENS
    longest: int | None = None
    if tokenizer is not None:
        longest = oracle_longest_prompt_tokens(tokenizer, dataset_path)
        if longest is not None:
            # The backend refuses generation when prompt tokens plus new
            # tokens exceed the declared window; size the suggestion to fit.
            new_tokens = max(1, min(_DEFAULT_NEW_TOKENS, window - longest))
    families = ", ".join(sorted(CATALOG))
    command = (
        "python -m model_protocol_lab run"
        f" --model {shlex.quote(str(model_path))}"
        f" --dataset {shlex.quote(str(dataset_path))}"
        f" --family FAMILY --mutate --boundary-ids {','.join(str(token) for token in ids)}"
        f" --max-new-tokens {new_tokens} --max-context {window} --device cpu"
        " --output NEW_REPORT_DIRECTORY"
    )
    note = (
        f"Replace FAMILY with one of: {families} (see the `catalog` command). "
        f"--max-context {window} is this checkpoint's declared window; "
    )
    if longest is not None:
        note += (
            f"--max-new-tokens {new_tokens} is sized so the longest fixture "
            f"prompt ({longest} tokens) fits that window; "
        )
    if longest is not None and longest >= window:
        note += (
            f"WARNING: the longest fixture prompt ({longest} tokens) already "
            f"fills the declared window ({window}) -- shorten the fixtures or "
            "use a checkpoint with a larger window before running; "
        )
    note += (
        "--output must be a fresh directory. "
        "Candidates that need a different boundary sequence take a repeatable "
        "--boundary-map NAME=ID,ID flag (see docs/USAGE.md section 4)."
    )
    return {"next_command": command, "note": note}


# ---------------------------------------------------------------------------
# Synthetic fixture builders (tmp dirs only; never download, never load weights)
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: object) -> None:
    """Write json."""
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_safetensors_file(path: Path, kind: str) -> None:
    """Hand-written safetensors files with valid / invalid byte layouts."""
    header = json.dumps(
        {
            "__metadata__": {},
            "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        }
    ).encode("utf-8")
    if kind == "valid":
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 8)
    elif kind == "truncated":
        # The 8-byte size prefix claims more bytes than the file holds.
        path.write_bytes(struct.pack("<Q", len(header) + 64) + header[: len(header) // 2])
    elif kind == "bad-magic":
        # Structurally sized prefix, but the header bytes are not JSON.
        path.write_bytes(struct.pack("<Q", 16) + b"NOT-A-JSON-HEADER!")
    else:
        raise AssertionError(f"unknown safetensors kind: {kind}")


def _write_wordlevel_tokenizer(root: Path, n_words: int = 24) -> None:
    """A genuinely loadable WordLevel tokenizer.json (tokenizers library)."""
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
    except ImportError:
        raise unittest.SkipTest("tokenizers library not installed") from None
    vocab = {f"w{i}": i for i in range(n_words)}
    vocab["[UNK]"] = n_words
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.add_special_tokens(["[UNK]", "[BOS]", "[EOS]"])
    tok.save(str(root / "tokenizer.json"))


def _write_dataset(path: Path, search_texts: list[str], heldout_texts: list[str]) -> None:
    """Write dataset."""

    def item(prefix: str, group: str, index: int, text: str) -> dict:
        """Item."""
        return {
            "id": f"{prefix}{index}",
            # Groups must not cross the search/held_out split (datasets rule).
            "group": group,
            "messages": [{"role": "user", "content": text}],
            "expected": "e",
        }

    _write_json(
        path,
        {
            "search": [item("s", "gs", i, t) for i, t in enumerate(search_texts)],
            "held_out": [item("h", "gh", i, t) for i, t in enumerate(heldout_texts)],
        },
    )


def _write_checkpoint(
    root: Path,
    *,
    config: object,
    weights: str = "valid",
    index_map: dict[str, str] | None = None,
    generation_config: object = None,
    raw_files: dict[str, bytes] | None = None,
) -> Path:
    """Assemble a synthetic checkpoint directory.

    weights: "valid" | "truncated" | "bad-magic" (single-file layout),
    "sharded" (index JSON + the shards in index_map), "none" (no weight files).
    """
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "config.json", config)
    _write_wordlevel_tokenizer(root)
    if weights in ("valid", "truncated", "bad-magic"):
        _write_safetensors_file(root / "model.safetensors", weights)
    elif weights == "sharded":
        assert index_map
        for shard in set(index_map.values()):
            _write_safetensors_file(root / shard, "valid")
        _write_json(root / "model.safetensors.index.json", {"weight_map": index_map})
    elif weights == "none":
        pass
    else:
        raise AssertionError(f"unknown weights layout: {weights}")
    if generation_config is not None:
        _write_json(root / "generation_config.json", generation_config)
    for name, data in (raw_files or {}).items():
        (root / name).write_bytes(data)
    return root


def _oracle_measurement_tokenizer(root: Path) -> object:
    """Oracle's own tokenizer instance for prompt measurement (no doctor helpers)."""
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_file=str(root / "tokenizer.json"), clean_up_tokenization_spaces=False
    )


def _build_suggestion_fixture(
    tmpdir: str, *, window_mode: str, prompt_words: int = 120
) -> tuple[Path, Path, int, int]:
    """Checkpoint + dataset with n_positions chosen relative to longest prompt.

    Returns (root, dataset_path, longest_prompt_tokens, window).
    """
    root = Path(tmpdir) / "ckpt"
    _write_checkpoint(root, config={"model_type": "gpt2", "n_positions": 10**6})
    ds = Path(tmpdir) / "ds.json"
    long_text = " ".join(f"w{i % 24}" for i in range(prompt_words))
    _write_dataset(ds, [long_text], ["w0 w1 w2"])
    longest = oracle_longest_prompt_tokens(_oracle_measurement_tokenizer(root), ds)
    assert longest is not None and longest > 0
    if window_mode == "fits":
        window = longest + 20
    elif window_mode == "over":
        window = max(1, longest - 5)
    else:
        raise AssertionError(f"unknown window_mode: {window_mode}")
    _write_json(root / "config.json", {"model_type": "gpt2", "n_positions": window})
    return root, ds, longest, window


# ---------------------------------------------------------------------------
# Differential comparison helpers
# ---------------------------------------------------------------------------


def _assert_report_matches(
    testcase: unittest.TestCase, expected: dict, actual: dict, *, label: str
) -> None:
    """Field-by-field verdict comparison; raises AssertionError on mismatch."""
    testcase.assertEqual(actual["verdict"], expected["verdict"], f"{label}: verdict")
    testcase.assertEqual(set(actual["model"]), set(expected["model"]), f"{label}: model keys")
    testcase.assertEqual(actual["model"]["checks"], expected["model"]["checks"], f"{label}: checks")
    testcase.assertEqual(actual["model"]["assets"], expected["model"]["assets"], f"{label}: assets")
    testcase.assertEqual(
        actual["model"]["special_tokens"],
        expected["model"]["special_tokens"],
        f"{label}: special_tokens",
    )
    testcase.assertEqual(
        actual["model"]["added_tokens"],
        expected["model"]["added_tokens"],
        f"{label}: added_tokens",
    )
    testcase.assertEqual(
        actual["model"]["context_window"],
        expected["model"]["context_window"],
        f"{label}: context_window",
    )


def _suggestion_inputs(
    root: Path, boundary_ids: tuple[int, ...] | None
) -> tuple[dict, object | None]:
    """Oracle examine report + oracle measurement tokenizer for suggestion tests."""
    o_report = oracle_examine_model(root, boundary_ids=boundary_ids)
    deep_ok = o_report["model"]["checks"].get("tokenizer_deep", {}).get("status") == "ok"
    tokenizer = _oracle_measurement_tokenizer(root) if deep_ok else None
    return o_report, tokenizer


def _expected_suggestion(
    o_report: dict,
    root: Path,
    ds: Path | None,
    boundary_ids: tuple[int, ...] | None,
    tokenizer: object | None,
) -> dict[str, str] | None:
    """Expected suggestion."""
    return oracle_suggestion(
        verdict_ok=o_report["verdict"] == "ok",
        dataset_path=ds,
        boundary_ids=boundary_ids,
        window=o_report["model"]["context_window"],
        tokenizer=tokenizer,
        model_path=root,
    )


# ---------------------------------------------------------------------------
# Mutant-package machinery for sensitivity controls
# ---------------------------------------------------------------------------

_MUTANT_PKG = "mplmut25"


def _make_mutant_package(tmpdir: str, mutations: list[tuple[str, str, str]]) -> Path:
    """Copy the package to a tmp dir under a fresh top-level name and mutate it."""
    src = Path(__file__).resolve().parent.parent / "model_protocol_lab"
    pkgdir = Path(tmpdir) / "mutpkg"
    shutil.copytree(src, pkgdir / _MUTANT_PKG, ignore=shutil.ignore_patterns("__pycache__"))
    for relpath, old, new in mutations:
        target = pkgdir / _MUTANT_PKG / relpath
        text = target.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise AssertionError(f"mutation anchor not unique in {relpath}: {count}")
        target.write_text(text.replace(old, new), encoding="utf-8")
    return pkgdir


def _import_mutant_doctor(pkgdir: Path):
    """Import mutant doctor."""
    sys.path.insert(0, str(pkgdir))
    try:
        return importlib.import_module(f"{_MUTANT_PKG}.doctor")
    finally:
        sys.path.remove(str(pkgdir))


def _cleanup_mutant_modules() -> None:
    """Cleanup mutant modules."""
    for name in [n for n in sys.modules if n == _MUTANT_PKG or n.startswith(_MUTANT_PKG + ".")]:
        del sys.modules[name]


class SafetensorsHeaderOracleTests(unittest.TestCase):
    """Sanity checks for the oracle's own struct-based header parser."""

    def test_valid_header_parses(self):
        """Test valid header parses."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.safetensors"
            _write_safetensors_file(path, "valid")
            self.assertEqual(oracle_parse_safetensors_header(path), (True, "ok"))

    def test_truncated_header_detected(self):
        """Test truncated header detected."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.safetensors"
            _write_safetensors_file(path, "truncated")
            self.assertEqual(oracle_parse_safetensors_header(path), (False, "truncated_header"))

    def test_bad_magic_detected(self):
        """Test bad magic detected."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.safetensors"
            _write_safetensors_file(path, "bad-magic")
            self.assertEqual(oracle_parse_safetensors_header(path), (False, "bad_header_json"))

    def test_short_file_detected(self):
        """Test short file detected."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.safetensors"
            path.write_bytes(b"\x01\x02")
            self.assertEqual(
                oracle_parse_safetensors_header(path), (False, "too_short_for_header_size")
            )


class ModelDifferentialTests(unittest.TestCase):
    """examine_model verdicts vs the independent oracle, field-by-field."""

    def test_valid_single_file_checkpoint(self):
        """Test valid single file checkpoint."""
        with TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), config={"model_type": "gpt2"})
            expected = oracle_examine_model(root, boundary_ids=(0, 1), boundary_map={"a": (0,)})
            actual = doctor_mod.examine_model(root, boundary_ids=(0, 1), boundary_map={"a": (0,)})
            _assert_report_matches(self, expected, actual, label="valid")
            self.assertEqual(actual["verdict"], "ok")
            self.assertEqual(
                actual["model"]["checks"]["asset_layout"],
                {"status": "ok", "detail": "2 assets"},
            )

    def test_truncated_safetensors_header_fails_weight_headers(self):
        """Byte-level header validation: a truncated header still passes the
        filename-level asset_layout check, but now fails weight_headers.
        """
        with TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), config={"model_type": "gpt2"}, weights="truncated")
            header_ok, reason = oracle_parse_safetensors_header(root / "model.safetensors")
            self.assertEqual((header_ok, reason), (False, "truncated_header"))
            expected = oracle_examine_model(root)
            actual = doctor_mod.examine_model(root)
            _assert_report_matches(self, expected, actual, label="truncated-header")
            self.assertEqual(actual["verdict"], "failed")
            self.assertEqual(actual["model"]["checks"]["asset_layout"]["status"], "ok")
            self.assertEqual(
                actual["model"]["checks"]["weight_headers"],
                {"status": "failed", "detail": "invalid_safetensors_header: model.safetensors"},
            )

    def test_bad_magic_safetensors_fails_weight_headers(self):
        """Byte-level header validation: non-JSON magic bytes fail weight_headers."""
        with TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), config={"model_type": "gpt2"}, weights="bad-magic")
            header_ok, reason = oracle_parse_safetensors_header(root / "model.safetensors")
            self.assertEqual((header_ok, reason), (False, "bad_header_json"))
            expected = oracle_examine_model(root)
            actual = doctor_mod.examine_model(root)
            _assert_report_matches(self, expected, actual, label="bad-magic")
            self.assertEqual(actual["verdict"], "failed")
            self.assertEqual(actual["model"]["checks"]["asset_layout"]["status"], "ok")
            self.assertEqual(
                actual["model"]["checks"]["weight_headers"],
                {"status": "failed", "detail": "invalid_safetensors_header: model.safetensors"},
            )

    def test_sharded_layout_with_index_ok(self):
        """Test sharded layout with index ok."""
        with TemporaryDirectory() as tmp:
            index_map = {
                "layer.w": "shard-a.safetensors",
                "layer.b": "shard-b.safetensors",
            }
            root = _write_checkpoint(
                Path(tmp),
                config={"model_type": "gpt2"},
                weights="sharded",
                index_map=index_map,
            )
            expected = oracle_examine_model(root)
            actual = doctor_mod.examine_model(root)
            _assert_report_matches(self, expected, actual, label="sharded-ok")
            self.assertEqual(
                [a["name"] for a in actual["model"]["assets"]],
                [
                    "config.json",
                    "model.safetensors.index.json",
                    "shard-a.safetensors",
                    "shard-b.safetensors",
                ],
            )

    def test_index_referencing_missing_shard_fails(self):
        """Test index referencing missing shard fails."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ckpt"
            root.mkdir()
            _write_json(root / "config.json", {"model_type": "gpt2"})
            _write_wordlevel_tokenizer(root)
            _write_safetensors_file(root / "shard-a.safetensors", "valid")
            _write_json(
                root / "model.safetensors.index.json",
                {
                    "weight_map": {
                        "w": "shard-a.safetensors",
                        "b": "shard-b.safetensors",
                    }
                },
            )
            expected = oracle_examine_model(root)
            self.assertEqual(
                expected["model"]["checks"]["asset_layout"],
                {"status": "failed", "detail": "missing_or_unsafe_asset"},
            )
            actual = doctor_mod.examine_model(root)
            _assert_report_matches(self, expected, actual, label="missing-shard")
            self.assertEqual(actual["verdict"], "failed")
            self.assertEqual(actual["model"]["assets"], [])

    def test_index_weight_map_variants(self):
        """Test index weight map variants."""
        cases = [
            ("list-weight-map", {"weight_map": ["x"]}, "invalid_weight_index"),
            ("missing-weight-map", {"other": 1}, "invalid_weight_index"),
            ("non-dict-index", ["w"], "invalid_asset_json"),
            (
                "path-traversal-shard",
                {"weight_map": {"w": "../evil.safetensors"}},
                "invalid_weight_index",
            ),
            (
                "non-string-shard",
                {"weight_map": {"w": 7}},
                "invalid_weight_index",
            ),
        ]
        for label, index_doc, expected_detail in cases:
            with self.subTest(label=label), TemporaryDirectory() as tmp:
                root = Path(tmp) / "ckpt"
                root.mkdir()
                _write_json(root / "config.json", {"model_type": "gpt2"})
                _write_wordlevel_tokenizer(root)
                _write_safetensors_file(root / "shard-a.safetensors", "valid")
                _write_json(root / "model.safetensors.index.json", index_doc)
                expected = oracle_examine_model(root)
                self.assertEqual(
                    expected["model"]["checks"]["asset_layout"],
                    {"status": "failed", "detail": expected_detail},
                )
                actual = doctor_mod.examine_model(root)
                _assert_report_matches(self, expected, actual, label=f"index-{label}")

    def test_layout_presence_variants(self):
        """Test layout presence variants."""
        for label, setup in (
            ("both-present", True),
            ("neither-present", False),
        ):
            with self.subTest(label=label), TemporaryDirectory() as tmp:
                root = Path(tmp) / "ckpt"
                root.mkdir()
                _write_json(root / "config.json", {"model_type": "gpt2"})
                _write_wordlevel_tokenizer(root)
                if setup:
                    _write_safetensors_file(root / "model.safetensors", "valid")
                    _write_json(
                        root / "model.safetensors.index.json",
                        {"weight_map": {"w": "model.safetensors"}},
                    )
                expected = oracle_examine_model(root)
                self.assertEqual(
                    expected["model"]["checks"]["asset_layout"],
                    {
                        "status": "failed",
                        "detail": "require_one_standard_safetensors_layout",
                    },
                )
                actual = doctor_mod.examine_model(root)
                _assert_report_matches(self, expected, actual, label=f"layout-{label}")

    def test_adapter_config_rejected(self):
        """Test adapter config rejected."""
        with TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), config={"model_type": "gpt2"})
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            expected = oracle_examine_model(root)
            self.assertEqual(
                expected["model"]["checks"]["asset_layout"],
                {"status": "failed", "detail": "adapters_not_supported"},
            )
            actual = doctor_mod.examine_model(root)
            _assert_report_matches(self, expected, actual, label="adapter")

    def test_generation_config_content_is_inert(self):
        """Documents the discrepancy: no generation_config content checks exist."""
        variants = [
            ("with-eos", {"eos_token_id": 2, "bos_token_id": 1}),
            ("without-eos", {"bos_token_id": 1}),
            ("absent", None),
        ]
        reports = {}
        for label, gen_config in variants:
            with self.subTest(label=label), TemporaryDirectory() as tmp:
                root = _write_checkpoint(
                    Path(tmp),
                    config={"model_type": "gpt2"},
                    generation_config=gen_config,
                )
                expected = oracle_examine_model(root)
                actual = doctor_mod.examine_model(root)
                _assert_report_matches(self, expected, actual, label=f"generation-config-{label}")
                self.assertEqual(actual["verdict"], "ok")
                reports[label] = actual

        # Inertness across variants: identical verdicts and check statuses --
        # the only legitimate difference is the asset count/detail and the
        # manifest listing the generation_config.json file itself (it is
        # hashed into the manifest; its *contents* change nothing).
        def normalized(report: dict) -> dict:
            """Normalized."""
            checks = {
                key: {**check, "detail": ""} if key in ("directory", "asset_layout") else check
                for key, check in report["model"]["checks"].items()
            }
            assets = sorted(
                a["name"]
                for a in report["model"]["assets"]
                if a["name"] != "generation_config.json"
            )
            return {"verdict": report["verdict"], "checks": checks, "assets": assets}

        self.assertEqual(normalized(reports["with-eos"]), normalized(reports["without-eos"]))
        self.assertEqual(normalized(reports["with-eos"]), normalized(reports["absent"]))
        self.assertEqual(
            [a["name"] for a in reports["with-eos"]["model"]["assets"]],
            ["config.json", "generation_config.json", "model.safetensors"],
        )

    def test_config_variants(self):
        """Test config variants."""

        def write_config(root: Path, kind: str) -> None:
            """Write config."""
            target = root / "config.json"
            if kind == "non-dict":
                target.write_text("[1, 2]", encoding="utf-8")
            elif kind == "bad-utf8":
                target.write_bytes(b"\xff\xfe\x00bad")
            elif kind == "unsupported-type":
                _write_json(target, {"model_type": "bert"})
            elif kind == "rejected-keys":
                _write_json(
                    target,
                    {"model_type": "gpt2", "quantization_config": {"bits": 4}},
                )
            # "missing": write nothing

        cases = [
            # (label, check key carrying the failure, expected detail)
            ("non-dict", "config", "invalid_asset_json"),
            ("bad-utf8", "config", "invalid_asset_json"),
            ("missing", "asset_layout", "missing_or_unsafe_asset"),
            ("unsupported-type", "config", "unsupported model_type: 'bert'"),
            ("rejected-keys", "config", "rejected config keys: quantization_config"),
        ]
        for label, check_key, expected_detail in cases:
            with self.subTest(label=label), TemporaryDirectory() as tmp:
                root = Path(tmp) / "ckpt"
                root.mkdir()
                _write_wordlevel_tokenizer(root)
                _write_safetensors_file(root / "model.safetensors", "valid")
                write_config(root, label)
                expected = oracle_examine_model(root)
                self.assertEqual(
                    expected["model"]["checks"][check_key],
                    {"status": "failed", "detail": expected_detail},
                )
                actual = doctor_mod.examine_model(root)
                _assert_report_matches(self, expected, actual, label=f"config-{label}")

    def test_tokenizer_file_variants(self):
        """Test tokenizer file variants."""

        def write_tokenizer(root: Path, kind: str) -> None:
            """Write tokenizer."""
            target = root / "tokenizer.json"
            if kind == "invalid-json":
                target.write_text("not json", encoding="utf-8")
            elif kind == "empty-dict":
                _write_json(target, {})
            elif kind == "stochastic":
                _write_json(target, {"model": {"type": "WordLevel", "vocab": {}, "dropout": 0.5}})
            elif kind == "oversized":
                with open(target, "wb") as handle:
                    handle.truncate(_JSON_CAP + 1)
            # "missing": write nothing

        cases = [
            ("missing", "missing_or_unsafe_asset"),
            ("invalid-json", "invalid_asset_json"),
            ("empty-dict", "tokenizer model section missing"),
            ("stochastic", "stochastic tokenizer not supported"),
            ("oversized", "oversized_asset_json"),
        ]
        for label, expected_detail in cases:
            with self.subTest(label=label), TemporaryDirectory() as tmp:
                root = Path(tmp) / "ckpt"
                root.mkdir()
                _write_json(root / "config.json", {"model_type": "gpt2"})
                _write_safetensors_file(root / "model.safetensors", "valid")
                write_tokenizer(root, label)
                expected = oracle_examine_model(root)
                self.assertEqual(
                    expected["model"]["checks"]["tokenizer_file"],
                    {"status": "failed", "detail": expected_detail},
                )
                actual = doctor_mod.examine_model(root)
                _assert_report_matches(self, expected, actual, label=f"tokenizer-{label}")

    def test_context_window_variants(self):
        """Test context window variants."""
        cases = [
            ("gpt2-window", {"model_type": "gpt2", "n_positions": 512}, 512),
            ("gpt2-no-window", {"model_type": "gpt2"}, None),
            ("gpt2-zero", {"model_type": "gpt2", "n_positions": 0}, None),
            ("gpt2-str", {"model_type": "gpt2", "n_positions": "512"}, None),
            (
                "llama-window",
                {"model_type": "llama", "max_position_embeddings": 256},
                256,
            ),
            ("llama-ignores-n-positions", {"model_type": "llama", "n_positions": 999}, None),
        ]
        for label, config, expected_window in cases:
            with self.subTest(label=label), TemporaryDirectory() as tmp:
                root = _write_checkpoint(Path(tmp), config=config)
                expected = oracle_examine_model(root)
                actual = doctor_mod.examine_model(root)
                _assert_report_matches(self, expected, actual, label=f"context-window-{label}")
                self.assertEqual(actual["model"]["context_window"], expected_window)

    def test_missing_directory(self):
        """Test missing directory."""
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope"
            expected = oracle_examine_model(missing)
            actual = doctor_mod.examine_model(missing)
            _assert_report_matches(self, expected, actual, label="missing-dir")
            self.assertEqual(
                actual["model"]["checks"]["directory"],
                {"status": "failed", "detail": "not a readable local directory"},
            )
            # str paths behave identically
            _assert_report_matches(
                self,
                oracle_examine_model(str(missing)),
                doctor_mod.examine_model(str(missing)),
                label="missing-dir-str",
            )


class WeightHeadersCheckTests(unittest.TestCase):
    """doctor's byte-level safetensors header check vs the oracle."""

    @staticmethod
    def _checkpoint(tmp: str, filename: str, payload: bytes) -> Path:
        """Checkpoint."""
        root = Path(tmp) / "ckpt"
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "config.json", {"model_type": "gpt2"})
        _write_wordlevel_tokenizer(root)
        (root / filename).write_bytes(payload)
        if filename != "model.safetensors":
            _write_json(
                root / "model.safetensors.index.json",
                {"weight_map": {"w": filename}},
            )
        return root

    @staticmethod
    def _valid_payload(header: dict) -> bytes:
        """Valid payload."""
        header_bytes = json.dumps(header).encode("utf-8")
        return struct.pack("<Q", len(header_bytes)) + header_bytes

    def test_valid_header_passes(self):
        """Test valid header passes."""
        with TemporaryDirectory() as tmp:
            root = self._checkpoint(
                tmp, "model.safetensors", self._valid_payload({"__metadata__": {}})
            )
            expected = oracle_examine_model(root)
            actual = doctor_mod.examine_model(root)
            _assert_report_matches(self, expected, actual, label="weight-headers-valid")
            self.assertEqual(
                actual["model"]["checks"]["weight_headers"],
                {"status": "ok", "detail": "safetensors headers valid"},
            )

    def test_short_file_fails(self):
        """Test short file fails."""
        with TemporaryDirectory() as tmp:
            root = self._checkpoint(tmp, "model.safetensors", b"\x01\x02")
            expected = oracle_examine_model(root)
            actual = doctor_mod.examine_model(root)
            _assert_report_matches(self, expected, actual, label="weight-headers-short")
            self.assertEqual(actual["verdict"], "failed")
            self.assertEqual(
                actual["model"]["checks"]["weight_headers"],
                {"status": "failed", "detail": "invalid_safetensors_header: model.safetensors"},
            )

    def test_oversized_declared_header_fails(self):
        """Test oversized declared header fails."""
        with TemporaryDirectory() as tmp:
            payload = struct.pack("<Q", 2**40) + b"{}"
            root = self._checkpoint(tmp, "model.safetensors", payload)
            expected = oracle_examine_model(root)
            actual = doctor_mod.examine_model(root)
            _assert_report_matches(self, expected, actual, label="weight-headers-oversized")
            self.assertEqual(actual["verdict"], "failed")

    def test_non_dict_json_header_fails(self):
        """Test non dict json header fails."""
        with TemporaryDirectory() as tmp:
            root = self._checkpoint(tmp, "model.safetensors", self._valid_payload([1, 2]))
            expected = oracle_examine_model(root)
            actual = doctor_mod.examine_model(root)
            _assert_report_matches(self, expected, actual, label="weight-headers-nondict")
            self.assertEqual(actual["verdict"], "failed")

    def test_truncated_header_body_fails(self):
        """Test truncated header body fails."""
        with TemporaryDirectory() as tmp:
            header_bytes = json.dumps({"a": 1}).encode("utf-8")
            payload = struct.pack("<Q", len(header_bytes) + 100) + header_bytes
            root = self._checkpoint(tmp, "model.safetensors", payload)
            expected = oracle_examine_model(root)
            actual = doctor_mod.examine_model(root)
            _assert_report_matches(self, expected, actual, label="weight-headers-truncated")
            self.assertEqual(actual["verdict"], "failed")

    def test_tensor_data_never_read(self):
        """Test tensor data never read."""
        # The header declares a huge tensor but no data follows: still
        # valid, because only the header is ever read (weights never load).
        with TemporaryDirectory() as tmp:
            header = {"w": {"dtype": "F32", "shape": [10**9], "data_offsets": [0, 4 * 10**9]}}
            root = self._checkpoint(tmp, "model.safetensors", self._valid_payload(header))
            expected = oracle_examine_model(root)
            actual = doctor_mod.examine_model(root)
            _assert_report_matches(self, expected, actual, label="weight-headers-nodata")
            self.assertEqual(actual["model"]["checks"]["weight_headers"]["status"], "ok")

    def test_sharded_layout_checks_every_shard(self):
        """Test sharded layout checks every shard."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ckpt"
            root.mkdir(parents=True, exist_ok=True)
            _write_json(root / "config.json", {"model_type": "gpt2"})
            _write_wordlevel_tokenizer(root)
            (root / "shard-a.safetensors").write_bytes(self._valid_payload({"__metadata__": {}}))
            (root / "shard-b.safetensors").write_bytes(b"not a header")
            _write_json(
                root / "model.safetensors.index.json",
                {"weight_map": {"a": "shard-a.safetensors", "b": "shard-b.safetensors"}},
            )
            expected = oracle_examine_model(root)
            actual = doctor_mod.examine_model(root)
            _assert_report_matches(self, expected, actual, label="weight-headers-sharded")
            self.assertEqual(actual["verdict"], "failed")
            # First invalid file in sorted order names the detail.
            self.assertEqual(
                actual["model"]["checks"]["weight_headers"],
                {"status": "failed", "detail": "invalid_safetensors_header: shard-b.safetensors"},
            )

    def test_detail_vocabulary_is_fixed(self):
        """Test detail vocabulary is fixed."""
        # No raw file bytes may reach the check detail, even for binary garbage.
        with TemporaryDirectory() as tmp:
            root = self._checkpoint(tmp, "model.safetensors", b"\xff\xfe\x00not-json-at-all")
            actual = doctor_mod.examine_model(root)
            detail = actual["model"]["checks"]["weight_headers"]["detail"]
            self.assertEqual(detail, "invalid_safetensors_header: model.safetensors")
            detail.encode("ascii")  # raises if non-ASCII slipped through


class TemplateConformanceDifferentialTests(unittest.TestCase):
    """tokenizer_config.json handling in template_conformance vs the oracle."""

    _FMT = Format(
        name="cycle25probe",
        bos="",
        user_prefix="U:",
        assistant_prefix="A:",
        turn_end="\n",
    )
    _CASE = Fixture(
        id="c1",
        messages=(Message(role="user", content="hello"),),
        expected="x",
        group="g",
    )

    @staticmethod
    def _conformance_dir(tmp: str, tokenizer_config: object) -> Path:
        """Conformance dir."""
        root = Path(tmp) / "ckpt"
        root.mkdir()
        _write_wordlevel_tokenizer(root)
        if tokenizer_config != "absent":
            if isinstance(tokenizer_config, bytes):
                (root / "tokenizer_config.json").write_bytes(tokenizer_config)
            else:
                _write_json(root / "tokenizer_config.json", tokenizer_config)
        return root

    def test_tokenizer_config_variants(self):
        """Test tokenizer config variants."""
        cases: list[tuple[str, object, tuple[str, str]]] = [
            ("missing", "absent", ("no_template_declared", "tokenizer_config_unreadable")),
            (
                "invalid-json",
                b"not json",
                ("no_template_declared", "tokenizer_config_unreadable"),
            ),
            (
                "non-dict",
                [1, 2],
                ("no_template_declared", "tokenizer_config_unreadable"),
            ),
            ("no-key", {}, ("no_template_declared", "no_chat_template_in_config")),
            (
                "null-template",
                {"chat_template": None},
                ("no_template_declared", "no_chat_template_in_config"),
            ),
            (
                "blank-template",
                {"chat_template": "   "},
                ("no_template_declared", "no_chat_template_in_config"),
            ),
            (
                "non-string-template",
                {"chat_template": 123},
                ("no_template_declared", "no_chat_template_in_config"),
            ),
            (
                "oversized-template",
                {"chat_template": "x" * (_TEMPLATE_CAP + 1)},
                ("unavailable", "chat_template_too_large"),
            ),
        ]
        for label, doc, expected in cases:
            with self.subTest(label=label), TemporaryDirectory() as tmp:
                root = self._conformance_dir(tmp, doc)
                self.assertEqual(oracle_template_conformance_status(root), expected)
                actual = doctor_mod.template_conformance(root, (self._FMT,), (self._CASE,))
                self.assertEqual((actual["status"], actual["detail"]), expected, f"variant {label}")
                self.assertEqual(actual["diffs"], [])

    def test_bad_model_path_reports_unavailable(self):
        """Per the docstring, any non-template problem reports "unavailable"."""
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope"
            expected = ("unavailable", "tokenizer_load_failed")
            self.assertEqual(oracle_template_conformance_status(missing), expected)
            actual = doctor_mod.template_conformance(missing, (self._FMT,), (self._CASE,))
            self.assertEqual((actual["status"], actual["detail"]), expected)


class SuggestionDifferentialTests(unittest.TestCase):
    """Context-window fitting of the suggested run command vs the oracle."""

    def _differential_suggestion(
        self, root: Path, ds: Path, boundary_ids: tuple[int, ...] | None, *, label: str
    ) -> dict:
        """Differential suggestion."""
        report = doctor_mod.doctor(root, dataset_path=ds, boundary_ids=boundary_ids)
        o_report, tokenizer = _suggestion_inputs(root, boundary_ids)
        _assert_report_matches(self, o_report, report, label=label)
        self.assertEqual(report["dataset"]["status"], "ok")
        expected = _expected_suggestion(o_report, root, ds, boundary_ids, tokenizer)
        self.assertEqual(report["suggestion"], expected, f"{label}: suggestion")
        return report

    def test_suggestion_fits_window(self):
        """Test suggestion fits window."""
        with TemporaryDirectory() as tmp:
            root, ds, longest, window = _build_suggestion_fixture(tmp, window_mode="fits")
            report = self._differential_suggestion(root, ds, (0, 1), label="fits")
            assert report["suggestion"] is not None
            want_new = max(1, min(64, window - longest))
            self.assertEqual(want_new, 20)
            self.assertIn(f"--max-new-tokens {want_new}", report["suggestion"]["next_command"])
            self.assertIn(f"--max-context {window}", report["suggestion"]["next_command"])
            self.assertNotIn("WARNING", report["suggestion"]["note"])

    def test_suggestion_warns_when_prompt_fills_window(self):
        """Test suggestion warns when prompt fills window."""
        with TemporaryDirectory() as tmp:
            root, ds, longest, window = _build_suggestion_fixture(tmp, window_mode="over")
            report = self._differential_suggestion(root, ds, (0, 1), label="over")
            assert report["suggestion"] is not None
            # window - longest is negative: budget collapses to 1 with a WARNING.
            self.assertEqual(max(1, min(64, window - longest)), 1)
            self.assertIn("--max-new-tokens 1", report["suggestion"]["next_command"])
            self.assertIn(f"--max-context {window}", report["suggestion"]["next_command"])
            self.assertIn("WARNING", report["suggestion"]["note"])
            self.assertIn(str(longest), report["suggestion"]["note"])

    def test_suggestion_none_without_declared_window(self):
        """Test suggestion none without declared window."""
        with TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), config={"model_type": "gpt2"})
            ds = Path(tmp) / "ds.json"
            _write_dataset(ds, ["w0 w1"], ["w2 w3"])
            report = doctor_mod.doctor(root, dataset_path=ds, boundary_ids=(0, 1))
            o_report, tokenizer = _suggestion_inputs(root, (0, 1))
            _assert_report_matches(self, o_report, report, label="no-window")
            self.assertIsNone(o_report["model"]["context_window"])
            self.assertIsNone(_expected_suggestion(o_report, root, ds, (0, 1), tokenizer))
            self.assertIsNone(report["suggestion"])

    def test_suggestion_none_on_failed_verdict(self):
        """Test suggestion none on failed verdict."""
        with TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), config={"model_type": "bert"})
            ds = Path(tmp) / "ds.json"
            _write_dataset(ds, ["w0 w1"], ["w2 w3"])
            report = doctor_mod.doctor(root, dataset_path=ds, boundary_ids=(0, 1))
            o_report, tokenizer = _suggestion_inputs(root, (0, 1))
            _assert_report_matches(self, o_report, report, label="failed-verdict")
            self.assertEqual(report["verdict"], "failed")
            self.assertIsNone(_expected_suggestion(o_report, root, ds, (0, 1), tokenizer))
            self.assertIsNone(report["suggestion"])

    def test_suggestion_none_without_dataset(self):
        """Test suggestion none without dataset."""
        with TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), config={"model_type": "gpt2", "n_positions": 512})
            report = doctor_mod.doctor(root, boundary_ids=(0, 1))
            o_report, tokenizer = _suggestion_inputs(root, (0, 1))
            _assert_report_matches(self, o_report, report, label="no-dataset")
            self.assertIsNone(_expected_suggestion(o_report, root, None, (0, 1), tokenizer))
            self.assertIsNone(report["suggestion"])

    def test_next_command_default_budget_without_tokenizer(self):
        """Without a measurable tokenizer the suggestion keeps the default 64."""
        with TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), config={"model_type": "gpt2", "n_positions": 512})
            ds = Path(tmp) / "ds.json"
            _write_dataset(ds, ["w0 w1"], ["w2 w3"])
            report = doctor_mod.examine_model(root, boundary_ids=(0, 1))
            actual = doctor_mod._next_command(root, ds, (0, 1), report, tokenizer=None)
            expected = oracle_suggestion(
                verdict_ok=True,
                dataset_path=ds,
                boundary_ids=(0, 1),
                window=512,
                tokenizer=None,
                model_path=root,
            )
            self.assertEqual(actual, expected)
            assert actual is not None
            self.assertIn("--max-new-tokens 64", actual["next_command"])
            self.assertNotIn("WARNING", actual["note"])

    def test_next_command_none_on_bad_boundary_ids(self):
        """Test next command none on bad boundary ids."""
        with TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), config={"model_type": "gpt2", "n_positions": 512})
            ds = Path(tmp) / "ds.json"
            _write_dataset(ds, ["w0 w1"], ["w2 w3"])
            report = doctor_mod.examine_model(root, boundary_ids=(0, 1))
            for label, bad_ids in (("empty", ()), ("negative", (0, -1))):
                with self.subTest(label=label):
                    self.assertIsNone(
                        doctor_mod._next_command(root, ds, bad_ids, report, tokenizer=None)
                    )
                    self.assertIsNone(
                        oracle_suggestion(
                            verdict_ok=True,
                            dataset_path=ds,
                            boundary_ids=bad_ids,
                            window=512,
                            tokenizer=None,
                            model_path=root,
                        )
                    )


class MutationTests(unittest.TestCase):
    """Sensitivity controls: mutated copies must be caught by the differential tests."""

    def _mutant(self, tmp: str, mutations: list[tuple[str, str, str]]):
        """Mutant."""
        pkgdir = _make_mutant_package(tmp, mutations)
        mutant = _import_mutant_doctor(pkgdir)
        self.addCleanup(_cleanup_mutant_modules)
        return mutant

    def test_mutation_miscomputed_new_tokens_is_caught(self):
        """Mutant: _next_command ignores the fitted budget (always 64)."""
        with TemporaryDirectory() as tmp:
            root, ds, _longest, _window = _build_suggestion_fixture(tmp, window_mode="over")
            mutant = self._mutant(
                tmp,
                [
                    (
                        "doctor.py",
                        "new_tokens = max(1, min(_DEFAULT_NEW_TOKENS, window - longest))",
                        "new_tokens = _DEFAULT_NEW_TOKENS",
                    )
                ],
            )
            report = mutant.doctor(root, dataset_path=ds, boundary_ids=(0, 1))
            o_report, tokenizer = _suggestion_inputs(root, (0, 1))
            expected = _expected_suggestion(o_report, root, ds, (0, 1), tokenizer)
            # The oracle still expects the fitted budget (1); the mutant emits 64.
            self.assertIn("--max-new-tokens 1", expected["next_command"])
            with self.assertRaises(AssertionError):
                self.assertEqual(report["suggestion"], expected)
            # Control: the mutation touches only the suggestion; model checks match.
            _assert_report_matches(self, o_report, report, label="mutant-A-control")

    def test_mutation_missing_shard_accepted_is_caught(self):
        """Mutant: _model_assets skips the existence check for weight files."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ckpt"
            root.mkdir()
            _write_json(root / "config.json", {"model_type": "gpt2"})
            _write_wordlevel_tokenizer(root)
            _write_safetensors_file(root / "shard-a.safetensors", "valid")
            _write_json(
                root / "model.safetensors.index.json",
                {"weight_map": {"w": "shard-a.safetensors", "b": "shard-b.safetensors"}},
            )
            mutant = self._mutant(
                tmp,
                [
                    (
                        "backends.py",
                        "    for name in names:\n"
                        "        _member(root, name)\n"
                        "    return tuple(sorted(names))",
                        "    for name in names:\n"
                        '        if not name.endswith(".safetensors"):\n'
                        "            _member(root, name)\n"
                        "    return tuple(sorted(names))",
                    )
                ],
            )
            expected = oracle_examine_model(root)
            self.assertEqual(
                expected["model"]["checks"]["asset_layout"],
                {"status": "failed", "detail": "missing_or_unsafe_asset"},
            )
            actual = mutant.examine_model(root)
            with self.assertRaises(AssertionError):
                _assert_report_matches(self, expected, actual, label="mutant-missing-shard")
            # Control: a fully-present layout still matches on the mutant.
            with TemporaryDirectory() as tmp2:
                valid = _write_checkpoint(
                    Path(tmp2),
                    config={"model_type": "gpt2"},
                    weights="sharded",
                    index_map={"w": "shard-a.safetensors"},
                )
                _assert_report_matches(
                    self,
                    oracle_examine_model(valid),
                    mutant.examine_model(valid),
                    label="mutant-B-control",
                )
