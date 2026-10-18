"""Pre-flight validation for local checkpoints and fixture files.

`doctor` inspects a model directory and (optionally) a fixture file without
loading model weights, executing model code, or touching the network. It is
meant to run before any experiment so configuration problems fail fast with
a clear report instead of mid-run.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shlex
import signal
import struct
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .backends import (
    _directory,
    _manifest,
    _member,
    _model_assets,
    _open_regular_read,
    _read_json,
)
from .catalog import CATALOG, candidates
from .core import (
    Fixture,
    Format,
    check_boundary_portability,
    template_diff,
    template_tokenizer_compatibility,
)
from .datasets import load_dataset
from .fixtures import _extract_special_tokens, audit_special_token_literals

_SUPPORTED_TYPES = ("gpt2", "llama")
_REJECTED_CONFIG_KEYS = (
    "auto_map",
    "quantization_config",
    "is_encoder_decoder",
    "add_cross_attention",
)
# Same 128 MB bound the backend enforces before its own HF load: the loader
# re-reads tokenizer.json after our capped _read_json parse, so the size is
# re-checked immediately before handing the path to transformers. Without
# this a swapped-in oversized tokenizer.json is parsed into memory uncapped.
_TOKENIZER_JSON_CAP = 128 * 1024 * 1024
# Bounds for the checkpoint-controlled Jinja chat template. Real templates
# are a few KB; a hostile one can render unbounded text (e.g.
# "{{ 'x' * 10**9 }}"), so the template source and its rendered output are
# both capped and oversized pairs degrade to "unavailable" instead of
# feeding a memory bomb to the encoder. The render itself runs under a
# transient address-space cap (memory bombs -> MemoryError) and a SIGALRM
# wall-clock budget (CPU-burn loops -> _RenderTimeBudgetExceeded); both
# degrade the pair to "unavailable". (Jinja still executes inside
# transformers' sandbox; resource exhaustion beyond these bounds -- e.g. on
# non-Unix platforms or off the main thread, where the OS guards cannot be
# installed -- is the same exposure as HF's own apply_chat_template. Run
# unfamiliar checkpoints with resource limits.)
_CHAT_TEMPLATE_MAX_CHARS = 1_000_000
_RENDERED_MAX_CHARS = 1_000_000
# Wall-clock budget for a single checkpoint-controlled Jinja render. The
# address-space cap in _render_with_memory_cap bounds memory bombs, but a
# template can also burn CPU without allocating (e.g.
# "{% for i in range(10**12) %}{% set x = i %}{% endfor %}"); the SIGALRM
# watchdog in _render_with_time_budget aborts such renders. 10 s matches the
# template_diff fragmentation budget; legitimate renders complete in
# milliseconds, so the budget only ever fires on hostile templates.
_RENDER_CPU_BUDGET_S = 10.0


def _tokenizer_file(root: Path) -> Path:
    """tokenizer.json member, with the pre-load size cap enforced."""
    path = _member(root, "tokenizer.json")
    if path.stat().st_size > _TOKENIZER_JSON_CAP:
        raise ValueError("oversized_tokenizer_file")
    return path


#: Fixed-vocabulary codes ``_build_primary_tokenizer`` may raise as
#: ValueError with a message that is safe to show. Anything else a loader
#: raises -- notably ``PreTrainedTokenizerFast`` parse failures, which echo
#: hostile checkpoint bytes into the exception text -- must never reach the
#: terminal and collapses to ``tokenizer_load_failed``.
_LOADER_FIXED_CODES = frozenset({"oversized_tokenizer_file"})


def _loader_failure_code(exc: BaseException) -> str:
    """Map a tokenizer-loader exception to a terminal-safe fixed code.

    Returns the exception's own message only on an exact allowlist match;
    every other failure (including non-ValueError parse blowups) maps to
    ``"tokenizer_load_failed"``. Sanitization boundary for all
    tokenizer-load call sites (debug, diff-backends): call it instead of
    interpolating the exception.
    """
    text = str(exc)
    return text if text in _LOADER_FIXED_CODES else "tokenizer_load_failed"


def _check(status: str, detail: str = "") -> dict[str, str]:
    return {"status": status, "detail": detail}


#: Cap on a safetensors JSON header read. Headers are metadata, never
#: weights; a larger declared header is not plausible (real headers are
#: kilobytes to low megabytes) and would turn the check into a memory sink.
_SAFETENSORS_HEADER_CAP = 64 * 1024 * 1024


def _check_weight_headers(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    """Byte-level validation of safetensors weight files; never loads weights.

    ``_model_assets`` is filename-level: a truncated download or a renamed
    foreign file passes the layout check. This check reads only the 8-byte
    little-endian header length plus the JSON header itself (tensor data is
    never read) and verifies the file is at least that long and the header
    parses as a JSON object. Details use the fixed error vocabulary -- raw
    file bytes never reach them (names come from ``_model_assets``, which
    restricts them to ``[A-Za-z0-9_.-]+\\.safetensors``).
    """
    for name in names:
        if not name.endswith(".safetensors"):
            continue
        try:
            # _open_regular_read (not a plain open): a FIFO swapped in for
            # the shard between validation and open must fail closed, never
            # block the opener forever.
            fd = _open_regular_read(root / name)
        except ValueError:
            return _check("failed", f"invalid_safetensors_header: {name}")
        try:
            with os.fdopen(fd, "rb") as handle:
                fd = -1  # ownership moved to the file object
                prefix = handle.read(8)
                if len(prefix) < 8:
                    return _check("failed", f"invalid_safetensors_header: {name}")
                (size,) = struct.unpack("<Q", prefix)
                if size > _SAFETENSORS_HEADER_CAP:
                    return _check("failed", f"invalid_safetensors_header: {name}")
                header_bytes = handle.read(size)
                if len(header_bytes) < size:
                    return _check("failed", f"invalid_safetensors_header: {name}")
        except OSError:
            return _check("failed", f"invalid_safetensors_header: {name}")
        finally:
            if fd != -1:
                os.close(fd)
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            # RecursionError: a deeply-nested header ([[[...]]]) exhausts the
            # parser stack -- a hostile but small file; fail closed like
            # _read_json does, never propagate.
            return _check("failed", f"invalid_safetensors_header: {name}")
        if type(header) is not dict:
            return _check("failed", f"invalid_safetensors_header: {name}")
    return _check("ok", "safetensors headers valid")


def _model_result(
    verdict: str,
    checks: dict[str, dict[str, str]],
    *,
    assets: list[dict[str, Any]] | None = None,
    special_tokens: dict[str, int] | None = None,
    added_tokens: dict[str, int] | None = None,
    context_window: int | None = None,
) -> dict[str, Any]:
    """Every examine_model return carries the same keys.

    Early-failure returns previously omitted "assets"/"special_tokens"/
    "added_tokens", so consumers can key on them unconditionally; the
    collections are simply empty when the corresponding stage never ran.
    "context_window" is the checkpoint's declared window
    (``n_positions``/``max_position_embeddings``) when the config supplied
    one, else None.
    """
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


def _portability_check(
    reference_model_path: str | Path,
    boundary_ids: tuple[int, ...] | None,
    primary_tokenizer: Any,
) -> dict[str, str]:
    """Cross-tokenizer boundary portability against a reference checkpoint.

    Decodes the boundary IDs with the primary tokenizer and re-encodes the
    marker text with the reference checkpoint's tokenizer (tokenizer only,
    never weights). Informational: "mismatch" does not fail the verdict,
    because boundary IDs are per-checkpoint by design. Never raises.
    """
    if primary_tokenizer is None:
        return _check("skipped", "primary tokenizer unavailable")
    if boundary_ids is None:
        return _check("skipped", "needs --boundary-ids")
    try:
        clean_ids = tuple(boundary_ids)
    except TypeError:
        clean_ids = ()
    if not clean_ids or any(type(token) is not int or token < 0 for token in clean_ids):
        return _check("skipped", "invalid boundary ids")
    try:
        boundary_text = primary_tokenizer.decode(list(clean_ids))
    except Exception:  # noqa: BLE001 -- auxiliary check degrades, never fails
        return _check("skipped", "boundary_decode_failed")
    if type(boundary_text) is not str or not boundary_text:
        return _check("skipped", "boundary_decode_failed")
    try:
        ref_root = _directory(reference_model_path)
        from transformers import PreTrainedTokenizerFast

        # Untyped transformers boundary: bind through a typed Callable so
        # strict mode stays clean with or without transformers installed.
        _factory: Callable[..., Any] = PreTrainedTokenizerFast
        ref_tokenizer: Any = _factory(
            tokenizer_file=str(_tokenizer_file(ref_root)),
            clean_up_tokenization_spaces=False,
        )

        def encode(text: str) -> tuple[int, ...]:
            return tuple(ref_tokenizer.encode(text, add_special_tokens=False))

        outcome = check_boundary_portability(boundary_text, clean_ids, encode)
    except Exception:  # noqa: BLE001 -- auxiliary check degrades, never fails
        return _check("skipped", "reference_tokenizer_unavailable")
    if outcome["detail"] == "identical":
        return _check("ok", "boundary ids identical under reference tokenizer")
    if outcome["detail"] in ("secondary_encode_failed", "secondary_encode_invalid"):
        return _check("skipped", "reference_tokenizer_unavailable")
    return _check(
        "mismatch",
        f"primary={outcome['primary_ids']} reference={outcome['secondary_ids']}",
    )


#: The asymmetry warning carried verbatim in every portability_matrix
#: report section and printed with the CLI table. "incompatible" is a
#: static exclusion answer; "compatible" is the absence of an exclusion,
#: never a prediction. Stated loudly because the misreading ("this format
#: will work over there") is the exact failure mode the research
#: (docs/RESEARCH_PORTABILITY_2026-10-15.md) rejected.
_PORTABILITY_ASYMMETRY = (
    "ASYMMETRY WARNING: 'incompatible' means the format cannot be this "
    "checkpoint's native format -- re-running the experiment here is "
    "scientifically pointless, and that answer cost one tokenizer load. "
    "'compatible' means the format's control tokens are merely encodable "
    "as atomic special tokens by this tokenizer; it predicts NOTHING "
    "about experimental success, which depends on the checkpoint's "
    "weights. This matrix never transfers a selection verdict across "
    "checkpoints and never emits a portability score."
)


def _secondary_registered(tokenizer: Any, doc: dict[str, Any]) -> dict[str, int]:
    """Added/special token map of a second checkpoint, literal -> id.

    Parsed from the checkpoint's own ``tokenizer.json`` document (the
    authority: transformers does not reliably surface added tokens when
    loading from ``tokenizer_file`` alone -- the same reason
    ``examine_model`` reads them from the document), supplemented with the
    loaded tokenizer's special-token attributes for literals the document
    does not list. Hostile content degrades, never raises: malformed
    entries are skipped, and a hostile ``convert_tokens_to_ids`` degrades
    per attribute.
    """
    registered: dict[str, int] = {}
    added_section = doc.get("added_tokens")
    if type(added_section) is list:
        for entry in added_section:
            if (
                type(entry) is dict
                and type(entry.get("content")) is str
                and entry["content"]
                and type(entry.get("id")) is int
                and entry["id"] >= 0
                and entry["content"] not in registered
            ):
                registered[entry["content"]] = entry["id"]
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
        if type(token) is str and token and token not in registered:
            with contextlib.suppress(Exception):
                token_id = tokenizer.convert_tokens_to_ids(token)
                if type(token_id) is int and token_id >= 0:
                    registered[token] = token_id
    return registered


def portability_matrix(second_checkpoint: str | Path) -> dict[str, Any]:
    """Static template<->tokenizer compatibility matrix for a second checkpoint.

    For every catalog candidate -- every base format plus its bounded local
    repairs (the same hypothesis space ``run --mutate`` measures), in
    deterministic family/name order -- checks the candidate's
    special-token inventory (``fixtures._extract_special_tokens``) against
    the second checkpoint's tokenizer via
    ``core.template_tokenizer_compatibility``: each inventory token must be
    registered in the checkpoint's added/special tokens, encode as exactly
    one token id, and encode to its registered id.

    Tokenizer only: never loads weights. Same cost class as the other
    doctor checks (one tokenizer load plus O(candidates x tokens) encodes),
    with the same 128 MB ``tokenizer.json`` cap and fixed-vocabulary error
    codes. Formats with an empty ``bos`` skip the BOS requirement -- a
    missing BOS token is not an incompatibility for them (their inventory
    simply contains no BOS literal); this is reported per row as
    ``bos_skipped``. A format whose inventory is empty reports "unknown",
    never "compatible".

    Never raises for checkpoint problems: an unusable second checkpoint
    yields status "unavailable" with a fixed-vocabulary detail (fail-closed;
    the matrix never fails the doctor verdict). See the "asymmetry" field
    -- stated loudly in the report, the CLI table, and docs/USAGE.md -- for
    what the verdicts do and do not mean.
    """
    if not isinstance(second_checkpoint, (str, Path)):
        raise ValueError("second_checkpoint must be a path")

    def unavailable(detail: str) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "detail": detail,
            "second_checkpoint": str(second_checkpoint),
            "asymmetry": _PORTABILITY_ASYMMETRY,
            "rows": [],
        }

    try:
        root = _directory(second_checkpoint)
    except ValueError:
        return unavailable("not_a_readable_directory")
    try:
        doc = _read_json(_tokenizer_file(root))
    except ValueError as exc:
        return unavailable(_loader_failure_code(exc))
    except OSError:
        # Unreadable tokenizer.json (e.g. permission denied for a non-root
        # user): the second checkpoint is unusable, not a crash. Fixed
        # vocabulary, per this function's "never raises" contract.
        return unavailable("missing_or_unsafe_asset")
    try:
        from transformers import PreTrainedTokenizerFast

        # Untyped transformers boundary: bind through a typed Callable so
        # strict mode stays clean with or without transformers installed.
        _factory: Callable[..., Any] = PreTrainedTokenizerFast
        tokenizer: Any = _factory(
            tokenizer_file=str(_tokenizer_file(root)),
            clean_up_tokenization_spaces=False,
        )
    except ImportError:
        return unavailable("transformers_not_installed")
    except Exception as exc:  # noqa: BLE001 -- sanitized fixed vocabulary only
        return unavailable(_loader_failure_code(exc))

    def encode(text: str) -> tuple[int, ...]:
        return tuple(tokenizer.encode(text, add_special_tokens=False))

    registered = _secondary_registered(tokenizer, doc)

    rows: list[dict[str, Any]] = []
    for family in sorted(CATALOG):
        for fmt in candidates(family, mutate=True):
            row = template_tokenizer_compatibility(
                fmt.name, _extract_special_tokens(fmt), registered, encode
            )
            # BOS-less formats (chatml/qwen3-style, bos="") carry no BOS
            # requirement: their inventory holds no BOS literal, and the
            # flag documents that the skip is by design, not an omission.
            row["bos_skipped"] = fmt.bos == ""
            rows.append(row)
    return {
        "status": "ok",
        "detail": f"{len(rows)} candidates checked (tokenizer only, no weights)",
        "second_checkpoint": str(root),
        "asymmetry": _PORTABILITY_ASYMMETRY,
        "rows": rows,
    }


# C0/C1 controls (incl. ESC), DEL, lone surrogates, and Unicode bidi
# controls are stripped from stderr-bound strings: they can inject terminal
# escape sequences or reorder displayed text (Trojan Source, CVE-2021-42574).
# Other invisible format characters (BOM, zero-width space, ...) are
# deliberately preserved for diagnostic fidelity, mirroring debug.py and
# cli.py (kept local so doctor.py stays importable without cli).
_CONTROL_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ud800-\udfff\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]"
)


def _sanitize_stderr(value: str) -> str:
    """Strip terminal-control characters from an operator-visible string."""
    return _CONTROL_CHARS.sub("", value)


def render_portability_table(section: dict[str, Any]) -> str:
    """Human-readable rendering of a ``portability_matrix`` report section.

    The JSON report stays the machine contract (printed on stdout by the
    CLI); this table is the operator-facing summary and goes to stderr so
    stdout remains parseable. The asymmetry warning is always included --
    it is the load-bearing part of the output. Every interpolated value
    is control-stripped at this render boundary: the fields are
    fixed-vocabulary or operator-supplied today, but stderr is a terminal
    and a future caller must not be able to inject escape sequences here.
    """
    second = _sanitize_stderr(str(section.get("second_checkpoint", "?")))
    lines = [f"portability matrix vs {second}"]
    if section.get("status") != "ok":
        status = _sanitize_stderr(str(section.get("status")))
        detail = _sanitize_stderr(str(section.get("detail", "")))
        lines.append(f"status: {status} ({detail})")
    else:
        rows = section.get("rows")
        rows = rows if type(rows) is list else []
        lines[0] += f": {len(rows)} candidates (tokenizer-only; no weights loaded)"
        names = [_sanitize_stderr(str(row.get("format", ""))) for row in rows if type(row) is dict]
        width = max([len(name) for name in names] + [len("format")])
        lines.append(f"{'format':<{width}}  {'verdict':<12}  {'reason':<14}".rstrip())
        for row in rows:
            if type(row) is not dict:
                continue
            line = (
                f"{_sanitize_stderr(str(row.get('format', ''))):<{width}}  "
                f"{_sanitize_stderr(str(row.get('verdict', ''))):<12}  "
                f"{_sanitize_stderr(str(row.get('reason', ''))):<14}"
            ).rstrip()
            if row.get("bos_skipped"):
                line += "  bos-skipped"
            lines.append(line)
    lines.extend(("", _PORTABILITY_ASYMMETRY))
    return "\n".join(lines)


def examine_model(
    model_path: str | Path,
    *,
    boundary_ids: tuple[int, ...] | None = None,
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    reference_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a local checkpoint directory; never loads weights.

    Weight files get byte-level header validation (the 8-byte length plus
    the JSON header only -- tensor data is never read), so a truncated
    download or a renamed foreign file fails here instead of mid-run.
    """
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
    except OSError:
        # Unreadable asset (e.g. permission denied): fail closed with the
        # fixed vocabulary, never a raw OSError.
        checks["asset_layout"] = _check("failed", "missing_or_unsafe_asset")
        return _model_result("failed", checks)
    try:
        manifest = _manifest(root, names)
    except ValueError as exc:
        # Asset enumeration succeeded, so the asset_layout stage is ok even
        # though manifesting failed (e.g. a file vanished between the two
        # stages): report both, like every other stage does.
        checks["asset_layout"] = _check("ok", f"{len(names)} assets")
        checks["asset_manifest"] = _check("failed", str(exc))
        return _model_result("failed", checks)
    except OSError:
        checks["asset_layout"] = _check("ok", f"{len(names)} assets")
        checks["asset_manifest"] = _check("failed", "missing_or_unsafe_asset")
        return _model_result("failed", checks)
    checks["asset_layout"] = _check("ok", f"{len(names)} assets")
    checks["asset_manifest"] = _check("ok", "sha256 recorded")
    checks["weight_headers"] = _check_weight_headers(root, names)
    if checks["weight_headers"]["status"] == "failed":
        return _model_result("failed", checks, assets=manifest)

    try:
        config = _read_json(_member(root, "config.json"))
    except ValueError as exc:
        checks["config"] = _check("failed", str(exc))
        return _model_result("failed", checks, assets=manifest)
    except OSError:
        # Unreadable config.json (e.g. permission denied for a non-root
        # user): fail closed with the fixed vocabulary, never a raw OSError.
        checks["config"] = _check("failed", "missing_or_unsafe_asset")
        return _model_result("failed", checks, assets=manifest)
    model_type = config.get("model_type")
    rejected = [key for key in _REJECTED_CONFIG_KEYS if config.get(key)]
    if model_type not in _SUPPORTED_TYPES:
        # model_type is checkpoint-controlled and unbounded (the asset cap is
        # 128MB); truncate the preview so a hostile config cannot bloat the
        # machine-readable report. The fixed code carries the verdict.
        preview = repr(model_type)
        if len(preview) > 200:
            preview = preview[:200] + "..."
        checks["config"] = _check("failed", f"unsupported model_type: {preview}")
    elif rejected:
        checks["config"] = _check("failed", f"rejected config keys: {', '.join(rejected)}")
    else:
        checks["config"] = _check("ok", f"model_type={model_type}")
    if checks["config"]["status"] == "failed":
        return _model_result("failed", checks, assets=manifest)
    # The declared context window feeds the backend's max_context gate and
    # the post-check `run` suggestion; a missing or malformed value leaves
    # it unknown rather than failing the check.
    declared = (
        config.get("n_positions") if model_type == "gpt2" else config.get("max_position_embeddings")
    )
    context_window = declared if type(declared) is int and declared > 0 else None

    try:
        tokenizer_doc = _read_json(_member(root, "tokenizer.json"))
    except ValueError as exc:
        checks["tokenizer_file"] = _check("failed", str(exc))
        return _model_result("failed", checks, assets=manifest, context_window=context_window)
    except OSError:
        # Unreadable tokenizer.json: fail closed with the fixed vocabulary.
        checks["tokenizer_file"] = _check("failed", "missing_or_unsafe_asset")
        return _model_result("failed", checks, assets=manifest, context_window=context_window)
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
    primary_tokenizer: Any = None
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
        tokenizer = _build_primary_tokenizer(root)
        primary_tokenizer = tokenizer
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
    except ImportError:
        checks["tokenizer_deep"] = _check(
            "skipped", "transformers not installed; install the 'hf' extra for deep checks"
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

    if boundary_map is not None:
        # Per-candidate boundaries (Cycle 2): every mapped candidate's IDs
        # are verified against the same vocabulary, reported under a
        # per-name check so one bad mapping cannot hide behind the default.
        if type(boundary_map) is not dict:
            checks["boundary_map"] = _check("failed", "invalid boundary map")
        elif vocab_size is None:
            for name in boundary_map:
                checks[f"boundary_ids:{name}"] = _check(
                    "skipped", "vocabulary size unknown without the 'hf' extra"
                )
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
                    checks[label] = _check("ok", f"{len(ids)} ids within vocabulary")
                else:
                    checks[label] = _check("failed", "invalid boundary ids")

    if reference_model_path is not None:
        checks["boundary_portability"] = _portability_check(
            reference_model_path, boundary_ids, primary_tokenizer
        )

    verdict = "failed" if any(c["status"] == "failed" for c in checks.values()) else "ok"
    return _model_result(
        verdict,
        checks,
        assets=manifest,
        special_tokens=specials,
        added_tokens=added_tokens,
        context_window=context_window,
    )


def examine_dataset(path: str | Path) -> dict[str, Any]:
    """Validate a fixture file; reports counts plus hygiene audits.

    The special-token audit scans fixture message text for special-token
    literals that collide with the catalog formats' special tokens (advisory
    GHSA-g5f9-3xfg-p9mf: such a literal can forge a role boundary when the
    fixture is composed into a prompt). The contamination report measures
    search/held-out near-duplicate overlap (see contamination.py). Both are
    reported with status ``"warning"`` / flagged pairs and never fail the
    verdict -- fixture hygiene is advisory, exactly as in ``fixtures
    build``.
    """
    try:
        # a bounded local JSON reader; not huggingface_hub.datasets.
        split = load_dataset(path)  # nosec B615 -- own bounded JSON reader, not HF datasets
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return {
            "verdict": "failed",
            "dataset": {"status": "failed", "detail": f"{type(exc).__name__}: {exc}"},
        }
    findings = audit_special_token_literals(split, tuple(CATALOG.values()))
    audit: dict[str, Any] = {
        "status": "warning" if findings else "ok",
        "detail": (
            f"{len(findings)} special-token literal(s) in fixture text; a literal "
            "that collides with a candidate format's special token can forge a "
            "role boundary when the fixture is composed into a prompt "
            "(advisory GHSA-g5f9-3xfg-p9mf)"
            if findings
            else "no special-token literals in fixture text"
        ),
        "findings": [
            {
                "fixture_id": finding.fixture_id,
                "field": finding.field,
                "literal": finding.literal,
                "format_name": finding.format_name,
                "offset": finding.offset,
            }
            for finding in findings
        ],
    }
    from .contamination import contamination_report

    return {
        "verdict": "ok",
        "dataset": {
            "status": "ok",
            "search_cases": len(split.search),
            "held_out_cases": len(split.held_out),
            "special_token_audit": audit,
            "contamination": contamination_report(split.search, split.held_out),
        },
    }


def _render_with_memory_cap(tokenizer: Any, conversation: list[dict[str, str]]) -> str:
    """Render ``apply_chat_template`` with a transient address-space cap.

    A hostile ``chat_template`` (e.g. ``{{ 'x' * 10**11 }}``) materializes
    its full output *before* the ``_RENDERED_MAX_CHARS`` length check can
    degrade the pair -- the 1MB template-source cap does not bound the
    *rendered* size, so a 26-byte template can transiently allocate
    gigabytes and OOM-kill the process. This bounds the render itself: a
    temporary ``RLIMIT_AS`` of (current virtual size + 256MB headroom) is
    installed around the single render call, so a bomb raises ``MemoryError``
    (caught by the caller's ``except Exception`` -> ``"unavailable"``)
    instead of killing the process. Legitimate renders peak far below the
    headroom (output is capped at 1MB chars by the post-render check).

    On platforms without ``resource(3)``/``/proc`` (non-Unix) or when the
    limit cannot be installed, the render runs unbounded -- the pre-existing
    behavior -- and the post-render length check still applies. An already
    tighter ambient limit is respected, never loosened.
    """

    def render() -> str:
        rendered = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        if type(rendered) is not str:
            raise ValueError("chat_template_did_not_render_text")
        return rendered

    try:
        import resource
    except ImportError:
        return render()
    vmsize: int | None = None
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmSize:"):
                    vmsize = int(line.split()[1]) * 1024
                    break
    except OSError:
        vmsize = None
    if vmsize is None:
        return render()
    try:
        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    except OSError:
        return render()
    if old_soft != resource.RLIM_INFINITY:
        # A tighter ambient limit already governs; respect it, don't touch.
        return render()
    new_soft = vmsize + 256 * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, old_hard))
    except (OSError, ValueError):
        return render()
    try:
        return render()
    finally:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (old_soft, old_hard))
        except _RenderTimeBudgetExceeded:
            # The watchdog alarm landed mid-restore. The one-shot alarm is
            # consumed, so retry the restore, then re-raise: the render's
            # budget expiry must reach the caller (which degrades the pair
            # to "unavailable"). Swallowing it here -- the budget exception
            # is a ValueError subclass, exactly what the cleanup guard
            # below tolerates -- would leave the address-space cap lowered
            # process-wide with nothing signaling the corruption.
            with contextlib.suppress(OSError, ValueError):
                resource.setrlimit(resource.RLIMIT_AS, (old_soft, old_hard))
            raise
        except (OSError, ValueError):
            pass


class _RenderTimeBudgetExceeded(ValueError):
    """A checkpoint's Jinja chat template exceeded the render CPU budget.

    A ``ValueError`` -- not ``KeyboardInterrupt`` -- so the per-pair handler
    in :func:`template_conformance` degrades the pair to ``"unavailable"``
    exactly like the memory-bomb path, and a budget expiry is never mistaken
    for a user interrupt.
    """


def _render_with_time_budget(
    tokenizer: Any,
    conversation: list[dict[str, str]],
    budget_s: float = _RENDER_CPU_BUDGET_S,
) -> str:
    """Render with a wall-clock budget around :func:`_render_with_memory_cap`.

    The address-space cap bounds memory bombs, but a hostile ``chat_template``
    can burn CPU without allocating -- e.g.
    ``{% for i in range(10**12) %}{% set x = i %}{% endfor %}``. On Unix, in
    the main thread, a one-shot ``SIGALRM`` (``setitimer(ITIMER_REAL)``)
    bounds the render: Jinja compiles templates to Python bytecode, so the
    interpreter services the signal promptly and the render aborts with
    :class:`_RenderTimeBudgetExceeded`, which the caller's ``except
    Exception`` degrades to ``"unavailable"``.

    The previous ``SIGALRM`` handler and timer are saved and restored, so a
    caller-owned alarm is never lost (its remaining time may shift by the
    render duration -- acceptable for a lab tool). A non-positive or
    non-finite ``budget_s`` disables the watchdog (documented escape hatch;
    ``inf``/``nan`` cannot arm ``setitimer``). Outside the main thread or on
    platforms without ``setitimer`` (Windows), the render runs unbounded --
    the pre-existing behavior -- and only the memory cap and the post-render
    length check apply. Teardown blocks ``SIGALRM`` while the timer is
    disarmed and the previous handler restored, so an alarm that becomes
    pending in that window cannot fire mid-restore (the budget exception is
    a ``ValueError``, which the cleanup guards would otherwise swallow,
    leaving the handler inconsistent).
    """
    if budget_s <= 0 or not math.isfinite(budget_s):
        return _render_with_memory_cap(tokenizer, conversation)
    if not hasattr(signal, "setitimer"):
        return _render_with_memory_cap(tokenizer, conversation)
    if threading.current_thread() is not threading.main_thread():
        # SIGALRM is delivered to the main thread only; arming a timer here
        # would never fire.
        return _render_with_memory_cap(tokenizer, conversation)

    def _on_alarm(signum: int, frame: Any) -> None:
        raise _RenderTimeBudgetExceeded(f"chat template render exceeded {budget_s:g}s CPU budget")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_delay, old_interval = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
    except (OSError, ValueError, RuntimeError):
        return _render_with_memory_cap(tokenizer, conversation)
    try:
        signal.setitimer(signal.ITIMER_REAL, budget_s)
    except (OSError, ValueError):
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            signal.signal(signal.SIGALRM, old_handler)
        return _render_with_memory_cap(tokenizer, conversation)
    try:
        return _render_with_memory_cap(tokenizer, conversation)
    finally:
        # Block SIGALRM across teardown: disarming the timer does not retract
        # an already-pending alarm, which would otherwise fire mid-restore.
        # A pending alarm delivered here is deferred until the mask is lifted
        # and then reaches the restored previous handler -- honest Unix
        # semantics, and the handler/timer restoration stays atomic.
        try:
            saved_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        except (AttributeError, OSError, ValueError, RuntimeError):
            saved_mask = None
        try:
            with contextlib.suppress(OSError, ValueError):
                signal.setitimer(signal.ITIMER_REAL, 0)
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                signal.signal(signal.SIGALRM, old_handler)
            if old_delay > 0:
                with contextlib.suppress(OSError, ValueError):
                    signal.setitimer(signal.ITIMER_REAL, old_delay, old_interval)
        finally:
            if saved_mask is not None:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    signal.pthread_sigmask(signal.SIG_SETMASK, saved_mask)


def template_firewall(model_path: str | Path) -> dict[str, Any]:
    """Static firewall scan of the checkpoint's ``chat_template``.

    Reads ``tokenizer_config.json``, extracts the ``chat_template`` source,
    and runs :func:`template_audit.scan_template_ast` — *before* any render.
    Returns ``{"status": "allowed"|"denied"|"no_template"|"unavailable",
    "findings": [...]}`` with fixed-vocabulary findings. Never raises and
    never renders: a denied template must not reach the Jinja compiler.
    """
    from .template_audit import scan_template_ast

    try:
        root = _directory(model_path)
        try:
            config_doc = _read_json(_member(root, "tokenizer_config.json"))
        except ValueError:
            return {
                "status": "unavailable",
                "detail": "tokenizer_config_unreadable",
                "findings": [],
            }
        template = config_doc.get("chat_template")
        if type(template) is not str or not template.strip():
            return {
                "status": "no_template",
                "detail": "no_chat_template_in_config",
                "findings": [],
            }
        if len(template) > _CHAT_TEMPLATE_MAX_CHARS:
            return {
                "status": "unavailable",
                "detail": "chat_template_too_large",
                "findings": [],
            }
    except Exception:  # noqa: BLE001 -- firewall must not fail the run
        return {"status": "unavailable", "detail": "firewall_check_failed", "findings": []}
    findings = scan_template_ast(template)
    if findings:
        return {
            "status": "denied",
            "detail": "template_firewall_denied",
            "findings": [{"rule": f.rule, "detail": f.detail} for f in findings],
        }
    return {"status": "allowed", "detail": "no_rce_primitives_found", "findings": []}


def template_conformance(
    model_path: str | Path,
    formats: tuple[Format, ...],
    cases: tuple[Fixture, ...],
) -> dict[str, Any]:
    """Diff hand-rendered hypotheses against the checkpoint's own chat template.

    Loads only the tokenizer (never model weights): reads the Jinja chat
    template from ``tokenizer_config.json``, renders it over each case's
    messages with ``add_generation_prompt=True`` (the template analogue of
    the lab's open-assistant-tail render), encodes both sides with
    ``add_special_tokens=False``, and reports the token-ID diff per
    (format, fixture) via ``core.template_diff``. ``tokenize=False`` plus an
    explicit ``encode`` is used instead of ``apply_chat_template``'s
    ``tokenize=True`` so the return type does not depend on the transformers
    version. Never raises: a missing template reports
    ``"no_template_declared"``; any other problem reports ``"unavailable"``
    with a fixed-vocabulary detail, so a run is never failed by this check.
    """
    if (
        type(formats) is not tuple
        or not formats
        or any(type(fmt) is not Format for fmt in formats)
        or type(cases) is not tuple
        or not cases
        or any(type(case) is not Fixture for case in cases)
    ):
        raise ValueError("template_conformance needs nonempty tuples of Format and Fixture")
    try:
        root = _directory(model_path)
        try:
            config_doc = _read_json(_member(root, "tokenizer_config.json"))
        except ValueError:
            return {
                "status": "no_template_declared",
                "detail": "tokenizer_config_unreadable",
                "diffs": [],
            }
        template = config_doc.get("chat_template")
        if type(template) is not str or not template.strip():
            return {
                "status": "no_template_declared",
                "detail": "no_chat_template_in_config",
                "diffs": [],
            }
        if len(template) > _CHAT_TEMPLATE_MAX_CHARS:
            return {
                "status": "unavailable",
                "detail": "chat_template_too_large",
                "diffs": [],
            }
        # Firewall: static AST scan before the template reaches any Jinja
        # compiler. A denied template is never rendered (fail closed).
        from .template_audit import scan_template_ast

        firewall_findings = scan_template_ast(template)
        if firewall_findings:
            return {
                "status": "denied",
                "detail": "template_firewall_denied",
                "findings": [{"rule": f.rule, "detail": f.detail} for f in firewall_findings],
                "diffs": [],
            }
        from transformers import PreTrainedTokenizerFast

        # Untyped transformers boundary: bind through a typed Callable so
        # strict mode stays clean with or without transformers installed.
        _factory: Callable[..., Any] = PreTrainedTokenizerFast
        tokenizer: Any = _factory(
            tokenizer_file=str(_tokenizer_file(root)),
            clean_up_tokenization_spaces=False,
        )
        tokenizer.chat_template = template
        # The template may reference bos_token; take it from the same config
        # document when it is a plain string (or a {"content": ...} mapping).
        bos_token = config_doc.get("bos_token")
        if type(bos_token) is dict:
            bos_token = bos_token.get("content")
        if type(bos_token) is str and bos_token:
            # The template may not reference bos_token; a setter failure
            # must not fail the check.
            with contextlib.suppress(Exception):
                tokenizer.bos_token = bos_token
    except ImportError:
        return {"status": "unavailable", "detail": "transformers_not_installed", "diffs": []}
    except Exception:  # noqa: BLE001 -- conformance must not fail the run
        return {"status": "unavailable", "detail": "tokenizer_load_failed", "diffs": []}

    diffs: list[dict[str, Any]] = []
    for fmt in formats:
        for case in cases:
            entry: dict[str, Any] = {
                "format": fmt.name,
                "fixture": case.id,
                "status": "ok",
                "diff": [],
                "hand_tokens": 0,
                "template_tokens": 0,
            }
            try:
                hand_ids = tuple(
                    tokenizer.encode(fmt.render(case.messages), add_special_tokens=False)
                )
                rendered = _render_with_time_budget(
                    tokenizer,
                    [{"role": m.role, "content": m.content} for m in case.messages],
                )
                if len(rendered) > _RENDERED_MAX_CHARS:
                    # Checkpoint-controlled template rendered unbounded text;
                    # degrade the pair instead of encoding a memory bomb.
                    # (The render itself was already memory-capped, so this
                    # only sees outputs that fit comfortably.)
                    entry["status"] = "unavailable"
                else:
                    template_ids = tuple(tokenizer.encode(rendered, add_special_tokens=False))
                    entry["diff"] = list(template_diff(hand_ids, template_ids))
                    entry["hand_tokens"] = len(hand_ids)
                    entry["template_tokens"] = len(template_ids)
            except Exception:  # noqa: BLE001 -- one bad pair degrades, not the run
                entry["status"] = "unavailable"
            diffs.append(entry)
    return {"status": "ok", "diffs": diffs}


def marker_isolation(
    model_path: str | Path,
    formats: tuple[Format, ...],
    cases: tuple[Fixture, ...],
) -> dict[str, Any]:
    """Chat-marker isolation conformance: do hostile marker strings stay inert?

    For each format x fixture, renders the benign prompt and re-renders it
    with each of :data:`control_tokens.HOSTILE_MARKER_STRINGS` injected into
    the user text, then encodes both with the checkpoint tokenizer (never
    model weights) using ``add_special_tokens=False`` — the same call the
    backend's ``encode`` makes. Counts ids in the tokenizer's special set
    (``all_special_ids`` union special-flagged added tokens) before and
    after: the number should not change. A nonzero delta means the hostile
    text's literal markers became real
    control tokens (the ChatInject/Phantom structural-injection class, via
    HF's ``split_special_tokens=False`` default).

    Never raises: missing/unreadable tokenizers degrade like
    :func:`template_conformance`; a tokenizer declaring no special ids
    reports ``"no_special_ids"`` (unmeasurable, not a failure); one bad
    format/fixture/marker triple degrades to ``"unavailable"`` without
    failing the rest.
    """
    from .control_tokens import (
        HOSTILE_MARKER_STRINGS,
        check_isolation,
        count_control_tokens,
        inject_hostile,
        summarize_probes,
    )

    if (
        type(formats) is not tuple
        or not formats
        or any(type(fmt) is not Format for fmt in formats)
        or type(cases) is not tuple
        or not cases
        or any(type(case) is not Fixture for case in cases)
    ):
        raise ValueError("marker_isolation needs nonempty tuples of Format and Fixture")
    try:
        root = _directory(model_path)
        try:
            tokenizer: Any = _build_primary_tokenizer(root)
        except ImportError:
            return {"status": "unavailable", "detail": "transformers_not_installed", "probes": []}
        raw_special = tokenizer.all_special_ids
        special_ids = (
            {int(i) for i in raw_special}
            if isinstance(raw_special, (list, tuple, set, frozenset))
            else set()
        )

        # Some tokenizers under-declare all_special_ids while the
        # tokenizer.json marks added tokens as special; union both so the
        # probe measures what the tokenizer actually treats as special.
        def _special_id(pair: Any) -> int | None:
            _id, _tok = pair
            if not getattr(_tok, "special", False):
                return None
            try:
                return int(_id)
            except (TypeError, ValueError):
                return None

        try:
            decoder = tokenizer.added_tokens_decoder
            items = decoder.items() if hasattr(decoder, "items") else []
        except Exception:  # noqa: BLE001 -- best effort, all_special_ids suffices
            items = []
        for pair in items:
            sid = _special_id(pair)
            if sid is not None:
                special_ids.add(sid)
    except Exception:  # noqa: BLE001 -- conformance must not fail the run
        return {"status": "unavailable", "detail": "tokenizer_load_failed", "probes": []}
    if not special_ids:
        return {
            "status": "no_special_ids",
            "detail": "tokenizer_declares_no_special_ids",
            "probes": [],
        }

    def _encode(text: str) -> tuple[int, ...]:
        # Same call the backend's encode makes: no added specials, so any
        # control token in the output came from the text itself.
        return tuple(tokenizer.encode(text, add_special_tokens=False))

    probes: list[dict[str, Any]] = []
    for fmt in formats:
        for case in cases:
            try:
                benign_ids = _encode(fmt.render(case.messages))
                baseline = count_control_tokens(benign_ids, special_ids)
            except Exception:  # noqa: BLE001 -- one bad pair degrades, not the run
                for marker in HOSTILE_MARKER_STRINGS:
                    probes.append(
                        {
                            "format": fmt.name,
                            "fixture": case.id,
                            "marker": marker,
                            "status": "unavailable",
                        }
                    )
                continue
            for marker in HOSTILE_MARKER_STRINGS:
                entry: dict[str, Any] = {
                    "format": fmt.name,
                    "fixture": case.id,
                    "marker": marker,
                    "status": "ok",
                    "baseline": baseline,
                }
                try:
                    hostile_ids = _encode(fmt.render(inject_hostile(case.messages, marker)))
                    probe = check_isolation(benign_ids, hostile_ids, special_ids)
                    entry["probed"] = probe.probed
                    entry["delta"] = probe.delta
                    entry["isolated"] = probe.isolated
                except Exception:  # noqa: BLE001 -- one bad triple degrades
                    entry["status"] = "unavailable"
                probes.append(entry)

    payload = summarize_probes(probes, len(special_ids))
    return payload


def _build_primary_tokenizer(root: Path) -> Any:
    """Load the checkpoint's tokenizer for measurement (never weights).

    Raises ImportError when transformers is unavailable, anything else when
    the tokenizer cannot be built. Callers decide the check status; the
    128 MB pre-load cap from ``_tokenizer_file`` applies.
    """
    from transformers import PreTrainedTokenizerFast

    # Untyped transformers boundary: bind through a typed Callable so
    # strict mode stays clean with or without transformers installed.
    _factory: Callable[..., Any] = PreTrainedTokenizerFast
    result: Any = _factory(
        tokenizer_file=str(_tokenizer_file(root)),
        clean_up_tokenization_spaces=False,
    )
    return result


_DEFAULT_NEW_TOKENS = 64


def _prompt_token_count(tokenizer: Any, fmt: Any, case: Any) -> int | None:
    """Encode one rendered fixture prompt; None when it cannot be measured."""
    try:
        text = fmt.render(tuple(case.messages))
        return len(
            tokenizer.encode(text, add_special_tokens=False, truncation=False, padding=False)
        )
    except Exception:  # noqa: BLE001 -- one bad pair degrades
        return None


def _longest_prompt_tokens(tokenizer: Any, dataset_path: str | Path) -> int | None:
    """Longest fixture prompt in tokens, over every catalog candidate.

    Renders each fixture's messages with every candidate of every format
    family (both mutate states) and encodes with the checkpoint tokenizer,
    so a budget derived from it fits whichever family the operator later
    picks. Mirrors the backend's ``encode`` (no special tokens added).
    Returns None when it cannot be measured; the caller keeps the default
    budget.
    """
    try:
        from .catalog import CATALOG, candidates
        from .datasets import load_dataset

        split = load_dataset(Path(dataset_path))  # nosec B615 -- own bounded JSON reader, not HF datasets
        fixtures = (*split.search, *split.held_out)
        longest = 0
        for family in CATALOG:
            for mutate in (False, True):
                for fmt in candidates(family, mutate=mutate):
                    for case in fixtures:
                        count = _prompt_token_count(tokenizer, fmt, case)
                        if count is not None:
                            longest = max(longest, count)
        return longest if longest > 0 else None
    except Exception:  # noqa: BLE001 -- measurement aid; degrade silently
        return None


def _next_command(
    model_path: str | Path,
    dataset_path: str | Path | None,
    boundary_ids: tuple[int, ...] | None,
    report: dict[str, Any],
    *,
    tokenizer: Any = None,
) -> dict[str, str] | None:
    """The exact `run` invocation implied by a clean pre-flight report.

    Emitted only when the verdict is ok and a dataset plus boundary IDs were
    supplied: every remaining choice is then determined except the format
    family, which stays an explicit placeholder (the family is the
    hypothesis under test, not something doctor can infer).
    ``--max-context`` comes from the checkpoint's declared window -- the
    value operators most often get wrong. ``--max-new-tokens`` is sized so
    the longest fixture prompt (over every catalog candidate) still fits
    that window; without a measurable tokenizer it keeps the default 64.
    Paths are shell-quoted. Returns None whenever the suggestion would be
    misleading.
    """
    if report.get("verdict") != "ok" or dataset_path is None:
        return None
    try:
        ids = tuple(boundary_ids or ())
    except TypeError:
        return None
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        return None
    model = report.get("model")
    window = model.get("context_window") if type(model) is dict else None
    if type(window) is not int:
        return None
    from .catalog import CATALOG

    new_tokens = _DEFAULT_NEW_TOKENS
    longest: int | None = None
    if tokenizer is not None:
        longest = _longest_prompt_tokens(tokenizer, dataset_path)
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


def _suggestion_tokenizer(report: dict[str, Any], model_path: str | Path) -> Any:
    """Tokenizer for sizing the suggested --max-new-tokens, or None.

    Only attempted when the deep tokenizer check passed; any failure
    degrades to None and the suggestion keeps the default budget.
    """
    try:
        model = report.get("model")
        checks = model.get("checks") if type(model) is dict else None
        deep = checks.get("tokenizer_deep") if type(checks) is dict else None
        if not (type(deep) is dict and deep.get("status") == "ok"):
            return None
        return _build_primary_tokenizer(_directory(model_path))
    except Exception:  # noqa: BLE001 -- suggestion aid; degrade silently
        return None


def doctor(
    model_path: str | Path,
    dataset_path: str | Path | None = None,
    *,
    boundary_ids: tuple[int, ...] | None = None,
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    reference_model_path: str | Path | None = None,
    portability_matrix_path: str | Path | None = None,
) -> dict[str, Any]:
    """Combined pre-flight report for a model directory and fixture file."""
    report = examine_model(
        model_path,
        boundary_ids=boundary_ids,
        boundary_map=boundary_map,
        reference_model_path=reference_model_path,
    )
    if dataset_path is not None:
        dataset_report = examine_dataset(dataset_path)
        report["dataset"] = dataset_report["dataset"]
        if dataset_report["verdict"] == "failed":
            report["verdict"] = "failed"
    if portability_matrix_path is not None:
        # Triage, never verdict: a broken second checkpoint degrades the
        # section instead of failing the pre-flight report.
        try:
            report["portability_matrix"] = portability_matrix(portability_matrix_path)
        except Exception:  # noqa: BLE001 -- matrix must never fail the report
            report["portability_matrix"] = {
                "status": "unavailable",
                "detail": "matrix_failed",
                "second_checkpoint": str(portability_matrix_path),
                "asymmetry": _PORTABILITY_ASYMMETRY,
                "rows": [],
            }
    report["suggestion"] = _next_command(
        model_path,
        dataset_path,
        boundary_ids,
        report,
        tokenizer=_suggestion_tokenizer(report, model_path),
    )
    return report
