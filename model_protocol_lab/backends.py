"""Local safetensors inference with greedy, raw-token, full-prefix decoding."""

from __future__ import annotations

import contextlib
import json
import os
import re
import stat
from bisect import bisect_left
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from threading import Event, RLock
from typing import TYPE_CHECKING, Any, TypeVar

from .core import Backend, Generation, _valid_token_ids, truncate_text_tail

if TYPE_CHECKING:
    import torch
    from torch import Tensor as _TorchTensor


def _directory(value: str | Path) -> Path:
    """Return the directory."""
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
    """Return the member."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("invalid_asset_name")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing_or_unsafe_asset")
    return path


def _open_regular_read(path: Path) -> int:
    """Open a checkpoint asset for reading without ever blocking on a FIFO.

    Opens with ``O_NONBLOCK`` so a file swapped for a named FIFO (no
    writer) between validation and open cannot hang the opener, then
    ``fstat``s the fd and refuses anything that is not a regular file.
    Checking the opened fd -- not the path -- closes the stat->open TOCTOU
    window. Raises ``ValueError`` with the fixed vocabulary on any problem;
    the caller owns the returned fd (``os.fdopen`` takes ownership).
    Mirrors ``datasets.load_dataset`` / ``diffreports.load_results``.
    """
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        raise ValueError("missing_or_unsafe_asset") from None
    try:
        is_regular = stat.S_ISREG(os.fstat(fd).st_mode)
    except OSError:
        is_regular = False
    if not is_regular:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise ValueError("missing_or_unsafe_asset")
    return fd


def _read_json(path: Path) -> dict[str, Any]:
    """Read json."""
    fd = _open_regular_read(path)
    try:
        with os.fdopen(fd, "rb") as handle:
            fd = -1  # ownership moved to the file object
            content = handle.read(128 * 1024 * 1024 + 1)
    finally:
        if fd != -1:
            os.close(fd)
    if len(content) > 128 * 1024 * 1024:
        raise ValueError("oversized_asset_json")
    try:
        value = json.loads(content.decode("utf-8"))
    except (RecursionError, ValueError):
        # Keep the fixed error vocabulary: raw codec/parser text (which
        # reflects the inspected file's bytes) must not reach doctor check
        # details. Mirrors datasets.parse_dataset, which maps the same
        # failure modes to its own vocabulary. The plain ValueError is
        # the interpreter's integer-length limit on >4300-digit literals
        # (not a JSONDecodeError subclass case — JSONDecodeError is one,
        # named for the record): before cycle 202 it escaped raw.
        raise ValueError("invalid_asset_json") from None
    if type(value) is not dict:
        raise ValueError("invalid_asset_json")
    return value


def _model_assets(root: Path) -> tuple[str, ...]:
    """Return the model assets."""
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


def _manifest_entries(
    root: Path, names: tuple[str, ...]
) -> Iterator[tuple[dict[str, Any], tuple[int, int, int]]]:
    """Yield ``(record, fingerprint)`` per asset.

    The record is the public manifest entry (name, byte size, sha256). The
    fingerprint is the cheap ``(st_size, st_mtime_ns, st_ino)`` stat triple
    captured from the same file that was hashed, for later re-verification
    without re-reading the file (see :func:`_verify_manifest_fingerprints`).
    """
    for name in names:
        path = _member(root, name)
        fd = _open_regular_read(path)
        try:
            # fstat the fd we actually read: the "before" snapshot of the
            # opened file, closing the stat->open race for it.
            before = os.fstat(fd)
            digest = sha256()
            with os.fdopen(fd, "rb") as source:
                fd = -1  # ownership moved to the file object
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
        finally:
            if fd != -1:
                os.close(fd)
        after = path.stat()
        fingerprint = (after.st_size, after.st_mtime_ns, after.st_ino)
        if (before.st_size, before.st_mtime_ns, before.st_ino) != fingerprint:
            raise ValueError("assets_changed")
        record = {"name": name, "bytes": after.st_size, "sha256": digest.hexdigest()}
        yield record, fingerprint


def _manifest(root: Path, names: tuple[str, ...]) -> list[dict[str, Any]]:
    """Return the manifest."""
    return [record for record, _ in _manifest_entries(root, names)]


def _manifest_with_fingerprints(
    root: Path, names: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int, int]]]:
    """Full sha256 manifest plus cheap stat fingerprints for later re-check."""
    records: list[dict[str, Any]] = []
    fingerprints: dict[str, tuple[int, int, int]] = {}
    for record, fingerprint in _manifest_entries(root, names):
        records.append(record)
        fingerprints[record["name"]] = fingerprint
    return records, fingerprints


def _verify_manifest_fingerprints(
    root: Path, names: tuple[str, ...], fingerprints: dict[str, tuple[int, int, int]]
) -> None:
    """Cheap re-check that assets have not changed since they were manifested.

    Compares the current ``(st_size, st_mtime_ns, st_ino)`` stat triple of
    each asset against the fingerprint captured while it was hashed, raising
    ``ValueError("assets_changed")`` on any mismatch. This detects file
    replacement (new inode), in-place modification (new mtime/size), and
    deletion -- everything the full re-hash detected except a content swap
    that preserves all three stat fields, which would require nanosecond
    mtime forgery during the load window. Used by the backend to verify
    assets are unchanged after transformers reads them, without hashing
    multi-gigabyte weight files a second time.
    """
    for name in names:
        path = _member(root, name)
        try:
            # skipcq: PYL-W0621 - the name is load-bearing (import binding / payload vocabulary)
            stat = path.stat()
        except OSError:
            raise ValueError("assets_changed") from None
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != fingerprints[name]:
            raise ValueError("assets_changed")


def _ids_valid(value: object, vocab_size: int) -> bool:
    """Return the ids valid."""
    return type(value) is tuple and all(
        type(token) is int and 0 <= token < vocab_size for token in value
    )


@dataclass(frozen=True)
class StopOutcome:
    """Result of emulating one competitor stop semantic on a recorded output.

    ``available=False`` means the emulation could not run (e.g. a decode it
    needed failed); ``detail`` uses a fixed vocabulary, never raw backend
    text. ``truncated_ids`` is the token sequence the competitor would have
    returned: a prefix of the recorded output. ``returned_text`` is the text
    the competitor would have returned when it differs from the decode of
    ``truncated_ids``: set by ``emulate_vllm_holdback`` (whose default
    ``include_stop_str_in_output=False`` strips the stop string from the
    returned text while keeping the completing token in ``truncated_ids``)
    and by ``emulate_sglang_stop`` (whose default ``no_stop_trim=False``
    likewise trims at the leftmost stop-string occurrence) -- token_ids and
    text can disagree, exactly as upstream. None everywhere else; the field
    defaults to None so old callers and positional constructions are
    unaffected.
    """

    available: bool
    fired: bool
    truncated_ids: tuple[int, ...]
    detail: str
    returned_text: str | None = None


# Token representations for the HF stop-string port: decoded strings on the
# classic path, raw token bytes on the byte-space path (see
# _hf_byte_token_mapper). The overlap algorithm only uses len(), slicing and
# prefix/suffix tests, which behave identically on both.
_TokSeq = TypeVar("_TokSeq", str, bytes)


def _hf_overhang_match(token_texts: Sequence[_TokSeq], stop: _TokSeq) -> bool:
    """Port of HF transformers' ``StopStringCriteria`` match rule.

    A match fires only when the stop string overlaps the FINAL token
    representation. The end-overlaps mirror the tensor implementation's
    ``_stop_string_get_matching_positions`` exactly: for every ``k`` in
    ``0..len(last)-1`` (units dropped from the token's end), letting
    ``L = len(last) - k``, an end-overlap of ``min(L, len(stop))`` is
    recorded when ``last[:L]`` ends with the stop string (``L >= len(stop)``)
    or with its length-``L`` suffix (``L < len(stop)``). In particular the
    stop string need only *occur* in the final token -- e.g. token
    ``" world"`` with stop ``"or"`` fires in real HF, as does the
    prefix-overhang ``stop[-i:] == last[:i]`` from the documented string
    algorithm. Every such end-overlap is then walked backwards, consuming
    each earlier token's full length when its suffix matches the remaining
    stop-string suffix (the tensor implementation's valid-position check),
    and the string counts as matched once consumption reaches its length.
    This mirrors both the documented string algorithm and the tensor
    implementation's parallel end-overlap paths (``stopping_criteria.py``,
    ``StopStringCriteria``); the docstring's "impossible to construct"
    claim does not hold for the implementation, which is what is ported
    here. A match can therefore fire when the stop string never appears as
    a literal suffix of the decoded text -- the token-boundary "overhang"
    cases from the HF docstring, e.g. ``["las", "topper"]`` matching
    ``"stop"``.

    The units are whatever the caller supplies: pass per-token decoded
    strings for the classic path, or raw token bytes (with the stop string
    UTF-8-encoded) for the byte-space path that current transformers uses
    for byte-level and byte-fallback decoders (see
    ``_hf_byte_token_mapper``). HF matches on *cleaned vocab strings* on the
    classic path; this port matches on per-token ``decode`` strings there,
    which agree except on exotic spacing edge cases.

    Two faithfulness details mirror the tensor implementation exactly:

    - The walk only consumes the last ``len(stop)`` representations: HF
      truncates ``input_ids`` to ``[:, -maximum_token_len:]`` before
      matching (``maximum_token_len`` is ``len(stop)`` for one stop
      string), so tokens earlier than that window can never participate.
      The bound is a no-op for non-empty representations, which consume at
      least one unit each and end the walk after at most ``len(stop)`` of
      them.
    - An empty representation is transparent to the walk: upstream assigns
      it valid positions ``1..len(stop)-1`` and it adds nothing to the
      running total, so a live walk (``matched < len(stop)``) always passes
      through it. (With a single-unit stop the walk completes at the
      end-overlap, so the empty case is unreachable there -- upstream
      gives it no valid positions then.) Only reachable with exotic
      empty-string vocab entries; out-of-vocabulary ids use the
      ``_OOVTokenBytes`` marker, which keeps killing the walk exactly like
      HF's dummy-row clamp.
    """
    if not stop or not token_texts:
        return False
    last = token_texts[-1]
    if type(last) is not type(stop):
        return False
    target = len(stop)
    token_len = len(last)
    end_overlaps: set[int] = set()
    for dropped in range(token_len):
        width = token_len - dropped
        # The token with ``dropped`` trailing units removed ends with the
        # stop string, or -- when shorter than the stop string -- with the
        # stop string's trailing ``width`` units (the prefix-overhang case).
        suffix = stop if width >= target else stop[target - width :]
        if last[:width].endswith(suffix):
            end_overlaps.add(min(width, target))
    for overlap in sorted(end_overlaps, reverse=True):
        matched = overlap
        index = len(token_texts) - 2
        # HF's per-call ``input_ids[:, -maximum_token_len:]`` truncation:
        # the walk may only consume the last ``target`` representations
        # (the max() keeps the bound from going negative, which would let
        # ``index`` wrap around to the sequence tail).
        earliest = max(0, len(token_texts) - target)
        while matched < target and index >= earliest:
            token = token_texts[index]
            if type(token) is not type(stop):
                break
            if len(token) == 0:
                # Transparent to the walk (see docstring); consumes window
                # budget but no stop-string units.
                index -= 1
                continue
            remainder = stop[: target - matched]
            width = min(len(token), len(remainder))
            if token[len(token) - width :] != remainder[len(remainder) - width :]:
                break
            matched += len(token)
            index -= 1
        if matched >= target:
            return True
    return False


def _first_firing_step(
    token_texts: Sequence[_TokSeq], stop: _TokSeq, base: int, total: int
) -> int | None:
    """1-based output step at which the stop first completes, if any."""
    for step in range(1, total + 1):
        if _hf_overhang_match(token_texts[: base + step], stop):
            return step
    return None


def _bytes_to_unicode() -> dict[int, str]:
    """Port of ``transformers.convert_slow_tokenizer.bytes_to_unicode``.

    The GPT-2 byte-level BPE alphabet: maps each byte value to the unicode
    character the ByteLevel pre-tokenizer represents it with. Inlined
    (rather than imported) so byte-space stop matching works without the HF
    stack; it is a fixed spec mapping, cross-checked against the real
    implementation whenever transformers is available (see tests).
    """
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = byte_values[:]
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, (chr(char) for char in chars), strict=True))


def _decoder_config_has_type(config: Any, decoder_type: str) -> bool:
    """Port of ``StopStringCriteria._decoder_has_type``: does the tokenizers
    decoder config tree contain a decoder of the given type?
    """
    if isinstance(config, dict):
        if config.get("type") == decoder_type:
            return True
        return any(_decoder_config_has_type(value, decoder_type) for value in config.values())
    if isinstance(config, list):
        return any(_decoder_config_has_type(value, decoder_type) for value in config)
    return False


class _OOVTokenBytes(bytes):
    """Byte-space marker for out-of-vocabulary token ids.

    Compares equal to ``b""`` (so an OOV id still "can never match"), but
    its distinct type keeps it from being confused with a genuinely empty
    token representation in :func:`_hf_overhang_match`: upstream clamps OOV
    ids to a dummy table row whose valid positions are all -1, killing any
    walk that reaches it, while a real empty-string token is transparent
    to the walk (valid positions ``1..len(stop)-1``). The walk's
    ``type(token) is not type(stop)`` guards implement both behaviors with
    no extra branches.
    """


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def _hf_byte_token_mapper(tokenizer: Any) -> Callable[[int], bytes | None] | None:
    """Byte-space token mapping mirroring HF ``StopStringCriteria``'s table build.

    Ports ``StopStringCriteria._get_stop_string_matching_mode`` and
    ``_token_to_bytes`` (the byte-space fix for transformers issue #46519):
    for byte-level and byte-fallback decoders, current transformers matches
    stop strings on raw token *bytes*, not decoded strings -- a stop string
    split across UTF-8-fragment tokens fires there while a decode-string
    port stays silent. Returns a per-token-id mapping, or None when the
    tokenizer does not use a byte-oriented decoder (the classic string path
    applies) or the mode cannot be determined. A per-token None means
    "unmappable in byte space" (e.g. special tokens under a byte-level
    decoder): the caller falls back to the isolated decode encoded as
    UTF-8, approximating HF's cleaned-vocab-string fallback for such
    tokens. Out-of-vocabulary ids map to an empty ``_OOVTokenBytes``
    marker, which compares equal to ``b""`` (so it can never match) but
    keeps killing the match walk -- mirroring HF clamping them to a dummy
    table row. Never raises: anything unexpected degrades to None (string
    path).
    """
    try:
        decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
        if decoder is None:
            return None
        if decoder.__class__.__name__ == "ByteLevel":
            mode = "byte_level"
        else:
            state = getattr(decoder, "__getstate__", lambda: None)()
            if isinstance(state, str):
                state = state.encode()
            config = None
            if isinstance(state, bytes):
                try:
                    config = json.loads(state)
                except ValueError:
                    config = None
            # Upstream prefers explicit "<0xNN>" byte-fallback tokens when
            # both markers appear.
            if _decoder_config_has_type(config, "ByteFallback"):
                mode = "byte_fallback"
            elif _decoder_config_has_type(config, "ByteLevel"):
                mode = "byte_level"
            else:
                return None
        byte_decoder: dict[str, int] | None = None
        if mode == "byte_level":
            byte_decoder = {char: byte for byte, char in _bytes_to_unicode().items()}
        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab) if isinstance(vocab, dict) else 0
        if vocab_size <= 0:
            return None
    except Exception:  # noqa: BLE001 -- byte path is optional; degrade silently
        return None

    def _map(token_id: int) -> bytes | None:
        """Return the map."""
        if type(token_id) is not int or token_id < 0:
            raise ValueError("token id must be a nonnegative integer")
        if token_id >= vocab_size:
            return _OOVTokenBytes()
        token = tokenizer.convert_ids_to_tokens(token_id)
        if type(token) is not str:
            return None
        if mode == "byte_level":
            if byte_decoder is not None and all(char in byte_decoder for char in token):
                return bytes(byte_decoder[char] for char in token)
            return None
        # byte_fallback: only explicit "<0xNN>" tokens map directly; the
        # caller falls back to the decoded string for everything else.
        if (
            len(token) == 6
            and token.startswith("<0x")
            and token.endswith(">")
            and all(char in "0123456789abcdefABCDEF" for char in token[3:5])
        ):
            return bytes([int(token[3:5], 16)])
        return None

    return _map


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def emulate_hf_stop_string(
    backend: Backend,
    prompt_ids: tuple[int, ...],
    output_ids: tuple[int, ...],
    stop_text: str,
    *,
    token_bytes: Callable[[int], bytes | None] | None = None,
) -> StopOutcome:
    """Emulate HF transformers' ``StopStringCriteria`` over a recorded output.

    Steps through the recorded output token by token, as ``generate()``
    would, checking the overhang match after each new token against the
    prompt plus generated token texts (the prompt supplies context for
    overhangs off the start, exactly as HF's full-``input_ids`` check does).
    On fire the output is truncated after the firing token, mirroring HF
    returning the sequence that completed the stop string.

    With ``token_bytes`` (a per-token-id byte mapping, e.g.
    ``LocalHFBackend.hf_stop_token_bytes``), matching runs in byte space --
    mirroring current transformers for byte-level and byte-fallback
    decoders, where the match table is built from raw token bytes and the
    stop string is UTF-8-encoded. Tokens the mapping cannot represent fall
    back to the isolated decode encoded as UTF-8 (an approximation of HF's
    cleaned-vocab-string fallback). A per-token None means "unmappable";
    out-of-vocabulary ids should map to empty bytes (the lab's own mapper
    returns the ``_OOVTokenBytes`` marker so the walk kills, mirroring HF's
    dummy-row clamp, rather than passing through). Without ``token_bytes``
    the classic string path is used: token texts come from per-token
    ``backend.decode``, an approximation of HF's cleaned vocab strings.

    HF always keeps the stop string: there is no include/exclude option
    upstream -- ``truncated_ids`` ends with the token that completed the
    stop string, and the returned text is exactly the decode of
    ``truncated_ids``. This is a hard divergence from vLLM's default
    ``include_stop_str_in_output=False``, which strips the stop string from
    the returned text (see ``emulate_vllm_holdback``). The
    ``StopOutcome.returned_text`` field stays None here: nothing is
    stripped, so the kept text is simply ``backend.decode(truncated_ids)``.

    Known divergences from real ``StopStringCriteria`` (all pinned by
    tests; none reachable with supported configurations):

    - Empty stop strings are rejected as ``invalid_emulation_input``
      (``available=False``). Every repo entry point rejects them at the
      API boundary (``diff_backends`` raises ``invalid_stop_texts``);
      ``stop_matrix`` degrades an undecodable boundary to an unavailable
      cell instead of failing. Upstream instead fires at step 1 for any
      input -- a pathological edge the lab deliberately does not
      reproduce.
    - Multi-stop runs compose per-stop cells, each bounded to its own
      ``len(stop)`` window (mirroring the single-stop tensor walk).
      Upstream's joint criteria bounds by the global longest stop, so
      with exotic empty-string vocab entries a short stop's walk can
      reach past its own window when a longer stop is present. Real
      tokenizers do not emit empty-string tokens.
    - The classic string path matches on per-token ``decode`` strings
      while upstream matches on cleaned vocab strings; the two agree
      except on exotic spacing edge cases (e.g. WordPiece inter-token
      spaces: stop ``"ab"`` over tokens ``["a", "b"]`` false-positives
      here while real HF stays silent on the true ``"a b"`` text;
      conversely upstream can fire a step early on its own cleaning
      artifacts, where this port fires at the true position).
    - Non-contiguous vocab ids are degenerate (real checkpoints use
      ``0..N-1``); the byte mapper bounds by ``len(vocab)`` and treats
      higher ids as walk-killing OOV, mirroring HF's dummy-row clamp.
    """
    if (
        not _valid_token_ids(prompt_ids)
        or not _valid_token_ids(output_ids)
        or type(stop_text) is not str
        or not stop_text
    ):
        return StopOutcome(False, False, tuple(output_ids), "invalid_emulation_input")
    if token_bytes is not None and not callable(token_bytes):
        return StopOutcome(False, False, tuple(output_ids), "invalid_emulation_input")
    base = len(prompt_ids)
    try:
        if token_bytes is None:
            texts = [backend.decode((token,)) for token in (*prompt_ids, *output_ids)]
            if any(type(text) is not str for text in texts):
                return StopOutcome(False, False, tuple(output_ids), "token_text_unavailable")
            step = _first_firing_step(texts, stop_text, base, len(output_ids))
        else:
            byte_texts: list[bytes] = []
            for token in (*prompt_ids, *output_ids):
                mapped = token_bytes(token)
                if mapped is None:
                    decoded = backend.decode((token,))
                    if type(decoded) is not str:
                        return StopOutcome(
                            False, False, tuple(output_ids), "token_text_unavailable"
                        )
                    mapped = decoded.encode("utf-8")
                if type(mapped) not in (bytes, _OOVTokenBytes):
                    return StopOutcome(False, False, tuple(output_ids), "token_text_unavailable")
                byte_texts.append(mapped)
            step = _first_firing_step(byte_texts, stop_text.encode("utf-8"), base, len(output_ids))
    except Exception:  # noqa: BLE001 -- emulation degrades, never fails the run
        return StopOutcome(False, False, tuple(output_ids), "token_text_unavailable")
    if step is None:
        return StopOutcome(True, False, tuple(output_ids), "no_stop_within_cap")
    return StopOutcome(True, True, tuple(output_ids[:step]), f"fired_at_step_{step}")


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def emulate_vllm_holdback(
    backend: Backend,
    output_ids: tuple[int, ...],
    stop_text: str,
    *,
    include_stop_str_in_output: bool = False,
    min_tokens: int = 0,
) -> StopOutcome:
    """Emulate vLLM's detokenizer stop handling over a recorded output.

    Ports ``vllm/v1/engine/detokenizer.py`` ``check_stop_strings``: after
    each new token the decoded text is searched with the same
    ``find(stop, 1 - new_char_count - stop_len)`` window, and the
    earliest-completing stop string wins (one stop string here, so no
    tie-break applies). With the default ``include_stop_str_in_output=False``
    -- vLLM's upstream default and a hard divergence from HF, which always
    keeps the stop string -- the returned text is truncated to the
    stop-string start; pass ``include_stop_str_in_output=True`` to model the
    kept variant, which returns the text through the stop string's end. The
    completing token stays in ``truncated_ids`` either way: tokens after the
    one completing the stop string are dropped via ``bisect_left`` over
    per-token text offsets, which for single-token updates keeps exactly the
    firing prefix. The exposed ``StopOutcome.returned_text`` is the text the
    competitor would return (truncated to the stop start, or to the stop end
    when kept). ``min_tokens`` ports vLLM's ``min_tokens`` gating (PR
    #22014): upstream skips the stop check while ``num_output_tokens <=
    min_tokens``. The check resumes at the first ungated step, but the find
    window only covers the tail region where a newly completing stop could
    begin (the negative ``find`` start counts from the end of the text), so
    a stop fully completed during the gated period is *not* refound -- only
    a stop completing at or after the gate (including one spanning the
    gate boundary) fires. If the recorded output ends while still gated, no
    check ever ran (``detail="gated_by_min_tokens"``). A step whose token
    decodes to no new text skips the check entirely, porting upstream
    ``check_stop_strings``' ``if not new_char_count ...: return None`` --
    without it the window start ``1 - len(stop)`` would be 0 for
    single-character stops and refind a stop completed during the gate,
    which upstream never refinds. The
    incremental text comes from ``backend.decode`` of each prefix, an
    approximation of vLLM's ``DecodeStream``; the ``stop_buffer_length``
    holdback only withholds streamed deltas and does not change this final
    truncation.
    """
    if not _valid_token_ids(output_ids) or type(stop_text) is not str or not stop_text:
        return StopOutcome(False, False, tuple(output_ids), "invalid_emulation_input")
    if type(include_stop_str_in_output) is not bool or not (
        type(min_tokens) is int and min_tokens >= 0
    ):
        return StopOutcome(False, False, tuple(output_ids), "invalid_emulation_input")
    stop_len = len(stop_text)
    texts: list[str] = []
    try:
        for step in range(1, len(output_ids) + 1):
            text = backend.decode(tuple(output_ids[:step]))
            if type(text) is not str:
                return StopOutcome(False, False, tuple(output_ids), "decode_unavailable")
            texts.append(text)
    except Exception:  # noqa: BLE001 -- emulation degrades, never fails the run
        return StopOutcome(False, False, tuple(output_ids), "decode_unavailable")
    offsets = [0]
    for text in texts:
        offsets.append(len(text))
    for step, text in enumerate(texts, start=1):
        new_char_count = len(text) - (len(texts[step - 2]) if step > 1 else 0)
        if not new_char_count:
            # Port of upstream check_stop_strings' `if not new_char_count
            # ...: return None`: a token decoding to no new text cannot
            # complete a stop. Without this, the window start
            # `1 - new_char_count - stop_len` is 0 for single-character
            # stops, which would refind a stop completed during the
            # min_tokens gate -- upstream never refinds it.
            continue
        # Same window as vLLM: already-searched text is excluded; a negative
        # start searches from 0, exactly like vLLM's str.find call.
        index = text.find(stop_text, 1 - new_char_count - stop_len)
        if index == -1:
            continue
        if step <= min_tokens:
            # vLLM skips the stop check while num_output_tokens <= min_tokens;
            # the check simply resumes at the first ungated step. The find
            # window (negative start counts from the end) only covers the
            # tail region, so a stop fully completed during the gate is not
            # refound -- only a stop completing at/after the gate fires.
            continue
        stop_end = index + stop_len
        kept = bisect_left(offsets, stop_end, lo=1)
        returned = text[:stop_end] if include_stop_str_in_output else text[:index]
        return StopOutcome(True, True, tuple(output_ids[:kept]), f"fired_at_step_{step}", returned)
    if min_tokens >= len(output_ids) and min_tokens > 0:
        return StopOutcome(True, False, tuple(output_ids), "gated_by_min_tokens")
    return StopOutcome(True, False, tuple(output_ids), "no_stop_within_cap")


def emulate_sglang_stop(
    backend: Backend,
    output_ids: tuple[int, ...],
    stop_text: str,
    *,
    no_stop_trim: bool = False,
) -> StopOutcome:
    """Emulate SGLang's str-based stop handling over a recorded output.

    Ports ``Req._check_str_based_finish``
    (``python/sglang/srt/managers/schedule_batch.py:1836-1847``,
    sgl-project/sglang @ 12dc623): each new output step's decoded text is
    searched for the stop string, and the first step whose decoded prefix
    contains it fires. On fire, ``DetokenizerManager.trim_matched_stop``
    (``python/sglang/srt/managers/detokenizer_manager.py:190-221``) cuts
    the returned text at the leftmost stop-string occurrence; the default
    ``no_stop_trim=False`` (``SamplingParams.no_stop_trim``,
    ``sampling_params.py:153``) strips -- like vLLM's default, unlike HF,
    which always keeps. Pass ``no_stop_trim=True`` to model the kept
    variant, which returns the text through the stop string's end.

    Deliberate divergences from the other engines, each pinned by the
    research log (``docs/RESEARCH_LOG.md``, Cycle 1, SGLang section):

    - No ``min_tokens`` equivalent: upstream ``min_new_tokens`` only clips
      the length budget (``scheduler.py:2631-2634``); the finish path never
      consults it, so this emulation takes no ``min_tokens`` parameter --
      vLLM gates all stops behind ``min_tokens``, SGLang gates nothing.
    - Stop string beats EOS in the same step: ``update_finish_state``
      (``schedule_batch.py:1913-1950``) runs the str-based check before the
      token-based (EOS) check -- the shipped fix for
      https://github.com/sgl-project/sglang/issues/28802. A stop past
      ``max_new_tokens`` demotes to a length finish (not emulated: the lab
      replays a recorded output, so there is no length budget).
    - Multi-stop selection is first-listed, not earliest-completing: the
      upstream check iterates ``stop_strs`` in list order (it even carries
      ``# TODO(lmzheng): handle the case where multiple stop strs are
      hit``) -- a hard divergence from vLLM PR #49391, whose Python side
      selects ``min(stop_index + len(stop_str))`` with ties broken by list
      order (pinned by ``tests/detokenizer/test_check_stop_strings.py``).
      That selection happens across the stop list in one check, so it is
      computed by the caller (``diffbackends``) from these per-stop cells:
      the SGLang winner is the fired stop with the smallest firing step,
      ties broken by list order -- exactly the research-log caveat that
      with 1 token/step the earlier-completing stop still fires at its own
      step and list order decides only when >=2 stops match within one
      check.

    ``truncated_ids`` keeps the ids through the firing step (the trim is
    text-level upstream); the exposed ``StopOutcome.returned_text`` is the
    text SGLang would return (trimmed at the stop start, or through the
    stop end when ``no_stop_trim=True``). SGLang's ``stop_regex`` is out of
    scope (it is not a stop string; see the USAGE.md note).
    """
    if not _valid_token_ids(output_ids) or type(stop_text) is not str or not stop_text:
        return StopOutcome(False, False, tuple(output_ids), "invalid_emulation_input")
    if type(no_stop_trim) is not bool:
        return StopOutcome(False, False, tuple(output_ids), "invalid_emulation_input")
    stop_len = len(stop_text)
    try:
        for step in range(1, len(output_ids) + 1):
            text = backend.decode(tuple(output_ids[:step]))
            if type(text) is not str:
                return StopOutcome(False, False, tuple(output_ids), "decode_unavailable")
            # Upstream searches the decoded output text for the stop string;
            # trim_matched_stop cuts at the leftmost occurrence.
            index = text.find(stop_text)
            if index == -1:
                continue
            stop_end = index + stop_len
            returned = text[:stop_end] if no_stop_trim else text[:index]
            return StopOutcome(
                True, True, tuple(output_ids[:step]), f"fired_at_step_{step}", returned
            )
    except Exception:  # noqa: BLE001 -- emulation degrades, never fails the run
        return StopOutcome(False, False, tuple(output_ids), "decode_unavailable")
    return StopOutcome(True, False, tuple(output_ids), "no_stop_within_cap")


def emulate_eos_stop(output_ids: tuple[int, ...], eos_id: int | None) -> StopOutcome:
    """Emulate an EOS-terminated competitor (lm-eval appends EOS to its stop
    sequences; vLLM and llama.cpp stop on EOS by default): truncate before
    the first EOS token. The lab itself generates through EOS, so this cell
    shows what changes if it had not.
    """
    ids = tuple(output_ids)
    if eos_id is None:
        return StopOutcome(False, False, ids, "eos_id_unavailable")
    if type(eos_id) is not int or eos_id < 0:
        return StopOutcome(False, False, ids, "invalid_eos_id")
    for index, token in enumerate(ids):
        if token == eos_id:
            return StopOutcome(True, True, ids[:index], f"fired_at_step_{index + 1}")
    return StopOutcome(True, False, ids, "no_stop_within_cap")


class TruncatedPromptBackend:
    """Wrap a backend, truncating every encoded prompt mid-token.

    ``encode(text)`` drops ``drop_bytes`` trailing bytes (whole-UTF-8-char
    granularity; see ``core.truncate_text_tail``) and re-encodes, so the
    model sees a prompt truncated in the middle of its final token.
    ``decode`` and ``generate`` pass through untouched. Used by the
    prompt-boundary sensitivity sweep.
    """

    def __init__(self, inner: Backend, drop_bytes: int) -> None:
        """Wrap ``inner``, truncating the prompt by whole bytes."""
        if type(drop_bytes) is not int or drop_bytes < 1:
            raise ValueError("drop_bytes must be a positive integer")
        inner_fingerprint = getattr(inner, "fingerprint", None)
        if type(inner_fingerprint) is not str:
            raise ValueError("Backend fingerprint must be a string")
        self._inner = inner
        self._drop_bytes = drop_bytes
        self.fingerprint = f"{inner_fingerprint}+prompt-trunc-{drop_bytes}b"
        # Truncation only rewrites prompts; the stop-string byte mapping is a
        # pure tokenizer property, so it forwards unchanged.
        self.hf_stop_token_bytes = getattr(inner, "hf_stop_token_bytes", None)

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode the complete rendered prompt into raw token IDs."""
        mutated = truncate_text_tail(text, self._drop_bytes)
        if mutated is None:
            raise ValueError("prompt_truncation_splits_codepoint")
        return self._inner.encode(mutated)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode token IDs without dropping special tokens."""
        return self._inner.decode(token_ids)

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        """Generate exactly the cap unless cancelled or failed."""
        return self._inner.generate(
            input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
        )


class TokenTruncatedPromptBackend:
    """Wrap a backend, dropping whole trailing tokens from every prompt.

    ``encode(text)`` encodes with the inner backend and drops the last
    ``drop_tokens`` token ids, so the model sees a prompt truncated by
    whole tokens -- never a mangled re-encode of a mid-token cut. This
    is the token-depth counterpart of ``TruncatedPromptBackend`` (which
    drops trailing *bytes* and re-encodes); the two wrappers must never
    be confused, hence the distinct ``t`` fingerprint suffix.

    Defined on the BASE prompt encoding: the wrapper drops tokens from
    whatever ``inner.encode`` returns, so stacking it over a backend
    whose ``encode`` appends ids (``PrefillBackend`` appends prefill
    ids) would drop the appended ids first and leave the prompt
    intact -- a silent wrong measurement. Construction therefore
    refuses any wrapper chain containing a backend carrying a
    ``prefill_ids`` attribute, checked at every depth of the chain,
    not just the direct inner (a prefill backend nested behind
    another wrapper evaded the direct-only check). A backend
    whose ``generate`` rewrites ids (``TokenHealingBackend``) is a
    separate eligibility matter for the sweep caller: the wrapper
    measures whatever ids ``generate`` actually consumes, and the
    fingerprint chain (inner fingerprint + this suffix) keeps that
    visible. ``decode`` and ``generate`` pass through untouched.
    """

    def __init__(self, inner: Backend, drop_tokens: int) -> None:
        """Wrap ``inner``, truncating the prompt by whole tokens."""
        if type(drop_tokens) is not int or drop_tokens < 1:
            raise ValueError("drop_tokens must be a positive integer")
        inner_fingerprint = getattr(inner, "fingerprint", None)
        if type(inner_fingerprint) is not str:
            raise ValueError("Backend fingerprint must be a string")
        # Walk the whole wrapper chain: every shipped wrapper stores
        # its wrapped backend as ``_inner``, so a prefill backend
        # nested behind another wrapper (e.g. the byte truncator) is
        # reached here. Cycle-guarded: a pathological _inner loop
        # terminates instead of hanging construction.
        seen: set[int] = set()
        node: Any = inner
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            if isinstance(getattr(node, "prefill_ids", None), tuple):
                raise ValueError(
                    "TokenTruncatedPromptBackend must wrap the base backend, "
                    "not a prefill-wrapped encoding"
                )
            node = getattr(node, "_inner", None)
        self._inner = inner
        self._drop_tokens = drop_tokens
        self.fingerprint = f"{inner_fingerprint}+prompt-trunc-{drop_tokens}t"
        # Truncation only rewrites prompts; the stop-string byte mapping is a
        # pure tokenizer property, so it forwards unchanged.
        self.hf_stop_token_bytes = getattr(inner, "hf_stop_token_bytes", None)

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode the complete rendered prompt into raw token IDs."""
        ids = self._inner.encode(text)
        if type(ids) is not tuple or self._drop_tokens >= len(ids):
            raise ValueError("prompt_token_truncation_beyond_length")
        return ids[: -self._drop_tokens]

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode token IDs without dropping special tokens."""
        return self._inner.decode(token_ids)

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        """Generate exactly the cap unless cancelled or failed."""
        return self._inner.generate(
            input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
        )


class TokenHealingBackend:
    """Wraps a backend; applies HF-style token healing to every prompt.

    ``generate`` first heals ``input_ids`` through the inner backend's
    ``heal_input_ids`` (the lab's port of ``transformers`` token healing);
    when healing is unavailable or fails, the native ids are used instead
    -- fail-closed, never a phantom prompt. ``n_healed`` / ``n_fallback``
    count healed vs native-fallback prompts so the experiment stage can
    report how much of the healed run actually healed. The fingerprint
    gains a ``+token-healing`` suffix so healed runs are distinguishable
    in reports.
    """

    def __init__(self, inner: Backend) -> None:
        """Wrap ``inner``, adding token healing to its generations."""
        inner_fingerprint = getattr(inner, "fingerprint", None)
        if type(inner_fingerprint) is not str:
            raise ValueError("Backend fingerprint must be a string")
        if not callable(getattr(inner, "heal_input_ids", None)):
            raise ValueError("Inner backend does not support token healing")
        self._inner = inner
        self.fingerprint = f"{inner_fingerprint}+token-healing"
        # Healing only rewrites prompts; the stop-string byte mapping is a
        # pure tokenizer property, so it forwards unchanged.
        self.hf_stop_token_bytes = getattr(inner, "hf_stop_token_bytes", None)
        self.n_healed = 0
        self.n_fallback = 0

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode the complete rendered prompt into raw token IDs."""
        return self._inner.encode(text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode token IDs without dropping special tokens."""
        return self._inner.decode(token_ids)

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        """Generate exactly the cap unless cancelled or failed."""
        heal = getattr(self._inner, "heal_input_ids", None)
        try:
            if not callable(heal):
                raise ValueError("healing_unavailable")
            healed = heal(input_ids)
            # Enforce the heal_input_ids contract at the boundary: a
            # misbehaving healer must degrade to the native prompt, never
            # feed malformed ids into generation.
            if type(healed) is not tuple or not all(type(t) is int and t >= 0 for t in healed):
                raise ValueError("healing_contract_violated")
        except Exception:  # noqa: BLE001 -- healing failure degrades, not the run
            healed = input_ids
            self.n_fallback += 1
        else:
            self.n_healed += 1
        return self._inner.generate(healed, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel)


class _ContextWindowError(Exception):
    """max_context exceeds the checkpoint's declared context window.

    Converted to a distinct fixed RuntimeError code (not collapsed into the
    generic init failure) so the operator learns the actionable cause:
    stock GPT-2 checkpoints declare n_positions=1024, below the CLI's
    default --max-context of 2048.
    """


class _MissingHfExtraError(Exception):
    """The ``hf`` extra (torch/transformers) is not installed.

    Converted to a distinct fixed RuntimeError code (not collapsed into the
    generic init failure) so the operator gets the actionable install hint
    instead of a dead-end message.
    """


_BATCH_FORWARD_BYTE_BUDGET = 1 << 33
"""Ceiling on one batched forward's estimated transient peak, in bytes.

The estimate (see :func:`_batch_peak_bytes`) covers the tensors a single
forward materializes. 8 GiB sits just above the batch-invariance audit's
default caller scale; the bound exists so an oversize batch degrades to
the labelled ``batch_size_exceeded`` refusal instead of an OOM kill,
which no Python exception handler could convert into the audit's
fail-closed payload. It is an estimate against a stated budget, not a
memory guarantee: weights are already resident and allocator behaviour
varies. See docs/BATCH_PATH_DESIGN.md.
"""


def _batch_peak_bytes(rows: int, length: int, vocab_size: int, heads: int) -> int:
    """Estimated peak transient bytes of one batched forward.

    ``length`` is the batch's final padded length (longest prompt +
    max_new_tokens): the last generation step is the largest, so sizing
    to it bounds every step. Terms: full-sequence fp32 logits
    ``rows * length * vocab``; eager attention scores
    ``rows * heads * length**2`` with a factor of two, because the
    matmul output and the softmax output are both live; the additive
    ``(rows, 1, length, length)`` fp32 mask held for the whole forward.
    Pure arithmetic -- no model, no torch.
    """
    return (
        4 * rows * length * vocab_size
        + 8 * rows * heads * length * length
        + 4 * rows * length * length
    )


class LocalHFBackend:
    """Built-in GPT-2/Llama only; self-contained local assets; no download fallback.

    Greedy decoding recomputes the full prefix and ignores automatic stops.
    The load dtype defaults to fp32 (the primary measurement); bfloat16 is
    available for the precision probe (cycle 198, docs/PRECISION_DESIGN.md)
    and is recorded in the backend identity. The seed is validated but does
    not change greedy selection. Constructor loading may affect global RNG
    state; requests do not seed global generators. Full-content fingerprints
    read selected assets twice during initialization. Cancellation is
    cooperative between forwards, not an interruption of kernels.
    """

    # skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        max_context: int = 2048,
        dtype: str = "float32",
    ) -> None:
        """Load the local Hugging Face checkpoint at ``model_path``."""
        self._lock = RLock()
        self._model = self._tokenizer = None
        try:
            if type(max_context) is not int or not 1 <= max_context <= 32768:
                raise ValueError("invalid_context_limit")
            if device not in ("cpu", "cuda"):
                raise ValueError("invalid_explicit_device")
            if dtype not in ("float32", "bfloat16"):
                raise ValueError("invalid_dtype")
            effective_device = "cuda:0" if device == "cuda" else "cpu"
            root = _directory(model_path)
            model_names = _model_assets(root)
            # Hash the weight files once (the sha256 record for the report).
            # The cheap stat fingerprints below re-verify the assets are
            # unchanged after transformers reads them, without hashing
            # multi-gigabyte files a second time.
            model_records, model_prints = _manifest_with_fingerprints(root, model_names)
            tok_records, tok_prints = _manifest_with_fingerprints(root, ("tokenizer.json",))
            before = {
                "model": model_records,
                "tokenizer": tok_records,
            }
            raw_config = _read_json(_member(root, "config.json"))
            declared_dtype = raw_config.get("dtype", raw_config.get("torch_dtype"))
            if type(declared_dtype) is not str:
                declared_dtype = None
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

            try:
                import torch
                from transformers import (  # noqa: E402 -- deferred optional import
                    AutoConfig,
                    AutoModelForCausalLM,
                    PreTrainedTokenizerFast,
                )
            except ImportError:
                # The hf extra is not installed: fail with the actionable code,
                # not the generic initialization failure. Fixed vocabulary.
                raise _MissingHfExtraError from None

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
            # transformers ships without type stubs (PreTrainedTokenizerFast
            # resolves to the untyped TokenizersBackend factory). Bind it
            # through a typed Callable so strict mode stays clean whether or
            # not transformers is installed (missing imports become Any).
            _factory: Callable[..., Any] = PreTrainedTokenizerFast
            tokenizer: Any = _factory(
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
                dtype=torch.float32 if dtype == "float32" else torch.bfloat16,
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
            # The identity records the verified dtype, never the requested
            # one and never the config's claim (after a bf16 load the config
            # object still reports float32). Parameters only: buffers such
            # as rotary inv_freq stay fp32 by design in HF Llama.
            wanted_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
            if {p.dtype for p in model.parameters()} != {wanted_dtype}:
                raise ValueError("dtype_verification_failed")
            model.eval()
            model.requires_grad_(False)
            model.to(self._device)
            vocab_size = model.get_input_embeddings().num_embeddings
            tokenizer_ids = frozenset(tokenizer.get_vocab().values())
            if not tokenizer_ids or any(
                type(t) is not int or not 0 <= t < vocab_size for t in tokenizer_ids
            ):
                raise ValueError("tokenizer_model_vocabulary_mismatch")
            _verify_manifest_fingerprints(root, model_names, model_prints)
            _verify_manifest_fingerprints(root, ("tokenizer.json",), tok_prints)
            identity = {
                "adapter": "local-hf-greedy-v1",
                "assets": before,
                "versions": {
                    p: version(p) for p in ("torch", "transformers", "tokenizers", "safetensors")
                },
                "device": effective_device,
                "dtype": dtype,
                "attention": "eager",
                "context_limit": max_context,
                "seed_used_for_selection": False,
                "tokenizer_policy": "raw-tokenizer-json-no-added-specials",
                "generation_policy": "greedy-full-prefix-no-stops-no-cache",
            }
            payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.fingerprint = "sha256:" + sha256(payload).hexdigest()
            self.dtype = dtype
            self.declared_dtype = declared_dtype
            self.seed_used_for_selection = False
            raw_eos = tokenizer.eos_token_id
            self.eos_token_id: int | None = raw_eos if type(raw_eos) is int else None
            self._context_limit, self._vocab_size = max_context, vocab_size
            self._tokenizer_ids, self._tokenizer, self._model = tokenizer_ids, tokenizer, model
            # Byte-space stop-string matching is optional diagnostics: if the
            # mode cannot be determined the HF emulation falls back to the
            # classic string path, so a mapper failure must never fail init.
            self.hf_stop_token_bytes: Callable[[int], bytes | None] | None
            try:
                self.hf_stop_token_bytes = _hf_byte_token_mapper(tokenizer)
            except Exception:  # noqa: BLE001 -- byte path is optional diagnostics
                self.hf_stop_token_bytes = None
        except _ContextWindowError:
            self._model = self._tokenizer = None
            raise RuntimeError("local_context_limit_exceeds_declared_window") from None
        except _MissingHfExtraError:
            self._model = self._tokenizer = None
            raise RuntimeError("local_hf_extra_not_installed") from None
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
        """Encode the complete rendered prompt into raw token IDs."""
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
        """Decode token IDs without dropping special tokens."""
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

    @property
    def vocab_size(self) -> int:
        """Model vocabulary size: valid token IDs satisfy ``0 <= t < vocab_size``."""
        size: int = self._vocab_size
        return size

    # skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
    def heal_input_ids(self, input_ids: tuple[int, ...]) -> tuple[int, ...]:
        """Apply HF-style token healing to a prompt (preprocessing step).

        Faithful port of ``transformers.GenerationMixin.heal_tokens`` into
        the lab's manual greedy loop (which never calls HF ``generate()``,
        so the documented ``token_healing`` flag has no native plumbing
        here): decode + strip + re-encode the prompt (re-canonicalization),
        find every vocabulary token extending the tail token's text, trim
        the tail, and greedily re-choose it with HF's sequence biases
        (+10.0 per extension, +1.0 on the original tail token).

        Deliberate, documented adaptations live in
        :mod:`model_protocol_lab.healing`. Raises ``ValueError`` /
        ``RuntimeError`` with fixed codes on any failure so the caller can
        fail closed to the native prompt.
        """
        from .healing import heal_prompt_ids

        if not _ids_valid(input_ids, self._vocab_size) or not input_ids:
            raise ValueError("invalid_input_ids")
        tokenizer, model = self._ready()
        torch = self._torch
        try:
            vocab = dict(tokenizer.get_vocab())
        except Exception as exc:  # noqa: BLE001 -- tokenizer introspection
            raise RuntimeError("local_heal_failed") from exc
        space_token_text: str | None = None
        try:
            space_id = tokenizer.convert_tokens_to_ids(" ")
            if type(space_id) is int:
                texts = tokenizer.convert_ids_to_tokens([space_id])
                if texts and type(texts[0]) is str:
                    space_token_text = texts[0]
        except Exception:  # noqa: BLE001 -- optional whitespace trick
            space_token_text = None

        def _decode_prompt(ids: tuple[int, ...]) -> str:
            # Skip-specials is False (deliberate adaptation, see healing):
            # the lab's prompts are exact id sequences that must round-trip.
            """Return the decode prompt."""
            text: str = tokenizer.decode(
                list(ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            return text

        def _decode_token(token_id: int) -> str:
            """Return the decode token."""
            text: str = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            return text

        def _reencode(text: str) -> tuple[int, ...]:
            """Return the reencode."""
            values = tuple(
                tokenizer.encode(text, add_special_tokens=False, truncation=False, padding=False)
            )
            if not _ids_valid(values, self._vocab_size):
                raise ValueError("invalid_healed_ids")
            return values

        def _biased_argmax(trimmed: tuple[int, ...], biases: dict[int, float]) -> int:
            """Return the biased argmax."""
            if len(trimmed) > self._context_limit:
                raise ValueError("heal_context_limit_exceeded")
            with torch.inference_mode():
                sequence = torch.tensor([trimmed], dtype=torch.long, device=self._device)
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
                    raise RuntimeError("local_heal_failed")
                for token_id, bias in biases.items():
                    if type(token_id) is int and 0 <= token_id < self._vocab_size:
                        logits[token_id] = logits[token_id] + bias
                return int(torch.argmax(logits).item())

        with self._lock:
            try:
                return heal_prompt_ids(
                    input_ids,
                    decode_prompt=_decode_prompt,
                    decode_token=_decode_token,
                    reencode=_reencode,
                    vocab=vocab,
                    space_token_text=space_token_text,
                    biased_argmax=_biased_argmax,
                )
            except (ValueError, RuntimeError):
                raise
            except Exception as exc:  # noqa: BLE001 -- healing must fail closed
                raise RuntimeError("local_heal_failed") from exc

    def forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        """Run one full-sequence forward pass; return ``[T, vocab]`` fp32 logits.

        This is the teacher-forcing primitive for likelihood scoring: unlike
        :meth:`generate` -- which exposes only the last position's logits
        and samples greedily -- it returns every position's logits so a
        caller can gather the causal-shifted log-probabilities of a target
        span from a single pass. Runs under the backend lock in inference
        mode on CPU; the returned tensor is detached and contiguous.
        Malformed input or a failed/degenerate forward raises ValueError or
        RuntimeError with the adapter's fixed error vocabulary, never a
        partial tensor.
        """
        if not _ids_valid(input_ids, self._vocab_size) or not input_ids:
            raise ValueError("invalid_input_ids")
        if len(input_ids) > self._context_limit:
            raise ValueError("context_limit_exceeded")
        _, model = self._ready()
        torch = self._torch
        with self._lock:
            try:
                with torch.inference_mode():
                    sequence = torch.tensor([input_ids], dtype=torch.long, device=self._device)
                    result = model(
                        input_ids=sequence,
                        attention_mask=torch.ones_like(sequence),
                        past_key_values=None,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    logits: _TorchTensor = result.logits[0]
                    if (
                        logits.ndim != 2
                        or logits.shape[0] != len(input_ids)
                        or logits.shape[1] != self._vocab_size
                        or not bool(torch.isfinite(logits).all().item())
                    ):
                        raise RuntimeError("local_forward_failed")
                    output: _TorchTensor = (
                        logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
                    )
                    return output
            except Exception:  # noqa: BLE001 -- inference errors become fixed codes
                raise RuntimeError("local_forward_failed") from None

    # skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        """Generate exactly the cap unless cancelled or failed."""
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

    # skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
    def generate_batch(
        self,
        prompts: tuple[tuple[int, ...], ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> tuple[tuple[int, ...], ...]:
        """Greedy-generate every prompt together in one batched execution.

        The batch path the batch-invariance audit (cycle 153) probes
        for: all prompts are decoded together -- left-padded, one
        forward per generation step over the whole batch -- under the
        same fp32 full-prefix recompute discipline as :meth:`generate`,
        so the audit compares solo vs batched continuations of
        identical token sequences. Returns one continuation tuple per
        prompt, in input order, each exactly ``max_new_tokens`` long.

        Raises ``ValueError`` / ``RuntimeError`` with the fixed
        vocabulary of docs/BATCH_PATH_DESIGN.md, validated in the
        order pinned there. The audit stage fails closed on any raise,
        so this method never returns partial rows: a cancelled batch
        raises the ``local_batch_cancelled`` code rather than returning
        rows a caller could mistake for a complete measurement, and a
        batch whose estimated forward peak exceeds the byte budget
        raises ``batch_size_exceeded`` before any forward runs.
        """
        if type(prompts) is not tuple or any(
            not _ids_valid(prompt, self._vocab_size) or not prompt for prompt in prompts
        ):
            raise ValueError("invalid_input_ids")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
            raise ValueError("invalid_output_cap")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("invalid_seed")
        if not isinstance(cancel, Event):
            raise ValueError("invalid_cancellation_event")
        if cancel.is_set():
            raise RuntimeError("local_batch_cancelled")
        if any(len(prompt) + max_new_tokens > self._context_limit for prompt in prompts):
            raise ValueError("context_limit_exceeded")
        _, model = self._ready()
        if not prompts:
            return ()
        config = model.config
        heads = getattr(config, "num_attention_heads", None)
        if type(heads) is not int or heads < 1:
            heads = getattr(config, "n_head", None)
        if type(heads) is not int or heads < 1:
            raise RuntimeError("local_batch_failed")
        longest = max(len(prompt) for prompt in prompts)
        peak = _batch_peak_bytes(len(prompts), longest + max_new_tokens, self._vocab_size, heads)
        if peak > _BATCH_FORWARD_BYTE_BUDGET:
            raise ValueError("batch_size_exceeded")
        torch = self._torch
        rows = len(prompts)
        outputs: list[list[int]] = [[] for _ in prompts]
        acquired = False
        while not acquired:
            if cancel.is_set():
                raise RuntimeError("local_batch_cancelled")
            acquired = self._lock.acquire(timeout=0.05)
        try:
            if cancel.is_set():
                raise RuntimeError("local_batch_cancelled")
            cancelled = False
            failed = False
            sequence = mask = result = logits = None
            try:
                with torch.inference_mode():
                    sequence = torch.tensor(
                        [[0] * (longest - len(prompt)) + list(prompt) for prompt in prompts],
                        dtype=torch.long,
                        device=self._device,
                    )
                    mask = torch.zeros_like(sequence)
                    for row, prompt in enumerate(prompts):
                        mask[row, longest - len(prompt) :] = 1
                    for step in range(max_new_tokens):
                        if cancel.is_set():
                            cancelled = True
                            break
                        # Position ids are recomputed from the current
                        # mask at every step: generated columns extend
                        # the mask, and each row's next position is its
                        # own count of real tokens, as in its solo run.
                        positions = (mask.cumsum(-1) - 1).clamp(min=0)
                        result = model(
                            input_ids=sequence,
                            attention_mask=mask,
                            position_ids=positions,
                            past_key_values=None,
                            use_cache=False,
                            output_attentions=False,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                        logits = result.logits[:, -1, :]
                        if (
                            logits.ndim != 2
                            or logits.shape[0] != rows
                            or logits.shape[1] != self._vocab_size
                            or not bool(torch.isfinite(logits).all().item())
                        ):
                            failed = True
                            break
                        tokens = [int(torch.argmax(row).item()) for row in logits]
                        # Drop the step's tensors before anything else
                        # can retain them: a kept slice would pin every
                        # step's full logits tensor for the whole call.
                        result = logits = None
                        if cancel.is_set():
                            cancelled = True
                            break
                        for row, token in enumerate(tokens):
                            outputs[row].append(token)
                        if step + 1 < max_new_tokens:
                            column = torch.tensor(
                                [[token] for token in tokens],
                                dtype=torch.long,
                                device=self._device,
                            )
                            sequence = torch.cat((sequence, column), dim=1)
                            mask = torch.cat(
                                (
                                    mask,
                                    torch.ones((rows, 1), dtype=mask.dtype, device=self._device),
                                ),
                                dim=1,
                            )
            except Exception:  # noqa: BLE001 -- inference errors become the fixed code below
                failed = True
            finally:
                sequence = mask = result = logits = None
            if cancelled:
                raise RuntimeError("local_batch_cancelled")
            if failed:
                raise RuntimeError("local_batch_failed")
            return tuple(tuple(row) for row in outputs)
        finally:
            self._lock.release()
