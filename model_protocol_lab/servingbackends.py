"""Execution backends: measured stop/cache behaviour of real serving stacks.

The lab's stop-semantics results come from *emulating* vendor stop
rules over recorded generations (:mod:`model_protocol_lab.diffbackends`).
This module is the measured counterpart: it drives a real serving
stack over HTTP with raw token ids under a pinned deterministic
sampler configuration ("Tier-A"), records what the stack actually
did — firing step, stop winner, disposition, cache reuse — and
compares the record against the lab's own predictor, so a fidelity
report can say, per stack × weights digest × configuration, whether
the emulation the lab ships agrees with the stack it describes.

Monday-morning utility: a team serving a checkpoint behind
llama.cpp can run one command and learn whether the stop behaviour
their serving stack exhibits is the stop behaviour the lab's
emulations assume — before a silent divergence (a stop string kept
when the emulation strips it, an EOS that loses to a later stop)
reaches their users.

Design commitments (docs/SERVING_BACKENDS_DESIGN.md):

* The :class:`ExecutionBackend` protocol lives *beside*
  ``core.Backend``, never inside it: ``core.Backend`` answers "what
  would this checkpoint generate", an execution backend answers
  "what did this serving stack do with these exact ids". The two
  protocols never merge.
* Capability is data, recorded in three places: the identity's
  posture block, the capabilities record, and per-cell
  ``capability_absent`` records. A request that exercises an
  unsupported capability produces an absent *cell*, never a degraded
  execution and never an omitted row.
* No silent fallback: a probe gate that fails stops the stage with a
  fixed-vocabulary detail; nothing retries under a weaker posture.
* Records carry full prompt ids and SHA-256 digests of text, never
  raw prompt or completion text, and stop winners as list indices /
  token ids, never raw stop text (§6.3 of the design record).
* This module is stdlib-only: HTTP via :mod:`urllib.request` pinned
  to loopback, no proxies, no redirects.

The llama.cpp arm is pinned to one build window (tag ``b12182``,
commit ``fd27b2034905f8dadd8bd6d2cc44595f4929d391``); identity binds
the commit prefix reported in ``build_info`` (the build counter is
recorded verbatim but is an acquisition artefact — a shallow clone
reports ``b1`` for the same commit), the resolved model path, and
the checkpoint's SHA-256. The vLLM arm is designed but not shipped
in this cycle: the request/record schema already carries its fields
(always null on the llama arm) so the schema never changes shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import diffbackends

__all__ = [
    "EXECUTION_BUDGET_DETAIL",
    "FINISH_CLASSES",
    "HARDWARE_CLASSES",
    "LLAMA_CACHE_QUANTUM",
    "LLAMACPP_COMMIT",
    "LLAMACPP_STACK",
    "LLAMACPP_TAG",
    "MAX_EXECUTIONS_PER_RUN",
    "MAX_RECORD_BYTES",
    "PROTOCOL_VERSION",
    "SENTINEL_TEXT",
    "ExecutionBackend",
    "ExecutionCancelled",
    "LlamacppHttpBackend",
    "RecordSink",
    "StackCapabilities",
    "StackIdentity",
    "StackObservation",
    "StackRequest",
    "aggregate_fidelity",
    "build_execution_record",
    "canonical_json",
    "config_fingerprint_for",
    "load_records_artifact",
    "predict_stop_outcome",
    "read_gguf_declared_tokens",
    "run_cache_sequence",
    "run_exec_backends",
    "run_stage",
    "sha256_hex",
    "tier_a_sampler_params",
]

#: The execution-protocol version string, bound into every identity
#: fingerprint: a verdict is only ever read against the protocol
#: that produced it.
PROTOCOL_VERSION = "exec-backends-protocol-1"

#: Stack identifier for the llama.cpp arm (the vLLM arm's ``"vllm"``
#: is a design value with no shipped constructor in this cycle).
LLAMACPP_STACK = "llamacpp"

#: The pinned llama.cpp build window (§8.6 of the design record):
#: tag ``b12182``; the commit is the identity — ``build_info``
#: reports a short prefix of it.
LLAMACPP_TAG = "b12182"
LLAMACPP_COMMIT = "fd27b2034905f8dadd8bd6d2cc44595f4929d391"

#: The probe sentinel: five tokens under the GPT-2 tokenizer; the
#: probe derives the ids from the *injected* tokenizer, never from a
#: hard-coded list (the ids are checkpoint evidence, not constants).
SENTINEL_TEXT = "The capital of France is"

#: One token: the granularity at which llama.cpp's slot layer
#: reports reuse (``timings.cache_n``). A cold or divergent arm
#: reporting more reuse than one quantum is contaminated (C-3).
LLAMA_CACHE_QUANTUM = 1

#: Fixed budgets (§8.5 of the design record): executions per run and
#: the records-artifact byte budget. Over-budget work degrades by
#: labelled omission, never by exception or silent truncation.
MAX_EXECUTIONS_PER_RUN = 512
MAX_RECORD_BYTES = 16_777_216
EXECUTION_BUDGET_DETAIL = "serving_backends_record_budget_exceeded"

#: Reader-side bound for load_records_artifact (RTI-04): the sink
#: budgets the record bytes it writes, but the header/footer envelope
#: lines and the finalized single-document re-wrap are not inside
#: that budget, so the reader allows the record budget plus a fixed
#: envelope allowance. Anything larger was not written by the sink
#: and is refused before parsing, never read into memory whole.
_MAX_RECORD_ARTIFACT_BYTES = MAX_RECORD_BYTES + 65_536

#: The fixed hardware-class vocabulary (§6.2). A class outside this
#: set is refused at construction, never recorded raw.
HARDWARE_CLASSES = ("cpu", "cuda", "metal", "rocm", "vulkan")

#: The A9(i) finish-class allowlist for vLLM-shaped stacks plus the
#: llama stop-type classes, unified: a finish outside this set is
#: ``stack_response_unclassified``, never exported raw.
FINISH_CLASSES = (
    "stop",
    "length",
    "abort",
    "error",
    "repetition",
    "eos",
    "word",
    "none",
    "unclassified",
)

#: Llama.cpp ``stop_type`` values the classifier accepts; anything
#: else is ``stack_response_unclassified``.
_LLAMA_STOP_TYPES = ("eos", "word", "limit", "none")

#: Cache positions (§5.4) plus the standalone stop/fidelity position.
CACHE_POSITIONS = (
    "cold",
    "warm_repeat",
    "shared_prefix",
    "divergent_control",
    "standalone",
)

#: Fixed detail used when an execution is refused because the stack
#: does not offer the exercised capability (capability-as-data).
CAPABILITY_ABSENT = "capability_absent"

_MAX_RESPONSE_BYTES = 16_777_216
_GGUF_MAX_METADATA_KEYS = 100_000
_GGUF_MAX_STRING_BYTES = 1_048_576


class ExecutionCancelled(Exception):
    """Raised inside a driver when the caller's cancel event is set.

    Cancellation produces no observation: cells that never executed
    are absent from the section (their absence is carried by the
    run's planned/executed counts, and by the incremental artifact's
    ``cell_not_executed`` rows when a run dies mid-plan).
    """


class _StackUnreachable(Exception):
    """Internal: the stack could not be reached at all."""


def sha256_hex(data: bytes) -> str:
    """SHA-256 hex digest of ``data`` (record digests are lowercase hex)."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(payload: Any) -> str:
    """The one canonical JSON form digests and fingerprints bind.

    Sorted keys, compact separators, ASCII-escaped: two runs that
    mean the same configuration produce the same bytes.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _text_digest(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def _ids_digest(ids: Iterable[int]) -> str:
    return sha256_hex(canonical_json(list(ids)).encode("utf-8"))


def tier_a_sampler_params(seed: int) -> dict[str, Any]:
    """The Tier-A deterministic sampler tuple (§5.3), at ``seed``.

    Temperature 0 with the sampler chain pinned to
    ``top_k → top_p → min_p → temperature``. ``repeat_penalty`` is
    pinned at exactly 1.0 — the value the spike's three executed
    reads at b12182 established as the server-neutral point (the
    earlier "server default 1.1" rationale was wrong and is deleted
    everywhere, per amendment R-1). The returned dict is the exact
    field set the G2 echo gate compares against the stack's
    ``generation_settings`` echo.
    """
    if type(seed) is not int or isinstance(seed, bool) or seed < 0:
        raise ValueError("stack_request_invalid")
    return {
        "samplers": ["top_k", "top_p", "min_p", "temperature"],
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "dynatemp_range": 0.0,
        "n_probs": 0,
        "seed": seed,
    }


def config_fingerprint_for(config: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON of an identity config block.

    The block itself is recorded verbatim in the identity, so the
    diff-reports mirror recomputes this fingerprint from the stored
    block and compares — a fingerprint is never trusted as reported.
    """
    return sha256_hex(canonical_json(dict(config)).encode("utf-8"))


# ---------------------------------------------------------------------------
# GGUF declared-token reader (A11)
#
# A minimal stdlib reader for the GGUF metadata section: enough to
# extract the tokenizer's declared BOS/EOS token ids. The format is
# the ggml-org GGUF spec: magic, version, tensor count, metadata KV
# count, then typed key/value pairs. Reads are capped; a malformed
# or hostile file is a checkpoint problem, never a crash.
# ---------------------------------------------------------------------------

_GGUF_SCALAR_SIZES = {
    0: 1,  # UINT8
    1: 1,  # INT8
    2: 2,  # UINT16
    3: 2,  # INT16
    4: 4,  # UINT32
    5: 4,  # INT32
    6: 4,  # FLOAT32
    7: 1,  # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


class _GgufReader:
    """Bounds-checked cursor over GGUF metadata bytes."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self._pos + count > len(self._data):
            raise ValueError("stack_checkpoint_unavailable")
        chunk = self._data[self._pos : self._pos + count]
        self._pos += count
        return chunk

    def u32(self) -> int:
        return int(struct.unpack("<I", self.take(4))[0])

    def u64(self) -> int:
        return int(struct.unpack("<Q", self.take(8))[0])

    def string(self) -> str:
        length = self.u64()
        if length > _GGUF_MAX_STRING_BYTES:
            raise ValueError("stack_checkpoint_unavailable")
        return self.take(length).decode("utf-8", errors="replace")

    def skip_value(self, value_type: int) -> None:
        if value_type in _GGUF_SCALAR_SIZES:
            self.take(_GGUF_SCALAR_SIZES[value_type])
        elif value_type == _GGUF_STRING:
            self.string()
        elif value_type == _GGUF_ARRAY:
            element_type = self.u32()
            count = self.u64()
            if count > _GGUF_MAX_STRING_BYTES:
                raise ValueError("stack_checkpoint_unavailable")
            if element_type in _GGUF_SCALAR_SIZES:
                self.take(_GGUF_SCALAR_SIZES[element_type] * count)
            elif element_type == _GGUF_STRING:
                for _ in range(count):
                    self.string()
            else:
                raise ValueError("stack_checkpoint_unavailable")
        else:
            raise ValueError("stack_checkpoint_unavailable")

    def value(self, value_type: int) -> Any:
        if value_type == 4:  # UINT32
            return self.u32()
        if value_type == 5:  # INT32
            raw = self.u32()
            return raw - 2**32 if raw >= 2**31 else raw
        if value_type == _GGUF_ARRAY:
            element_type = self.u32()
            count = self.u64()
            if count > 4096:
                raise ValueError("stack_checkpoint_unavailable")
            if element_type in (4, 5):
                return [self.value(element_type) for _ in range(count)]
            self._pos -= 12
            self.skip_value(value_type)
            return None
        self.skip_value(value_type)
        return None


def read_gguf_declared_tokens(path: Path) -> dict[str, Any]:
    """Declared BOS/EOS token ids from a GGUF file's metadata (A11).

    Returns ``{"eos_token_ids": [...], "bos_token_id": int | None,
    "file_type": int | None}``. Raises
    ``ValueError("stack_checkpoint_unavailable")`` for a missing,
    unreadable, or malformed file: EOS attribution binds the
    *declared* ids, so an unreadable declaration fails closed.
    """
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
            if len(header) < 24 or header[:4] != b"GGUF":
                raise ValueError("stack_checkpoint_unavailable")
            _version, _tensors, kv_count = struct.unpack("<IQQ", header[4:24])
            if kv_count > _GGUF_MAX_METADATA_KEYS:
                raise ValueError("stack_checkpoint_unavailable")
            # Metadata for real checkpoints is small; cap the read so
            # a hostile header cannot make us slurp the tensor data.
            blob = handle.read(64 * 1024 * 1024)
    except OSError:
        raise ValueError("stack_checkpoint_unavailable") from None
    reader = _GgufReader(blob)
    found: dict[str, Any] = {}
    wanted = {
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.bos_token_id",
        "general.file_type",
    }
    for _ in range(kv_count):
        key = reader.string()
        value_type = reader.u32()
        if key in wanted:
            found[key] = reader.value(value_type)
        else:
            reader.skip_value(value_type)

    def _id_list(raw: Any) -> list[int]:
        if type(raw) is int and not isinstance(raw, bool):
            return [raw]
        if type(raw) is list:
            return [item for item in raw if type(item) is int and not isinstance(item, bool)]
        return []

    eos_ids = _id_list(found.get("tokenizer.ggml.eos_token_id"))
    bos_list = _id_list(found.get("tokenizer.ggml.bos_token_id"))
    file_type = found.get("general.file_type")
    return {
        "eos_token_ids": eos_ids,
        "bos_token_id": bos_list[0] if bos_list else None,
        "file_type": file_type if type(file_type) is int else None,
    }


# ---------------------------------------------------------------------------
# Request / capability / identity / observation records (§5, §6)
# ---------------------------------------------------------------------------


def _checked_token_ids(value: Any, code: str) -> tuple[int, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(code)
    out: list[int] = []
    for item in value:
        if type(item) is not int or isinstance(item, bool) or item < 0:
            raise ValueError(code)
        out.append(item)
    return tuple(out)


def _checked_seed(value: Any) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ValueError("stack_request_invalid")
    return value


@dataclass(frozen=True)
class StackRequest:
    """One execution request, carrying every pinned field (§5.2).

    The request carries the vLLM-shaped fields (``stop_token_ids``,
    ``min_tokens``, ``cache_salt``, ``include_stop_str_in_output``)
    even though the llama arm cannot honour them: exercising one
    produces a ``capability_absent`` cell, so the schema is frozen
    once and stack differences stay visible as data.

    ``seed`` has no default on purpose: request seeds derive from
    the run seed via ``core.request_seed_for`` at the driver, so two
    requests in one run never silently share a seed.
    """

    prompt_ids: tuple[int, ...]
    seed: int
    max_new_tokens: int = 64
    stop_texts: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    eos_id: int | None = None
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    cache_salt: str | None = None
    cache_mode: str = "standalone"

    def __post_init__(self) -> None:
        prompt = _checked_token_ids(self.prompt_ids, "stack_request_invalid")
        if not prompt:
            raise ValueError("stack_request_invalid")
        if len(prompt) > diffbackends.MAX_PROMPT_IDS:
            raise ValueError("stack_request_invalid")
        object.__setattr__(self, "prompt_ids", prompt)
        _checked_seed(self.seed)
        if (
            type(self.max_new_tokens) is not int
            or isinstance(self.max_new_tokens, bool)
            or not 1 <= self.max_new_tokens <= 4096
        ):
            raise ValueError("stack_request_invalid")
        stops = diffbackends._checked_stop_texts(self.stop_texts) if self.stop_texts else ()
        object.__setattr__(self, "stop_texts", stops)
        object.__setattr__(
            self,
            "stop_token_ids",
            _checked_token_ids(self.stop_token_ids, "stack_request_invalid"),
        )
        if self.eos_id is not None and (
            type(self.eos_id) is not int or isinstance(self.eos_id, bool) or self.eos_id < 0
        ):
            raise ValueError("stack_request_invalid")
        for flag in (self.include_stop_str_in_output, self.ignore_eos):
            if type(flag) is not bool:
                raise ValueError("stack_request_invalid")
        if (
            type(self.min_tokens) is not int
            or isinstance(self.min_tokens, bool)
            or self.min_tokens < 0
        ):
            raise ValueError("stack_request_invalid")
        if self.cache_salt is not None and type(self.cache_salt) is not str:
            raise ValueError("stack_request_invalid")
        if self.cache_mode not in CACHE_POSITIONS:
            raise ValueError("stack_request_invalid")

    def request_config(self) -> dict[str, Any]:
        """The recorded request configuration (§6.3): verbatim fields.

        Stop texts appear as digests, never raw (the winner is a list
        index, so the texts themselves are never needed downstream).
        """
        return {
            "max_new_tokens": self.max_new_tokens,
            "stop_text_digests": [_text_digest(text) for text in self.stop_texts],
            "stop_token_ids": list(self.stop_token_ids),
            "eos_id": self.eos_id,
            "include_stop_str_in_output": self.include_stop_str_in_output,
            "ignore_eos": self.ignore_eos,
            "min_tokens": self.min_tokens,
            "cache_salt_present": self.cache_salt is not None,
            "cache_mode": self.cache_mode,
            "seed": self.seed,
            "sampler": tier_a_sampler_params(self.seed),
        }


@dataclass(frozen=True)
class StackCapabilities:
    """What the probed stack offers, as data (§5.2/A2).

    ``False`` never degrades an execution; it routes the cells that
    exercise the capability to ``capability_absent`` records.
    """

    raw_token_ids: bool
    stop_token_ids: bool
    cached_token_reporting: bool
    token_id_echo: bool
    prompt_echo: bool
    min_tokens: bool
    include_stop_str_in_output: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "raw_token_ids": self.raw_token_ids,
            "stop_token_ids": self.stop_token_ids,
            "cached_token_reporting": self.cached_token_reporting,
            "token_id_echo": self.token_id_echo,
            "prompt_echo": self.prompt_echo,
            "min_tokens": self.min_tokens,
            "include_stop_str_in_output": self.include_stop_str_in_output,
        }


@dataclass(frozen=True)
class StackIdentity:
    """The identity a run's verdicts are keyed by (§6.2).

    ``config`` is the exact fingerprint input block, recorded
    verbatim; ``config_fingerprint`` is its SHA-256 and is recomputed
    by the diff-reports mirror from the stored block.
    """

    stack: str
    stack_version: str
    checkpoint_digest: str
    checkpoint_bytes: int
    tokenizer_fingerprint: str
    hardware_class: str
    declared_eos_ids: tuple[int, ...]
    config: dict[str, Any]
    config_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "stack": self.stack,
            "stack_version": self.stack_version,
            "checkpoint_digest": self.checkpoint_digest,
            "checkpoint_bytes": self.checkpoint_bytes,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "hardware_class": self.hardware_class,
            "declared_eos_ids": list(self.declared_eos_ids),
            "config": self.config,
            "config_fingerprint": self.config_fingerprint,
        }


@dataclass(frozen=True)
class StackObservation:
    """What one execution returned, classified into fixed vocabulary.

    ``detail`` is ``""`` for a clean observation; any other value is
    a code from the §8.1 vocabulary. ``returned_ids`` is ``None``
    when no stream exists to record (capability-absent,
    execution-failed, invalid and unclassified outcomes).
    """

    returned_ids: tuple[int, ...] | None
    finish_class: str
    stop_winner: dict[str, Any] | None
    cache_n_reported: int | None
    tokens_cached_reported: int | None
    tokens_evaluated_reported: int | None
    returned_text_digest: str | None
    server_detok_digest: str | None
    config_echo_digest: str | None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "returned_ids": list(self.returned_ids) if self.returned_ids is not None else None,
            "finish_class": self.finish_class,
            "stop_winner": self.stop_winner,
            "cache_n_reported": self.cache_n_reported,
            "tokens_cached_reported": self.tokens_cached_reported,
            "tokens_evaluated_reported": self.tokens_evaluated_reported,
            "returned_text_digest": self.returned_text_digest,
            "server_detok_digest": self.server_detok_digest,
            "config_echo_digest": self.config_echo_digest,
            "detail": self.detail,
        }


class ExecutionBackend(Protocol):
    """The execution-backend protocol, beside ``core.Backend`` (C7).

    An implementation probes once (identity + gates G1–G5), then
    executes pinned requests. ``probe`` raises ``ValueError`` with a
    §8.1 detail code when a gate fails; ``execute`` never raises for
    a stack-side failure — it returns an observation whose ``detail``
    carries the fixed-vocabulary outcome — except
    :class:`ExecutionCancelled`, which produces no observation.
    """

    identity: StackIdentity | None

    def probe(self) -> StackIdentity: ...

    def execute(
        self, request: StackRequest, *, cancel: threading.Event | None = None
    ) -> StackObservation: ...


def _absent_observation() -> StackObservation:
    return StackObservation(
        returned_ids=None,
        finish_class="unclassified",
        stop_winner=None,
        cache_n_reported=None,
        tokens_cached_reported=None,
        tokens_evaluated_reported=None,
        returned_text_digest=None,
        server_detok_digest=None,
        config_echo_digest=None,
        detail=CAPABILITY_ABSENT,
    )


# ---------------------------------------------------------------------------
# llama.cpp HTTP backend (pinned window: tag b12182)
#
# The documented launch posture (design doc §4, adjudicated Q1):
# loopback bind, --parallel 1, --api-key, --no-webui,
# --slot-save-path <dedicated operator dir, created before launch>,
# --cache-ram 0. The backend never launches a server; it measures
# the operator's server *as found* and fingerprints the posture it
# can prove (an unauthenticated request is rejected; slot erase is
# available; the web UI reports disabled). --cache-ram 0 cannot be
# read back from the API at b12182, so the fingerprint binds the
# measured cache probe instead, and the cache *sequence* verifies
# coldness itself (C-3) rather than trusting the launch line.
# ---------------------------------------------------------------------------

_PROBE_SEED = 1234
_PROBE_MAX_NEW_TOKENS = 8
_G5_MAX_NEW_TOKENS = 4


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: a redirect is a different endpoint."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


def _opener() -> urllib.request.OpenerDirector:
    # Empty ProxyHandler: environment proxies must never see (or
    # reroute) loopback measurement traffic.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _build_info_commit(build_info: Any) -> str | None:
    """The commit prefix in a ``b<N>-<hex>`` build_info, or None."""
    if type(build_info) is not str:
        return None
    head, sep, tail = build_info.partition("-")
    if not sep or not head.startswith("b") or not head[1:].isdigit():
        return None
    if len(tail) < 7 or len(tail) > 40:
        return None
    if any(char not in "0123456789abcdef" for char in tail):
        return None
    return tail


def _echo_matches(echo: Any, params: Mapping[str, Any]) -> bool:
    """Exact Tier-A echo comparison (G2): every carried field, verbatim."""
    if type(echo) is not dict:
        return False
    for key, expected in params.items():
        if key not in echo:
            return False
        seen = echo[key]
        if type(expected) is list:
            if type(seen) is not list or seen != expected:
                return False
        elif type(expected) is float:
            if type(seen) not in (int, float) or isinstance(seen, bool):
                return False
            # Exact comparison, never a float() conversion (S3-1):
            # Python compares int/float mathematically exactly, while
            # float(10**400) raised OverflowError on a hostile echo.
            if seen != expected:
                return False
        elif type(expected) is int:
            if type(seen) is not int or isinstance(seen, bool) or seen != expected:
                return False
        elif seen != expected:
            return False
    return True


def _int_field(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if type(value) is int and not isinstance(value, bool):
        return value
    return None


class LlamacppHttpBackend:
    """An :class:`ExecutionBackend` driving llama.cpp's HTTP API.

    Construction validates the endpoint (loopback ``http`` only, no
    credentials/query/fragment in the URL, an API key is required —
    the ratified posture has no unguarded form) and binds the
    checkpoint: the GGUF file's SHA-256 and its declared BOS/EOS ids
    are read at construction, so an unreadable checkpoint fails
    before any server traffic.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        gguf_path: Path,
        tokenizer: Any,
        tokenizer_fingerprint: str,
        hardware_class: str = "cpu",
        timeout_s: float = 120.0,
    ) -> None:
        import urllib.parse

        parts = urllib.parse.urlsplit(base_url)
        if (
            parts.scheme != "http"
            or parts.hostname not in ("127.0.0.1", "::1", "localhost")
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or type(api_key) is not str
            or not api_key
            or hardware_class not in HARDWARE_CLASSES
        ):
            raise ValueError("stack_endpoint_refused")
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._gguf_path = Path(gguf_path)
        self._tokenizer = tokenizer
        self._tokenizer_fingerprint = tokenizer_fingerprint
        self._hardware_class = hardware_class
        self._timeout_s = timeout_s
        try:
            digest = hashlib.sha256()
            size = 0
            with self._gguf_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError:
            raise ValueError("stack_checkpoint_unavailable") from None
        self._checkpoint_digest = digest.hexdigest()
        self._checkpoint_bytes = size
        declared = read_gguf_declared_tokens(self._gguf_path)
        if not declared["eos_token_ids"]:
            raise ValueError("stack_checkpoint_unavailable")
        self._declared_eos = tuple(sorted(set(declared["eos_token_ids"])))
        self._declared_bos = declared["bos_token_id"]
        self.identity: StackIdentity | None = None
        self._props_snapshot: dict[str, Any] | None = None
        self._slot_erase_available = False
        self._opener = _opener()

    # -- tokenizer plumbing (the tokenizer is injected; this module
    #    never imports transformers) --

    def _tok_encode(self, text: str) -> tuple[int, ...]:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return _checked_token_ids(ids, "token_vocab_mismatch")

    def _tok_decode(self, ids: Iterable[int]) -> str:
        text = self._tokenizer.decode(list(ids), skip_special_tokens=False)
        if type(text) is not str:
            raise ValueError("token_vocab_mismatch")
        return text

    def decode_ids(self, ids: tuple[int, ...]) -> str:
        """Detokenize with the injected tokenizer (the predictor's view)."""
        return self._tok_decode(ids)

    # -- HTTP plumbing --

    def _call(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None,
        *,
        authenticated: bool = True,
    ) -> tuple[int, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if authenticated:
            headers["Authorization"] = f"Bearer {self._api_key}"
        # The base URL was validated at construction (plain HTTP,
        # loopback host, no userinfo/query/fragment) and the path
        # is an internal constant: opening it is the validated
        # endpoint, not a new exposure.
        request = urllib.request.Request(  # noqa: S310
            self._base + path, data=data, headers=headers, method=method
        )
        try:
            with self._opener.open(request, timeout=self._timeout_s) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                status = int(response.status)
        except urllib.error.HTTPError as exc:
            raw = exc.read(_MAX_RESPONSE_BYTES + 1)
            status = int(exc.code)
        except (urllib.error.URLError, TimeoutError, OSError):
            raise _StackUnreachable from None
        if len(raw) > _MAX_RESPONSE_BYTES:
            return status, None
        try:
            return status, json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return status, None

    def _wait_health(self) -> None:
        import time

        deadline = time.monotonic() + min(self._timeout_s, 60.0)
        while True:
            try:
                status, payload = self._call("GET", "/health", None, authenticated=False)
                if status == 200 and type(payload) is dict and payload.get("status") == "ok":
                    return
            except _StackUnreachable:
                pass
            if time.monotonic() >= deadline:
                raise ValueError("stack_unreachable")
            time.sleep(0.25)

    def erase_slot(self) -> bool:
        """Erase slot 0 (A2 hygiene); availability is posture data."""
        try:
            status, _payload = self._call("POST", "/slots/0?action=erase", {})
        except _StackUnreachable:
            return False
        return status == 200

    # -- probe: gates A3, G1–G5 in order --

    def probe(self) -> StackIdentity:
        self._wait_health()
        # A3 (auth posture): the slot/props surface must reject an
        # unauthenticated caller and accept the operator key. C-1(a)
        # was discharged by execution at b12182 (pre-routing 401s);
        # the probe re-verifies it against the server as found.
        try:
            unauth_status, _ = self._call("GET", "/props", None, authenticated=False)
            auth_status, props = self._call("GET", "/props", None, authenticated=True)
        except _StackUnreachable:
            raise ValueError("stack_unreachable") from None
        if unauth_status != 401 or auth_status != 200 or type(props) is not dict:
            raise ValueError("stack_endpoint_refused")
        # G1 (identity window): commit prefix + resolved model path
        # + the single-slot topology the protocol pins.
        build_info = props.get("build_info")
        commit = _build_info_commit(build_info)
        if commit is None or not LLAMACPP_COMMIT.startswith(commit):
            if commit is None and type(build_info) is str and build_info:
                raise ValueError("stack_identity_unavailable")
            raise ValueError("stack_version_unsupported")
        model_path = props.get("model_path")
        if type(model_path) is not str or Path(model_path).resolve() != self._gguf_path.resolve():
            raise ValueError("stack_identity_unavailable")
        if props.get("total_slots") != 1:
            raise ValueError("stack_topology_mismatch")
        self._props_snapshot = {
            "build_info": build_info,
            "model_path": model_path,
            "total_slots": 1,
        }
        # Slot-erase availability is posture data (the fallback
        # posture records capability-absent cache cells); the probe
        # erase also makes the G5 cold arm genuinely post-erase.
        self._slot_erase_available = self.erase_slot()
        # G2 (config echo) on the sentinel, Tier-A pinned.
        try:
            sentinel_ids = self._tok_encode(SENTINEL_TEXT)
        except ValueError:
            raise ValueError("token_vocab_mismatch") from None
        if len(sentinel_ids) < 2:
            raise ValueError("token_vocab_mismatch")
        probe_request = StackRequest(
            prompt_ids=sentinel_ids, seed=_PROBE_SEED, max_new_tokens=_PROBE_MAX_NEW_TOKENS
        )
        observation = self._raw_completion(probe_request, cache_prompt=False)
        if observation.detail == "stack_unreachable":
            raise ValueError("stack_unreachable")
        if observation.detail != "" or observation.returned_ids is None:
            detail = observation.detail or "stack_response_invalid"
            raise ValueError(detail)
        # G3 (prompt echo, text form): the response's prompt text
        # must equal the local detokenization AND the server's own
        # /detokenize of the sent ids (amendment R-5: text equality,
        # never an ids-valued prompt assertion).
        status, completion = self._call(
            "POST", "/completion", self._completion_body(probe_request, cache_prompt=False)
        )
        if status != 200 or type(completion) is not dict:
            raise ValueError("stack_execution_failed")
        prompt_text = completion.get("prompt")
        detok_text = self._server_detokenize(sentinel_ids)
        try:
            local_text = self._tok_decode(sentinel_ids)
        except ValueError:
            raise ValueError("token_vocab_mismatch") from None
        if (
            type(prompt_text) is not str
            or prompt_text != local_text
            or detok_text is None
            or detok_text != local_text
        ):
            raise ValueError("prompt_echo_mismatch")
        # G4 (vocab roundtrip incl. EOT): detokenize the sentinel
        # plus the declared EOT, re-tokenize both ways.
        eot = self._declared_eos[0]
        extended = (*sentinel_ids, eot)
        extended_text = self._server_detokenize(extended)
        if extended_text is None:
            raise ValueError("token_vocab_mismatch")
        server_ids = self._server_tokenize(extended_text)
        try:
            local_ids = self._tok_encode(extended_text)
        except ValueError:
            raise ValueError("token_vocab_mismatch") from None
        if server_ids != extended or local_ids != extended:
            raise ValueError("token_vocab_mismatch")
        # G5 (cache quantity is cache_n, amendment R-2): a cold arm
        # and a warm repeat of the sentinel; discrimination is
        # warm > cold. Missing cache_n is capability data, not a
        # probe failure.
        g5_request = StackRequest(
            prompt_ids=sentinel_ids, seed=_PROBE_SEED, max_new_tokens=_G5_MAX_NEW_TOKENS
        )
        cold = self._raw_completion(g5_request, cache_prompt=True)
        warm = self._raw_completion(g5_request, cache_prompt=True)
        cached_reporting = (
            cold.cache_n_reported is not None
            and warm.cache_n_reported is not None
            and warm.cache_n_reported > cold.cache_n_reported
        )
        capabilities = StackCapabilities(
            raw_token_ids=True,
            stop_token_ids=False,  # U1: zero stop_token surface at b12182
            cached_token_reporting=cached_reporting,
            token_id_echo=True,
            prompt_echo=True,
            min_tokens=False,
            include_stop_str_in_output=False,
        )
        settings = props.get("default_generation_settings")
        tier_a = tier_a_sampler_params(_PROBE_SEED)
        tier_a.pop("seed")
        config: dict[str, Any] = {
            "protocol": PROTOCOL_VERSION,
            "stack": LLAMACPP_STACK,
            "build_info": build_info,
            "total_slots": 1,
            "model_path_digest": _text_digest(model_path),
            "checkpoint_sha256": self._checkpoint_digest,
            "declared_eos_ids": list(self._declared_eos),
            "generation_settings_digest": sha256_hex(canonical_json(settings).encode("utf-8"))
            if settings is not None
            else None,
            "posture": {
                "api_key_required": True,
                "slot_erase_available": self._slot_erase_available,
                "endpoint_slots": props.get("endpoint_slots") is True,
                "ui_enabled": props.get("ui") is True,
            },
            "cache_probe": {
                "cold_cache_n": cold.cache_n_reported,
                "warm_cache_n": warm.cache_n_reported,
            },
            "tier_a": tier_a,
        }
        identity = StackIdentity(
            stack=LLAMACPP_STACK,
            stack_version=str(build_info),
            checkpoint_digest=self._checkpoint_digest,
            checkpoint_bytes=self._checkpoint_bytes,
            tokenizer_fingerprint=self._tokenizer_fingerprint,
            hardware_class=self._hardware_class,
            declared_eos_ids=self._declared_eos,
            config=config,
            config_fingerprint=config_fingerprint_for(config),
        )
        self.identity = identity
        self._capabilities = capabilities
        return identity

    @property
    def capabilities(self) -> StackCapabilities:
        """The probed capabilities (valid after :meth:`probe`)."""
        caps: StackCapabilities | None = getattr(self, "_capabilities", None)
        if caps is None:
            raise ValueError("stack_identity_unavailable")
        return caps

    def verify_identity(self) -> bool:
        """Re-read /props and compare the identity-bearing fields (A8).

        Any change — or an unreachable server — returns False; the
        caller decides which cells the change invalidates.
        """
        if self._props_snapshot is None:
            return False
        try:
            status, props = self._call("GET", "/props", None, authenticated=True)
        except _StackUnreachable:
            return False
        if status != 200 or type(props) is not dict:
            return False
        return all(props.get(key) == value for key, value in self._props_snapshot.items())

    # -- completion plumbing --

    def _completion_body(self, request: StackRequest, *, cache_prompt: bool) -> dict[str, Any]:
        body: dict[str, Any] = dict(tier_a_sampler_params(request.seed))
        body.update(
            {
                "prompt": list(request.prompt_ids),
                "n_predict": request.max_new_tokens,
                "stop": list(request.stop_texts),
                "ignore_eos": request.ignore_eos,
                "cache_prompt": cache_prompt,
                "id_slot": 0,
                "return_tokens": True,
                "stream": False,
            }
        )
        return body

    def _server_detokenize(self, ids: Iterable[int]) -> str | None:
        try:
            status, payload = self._call("POST", "/detokenize", {"tokens": list(ids)})
        except _StackUnreachable:
            return None
        if status != 200 or type(payload) is not dict:
            return None
        content = payload.get("content")
        return content if type(content) is str else None

    def _server_tokenize(self, text: str) -> tuple[int, ...] | None:
        try:
            status, payload = self._call("POST", "/tokenize", {"content": text})
        except _StackUnreachable:
            return None
        if status != 200 or type(payload) is not dict:
            return None
        try:
            return _checked_token_ids(payload.get("tokens"), "token_vocab_mismatch")
        except ValueError:
            return None

    def _raw_completion(self, request: StackRequest, *, cache_prompt: bool) -> StackObservation:
        """One /completion call, classified; never raises for stack faults."""
        try:
            status, payload = self._call(
                "POST", "/completion", self._completion_body(request, cache_prompt=cache_prompt)
            )
        except _StackUnreachable:
            return StackObservation(
                returned_ids=None,
                finish_class="unclassified",
                stop_winner=None,
                cache_n_reported=None,
                tokens_cached_reported=None,
                tokens_evaluated_reported=None,
                returned_text_digest=None,
                server_detok_digest=None,
                config_echo_digest=None,
                detail="stack_unreachable",
            )
        if status != 200 or type(payload) is not dict:
            return StackObservation(
                returned_ids=None,
                finish_class="unclassified",
                stop_winner=None,
                cache_n_reported=None,
                tokens_cached_reported=None,
                tokens_evaluated_reported=None,
                returned_text_digest=None,
                server_detok_digest=None,
                config_echo_digest=None,
                detail="stack_execution_failed",
            )
        return self._classify(request, payload)

    def _classify(self, request: StackRequest, payload: dict[str, Any]) -> StackObservation:
        tokens_raw = payload.get("tokens")
        content = payload.get("content")
        stop_type = payload.get("stop_type")
        settings = payload.get("generation_settings")
        echo_digest = (
            sha256_hex(canonical_json(settings).encode("utf-8")) if type(settings) is dict else None
        )
        timings = payload.get("timings")
        cache_n = _int_field(timings, "cache_n") if type(timings) is dict else None
        base: dict[str, Any] = {
            "cache_n_reported": cache_n,
            "tokens_cached_reported": _int_field(payload, "tokens_cached"),
            "tokens_evaluated_reported": _int_field(payload, "tokens_evaluated"),
            "config_echo_digest": echo_digest,
        }
        if (
            type(tokens_raw) is not list
            or any(type(t) is not int or isinstance(t, bool) for t in tokens_raw)
            or type(content) is not str
            or type(stop_type) is not str
        ):
            return StackObservation(
                returned_ids=None,
                finish_class="unclassified",
                stop_winner=None,
                returned_text_digest=None,
                server_detok_digest=None,
                detail="stack_response_invalid",
                **base,
            )
        tokens = tuple(tokens_raw)
        # The decode is inlined (not _tok_decode) because the two
        # failure classes must stay distinct: a decoder EXCEPTION on
        # these ids means the stack returned ids outside the local
        # vocabulary (S3-2) -- the response is invalid, and pre-fix
        # the ids were banked as a normal observation whose predictor
        # decode then raised a raw KeyError out of execute_cell. A
        # non-str decode RESULT is a tokenizer-health fault, which
        # keeps raising token_vocab_mismatch exactly as the probe
        # gates have always required.
        try:
            decoded_locally = self._tokenizer.decode(list(tokens), skip_special_tokens=False)
        except (KeyError, IndexError, ValueError, OverflowError):
            return StackObservation(
                returned_ids=None,
                finish_class="unclassified",
                stop_winner=None,
                returned_text_digest=None,
                server_detok_digest=None,
                detail="stack_response_invalid",
                **base,
            )
        if type(decoded_locally) is not str:
            raise ValueError("token_vocab_mismatch")
        detok = self._server_detokenize(tokens)
        common: dict[str, Any] = {
            "returned_ids": tokens,
            "returned_text_digest": _text_digest(content),
            "server_detok_digest": _text_digest(detok) if detok is not None else None,
            **base,
        }
        if not _echo_matches(settings, tier_a_sampler_params(request.seed)):
            # §7.2: the per-request echo is checked on the response
            # itself, every time — a sampler field the stack silently
            # re-pins invalidates the cell, not the run.
            return StackObservation(
                finish_class="unclassified",
                stop_winner=None,
                detail="config_echo_mismatch",
                **common,
            )
        if stop_type not in _LLAMA_STOP_TYPES:
            return StackObservation(
                finish_class="unclassified",
                stop_winner=None,
                detail="stack_response_unclassified",
                **common,
            )
        if len(tokens) > request.max_new_tokens + 1:
            # §8.5/A6: the returned length may exceed the cap by at
            # most one token (partial-multibyte overshoot).
            finish = {"eos": "eos", "word": "word", "limit": "length", "none": "none"}[stop_type]
            return StackObservation(
                finish_class=finish,
                stop_winner=None,
                detail="stack_response_invalid",
                **common,
            )
        if stop_type == "limit":
            return StackObservation(finish_class="length", stop_winner=None, detail="", **common)
        if stop_type == "none":
            return StackObservation(finish_class="none", stop_winner=None, detail="", **common)
        if stop_type == "eos":
            winner = {"kind": "eos", "id": tokens[-1]} if tokens else None
            if winner is None:
                return StackObservation(
                    finish_class="unclassified",
                    stop_winner=None,
                    detail="stack_response_invalid",
                    **common,
                )
            return StackObservation(finish_class="eos", stop_winner=winner, detail="", **common)
        # stop_type == "word"
        word = payload.get("stopping_word")
        if type(word) is not str or not word:
            return StackObservation(
                finish_class="unclassified",
                stop_winner=None,
                detail="stack_response_invalid",
                **common,
            )
        index = request.stop_texts.index(word) if word in request.stop_texts else None
        winner = {"kind": "stop_text", "index": index, "digest": _text_digest(word)}
        return StackObservation(finish_class="word", stop_winner=winner, detail="", **common)

    # -- execute (the ExecutionBackend surface) --

    def execute(
        self, request: StackRequest, *, cancel: threading.Event | None = None
    ) -> StackObservation:
        if self.identity is None:
            raise ValueError("stack_identity_unavailable")
        if cancel is not None and cancel.is_set():
            raise ExecutionCancelled
        # Capability-as-data: each unsupported exercised capability
        # is an absent cell, never a degraded execution.
        if (
            request.stop_token_ids
            or request.min_tokens > 0
            or request.include_stop_str_in_output
            or request.cache_salt is not None
        ):
            return _absent_observation()
        observation = self._raw_completion(request, cache_prompt=request.cache_mode != "standalone")
        if observation.detail == "stack_unreachable":
            return StackObservation(
                returned_ids=None,
                finish_class="unclassified",
                stop_winner=None,
                cache_n_reported=None,
                tokens_cached_reported=None,
                tokens_evaluated_reported=None,
                returned_text_digest=None,
                server_detok_digest=None,
                config_echo_digest=None,
                detail="stack_execution_failed",
            )
        return observation


# ---------------------------------------------------------------------------
# The mirror predictor (§5.5) and the execution record (§6.3)
#
# The predictor derives, from the returned ids alone, the firing
# step and winner the stack *should* have reported under the generic
# scan semantics: decode prefixes in order; a stop text completes at
# the first step whose decoded prefix contains it; the EOS fires at
# its first position (unless ignore_eos); the earliest completion
# wins. A stop text and the EOS completing at the same step are a
# documented tie: the step is certain, the winner is the tied set.
# ---------------------------------------------------------------------------


def predict_stop_outcome(
    *,
    decode: Callable[[tuple[int, ...]], str],
    returned_ids: tuple[int, ...],
    stop_texts: tuple[str, ...],
    eos_id: int | None,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    """Predict the stop outcome of a returned stream (generic scan).

    Returns the predicted firing step (1-based position of the
    completing token, or None), the predicted winner, the per-stop
    completion steps, the tie flag, and the disposition basis
    digests (local detokenization of the full stream, and of the
    stream truncated at the winning stop's first occurrence).
    """
    # Stop texts are strings (S4-2): the same invalid_stop_texts code
    # the CLI validator (diffbackends._checked_stop_texts) raises. A
    # bytes/int member previously died as a raw TypeError inside the
    # containment scan, after the caller's decode work had run.
    if any(type(text) is not str for text in stop_texts):
        raise ValueError("invalid_stop_texts")
    prefixes: list[str] = []
    for step in range(1, len(returned_ids) + 1):
        prefixes.append(decode(returned_ids[:step]))
    full_text = prefixes[-1] if prefixes else decode(())
    stop_steps: dict[int, int] = {}
    for index, text in enumerate(stop_texts):
        for step, prefix in enumerate(prefixes, start=1):
            if text in prefix:
                stop_steps[index] = step
                break
    eos_step: int | None = None
    if eos_id is not None and not ignore_eos:
        for position, token in enumerate(returned_ids):
            if token == eos_id:
                eos_step = position + 1
                break
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, step in stop_steps.items():
        candidates.append((step, 0, {"kind": "stop_text", "index": index}))
    if eos_step is not None:
        candidates.append((eos_step, 1, {"kind": "eos", "id": eos_id}))
    firing_step: int | None = None
    winner: dict[str, Any] | None = None
    tie = False
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]))
        firing_step = candidates[0][0]
        winner = candidates[0][2]
        tie = len(candidates) > 1 and candidates[1][0] == firing_step
    stripped_digest: str | None = None
    predicted_disposition: str | None = None
    if winner is not None and winner["kind"] == "stop_text":
        stop_text = stop_texts[int(winner["index"])]
        start = full_text.find(stop_text)
        if start >= 0:
            stripped_digest = _text_digest(full_text[:start])
        predicted_disposition = "stripped"
    return {
        "predicted_firing_step": firing_step,
        "predicted_winner": winner,
        "stop_steps": {str(index): step for index, step in stop_steps.items()},
        "precedence_tie": tie,
        "local_text_digest": _text_digest(full_text),
        "stripped_text_digest": stripped_digest,
        "predicted_disposition": predicted_disposition,
    }


def _observed_disposition(
    *,
    observation: StackObservation,
    prediction: Mapping[str, Any],
) -> str | None:
    """Disposition from the digest triple (§5.1): kept / stripped /
    unobservable, computed without ever holding raw text."""
    if observation.stop_winner is None or observation.stop_winner.get("kind") != "stop_text":
        return None
    stack_digest = observation.returned_text_digest
    if stack_digest is None:
        return "unobservable"
    if stack_digest == prediction["local_text_digest"]:
        return "kept"
    stripped = prediction["stripped_text_digest"]
    if stripped is not None and stack_digest == stripped:
        return "stripped"
    return "unobservable"


def _fidelity_verdict(
    *,
    request: StackRequest,
    observation: StackObservation,
    prediction: Mapping[str, Any],
    declared_eos_ids: tuple[int, ...],
) -> tuple[str, str | None, str, str | None]:
    """(verdict, divergence_class, divergence_detail, cell_detail).

    The comparison of one measured cell against the predictor, in
    the divergence-class vocabulary the lab already ships
    (:mod:`model_protocol_lab.diffbackends`).
    """
    if observation.detail == CAPABILITY_ABSENT:
        return "capability_absent", None, "", CAPABILITY_ABSENT
    if observation.detail in (
        "stack_response_invalid",
        "stack_response_unclassified",
        "config_echo_mismatch",
        "stack_execution_failed",
    ):
        return "unobservable", None, observation.detail, observation.detail
    if (
        observation.server_detok_digest is not None
        and observation.server_detok_digest != prediction["local_text_digest"]
    ):
        # A7: the stack's own detokenizer disagrees with the
        # predictor's — an instrument-side finding, aggregated in
        # its own row, never as a vendor divergence.
        return (
            "predictor_detokenization_mismatch",
            None,
            "predictor_detokenization_mismatch",
            "predictor_detokenization_mismatch",
        )
    winner = observation.stop_winner
    if (
        winner is not None
        and winner.get("kind") == "eos"
        and winner.get("id") not in declared_eos_ids
    ):
        # A11: EOS attribution binds the GGUF-declared ids.
        return "diverge", None, "eos_attribution_mismatch", "eos_attribution_mismatch"
    predicted_winner = prediction["predicted_winner"]
    predicted_step = prediction["predicted_firing_step"]
    observed_fired = winner is not None
    observed_step = len(observation.returned_ids or ())
    if observation.finish_class == "none":
        return "unobservable", None, "stack_response_unfinished", ""
    if not observed_fired and predicted_winner is not None:
        return "diverge", "MISSING_STOP_AT_CAP", "predicted firing never observed", ""
    if observed_fired and predicted_winner is None:
        return "diverge", "SPURIOUS_STOP", "stack fired with no predicted completion", ""
    if predicted_winner is not None and winner is not None:
        if predicted_step is None:
            return "unobservable", None, "predictor_step_missing", ""
        if predicted_step != observed_step:
            return (
                "diverge",
                "FIRING_STEP_DIFFERS",
                f"predicted step {predicted_step}, observed step {observed_step}",
                "",
            )
        predicted_kind = predicted_winner["kind"]
        observed_kind = winner["kind"]
        if predicted_kind != observed_kind:
            if prediction["precedence_tie"]:
                tied = {"stop_text", "eos"}
                if observed_kind in tied:
                    return "agree", None, "", ""
            if observed_kind == "eos":
                return "diverge", "EOS_OVER_STR_PRECEDENCE", "eos won over a stop text", ""
            return "diverge", "STR_OVER_EOS_PRECEDENCE", "stop text won over eos", ""
        if observed_kind == "stop_text":
            stack_index = winner.get("index")
            if stack_index is not None and stack_index != predicted_winner["index"]:
                stack_step = prediction["stop_steps"].get(str(stack_index))
                if stack_step is not None and int(stack_step) != observed_step:
                    return (
                        "diverge",
                        "FIRING_STEP_DIFFERS",
                        "stack's stop completes at a different step",
                        "",
                    )
                # Same step, different list order: a tie the stack
                # resolved the other way — the firing event agrees.
                return "agree", None, "", ""
            disposition = _observed_disposition(observation=observation, prediction=prediction)
            if disposition == "kept":
                return "diverge", "KEPT_VS_STRIPPED", "stack kept the stop text", ""
            if disposition == "unobservable":
                return "unobservable", None, "disposition_unobservable", ""
    return "agree", None, "", ""


def build_execution_record(
    *,
    request: StackRequest,
    observation: StackObservation,
    prediction: Mapping[str, Any],
    identity: StackIdentity,
    candidate: str,
    fixture: str,
    cache_position: str,
) -> dict[str, Any]:
    """Assemble one §6.3 execution record from its three sources.

    Ids and digests only: the prompt ids in full (they are the
    measurement), every text as a SHA-256 digest, the stop winner as
    an index/id. ``request_digest`` binds the request configuration
    to the prompt digest, so a record cannot be re-pointed at a
    different prompt without changing its digest.
    """
    verdict, divergence_class, divergence_detail, cell_detail = _fidelity_verdict(
        request=request,
        observation=observation,
        prediction=prediction,
        declared_eos_ids=identity.declared_eos_ids,
    )
    if not cell_detail and observation.detail:
        cell_detail = observation.detail
    prompt_digest = _ids_digest(request.prompt_ids)
    request_config = request.request_config()
    request_digest = sha256_hex(
        canonical_json({"config": request_config, "prompt_id_digest": prompt_digest}).encode(
            "utf-8"
        )
    )
    observed_disposition = _observed_disposition(observation=observation, prediction=prediction)
    mirror: dict[str, Any] = {
        "predicted_firing_step": prediction["predicted_firing_step"],
        "predicted_winner": prediction["predicted_winner"],
        "precedence_tie": bool(prediction["precedence_tie"]),
        "local_text_digest": prediction["local_text_digest"],
        "stripped_text_digest": prediction["stripped_text_digest"],
        "server_detok_digest": observation.server_detok_digest,
        "predicted_disposition": prediction["predicted_disposition"],
        "observed_disposition": observed_disposition,
        "fidelity_verdict": verdict,
        "divergence_class": divergence_class,
        "divergence_detail": divergence_detail,
        "hf_stream": None,
    }
    return {
        "request_digest": request_digest,
        "prompt_ids": list(request.prompt_ids),
        "prompt_id_digest": prompt_digest,
        "candidate": candidate,
        "fixture": fixture,
        "cache_position": cache_position,
        "request_config": request_config,
        "returned_ids": list(observation.returned_ids)
        if observation.returned_ids is not None
        else None,
        "returned_id_digest": _ids_digest(observation.returned_ids)
        if observation.returned_ids is not None
        else None,
        "returned_text_digest": observation.returned_text_digest,
        "returned_text_digest_no_special": None,  # vLLM-shaped; always null on llama
        "finish_class": observation.finish_class,
        "stop_winner": observation.stop_winner,
        "cache_n_reported": observation.cache_n_reported,
        "tokens_cached_reported": observation.tokens_cached_reported,
        "tokens_evaluated_reported": observation.tokens_evaluated_reported,
        "cache_creation_tokens_reported": None,  # vLLM-shaped; always null on llama
        "mirror": mirror,
        "cell_detail": cell_detail,
    }


def _absent_record(
    *,
    request: StackRequest,
    identity: StackIdentity,
    candidate: str,
    fixture: str,
    cache_position: str,
    detail: str = CAPABILITY_ABSENT,
) -> dict[str, Any]:
    """A record for a cell that was never executed, with its cause."""
    observation = StackObservation(
        returned_ids=None,
        finish_class="unclassified",
        stop_winner=None,
        cache_n_reported=None,
        tokens_cached_reported=None,
        tokens_evaluated_reported=None,
        returned_text_digest=None,
        server_detok_digest=None,
        config_echo_digest=None,
        detail=detail,
    )
    prediction: dict[str, Any] = {
        "predicted_firing_step": None,
        "predicted_winner": None,
        "stop_steps": {},
        "precedence_tie": False,
        "local_text_digest": _text_digest(""),
        "stripped_text_digest": None,
        "predicted_disposition": None,
    }
    record = build_execution_record(
        request=request,
        observation=observation,
        prediction=prediction,
        identity=identity,
        candidate=candidate,
        fixture=fixture,
        cache_position=cache_position,
    )
    record["mirror"]["fidelity_verdict"] = (
        "capability_absent" if detail == CAPABILITY_ABSENT else "unobservable"
    )
    record["cell_detail"] = detail
    return record


def aggregate_fidelity(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The §9.3 section verdict over a set of execution records.

    Any divergence makes the section ``fidelity_diverge``; otherwise
    any unobservable / unavailable / predictor-mismatch row makes it
    ``fidelity_unavailable`` (partial coverage resolves toward
    unavailable-flavoured honesty); only a fully observed, fully
    agreeing set is ``fidelity_agree``. The predictor-mismatch count
    is its own row, never folded into the divergence count (A7).
    """
    counts = {
        "agree": 0,
        "diverge": 0,
        "unobservable": 0,
        "capability_absent": 0,
        "predictor_detokenization_mismatch": 0,
    }
    classes: dict[str, int] = {}
    total = 0
    for record in records:
        total += 1
        mirror = record.get("mirror") if type(record) is dict else None
        verdict = mirror.get("fidelity_verdict") if type(mirror) is dict else None
        # Only a verdict string can key the counts (S4-1): an
        # unhashable verdict (list/dict) raised TypeError here; it is
        # simply not a verdict and counts as unobservable, like None.
        if type(verdict) is str and verdict in counts:
            counts[verdict] += 1
        else:
            counts["unobservable"] += 1
        if verdict == "diverge":
            divergence_class = mirror.get("divergence_class") if type(mirror) is dict else None
            if type(divergence_class) is str and divergence_class:
                classes[divergence_class] = classes.get(divergence_class, 0) + 1
    if total == 0:
        section_verdict = "fidelity_unavailable"
    elif counts["diverge"]:
        section_verdict = "fidelity_diverge"
    elif (
        counts["unobservable"]
        or counts["capability_absent"]
        or counts["predictor_detokenization_mismatch"]
    ):
        section_verdict = "fidelity_unavailable"
    else:
        section_verdict = "fidelity_agree"
    return {
        "available": total > 0,
        "verdict": section_verdict,
        "counts": counts,
        "divergence_classes": classes,
        "n_records": total,
    }


# ---------------------------------------------------------------------------
# Drivers: single cells, the §5.4 cache sequence, the stage
# ---------------------------------------------------------------------------


def _empty_prediction() -> dict[str, Any]:
    return {
        "predicted_firing_step": None,
        "predicted_winner": None,
        "stop_steps": {},
        "precedence_tie": False,
        "local_text_digest": _text_digest(""),
        "stripped_text_digest": None,
        "predicted_disposition": None,
    }


def execute_cell(
    backend: ExecutionBackend,
    *,
    request: StackRequest,
    decode: Callable[[tuple[int, ...]], str],
    candidate: str,
    fixture: str,
    cache_position: str = "standalone",
) -> dict[str, Any]:
    """Execute one request and assemble its record (predictor included)."""
    identity = backend.identity
    if identity is None:
        raise ValueError("stack_identity_unavailable")
    observation = backend.execute(request)
    if observation.returned_ids is not None:
        prediction = predict_stop_outcome(
            decode=decode,
            returned_ids=observation.returned_ids,
            stop_texts=request.stop_texts,
            eos_id=request.eos_id,
            ignore_eos=request.ignore_eos,
        )
    else:
        prediction = _empty_prediction()
    return build_execution_record(
        request=request,
        observation=observation,
        prediction=prediction,
        identity=identity,
        candidate=candidate,
        fixture=fixture,
        cache_position=cache_position,
    )


def run_cache_sequence(
    backend: LlamacppHttpBackend,
    *,
    cold_ids: tuple[int, ...],
    shared_ids: tuple[int, ...],
    divergent_ids: tuple[int, ...],
    seed: int,
    decode: Callable[[tuple[int, ...]], str],
    candidate: str = "cache-probe",
    on_record: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """The measured cache companion protocol (§5.4), in order.

    Identity re-read at the start (A8(i)); erase; the four arms
    (cold, warm repeat, shared prefix, divergent control); erase
    again; identity re-read at the end. Cold follows C-3: the first
    arm after a completed erase on the probed server — a cold or
    divergent arm reporting reuse above one quantum marks the
    sequence ``cache_protocol_contaminated`` whatever erase returned.
    On a server without slot erase (the fallback posture) all four
    cells are recorded ``capability_absent`` and no arm executes.
    """
    from .core import request_seed_for

    identity = backend.identity
    if identity is None:
        raise ValueError("stack_identity_unavailable")
    prompts = {
        "cold": cold_ids,
        "warm_repeat": cold_ids,
        "shared_prefix": shared_ids,
        "divergent_control": divergent_ids,
    }
    records: list[dict[str, Any]] = []

    def emit(record: dict[str, Any]) -> None:
        records.append(record)
        if on_record is not None:
            on_record(record)

    def request_for(position: str, ids: tuple[int, ...]) -> StackRequest:
        return StackRequest(
            prompt_ids=ids,
            seed=request_seed_for(seed, f"cache:{position}"),
            max_new_tokens=16,
            eos_id=identity.declared_eos_ids[0],
            cache_mode=position,
        )

    posture = identity.config.get("posture")
    erase_available = type(posture) is dict and posture.get("slot_erase_available") is True
    if not erase_available:
        for position, ids in prompts.items():
            emit(
                _absent_record(
                    request=request_for(position, ids),
                    identity=identity,
                    candidate=candidate,
                    fixture=f"cache-{position}",
                    cache_position=position,
                )
            )
        return {
            "attempted": True,
            "available": False,
            "detail": CAPABILITY_ABSENT,
            "records": records,
            "positions": {},
        }
    if not backend.verify_identity():
        return {
            "attempted": True,
            "available": False,
            "detail": "stack_identity_changed_during_run",
            "records": [],
            "positions": {},
        }
    if not backend.erase_slot():
        for position, ids in prompts.items():
            emit(
                _absent_record(
                    request=request_for(position, ids),
                    identity=identity,
                    candidate=candidate,
                    fixture=f"cache-{position}",
                    cache_position=position,
                )
            )
        return {
            "attempted": True,
            "available": False,
            "detail": CAPABILITY_ABSENT,
            "records": records,
            "positions": {},
        }
    for position, ids in prompts.items():
        if cancel is not None and cancel.is_set():
            raise ExecutionCancelled
        emit(
            execute_cell(
                backend,
                request=request_for(position, ids),
                decode=decode,
                candidate=candidate,
                fixture=f"cache-{position}",
                cache_position=position,
            )
        )
    backend.erase_slot()
    detail = ""
    if not backend.verify_identity():
        detail = "stack_identity_changed_during_run"
    by_position = {record["cache_position"]: record for record in records}
    cold_n = by_position["cold"]["cache_n_reported"]
    divergent_n = by_position["divergent_control"]["cache_n_reported"]
    if cold_n is None or divergent_n is None:
        detail = detail or CAPABILITY_ABSENT
    elif cold_n > LLAMA_CACHE_QUANTUM or divergent_n > LLAMA_CACHE_QUANTUM:
        detail = "cache_protocol_contaminated"
    positions = {
        position: {
            "cache_n": record["cache_n_reported"],
            "tokens_cached": record["tokens_cached_reported"],
            "tokens_evaluated": record["tokens_evaluated_reported"],
        }
        for position, record in by_position.items()
    }
    return {
        "attempted": True,
        "available": True,
        "detail": detail,
        "records": records,
        "positions": positions,
    }


# ---------------------------------------------------------------------------
# Incremental record persistence (A8(iv))
#
# While a run executes, the artifact is JSON Lines: a header (with
# the full plan), one line per record as its execution completes
# (flushed and fsynced per record), and a footer on completion. A
# run that dies leaves a valid prefix; the reader reconstructs the
# planned-but-absent cells as cell_not_executed rows. On completion
# the CLI replaces the artifact with the finalized section document;
# readers accept both forms.
# ---------------------------------------------------------------------------


class RecordSink:
    """Append-only JSONL writer for execution records."""

    def __init__(self, path: Path, *, header: Mapping[str, Any]) -> None:
        self._path = Path(path)
        self._handle = self._path.open("x", encoding="utf-8")
        self._bytes = 0
        self._budget_tripped = False
        self._closed = False
        self._write_line({"header": dict(header)})

    def _write_line(self, payload: Mapping[str, Any]) -> None:
        line = canonical_json(payload) + "\n"
        self._handle.write(line)
        self._handle.flush()
        import os

        os.fsync(self._handle.fileno())
        self._bytes += len(line.encode("utf-8"))

    def append(self, record: Mapping[str, Any]) -> bool:
        """Write one record; False once the byte budget is spent.

        The first over-budget record is replaced by a single
        budget-exceeded marker; later records are dropped (the
        driver records them as budget-exceeded cells instead).
        """
        if self._closed:
            raise ValueError("serving_backends_artifact_closed")
        line_bytes = len(canonical_json({"record": dict(record)}).encode("utf-8")) + 1
        if self._bytes + line_bytes > MAX_RECORD_BYTES:
            if not self._budget_tripped:
                self._budget_tripped = True
                self._write_line({"budget_exceeded": True})
            return False
        self._write_line({"record": dict(record)})
        return True

    @property
    def budget_tripped(self) -> bool:
        return self._budget_tripped

    def finalize(self, summary: Mapping[str, Any]) -> None:
        if not self._closed:
            self._write_line({"footer": {"complete": True, **dict(summary)}})
            self._handle.close()
            self._closed = True

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True


def _reject_artifact_constant(value: str) -> Any:
    """``parse_constant`` hook: NaN/Infinity are not strict JSON.

    The ValueError is mapped to the reader's fixed code by both call
    sites in :func:`load_records_artifact`.
    """
    raise ValueError(f"non-finite JSON constant in records artifact: {value!r}")


def load_records_artifact(path: Path) -> dict[str, Any]:
    """Read an exec-backends artifact, finalized or interrupted.

    Returns ``{"finalized", "complete", "header", "records",
    "missing"}`` where ``missing`` carries the planned cells an
    interrupted run never executed, each with its cause.
    """
    # Bounded, strict reading (RTI-04): the read is capped at the
    # sink's own budget plus its envelope allowance, the bytes must
    # be UTF-8, and both JSON parses reject the non-finite constants
    # Python's decoder otherwise accepts. Every failure is the one
    # fixed code -- pre-fix an unbounded read_text leaked
    # UnicodeDecodeError verbatim and accepted NaN literals.
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(_MAX_RECORD_ARTIFACT_BYTES + 1)
    except OSError:
        raise ValueError("serving_backends_artifact_invalid") from None
    if len(raw) > _MAX_RECORD_ARTIFACT_BYTES:
        raise ValueError("serving_backends_artifact_invalid")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("serving_backends_artifact_invalid") from None
    stripped = text.strip()
    if stripped.startswith("{") and "\n" not in stripped:
        try:
            doc = json.loads(stripped, parse_constant=_reject_artifact_constant)
        except ValueError:
            raise ValueError("serving_backends_artifact_invalid") from None
        if type(doc) is dict and type(doc.get("serving_backends")) is dict:
            section = doc["serving_backends"]
            stored_records = section.get("records")
            return {
                "finalized": True,
                "complete": True,
                "header": None,
                "records": stored_records if type(stored_records) is list else [],
                "missing": [],
            }
        raise ValueError("serving_backends_artifact_invalid")
    header: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    complete = False
    budget_exceeded = False
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line, parse_constant=_reject_artifact_constant)
        except ValueError:
            raise ValueError("serving_backends_artifact_invalid") from None
        if type(payload) is not dict:
            raise ValueError("serving_backends_artifact_invalid")
        if type(payload.get("header")) is dict:
            header = payload["header"]
        elif type(payload.get("record")) is dict:
            records.append(payload["record"])
        elif payload.get("budget_exceeded") is True:
            budget_exceeded = True
        elif type(payload.get("footer")) is dict:
            complete = payload["footer"].get("complete") is True
    if header is None:
        raise ValueError("serving_backends_artifact_invalid")
    plan = header.get("plan")
    # Dedup/matching keys are built only from validated string fields
    # (RTI-04 / wave-9 A8): an unhashable key field (a list candidate)
    # raised TypeError here. Such a record can never key a plan cell
    # -- it stays in the records list and its cell, if any, is
    # accounted missing like any other unexecuted cell; a descriptor
    # with non-string key fields is likewise unmatchable by
    # construction and reports missing.
    seen = {
        (record.get("candidate"), record.get("fixture"), record.get("cache_position"))
        for record in records
        if type(record.get("candidate")) is str
        and type(record.get("fixture")) is str
        and type(record.get("cache_position")) is str
    }
    missing: list[dict[str, Any]] = []
    if type(plan) is list:
        for descriptor in plan:
            if type(descriptor) is not dict:
                continue
            key = (
                descriptor.get("candidate"),
                descriptor.get("fixture"),
                descriptor.get("cache_position"),
            )
            if not all(type(part) is str for part in key) or key not in seen:
                detail = EXECUTION_BUDGET_DETAIL if budget_exceeded else "cell_not_executed"
                missing.append({**descriptor, "cell_detail": detail})
    return {
        "finalized": False,
        "complete": complete,
        "header": header,
        "records": records,
        "missing": missing,
    }


class _BudgetSpent(Exception):
    """Internal: the records-artifact byte budget is spent."""


def run_stage(
    backend: LlamacppHttpBackend,
    *,
    plan_cells: list[dict[str, Any]],
    fidelity_input: dict[str, Any] | None,
    cache_prompts: dict[str, tuple[int, ...]] | None,
    seed: int,
    decode: Callable[[tuple[int, ...]], str],
    capabilities: StackCapabilities,
    on_record: Callable[[dict[str, Any]], bool] | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """Run one exec-backends stage: standalone cells, fidelity mode,
    and the cache sequence, in that order, under the run budgets.

    ``on_record`` is the incremental sink (A8(iv)): it is called as
    each record completes and returns False when the artifact byte
    budget is spent, after which every remaining planned cell is
    recorded as a budget-exceeded omission and execution stops.
    """
    identity = backend.identity
    if identity is None:
        raise ValueError("stack_identity_unavailable")
    records: list[dict[str, Any]] = []
    budget_spent = False

    def emit(record: dict[str, Any]) -> None:
        records.append(record)
        if on_record is not None and not on_record(record):
            raise _BudgetSpent

    def budget_record(cell: Mapping[str, Any]) -> dict[str, Any]:
        return _absent_record(
            request=cell["request"],
            identity=identity,
            candidate=str(cell["candidate"]),
            fixture=str(cell["fixture"]),
            cache_position=str(cell.get("cache_position", "standalone")),
            detail=EXECUTION_BUDGET_DETAIL,
        )

    n_executions = 0
    index = 0
    try:
        for index, cell in enumerate(plan_cells):
            if cancel is not None and cancel.is_set():
                raise ExecutionCancelled
            if n_executions >= MAX_EXECUTIONS_PER_RUN:
                for remaining in plan_cells[index:]:
                    records.append(budget_record(remaining))
                budget_spent = True
                index = len(plan_cells)
                break
            n_executions += 1
            emit(
                execute_cell(
                    backend,
                    request=cell["request"],
                    decode=decode,
                    candidate=str(cell["candidate"]),
                    fixture=str(cell["fixture"]),
                    cache_position=str(cell.get("cache_position", "standalone")),
                )
            )
        else:
            index = len(plan_cells)
    except _BudgetSpent:
        budget_spent = True
        for remaining in plan_cells[index + 1 :]:
            records.append(budget_record(remaining))
    fidelity_result: dict[str, Any] | None = None
    if fidelity_input is not None and not budget_spent:
        fidelity_result = diffbackends.fidelity_section_data(
            observations=fidelity_input["observations"],
            stop_texts=fidelity_input["stop_texts"],
            eos_id=fidelity_input["eos_id"],
            min_tokens=fidelity_input["min_tokens"],
            backend=backend,
            decode=decode,
            cancel=cancel,
        )
        if fidelity_result is not None:
            try:
                for record in fidelity_result["records"]:
                    emit(record)
            except _BudgetSpent:
                budget_spent = True
    cache_result: dict[str, Any] | None = None
    if cache_prompts is not None:
        if budget_spent:
            for position in ("cold", "warm_repeat", "shared_prefix", "divergent_control"):
                ids_key = "cold_ids" if position in ("cold", "warm_repeat") else f"{position}_ids"
                records.append(
                    _absent_record(
                        request=StackRequest(
                            prompt_ids=cache_prompts[ids_key],
                            seed=seed,
                            max_new_tokens=16,
                            eos_id=identity.declared_eos_ids[0],
                            cache_mode=position,
                        ),
                        identity=identity,
                        candidate="cache-probe",
                        fixture=f"cache-{position}",
                        cache_position=position,
                        detail=EXECUTION_BUDGET_DETAIL,
                    )
                )
            cache_result = {
                "attempted": True,
                "available": False,
                "detail": EXECUTION_BUDGET_DETAIL,
                "positions": {},
            }
        else:
            try:
                sequence = run_cache_sequence(
                    backend,
                    cold_ids=cache_prompts["cold_ids"],
                    shared_ids=cache_prompts["shared_prefix_ids"],
                    divergent_ids=cache_prompts["divergent_control_ids"],
                    seed=seed,
                    decode=decode,
                    on_record=emit,
                    cancel=cancel,
                )
            except _BudgetSpent:
                budget_spent = True
                sequence = None
            if sequence is not None:
                cache_result = {
                    "attempted": True,
                    "available": bool(sequence["available"]),
                    "detail": str(sequence["detail"]),
                    "positions": sequence["positions"],
                }
            else:
                cache_result = {
                    "attempted": True,
                    "available": False,
                    "detail": EXECUTION_BUDGET_DETAIL,
                    "positions": {},
                }
    return {
        "available": True,
        "detail": EXECUTION_BUDGET_DETAIL if budget_spent else "",
        "identity": identity.as_dict(),
        "capabilities": capabilities.as_dict(),
        "records": records,
        "cache_sequence": cache_result,
        "fidelity": fidelity_result,
        "audit": {
            "n_records": len(records),
            "n_planned_cells": len(plan_cells),
            "n_executions": n_executions,
            "budget_spent": budget_spent,
        },
    }


# ---------------------------------------------------------------------------
# The exec-backends CLI handler (wired in cli.py like diff-backends)
# ---------------------------------------------------------------------------

_TOKENIZER_ASSETS = (
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
)


def _tokenizer_fingerprint(model_dir: Path) -> str:
    """SHA-256 over the manifest of tokenizer assets present.

    Uses the same manifest machinery as the HF backend, so the
    execution identity's tokenizer fingerprint and the generation
    backend's agree on what "the tokenizer" is.
    """
    from .backends import _manifest

    present = tuple(name for name in _TOKENIZER_ASSETS if (model_dir / name).is_file())
    manifest = _manifest(model_dir, present) if present else []
    return "sha256:" + sha256_hex(canonical_json(manifest).encode("utf-8"))


def _parse_id_list(text: str) -> tuple[int, ...] | None:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            return None
        try:
            value = int(part)
        except ValueError:
            return None
        if value < 0:
            return None
        out.append(value)
    return tuple(out)


def _extract_observations(doc: Any) -> list[dict[str, Any]]:
    """Every recorded observation in a run-results document."""
    observations: list[dict[str, Any]] = []
    if type(doc) is not dict:
        return observations
    search = doc.get("search")
    if type(search) is list:
        for entry in search:
            if type(entry) is dict and type(entry.get("observations")) is list:
                observations.extend(obs for obs in entry["observations"] if type(obs) is dict)
    held = doc.get("held_out")
    if type(held) is dict and type(held.get("observations")) is list:
        observations.extend(obs for obs in held["observations"] if type(obs) is dict)
    return observations


def run_exec_backends(args: Any, parser: argparse.ArgumentParser) -> int:
    """Run the ``exec-backends`` subcommand; return the exit code.

    Exit codes (stated identically in the subcommand's help epilog
    and in docs/USAGE.md): 0 when a report was produced (the
    artifact is the output, whatever it finds — including a stage
    that failed closed at a probe gate); 2 for invalid input
    (endpoint refusal, unparsable arguments, an unusable results
    file); 1 when no report could be produced (the tokenizer or
    checkpoint could not be loaded, the artifact could not be
    written, or the run was interrupted before finalization).
    """
    import sys

    from .core import request_seed_for

    out = Path(args.out)
    if out.exists():
        parser.error(f"exec-backends: output artifact already exists: {out}")
    # Every CLI-local validation runs before any expensive setup
    # (RTI-03): the tokenizer load, the checkpoint-bound backend
    # construction and the probe. Pre-fix an invalid invocation paid
    # for all three first -- and a probe failure answered
    # finish_unavailable (exit 0) before an invalid --stops, an
    # unusable --results file or an out-of-range cap was ever looked
    # at, so the invalid input was silently masked by the stage
    # outcome. Each failure below keeps its established exit code.
    stops: tuple[str, ...] = ()
    if args.stops is not None:
        parts = tuple(args.stops.split(","))
        if any(not part for part in parts):
            parser.error("--stops must be comma-separated non-empty stop texts")
        try:
            stops = diffbackends._checked_stop_texts(parts)
        except ValueError as exc:
            print(f"exec-backends: {exc}", file=sys.stderr)
            return 2
    prompt_ids: tuple[int, ...] | None = None
    if args.prompt_ids is not None:
        prompt_ids = _parse_id_list(args.prompt_ids)
        if not prompt_ids:
            parser.error("exec-backends: --prompt-ids must be a comma-separated list of token ids")
        try:
            # Provisional construction validates the request's scalar
            # fields now; the binding construction after the probe
            # resolves the eos default from the probed identity.
            StackRequest(
                prompt_ids=prompt_ids,
                seed=request_seed_for(int(args.seed), "exec:standalone"),
                max_new_tokens=int(args.max_new_tokens),
                stop_texts=stops,
                eos_id=args.eos_id if args.eos_id is not None else 0,
                min_tokens=int(args.min_tokens),
            )
        except ValueError as exc:
            print(f"exec-backends: {exc}", file=sys.stderr)
            return 2
    observations: list[dict[str, Any]] | None = None
    if args.results is not None:
        if not stops:
            parser.error("exec-backends: --results requires at least one --stop")
        from .diffreports import MAX_RESULTS_BYTES

        try:
            with Path(args.results).open("rb") as handle:
                content = handle.read(MAX_RESULTS_BYTES + 1)
            if len(content) > MAX_RESULTS_BYTES:
                raise ValueError("results file exceeds the size budget")
            doc = json.loads(content.decode("utf-8"))
        except (OSError, ValueError):
            print("exec-backends: cannot read the results file", file=sys.stderr)
            return 2
        observations = _extract_observations(doc)
        if not observations:
            parser.error("exec-backends: the results file carries no observations")
    cache_prompts: dict[str, tuple[int, ...]] | None = None
    cache_args = (args.cache_cold_ids, args.cache_shared_ids, args.cache_divergent_ids)
    if any(value is not None for value in cache_args):
        if any(value is None for value in cache_args):
            parser.error(
                "exec-backends: the cache protocol needs --cache-cold-ids, "
                "--cache-shared-ids and --cache-divergent-ids together"
            )
        parsed: dict[str, tuple[int, ...]] = {}
        for key, raw in zip(
            ("cold_ids", "shared_prefix_ids", "divergent_control_ids"),
            cache_args,
            strict=True,
        ):
            ids = _parse_id_list(str(raw))
            if not ids:
                parser.error("exec-backends: --cache-*-ids must be comma-separated token ids")
            parsed[key] = ids
        cache_prompts = parsed
    from .doctor import _build_primary_tokenizer, _loader_failure_code

    model_dir = Path(args.model)
    try:
        tokenizer = _build_primary_tokenizer(model_dir)
        fingerprint = _tokenizer_fingerprint(model_dir)
    except Exception as exc:  # noqa: BLE001 -- loader failures map to exit 1
        print(
            f"exec-backends: tokenizer load failed: {_loader_failure_code(exc)}",
            file=sys.stderr,
        )
        return 1
    try:
        backend = LlamacppHttpBackend(
            args.base_url,
            api_key=args.api_key,
            gguf_path=Path(args.gguf),
            tokenizer=tokenizer,
            tokenizer_fingerprint=fingerprint,
            hardware_class=args.hardware_class,
            timeout_s=float(args.timeout),
        )
    except ValueError as exc:
        print(f"exec-backends: {exc}", file=sys.stderr)
        return 2 if str(exc) == "stack_endpoint_refused" else 1

    from .experiments import serving_backends_analysis

    def finish_unavailable(detail: str) -> int:
        section = serving_backends_analysis({"available": False, "detail": detail})
        doc = {"schema_version": 1, "serving_backends": section}
        try:
            out.write_text(canonical_json(doc) + "\n", encoding="utf-8")
        except OSError:
            print("exec-backends: cannot write the output artifact", file=sys.stderr)
            return 1
        print(canonical_json({"available": False, "detail": detail, "artifact": str(out)}))
        return 0

    try:
        identity = backend.probe()
    except ValueError as exc:
        return finish_unavailable(str(exc))
    plan_cells: list[dict[str, Any]] = []
    if prompt_ids is not None:
        try:
            request = StackRequest(
                prompt_ids=prompt_ids,
                seed=request_seed_for(int(args.seed), "exec:standalone"),
                max_new_tokens=int(args.max_new_tokens),
                stop_texts=stops,
                eos_id=args.eos_id if args.eos_id is not None else identity.declared_eos_ids[0],
                min_tokens=int(args.min_tokens),
            )
        except ValueError as exc:
            print(f"exec-backends: {exc}", file=sys.stderr)
            return 2
        plan_cells.append(
            {
                "candidate": "cli",
                "fixture": "cli",
                "request": request,
                "cache_position": "standalone",
            }
        )
    fidelity_input: dict[str, Any] | None = None
    if observations is not None:
        fidelity_input = {
            "observations": observations,
            "stop_texts": stops,
            "eos_id": args.eos_id if args.eos_id is not None else identity.declared_eos_ids[0],
            "min_tokens": int(args.min_tokens),
        }
    plan_descriptors = [
        {
            "candidate": cell["candidate"],
            "fixture": cell["fixture"],
            "cache_position": cell["cache_position"],
        }
        for cell in plan_cells
    ]
    if cache_prompts is not None:
        plan_descriptors.extend(
            {
                "candidate": "cache-probe",
                "fixture": f"cache-{position}",
                "cache_position": position,
            }
            for position in ("cold", "warm_repeat", "shared_prefix", "divergent_control")
        )
    try:
        sink = RecordSink(
            out,
            header={
                "kind": "exec-backends-records",
                "schema_version": 1,
                "identity": identity.as_dict(),
                "capabilities": backend.capabilities.as_dict(),
                "plan": plan_descriptors,
            },
        )
    except OSError:
        print("exec-backends: cannot write the output artifact", file=sys.stderr)
        return 1
    try:
        stage = run_stage(
            backend,
            plan_cells=plan_cells,
            fidelity_input=fidelity_input,
            cache_prompts=cache_prompts,
            seed=int(args.seed),
            decode=backend.decode_ids,
            capabilities=backend.capabilities,
            on_record=sink.append,
        )
    except (ExecutionCancelled, ValueError) as exc:
        sink.close()
        print(f"exec-backends: run interrupted: {exc}", file=sys.stderr)
        return 1
    sink.finalize({"n_records": len(stage["records"])})
    section = serving_backends_analysis(stage)
    doc = {"schema_version": 1, "serving_backends": section}
    tmp = out.with_name(out.name + ".tmp")
    try:
        tmp.write_text(canonical_json(doc) + "\n", encoding="utf-8")
        tmp.replace(out)
    except OSError:
        print("exec-backends: cannot write the output artifact", file=sys.stderr)
        return 1
    fidelity = section.get("fidelity")
    summary = {
        "available": section.get("available"),
        "detail": section.get("detail"),
        "fidelity_verdict": fidelity.get("verdict") if type(fidelity) is dict else None,
        "n_records": len(stage["records"]),
        "artifact": str(out),
    }
    print(canonical_json(summary))
    return 0
