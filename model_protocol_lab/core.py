"""Immutable conversation formats and raw-token evaluation contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from threading import Event
from typing import Literal, Protocol


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError("Only user/assistant roles are supported")
        if type(self.content) is not str:
            raise ValueError("Message content must be a string")


@dataclass(frozen=True)
class Format:
    name: str
    bos: str
    user_prefix: str
    assistant_prefix: str
    turn_end: str
    repeat_bos: bool = False

    def __post_init__(self) -> None:
        fields = (self.name, self.bos, self.user_prefix,
                  self.assistant_prefix, self.turn_end)
        if any(type(value) is not str for value in fields):
            raise ValueError("Format text fields must be strings")
        if not self.name.strip():
            raise ValueError("Format name must be nonblank")
        if type(self.repeat_bos) is not bool:
            raise ValueError("repeat_bos must be a bool")

    def render(self, messages: tuple[Message, ...]) -> str:
        """repeat_bos means BOS before every message, including the open reply."""
        if (type(messages) is not tuple or not messages
                or any(type(message) is not Message for message in messages)):
            raise ValueError("Messages must be a nonempty tuple of Message")
        if messages[-1].role != "user":
            raise ValueError("This initial renderer requires a final user turn")
        parts = [self.bos]
        for index, message in enumerate(messages):
            expected = "user" if index % 2 == 0 else "assistant"
            if message.role != expected:
                raise ValueError("Messages must alternate, starting with user")
            if index and self.repeat_bos:
                parts.append(self.bos)
            prefix = (self.user_prefix if message.role == "user"
                      else self.assistant_prefix)
            parts.extend((prefix, message.content, self.turn_end))
        if self.repeat_bos:
            parts.append(self.bos)
        parts.append(self.assistant_prefix)
        return "".join(parts)


def local_repairs(base: Format) -> tuple[Format, ...]:
    """Small, explicit search space; no automatic discovery claim."""
    variants = [
        base,
        replace(base, name=base.name + "/toggle-bos", repeat_bos=not base.repeat_bos),
        replace(base, name=base.name + "/no-bos", bos=""),
        replace(base, name=base.name + "/prefix-newline",
                assistant_prefix=base.assistant_prefix + "\n"),
        replace(base, name=base.name + "/prefix-trim",
                assistant_prefix=base.assistant_prefix.rstrip()),
    ]
    unique: dict[tuple[object, ...], Format] = {}
    for candidate in variants:
        key = (candidate.bos, candidate.user_prefix, candidate.assistant_prefix,
               candidate.turn_end, candidate.repeat_bos if candidate.bos else False)
        unique.setdefault(key, candidate)
    return tuple(unique.values())


@dataclass(frozen=True)
class Fixture:
    id: str
    messages: tuple[Message, ...]
    expected: str
    group: str  # Related/paraphrased examples share one group.

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id.strip():
            raise ValueError("Fixture ID must be a nonblank string")
        if type(self.group) is not str or not self.group.strip():
            raise ValueError("Fixture group must be a nonblank string")
        if type(self.expected) is not str:
            raise ValueError("Expected answer must be a string")
        if (type(self.messages) is not tuple or not self.messages
                or any(type(message) is not Message for message in self.messages)):
            raise ValueError("Fixture messages must be a nonempty tuple")


@dataclass(frozen=True)
class Split:
    search: tuple[Fixture, ...]
    held_out: tuple[Fixture, ...]

    def __post_init__(self) -> None:
        if (type(self.search) is not tuple or type(self.held_out) is not tuple
                or not self.search or not self.held_out):
            raise ValueError("Both splits must be nonempty tuples")
        all_cases = self.search + self.held_out
        if any(type(case) is not Fixture for case in all_cases):
            raise ValueError("Splits must contain Fixture values")
        if len({case.id for case in all_cases}) != len(all_cases):
            raise ValueError("Fixture IDs must be globally unique")
        if {c.group for c in self.search} & {c.group for c in self.held_out}:
            raise ValueError("Related fixtures cannot cross the split")
        def message_key(case: Fixture) -> tuple[tuple[str, str], ...]:
            return tuple((message.role, message.content) for message in case.messages)
        if ({message_key(case) for case in self.search}
                & {message_key(case) for case in self.held_out}):
            raise ValueError("Identical message inputs cannot cross the split")


@dataclass(frozen=True)
class Generation:
    token_ids: tuple[int, ...]
    reason: Literal["limit", "cancelled", "error"]
    error: str | None = None


class Backend(Protocol):
    """Raw-token backend contract.

    Encode the complete rendered prompt with no implicit BOS/EOS insertion.
    Generate exactly the cap unless cancelled or failed. Never stop at EOS or
    candidate boundaries. reason='limit' means the cap was reached. Return
    continuation-only token IDs, excluding prompt tokens, as tuples of nonnegative
    integers (not bool). Use a request-local RNG. Check cancellation in inference.
    Error strings are private adapter diagnostics and are never exported.
    Decode without dropping special tokens. Never execute remote model code.
    """

    fingerprint: str

    def encode(self, text: str) -> tuple[int, ...]: ...

    def decode(self, token_ids: tuple[int, ...]) -> str: ...

    def generate(self, input_ids: tuple[int, ...], *, max_new_tokens: int,
                 seed: int, cancel: Event) -> Generation: ...


RunStatus = Literal["completed", "cancelled", "failed"]


@dataclass(frozen=True)
class Observation:
    candidate: str
    fixture: str
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    answer: str
    exact_match: bool
    boundary_found: bool
    raw_reason: str | None
    error: str | None
    status: RunStatus


@dataclass(frozen=True)
class EvaluationReport:
    observations: tuple[Observation, ...]
    planned_count: int
    status: RunStatus

    def __post_init__(self) -> None:
        if type(self.planned_count) is not int or self.planned_count < 1:
            raise ValueError("planned_count must be a positive integer")
        if (type(self.observations) is not tuple
                or any(type(item) is not Observation for item in self.observations)):
            raise ValueError("observations must be a tuple of Observation")
        if self.status not in ("completed", "cancelled", "failed"):
            raise ValueError("Invalid report status")
        if len(self.observations) > self.planned_count:
            raise ValueError("Attempts cannot exceed planned_count")
        for index, observation in enumerate(self.observations):
            if observation.status not in ("completed", "cancelled", "failed"):
                raise ValueError("Invalid observation status")
            if observation.status != "completed":
                if (index != len(self.observations) - 1
                        or observation.status != self.status):
                    raise ValueError("Only the final attempt may be incomplete")
        if self.status == "completed":
            if (len(self.observations) != self.planned_count
                    or any(item.status != "completed" for item in self.observations)):
                raise ValueError("A completed report must cover every fixture")
        if self.status == "failed":
            if not self.observations or self.observations[-1].status != "failed":
                raise ValueError("A failed report needs a failed observation")

    @property
    def attempted_count(self) -> int:
        return len(self.observations)

    @property
    def complete(self) -> bool:
        return self.status == "completed" and self.attempted_count == self.planned_count

    @property
    def aggregate_eligible(self) -> bool:
        return self.complete


def _valid_token_ids(value: object, *, nonempty: bool = False) -> bool:
    return (type(value) is tuple and (not nonempty or bool(value))
            and all(type(token) is int and token >= 0 for token in value))


def _valid_generation(result: object, cap: int) -> bool:
    if type(result) is not Generation or not _valid_token_ids(result.token_ids):
        return False
    if len(result.token_ids) > cap or type(result.reason) is not str:
        return False
    if result.reason not in ("limit", "cancelled", "error"):
        return False
    if result.error is not None and type(result.error) is not str:
        return False
    if result.reason != "error" and result.error is not None:
        return False
    if result.reason == "limit" and len(result.token_ids) != cap:
        return False
    return True


def first_boundary(tokens: tuple[int, ...], stop: tuple[int, ...]) -> int | None:
    if not stop:
        raise ValueError("A nonempty boundary is required")
    for index in range(len(tokens) - len(stop) + 1):
        if tokens[index:index + len(stop)] == stop:
            return index
    return None


def evaluate(backend: Backend, candidate: Format, cases: tuple[Fixture, ...],
             *, boundary: tuple[int, ...], max_new_tokens: int = 64,
             seed: int = 0, cancel: Event | None = None) -> EvaluationReport:
    """Measure a frozen candidate, retaining incomplete-run evidence.

    Only aggregate_eligible reports may be ranked or aggregated. Exact match and
    boundary presence are separate. A limit result has exactly the requested cap.
    No backend exception text is exported. Total compute accounting, confidence,
    provenance and candidate selection remain separate unimplemented features.
    Boundary IDs must be verified in context; suffix tokens remain in evidence.
    """
    if type(candidate) is not Format:
        raise ValueError("candidate must be a Format")
    if (type(cases) is not tuple or not cases
            or any(type(case) is not Fixture for case in cases)):
        raise ValueError("cases must be a nonempty tuple of Fixture")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Evaluation fixture IDs must be unique")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("Output cap must be an integer between 1 and 4096")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    event = cancel if cancel is not None else Event()
    records: list[Observation] = []

    def report(status: RunStatus) -> EvaluationReport:
        return EvaluationReport(tuple(records), len(cases), status)

    for case in cases:
        if event.is_set():
            return report("cancelled")
        input_ids: tuple[int, ...] = ()
        output_ids: tuple[int, ...] = ()
        answer = ""
        raw_reason: str | None = None
        cut: int | None = None

        def record(status: RunStatus, error: str | None = None) -> None:
            records.append(Observation(
                candidate=candidate.name, fixture=case.id, input_ids=input_ids,
                output_ids=output_ids, answer=answer,
                exact_match=status == "completed" and answer == case.expected,
                boundary_found=cut is not None, raw_reason=raw_reason,
                error=error, status=status,
            ))

        def terminate(status: RunStatus, error: str | None = None) -> EvaluationReport:
            record(status, error)
            return report(status)

        try:
            prompt = candidate.render(case.messages)
        except Exception:
            return terminate("failed", "render_failed")
        if event.is_set():
            return terminate("cancelled")
        try:
            encoded = backend.encode(prompt)
        except Exception:
            return terminate("failed", "encode_failed")
        if not _valid_token_ids(encoded):
            return terminate("failed", "invalid_input_ids")
        input_ids = encoded
        if event.is_set():
            return terminate("cancelled")
        material = f"{seed}\0{case.id}".encode("utf-8", errors="surrogatepass")
        request_seed = int.from_bytes(sha256(material).digest()[:4], "big")
        try:
            result = backend.generate(input_ids, max_new_tokens=max_new_tokens,
                                      seed=request_seed, cancel=event)
        except Exception:
            return terminate("failed", "generate_failed")
        if not _valid_generation(result, max_new_tokens):
            return terminate("failed", "invalid_generation")
        output_ids = result.token_ids
        raw_reason = result.reason
        cut = first_boundary(output_ids, boundary)
        if result.reason == "error":
            return terminate("failed", "backend_reported_error")
        if result.reason == "cancelled" or event.is_set():
            return terminate("cancelled")
        answer_ids = output_ids if cut is None else output_ids[:cut]
        try:
            decoded = backend.decode(answer_ids)
        except Exception:
            return terminate("failed", "decode_failed")
        if type(decoded) is not str:
            return terminate("failed", "decode_failed")
        answer = decoded
        if event.is_set():
            return terminate("cancelled")
        record("completed")
    return report("cancelled" if event.is_set() else "completed")


