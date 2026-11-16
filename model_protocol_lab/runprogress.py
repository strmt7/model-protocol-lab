"""Crash-safe progress journal for experiment runs (W4-3, RR-14b).

A ``run`` can take hours; before W4, a process that died mid-run
(host replacement, kill -9, OOM) lost every completed cell, because
the report pair is written only at the end. This module journals
observations to ``progress.jsonl`` inside the claimed output
directory as they complete, so ``run --resume`` can re-execute only
the missing cells.

Journal format (one canonical-JSON object per line):

* header — ``{"header": {...}}`` binding the run's identity:
  the progress schema, the package version, and the configuration
  digest (audit adjudication D-3). The budget cap is deliberately
  NOT part of the identity: the resuming invocation's cap governs,
  enforced by the same full-run preflight bound as a fresh run.
* records — ``{"stage", "candidate", "observation",
  "cell_sha256"}``; the hash chains over the previous line's hash
  and the record body, with the header line's hash as genesis, so
  a tampered or reordered middle line breaks verification.
* footer — ``{"footer": {"complete": true, ...}}``, written once
  the report pair is saved.

Verification on load is total and refuses by fixed code:
``resume_configuration_mismatch`` (schema, package version or
digest differs), ``resume_progress_corrupt`` (chain break, bad
record shape, records after a terminal observation, a record
after the footer), ``resume_stage_order_violation`` (held-out
records ahead of a complete search stage — an order no real run
can produce). A torn LAST line — the kill landed mid-write — is
not corruption: it is dropped, counted in ``torn_lines``, and
exactly that cell re-executes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .core import Format, Observation, Split

PROGRESS_SCHEMA = "mpl-progress-1"
PROGRESS_FILENAME = "progress.jsonl"

_RECORD_FIELDS = frozenset(
    {
        "candidate",
        "fixture",
        "input_ids",
        "output_ids",
        "answer",
        "exact_match",
        "boundary_found",
        "boundary_text_match",
        "raw_reason",
        "error",
        "status",
        "input_text",
    }
)
_STATUSES = frozenset({"completed", "cancelled", "failed"})


def _canonical(payload: Any) -> str:
    """The journal's canonical JSON form (sorted keys, compact, ASCII)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _chain_hash(previous: str, body: dict[str, Any]) -> str:
    """Hash one record body onto the chain head."""
    return hashlib.sha256((previous + _canonical(body)).encode("utf-8")).hexdigest()


class LoadedProgress:
    """A verified journal: reusable observations plus resume state."""

    def __init__(
        self,
        *,
        observations: dict[tuple[str, str, str], Observation],
        footer: dict[str, Any] | None,
        torn_lines: int,
        valid_bytes: int,
        last_hash: str,
    ) -> None:
        """Init."""
        self.observations = observations
        self.footer = footer
        self.torn_lines = torn_lines
        self.valid_bytes = valid_bytes
        self.last_hash = last_hash


class ProgressSink:
    """Append-only journal writer (flush + fsync per line).

    Mirrors the serving stack's RecordSink discipline: a record is
    durable before the run moves on, so a kill between cells loses
    at most the cell in flight.
    """

    def __init__(self, path: Path, *, mode: str, last_hash: str, records: int) -> None:
        """Init."""
        self._path = Path(path)
        self._handle = self._path.open(mode, encoding="utf-8")
        self._last_hash = last_hash
        self._records = records
        self._closed = False

    @classmethod
    def create(cls, path: Path, *, header: dict[str, Any]) -> ProgressSink:
        """Create a fresh journal and write its identity header."""
        sink = cls(Path(path), mode="x", last_hash="", records=0)
        line = _canonical({"header": dict(header)})
        sink._write_raw(line)
        sink._last_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
        return sink

    @classmethod
    def resume(cls, path: Path, loaded: LoadedProgress) -> ProgressSink:
        """Reopen a verified journal for appending.

        Truncates a torn tail first; refuses a finalized journal
        (its run completed — the CLI reports that state as
        ``resume_already_complete`` before reaching here).
        """
        if loaded.footer is not None:
            raise ValueError("resume_already_complete")
        with Path(path).open("r+b") as handle:
            handle.truncate(loaded.valid_bytes)
        return cls(
            Path(path),
            mode="a",
            last_hash=loaded.last_hash,
            records=len(loaded.observations),
        )

    @property
    def records_written(self) -> int:
        """Records in the journal, including any resumed prefix."""
        return self._records

    def _write_raw(self, line: str) -> None:
        """Write raw."""
        self._handle.write(line + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def append(self, stage: str, candidate: str, observation: Observation) -> None:
        """Journal one freshly produced observation."""
        if self._closed:
            raise ValueError("progress_journal_closed")
        body = {
            "stage": stage,
            "candidate": candidate,
            "observation": asdict(observation),
        }
        digest = _chain_hash(self._last_hash, body)
        self._write_raw(_canonical({**body, "cell_sha256": digest}))
        self._last_hash = digest
        self._records += 1

    def finalize(self, summary: dict[str, Any]) -> None:
        """Write the completion footer and close."""
        if not self._closed:
            self._write_raw(_canonical({"footer": {"complete": True, **summary}}))
            self._handle.close()
            self._closed = True

    def close(self) -> None:
        """Close without a footer (the run did not complete)."""
        if not self._closed:
            self._handle.close()
            self._closed = True


def _reconstruct_observation(payload: Any) -> Observation:
    """Rebuild one Observation from its journal dict, strictly."""
    if type(payload) is not dict or frozenset(payload) != _RECORD_FIELDS:
        raise ValueError("resume_progress_corrupt")
    for field in ("input_ids", "output_ids"):
        ids = payload[field]
        if type(ids) is not list or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError("resume_progress_corrupt")
    if (
        type(payload["candidate"]) is not str
        or type(payload["fixture"]) is not str
        or type(payload["answer"]) is not str
        or type(payload["exact_match"]) is not bool
        or type(payload["boundary_found"]) is not bool
        or not (
            payload["boundary_text_match"] is None or type(payload["boundary_text_match"]) is bool
        )
        or not (payload["raw_reason"] is None or type(payload["raw_reason"]) is str)
        or not (payload["error"] is None or type(payload["error"]) is str)
        or payload["status"] not in _STATUSES
        or type(payload["input_text"]) is not str
    ):
        raise ValueError("resume_progress_corrupt")
    return Observation(
        candidate=payload["candidate"],
        fixture=payload["fixture"],
        input_ids=tuple(payload["input_ids"]),
        output_ids=tuple(payload["output_ids"]),
        answer=payload["answer"],
        exact_match=payload["exact_match"],
        boundary_found=payload["boundary_found"],
        boundary_text_match=payload["boundary_text_match"],
        raw_reason=payload["raw_reason"],
        error=payload["error"],
        status=payload["status"],
        input_text=payload["input_text"],
    )


def load_progress(
    path: Path,
    *,
    expected_digest: str,
    expected_package_version: str,
    formats: tuple[Format, ...],
    split: Split,
) -> LoadedProgress:
    """Read and fully verify a journal before any cell is reused.

    The expected plan is derived from ``formats``/``split`` (whose
    content the configuration digest binds), never from the file:
    search records must form an exact prefix of the plan order,
    and held-out records may appear only after a complete,
    all-completed search stage.
    """
    raw = Path(path).read_bytes()
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    if not lines:
        raise ValueError("resume_progress_corrupt")
    try:
        first = json.loads(lines[0])
    except (ValueError, UnicodeDecodeError):
        raise ValueError("resume_progress_corrupt") from None
    if type(first) is not dict or frozenset(first) != {"header"}:
        raise ValueError("resume_progress_corrupt")
    header = first["header"]
    if type(header) is not dict or header.get("schema") != PROGRESS_SCHEMA:
        raise ValueError("resume_configuration_mismatch")
    if header.get("package_version") != expected_package_version:
        raise ValueError("resume_configuration_mismatch")
    if header.get("configuration_digest") != expected_digest:
        raise ValueError("resume_configuration_mismatch")

    search_plan = [(fmt.name, case.id) for fmt in formats for case in split.search]
    held_out_ids = [case.id for case in split.held_out]
    candidate_names = {fmt.name for fmt in formats}
    observations: dict[tuple[str, str, str], Observation] = {}
    last_hash = hashlib.sha256(lines[0]).hexdigest()
    valid_bytes = len(lines[0]) + 1
    torn_lines = 0
    footer: dict[str, Any] | None = None
    search_index = 0
    held_candidate: str | None = None
    held_index = 0
    search_terminal = False

    for index, raw_line in enumerate(lines[1:], start=1):
        is_last = index == len(lines) - 1
        try:
            entry = json.loads(raw_line)
        except (ValueError, UnicodeDecodeError):
            if is_last:
                torn_lines += 1
                break
            raise ValueError("resume_progress_corrupt") from None
        if type(entry) is dict and frozenset(entry) == {"footer"}:
            if not is_last or type(entry["footer"]) is not dict:
                raise ValueError("resume_progress_corrupt")
            if entry["footer"].get("complete") is not True:
                raise ValueError("resume_progress_corrupt")
            footer = entry["footer"]
            valid_bytes += len(raw_line) + 1
            continue
        if footer is not None:
            raise ValueError("resume_progress_corrupt")
        if (
            type(entry) is not dict
            or frozenset(entry) != {"stage", "candidate", "observation", "cell_sha256"}
            or entry["cell_sha256"]
            != _chain_hash(
                last_hash,
                {
                    "stage": entry["stage"],
                    "candidate": entry["candidate"],
                    "observation": entry["observation"],
                },
            )
        ):
            raise ValueError("resume_progress_corrupt")
        stage = entry["stage"]
        candidate = entry["candidate"]
        observation = _reconstruct_observation(entry["observation"])
        if observation.candidate != candidate:
            raise ValueError("resume_progress_corrupt")
        if search_terminal:
            raise ValueError("resume_progress_corrupt")
        if stage == "search":
            if search_index >= len(search_plan) or search_plan[search_index] != (
                candidate,
                observation.fixture,
            ):
                raise ValueError("resume_progress_corrupt")
            search_index += 1
            if observation.status != "completed":
                search_terminal = True
        elif stage == "held_out":
            if search_index != len(search_plan):
                raise ValueError("resume_stage_order_violation")
            if candidate not in candidate_names:
                raise ValueError("resume_progress_corrupt")
            if held_candidate is None:
                held_candidate = candidate
            if candidate != held_candidate:
                raise ValueError("resume_progress_corrupt")
            if held_index >= len(held_out_ids) or held_out_ids[held_index] != (observation.fixture):
                raise ValueError("resume_progress_corrupt")
            held_index += 1
            if observation.status != "completed":
                search_terminal = True
        else:
            raise ValueError("resume_progress_corrupt")
        key = (stage, candidate, observation.fixture)
        if key in observations:
            raise ValueError("resume_progress_corrupt")
        observations[key] = observation
        last_hash = entry["cell_sha256"]
        valid_bytes += len(raw_line) + 1

    return LoadedProgress(
        observations=observations,
        footer=footer,
        torn_lines=torn_lines,
        valid_bytes=valid_bytes,
        last_hash=last_hash,
    )
