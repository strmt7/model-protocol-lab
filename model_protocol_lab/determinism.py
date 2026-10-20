"""Determinism self-audit (``verify --determinism``).

Re-executes a recorded run's search observations against a locally
loaded backend and compares the replayed token IDs with the recorded
``output_ids`` exactly, cell by cell. The audit emits a determinism
certificate when every claimed cell reproduces, or a per-cell
divergence table when some do not.

What the audit proves is deliberately narrow (the full contract is
``docs/DETERMINISM_AUDIT_DESIGN.md``): it certifies the re-execution
in this audit run, never the original run; a divergence is a
backend-determinism finding, not an instrument failure and not an
artifact-integrity failure; and the certificate is unsigned and
self-issued. Recorded cells are validated against the recording
pipeline's own invariants before any blame is assigned, so a tampered
or foreign artifact reads as an artifact defect, never as backend
nondeterminism. The audit never raises on malformed input: every
defect lands in a fixed detail vocabulary, and backend error text is
classified through the same allowlist ``core`` applies to recorded
observations, never exported raw.
"""

from __future__ import annotations

import hashlib
import json
from threading import Event
from typing import Any

from .core import _BACKEND_ERROR_CODES, _valid_generation, request_seed_for

#: Largest search phase ``run_experiment`` can record: 24 candidates
#: (its candidate cap) x 100 search fixtures (its fixture cap). A
#: foreign artifact can carry far more recorded cells than any real
#: run; the audit attempts at most this many and reports the rest as
#: budget skips, so the audit never costs more than re-running the
#: largest experiment the lab itself can produce.
MAX_AUDIT_CELLS = 2400

_REPRODUCED = "reproduced"
_DIVERGED = "diverged"
_ERROR = "error"
_SKIPPED = "skipped"

#: Skip details that mean "a cell the artifact claims as a completed
#: generation could not be audited". Any one of them caps the verdict
#: at "inconclusive" -- a certificate over a partial cell set that hid
#: the gap would be dishonest. ``observation_not_completed`` is
#: deliberately absent: the original run never claimed that cell.
_BLOCKING_SKIP_DETAILS = frozenset(
    {
        "recorded_length_inconsistent",
        "recorded_ids_out_of_vocab",
        "replay_context_limit_exceeded",
        "audit_cell_budget_exceeded",
    }
)


def compare_token_ids(recorded: tuple[int, ...], replayed: tuple[int, ...]) -> dict[str, Any]:
    """Compare two token sequences exactly (pure).

    ``first_divergence`` is the index of the first position where the
    sequences differ, or the shared-prefix length when one sequence is
    a strict prefix of the other; it is ``None`` on an exact match.
    """
    shared = 0
    for recorded_id, replayed_id in zip(recorded, replayed, strict=False):
        if recorded_id != replayed_id:
            break
        shared += 1
    match = shared == len(recorded) == len(replayed)
    return {
        "match": match,
        "first_divergence": None if match else shared,
        "shared_prefix": shared,
        "recorded_length": len(recorded),
        "replayed_length": len(replayed),
    }


def _id_tuple(value: Any, *, nonempty: bool) -> tuple[int, ...] | None:
    """Recorded IDs as stored in JSON, validated and tupled, else None.

    Mirrors the strictness of ``core._valid_token_ids`` for the stored
    (list) form: booleans are not token IDs, and a completed cell's
    input is never empty. Returning the tuple (rather than a bool)
    keeps the validated value narrowed for the caller.
    """
    if (
        type(value) is not list
        or (nonempty and not value)
        or not all(type(token) is int and token >= 0 for token in value)
    ):
        return None
    return tuple(value)


def _entry_name(entry: dict[str, Any]) -> str | None:
    # Kept local, mirroring verify._entry_name, so this module does not
    # import the verify CLI layer.
    fmt = entry.get("format")
    name = fmt.get("name") if type(fmt) is dict else None
    return name if type(name) is str and name else None


def _header_problem(data: Any) -> str | None:
    """The artifact fields the audit depends on, validated fail-closed."""
    if type(data) is not dict:
        return "artifact_malformed"
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        return "unsupported_schema_version"
    seed = data.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**64:
        return "artifact_malformed"
    cap = data.get("max_new_tokens")
    if type(cap) is not int or not 1 <= cap <= 4096:
        return "artifact_malformed"
    fingerprint = data.get("backend_fingerprint")
    if type(fingerprint) is not str or not fingerprint:
        return "artifact_malformed"
    if type(data.get("search")) is not list:
        return "artifact_malformed"
    return None


def _search_incomplete(data: dict[str, Any]) -> bool:
    """True when the artifact records a search that never completed.

    Mirrors verify's replay rule: an aborted run either carries the
    abort decision itself or leaves a non-aggregate-eligible entry.
    """
    if data.get("decision") in ("cancelled", "failed"):
        return True
    for entry in data["search"]:
        summary = entry.get("summary") if type(entry) is dict else None
        if type(summary) is not dict or not summary.get("aggregate_eligible"):
            return True
    return False


def _plan(
    data: Any,
) -> tuple[
    str | None,
    list[dict[str, Any]],
    list[tuple[dict[str, Any], tuple[int, ...], tuple[int, ...]]],
    dict[str, Any],
]:
    """Classify every recorded search observation before any replay.

    Returns ``(problem, cells, pending, meta)``. ``problem`` is a
    global detail code when the audit cannot run at all; otherwise
    ``cells`` holds one output dict per recorded observation (with
    pre-replay skips already assigned) and ``pending`` the cells still
    to re-execute, each with its recorded ID tuples.
    """
    meta: dict[str, Any] = {"n_entries_stripped": 0, "search_incomplete": False}
    problem = _header_problem(data)
    if problem is not None:
        return problem, [], [], meta
    cap = data["max_new_tokens"]
    cells: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], tuple[int, ...], tuple[int, ...]]] = []
    entries = data["search"]
    for index, entry in enumerate(entries):
        if type(entry) is not dict:
            return "artifact_malformed", [], [], meta
        candidate = _entry_name(entry)
        if "observations" not in entry:
            meta["n_entries_stripped"] += 1
            continue
        observations = entry["observations"]
        if type(observations) is not list:
            return "artifact_malformed", [], [], meta
        for observation in observations:
            cell: dict[str, Any] = {
                "entry": index,
                "candidate": candidate,
                "fixture": None,
                "status": _SKIPPED,
                "detail": "",
            }
            cells.append(cell)
            if type(observation) is not dict:
                cell["detail"] = "recorded_ids_invalid"
                continue
            fixture = observation.get("fixture")
            if type(fixture) is str:
                cell["fixture"] = fixture
            input_ids = _id_tuple(observation.get("input_ids"), nonempty=True)
            output_ids = _id_tuple(observation.get("output_ids"), nonempty=False)
            if type(fixture) is not str or input_ids is None or output_ids is None:
                cell["detail"] = "recorded_ids_invalid"
                continue
            if observation.get("status") != "completed":
                cell["detail"] = "observation_not_completed"
                continue
            if len(output_ids) != cap:
                # A completed generation reached the token cap exactly
                # (core._valid_generation's limit rule), so a stored
                # completed cell of any other length was not recorded
                # by this pipeline: an artifact defect, not divergence.
                cell["detail"] = "recorded_length_inconsistent"
                continue
            pending.append((cell, input_ids, output_ids))
    meta["search_incomplete"] = _search_incomplete(data)
    if entries and meta["n_entries_stripped"] == len(entries):
        return "observations_stripped", cells, pending, meta
    if not pending:
        return "no_replayable_cells", cells, pending, meta
    return None, cells, pending, meta


def precheck(data: Any) -> str | None:
    """Global audit availability for an artifact, without any backend.

    Returns a fixed detail code (``artifact_malformed``,
    ``unsupported_schema_version``, ``observations_stripped``,
    ``no_replayable_cells``) or ``None`` when the audit can run.
    ``verify`` consults this before constructing a backend, so a
    malformed artifact never triggers a checkpoint load.
    """
    problem, _cells, _pending, _meta = _plan(data)
    return problem


def _backend_vocab(backend: Any) -> int | None:
    """The backend's vocabulary bound, when it exposes one.

    ``LocalHFBackend.vocab_size`` is a property; other backends may
    expose a method or nothing. An unreadable bound means the audit
    cannot vocab-validate recorded IDs and proceeds without that
    check rather than failing closed on its own instrumentation.
    """
    try:
        value = getattr(backend, "vocab_size", None)
        if callable(value):
            value = value()
    except Exception:  # noqa: BLE001 -- a hostile double must not crash the audit
        return None
    return value if type(value) is int and value > 0 else None


def _certificate(
    data: dict[str, Any],
    digest_records: list[dict[str, Any]],
    *,
    n_reproduced: int,
    n_not_completed: int,
    replay: dict[str, Any] | None,
) -> dict[str, Any]:
    """The determinism certificate: unsigned, self-issued, cell-bound.

    The digest covers exactly the attempted cells in recorded order,
    canonically encoded (``ensure_ascii=True`` is load-bearing: it
    keeps lone-surrogate fixture IDs encodable instead of raising).
    """
    payload = json.dumps(digest_records, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    certificate: dict[str, Any] = {
        "kind": "determinism_certificate",
        "unsigned": True,
        "issuer": "self",
        "claim": (
            "every re-executed recorded search observation reproduced "
            "token-identically in this audit run"
        ),
        "scope": "search observations only; held-out observations were not re-executed",
        "certifies": "the re-execution in this audit run, not the original run",
        "not_certified": (
            "recorded prompts are replayed as stored; their authenticity "
            "against the fixture files and format definitions is not certified"
        ),
        "cells_digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "backend_fingerprint": data["backend_fingerprint"],
        "seed": data["seed"],
        "max_new_tokens": data["max_new_tokens"],
        "n_reproduced": n_reproduced,
        "n_not_completed_in_original": n_not_completed,
    }
    if replay is not None:
        certificate["replay"] = replay
    return certificate


def audit_determinism(
    data: Any, backend: Any, *, replay: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Re-execute an artifact's recorded search cells against ``backend``.

    ``backend`` is any object satisfying the ``core.Backend`` protocol
    (only ``generate`` and ``fingerprint`` are used, plus ``vocab_size``
    when exposed). ``replay`` is operator context (device, max-context)
    echoed into the report and the certificate so the claim's scope is
    on the record. Never raises on malformed artifacts; backend
    exceptions become per-cell ``error`` outcomes.
    """
    problem, cells, pending, meta = _plan(data)
    n_entries = (
        len(data["search"]) if type(data) is dict and type(data.get("search")) is list else 0
    )
    result: dict[str, Any] = {
        "available": False,
        "detail": problem or "",
        "verdict": None,
        "recorded_backend_fingerprint": (
            data.get("backend_fingerprint") if type(data) is dict else None
        ),
        "replay_backend_fingerprint": None,
        "seed": data.get("seed") if type(data) is dict else None,
        "max_new_tokens": data.get("max_new_tokens") if type(data) is dict else None,
        "replay": replay,
        "n_entries": n_entries,
        "n_entries_stripped": meta["n_entries_stripped"],
        "search_incomplete": meta["search_incomplete"],
        "n_cells": len(cells),
        "n_attempted": 0,
        "n_reproduced": 0,
        "n_diverged": 0,
        "n_error": 0,
        "n_skipped": 0,
        "cells": cells,
        "certificate": None,
    }
    if problem is not None:
        result["n_skipped"] = len(cells)
        return result
    recorded_fingerprint = data["backend_fingerprint"]
    replay_fingerprint = getattr(backend, "fingerprint", None)
    result["replay_backend_fingerprint"] = (
        replay_fingerprint if type(replay_fingerprint) is str else None
    )
    if type(replay_fingerprint) is not str or replay_fingerprint != recorded_fingerprint:
        # A different checkpoint (or device, or context limit -- all
        # bound into the fingerprint) is a different experiment, never
        # nondeterminism. No generate call has happened yet.
        result["detail"] = "backend_fingerprint_mismatch"
        result["n_skipped"] = len(cells)
        return result
    result["available"] = True
    result["detail"] = ""
    seed = data["seed"]
    cap = data["max_new_tokens"]
    vocab = _backend_vocab(backend)
    event = Event()
    digest_records: list[dict[str, Any]] = []
    attempted = 0
    for cell, input_ids, output_ids in pending:
        if vocab is not None and (max(input_ids) >= vocab or max(output_ids) >= vocab):
            # IDs this backend cannot consume or emit: the artifact
            # cannot record a genuine generation by it. Artifact
            # defect, reported as a skip -- never as divergence.
            cell["detail"] = "recorded_ids_out_of_vocab"
            continue
        if attempted >= MAX_AUDIT_CELLS:
            cell["detail"] = "audit_cell_budget_exceeded"
            continue
        attempted += 1
        digest_records.append(
            {
                "candidate": cell["candidate"],
                "fixture": cell["fixture"],
                "input_ids": list(input_ids),
                "output_ids": list(output_ids),
            }
        )
        # The fixture ID of a pending cell is always a string: the
        # planner assigned recorded_ids_invalid otherwise.
        request_seed = request_seed_for(seed, str(cell["fixture"]))
        try:
            generation = backend.generate(
                input_ids, max_new_tokens=cap, seed=request_seed, cancel=event
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # noqa: BLE001 -- a misbehaving backend is a cell outcome
            cell["status"] = _ERROR
            cell["detail"] = "generate_failed"
            continue
        if not _valid_generation(generation, cap):
            cell["status"] = _ERROR
            cell["detail"] = "invalid_generation"
            continue
        if generation.reason == "error":
            code = (
                generation.error
                if type(generation.error) is str and generation.error in _BACKEND_ERROR_CODES
                else "backend_reported_error"
            )
            if code == "context_limit_exceeded":
                cell["detail"] = "replay_context_limit_exceeded"
            else:
                cell["status"] = _ERROR
                cell["detail"] = code
            continue
        if generation.reason == "cancelled":
            # The audit's cancel event is never set, so a cancelled
            # replay is a backend contract surprise, not an outcome.
            cell["status"] = _ERROR
            cell["detail"] = "backend_reported_cancelled"
            continue
        comparison = compare_token_ids(output_ids, generation.token_ids)
        cell["recorded_length"] = comparison["recorded_length"]
        cell["replayed_length"] = comparison["replayed_length"]
        if comparison["match"]:
            cell["status"] = _REPRODUCED
        else:
            cell["status"] = _DIVERGED
            cell["first_divergence"] = comparison["first_divergence"]
            cell["shared_prefix"] = comparison["shared_prefix"]
    n_reproduced = sum(1 for c in cells if c["status"] == _REPRODUCED)
    n_diverged = sum(1 for c in cells if c["status"] == _DIVERGED)
    n_error = sum(1 for c in cells if c["status"] == _ERROR)
    blocking = sum(1 for c in cells if c["detail"] in _BLOCKING_SKIP_DETAILS)
    result["n_attempted"] = attempted
    result["n_reproduced"] = n_reproduced
    result["n_diverged"] = n_diverged
    result["n_error"] = n_error
    result["n_skipped"] = len(cells) - n_reproduced - n_diverged - n_error
    if n_diverged:
        result["verdict"] = "diverged"
    elif n_error or blocking or meta["n_entries_stripped"] or meta["search_incomplete"]:
        result["verdict"] = "inconclusive"
    else:
        result["verdict"] = "deterministic"
        n_not_completed = sum(1 for c in cells if c["detail"] == "observation_not_completed")
        result["certificate"] = _certificate(
            data,
            digest_records,
            n_reproduced=n_reproduced,
            n_not_completed=n_not_completed,
            replay=replay,
        )
    return result
