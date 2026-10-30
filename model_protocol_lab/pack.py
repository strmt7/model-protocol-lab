"""Sealed result bundles (``model-protocol-lab pack`` / ``verify --bundle``).

Seals a report directory into one hash-bound ZIP bundle so an auditor can
later check that the files are unchanged since packing, offline, and then
re-derive the experiment's decision from the sealed ``results.json`` via
:func:`model_protocol_lab.verify.verify_results` as an independent second
layer.

The seal is hash-based and unsigned. It proves bit-integrity since
packing **only when the bundle digest is compared against a digest
obtained through an independent channel** (the hex ``pack`` prints, or
``verify --bundle --expect-digest``). It does not authenticate origin,
does not prove the experiment was run, and carries no freshness or
rollback protection. The full format contract lives in
``docs/SEALED_BUNDLE_DESIGN.md``.

No model is loaded and no network is used; packing and verification are
deterministic.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import lzma
import os
import re
import stat
import sys
import zipfile
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import __version__
from .catalog import CATALOG
from .verify import _display_id

MONDAY_MORNING_USE = (
    "an auditor asks to prove a run's result files are unchanged since "
    "packing — pack seals the report directory into one hash-bound "
    "bundle and verify --bundle re-checks it offline"
)

FORMAT_ID = "mpl-bundle/1"
MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.json"
_COMMENT_PREFIX = b"mpl-bundle-sha256:"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PATH_BAD_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")

# Caps (docs/SEALED_BUNDLE_DESIGN.md, "Limits"). Enforcement order is part
# of the contract: outer size from fstat before reading, entry count from
# the EOCD pre-parse before ZipFile construction, per-entry and total
# uncompressed sizes from the central directory before any entry is read.
MAX_BUNDLE_ZIP_BYTES = 288 * 1024 * 1024
MAX_BUNDLE_ENTRIES = 1024
MAX_BUNDLE_FILE_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_TOTAL_BYTES = 256 * 1024 * 1024

_READ_CHUNK = 1 << 20
_EOCD_SIGNATURE = b"PK\x05\x06"
_EOCD_MIN_BYTES = 22
_MAX_COMMENT_BYTES = 65535

_SEAL_CHECKS = (
    "seal_container",
    "seal_manifest_schema",
    "seal_comment_digest",
    "seal_file_set",
    "seal_file_hashes",
)


def _record(checks: list[dict[str, str]], check_id: str, status: str, detail: str = "") -> None:
    """Record."""
    checks.append({"id": check_id, "status": status, "detail": detail})


def _sha256_hex(data: bytes) -> str:
    """Return the sha256 hex."""
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical json bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def catalog_digest() -> str:
    """SHA-256 over the canonical JSON of the catalog's definitions.

    The digest covers the template definitions themselves (each family's
    ``Format`` fields), never source text, so a comment edit in
    ``catalog.py`` does not move it and a delimiter change always does.
    """
    definitions = {family: asdict(fmt) for family, fmt in CATALOG.items()}
    return _sha256_hex(_canonical_json_bytes(definitions))


def _entry_type_not_regular(info: zipfile.ZipInfo) -> bool:
    """True when the entry's stored Unix file type is explicitly NOT
    a regular file — symlink, FIFO, device, socket, or a directory
    entry whose name lacks the trailing slash the name check keys
    on (oversight security S-8; the decision is recorded in
    docs/SEALED_BUNDLE_DESIGN.md). Entries from non-Unix creators,
    and Unix entries whose type bits were never set, make no type
    claim and are judged by the name/content checks alone; the
    lab's own writer leaves the type bits unset.
    """
    if info.create_system != 3:
        return False
    file_type = stat.S_IFMT(info.external_attr >> 16)
    return file_type not in (0, stat.S_IFREG)


def _valid_segment(segment: str) -> bool:
    """One canonical path segment: non-empty, not ``.``/``..``, no
    separator, backslash, drive colon at path start handled by the caller,
    and no control characters. A segment that cannot be UTF-8
    encoded (a lone surrogate, e.g. from a surrogateescape filename
    on the pack side or a JSON escape in a manifest path) is not a
    segment at all — it must fail here in the fixed vocabulary, not
    later as a raw codec error during serialisation (oversight
    security S-10).
    """
    try:
        segment.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return (
        segment not in ("", ".", "..")
        and "/" not in segment
        and "\\" not in segment
        and _PATH_BAD_CHARS.search(segment) is None
    )


def _valid_bundle_path(path: Any) -> bool:
    """The canonical relative-path grammar for manifest/archive paths."""
    if type(path) is not str or not path:
        return False
    if path.startswith("/") or "\\" in path or _DRIVE_RE.match(path):
        return False
    if _PATH_BAD_CHARS.search(path):
        return False
    return all(_valid_segment(segment) for segment in path.split("/"))


class _ManifestDuplicateField(ValueError):
    """A bundle manifest carried the same field more than once."""


def _unique_manifest_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Return the unique manifest object."""
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise _ManifestDuplicateField("bundle_manifest_duplicate_field")
        obj[key] = value
    return obj


def _reject_manifest_constant(_value: str) -> Any:
    """``parse_constant`` hook: NaN/Infinity are not manifest JSON
    (the tree-wide nonfinite discipline; oversight lead L-C). The
    ValueError is normalised to ``manifest_unparseable`` by the
    caller's widened catch.
    """
    raise ValueError("manifest_nonfinite_constant")


def build_manifest(files: list[dict[str, Any]], context: dict[str, str]) -> dict[str, Any]:
    """Assemble the manifest dict (pure; canonicalisation happens at
    serialisation time in :func:`manifest_bytes`).
    """
    return {"context": context, "files": files, "format": FORMAT_ID}


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """Serialise the manifest as canonical JSON (the exact bytes the
    bundle digest is computed over).
    """
    return _canonical_json_bytes(manifest)


def bundle_digest(manifest_data: bytes) -> str:
    """The bundle digest: SHA-256 of the exact stored manifest bytes."""
    return _sha256_hex(manifest_data)


def _zip_info(name: str) -> zipfile.ZipInfo:
    """Return the zip info."""
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 0
    info.external_attr = (0o644 & 0xFFFF) << 16
    return info


def build_bundle_bytes(manifest_data: bytes, files: list[tuple[str, bytes]]) -> bytes:
    """Serialise a bundle (pure): entries sorted by name, every ZipInfo
    normalised, the manifest digest in the archive comment.
    """
    entries = [(MANIFEST_NAME, manifest_data), *files]
    entries.sort(key=lambda item: item[0])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries:
            archive.writestr(_zip_info(name), data)
        archive.comment = _COMMENT_PREFIX + bundle_digest(manifest_data).encode("ascii")
    return buffer.getvalue()


def _read_regular_file(path: Path, *, too_large_code: str, cap: int) -> bytes:
    """Open ``path`` once (O_NOFOLLOW/O_NONBLOCK), require a regular
    file, and return its bytes — at most ``cap`` of them. Mirrors the
    single-fd discipline of ``diffreports.load_results``.
    """
    try:
        fd = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
    except OSError:
        raise ValueError("pack_entry_unreadable") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("pack_entry_not_regular")
        with os.fdopen(fd, "rb") as handle:
            fd = -1  # ownership moved to the file object
            content = handle.read(cap + 1)
    finally:
        if fd != -1:
            os.close(fd)
    if len(content) > cap:
        raise ValueError(too_large_code)
    return content


def collect_pack_files(
    report_dir: Path, includes: list[Path]
) -> tuple[list[tuple[str, bytes]], dict[str, Any]]:
    """Read every file to seal (single-read, fail-closed) and parse the
    report's ``results.json`` through the shared bounded reader.

    Returns ``(entries, results_data)`` where entries are
    ``(arcname, bytes)`` pairs sorted by arcname. Raises ``ValueError``
    with a fixed code on any problem — nothing is silently skipped.
    """
    from .diffreports import parse_results_bytes

    if not report_dir.is_dir():
        raise ValueError("pack_report_dir_missing")
    entries: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    with os.scandir(report_dir) as scan:
        children = sorted(scan, key=lambda entry: entry.name)
    for child in children:
        name = child.name
        if child.is_dir(follow_symlinks=False):
            raise ValueError("pack_subdirectory_not_allowed")
        if not _valid_segment(name):
            raise ValueError("pack_entry_name_invalid")
        if name == MANIFEST_NAME:
            # The manifest name is reserved by the bundle format
            # itself (the verifier rejects it as a sealed path):
            # sealing a report file under it would build a bundle
            # with two manifest entries that pack's own verifier
            # refuses -- fail here, at collection, instead.
            raise ValueError("pack_entry_name_reserved")
        data = _read_regular_file(
            report_dir / name,
            too_large_code="pack_entry_too_large",
            cap=MAX_BUNDLE_FILE_BYTES,
        )
        entries.append((name, data))
        seen.add(name)
    if RESULTS_NAME not in seen:
        raise ValueError("pack_results_missing")
    results_data = parse_results_bytes(dict(entries)[RESULTS_NAME])
    for include in includes:
        segment = include.name
        if not _valid_segment(segment):
            raise ValueError("pack_include_name_invalid")
        arcname = "inputs/" + segment
        if arcname in seen:
            raise ValueError("pack_include_duplicate")
        data = _read_regular_file(
            include,
            too_large_code="pack_entry_too_large",
            cap=MAX_BUNDLE_FILE_BYTES,
        )
        entries.append((arcname, data))
        seen.add(arcname)
    entries.sort(key=lambda item: item[0])
    return entries, results_data


def _role_for(arcname: str) -> str:
    """Return the bundle role of an archive member name."""
    if arcname == RESULTS_NAME:
        return "results"
    if arcname.startswith("inputs/"):
        return "input"
    return "report"


def pack_bundle(report_dir: Path, includes: list[Path]) -> tuple[bytes, dict[str, Any], str]:
    """Build the bundle bytes for a report directory (pure given the
    directory contents): returns ``(bundle_bytes, manifest, digest)``.
    """
    entries, results_data = collect_pack_files(report_dir, includes)
    fingerprint = results_data.get("backend_fingerprint")
    if type(fingerprint) is not str or not fingerprint:
        raise ValueError("pack_results_missing_fingerprint")
    files = [
        {"path": name, "role": _role_for(name), "sha256": _sha256_hex(data), "size": len(data)}
        for name, data in entries
    ]
    context = {
        "backend_fingerprint": fingerprint,
        "catalog_sha256": catalog_digest(),
        "package_version": __version__,
    }
    manifest = build_manifest(files, context)
    data = manifest_bytes(manifest)
    return build_bundle_bytes(data, entries), manifest, bundle_digest(data)


def load_bundle_bytes(path: Path) -> bytes:
    """Read a bundle file with ``load_results``' single-fd discipline.

    Raises ``ValueError`` with a fixed ``bundle_*`` code; a FIFO cannot
    block the read, and the size cap is enforced from ``fstat`` before
    any bytes are read. The open refuses symlinks (``O_NOFOLLOW``),
    matching :func:`_read_regular_file` in this module: before cycle
    202 the two readers disagreed, and a symlinked bundle was followed
    here while a symlinked pack entry was refused there.
    """
    if not path.exists():
        raise ValueError("bundle_path_missing")
    try:
        fd = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
    except OSError:
        raise ValueError("bundle_unreadable") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("bundle_not_regular_file")
        if info.st_size > MAX_BUNDLE_ZIP_BYTES:
            raise ValueError("bundle_too_large")
        with os.fdopen(fd, "rb") as handle:
            fd = -1  # ownership moved to the file object
            content = handle.read(MAX_BUNDLE_ZIP_BYTES + 1)
    finally:
        if fd != -1:
            os.close(fd)
    if len(content) > MAX_BUNDLE_ZIP_BYTES:
        raise ValueError("bundle_too_large")
    return content


def _preparse_container(data: bytes) -> tuple[bytes, int]:
    """Validate the strict container shape and return ``(comment,
    entry_count)`` before ``ZipFile`` parses anything.

    Rejects (``ValueError("bundle_container_invalid")``, or
    ``"bundle_too_many_entries"``): no EOCD, trailing bytes after the
    EOCD, a first local header not at offset 0, multi-disk fields,
    ZIP64 sentinels, and central-directory bounds that do not end
    exactly at the EOCD.
    """
    window_start = max(0, len(data) - (_EOCD_MIN_BYTES + _MAX_COMMENT_BYTES))
    window = data[window_start:]
    eocd_pos = -1
    search_end = len(window)
    while True:
        idx = window.rfind(_EOCD_SIGNATURE, 0, search_end)
        if idx == -1:
            break
        if idx + _EOCD_MIN_BYTES <= len(window):
            comment_len = int.from_bytes(window[idx + 20 : idx + 22], "little")
            if window_start + idx + _EOCD_MIN_BYTES + comment_len == len(data):
                eocd_pos = window_start + idx
                break
        search_end = idx
    if eocd_pos == -1:
        raise ValueError("bundle_container_invalid")
    record = data[eocd_pos : eocd_pos + _EOCD_MIN_BYTES]
    disk_no = int.from_bytes(record[4:6], "little")
    cd_disk = int.from_bytes(record[6:8], "little")
    entries_disk = int.from_bytes(record[8:10], "little")
    total_entries = int.from_bytes(record[10:12], "little")
    cd_size = int.from_bytes(record[12:16], "little")
    cd_offset = int.from_bytes(record[16:20], "little")
    if (
        disk_no != 0
        or cd_disk != 0
        or entries_disk != total_entries
        or total_entries == 0xFFFF
        or cd_size == 0xFFFFFFFF
        or cd_offset == 0xFFFFFFFF
    ):
        raise ValueError("bundle_container_invalid")
    if total_entries > MAX_BUNDLE_ENTRIES:
        raise ValueError("bundle_too_many_entries")
    if total_entries < 1 or cd_offset + cd_size != eocd_pos:
        raise ValueError("bundle_container_invalid")
    if data[0:4] != b"PK\x03\x04":
        raise ValueError("bundle_container_invalid")
    comment = data[eocd_pos + _EOCD_MIN_BYTES :]
    return comment, total_entries


def _valid_manifest(manifest: Any) -> str | None:
    """Return ``None`` when the manifest matches the format-1 schema,
    else a short fixed detail token. Exact types throughout; unknown
    keys and unknown roles are accepted (forward compatibility).
    """
    if type(manifest) is not dict:
        return "manifest_not_an_object"
    if manifest.get("format") != FORMAT_ID:
        return "manifest_format_mismatch"
    files = manifest.get("files")
    if type(files) is not list or not files:
        return "manifest_files_invalid"
    paths: set[str] = set()
    for entry in files:
        if type(entry) is not dict:
            return "manifest_file_entry_invalid"
        path = entry.get("path")
        if type(path) is not str or not _valid_bundle_path(path) or path == MANIFEST_NAME:
            return "manifest_path_invalid"
        if path in paths:
            return "manifest_path_duplicate"
        paths.add(path)
        sha = entry.get("sha256")
        if type(sha) is not str or _SHA256_RE.match(sha) is None:
            return "manifest_sha256_invalid"
        size = entry.get("size")
        if type(size) is not int or size < 0:
            return "manifest_size_invalid"
        if type(entry.get("role")) is not str:
            return "manifest_role_invalid"
    if RESULTS_NAME not in paths:
        return "manifest_results_missing"
    context = manifest.get("context")
    if type(context) is not dict:
        return "manifest_context_invalid"
    if type(context.get("package_version")) is not str or not context.get("package_version"):
        return "manifest_context_invalid"
    catalog_sha = context.get("catalog_sha256")
    if type(catalog_sha) is not str or _SHA256_RE.match(catalog_sha) is None:
        return "manifest_context_invalid"
    fingerprint = context.get("backend_fingerprint")
    if type(fingerprint) is not str or not fingerprint:
        return "manifest_context_invalid"
    return None


def _read_entry(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[bytes, str]:
    """Stream-hash one entry in bounded chunks; returns its bytes too
    (callers keep only the entries they need). Raises ``ValueError``
    with a fixed code on CRC/encryption/read failures.
    """
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    try:
        with archive.open(info, "r") as handle:
            while True:
                chunk = handle.read(_READ_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                chunks.append(chunk)
                total += len(chunk)
                if total > info.file_size:
                    # The stream is yielding more than the central
                    # directory declared: abort now instead of buffering
                    # an unbounded lie.
                    raise ValueError("bundle_entry_read_failed")
    except (
        zipfile.BadZipFile,
        RuntimeError,
        zlib.error,
        lzma.LZMAError,
        EOFError,
        OSError,
    ):
        # Codec corruption escapes zipfile as the codec's own exception
        # type (zlib.error for DEFLATE, lzma.LZMAError, OSError for
        # BZip2), not as BadZipFile: the shipped writer only emits
        # STORED entries, so before cycle 202 a hostile compressed
        # entry reached verify --bundle as a raw traceback. Every read
        # failure at this boundary is the same fixed code.
        raise ValueError("bundle_entry_read_failed") from None
    if total != info.file_size:
        raise ValueError("bundle_entry_read_failed")
    return b"".join(chunks), digest.hexdigest()


def verify_bundle_bytes(data: bytes, expect_digest: str | None = None) -> dict[str, Any]:
    """Verify a bundle's seal and replay its ``results.json``.

    Returns the report dict ``{"verdict", "checks", "bundle_digest",
    "context"}`` mirroring :func:`verify_results`. Container-level
    problems (not a strict format-1 container at all) and a sealed
    ``results.json`` that the bounded reader rejects raise
    ``ValueError`` with a fixed code — the caller maps those to the
    unreadable exit class, matching plain ``verify``.
    """
    from .diffreports import parse_results_bytes
    from .verify import verify_results

    comment, entry_count = _preparse_container(data)
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        infos = archive.infolist()
    except (zipfile.BadZipFile, UnicodeDecodeError):
        # A name carrying the UTF-8 flag with invalid UTF-8 bytes
        # fails filename decoding (oversight security S-6b): that is
        # a container defect, in the fixed vocabulary — never a raw
        # codec traceback.
        raise ValueError("bundle_container_invalid") from None
    checks: list[dict[str, str]] = []
    names = [info.filename for info in infos]

    # seal_container: duplicates, path grammar, directory entries, and
    # the pre-parsed entry count — all before any entry data is read.
    container_detail = ""
    if len(infos) != entry_count:
        container_detail = "entry_count_mismatch"
    elif len(set(names)) != len(names):
        container_detail = "duplicate_entry_name"
    elif any(name.endswith("/") for name in names):
        container_detail = "directory_entry_present"
    elif any(_entry_type_not_regular(info) for info in infos):
        container_detail = "entry_type_not_regular"
    elif any(not _valid_bundle_path(name) for name in names):
        container_detail = "entry_path_invalid"
    _record(
        checks,
        "seal_container",
        "failed" if container_detail else "passed",
        container_detail,
    )

    # Caps from the central directory, before reading any entry.
    caps_detail = ""
    total_size = 0
    for info in infos:
        total_size += info.file_size
        if info.file_size > MAX_BUNDLE_FILE_BYTES:
            caps_detail = "entry_too_large"
    if total_size > MAX_BUNDLE_TOTAL_BYTES:
        caps_detail = "total_too_large"
    caps_ok = not caps_detail

    manifest: dict[str, Any] | None = None
    manifest_raw: bytes | None = None
    stored: dict[str, tuple[bytes, str]] = {}
    read_error = ""
    if not container_detail and caps_ok:
        by_name = {info.filename: info for info in infos}
        for target in (MANIFEST_NAME, RESULTS_NAME):
            if target in by_name:
                try:
                    stored[target] = _read_entry(archive, by_name[target])
                except ValueError as exc:
                    read_error = str(exc)
        if MANIFEST_NAME in stored:
            manifest_raw = stored[MANIFEST_NAME][0]

    # seal_manifest_schema.
    schema_detail = ""
    if container_detail:
        schema_detail = "container_failed"
    elif caps_detail:
        schema_detail = caps_detail
    elif read_error:
        schema_detail = read_error
    elif manifest_raw is None:
        schema_detail = "manifest_missing"
    else:
        try:
            parsed = json.loads(
                manifest_raw.decode("utf-8"),
                object_pairs_hook=_unique_manifest_object,
                parse_constant=_reject_manifest_constant,
            )
            schema_detail = _valid_manifest(parsed) or ""
        except (RecursionError, ValueError):
            # The manifest boundary normalises EVERY parse/validation
            # escape into the documented code (oversight security
            # S-4/S-6): a deeply-nested manifest (RecursionError),
            # an over-limit integer literal (the interpreter's plain
            # ValueError), and nonfinite constants (the hook above)
            # are all "unparseable" — datasets.parse_dataset and
            # diffreports.parse_results_bytes already convert the
            # same classes; a bare traceback here broke the verify
            # report contract. JSONDecodeError and
            # _ManifestDuplicateField are ValueError subclasses,
            # named for the record.
            schema_detail = "manifest_unparseable"
        else:
            if not schema_detail:
                manifest = parsed
    _record(
        checks,
        "seal_manifest_schema",
        "failed" if schema_detail else "passed",
        schema_detail,
    )

    digest_hex = bundle_digest(manifest_raw) if manifest_raw is not None else ""
    # seal_comment_digest: the comment must be exactly the prefix + the
    # digest of the stored manifest bytes.
    if manifest is None:
        _record(checks, "seal_comment_digest", "unavailable", "manifest_unusable")
    else:
        expected_comment = _COMMENT_PREFIX + digest_hex.encode("ascii")
        _record(
            checks,
            "seal_comment_digest",
            "passed" if comment == expected_comment else "failed",
            "" if comment == expected_comment else "comment_digest_mismatch",
        )
    # seal_expected_digest (only when the caller supplies one out-of-band).
    if expect_digest is not None:
        if manifest is None:
            _record(checks, "seal_expected_digest", "unavailable", "manifest_unusable")
        else:
            _record(
                checks,
                "seal_expected_digest",
                "passed" if digest_hex == expect_digest else "failed",
                "" if digest_hex == expect_digest else "expected_digest_mismatch",
            )

    # seal_file_set + seal_file_hashes over the manifest's file list.
    if manifest is None:
        _record(checks, "seal_file_set", "unavailable", "manifest_unusable")
        _record(checks, "seal_file_hashes", "unavailable", "manifest_unusable")
    else:
        listed = {entry["path"] for entry in manifest["files"]}
        actual = set(names)
        expected_set = listed | {MANIFEST_NAME}
        if actual == expected_set:
            _record(checks, "seal_file_set", "passed")
        else:
            missing = sorted(expected_set - actual)
            extra = sorted(actual - expected_set)
            detail_parts = []
            if missing:
                detail_parts.append(
                    "missing:" + ",".join(_display_id(name) for name in missing[:5])
                )
            if extra:
                detail_parts.append("unlisted:" + ",".join(_display_id(name) for name in extra[:5]))
            _record(checks, "seal_file_set", "failed", ";".join(detail_parts))
        hash_detail = ""
        by_name = {info.filename: info for info in infos}
        for entry in manifest["files"]:
            path = entry["path"]
            zinfo = by_name.get(path)
            if zinfo is None:
                hash_detail = "entry_missing"
                break
            if path in stored:
                entry_bytes, entry_digest = stored[path]
            else:
                try:
                    entry_bytes, entry_digest = _read_entry(archive, zinfo)
                except ValueError as exc:
                    hash_detail = str(exc)
                    break
            if len(entry_bytes) != entry["size"] or entry_digest != entry["sha256"]:
                hash_detail = "hash_or_size_mismatch:" + _display_id(path)
                break
        _record(
            checks,
            "seal_file_hashes",
            "failed" if hash_detail else "passed",
            hash_detail,
        )

    seal_failed = any(c["status"] == "failed" for c in checks)
    context = manifest.get("context") if manifest is not None else None
    if seal_failed:
        for replay_check in verify_results(None)["checks"]:
            _record(checks, replay_check["id"], "unavailable", "seal_failed")
    else:
        results_bytes = stored[RESULTS_NAME][0]
        results_data = parse_results_bytes(results_bytes)
        for replay_check in verify_results(results_data)["checks"]:
            checks.append(replay_check)
    verdict = "ok" if all(c["status"] != "failed" for c in checks) else "failed"
    return {"verdict": verdict, "checks": checks, "bundle_digest": digest_hex, "context": context}


def build_pack_parser(sub: Any) -> None:
    """Register the ``pack`` subcommand (called from cli.build_parser)."""
    pack_cmd = sub.add_parser(
        "pack",
        help="Seal a report directory into a hash-bound bundle: " + MONDAY_MORNING_USE,
        allow_abbrev=False,
    )
    pack_cmd.add_argument(
        "report_dir",
        type=Path,
        help="Report directory to seal (must contain results.json)",
    )
    pack_cmd.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Bundle file to create (default: <report-dir>.bundle.zip as a "
        "sibling); an existing file is never overwritten",
    )
    pack_cmd.add_argument(
        "--include",
        type=Path,
        action="append",
        default=None,
        help="Extra file to seal under inputs/<name> (repeatable)",
    )


def _default_output(report_dir: Path) -> Path:
    """Return the default output."""
    return report_dir.with_name(report_dir.name + ".bundle.zip")


def run_pack(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    """Implement ``model-protocol-lab pack``; returns the process exit code.

    JSON goes to stdout, the human-readable summary to stderr. Exit 0 on
    success, 1 for any input/usage/output failure. The bundle digest is
    printed prominently: it is the out-of-band channel the seal needs.
    """
    report_dir = Path(args.report_dir)
    includes = [Path(p) for p in (args.include or [])]
    output = Path(args.output) if args.output is not None else _default_output(report_dir)
    try:
        report_resolved = report_dir.resolve()
        output_resolved = output.resolve()
    except OSError:
        print("pack: path_unresolvable", file=sys.stderr)
        return 1
    try:
        output_resolved.relative_to(report_resolved)
    except ValueError:
        pass
    else:
        print("pack: output_inside_report_dir", file=sys.stderr)
        return 1
    for include in includes:
        try:
            same = include.resolve() == output_resolved
        except OSError:
            print("pack: path_unresolvable", file=sys.stderr)
            return 1
        if same:
            print("pack: include_is_output", file=sys.stderr)
            return 1
    try:
        bundle_bytes, manifest, digest = pack_bundle(report_dir, includes)
    except ValueError as exc:
        print(f"pack: {exc}", file=sys.stderr)
        return 1
    try:
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError:
        print("pack: output_unwritable_or_exists", file=sys.stderr)
        return 1
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1  # ownership moved to the file object
            handle.write(bundle_bytes)
    except OSError:
        with contextlib.suppress(OSError):
            output.unlink()
        print("pack: output_write_failed", file=sys.stderr)
        return 1
    finally:
        if fd != -1:
            os.close(fd)
    print(json.dumps({"bundle": str(output), "digest": digest}, indent=2, sort_keys=True))
    print(
        "pack: sealed " + str(len(manifest["files"])) + " files into " + _display_id(str(output)),
        file=sys.stderr,
    )
    print("pack: bundle digest (publish alongside the result): " + digest, file=sys.stderr)
    return 0


def _render_bundle_summary(report: dict[str, Any], label: str) -> str:
    """Render bundle summary."""
    checks = report["checks"]
    passed = sum(1 for c in checks if c["status"] == "passed")
    failed = [c for c in checks if c["status"] == "failed"]
    unavailable = [c for c in checks if c["status"] == "unavailable"]
    lines = [
        "verify --bundle: "
        + _display_id(label)
        + ": "
        + str(passed)
        + "/"
        + str(len(checks))
        + " checks passed"
        + (", " + str(len(unavailable)) + " unavailable" if unavailable else "")
        + ", verdict "
        + report["verdict"]
    ]
    for check in failed:
        detail = " (" + check["detail"] + ")" if check["detail"] else ""
        lines.append("FAILED: " + check["id"] + detail)
    for check in unavailable:
        detail = " (" + check["detail"] + ")" if check["detail"] else ""
        lines.append("unavailable: " + check["id"] + detail)
    lines.append("seal: integrity since packing only — origin not authenticated")
    return "\n".join(lines)


def run_verify_bundle(args: argparse.Namespace) -> int:
    """Implement ``model-protocol-lab verify --bundle``; exit codes mirror
    ``run_verify``: 0 when seal and replay pass, 2 when either fails, 1
    for unreadable input.
    """
    try:
        data = load_bundle_bytes(Path(args.bundle))
    except ValueError as exc:
        print(f"verify: {exc}", file=sys.stderr)
        return 1
    try:
        report = verify_bundle_bytes(data, expect_digest=args.expect_digest)
    except ValueError as exc:
        print(f"verify: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
    print(_render_bundle_summary(report, str(args.bundle)), file=sys.stderr)
    return 0 if report["verdict"] == "ok" else 2
