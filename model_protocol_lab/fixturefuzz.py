"""Seeded contract fuzzer for the lab's own fixture loaders.

``model-protocol-lab fixtures fuzz`` mutates known-good fixture files
(JSON runtime format and CSV authoring format) from a fixed, published
mutator table and checks the lab's own loaders against their published
accept/reject contract (``docs/FIXTURE_FUZZ_DESIGN.md``). It tests the
loaders -- never a model, never fixture quality. The corpus is not
coverage-guided and does not evolve: the seed only varies mutator
parameters (lengths, padding), never which outcome a case expects, so
a run is byte-for-byte reproducible from its seed.

Fail-closed vocabulary: a case outcome is ``accept``, ``reject`` (any
``ValueError``), or ``accept_warn`` (CSV only: the loader returns and
names the fixture in its empty-``expected`` list). A case that escapes
with anything but ``ValueError`` is a ``crash`` finding; a case whose
outcome differs from the table is a finding, never a pass.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import Split
from .datasets import load_dataset, parse_dataset
from .fixtures import parse_fixture_csv, read_authoring_file

MONDAY_MORNING_USE = (
    "before trusting fixture files with an experiment, prove the loaders "
    "still refuse hostile or malformed ones gracefully -- fixtures fuzz "
    "mutates known-good JSON and CSV fixtures from a fixed published table "
    "and checks every mutation against the loaders' accept/reject contract"
)

SCHEMA_ID = "model-protocol-lab.fixture-fuzz/v1"
TARGETS = ("json", "csv", "json_file", "csv_file")
OUTCOMES = ("accept", "reject", "accept_warn")
FILE_KINDS = ("regular", "directory", "missing", "fifo", "symlink")
_STORED_FIELDS = ("id", "group", "expected", "content", "content_last")

ParseJson = Callable[[str], Split]
ParseCsv = Callable[[str], "tuple[Split, list[str]]"]
LoadJsonFile = Callable[[Path], Split]
ReadCsvFile = Callable[[Path], str]


@dataclass(frozen=True)
class FuzzCase:
    """One published mutation and the outcome the contract requires.

    ``stored`` pins the loader's decoded value for one field of one
    fixture: ``(partition, index, field, value)`` with field one of
    ``id``/``group``/``expected``/``content`` (first message) /
    ``content_last`` (final message). The
    comparison is against the decoded value -- JSON escapes resolved,
    CSV quoting removed -- never the raw source bytes.
    ``message_pin`` strengthens a ``reject`` case: the exception must
    be exactly ``ValueError`` and its message must contain the pin.
    Pins are used only where the loaders' wording is their own fixed
    vocabulary; interpreter-owned messages (JSON syntax, nesting)
    are deliberately unpinned because they vary by version.
    """

    name: str
    target: str
    payload: str | bytes
    expected: str
    clause: str
    message_pin: str | None = None
    stored: tuple[str, int, str, str] | None = None
    counts: tuple[int, int] | None = None
    warn_id: str | None = None
    file_kind: str = "regular"

    # skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
    def __post_init__(self) -> None:
        """Validate the fuzz target and outcome vocabulary."""
        if self.target not in TARGETS:
            raise ValueError(f"unknown fuzz target {self.target!r}")
        if self.expected not in OUTCOMES:
            raise ValueError(f"unknown fuzz outcome {self.expected!r}")
        if self.target in ("json", "csv"):
            if type(self.payload) is not str:
                raise ValueError(f"{self.name}: text target payload must be str")
        else:
            if type(self.payload) is not bytes:
                raise ValueError(f"{self.name}: file target payload must be bytes")
            if self.file_kind not in FILE_KINDS:
                raise ValueError(f"{self.name}: unknown file kind {self.file_kind!r}")
            if self.expected == "accept_warn":
                raise ValueError(f"{self.name}: file targets have no warning channel")
        if self.expected == "accept_warn" and (self.target != "csv" or self.warn_id is None):
            raise ValueError(f"{self.name}: accept_warn needs the csv target and a warn_id")
        if self.message_pin is not None and self.expected != "reject":
            raise ValueError(f"{self.name}: message_pin only applies to reject cases")
        if self.stored is not None:
            partition, _index, field, _value = self.stored
            if partition not in ("search", "held_out") or field not in _STORED_FIELDS:
                raise ValueError(f"{self.name}: malformed stored check {self.stored!r}")


@dataclass(frozen=True)
class CaseResult:
    """The observed outcome of one case and whether it conforms."""

    name: str
    target: str
    expected: str
    observed: str
    conform: bool
    kind: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize with a fixed key order (report determinism)."""
        return {
            "name": self.name,
            "target": self.target,
            "expected": self.expected,
            "observed": self.observed,
            "conform": self.conform,
            "kind": self.kind,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FuzzReport:
    """A complete fuzz run: per-case results plus the corpus digest."""

    seed: int
    digest: str
    results: tuple[CaseResult, ...]

    @property
    def total(self) -> int:
        """Number of cases evaluated."""
        return len(self.results)

    @property
    def conforming(self) -> int:
        """Number of cases whose observed outcome matched the contract."""
        return sum(1 for result in self.results if result.conform)

    @property
    def ok(self) -> bool:
        """True iff every case conformed (no findings)."""
        return self.conforming == self.total

    def to_dict(self) -> dict[str, Any]:
        """Serialize with a fixed key order (report determinism)."""
        by_target: dict[str, dict[str, int]] = {}
        for target in TARGETS:
            rows = [r for r in self.results if r.target == target]
            if rows:
                by_target[target] = {
                    "cases": len(rows),
                    "conform": sum(1 for r in rows if r.conform),
                }
        return {
            "schema": SCHEMA_ID,
            "seed": self.seed,
            "digest": self.digest,
            "cases": self.total,
            "conform": self.conforming,
            "findings": self.total - self.conforming,
            "ok": self.ok,
            "by_target": by_target,
            "results": [r.to_dict() for r in self.results],
        }


def _derive(seed: int, case: str, param: str, modulus: int) -> int:
    """Deterministic parameter derivation (SHA-256, never ``random``).

    CPython guarantees cross-version reproducibility only for
    ``random.random()``; deriving every parameter from a hash keeps
    the corpus identical on every supported interpreter.
    """
    material = f"fixturefuzz/v1:{seed}:{case}:{param}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % modulus


def corpus_digest(cases: tuple[FuzzCase, ...]) -> str:
    """SHA-256 over case identities and payload bytes.

    File-case payloads are their bytes; no temp path ever enters the
    digest, so two runs of one seed agree on any machine.
    """
    hasher = hashlib.sha256()
    for case in cases:
        payload = case.payload
        raw = payload.encode("utf-8", "surrogatepass") if isinstance(payload, str) else payload
        hasher.update(case.name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(case.target.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(case.expected.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update((case.message_pin or "").encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(case.file_kind.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(raw)
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _materialize(case: FuzzCase, workdir: Path) -> Path:
    """Create the on-disk shape a file case points its loader at."""
    if case.file_kind == "directory":
        return workdir
    if case.file_kind == "missing":
        return workdir / f"absent-{case.name}"
    path = workdir / f"{case.name}.fixture"
    if case.file_kind == "fifo":
        os.mkfifo(path)
        return path
    payload = case.payload
    if type(payload) is not bytes:  # constructor-enforced
        raise TypeError(f"{case.name}: file payload must be bytes")
    path.write_bytes(payload)
    if case.file_kind == "symlink":
        link = workdir / f"{case.name}.link"
        os.symlink(path, link)
        return link
    return path


def _stored_value(split: Split, stored: tuple[str, int, str, str]) -> str | None:
    """Return the stored value."""
    partition, index, field, _value = stored
    fixtures = split.search if partition == "search" else split.held_out
    if not 0 <= index < len(fixtures):
        return None
    fixture = fixtures[index]
    if field == "id":
        return fixture.id
    if field == "group":
        return fixture.group
    if field == "expected":
        return fixture.expected
    if field == "content_last":
        return fixture.messages[-1].content
    return fixture.messages[0].content


def _observe(
    case: FuzzCase,
    workdir: Path,
    parse_json: ParseJson,
    parse_csv: ParseCsv,
    load_json_file: LoadJsonFile,
    read_csv_file: ReadCsvFile,
) -> tuple[str, Split | None, list[str], Exception | None]:
    """Run one case; return (observed, split, warnings, exception).

    Only ``ValueError`` counts as a rejection. Any other exception is
    reported as a crash finding by the caller -- that distinction is
    the fuzzer's core claim (F1), so it is never blurred here.
    """
    try:
        if case.target == "json":
            text = case.payload
            if type(text) is not str:  # constructor-enforced
                raise TypeError(f"{case.name}: text payload must be str")
            return "accept", parse_json(text), [], None
        if case.target == "csv":
            text = case.payload
            if type(text) is not str:  # constructor-enforced
                raise TypeError(f"{case.name}: text payload must be str")
            split, warnings = parse_csv(text)
            return ("accept_warn" if warnings else "accept"), split, warnings, None
        path = _materialize(case, workdir)
        if case.target == "json_file":
            return "accept", load_json_file(path), [], None
        split, warnings = parse_csv(read_csv_file(path))
        return ("accept_warn" if warnings else "accept"), split, warnings, None
    except ValueError as exc:
        return "reject", None, [], exc
    # skipcq: FLK-E501 - ruff's E501 exemptions cover this frozen, test-pinned literal
    except Exception as exc:  # noqa: BLE001 -- deliberate: any non-ValueError escape IS the crash finding this harness exists to report; it is classified, never propagated
        return "crash", None, [], exc


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def evaluate_case(
    case: FuzzCase,
    workdir: Path,
    *,
    parse_json: ParseJson = parse_dataset,
    parse_csv: ParseCsv = parse_fixture_csv,
    load_json_file: LoadJsonFile = load_dataset,
    read_csv_file: ReadCsvFile = read_authoring_file,
) -> CaseResult:
    """Evaluate one case against the contract table.

    Report details are fixed tokens plus, for a crash, the exception
    class name -- never exception message text, whose wording varies
    by interpreter version even when the outcome does not.
    """
    observed, split, warnings, exc = _observe(
        case, workdir, parse_json, parse_csv, load_json_file, read_csv_file
    )
    if observed == "crash":
        return CaseResult(
            case.name,
            case.target,
            case.expected,
            observed,
            False,
            "crash",
            f"uncaught {type(exc).__name__}",
        )
    if observed != case.expected:
        if observed == "reject":
            kind = "unexpected_reject"
        elif case.expected == "reject":
            kind = "unexpected_accept"
        else:
            kind = "warning_mismatch"
        return CaseResult(
            case.name,
            case.target,
            case.expected,
            observed,
            False,
            kind,
            f"expected {case.expected}, observed {observed}",
        )
    if observed == "reject":
        if case.message_pin is not None:
            if exc is None or type(exc) is not ValueError or case.message_pin not in str(exc):
                return CaseResult(
                    case.name,
                    case.target,
                    case.expected,
                    observed,
                    False,
                    "message_mismatch",
                    f"expected ValueError mentioning {case.message_pin!r}, "
                    f"got {type(exc).__name__}",
                )
            detail = "rejected with the pinned message"
        else:
            detail = "rejected with ValueError"
        return CaseResult(case.name, case.target, case.expected, observed, True, "ok", detail)
    # Accepted as expected: the warning roster and stored data are
    # part of the contract, so both are checked before conforming.
    if case.expected == "accept_warn" and case.warn_id not in warnings:
        return CaseResult(
            case.name,
            case.target,
            case.expected,
            observed,
            False,
            "warning_mismatch",
            "empty-expected warning names a different fixture",
        )
    if (
        case.counts is not None
        and split is not None
        and (len(split.search), len(split.held_out)) != case.counts
    ):
        return CaseResult(
            case.name,
            case.target,
            case.expected,
            observed,
            False,
            "stored_mismatch",
            f"partition sizes {(len(split.search), len(split.held_out))} "
            f"!= contracted {case.counts}",
        )
    if (
        case.stored is not None
        and split is not None
        and _stored_value(split, case.stored) != case.stored[3]
    ):
        return CaseResult(
            case.name,
            case.target,
            case.expected,
            observed,
            False,
            "stored_mismatch",
            f"stored {case.stored[2]} of {case.stored[0]}[{case.stored[1]}] "
            "differs from the decoded source value",
        )
    detail = (
        "accepted with the documented empty-expected warning"
        if case.expected == "accept_warn"
        else "accepted as contracted"
    )
    return CaseResult(case.name, case.target, case.expected, observed, True, "ok", detail)


def run_fuzz(seed: int = 0) -> FuzzReport:
    """Generate the corpus for ``seed`` and evaluate every case.

    Cases run on the calling thread in a fresh temp directory; the
    report contains no paths and no timestamps, so the same seed
    reproduces it byte-for-byte.
    """
    cases = generate_cases(seed)
    with tempfile.TemporaryDirectory(prefix="mpl-fixture-fuzz-") as tmp:
        workdir = Path(tmp)
        results = tuple(evaluate_case(case, workdir) for case in cases)
    return FuzzReport(seed=seed, digest=corpus_digest(cases), results=results)


def run_fixtures_fuzz(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Implement ``model-protocol-lab fixtures fuzz``; returns the exit code.

    The JSON report goes to stdout and a one-line summary to stderr
    (the doctor/pack convention). Exit 0 when every case conforms,
    1 on any finding.
    """
    # skipcq: PTC-W0043 - the del is a deliberate tripwire: the protocol parameter is unused
    del parser  # the subcommand takes no parser-level validation
    seed = int(args.seed)
    report = run_fuzz(seed)
    print(json.dumps(report.to_dict(), indent=2))
    print(
        f"fixtures fuzz: {report.conforming}/{report.total} cases conform, "
        f"{report.total - report.conforming} finding(s), "
        f"corpus digest {report.digest[:16]}..., seed {seed}",
        file=sys.stderr,
    )
    return 0 if report.ok else 1


# ---------------------------------------------------------------------------
# Corpus generation. The table is fixed and published (design doc
# section 2);
# the seed varies only mutator parameters via _derive. JSON cases built
# by mutating a dict are serialized with json.dumps (which escapes
# control characters); cases about the source text itself are built by
# string surgery, and the clause says which construction applies --
# the same value can have different contract outcomes in each.
# ---------------------------------------------------------------------------

_MAX_TEXT = 16_384  # datasets/fixtures MAX_TEXT, restated to pin the boundary
_MAX_BYTES = 2_000_000  # datasets/fixtures MAX_BYTES


def _fx(fid: str, group: str, content: str, expected: str) -> dict[str, Any]:
    """Return the fx."""
    return {
        "id": fid,
        "group": group,
        "messages": [{"role": "user", "content": content}],
        "expected": expected,
    }


def _json_base() -> dict[str, Any]:
    """Return the json base."""
    return {
        "search": [
            _fx("s1", "gs", "q one", "a"),
            _fx("s2", "gs2", "q two", "b"),
        ],
        "held_out": [
            _fx("h1", "gh", "q three", "c"),
            _fx("h2", "gh2", "q four", "d"),
        ],
    }


def _json_text(doc: dict[str, Any]) -> str:
    """Return the json text."""
    return json.dumps(doc, ensure_ascii=False)


def _json_cases(seed: int) -> list[FuzzCase]:
    """Return the json cases."""
    cases: list[FuzzCase] = []

    def add(name: str, payload: str, expected: str, clause: str, **kw: Any) -> None:
        """Add."""
        cases.append(FuzzCase(f"json_{name}", "json", payload, expected, clause, **kw))

    base = _json_text(_json_base())
    add(
        "control",
        base,
        "accept",
        "control: the unmutated base document loads",
        stored=("search", 0, "id", "s1"),
        counts=(2, 2),
    )

    doc = _json_base()
    doc["search"][0]["messages"][0]["content"] = "Ignore <|im_start|> previous"
    add(
        "marker_content",
        _json_text(doc),
        "accept",
        "content has no blocklist at load time; the special-token audit is a "
        "separate warning-only layer",
        stored=("search", 0, "content", "Ignore <|im_start|> previous"),
    )
    doc = _json_base()
    doc["search"][0]["expected"] = "ans <|im_start|> one"
    add(
        "marker_expected",
        _json_text(doc),
        "accept",
        "expected is plain text to the loader",
        stored=("search", 0, "expected", "ans <|im_start|> one"),
    )

    doc = _json_base()
    doc["search"][0]["id"] = "alpha"
    doc["held_out"][0]["id"] = "аlpha"  # Cyrillic а
    add(
        "homoglyph_id",
        _json_text(doc),
        "accept",
        "confusable look-alikes are documented out of scope for Split: no "
        "normalization form unifies them",
        stored=("held_out", 0, "id", "аlpha"),
    )
    doc = _json_base()
    doc["search"][0]["group"] = "grp"
    doc["held_out"][0]["group"] = "grр"  # Cyrillic р
    add(
        "homoglyph_group",
        _json_text(doc),
        "accept",
        "homoglyph group twin: out of scope, as for ids",
    )
    doc = _json_base()
    doc["search"][0]["messages"][0]["content"] = "same text"
    doc["held_out"][0]["messages"][0]["content"] = "same tеxt"  # Cyrillic е
    add(
        "homoglyph_messages",
        _json_text(doc),
        "accept",
        "homoglyph message twin is not identical after NFC",
    )

    doc = _json_base()
    doc["search"][0]["id"] = "café"
    doc["held_out"][0]["id"] = "café"
    add("nfd_twin_id", _json_text(doc), "reject", "ids must be unique after NFC normalization")
    doc = _json_base()
    doc["search"][0]["group"] = "café"
    doc["held_out"][0]["group"] = "café"
    add(
        "nfd_twin_group",
        _json_text(doc),
        "reject",
        "groups must not overlap across partitions after NFC",
    )
    doc = _json_base()
    doc["search"][0]["messages"][0]["content"] = "café menu"
    doc["held_out"][0]["messages"][0]["content"] = "café menu"
    add(
        "nfd_twin_messages",
        _json_text(doc),
        "reject",
        "identical message sets across partitions after NFC are leakage",
    )
    doc = _json_base()
    doc["search"][0]["messages"][0]["content"] = "unit Ω sign"
    doc["held_out"][0]["messages"][0]["content"] = "unit Ω sign"
    add(
        "ohm_twin_messages",
        _json_text(doc),
        "reject",
        "U+2126 OHM is a canonical singleton: NFC folds it to U+03A9",
    )
    doc = _json_base()
    doc["search"][0]["messages"][0]["content"] = "fish dish"
    doc["held_out"][0]["messages"][0]["content"] = "ﬁsh dish"
    add(
        "ligature_messages",
        _json_text(doc),
        "accept",
        "U+FB01 is a compatibility ligature: NFC does not fold it",
    )
    doc = _json_base()
    doc["search"][1]["messages"][0]["content"] = "q one"
    add(
        "within_partition_identical_messages",
        _json_text(doc),
        "accept",
        "Split intersects message sets across partitions only",
    )
    doc = _json_base()
    doc["search"][0]["group"] = "gspace"
    doc["held_out"][0]["group"] = "gspace "
    add(
        "group_trailing_space",
        _json_text(doc),
        "accept",
        "discipline keys are NFC-normalized but never stripped",
    )
    doc = _json_base()
    doc["search"][0]["group"] = "GroupA"
    doc["held_out"][0]["group"] = "groupa"
    add("group_case_differs", _json_text(doc), "accept", "discipline comparison is case-sensitive")

    doc = _json_base()
    doc["held_out"][1]["id"] = "h1"
    add(
        "duplicate_id_in_partition",
        _json_text(doc),
        "reject",
        "ids are globally unique, including within one partition",
    )
    doc = _json_base()
    doc["held_out"][0]["id"] = "s1"
    add(
        "duplicate_id_cross_partition",
        _json_text(doc),
        "reject",
        "ids are globally unique across partitions",
    )
    doc = _json_base()
    doc["held_out"][0]["group"] = "gs"
    add(
        "group_cross_partition", _json_text(doc), "reject", "a group lives in exactly one partition"
    )
    doc = _json_base()
    doc["held_out"][0]["messages"][0]["content"] = "q one"
    add(
        "identical_messages_cross_partition",
        _json_text(doc),
        "reject",
        "identical message sets across partitions are leakage",
    )

    over = _MAX_TEXT + 1 + _derive(seed, "json_overlong", "over", 512)
    doc = _json_base()
    doc["search"][0]["messages"][0]["content"] = "x" * over
    add(
        "overlong_content",
        _json_text(doc),
        "reject",
        "no text field may exceed MAX_TEXT characters",
    )
    doc = _json_base()
    doc["search"][0]["id"] = "x" * over
    add("overlong_id", _json_text(doc), "reject", "the limit covers ids too")
    doc = _json_base()
    doc["search"][0]["group"] = "x" * over
    add("overlong_group", _json_text(doc), "reject", "the limit covers groups too")
    doc = _json_base()
    doc["search"][0]["expected"] = "x" * over
    add("overlong_expected", _json_text(doc), "reject", "the limit covers expected too")
    doc = _json_base()
    doc["search"][0]["messages"][0]["content"] = "y" * _MAX_TEXT
    add(
        "at_limit_content",
        _json_text(doc),
        "accept",
        "exactly MAX_TEXT characters is within the limit",
        stored=("search", 0, "content", "y" * _MAX_TEXT),
    )
    doc = _json_base()
    doc["search"][0]["messages"][0]["content"] = "\U0001f600" * _MAX_TEXT
    add(
        "at_limit_astral_content",
        _json_text(doc),
        "accept",
        "the limit counts characters, not bytes: MAX_TEXT astral chars "
        "(4x MAX_TEXT bytes) still conform",
        stored=("search", 0, "content", "\U0001f600" * _MAX_TEXT),
    )

    doc = _json_base()
    doc["search"][0]["expected"] = ""
    add(
        "empty_expected",
        _json_text(doc),
        "accept",
        "JSON has no warning channel: an empty expected is a plain accept",
    )
    doc = _json_base()
    doc["search"][0]["expected"] = " "
    add(
        "whitespace_expected", _json_text(doc), "accept", "a whitespace expected is stored verbatim"
    )

    doc = _json_base()
    del doc["held_out"]
    add("missing_top_key", _json_text(doc), "reject", "both partitions are required")
    doc = _json_base()
    doc["extra"] = []
    add("extra_top_key", _json_text(doc), "reject", "no top-level keys beyond search/held_out")
    add(
        "duplicate_key_top",
        base[:-1] + ', "search": []}',
        "reject",
        "duplicate object keys reject at every level",
    )
    add(
        "duplicate_key_nested",
        base.replace('"id": "s1"', '"id": "s1", "id": "s1x"', 1),
        "reject",
        "duplicate keys reject inside fixture objects too",
    )
    add(
        "duplicate_key_escape_equivalent",
        base[:-1] + ', "se\\u0061rch": []}',
        "reject",
        "the duplicate-key hook compares decoded keys",
    )
    add(
        "escaped_key_spelling",
        base.replace('"search"', '"se\\u0061rch"', 1),
        "accept",
        "a key spelled with escapes is the same key",
        stored=("search", 0, "id", "s1"),
    )

    add(
        "nan_constant",
        base.replace('"expected": "a"', '"expected": NaN'),
        "reject",
        "NaN is not JSON and never a stored value",
        message_pin="Nonfinite JSON",
    )
    add(
        "infinity_constant",
        base.replace('"expected": "a"', '"expected": -Infinity'),
        "reject",
        "Infinity is not JSON and never a stored value",
        message_pin="Nonfinite JSON",
    )
    add(
        "overflow_number",
        base.replace('"expected": "a"', '"expected": 1e999'),
        "reject",
        "1e999 overflows to inf, which is not a text field",
    )
    add(
        "huge_int",
        base.replace('"expected": "a"', '"expected": ' + "9" * 5000),
        "reject",
        "a 5000-digit integer is not a text field on any interpreter configuration",
    )
    add(
        "deep_nesting",
        base.replace('"expected": "a"', '"expected": ' + "[" * 5000 + "]" * 5000),
        "reject",
        "5000-deep nesting: outcome asserted only -- the guard "
        "that fires is depth- and interpreter-dependent",
    )
    add("truncated", base[: len(base) // 2], "reject", "a truncated document")
    add("trailing_garbage", base + " garbage", "reject", "nothing may follow the document")
    add("bom_prefix", "﻿" + base, "reject", "a BOM is not JSON whitespace")
    add("empty_document", "", "reject", "an empty document has no partitions")
    add("whitespace_document", "  \n\t ", "reject", "a whitespace-only document has no partitions")
    add("top_level_list", "[]", "reject", "the document is an object")

    doc = _json_base()
    doc["search"][0]["id"] = 42
    add("wrong_type_id", _json_text(doc), "reject", "ids are text")
    doc = _json_base()
    doc["search"][0]["expected"] = None
    add("null_expected", _json_text(doc), "reject", "null is not text")
    doc = _json_base()
    doc["search"][0]["id"] = 1.5
    add("float_id", _json_text(doc), "reject", "a float is not an id")
    doc = _json_base()
    del doc["search"][0]["expected"]
    add(
        "fixture_missing_field",
        _json_text(doc),
        "reject",
        "a fixture has exactly id/group/messages/expected",
    )
    doc = _json_base()
    doc["search"][0]["note"] = "x"
    add("fixture_extra_field", _json_text(doc), "reject", "no fixture fields beyond the four")
    doc = _json_base()
    doc["search"][0]["messages"][0]["lang"] = "en"
    add("message_extra_field", _json_text(doc), "reject", "a message has exactly role/content")
    doc = _json_base()
    doc["search"][0]["messages"] = {"role": "user", "content": "q"}
    add("messages_not_list", _json_text(doc), "reject", "messages are a list")
    doc = _json_base()
    doc["search"][0]["messages"] = "q"
    add("messages_string", _json_text(doc), "reject", "messages are not a string")
    doc = _json_base()
    doc["search"][0]["messages"] = []
    add("messages_empty", _json_text(doc), "reject", "a conversation has at least one message")
    doc = _json_base()
    doc["search"][0]["messages"] = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(23)
    ]
    add(
        "messages_too_many",
        _json_text(doc),
        "reject",
        "a conversation has at most MAX_MESSAGES messages",
    )
    doc = _json_base()
    doc["search"][0]["messages"] = [
        {"role": "user", "content": "q"},
        {"role": "user", "content": "q again"},
    ]
    add("messages_misalternating", _json_text(doc), "reject", "roles alternate user/assistant")
    doc = _json_base()
    doc["search"][0]["messages"] = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    add(
        "messages_end_assistant",
        _json_text(doc),
        "reject",
        "a conversation ends with a user message",
    )

    doc = _json_base()
    doc["search"][0]["id"] = " "
    add("blank_id", _json_text(doc), "reject", "ids are non-blank")
    doc = _json_base()
    doc["search"][0]["group"] = " "
    add(
        "blank_group_nbsp",
        _json_text(doc),
        "reject",
        "U+00A0 strips: a group of only NBSP is blank",
    )
    doc = _json_base()
    doc["search"][0]["id"] = "​"
    add(
        "zwsp_id",
        _json_text(doc),
        "accept",
        "U+200B does not strip: a ZWSP-only id is (odd but) non-blank",
        stored=("search", 0, "id", "​"),
    )

    doc = _json_base()
    doc["held_out"] = []
    add("partition_empty", _json_text(doc), "reject", "each partition holds at least one fixture")
    many = {
        "search": [_fx(f"s{i}", f"sg{i}", f"q s{i}", "a") for i in range(101)],
        "held_out": [_fx("h0", "hg0", "q h0", "c")],
    }
    add(
        "partition_too_many",
        _json_text(many),
        "reject",
        "each partition holds at most MAX_CASES_PER_SPLIT fixtures",
    )
    full = {
        "search": [_fx(f"s{i}", f"sg{i}", f"q s{i}", "a") for i in range(100)],
        "held_out": [_fx(f"h{i}", f"hg{i}", f"q h{i}", "c") for i in range(100)],
    }
    add(
        "partitions_at_limit",
        _json_text(full),
        "accept",
        "100 fixtures per partition is the boundary and conforms",
        counts=(100, 100),
    )

    doc = _json_base()
    doc["search"][0]["expected"] = "x" * _MAX_BYTES
    add(
        "oversize_document",
        _json_text(doc),
        "reject",
        "the byte cap fires before parsing",
        message_pin="exceeds size limit",
    )
    pad = _MAX_BYTES - len(base.encode("utf-8"))
    add(
        "exact_size_document",
        base + " " * pad,
        "accept",
        "a document of exactly MAX_BYTES bytes conforms (trailing whitespace is JSON-legal)",
        counts=(2, 2),
    )

    add(
        "raw_control_content",
        base.replace('"q one"', '"q\x01one"'),
        "reject",
        "raw control characters in a string literal: strict JSON rejects (construction: raw text)",
    )
    add(
        "escaped_control_content",
        base.replace('"q one"', '"q\\u0001one"'),
        "accept",
        "the same control character in escape form conforms (construction: raw text)",
        stored=("search", 0, "content", "q\x01one"),
    )
    doc = _json_base()
    doc["search"][0]["messages"][0]["content"] = "q​one"
    add(
        "zero_width_content",
        _json_text(doc),
        "accept",
        "zero-width characters are ordinary content",
        stored=("search", 0, "content", "q​one"),
    )

    add(
        "surrogate_escape_content",
        base.replace('"q one"', '"a\\ud800"'),
        "reject",
        "an escape-form lone surrogate decodes to text no UTF-8 fixture file can store (cycle 172)",
        message_pin="lone surrogate",
    )
    add(
        "surrogate_escape_id",
        base.replace('"s1"', '"a\\ud800"', 1),
        "reject",
        "the surrogate rule covers ids",
        message_pin="lone surrogate",
    )
    add(
        "surrogate_escape_group",
        base.replace('"gs"', '"a\\ud800"', 1),
        "reject",
        "the surrogate rule covers groups",
        message_pin="lone surrogate",
    )
    add(
        "surrogate_escape_expected",
        base.replace('"expected": "a"', '"expected": "a\\ud800"'),
        "reject",
        "the surrogate rule covers expected",
        message_pin="lone surrogate",
    )
    add(
        "surrogate_low_alone",
        base.replace('"q one"', '"x\\udc00"'),
        "reject",
        "a lone low surrogate is as unstorable as a high one",
        message_pin="lone surrogate",
    )
    add(
        "surrogate_pair_escape",
        base.replace('"q one"', '"\\ud83d\\ude00"'),
        "accept",
        "a valid surrogate pair decodes to one astral character and "
        "must stay accepted -- the check scans decoded values",
        stored=("search", 0, "content", "\U0001f600"),
    )
    doc = _json_base()
    doc["search"][0]["messages"][0]["content"] = "q\ud800one"
    add(
        "raw_surrogate_text",
        _json_text(doc),
        "reject",
        "raw surrogate text must fail with the loader's deliberate "
        "message, not a bare codec error (the pin is what makes this "
        "case non-vacuous)",
        message_pin="lone surrogate",
    )

    doc = _json_base()
    doc["search"][0] = {
        "expected": "a",
        "messages": [{"content": "q one", "role": "user"}],
        "group": "gs",
        "id": "s1",
    }
    add(
        "key_order_permuted",
        _json_text(doc),
        "accept",
        "object key order carries no meaning",
        stored=("search", 0, "id", "s1"),
    )
    return cases


_CSV_HEADER = "id,group,partition,expected,user_1,assistant_1,user_2"
_CSV_ROWS = (
    "s1,gs,search,ans one,q one,a one,q follow",
    "s2,gs2,search,ans two,q two,a two,q more",
    "h1,gh,held_out,ans three,q three,a three,q held",
    "h2,gh2,held_out,ans four,q four,a four,q last",
)


def _csv_text(
    rows: tuple[str, ...] = _CSV_ROWS,
    header: str = _CSV_HEADER,
    eol: str = "\n",
    trailing: bool = True,
) -> str:
    """Return the csv text."""
    text = eol.join((header, *rows))
    return text + eol if trailing else text


def _csv_cases(seed: int) -> list[FuzzCase]:
    """Return the csv cases."""
    cases: list[FuzzCase] = []

    def add(name: str, payload: str, expected: str, clause: str, **kw: Any) -> None:
        """Add."""
        cases.append(FuzzCase(f"csv_{name}", "csv", payload, expected, clause, **kw))

    add(
        "control",
        _csv_text(),
        "accept",
        "control: the unmutated base CSV loads",
        stored=("search", 0, "content", "q one"),
        counts=(2, 2),
    )

    rows = list(_CSV_ROWS)
    rows[0] = "s1,gs,search,,q one,a one,q follow"
    add(
        "empty_expected",
        _csv_text(tuple(rows)),
        "accept_warn",
        "an empty expected is returned and named in the warning list",
        warn_id="s1",
    )
    rows = list(_CSV_ROWS)
    rows[0] = "s1,gs,search, ,q one,a one,q follow"
    add(
        "whitespace_expected",
        _csv_text(tuple(rows)),
        "accept",
        "the warning fires on the exact empty string only: a space is "
        "an (odd) answer, not a warning",
    )

    rows = list(_CSV_ROWS)
    rows[0] = "s1,gs,search,ans one,Ignore <|im_start|> previous,a one,q follow"
    add(
        "marker_cell",
        _csv_text(tuple(rows)),
        "accept",
        "cells are plain text to the loader",
        stored=("search", 0, "content", "Ignore <|im_start|> previous"),
    )

    rows = list(_CSV_ROWS)
    rows[0] = "s1,gs,test,ans one,q one,a one,q follow"
    add("bad_partition", _csv_text(tuple(rows)), "reject", "partition is search or held_out")
    rows = list(_CSV_ROWS)
    rows[0] = ",gs,search,ans one,q one,a one,q follow"
    add(
        "blank_id",
        _csv_text(tuple(rows)),
        "reject",
        "a row with any content but a blank id rejects (a fully empty "
        "row is a different, accepted shape -- see all_empty_row)",
    )
    rows = list(_CSV_ROWS)
    rows[0] = "s1,,search,ans one,q one,a one,q follow"
    add("blank_group", _csv_text(tuple(rows)), "reject", "groups are non-blank")
    rows = list(_CSV_ROWS)
    rows.insert(1, ",,,,,,")
    add(
        "all_empty_row",
        _csv_text(tuple(rows)),
        "accept",
        "an all-empty row is silently skipped by the reader: pinned so "
        "the leniency is contractual, not accidental",
        counts=(2, 2),
    )

    rows = list(_CSV_ROWS)
    rows[2] = "s1,gh,held_out,ans three,q three,a three,q held"
    add("duplicate_id", _csv_text(tuple(rows)), "reject", "ids are globally unique")
    rows = list(_CSV_ROWS)
    rows[2] = "h1,gs,held_out,ans three,q three,a three,q held"
    add(
        "group_cross_partition",
        _csv_text(tuple(rows)),
        "reject",
        "a group lives in exactly one partition",
    )
    rows = list(_CSV_ROWS)
    rows[2] = "h1,gh,held_out,ans three,q one,a one,q follow"
    add(
        "identical_messages_cross_partition",
        _csv_text(tuple(rows)),
        "reject",
        "identical message sets across partitions are leakage",
    )
    rows = list(_CSV_ROWS)
    rows[0] = "café,gs,search,ans one,q one,a one,q follow"
    rows[2] = "café,gh,held_out,ans three,q three,a three,q held"
    add(
        "nfd_twin_id",
        _csv_text(tuple(rows)),
        "reject",
        "ids must be unique after NFC normalization",
    )

    add(
        "bad_header_case",
        _csv_text(header="ID,group,partition,expected,user_1,assistant_1,user_2"),
        "reject",
        "the header is exact and case-sensitive",
    )
    add(
        "bom_header",
        "﻿" + _csv_text(),
        "reject",
        "a BOM lands in the first header cell and is deliberately visible",
    )
    add(
        "quoted_header",
        _csv_text(header='"id","group","partition","expected","user_1","assistant_1","user_2"'),
        "accept",
        "a fully quoted header parses to the same header",
        counts=(2, 2),
    )
    add(
        "header_trailing_empty_column",
        _csv_text(header=_CSV_HEADER + ","),
        "reject",
        "a trailing empty column is not a message column",
    )
    add(
        "message_columns_out_of_order",
        _csv_text(header="id,group,partition,expected,user_1,user_2,assistant_1"),
        "reject",
        "message columns alternate user/assistant in order",
    )
    add(
        "message_columns_too_many",
        _csv_text(
            header="id,group,partition,expected,"
            + ",".join(f"{'user' if i % 2 == 0 else 'assistant'}_{i // 2 + 1}" for i in range(22))
        ),
        "reject",
        "at most MAX_MESSAGES message columns",
    )
    add(
        "message_columns_none",
        _csv_text(header="id,group,partition,expected"),
        "reject",
        "at least one message column is required",
    )

    over = _MAX_TEXT + 1 + _derive(seed, "csv_overlong", "over", 512)
    size_pin = "text field exceeds size limit"
    rows = list(_CSV_ROWS)
    rows[0] = f"s1,gs,search,ans one,q one,a one,{'x' * over}"
    add(
        "overlong_cell",
        _csv_text(tuple(rows)),
        "reject",
        "no message cell may exceed MAX_TEXT characters",
        message_pin=size_pin,
    )
    rows = list(_CSV_ROWS)
    rows[0] = f"{'x' * over},gs,search,ans one,q one,a one,q follow"
    add(
        "overlong_id",
        _csv_text(tuple(rows)),
        "reject",
        "the limit covers ids: split_to_json_text refuses oversized "
        "identifiers, so the authoring path must too (cycle 172)",
        message_pin=size_pin,
    )
    rows = list(_CSV_ROWS)
    rows[0] = f"s1,{'x' * over},search,ans one,q one,a one,q follow"
    add(
        "overlong_group",
        _csv_text(tuple(rows)),
        "reject",
        "the limit covers groups, for the same reason",
        message_pin=size_pin,
    )
    rows = list(_CSV_ROWS)
    rows[0] = f"s1,gs,search,{'x' * over},q one,a one,q follow"
    add(
        "overlong_expected",
        _csv_text(tuple(rows)),
        "reject",
        "the limit covers expected",
        message_pin=size_pin,
    )
    rows = list(_CSV_ROWS)
    rows[0] = f"{'y' * _MAX_TEXT},gs,search,ans one,q one,a one,q follow"
    add(
        "id_at_limit",
        _csv_text(tuple(rows)),
        "accept",
        "an id of exactly MAX_TEXT characters conforms",
    )

    rows = list(_CSV_ROWS)
    rows[0] = "s1,gs,search,ans one,,a one,q follow"
    add("first_message_empty", _csv_text(tuple(rows)), "reject", "a fixture needs a user_1 message")
    rows = list(_CSV_ROWS)
    rows[0] = "s1,gs,search,ans one,q one,,q follow"
    add(
        "gap_then_content",
        _csv_text(tuple(rows)),
        "reject",
        "an empty message followed by content is malformed",
    )
    add(
        "ends_with_assistant",
        _csv_text(
            rows=("s1,gs,search,ans one,q one,a one", "h1,gh,held_out,ans three,q three,a three"),
            header="id,group,partition,expected,user_1,assistant_1",
        ),
        "reject",
        "a conversation ends with a user message",
    )
    rows = list(_CSV_ROWS)
    rows[0] = "s1,gs,search,ans one,q one"
    add(
        "short_row_after_user",
        _csv_text(tuple(rows)),
        "accept",
        "a short row is padded: ending after a user message conforms",
        stored=("search", 0, "content", "q one"),
        counts=(2, 2),
    )
    many_rows = tuple(
        f"r{i},rg{i},{'search' if i % 2 == 0 else 'held_out'},a{i},q {i}" for i in range(201)
    )
    add(
        "too_many_rows",
        _csv_text(rows=many_rows),
        "reject",
        "at most 2 * MAX_CASES_PER_SPLIT rows in total",
    )
    overflow_rows = tuple(
        [f"s{i},sg{i},search,a,q s{i}" for i in range(101)] + ["h0,hg0,held_out,a,q h0"]
    )
    add(
        "partition_too_many",
        _csv_text(rows=overflow_rows),
        "reject",
        "each partition holds at most MAX_CASES_PER_SPLIT fixtures",
    )
    add(
        "missing_partition", _csv_text(rows=_CSV_ROWS[:2]), "reject", "both partitions are required"
    )

    rows = list(_CSV_ROWS)
    rows[0] = f"s1,gs,search,ans one,{'z' * 131_073},a one,q follow"
    add(
        "field_over_parser_limit",
        _csv_text(tuple(rows)),
        "reject",
        "a field over the csv module's own limit reports as a parser "
        "limit, in the loader's vocabulary",
        message_pin="parser limit",
    )
    add(
        "bare_cr_endings",
        _csv_text(eol="\r"),
        "reject",
        "bare-CR line endings are a parse error -- and must not be "
        "misreported as a field-size problem (cycle 172)",
        message_pin="CSV parse error",
    )
    rows = list(_CSV_ROWS)
    rows[0] = "s1,gs,search,ans one,q one,a one,q follow,extra"
    add(
        "extra_cells_in_row",
        _csv_text(tuple(rows)),
        "reject",
        "a row may not have more cells than the header",
    )

    rows = list(_CSV_ROWS)
    rows[0] = 's1,gs,search,ans one,"q\none",a one,q follow'
    add(
        "quoted_newline_cell",
        _csv_text(tuple(rows)),
        "accept",
        "a quoted cell may contain a newline (documented)",
        stored=("search", 0, "content", "q\none"),
    )
    add(
        "crlf_endings",
        _csv_text(eol="\r\n"),
        "accept",
        "CRLF line endings are accepted and never leak into cells",
        stored=("search", 0, "content", "q one"),
    )
    rows = list(_CSV_ROWS)
    rows[3] = 'h2,gh2,held_out,ans four,q four,a four,"q last'
    add(
        "unterminated_quote_eof",
        _csv_text(tuple(rows), trailing=False),
        "accept",
        "an unterminated quote at EOF is silently repaired by the "
        "non-strict reader; the stored value is pinned",
        stored=("held_out", 1, "content_last", "q last"),
    )
    rows = list(_CSV_ROWS)
    rows[0] = 's1,gs,search,ans one,"q one"xyz,a one,q follow'
    add(
        "quoted_field_trailing_junk",
        _csv_text(tuple(rows)),
        "accept",
        "junk after a closing quote is kept literally by the "
        "non-strict reader; the stored value is pinned",
        stored=("search", 0, "content", "q onexyz"),
    )
    rows = list(_CSV_ROWS)
    rows[0] = 's1,gs,search,ans one,q"one,a one,q follow'
    add(
        "quote_mid_field",
        _csv_text(tuple(rows)),
        "accept",
        "a quote not at a field start is literal text",
        stored=("search", 0, "content", 'q"one'),
    )

    rows = list(_CSV_ROWS)
    rows[0] = "s1,gs,search,ans one,q\ud800one,a one,q follow"
    add(
        "lone_surrogate_cell",
        _csv_text(tuple(rows)),
        "reject",
        "raw surrogate text must fail with the loader's deliberate message, not a bare codec error",
        message_pin="lone surrogate",
    )
    rows = list(_CSV_ROWS)
    rows[0] = "s1,gs,search,ans one,bad\\ud800text,a one,q follow"
    add(
        "surrogate_literal_text",
        _csv_text(tuple(rows)),
        "accept",
        "CSV has no escape form: the six characters \\ud800 are ordinary text",
        stored=("search", 0, "content", "bad\\ud800text"),
    )
    rows = list(_CSV_ROWS)
    rows[0] = "null,123,search,ans one,q one,a one,q follow"
    add(
        "null_literal_fields",
        _csv_text(tuple(rows)),
        "accept",
        "CSV fields are strings: 'null' and '123' are just text",
        stored=("search", 0, "id", "null"),
    )

    add("empty_document", "", "reject", "an empty document has no header row")
    add(
        "whitespace_only_document",
        " ",
        "reject",
        "a whitespace-only document fails the header check",
    )
    return cases


def _file_cases() -> list[FuzzCase]:
    """Return the file cases."""
    cases: list[FuzzCase] = []
    json_bytes = _json_text(_json_base()).encode("utf-8")
    csv_bytes = _csv_text().encode("utf-8")

    def add(name: str, target: str, payload: bytes, expected: str, clause: str, **kw: Any) -> None:
        """Add."""
        cases.append(FuzzCase(name, target, payload, expected, clause, **kw))

    add(
        "jfile_control",
        "json_file",
        json_bytes,
        "accept",
        "control: a valid fixture file loads",
        counts=(2, 2),
    )
    add(
        "jfile_invalid_utf8",
        "json_file",
        json_bytes.replace(b'"q one"', b'"q \xffone"'),
        "reject",
        "invalid UTF-8 bytes reject at decode",
        message_pin="not valid UTF-8",
    )
    add(
        "jfile_cesu_surrogate_bytes",
        "json_file",
        json_bytes.replace(b'"q one"', b'"q \xed\xa0\x80one"'),
        "reject",
        "CESU-8 surrogate bytes are not valid UTF-8: the file-level surrogate case",
        message_pin="not valid UTF-8",
    )
    add(
        "jfile_bom_bytes",
        "json_file",
        b"\xef\xbb\xbf" + json_bytes,
        "reject",
        "BOM bytes decode to U+FEFF, which the parser rejects",
    )
    add(
        "jfile_oversize",
        "json_file",
        json_bytes + b" " * _MAX_BYTES,
        "reject",
        "the byte cap is checked on the file's bytes",
        message_pin="exceeds size limit",
    )
    add(
        "jfile_directory",
        "json_file",
        b"",
        "reject",
        "a directory is not a regular file",
        file_kind="directory",
    )
    add(
        "jfile_missing",
        "json_file",
        b"",
        "reject",
        "a missing file is unreadable",
        file_kind="missing",
    )
    add(
        "jfile_fifo",
        "json_file",
        b"",
        "reject",
        "a FIFO is refused without blocking (O_NONBLOCK + fstat)",
        file_kind="fifo",
    )
    add(
        "jfile_symlink",
        "json_file",
        json_bytes,
        "accept",
        "a symlink to a regular file is followed by design",
        file_kind="symlink",
        counts=(2, 2),
    )

    add(
        "cfile_control",
        "csv_file",
        csv_bytes,
        "accept",
        "control: a valid authoring file loads",
        counts=(2, 2),
    )
    add(
        "cfile_invalid_utf8",
        "csv_file",
        csv_bytes.replace(b"q one", b"q \xffone"),
        "reject",
        "invalid UTF-8 bytes reject at decode",
        message_pin="not valid UTF-8",
    )
    add(
        "cfile_oversize",
        "csv_file",
        csv_bytes + b"\n" + b"x" * _MAX_BYTES,
        "reject",
        "the byte cap is checked on the file's bytes",
        message_pin="exceeds size limit",
    )
    add(
        "cfile_directory",
        "csv_file",
        b"",
        "reject",
        "a directory is not a regular file",
        file_kind="directory",
    )
    add(
        "cfile_missing",
        "csv_file",
        b"",
        "reject",
        "a missing file is unreadable",
        file_kind="missing",
    )
    add(
        "cfile_fifo",
        "csv_file",
        b"",
        "reject",
        "a FIFO is refused without blocking",
        file_kind="fifo",
    )
    return cases


def generate_cases(seed: int = 0) -> tuple[FuzzCase, ...]:
    """Build the full published corpus for ``seed``.

    Controls come first: if a control case ever fails, the harness --
    not the loaders -- is the first suspect.
    """
    return tuple(_json_cases(seed) + _csv_cases(seed) + _file_cases())
