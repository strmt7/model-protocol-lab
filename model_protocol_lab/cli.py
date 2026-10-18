"""Command-line entry points for local, bounded experiments."""

import argparse
import contextlib
import json
import os
import re
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from threading import Event
from typing import Any

from .catalog import CATALOG, candidates
from .datasets import load_dataset
from .experiments import run_experiment
from .reporting import save_report

_CONTROL_CHARS = re.compile(
    r"[\x00-\x1f\x7f-\x9f\ud800-\udfff\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]"
)
# Lone surrogates are stripped too: they crash strict stream writes with
# UnicodeEncodeError and are never valid text on their own. The \u200e-\u2069
# ranges strip Unicode bidi controls: a hostile fixture ID like
# "safe-\u202e-evil" renders right-to-left in the operator's terminal
# (Trojan Source spoofing, CVE-2021-42574), and U+2028/U+2029 break line
# discipline. Other invisible format characters (BOM, zero-width space, soft
# hyphen, ...) are deliberately preserved: they are diagnostically valuable
# in error messages (e.g. a BOM-prefixed CSV header shows the operator exactly
# why it was rejected) and cannot reorder displayed text.


def _display_id(value: str) -> str:
    """Fixture IDs are operator-visible; strip control characters so a
    hostile fixture file cannot inject terminal escape sequences into
    stderr warnings."""
    return _CONTROL_CHARS.sub("", value)


def _warn_contamination(command: str, contamination: dict[str, Any]) -> None:
    """Print the contamination tier table to stderr as a non-fatal warning.

    Advisory only: the tiers are heuristic, so a flagged pair never fails
    the command -- it tells the dataset author the held-out split may be
    contaminated by near-duplicates of search fixtures. At most 10 pairs
    print; the machine JSON carries the full list.
    """
    pairs = contamination.get("pairs")
    if type(pairs) is not list or not pairs:
        return
    lines = [
        f"{command}: warning: search/held-out contamination: "
        f"worst tier {contamination.get('worst_tier')} ({len(pairs)} pair(s)):"
    ]
    for pair in pairs[:10]:
        if type(pair) is not dict:
            continue
        lines.append(
            f"  held_out {_display_id(str(pair.get('held_out')))} ~ "
            f"search {_display_id(str(pair.get('search')))} "
            f"(ratio {pair.get('ratio')}, jaccard {pair.get('jaccard')}, {pair.get('tier')})"
        )
    if len(pairs) > 10:
        lines.append(f"  ... and {len(pairs) - 10} more (see JSON)")
    print("\n".join(lines), file=sys.stderr)


def parse_boundary(value: str) -> tuple[int, ...]:
    """Argparse type: parse 1-16 comma-separated ASCII token IDs.

    Only ASCII digits are accepted (Unicode digits would silently coerce via
    int()); IDs must fit below 2**31. Raises argparse.ArgumentTypeError on
    any malformed input.
    """

    parts = value.split(",")
    # ASCII digits only: isdecimal() accepts Unicode digits (e.g. fullwidth),
    # which would silently coerce. Token IDs are ASCII by convention.
    if not 1 <= len(parts) <= 16 or any(
        not p.strip() or any(not ("0" <= ch <= "9") for ch in p.strip()) for p in parts
    ):
        raise argparse.ArgumentTypeError("Use 1-16 comma-separated ASCII token IDs")
    try:
        result = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid token IDs") from exc
    if any(token > 2**31 - 1 for token in result):
        raise argparse.ArgumentTypeError("Token IDs must be < 2^31")
    return result


def parse_boundary_map(value: str) -> tuple[str, tuple[int, ...]]:
    """Parse one NAME=ID,ID per-candidate boundary override.

    Candidate names may contain "/" (e.g. "chatml/toggle-bos") but never
    "=", so the split is on the first "=".
    """
    name, separator, ids = value.partition("=")
    name = name.strip()
    if not separator or not name:
        raise argparse.ArgumentTypeError("Use NAME=ID,ID (for example chatml=198)")
    return name, parse_boundary(ids)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands.

    Abbreviation is disabled on every parser so option prefixes can never
    silently resolve to the wrong flag. New top-level commands are dispatched
    in main() at the ANCHOR comment.
    """

    parser = argparse.ArgumentParser(prog="model-protocol-lab", allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog", help="List bounded text-only protocol hypotheses", allow_abbrev=False)
    inspect = sub.add_parser(
        "inspect", help="Validate a fixture file without loading a model", allow_abbrev=False
    )
    inspect.add_argument("dataset", type=Path)
    doctor_cmd = sub.add_parser(
        "doctor",
        help="Pre-flight check of a local checkpoint (never loads weights)",
        allow_abbrev=False,
    )
    doctor_cmd.add_argument("--model", required=True, type=Path)
    doctor_cmd.add_argument("--dataset", type=Path, default=None)
    doctor_cmd.add_argument("--boundary-ids", type=parse_boundary, default=None)
    doctor_cmd.add_argument(
        "--boundary-map",
        type=parse_boundary_map,
        action="append",
        default=None,
        help="Per-candidate boundary override as NAME=ID,ID (repeatable); "
        "validated against the checkpoint vocabulary like --boundary-ids",
    )
    doctor_cmd.add_argument(
        "--reference-model",
        type=Path,
        default=None,
        help="Optional second local checkpoint: its tokenizer re-encodes the "
        "boundary marker to check cross-tokenizer boundary portability",
    )
    doctor_cmd.add_argument(
        "--portability-matrix",
        type=Path,
        default=None,
        help="Optional second local checkpoint: every catalog candidate's "
        "special-token inventory is checked against its tokenizer "
        "(registered, single-token, id-consistent) and reported as a "
        "compatibility matrix. Tokenizer only, never weights. "
        "'incompatible' rules the checkpoint out; 'compatible' predicts "
        "nothing about experimental success (see docs/USAGE.md section 2).",
    )
    run = sub.add_parser(
        "run", help="Evaluate a local checkpoint and save a report", allow_abbrev=False
    )
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--dataset", required=True, type=Path)
    run.add_argument("--family", choices=sorted(CATALOG), required=True)
    run.add_argument("--mutate", action="store_true", help="Include bounded local repairs")
    run.add_argument("--boundary-ids", type=parse_boundary, required=True)
    run.add_argument(
        "--boundary-map",
        type=parse_boundary_map,
        action="append",
        default=None,
        help="Per-candidate boundary override as NAME=ID,ID (repeatable). "
        "Each candidate is measured with its own boundary; the default "
        "--boundary-ids applies to candidates without an entry. Names must "
        "match the evaluated candidates (see `catalog`, plus /toggle-bos, "
        "/no-bos, /prefix-newline, /prefix-trim with --mutate, and the "
        "thinking/effort variants: kolibri/thinking-off, "
        "nemotron35/thinking-off, ling/thinking-off, laguna/thinking-on, "
        "qwen38/thinking-off, nanbeige42/thinking-off, "
        "k2horizon/effort-medium, k2horizon/effort-low, hy4/no-think, "
        "minimax3/thinking-disabled, minimax3/thinking-enabled).",
    )
    run.add_argument("--max-new-tokens", type=int, default=64)
    run.add_argument("--max-context", type=int, default=2048)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--output", required=True, type=Path, help="New local report directory")
    run.add_argument(
        "--include-evidence",
        action="store_true",
        help="Include potentially sensitive raw answer/token traces in JSON",
    )
    run.add_argument(
        "--sensitivity-sweep",
        action="store_true",
        help="Re-run search with the prompt truncated mid-token at each depth "
        "(1..--max-drop-bytes) to measure boundary sensitivity; multiplies "
        "search compute",
    )
    run.add_argument(
        "--max-drop-bytes",
        type=int,
        default=3,
        help="Truncation depths for --sensitivity-sweep (1-8)",
    )
    run.add_argument(
        "--logprob-diagnostic",
        action="store_true",
        help="Teacher-force each candidate's expected answer after search "
        "(one forward pass per candidate x fixture) and report "
        "per-candidate log-likelihoods; diagnostic only, never changes the "
        "selection rule",
    )
    fixtures_cmd = sub.add_parser(
        "fixtures",
        help="Authoring tools for fixture files (never loads a model)",
        allow_abbrev=False,
    )
    fixtures_sub = fixtures_cmd.add_subparsers(dest="fixtures_command", required=True)
    fixtures_build = fixtures_sub.add_parser(
        "build",
        help="Build a strict fixture JSON file from a compact authoring CSV",
        allow_abbrev=False,
    )
    fixtures_build.add_argument("source", type=Path, help="Authoring CSV file")
    fixtures_build.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New fixture JSON file; an existing path is never overwritten",
    )
    # ANCHOR: new top-level subparsers are registered here (one block per command).
    from .verify import build_verify_parser

    build_verify_parser(sub)
    diff_reports_cmd = sub.add_parser(
        "diff-reports",
        help="Compare two experiment result directories (or results.json files)",
        allow_abbrev=False,
    )
    diff_reports_cmd.add_argument(
        "old_results",
        type=Path,
        help="Old report directory (containing results.json) or results.json path",
    )
    diff_reports_cmd.add_argument(
        "new_results",
        type=Path,
        help="New report directory (containing results.json) or results.json path",
    )
    diff_reports_cmd.add_argument(
        "--all",
        action="store_true",
        help="List every compared fixture outcome, not just changed ones "
        "(requires --include-evidence on both runs)",
    )
    diff_cmd = sub.add_parser(
        "diff-backends",
        help="Differential-test emulated backend stop semantics on one token sequence",
        allow_abbrev=False,
    )
    diff_cmd.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Local checkpoint directory (tokenizer only; weights are never loaded)",
    )
    diff_cmd.add_argument(
        "--prompt-text",
        required=True,
        help="Prompt text (encoded with add_special_tokens=False)",
    )
    diff_cmd.add_argument(
        "--output-text",
        required=True,
        help="Recorded output text (encoded with add_special_tokens=False)",
    )
    diff_cmd.add_argument(
        "--stops",
        required=True,
        help='Comma-separated stop texts, e.g. "STOP,</answer>"',
    )
    diff_cmd.add_argument(
        "--eos-id",
        type=int,
        default=None,
        help="EOS token id for the eos_appended cell",
    )
    diff_cmd.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="vLLM min_tokens gating applied to the holdback cells",
    )
    debug_cmd = sub.add_parser(
        "debug",
        help="Interactive protocol debugger REPL (offline; tokenizer-only)",
        allow_abbrev=False,
    )
    debug_cmd.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Fixture JSON file loaded via datasets.load_dataset",
    )
    debug_cmd.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Local checkpoint directory; attaches its tokenizer only "
        "(never loads model weights) for exact token IDs",
    )
    debug_cmd.add_argument(
        "--script",
        type=Path,
        default=None,
        help="Run a .read script non-interactively and exit",
    )
    return parser


def _backend_eos_id(backend: Any) -> int | None:
    """EOS token ID from a backend that exposes one, else None."""
    eos = getattr(backend, "eos_token_id", None)
    return eos if type(eos) is int and eos >= 0 else None


def _resolve_boundary_map(
    entries: list[tuple[str, tuple[int, ...]]] | None,
    candidate_names: set[str] | None,
    parser: argparse.ArgumentParser,
) -> dict[str, tuple[int, ...]] | None:
    """Validate repeatable --boundary-map flags.

    Fails fast (exit 2) on duplicate names; when the candidate set is known
    (``run``), unknown names also fail fast so a typo cannot silently run a
    whole experiment with the default boundary. ``doctor`` does not know the
    candidate set, so it checks duplicates only. The returned dict maps
    candidate name to its override boundary IDs.
    """
    if not entries:
        return None
    resolved: dict[str, tuple[int, ...]] = {}
    for name, ids in entries:
        if name in resolved:
            parser.error(f"Duplicate --boundary-map candidate: {name!r}")
        if candidate_names is not None and name not in candidate_names:
            parser.error(
                f"--boundary-map names unknown candidate {name!r}; "
                f"valid names: {', '.join(sorted(candidate_names))}"
            )
        resolved[name] = ids
    return resolved


def _matrix_stage_rows(
    result: dict[str, Any],
    stage_key: str,
    winner_name: str,
    cases: tuple[Any, ...],
    backend: Any,
    boundary: tuple[int, ...],
    eos_id: int | None,
    token_bytes: Any = None,
) -> list[dict[str, Any]] | None:
    """Build stop-matrix rows for one stage from a run_experiment result."""
    if stage_key == "search":
        entries = result.get("search") or []
        observations: list[dict[str, Any]] | None = None
        for entry in entries:
            if (
                type(entry) is dict
                and type(entry.get("format")) is dict
                and entry["format"].get("name") == winner_name
            ):
                observations = entry.get("observations")
                break
    else:
        held = result.get("held_out")
        observations = held.get("observations") if type(held) is dict else None
    if type(observations) is not list or len(observations) != len(cases):
        return None
    from .experiments import stop_matrix

    return stop_matrix(
        backend,
        winner_name,
        cases,
        tuple(observations),
        boundary=boundary,
        eos_id=eos_id,
        token_bytes=token_bytes,
    )


_KNOWN_LOCAL_FAILURE_CODES = (
    "local_context_limit_exceeds_declared_window",
    "local_hf_extra_not_installed",
    "local_backend_initialization_failed",
    "local_backend_not_initialized",
    "local_encode_failed",
    "local_decode_failed",
    "local_forward_failed",
)
"""Backend RuntimeError codes the CLI is allowed to echo.

Every ``raise RuntimeError("local_*")`` in ``backends.py`` is a fixed code
raised ``from None``; the allowlist below is checked against that exact set
(a test pins the membership). A prefix match would echo any
``local_``-prefixed text, so unknown codes stay generic.
"""


def _backend_failure_message(message: str) -> str:
    """Render a backend RuntimeError code as a sanitized CLI message.

    Only fixed ``local_*`` codes are shown; anything else stays generic so
    arbitrary runtime text never reaches the terminal. The missing-extra code
    carries its own actionable install hint.
    """
    if message == "local_hf_extra_not_installed":
        return (
            "Operation failed [local_hf_extra_not_installed]: this command needs "
            'the Hugging Face extra. Install it with: pip install "model-protocol-lab[hf]".'
        )
    if message in _KNOWN_LOCAL_FAILURE_CODES:
        return f"Operation failed [{message}]. See docs/USAGE.md."
    return (
        "Operation failed. Check input paths, local model support, "
        "dependencies and output permissions."
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse argv and dispatch to the requested subcommand.

    Returns the process exit code (0 on success). Accepts an optional argv
    list for programmatic/testing use; defaults to sys.argv.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            print(json.dumps([asdict(fmt) for fmt in CATALOG.values()], indent=2))
            return 0
        # ANCHOR: new top-level command dispatch goes here (one block per command).
        if args.command == "verify":
            from .verify import run_verify

            return run_verify(args, parser)
        if args.command == "diff-reports":
            from .diffreports import run_diff_reports

            return run_diff_reports(args, parser)
        if args.command == "diff-backends":
            from .diffbackends import run_diff_backends

            return run_diff_backends(args, parser)
        if args.command == "debug":
            from .debug import run_debug

            return run_debug(args)
        if args.command == "inspect":
            try:
                split = load_dataset(args.dataset)  # nosec B615 -- own bounded JSON reader
            except ValueError as exc:
                # inspect exists to diagnose fixture files; show the reason.
                print(f"inspect: invalid fixture file: {exc}", file=sys.stderr)
                return 1
            empty_expected = [
                case.id for case in (*split.search, *split.held_out) if not case.expected
            ]
            if empty_expected:
                # An empty expected answer is trivially satisfiable: any
                # output that emits only the boundary token passes. Warn so
                # the dataset author knows the measurement is vacuous there.
                print(
                    "inspect: warning: empty expected answer in fixture(s) "
                    f"{', '.join(_display_id(i) for i in empty_expected)}; these pass vacuously",
                    file=sys.stderr,
                )
            from .contamination import contamination_report

            contamination = contamination_report(split.search, split.held_out)
            _warn_contamination("inspect", contamination)
            print(
                json.dumps(
                    {
                        "search_cases": len(split.search),
                        "held_out_cases": len(split.held_out),
                        "valid": True,
                        "contamination": contamination,
                    }
                )
            )
            return 0
        if args.command == "doctor":
            from .doctor import doctor

            result = doctor(
                args.model,
                args.dataset,
                boundary_ids=args.boundary_ids,
                boundary_map=_resolve_boundary_map(args.boundary_map, None, parser),
                reference_model_path=args.reference_model,
                portability_matrix_path=args.portability_matrix,
            )
            print(json.dumps(result, indent=2))
            if args.portability_matrix is not None:
                # The human-readable matrix goes to stderr so stdout stays
                # a parseable JSON report.
                from .doctor import render_portability_table

                section = result.get("portability_matrix")
                if type(section) is dict:
                    print(render_portability_table(section), file=sys.stderr)
            return 0 if result["verdict"] == "ok" else 1
        if args.command == "fixtures":
            from .fixtures import (
                audit_findings_json,
                audit_special_token_literals,
                parse_fixture_csv,
                read_authoring_file,
                split_to_json_text,
                write_fixture_file,
            )

            try:
                authoring = read_authoring_file(args.source)  # nosec B615 -- own bounded reader
                split, empty_expected = parse_fixture_csv(authoring)
                if empty_expected:
                    print(
                        "fixtures: warning: empty expected answer in fixture(s) "
                        f"{', '.join(_display_id(i) for i in empty_expected)}; "
                        "these pass vacuously",
                        file=sys.stderr,
                    )
                # Fixture hygiene (advisory GHSA-g5f9-3xfg-p9mf): a literal
                # special token in fixture text would forge a role boundary
                # when the fixture is composed into a prompt. The inventory
                # comes from the base catalog formats; mutated candidates
                # derive from the same framing strings, so the base
                # inventory covers them. Findings warn only -- the build
                # still succeeds and the exit code is unchanged.
                findings = audit_special_token_literals(split, tuple(CATALOG.values()))
                for finding in findings:
                    print(
                        "fixtures: warning: special-token literal "
                        f"{finding.literal!r} in fixture "
                        f"{_display_id(finding.fixture_id)!r} field "
                        f"{finding.field!r} at offset {finding.offset} "
                        f"collides with format {finding.format_name!r}",
                        file=sys.stderr,
                    )
                # The strict fixture schema rejects extra top-level keys, so
                # findings are recorded in a sidecar, never in the fixture
                # file itself. Pre-check the sidecar path so a collision is
                # reported before the fixture file is written.
                sidecar = Path(str(args.output) + ".audit.json")
                if sidecar.exists():
                    raise ValueError("audit sidecar already exists; use a fresh path")
                written = write_fixture_file(split_to_json_text(split), args.output)
                try:
                    write_fixture_file(audit_findings_json(findings, source=str(written)), sidecar)
                except ValueError as sidecar_exc:
                    # A concurrent writer won the sidecar race after our
                    # pre-check: remove the fixture just written so no
                    # half-built pair is left, and report the collision
                    # distinctly from authoring-source errors.
                    written.unlink(missing_ok=True)
                    print(
                        f"fixtures: output collision, nothing written: {sidecar_exc}",
                        file=sys.stderr,
                    )
                    return 1
            except ValueError as exc:
                # fixtures exists to diagnose authoring input; show the reason.
                print(f"fixtures: invalid authoring source: {exc}", file=sys.stderr)
                return 1
            from .contamination import contamination_report

            build_contamination = contamination_report(split.search, split.held_out)
            _warn_contamination("fixtures", build_contamination)
            print(
                json.dumps(
                    {
                        "search_cases": len(split.search),
                        "held_out_cases": len(split.held_out),
                        "output": str(written),
                        "audit_findings": len(findings),
                        "audit_path": str(sidecar),
                        "contamination": build_contamination,
                    }
                )
            )
            return 0
        if not 1 <= args.max_new_tokens <= 4096 or not 1 <= args.max_context <= 32768:
            parser.error(
                "Token/context limits are outside supported ranges: "
                "--max-new-tokens 1-4096, --max-context 1-32768"
            )
        if not 0 <= args.seed < 2**64:
            parser.error("Seed must be an unsigned 64-bit integer")
        if not 1 <= args.max_drop_bytes <= 8:
            parser.error("--max-drop-bytes must be between 1 and 8")
        if args.output.exists():
            parser.error("Output path already exists; use a fresh directory")
        # The candidate set is pure (no model involved): compute it before
        # any expensive work so a mistyped --boundary-map name fails fast
        # instead of silently running with the default boundary.
        formats = candidates(args.family, mutate=args.mutate)
        boundary_map = _resolve_boundary_map(
            args.boundary_map, {fmt.name for fmt in formats}, parser
        )
        # Claim the output directory BEFORE the expensive backend init and
        # experiment run: an unwritable parent or a path through a regular
        # file fails here, cheaply, instead of discarding hours of compute
        # after the run completes. On failure before the report is saved the
        # directory is removed again so a retry still sees a fresh path.
        from .reporting import prepare_report_dir

        # Remember which parents did not exist before the claim, so cleanup
        # can remove the whole created chain (not just the leaf).
        _pre_existing = set()
        _probe = Path(args.output)
        while not _probe.exists():
            _probe = _probe.parent
        _pre_existing.add(_probe)
        output_dir = prepare_report_dir(args.output)
        _created_chain = []
        _probe = output_dir
        while _probe not in _pre_existing:
            _created_chain.append(_probe)
            _probe = _probe.parent
        report_saved = False
        try:
            # own bounded local JSON reader; not huggingface_hub.datasets.
            split = load_dataset(args.dataset)  # nosec B615
            empty_expected = [
                case.id for case in (*split.search, *split.held_out) if not case.expected
            ]
            if empty_expected:
                print(
                    "run: warning: empty expected answer in fixture(s) "
                    f"{', '.join(_display_id(i) for i in empty_expected)}; these pass vacuously",
                    file=sys.stderr,
                )
            from .backends import LocalHFBackend

            backend = LocalHFBackend(args.model, device=args.device, max_context=args.max_context)
            eos_id = _backend_eos_id(backend)
            cancel = Event()
            previous_int = signal.getsignal(signal.SIGINT)
            previous_term = signal.getsignal(signal.SIGTERM)
            # Both SIGINT (Ctrl-C) and SIGTERM (orchestrated shutdown) drive
            # the same cooperative cancellation: the experiment drains to a
            # "cancelled" verdict and a valid report instead of dying
            # mid-write and leaving a claimed output directory behind.
            signal.signal(signal.SIGINT, lambda _signum, _frame: cancel.set())
            signal.signal(signal.SIGTERM, lambda _signum, _frame: cancel.set())
            # Test-only readiness hook (no effect unless the env var is set):
            # an orchestrating integration test cannot otherwise observe when
            # the signal handlers are installed, forcing racy timing guesses.
            # Writing the marker file makes the "handlers installed" moment
            # observable deterministically. The name deliberately avoids the
            # three-letter MPL prefix: the package contract (pinned by
            # test_package_reads_no_mpl_env_vars) forbids such env vars.
            _ready_marker = os.environ.get("MODEL_PROTOCOL_LAB_TEST_SIGNAL_READY_FILE")
            if _ready_marker:
                try:
                    with open(_ready_marker, "w", encoding="utf-8") as _rf:
                        _rf.write("ready")
                except OSError:
                    pass
            try:
                result = run_experiment(
                    backend,
                    formats,
                    split,
                    boundary=args.boundary_ids,
                    boundary_map=boundary_map,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    cancel=cancel,
                    eos_id=eos_id,
                )
            finally:
                signal.signal(signal.SIGINT, previous_int)
                signal.signal(signal.SIGTERM, previous_term)
            # Fixture hygiene (pure, advisory): search/held-out
            # near-duplicate overlap, so the report records whether the
            # held-out split was contaminated. Never fails the run.
            from .contamination import contamination_report

            result["contamination"] = contamination_report(split.search, split.held_out)
            # Diagnosis sections; none of these may fail the run.
            try:
                from .doctor import marker_isolation, template_conformance, template_firewall

                # Byte-space HF stop matching when the backend's tokenizer is
                # byte-oriented; otherwise the classic string path is used.
                token_bytes = getattr(backend, "hf_stop_token_bytes", None)
                if not callable(token_bytes):
                    token_bytes = None
                result["template_conformance"] = template_conformance(
                    args.model, formats, split.search
                )
                # Template firewall (research R3): static AST scan of the
                # checkpoint's chat_template for RCE primitives, before any
                # render. Fail-closed; never fails the run.
                result["template_firewall"] = template_firewall(args.model)
                # Chat-marker isolation (research R2/R4 line): hostile marker
                # strings injected into user text must not become real
                # control tokens. Pure tokenizer check, never fails the run.
                result["marker_isolation"] = marker_isolation(args.model, formats, split.search)
                if result["selected_candidate"] is not None:
                    winner = result["selected_candidate"]
                    # The stop matrix re-measures the winner under its own
                    # boundary (per-candidate boundaries, Cycle 2).
                    winner_boundary = (
                        boundary_map.get(winner, args.boundary_ids)
                        if boundary_map
                        else args.boundary_ids
                    )
                    search_rows = _matrix_stage_rows(
                        result,
                        "search",
                        winner,
                        split.search,
                        backend,
                        winner_boundary,
                        eos_id,
                        token_bytes,
                    )
                    held_rows = _matrix_stage_rows(
                        result,
                        "held_out",
                        winner,
                        split.held_out,
                        backend,
                        winner_boundary,
                        eos_id,
                        token_bytes,
                    )
                    result["stop_matrix"] = {
                        "candidate": result["selected_candidate"],
                        "semantics": [
                            {
                                "id": "native",
                                "label": "Native (lab)",
                                "description": (
                                    "The lab's own behavior: the full requested "
                                    "token cap is generated with no stop; the "
                                    "boundary is found by token-ID subsequence "
                                    "match afterwards."
                                ),
                            },
                            {
                                "id": "hf_stop_string",
                                "label": "HF StopStringCriteria",
                                "description": (
                                    "Emulates HF transformers' StopStringCriteria "
                                    "overhang matching on the decoded boundary "
                                    "text -- byte-space matching for byte-level "
                                    "or byte-fallback tokenizers, mirroring "
                                    "current transformers: generation stops "
                                    "after the token that completes the stop "
                                    "string."
                                ),
                            },
                            {
                                "id": "vllm_holdback",
                                "label": "vLLM detokenizer",
                                "description": (
                                    "Emulates vLLM's detokenizer stop handling: "
                                    "the earliest-completing stop string in the "
                                    "decoded text truncates the output before it "
                                    "(exclude semantics)."
                                ),
                            },
                            {
                                "id": "eos_appended",
                                "label": "EOS-appended",
                                "description": (
                                    "Emulates a competitor that stops on EOS "
                                    "(lm-eval appends EOS to its stops; vLLM and "
                                    "llama.cpp stop on EOS by default): truncates "
                                    "before the first EOS token."
                                ),
                            },
                        ],
                        "search": search_rows,
                        "held_out": held_rows,
                    }
                    # Backend-divergence diagnosis: differential-test the
                    # emulated stop semantics over the winner's recorded
                    # observations (search and held_out), with the decoded
                    # boundary as the stop text -- mirroring how the
                    # stop-matrix path derives it. Omitted when no
                    # observations are usable; like every diagnosis here,
                    # this must not fail the run.
                    _divergence_obs: list[dict[str, Any]] = []
                    for _entry in result.get("search") or []:
                        if (
                            type(_entry) is dict
                            and type(_entry.get("format")) is dict
                            and _entry["format"].get("name") == winner
                            and type(_entry.get("observations")) is list
                        ):
                            _divergence_obs.extend(
                                _obs for _obs in _entry["observations"] if type(_obs) is dict
                            )
                            break
                    _held = result.get("held_out")
                    _held_obs = _held.get("observations") if type(_held) is dict else None
                    if type(_held_obs) is list:
                        _divergence_obs.extend(_obs for _obs in _held_obs if type(_obs) is dict)
                    try:
                        _boundary_text: Any = backend.decode(winner_boundary)
                    except Exception:  # noqa: BLE001 -- string cells degrade, never fail
                        _boundary_text = None
                    if type(_boundary_text) is str and _boundary_text and _divergence_obs:
                        from .diffbackends import divergence_section_data

                        try:
                            _divergence = divergence_section_data(
                                observations=_divergence_obs,
                                stop_texts=(_boundary_text,),
                                eos_id=eos_id,
                                min_tokens=0,
                                backend=backend,
                            )
                        except Exception:  # noqa: BLE001 -- new section degrades alone
                            _divergence = None
                        if _divergence is not None:
                            result["backend_divergences"] = _divergence
                # Token-budget frontier: post-hoc, exact, no model calls.
                # Kept out of the stop-matrix's `if` so ambiguous and
                # no-candidate runs get one too; it degrades to an
                # "unavailable" marker instead of failing the run.
                try:
                    from .experiments import budget_frontier

                    if result["decision"] in (
                        "compatible_on_fixture_set",
                        "held_out_failed",
                        "ambiguous",
                        "no_candidate",
                    ):
                        result["budget_frontier"] = budget_frontier(
                            result.get("search") or [],
                            boundary=args.boundary_ids,
                            max_new_tokens=args.max_new_tokens,
                            native_decision=result["decision"],
                            native_winner=result.get("selected_candidate"),
                        )
                    else:
                        result["budget_frontier"] = {
                            "available": False,
                            "detail": "search_incomplete",
                        }
                except Exception:  # noqa: BLE001 -- diagnostics must not fail the run
                    result["budget_frontier"] = {
                        "available": False,
                        "detail": "frontier_unavailable",
                    }
                # Selection robustness under stop semantics: post-hoc,
                # exact, no model calls. Re-applies the selection rule to
                # every search candidate under each emulated stop semantic
                # so a user deploying on a different serving stack sees
                # whether the winner would change. Degrades to an
                # "unavailable" marker instead of failing the run.
                try:
                    from .experiments import selection_robustness

                    result["selection_robustness"] = selection_robustness(
                        backend,
                        result,
                        tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        eos_id=eos_id,
                        token_bytes=token_bytes,
                    )
                except Exception:  # noqa: BLE001 -- diagnostics must not fail the run
                    result["selection_robustness"] = {
                        "available": False,
                        "detail": "selection_robustness_unavailable",
                    }
                # Held-out budget floor: post-hoc, exact, no model calls.
                # Only meaningful when the held-out stage completed; it
                # degrades to an "unavailable" marker instead of failing.
                try:
                    from .experiments import held_out_budget_floor

                    held_out = result.get("held_out") or {}
                    held_observations = held_out.get("observations") or []
                    held_boundary = result.get("selected_boundary_token_ids")
                    if (
                        result["decision"] in ("compatible_on_fixture_set", "held_out_failed")
                        and held_observations
                        and held_boundary
                    ):
                        result["held_out_budget_floor"] = held_out_budget_floor(
                            held_observations,
                            boundary=tuple(held_boundary),
                            max_new_tokens=args.max_new_tokens,
                            native_verdict=result["decision"],
                        )
                    else:
                        result["held_out_budget_floor"] = {
                            "available": False,
                            "detail": "held_out_incomplete",
                        }
                except Exception:  # noqa: BLE001 -- diagnostics must not fail the run
                    result["held_out_budget_floor"] = {
                        "available": False,
                        "detail": "held_out_floor_unavailable",
                    }
                # Prompt-boundary canonicality audit (research R2): re-encode
                # each stored prompt+completion and require the joint encoding
                # to equal prompt_ids + completion_ids. Pure post-hoc check
                # over stored artifacts; per-observation encode/decode
                # failures degrade to "unusable", never a run failure.
                try:
                    from .boundary_audit import audit_search_observations

                    result["boundary_audit"] = audit_search_observations(
                        result.get("search") or [],
                        backend.encode,
                        backend.decode,
                    )
                except Exception:  # noqa: BLE001 -- diagnostics must not fail the run
                    result["boundary_audit"] = {
                        "available": False,
                        "detail": "boundary_audit_unavailable",
                    }
                # Teacher-forced log-likelihood diagnostic (opt-in: one extra
                # forward pass per candidate x fixture). Scores each
                # candidate's expected answer against the recorded search
                # prompts; diagnostic only, never changes the selection rule.
                if args.logprob_diagnostic:
                    from .experiments import logprob_diagnostic

                    try:
                        result["logprob_diagnostic"] = logprob_diagnostic(
                            backend, result, split.search, cancel=cancel
                        )
                    except Exception:  # noqa: BLE001 -- diagnostics must not fail the run
                        result["logprob_diagnostic"] = {
                            "available": False,
                            "detail": "diagnostic_failed",
                            "method": "",
                            "thresholds": {},
                            "candidates": [],
                        }
                if args.sensitivity_sweep:
                    from .experiments import prompt_sensitivity_sweep

                    result["sensitivity_sweep"] = prompt_sensitivity_sweep(
                        backend,
                        formats,
                        split,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_drop_bytes=args.max_drop_bytes,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                        cancel=cancel,
                    )
            except Exception:  # noqa: BLE001 -- diagnostics must not fail the run
                result["diagnostics_unavailable"] = True
            results_path, report_path = save_report(
                result,
                args.output,
                include_evidence=args.include_evidence,
                _allow_empty_existing=True,  # pre-claimed above
            )
            report_saved = True
        finally:
            if not report_saved:
                # The run produced no report: release the claimed directory
                # (and any partial files / newly-created parents) so a retry
                # does not trip the freshness check.
                for _component in _created_chain:
                    with contextlib.suppress(OSError):
                        if _component.is_dir():
                            for _child in _component.iterdir():
                                with contextlib.suppress(OSError):
                                    _child.unlink()
                            _component.rmdir()
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "selected_candidate": result["selected_candidate"],
                    # Point the operator at the artifacts: the first-run
                    # walk showed the bare decision JSON left new users
                    # guessing where the report went.
                    "results": str(results_path),
                    "report": str(report_path),
                }
            )
        )
        if result["decision"] == "cancelled":
            return 130
        if result["decision"] == "failed":
            return 1
        return 0 if result["decision"] == "compatible_on_fixture_set" else 2
    except KeyboardInterrupt:
        print("Interrupted before a complete report was saved.", file=sys.stderr)
        return 130
    except ValueError as exc:
        # ValueError text here is the deliberate sanitized vocabulary (fixed
        # error codes like "unsupported_model_type"); safe and useful to show.
        print(f"Operation failed [{exc}]. See docs/USAGE.md for this check.", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        # Backend failures use fixed "local_*" codes; show those, but not
        # arbitrary runtime text (see _backend_failure_message).
        print(_backend_failure_message(str(exc)), file=sys.stderr)
        return 1
    except (OSError, ImportError):
        # These may carry local paths or dependency text; keep them generic.
        print(
            "Operation failed. Check input paths, local model support, "
            "dependencies and output permissions.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
