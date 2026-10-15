"""Command-line entry points for local, bounded experiments."""

import argparse
import contextlib
import json
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

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _display_id(value: str) -> str:
    """Fixture IDs are operator-visible; strip control characters so a
    hostile fixture file cannot inject terminal escape sequences into
    stderr warnings."""
    return _CONTROL_CHARS.sub("", value)


def parse_boundary(value: str) -> tuple[int, ...]:
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
        "/no-bos, /prefix-newline, /prefix-trim with --mutate, and "
        "kolibri/thinking-off).",
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
        observations: list | None = None
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


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            print(json.dumps([asdict(fmt) for fmt in CATALOG.values()], indent=2))
            return 0
        # ANCHOR: new top-level command dispatch goes here (one block per command).
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
            print(
                json.dumps(
                    {
                        "search_cases": len(split.search),
                        "held_out_cases": len(split.held_out),
                        "valid": True,
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
            )
            print(json.dumps(result, indent=2))
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
            print(
                json.dumps(
                    {
                        "search_cases": len(split.search),
                        "held_out_cases": len(split.held_out),
                        "output": str(written),
                        "audit_findings": len(findings),
                        "audit_path": str(sidecar),
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
            previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda _signum, _frame: cancel.set())
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
                signal.signal(signal.SIGINT, previous)
            # Diagnosis sections; none of these may fail the run.
            try:
                from .doctor import template_conformance

                # Byte-space HF stop matching when the backend's tokenizer is
                # byte-oriented; otherwise the classic string path is used.
                token_bytes = getattr(backend, "hf_stop_token_bytes", None)
                if not callable(token_bytes):
                    token_bytes = None
                result["template_conformance"] = template_conformance(
                    args.model, formats, split.search
                )
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
                # Teacher-forced log-likelihood diagnostic (opt-in: one extra
                # forward pass per candidate x fixture). Scores each
                # candidate's expected answer against the recorded search
                # prompts; diagnostic only, never changes the selection rule.
                if args.logprob_diagnostic:
                    from .experiments import logprob_diagnostic

                    try:
                        result["logprob_diagnostic"] = logprob_diagnostic(
                            backend, result, split.search
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
        # arbitrary runtime text.
        message = str(exc)
        if message.startswith("local_"):
            print(f"Operation failed [{message}]. See docs/USAGE.md.", file=sys.stderr)
        else:
            print(
                "Operation failed. Check input paths, local model support, "
                "dependencies and output permissions.",
                file=sys.stderr,
            )
        return 1
    except (OSError, ImportError):
        # These may carry local paths or dependency text; keep them generic.
        print(
            "Operation failed. Check input paths, local model support, "
            "dependencies and output permissions.",
            file=sys.stderr,
        )
        return 1
