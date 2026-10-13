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
    run = sub.add_parser(
        "run", help="Evaluate a local checkpoint and save a report", allow_abbrev=False
    )
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--dataset", required=True, type=Path)
    run.add_argument("--family", choices=sorted(CATALOG), required=True)
    run.add_argument("--mutate", action="store_true", help="Include bounded local repairs")
    run.add_argument("--boundary-ids", type=parse_boundary, required=True)
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
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            print(json.dumps([asdict(fmt) for fmt in CATALOG.values()], indent=2))
            return 0
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

            result = doctor(args.model, args.dataset, boundary_ids=args.boundary_ids)
            print(json.dumps(result, indent=2))
            return 0 if result["verdict"] == "ok" else 1
        if not 1 <= args.max_new_tokens <= 4096 or not 1 <= args.max_context <= 32768:
            parser.error(
                "Token/context limits are outside supported ranges: "
                "--max-new-tokens 1-4096, --max-context 1-32768"
            )
        if not 0 <= args.seed < 2**64:
            parser.error("Seed must be an unsigned 64-bit integer")
        if args.output.exists():
            parser.error("Output path already exists; use a fresh directory")
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
            cancel = Event()
            previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda _signum, _frame: cancel.set())
            try:
                result = run_experiment(
                    backend,
                    candidates(args.family, mutate=args.mutate),
                    split,
                    boundary=args.boundary_ids,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    cancel=cancel,
                )
            finally:
                signal.signal(signal.SIGINT, previous)
            save_report(
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
                {"decision": result["decision"], "selected_candidate": result["selected_candidate"]}
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
