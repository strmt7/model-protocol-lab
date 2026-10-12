"""Command-line entry points for local, bounded experiments."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
from threading import Event

from .catalog import CATALOG, candidates
from .datasets import load_dataset
from .experiments import run_experiment
from .reporting import save_report


def parse_boundary(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if not 1 <= len(parts) <= 16 or any(not p.strip().isdecimal() for p in parts):
        raise argparse.ArgumentTypeError("Use 1-16 comma-separated nonnegative token IDs")
    try:
        result = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid token IDs") from exc
    if any(token > 2**31 - 1 for token in result):
        raise argparse.ArgumentTypeError("Token IDs exceed supported range")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-protocol-lab", allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog", help="List bounded text-only protocol hypotheses", allow_abbrev=False)
    inspect = sub.add_parser("inspect", help="Validate a fixture file without loading a model", allow_abbrev=False)
    inspect.add_argument("dataset", type=Path)
    run = sub.add_parser("run", help="Evaluate a local checkpoint and save a report", allow_abbrev=False)
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
    run.add_argument("--include-evidence", action="store_true",
                     help="Include potentially sensitive raw answer/token traces in JSON")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            print(json.dumps([asdict(fmt) for fmt in CATALOG.values()], indent=2))
            return 0
        if args.command == "inspect":
            split = load_dataset(args.dataset)
            print(json.dumps({"search_cases": len(split.search),
                              "held_out_cases": len(split.held_out), "valid": True}))
            return 0
        if not 1 <= args.max_new_tokens <= 4096 or not 1 <= args.max_context <= 32768:
            parser.error("Token/context limits are outside supported ranges")
        if not 0 <= args.seed < 2**64:
            parser.error("Seed must be an unsigned 64-bit integer")
        if args.output.exists():
            parser.error("Output directory already exists; use a fresh directory")
        split = load_dataset(args.dataset)
        from .backends import LocalHFBackend
        backend = LocalHFBackend(args.model, device=args.device, max_context=args.max_context)
        cancel = Event()
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda _signum, _frame: cancel.set())
        try:
            result = run_experiment(backend, candidates(args.family, mutate=args.mutate),
                                    split, boundary=args.boundary_ids,
                                    max_new_tokens=args.max_new_tokens,
                                    seed=args.seed, cancel=cancel)
        finally:
            signal.signal(signal.SIGINT, previous)
        save_report(result, args.output, include_evidence=args.include_evidence)
        print(json.dumps({"decision": result["decision"],
                          "selected_candidate": result["selected_candidate"]}))
        if result["decision"] == "cancelled":
            return 130
        if result["decision"] == "failed":
            return 1
        return 0 if result["decision"] == "compatible_on_fixture_set" else 2
    except KeyboardInterrupt:
        print("Interrupted before a complete report was saved.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError):
        print("Operation failed. Check local model support, input schema, dependencies and output permissions.",
              file=sys.stderr)
        return 1
