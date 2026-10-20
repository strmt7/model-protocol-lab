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
        "--transfer-model",
        type=Path,
        action="append",
        default=None,
        help="Additional local checkpoint for the cross-checkpoint transfer "
        "experiment (repeatable, max 4). After the primary run, every "
        "search candidate is re-evaluated against each transfer checkpoint "
        "on the search split, and the selection rule is re-applied per "
        "checkpoint to report whether the primary winner transfers. "
        "Multiplies compute; each checkpoint is recorded with its "
        "fingerprint.",
    )
    run.add_argument(
        "--perturb",
        type=int,
        default=0,
        help="Fixture-perturbation robustness experiment: re-run every "
        "search candidate on N deterministically perturbed fixture "
        "variants (1-8; typos, case flips, whitespace, Unicode variants, "
        "punctuation stripping) and re-apply the selection rule per "
        "variant, reporting whether the primary winner survives "
        "phrasing noise. Multiplies compute by (1 + N); the stage never "
        "fails the run.",
    )
    run.add_argument(
        "--token-healing",
        action="store_true",
        help="Token-healing selection sensitivity experiment: re-run every "
        "search candidate with HF-style token healing applied to each "
        "prompt (the tail token is re-chosen among vocabulary extensions "
        "of its text) and re-apply the selection rule, reporting whether "
        "the winner flips. The lab's own boundary-canonicality audit "
        "predicts which candidates are healing-sensitive, so the report "
        "includes the falsification table. Roughly doubles search "
        "compute; the stage never fails the run.",
    )
    run.add_argument(
        "--hierarchy",
        action="store_true",
        help="Instruction-hierarchy adherence probe: re-run every search "
        "candidate on synthetic system-vs-user conflict fixtures (the "
        "system instruction is delivered as a privilege-marked block "
        "folded into the user turn, before and after the user "
        "instruction) and re-apply the selection rule per position, "
        "reporting per-candidate system/user adherence rates, "
        "winner-vs-clean flips, and the documentation-derived "
        "dedicated-frame vs folded family comparison. Roughly triples "
        "search compute; the stage never fails the run.",
    )
    run.add_argument(
        "--prefill",
        action="store_true",
        help="Assistant-prefill sensitivity experiment: re-run every search "
        "candidate with benign format-lock prefills appended after the "
        "generation prompt (the model generates the continuation, scored "
        "as prefill + generated text) and re-apply the selection rule "
        "per prefill, reporting per-candidate honoring rates (does the "
        "model accept the prefill as its own opening or restart its "
        "turn?), winner-vs-native flips, and the think-channel group "
        "comparison. Roughly quadruples search compute; the stage never "
        "fails the run.",
    )
    run.add_argument(
        "--synth-history",
        action="store_true",
        help="Synthetic-history selection stability experiment: prepend "
        "fixed benign synthetic prior conversation turns (one and three "
        "exchanges) to every fixture and re-apply the selection rule per "
        "history condition, reporting per-candidate history pass rates, "
        "winner-vs-clean flips, and the delimiter-explicitness group "
        "comparison (families with unambiguous per-turn open/close markers "
        "vs loose delimiting). Roughly triples search compute; the stage "
        "never fails the run.",
    )
    run.add_argument(
        "--few-shot",
        action="store_true",
        help="Few-shot demonstration gain x template experiment: sample 2 "
        "demonstration fixtures from the search split (seeded; never "
        "held-out), evaluate every candidate 0-shot and with both demo "
        "orders on the held-out fixtures, and report per-candidate gain "
        "(zero-shot -> k-shot verdict-change rate), order sensitivity "
        "(per-order rate dispersion), and per-arm (baseline/order_a/order_b) "
        "winner-vs-primary flips. Roughly triples held-out compute; the "
        "stage never fails the run.",
    )
    run.add_argument(
        "--batch-invariance",
        action="store_true",
        help="Batch-invariance audit: re-run every search fixture's exact "
        "prompt token sequence through the backend's batch path (when one "
        "exists) and compare solo vs batched greedy continuation ids "
        "token-for-token, reporting the divergence rate and first-"
        "divergence positions. Backends without a batch path record "
        "batch_unsupported instead of failing; the stage never fails "
        "the run.",
    )
    run.add_argument(
        "--position",
        action="store_true",
        help="Position-sensitivity experiment: re-run every search "
        "candidate on synthetic multiple-choice fixtures whose correct "
        "option is permuted between the first, middle, and last positions "
        "(seeded distractor order) and re-apply the selection rule per "
        "condition, reporting per-candidate correct rates, the primacy "
        "index (P(correct | first) - P(correct | last)), winner-vs-clean "
        "flips, and the condition effect. Roughly triples search "
        "compute; the stage never fails the run.",
    )
    run.add_argument(
        "--stop-cause",
        action="store_true",
        help="Stop-cause termination audit: post-hoc classification of "
        "every search candidate's recorded generations into eos_id / "
        "stop_string / max_length / stop_unknown, reporting per-family "
        "termination-cause distributions and over-generation bleed "
        "past the first end-of-turn token. No new generations; the "
        "stage never fails the run.",
    )
    run.add_argument(
        "--ws-scan",
        action="store_true",
        help="Whitespace-canonicalization scan: re-encode every search "
        "candidate's rendered prompts under 8 whitespace mutations of "
        "the render string (trailing/leading space, newline, tab, "
        "stripping, collapsing), compare id sequences exactly, and "
        "re-run fixtures on divergent variants to measure verdict "
        "flips. Reports per-family whitespace fragility. Roughly 9x "
        "search compute; the stage never fails the run.",
    )
    run.add_argument(
        "--structured",
        action="store_true",
        help="Structured-output adherence probe: evaluate every search "
        "candidate on synthetic JSON fixtures (extraction / "
        "classification / routing shapes with fixed schemas) and "
        "measure prompt-only parse-valid and schema-valid rates per "
        "template -- the honest baseline no constrained-decoding paper "
        "publishes. No constrained decoder; roughly 1x search "
        "compute; the stage never fails the run.",
    )
    run.add_argument(
        "--deliforge",
        action="store_true",
        help="Delimiter-authority probe (research wave-4 idea 1): forge "
        "chat-template markers (e.g. <|im_start|>) into user content in "
        "two encodings of identical decoded text -- the family's native "
        "single-id (reserved) encoding vs forced subword tokens -- and "
        "measure which encoding flips task verdicts more often per "
        "template family, testing arXiv:2609.35932's reserved-token "
        "authority claim across all families. Roughly 16x search "
        "compute (2 payloads x 4 markers x 2 arms); the stage never "
        "fails the run.",
    )
    run.add_argument(
        "--fake-turn",
        action="store_true",
        help="Fake-turn history-tampering probe (research wave-4 idea 3): "
        "forge prior turns (fake assistant asserting a wrong answer, "
        "fake system contradicting the task) into user content at the "
        "family's native encoding, and measure the verdict-flip rate "
        "per template family with the forged delimiters wearing the "
        "family's own token set (same) vs another family's token set "
        "(cross), testing arXiv:2405.20234's token-set-dependent "
        "tampering claim across all families. Roughly 8x search "
        "compute (2 payloads x 4 markers); the stage never fails the run.",
    )
    run.add_argument(
        "--tool-call",
        action="store_true",
        help="Prompt-only tool-call syntax adherence probe (research "
        "wave-4 idea 4): describe a fixed 4-function tool list in prompt "
        "text and measure, per template family, correct function "
        "selection, typed argument extraction, and abstention "
        "(NO_CALL) when no tool fits -- the BFCL prompt-mode task with "
        "the chat template as the independent variable. 9 synthetic "
        "fixtures; the stage never fails the run.",
    )
    run.add_argument(
        "--verbosity",
        action="store_true",
        help="Response verbosity profile (research wave-4 idea 5): "
        "measure, per template family, the distribution of excess "
        "generated tokens over brevity-ideal fixtures (short stable "
        "answers) -- post-hoc over the main run's observations, zero "
        "extra model calls. Reports median/IQR excess per family and "
        "the budget-saturation share (capped generations are lower "
        "bounds on true verbosity). Behavior/cost profile only -- "
        "shorter is not better. The stage never fails the run.",
    )
    run.add_argument(
        "--cache-profile",
        action="store_true",
        help="Prefix-cache sharability profile (cycle 168, research wave-6 "
        "idea 2): per template family, measure the fraction of prompt tokens "
        "that exact-prefix KV-cache reuse (vLLM automatic prefix caching) "
        "could skip -- computed offline from the run's recorded input_ids, "
        "zero extra model calls, with block quantization for vLLM-style "
        "16-token blocks. Headline: exact achievable block fraction (cost "
        "input, not a cost verdict). Informational only -- never changes "
        "selection. The stage never fails the run.",
    )
    run.add_argument(
        "--cache-profile-block-size",
        type=int,
        default=16,
        help="Block size in tokens for --cache-profile block quantization "
        "(default 16, vLLM's block size). Recorded in the output.",
    )
    run.add_argument(
        "--confidentiality",
        action="store_true",
        help="Labeled-block confidentiality probe (cycle 169, research wave-6 "
        "idea 3): a fixed nonce marker is rendered inside the "
        "privilege-labeled block and, as a control, bare at the same "
        "position; frozen extraction elicitations then ask for the block "
        "quoted verbatim. Reports per template whether the label reduces "
        "verbatim exposure (a lower bound: paraphrase leakage is out of "
        "scope). Answers: 'does this template stop the model quoting my "
        "standing instructions back?' Informational only -- never changes "
        "selection. Costs 2 arms x 12 fixtures x C extra generations. "
        "The stage never fails the run.",
    )
    run.add_argument(
        "--answer-collision",
        action="store_true",
        help="Answer-collision audit (cycle 170, research wave-6 idea 4): "
        "the lab audits its OWN boundary detector. Synthetic expected "
        "answers embedding marker-like text are emitted verbatim through "
        "the evaluation path, and each is classified collision (the "
        "detector cut a correct answer and discarded content), "
        "ends-at-marker, or intact -- per boundary configuration, in a "
        "reserved-token arm and a subword control arm. Answers: 'could "
        "my boundary marker silently truncate correct answers that quote "
        "the marker?' Scope: configured boundaries and this tokenizer "
        "only; scripted emission measures the detector, never a model. "
        "Informational only -- never changes selection. The stage never "
        "fails the run.",
    )
    run.add_argument(
        "--template-robustness",
        action="store_true",
        help="Per-template surface-noise robustness (cycle 166): for each "
        "candidate template, measure the verdict preservation rate -- the "
        "fraction of its correct fixture verdicts that survive "
        "presentation-level surface noise (whitespace, case, unicode "
        "variants) applied to user message contents. Template is the "
        "independent variable; weights fixed. Informational only, never "
        "changes winner selection. Opt-in: costs about C*F*17 extra "
        "generations. The stage never fails the run.",
    )
    run.add_argument(
        "--ablate-fields",
        action="store_true",
        help="Template field ablation (cycle 167): for the top-2 candidates "
        "by search accuracy, replace one Format structural field at a time "
        "(bos, role prefixes, turn terminators, open_suffix) with its "
        "neutral value and re-evaluate, attributing verdict flips per "
        "field. Answers 'what about the winner won' -- only the causal "
        "field transfers to the next template you write. Main effects "
        "only; informational, never changes winner selection. Opt-in: "
        "costs about 2*(1+8) extra evaluations. The stage never fails "
        "the run.",
    )
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
    fixtures_fuzz = fixtures_sub.add_parser(
        "fuzz",
        help="Fuzz the fixture loaders against their published contract: "
        + "mutate known-good JSON and CSV fixtures from a fixed table and "
        "check every outcome (never loads a model)",
        allow_abbrev=False,
    )
    fixtures_fuzz.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Corpus seed (default 0); the same seed reproduces the same "
        "corpus and report byte-for-byte",
    )
    # ANCHOR: new top-level subparsers are registered here (one block per command).
    from .surfacenoise import build_surface_noise_parser

    build_surface_noise_parser(sub)
    from .verify import build_verify_parser

    build_verify_parser(sub)
    from .pack import build_pack_parser

    build_pack_parser(sub)
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
    "local_heal_failed",
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


def _run_hierarchy_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the instruction-hierarchy probe; never raises.

    Re-runs every search candidate on the synthetic system-vs-user
    conflict fixtures (both privilege-block positions) and stores the
    ``hierarchy`` analysis payload on ``result``. Any failure degrades
    to an "unavailable" marker instead of failing the run.
    """
    try:
        from .experiments import hierarchy_adherence, hierarchy_stage
        from .hierarchy import (
            CONFLICT_SPECS,
            POSITIONS,
            build_conflict_fixtures,
            validate_conflict_specs,
        )

        specs = validate_conflict_specs(CONFLICT_SPECS)
        stages: dict[str, Any] = {}
        for position in POSITIONS:
            fixtures, expectations = build_conflict_fixtures(specs, position=position)
            stages[position] = hierarchy_stage(
                backend,
                formats,
                fixtures,
                expectations,
                position=position,
                boundary=boundary,
                boundary_map=boundary_map,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        result["hierarchy"] = hierarchy_adherence(
            result.get("selected_candidate"),
            result["decision"],
            tuple(fmt.name for fmt in formats),
            stages,
        )
    except Exception:  # noqa: BLE001 -- hierarchy must not fail the run
        result["hierarchy"] = {
            "available": False,
            "detail": "hierarchy_unavailable",
        }


def _run_position_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the position-sensitivity experiment; never raises.

    Re-runs every search candidate on synthetic multiple-choice
    fixtures whose correct option is permuted between the first,
    middle, and last positions (seeded distractor order) and stores
    the ``position`` analysis payload on ``result``. Any failure
    degrades to an "unavailable" marker instead of failing the run.
    """
    try:
        from .experiments import position_sensitivity, position_stage
        from .position import (
            CONDITIONS,
            POSITION_SPECS,
            build_position_fixtures,
            validate_position_specs,
        )

        specs = validate_position_specs(POSITION_SPECS)
        stages: dict[str, Any] = {}
        for condition in CONDITIONS:
            fixtures = build_position_fixtures(specs, condition=condition, seed=seed)
            stages[condition] = position_stage(
                backend,
                formats,
                fixtures,
                condition=condition,
                boundary=boundary,
                boundary_map=boundary_map,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        result["position"] = position_sensitivity(
            result.get("selected_candidate"),
            result["decision"],
            tuple(fmt.name for fmt in formats),
            stages,
        )
    except Exception:  # noqa: BLE001 -- position probe must not fail the run
        result["position"] = {
            "available": False,
            "detail": "position_unavailable",
        }


def _run_stop_probe(
    backend: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    boundary: Any,
    boundary_map: Any,
    eos_id: Any,
    token_bytes: Any = None,
) -> None:
    """Run the stop-cause termination audit; never raises.

    Post-hoc classification (no new generations) of every search
    candidate's recorded generations into eos_id / stop_string /
    max_length / stop_unknown, stored as the ``stop_cause`` analysis
    payload on ``result``. Any failure degrades to an "unavailable"
    marker instead of failing the run.
    """
    try:
        from .experiments import stop_stage

        result["stop_cause"] = stop_stage(
            result,
            tuple(cases),
            boundary=boundary,
            boundary_map=boundary_map,
            eos_id=eos_id,
            backend=backend,
            token_bytes=token_bytes,
        )
    except Exception:  # noqa: BLE001 -- stop-cause probe must not fail the run
        result["stop_cause"] = {
            "available": False,
            "detail": "stop_unavailable",
        }


def _run_ws_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the whitespace-canonicalization scan; never raises.

    Re-encodes every search candidate's rendered prompts under the 8
    whitespace mutations of ``wscanonical.MUTATIONS``, compares id
    sequences exactly, re-runs fixtures on divergent variants through
    the normal evaluate pipeline, and stores the ``ws_canonical``
    analysis payload on ``result``. Any failure degrades to an
    "unavailable" marker instead of failing the run.
    """
    try:
        from .experiments import (
            native_search_input_ids,
            native_search_outcomes,
            ws_canonicalization,
            ws_stage,
        )
        from .wscanonical import MUTATIONS

        candidates = tuple(fmt.name for fmt in formats)
        stage = ws_stage(
            backend,
            formats,
            tuple(cases),
            mutations=MUTATIONS,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        result["ws_canonical"] = ws_canonicalization(
            native_search_outcomes(result.get("search") or [], candidates),
            native_search_input_ids(result.get("search") or [], candidates),
            stage,
            candidates,
        )
    except Exception:  # noqa: BLE001 -- ws-scan probe must not fail the run
        result["ws_canonical"] = {
            "available": False,
            "detail": "ws_scan_unavailable",
        }


def _run_structured_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the structured-output adherence probe; never raises.

    Evaluates every search candidate on the synthetic JSON fixtures
    from ``structured.STRUCTURED_SPECS`` and stores the ``structured``
    analysis payload on ``result``. Any failure degrades to an
    "unavailable" marker instead of failing the run.
    """
    try:
        from .experiments import structured_adherence, structured_stage
        from .structured import (
            STRUCTURED_SPECS,
            build_structured_fixtures,
            schema_map,
            validate_structured_specs,
        )

        specs = validate_structured_specs(STRUCTURED_SPECS)
        fixtures = build_structured_fixtures(specs)
        schemas = schema_map(specs)
        stage = structured_stage(
            backend,
            formats,
            fixtures,
            schemas=schemas,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        result["structured"] = structured_adherence(
            tuple(fmt.name for fmt in formats),
            stage,
        )
    except Exception:  # noqa: BLE001 -- structured probe must not fail the run
        result["structured"] = {
            "available": False,
            "detail": "structured_unavailable",
        }


def _run_deliforge_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the delimiter-authority probe; never raises.

    Forges chat-template markers into the search fixtures' user
    content and evaluates every candidate with the forged markers in
    two encodings of identical decoded text (reserved single-id vs
    forced subword), storing the ``deliforge`` authority analysis on
    ``result``. Any failure degrades to an "unavailable" marker
    instead of failing the run.
    """
    try:
        from .experiments import deliforge_authority, deliforge_stage, native_search_outcomes

        candidates = tuple(fmt.name for fmt in formats)
        stage = deliforge_stage(
            backend,
            formats,
            tuple(cases),
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        result["deliforge"] = deliforge_authority(
            native_search_outcomes(result.get("search") or [], candidates),
            stage,
            candidates,
        )
    except Exception:  # noqa: BLE001 -- deliforge probe must not fail the run
        result["deliforge"] = {
            "available": False,
            "detail": "deliforge_unavailable",
        }


def _run_faketurn_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the fake-turn history-tampering probe; never raises.

    Forges prior turns (fake assistant asserting a wrong answer, fake
    system contradicting the task) into the search fixtures' user
    content at each family's native encoding, classifies each forged
    marker's token provenance (same-family vs cross-family
    delimiters), and stores the ``faketurn`` robustness analysis on
    ``result``. Any failure degrades to an "unavailable" marker
    instead of failing the run.
    """
    try:
        from .experiments import faketurn_robustness, faketurn_stage, native_search_outcomes

        candidates = tuple(fmt.name for fmt in formats)
        stage = faketurn_stage(
            backend,
            formats,
            tuple(cases),
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        result["faketurn"] = faketurn_robustness(
            native_search_outcomes(result.get("search") or [], candidates),
            stage,
            candidates,
        )
    except Exception:  # noqa: BLE001 -- faketurn probe must not fail the run
        result["faketurn"] = {
            "available": False,
            "detail": "faketurn_unavailable",
        }


def _run_toolcall_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the prompt-only tool-call adherence probe; never raises.

    Describes the fixed 4-function tool list in prompt text (no native
    function-calling APIs), evaluates each candidate on the 9
    synthetic tool-call fixtures, classifies each decoded response
    (selection / args / relevance) and stores the ``toolcall``
    adherence analysis on ``result``. Any failure degrades to an
    "unavailable" marker instead of failing the run.
    """
    try:
        from .experiments import toolcall_adherence, toolcall_stage
        from .toolcall import (
            TOOLCALL_QUERIES,
            build_toolcall_fixtures,
            query_calls,
            validate_toolcall_queries,
        )

        queries = validate_toolcall_queries(TOOLCALL_QUERIES)
        candidates = tuple(fmt.name for fmt in formats)
        # Floor the token budget: the longest canonical call is 79
        # characters, and real tokenizers may split JSON punctuation,
        # so the probe must not let an unrelated budget parameter
        # truncate calls into parse_failed (that would confound the
        # measurement with the budget).
        stage = toolcall_stage(
            backend,
            formats,
            build_toolcall_fixtures(queries),
            calls=query_calls(queries),
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max(128, max_new_tokens),
            seed=seed,
        )
        result["toolcall"] = toolcall_adherence(candidates, stage)
    except Exception:  # noqa: BLE001 -- toolcall probe must not fail the run
        result["toolcall"] = {
            "available": False,
            "detail": "toolcall_unavailable",
        }


def _run_verbosity_probe(
    result: Any,
    formats: Any,
    *,
    backend: Any,
    search_fixtures: Any,
    max_new_tokens: int,
) -> None:
    """Run the response-verbosity profile probe; never raises.

    Post-hoc over the main run's observations (zero extra model
    calls): for every brevity-ideal search fixture, the generated
    token count (``len(output_ids)``) is measured against the exact
    tokenization of the expected answer, and the per-candidate excess
    distributions plus budget-saturation shares are stored as the
    ``verbosity`` profile on ``result``. Any failure degrades to an
    "unavailable" marker instead of failing the run.
    """
    try:
        from .experiments import verbosity_profile, verbosity_stage

        entries = result.get("search")
        if type(entries) is not list:
            raise ValueError("result has no search entries")
        observations: list[dict[str, Any]] = []
        for entry in entries:
            if type(entry) is not dict:
                continue
            entry_obs = entry.get("observations")
            if type(entry_obs) is not list:
                continue
            observations.extend(o for o in entry_obs if type(o) is dict)
        expected_by_fixture = {f.id: f.expected for f in search_fixtures}
        candidates = tuple(fmt.name for fmt in formats)
        stage = verbosity_stage(
            observations=observations,
            expected_by_fixture=expected_by_fixture,
            backend=backend,
            max_new_tokens=max_new_tokens,
        )
        result["verbosity"] = verbosity_profile(candidates, stage)
    except Exception:  # noqa: BLE001 -- verbosity probe must not fail the run
        result["verbosity"] = {
            "available": False,
            "detail": "verbosity_unavailable",
        }


def _run_cache_profile_probe(
    result: Any,
    formats: Any,
    *,
    block_size: int = 16,
) -> None:
    """Run the prefix-cache sharability probe; never raises.

    Post-hoc over the main run's observations (zero extra model
    calls): for every completed observation, the recorded ``input_ids``
    are grouped by candidate and aggregated by
    :func:`cacheprofile.candidate_cache_profile` (exact achievable
    sharability fractions plus the potential upper bound,
    block-quantized for vLLM-style prefix caching). The analysis is
    stored as ``result["cache_profile"]``. Informational only -- never
    changes selection. Any failure degrades to an "unavailable" marker
    instead of failing the run.
    """
    try:
        from .experiments import cache_profile_profile, cache_profile_stage

        entries = result.get("search")
        if type(entries) is not list:
            raise ValueError("result has no search entries")
        observations: list[dict[str, Any]] = []
        for entry in entries:
            if type(entry) is not dict:
                continue
            entry_obs = entry.get("observations")
            if type(entry_obs) is not list:
                continue
            observations.extend(o for o in entry_obs if type(o) is dict)
        candidates = tuple(fmt.name for fmt in formats)
        stage = cache_profile_stage(
            observations=observations,
            block_size=block_size,
        )
        result["cache_profile"] = cache_profile_profile(candidates, stage)
    except Exception:  # noqa: BLE001 -- cache-profile probe must not fail the run
        result["cache_profile"] = {
            "available": False,
            "detail": "cache_profile_unavailable",
        }


def _run_confidentiality_probe(
    result: Any,
    formats: Any,
    *,
    backend: Any,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the labeled-block confidentiality probe; never raises.

    Per candidate, the synthetic confidentiality fixtures are evaluated
    in both arms (privilege-labeled block vs bare marker, same position,
    mirrored extraction elicitations) and detection runs on the full
    decoded output. Results are stored as ``result["confidentiality"]``;
    informational only, never changes selection. Any failure degrades to
    an "unavailable" marker instead of failing the run.
    """
    try:
        from .confidentiality import (
            CONFIDENTIALITY_SPECS,
            build_arm_fixtures,
            validate_confidentiality_specs,
        )
        from .experiments import confidentiality_profile, confidentiality_stage

        specs = validate_confidentiality_specs(CONFIDENTIALITY_SPECS)
        stage = confidentiality_stage(
            backend,
            tuple(formats),
            build_arm_fixtures("treatment", specs),
            build_arm_fixtures("control", specs),
            boundary=tuple(boundary),
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        candidates = tuple(fmt.name for fmt in formats)
        result["confidentiality"] = confidentiality_profile(candidates, stage)
    except Exception:  # noqa: BLE001 -- confidentiality probe must not fail the run
        result["confidentiality"] = {
            "available": False,
            "detail": "confidentiality_unavailable",
        }


def _run_answer_collision_probe(
    result: Any,
    formats: Any,
    *,
    backend: Any,
    boundary: Any,
    boundary_map: Any,
    seed: int,
) -> None:
    """Run the answer-collision audit; never raises.

    Per candidate boundary configuration, synthetic marker-bearing
    expected answers are emitted verbatim (scripted backends wrapping
    the run backend's encode/decode) through the evaluation path, and
    the detector's effect on each is classified collision /
    ends-at-marker / intact. Results are stored as
    ``result["answer_collision"]``; informational only, never changes
    selection. Any failure degrades to an "unavailable" marker instead
    of failing the run.
    """
    try:
        from .experiments import answer_collision_profile, answer_collision_stage

        stage = answer_collision_stage(
            backend,
            tuple(formats),
            boundary=tuple(boundary),
            boundary_map=boundary_map,
            seed=seed,
        )
        candidates = tuple(fmt.name for fmt in formats)
        result["answer_collision"] = answer_collision_profile(candidates, stage)
    except Exception:  # noqa: BLE001 -- answer-collision audit must not fail the run
        result["answer_collision"] = {
            "available": False,
            "detail": "answer_collision_unavailable",
        }


def _run_template_robustness_probe(
    result: Any,
    formats: Any,
    *,
    backend: Any,
    search_fixtures: Any,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the per-template surface-noise robustness stage; never raises.

    For each candidate template, probes verdict preservation under
    presentation-level surface noise on user message contents (cycle 166).
    Each candidate is judged under its own boundary (``boundary_map``), the
    same honest-boundary rule as the main evaluation. Informational only:
    results are stored as ``result["template_robustness"]`` and never change
    winner selection. Any failure degrades to an "unavailable" marker instead
    of failing the run.
    """
    try:
        from .experiments import template_robustness_stage

        stage = template_robustness_stage(
            tuple(formats),
            tuple(search_fixtures),
            backend,
            boundary=tuple(boundary),
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        result["template_robustness"] = {
            "available": True,
            "candidates": stage["candidates"],
            "protocol_fingerprint": stage["protocol"].get("template_robustness_code_sha256"),
        }
    except Exception:  # noqa: BLE001 -- robustness probe must not fail the run
        result["template_robustness"] = {
            "available": False,
            "detail": "template_robustness_unavailable",
        }


def _run_ablation_probe(
    result: Any,
    formats: Any,
    *,
    backend: Any,
    search_fixtures: Any,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the template field ablation stage; never raises.

    For the top-2 candidates by search accuracy, evaluates one
    single-field ablation arm per structural Format field (cycle 167).
    Each candidate is judged under its own boundary (``boundary_map``), the
    same honest-boundary rule as the main evaluation. Informational only:
    results are stored as ``result["ablation"]`` and never change winner
    selection. Any failure degrades to an "unavailable" marker instead of
    failing the run.
    """
    try:
        from .experiments import ablation_stage

        stage = ablation_stage(
            tuple(formats),
            tuple(search_fixtures),
            backend,
            boundary=tuple(boundary),
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        result["ablation"] = {
            "available": True,
            "candidates": stage["candidates"],
            "n_focus": stage["n_focus"],
        }
    except Exception:  # noqa: BLE001 -- ablation probe must not fail the run
        result["ablation"] = {
            "available": False,
            "detail": "ablation_unavailable",
        }


def _run_prefill_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the assistant-prefill sensitivity experiment; never raises.

    Re-runs every search candidate with each benign format-lock prefill
    appended after the generation prompt and stores the ``prefill``
    analysis payload on ``result``. Any failure degrades to an
    "unavailable" marker instead of failing the run.
    """
    try:
        from .experiments import prefill_adherence, prefill_stage
        from .prefill import PREFILL_SPECS, validate_prefill_specs

        specs = validate_prefill_specs(PREFILL_SPECS)
        stages: dict[str, Any] = {}
        for spec in specs:
            stages[spec["id"]] = prefill_stage(
                backend,
                formats,
                cases,
                prefill=spec,
                boundary=boundary,
                boundary_map=boundary_map,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        result["prefill"] = prefill_adherence(
            result.get("selected_candidate"),
            result["decision"],
            tuple(fmt.name for fmt in formats),
            stages,
        )
    except Exception:  # noqa: BLE001 -- prefill must not fail the run
        result["prefill"] = {
            "available": False,
            "detail": "prefill_unavailable",
        }


def _run_synthhistory_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the synthetic-history stability experiment; never raises.

    Prepends the fixed benign synthetic prior turns (one and three
    exchanges) to every search fixture, re-runs every search candidate
    per history condition, and stores the ``synthhistory`` analysis
    payload on ``result``. Any failure degrades to an "unavailable"
    marker instead of failing the run.
    """
    try:
        from .experiments import synthhistory_stability, synthhistory_stage
        from .synthhistory import CONDITIONS, build_history_fixtures

        stages: dict[str, Any] = {}
        for condition in CONDITIONS:
            fixtures, _n_turns = build_history_fixtures(cases, condition=condition)
            stages[condition] = synthhistory_stage(
                backend,
                formats,
                fixtures,
                condition=condition,
                boundary=boundary,
                boundary_map=boundary_map,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        result["synthhistory"] = synthhistory_stability(
            result.get("selected_candidate"),
            result["decision"],
            tuple(fmt.name for fmt in formats),
            stages,
        )
    except Exception:  # noqa: BLE001 -- synthhistory must not fail the run
        result["synthhistory"] = {
            "available": False,
            "detail": "synthhistory_unavailable",
        }


def _run_fewshot_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    demo_cases: Any,
    eval_cases: Any,
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the few-shot x template experiment; never raises.

    Samples 2 demonstration fixtures from the search split (seeded),
    re-runs every candidate 0-shot and with both demo orders on the
    held-out fixtures, and stores the ``fewshot`` analysis payload on
    ``result``. Any failure degrades to an "unavailable" marker instead
    of failing the run.
    """
    try:
        from .experiments import fewshot_gain, fewshot_stage

        stage = fewshot_stage(
            backend,
            formats,
            tuple(demo_cases),
            tuple(eval_cases),
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        result["fewshot"] = fewshot_gain(
            result.get("selected_candidate"),
            result["decision"],
            tuple(fmt.name for fmt in formats),
            stage,
        )
    except Exception:  # noqa: BLE001 -- fewshot must not fail the run
        result["fewshot"] = {
            "available": False,
            "detail": "fewshot_unavailable",
        }


def _encode_probe_prompt(backend: Any, reference: Any, case: Any) -> tuple[int, ...] | None:
    """Encode one probe fixture's prompt; None when the fixture degrades.

    A fixture that cannot be rendered or encoded is skipped by the
    batch-invariance probe instead of failing the run.
    """
    try:
        encoded: tuple[int, ...] | None = backend.encode(reference.render(case.messages))
    except Exception:  # noqa: BLE001 -- one fixture degrades, not the probe
        return None
    return encoded


def _run_batch_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Run the batch-invariance audit; never raises.

    Encodes every search fixture with the first search candidate's
    rendering (the exact token sequences the lab actually sends) and
    runs :func:`experiments.batch_stage`: solo vs batched greedy
    decoding compared token-for-token. A backend without a batch path
    records ``batch_unsupported``; any failure degrades to an
    "unavailable" marker instead of failing the run.
    """
    try:
        from .experiments import batch_stage

        reference = formats[0]
        prompts: dict[str, tuple[int, ...]] = {}
        for case in cases:
            encoded = _encode_probe_prompt(backend, reference, case)
            if encoded is not None:
                prompts[case.id] = encoded
        if not prompts:
            result["batch_invariance"] = {
                "available": False,
                "detail": "no_prompts",
            }
            return
        result["batch_invariance"] = batch_stage(
            backend,
            prompts,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
    except Exception:  # noqa: BLE001 -- batch-invariance must not fail the run
        result["batch_invariance"] = {
            "available": False,
            "detail": "batch_unavailable",
        }


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
        if args.command == "surface-noise":
            from .surfacenoise import run_surface_noise

            return run_surface_noise(args, parser)
        if args.command == "verify":
            from .verify import run_verify

            return run_verify(args, parser)
        if args.command == "pack":
            from .pack import run_pack

            return run_pack(args, parser)
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
            if args.fixtures_command == "fuzz":
                from .fixturefuzz import run_fixtures_fuzz

                return run_fixtures_fuzz(args, parser)
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
                # Oracle-mixture / protocol-portfolio gap (research
                # wave-2 idea 4): pure post-hoc over stored search
                # observations, no model calls. Answers "how much does
                # the single global winner leave on the table versus a
                # per-fixture oracle, and which small protocol portfolio
                # covers every solvable fixture?" Degrades to an
                # "unavailable" marker instead of failing the run.
                try:
                    from .experiments import portfolio_analysis

                    result["portfolio"] = portfolio_analysis(result, tuple(split.search))
                except Exception:  # noqa: BLE001 -- diagnostics must not fail the run
                    result["portfolio"] = {
                        "available": False,
                        "detail": "portfolio_unavailable",
                    }
                # Cross-checkpoint transfer experiment (IDEAS #1): re-evaluate
                # every search candidate against each --transfer-model
                # checkpoint on the search split, then re-apply the
                # selection rule per checkpoint. Answers "does the winner
                # transfer to my other checkpoints?" Capped at 4
                # checkpoints; a failed checkpoint degrades to
                # checkpoint_incompatible, never silently skipped, and
                # never fails the run.
                if args.transfer_model:
                    try:
                        from .experiments import transfer_portability, transfer_stage

                        _transfer_paths = list(args.transfer_model)
                        if len(_transfer_paths) > 4:
                            raise ValueError("at most 4 --transfer-model checkpoints")
                        _transfer_ids = [p.name for p in _transfer_paths]
                        if len(set(_transfer_ids)) != len(_transfer_ids):
                            raise ValueError("transfer checkpoint names must be unique")
                        from .backends import LocalHFBackend

                        _transfer_backends: dict[str, Any] = {}
                        _transfer_fingerprints: dict[str, str | None] = {}
                        for _cid, _path in zip(_transfer_ids, _transfer_paths, strict=True):
                            try:
                                _be = LocalHFBackend(
                                    _path,
                                    device=args.device,
                                    max_context=args.max_context,
                                )
                                _transfer_backends[_cid] = _be
                                _fp = getattr(_be, "fingerprint", None)
                                _transfer_fingerprints[_cid] = _fp if type(_fp) is str else None
                            except Exception:  # noqa: BLE001 -- per-checkpoint fail-closed
                                _transfer_fingerprints[_cid] = None

                        def _transfer_backend(ckpt_id: str) -> Any:
                            if ckpt_id not in _transfer_backends:
                                raise RuntimeError(f"transfer checkpoint unavailable: {ckpt_id}")
                            return _transfer_backends[ckpt_id]

                        _transfer_outcomes = transfer_stage(
                            _transfer_backend,
                            tuple(_transfer_ids),
                            formats,
                            tuple(split.search),
                            boundary=args.boundary_ids,
                            boundary_map=boundary_map,
                            max_new_tokens=args.max_new_tokens,
                            seed=args.seed,
                        )
                        _candidate_names = tuple(fmt.name for fmt in formats)
                        _transfer_result = transfer_portability(
                            result.get("selected_candidate"),
                            result["decision"],
                            _candidate_names,
                            _transfer_outcomes,
                        )
                        _transfer_result["checkpoints"] = [
                            {
                                "id": cid,
                                "path": str(p),
                                "fingerprint": _transfer_fingerprints.get(cid),
                            }
                            for cid, p in zip(_transfer_ids, _transfer_paths, strict=True)
                        ]
                        result["transfer"] = _transfer_result
                    except Exception:  # noqa: BLE001 -- transfer must not fail the run
                        result["transfer"] = {
                            "available": False,
                            "detail": "transfer_unavailable",
                        }
                    # Fixture-perturbation robustness experiment (cycle 144):
                    # re-evaluate every search candidate against the primary
                    # checkpoint on N deterministically perturbed fixture
                    # variants (search split only), then re-apply the
                    # selection rule per variant. Answers "does the winner
                    # survive fixture phrasing perturbations, or did it win
                    # on phrasing luck?" A failed variant degrades to
                    # "unavailable", never a phantom verdict, and never
                    # fails the run.
                    if args.perturb:
                        try:
                            from .experiments import perturbation_robustness, perturbation_stage
                            from .perturbation import PERTURBATION_OPERATORS

                            _perturb_n = args.perturb
                            if type(_perturb_n) is not int or not 1 <= _perturb_n <= 8:
                                raise ValueError("--perturb must be an integer between 1 and 8")
                            _perturb_stage = perturbation_stage(
                                backend,
                                formats,
                                tuple(split.search),
                                n_variants=_perturb_n,
                                operators=PERTURBATION_OPERATORS,
                                seed=args.seed,
                                boundary=args.boundary_ids,
                                boundary_map=boundary_map,
                                max_new_tokens=args.max_new_tokens,
                            )
                            result["perturbation"] = perturbation_robustness(
                                result.get("selected_candidate"),
                                result["decision"],
                                tuple(fmt.name for fmt in formats),
                                _perturb_stage,
                            )
                        except Exception:  # noqa: BLE001 -- perturbation must not fail the run
                            result["perturbation"] = {
                                "available": False,
                                "detail": "perturbation_unavailable",
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
                # Token-healing selection sensitivity experiment (research
                # wave-2 idea 3): re-run every search candidate with
                # HF-style token healing applied to each prompt, then
                # re-apply the selection rule to the healed verdicts.
                # Answers "does the winning protocol change under token
                # healing?" The lab's own boundary-canonicality audit
                # (above) predicts which candidates are healing-sensitive,
                # so the falsification table tests that prediction --
                # a negative result is as informative as a positive one.
                # Healing failures degrade to native prompts per
                # observation (counted, never silent) and never fail
                # the run.
                if args.token_healing:
                    try:
                        from .experiments import (
                            healing_delta,
                            healing_stage,
                            native_search_outcomes,
                        )

                        _candidate_names = tuple(fmt.name for fmt in formats)
                        _healed_stage = healing_stage(
                            backend,
                            formats,
                            tuple(split.search),
                            boundary=args.boundary_ids,
                            boundary_map=boundary_map,
                            max_new_tokens=args.max_new_tokens,
                            seed=args.seed,
                        )
                        _native_outcomes = native_search_outcomes(
                            result.get("search") or [], _candidate_names
                        )
                        result["token_healing"] = healing_delta(
                            result.get("selected_candidate"),
                            result["decision"],
                            _candidate_names,
                            _native_outcomes,
                            _healed_stage,
                            result.get("boundary_audit"),
                        )
                    except Exception:  # noqa: BLE001 -- healing must not fail the run
                        result["token_healing"] = {
                            "available": False,
                            "detail": "token_healing_unavailable",
                        }
                # Instruction-hierarchy adherence probe (research wave-2
                # idea 1): re-run every search candidate on synthetic
                # system-vs-user conflict fixtures. The lab never sends
                # system messages (core.Message is user/assistant only),
                # so the system instruction is delivered as a
                # privilege-marked block folded into the user turn, in
                # two positions (before/after the user instruction);
                # the selection rule is re-applied per position.
                # Answers "which instruction does each template lead the
                # model to follow, and does the winner change under
                # conflict?" A failed position degrades to "unavailable"
                # inside the analysis, never a phantom verdict, and never
                # fails the run.
                if args.hierarchy:
                    _run_hierarchy_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                # Assistant-prefill sensitivity experiment (research
                # wave-2 idea 2): re-run every search candidate with
                # benign format-lock prefills appended after the
                # generation prompt, then re-apply the selection rule
                # per prefill. Answers "does the winning protocol
                # survive prefill, and which families honor vs ignore
                # it?" A failed prefill degrades to "unavailable"
                # inside the analysis, never a phantom verdict, and
                # never fails the run.
                if args.prefill:
                    _run_prefill_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                # Synthetic-history selection stability experiment
                # (research wave-2 idea 5): prepend fixed benign synthetic
                # prior turns (one and three exchanges) to every fixture
                # and re-apply the selection rule per history condition.
                # Answers "does the winning protocol survive conversational
                # context, and do explicitly-delimited templates degrade
                # less?" A failed condition degrades to "unavailable"
                # inside the analysis, never a phantom verdict, and
                # never fails the run.
                if args.synth_history:
                    _run_synthhistory_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                # Few-shot demonstration gain x template experiment
                # (research wave-4 idea 2): sample 2 demo fixtures from
                # the search split (seeded; never held-out), evaluate
                # every candidate 0-shot and with both demo orders on
                # the held-out fixtures, then report per-candidate gain
                # (0-shot -> k-shot verdict-change rate), order
                # sensitivity, and per-arm winner-vs-primary flips.
                # Answers "does in-context learning work the same under
                # every template, or does the template determine whether
                # demos help?" Demos are context, never the answer; a
                # failed stage degrades to "unavailable" and never fails
                # the run.
                if args.few_shot:
                    _run_fewshot_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        tuple(split.held_out),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                # Batch-invariance audit (opt-in: one solo generation per
                # search fixture plus one batched call when the backend
                # exposes a batch path). Re-runs every search fixture's
                # exact prompt token sequence solo vs co-batched and
                # compares continuation ids token-for-token -- the audit
                # of the lab's "exact token sequences" invariant against
                # batch-composition nondeterminism. A backend without a
                # batch path records batch_unsupported; the probe never
                # fails the run.
                if args.batch_invariance:
                    _run_batch_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                # Position-sensitivity experiment (research wave-3 idea 1):
                # re-run every search candidate on synthetic multiple-choice
                # fixtures whose correct option is permuted between the
                # first, middle, and last positions (seeded distractor
                # order) and re-apply the selection rule per condition.
                # Answers "does the winner survive option repositioning,
                # and do templates differ in primacy bias?" A failed run
                # degrades to "unavailable" inside the analysis, never a
                # phantom verdict, and never fails the run.
                if args.position:
                    _run_position_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                # Stop-cause termination audit (research wave-3 idea 3):
                # post-hoc classification of every search candidate's
                # recorded generations into eos_id / stop_string /
                # max_length / stop_unknown, reporting per-family
                # termination-cause distributions and over-generation
                # bleed. No new generations; a stop-aware engine's
                # would-have-fired mechanism is classified from the
                # recorded token ids. Never fails the run.
                if args.stop_cause:
                    _run_stop_probe(
                        backend,
                        result,
                        tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        eos_id=eos_id,
                        token_bytes=token_bytes,
                    )
                # Whitespace-canonicalization scan (research wave-3
                # idea 4): re-encode every search candidate's rendered
                # prompts under whitespace mutations of the render
                # string, compare id sequences exactly, and re-run
                # fixtures on divergent variants to measure verdict
                # flips. Never fails the run.
                if args.ws_scan:
                    _run_ws_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                # Structured-output adherence probe (research wave-3
                # idea 5): evaluate every search candidate on synthetic
                # JSON fixtures and measure prompt-only parse/schema
                # adherence per template. Never fails the run.
                if args.structured:
                    _run_structured_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                # Delimiter-authority probe (research wave-4 idea 1):
                # forged chat-template markers in reserved vs subword
                # encoding of identical text, per template family.
                # Never fails the run.
                if args.deliforge:
                    _run_deliforge_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                # Fake-turn history-tampering probe (research wave-4
                # idea 3): forged prior turns in user content, encoded
                # at the family's native form, with the forged
                # delimiters' provenance (same-family vs cross-family
                # token set) as the independent variable. Never fails
                # the run.
                if args.fake_turn:
                    _run_faketurn_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                # Prompt-only tool-call adherence probe (research wave-4
                # idea 4): fixed 4-function tool list described in
                # prompt text; selection / typed args / NO_CALL
                # abstention measured per family. Never fails the run.
                if args.tool_call:
                    _run_toolcall_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                # Response verbosity profile (research wave-4 idea 5):
                # per-family excess-token distribution on brevity-ideal
                # fixtures, post-hoc over the main run's observations
                # (zero extra model calls). Behavior/cost profile only.
                # Never fails the run.
                if args.verbosity:
                    _run_verbosity_probe(
                        result,
                        formats,
                        backend=backend,
                        search_fixtures=tuple(split.search),
                        max_new_tokens=args.max_new_tokens,
                    )
                # Prefix-cache sharability profile (cycle 168, research
                # wave-6 idea 2): per-family exact-prefix sharability of
                # the recorded input_ids, post-hoc over the main run's
                # observations (zero extra model calls). Informational
                # only -- never changes selection. Never fails the run.
                if args.cache_profile:
                    _run_cache_profile_probe(
                        result,
                        formats,
                        block_size=args.cache_profile_block_size,
                    )
                # Labeled-block confidentiality (cycle 169, research
                # wave-6 idea 3): verbatim exposure of a fixed nonce in a
                # privilege-labeled block vs the bare marker, under
                # mirrored extraction elicitations. Informational only --
                # never changes selection. Never fails the run.
                if args.confidentiality:
                    _run_confidentiality_probe(
                        result,
                        formats,
                        backend=backend,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                # Answer-collision audit (cycle 170, research wave-6
                # idea 4): the lab's own boundary detector vs verbatim
                # marker-bearing expected answers, per boundary
                # configuration. Informational only -- never changes
                # selection. Never fails the run.
                if args.answer_collision:
                    _run_answer_collision_probe(
                        result,
                        formats,
                        backend=backend,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        seed=args.seed,
                    )
                # Per-template surface-noise robustness (cycle 166):
                # verdict preservation rate per candidate under
                # presentation-level noise on user content. Informational
                # only -- never changes selection. Never fails the run.
                if args.template_robustness:
                    _run_template_robustness_probe(
                        result,
                        formats,
                        backend=backend,
                        search_fixtures=tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                # Template field ablation (cycle 167): per-field causal
                # attribution for the top-2 candidates. Informational
                # only -- never changes selection. Never fails the run.
                if args.ablate_fields:
                    _run_ablation_probe(
                        result,
                        formats,
                        backend=backend,
                        search_fixtures=tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
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
