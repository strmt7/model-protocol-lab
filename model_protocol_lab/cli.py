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
    stderr warnings, and apply the R4-7 200-character preview cap so
    a hostile ID cannot flood the output either (oversight security
    S-9)."""
    cleaned = _CONTROL_CHARS.sub("", value)
    if len(cleaned) > 200:
        return cleaned[:200] + "..."
    return cleaned


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
    doctor_cmd.add_argument(
        "--template-probe",
        action="store_true",
        default=False,
        help="Behavioural probe of the checkpoint's chat template: renders "
        "frozen sentinel-bracketed probe conversations in word-swap pairs "
        "through the checkpoint's own tokenizer and reports per-pair "
        "facts about content-conditional template logic the static "
        "firewall cannot see (an injected instruction that only appears "
        "for some user messages). Monday-morning use: run it on a "
        "downloaded checkpoint before trusting its template. Findings "
        "only; never changes the doctor verdict (see docs/USAGE.md "
        "section 2).",
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
        "--refusals",
        action="store_true",
        help="Suspected-refusal profile (cycle 174, research wave-6 "
        "idea 6): classify each candidate's recorded search answers "
        "against a frozen, published refusal-opener list (XSTest "
        "Appendix D core plus itemised supplements) and report the "
        "per-template suspected-refusal rate on benign fixtures -- "
        "post-hoc over the main run's observations, zero extra model "
        "calls. Monday morning: 'does this template make the model "
        "refuse questions it would otherwise answer?' Detector is "
        "heuristic and beginning-anchored; hits name the fixture and "
        "the opener, never the answer text. Informational only; "
        "never touches selection. The stage never fails the run.",
    )
    run.add_argument(
        "--scorer-lens",
        action="store_true",
        help="Scorer-lens regrade (cycle 180, research wave-7 rank 1): "
        "regrade each candidate's recorded search answers under a "
        "frozen normalisation ladder (exact; whitespace; case/Unicode; "
        "edge punctuation and markdown fences; numeric value) and "
        "report how many failures each rung recovers, which fixture "
        "flips first at which rung, and whether the search-phase "
        "winner changes under a more forgiving scorer -- post-hoc "
        "over the main run's observations, zero extra model calls. "
        "Monday morning: 'is this template actually worse, or is the "
        "byte-exact scorer discarding answers a human would accept?' "
        "Counts and fixture IDs only; no answer text. Informational "
        "only; never touches selection. The stage never fails the run.",
    )
    run.add_argument(
        "--aliasing",
        action="store_true",
        help="Candidate aliasing audit (cycle 181, research wave-7 "
        "rank 2): group the run's candidates into token-identity "
        "equivalence classes over the recorded search input_ids -- "
        "render classes (same prompt token sequences) and measurement "
        "classes (same prompts and same response boundary) -- and "
        "report effective vs recorded candidate counts, redundant "
        "completed observations, per-pair coincidence, determinism-"
        "suspect cells inside a class, and whether an ambiguous "
        "abstention is contained in one measurement class -- "
        "post-hoc over the main run's observations, zero extra model "
        "calls. Monday morning: 'did I actually test 24 templates, "
        "or 17 token sequences -- and was that tie two templates, or "
        "one sequence wearing two name tags?' Digests and counts "
        "only; no token IDs or text. Informational only; never "
        "touches selection. The stage never fails the run.",
    )
    run.add_argument(
        "--repeat",
        action="store_true",
        help="Prompt-repetition probe (cycle 183, research wave-7 "
        "rank 4): each search fixture's last user message is repeated "
        "verbatim inside the same turn (x2 arm) and compared against "
        "the recorded native run (x1) and a token-length-matched "
        "padding control (pad arm, frozen neutral filler), with the "
        "conservative selection re-applied per arm and per-candidate "
        "repetition effects reported over paired denominators. "
        "Monday morning: 'does repeating the request actually help "
        "my checkpoint -- and does the answer depend on which chat "
        "template carries the duplicate?' Costs 2 extra generations "
        "per candidate per fixture (4,800 at the legal maximum); "
        "doubled prompts cost real compute in this lab (full-prefix "
        "recompute, no KV cache). Informational only; never touches "
        "selection. The stage never fails the run.",
    )
    run.add_argument(
        "--distract",
        action="store_true",
        help="Distractor-dilution probe (cycle 184, research wave-7 "
        "rank 5): deterministic irrelevant filler is inserted into "
        "each search fixture's last user message at exact rendered-"
        "token doses -- neutral prose at 64/256/1024 tokens and a "
        "frozen near-miss template carrying a planted wrong answer "
        "at 64/256, before or after the content -- with verdict "
        "curves, first-flip counts, and the conservative selection "
        "re-applied per condition. Monday morning: 'which template "
        "keeps my checkpoint accurate as retrieved bulk grows "
        "around the question?' Near-miss capture is reported as an "
        "upper bound beside a neutral-control excess; near-miss "
        "covers numeric-expected fixtures with plain ASCII "
        "formatting only and is empty by construction on "
        "instruction/copy fixture sets; near-miss curves terminate "
        "at 256 tokens (no capture rate at 1024). Costs 10 extra "
        "generations per candidate per fixture (24,000 at the "
        "legal maximum), each at up to ~1024 extra prompt tokens "
        "under full-prefix recompute (forwards per generation "
        "scale as M*L + M(M-1)/2 in the generated length M and "
        "prompt length L; no KV cache). Informational only; never "
        "touches selection. The stage never fails the run.",
    )
    run.add_argument(
        "--leading",
        action="store_true",
        help="Leading-claim caving probe (cycle 185, research wave-7 "
        "rank 6): a frozen claim sentence is appended to each search "
        "fixture's last user message -- a wrong claim (another "
        "fixture's answer from a different group, or a digit-surgery "
        "plant of the fixture's own) and a right-claim control, each "
        "in first-person and third-person framings -- against the "
        "recorded native run, with caving counted per candidate per "
        "framing (pass native, pass right-claim, fail wrong-claim; "
        "adoption of the planted answer counted byte-exactly) and "
        "the conservative selection re-applied per condition. "
        "Monday morning: 'which template keeps my checkpoint from "
        "folding when a user confidently asserts a wrong answer?' "
        "Costs 4 extra generations per candidate per fixture (9,600 "
        "at the legal maximum); the native arm is reused at zero "
        "cost. Meaningful readouts need a materially larger search "
        "set than the shipped example -- on two fixtures every "
        "classification is 'insufficient' by design. Factual caving "
        "only (exact match, no LLM judge); not social sycophancy. "
        "Informational only; never touches selection. The stage "
        "never fails the run.",
    )
    run.add_argument(
        "--numform",
        action="store_true",
        help="Numeric surface-form probe (cycle 186, research wave-7 "
        "rank 7): generated integer fixtures (answers computed, not "
        "recorded) for copy / compare / add tasks, the same values "
        "written in four forms -- plain, comma-grouped (3-3-3), "
        "space-grouped (U+202F narrow no-break space), scientific -- "
        "with the expected answer in the operands' form, per "
        "candidate x task x form pass rates decomposed into exact "
        "pass / right value in the wrong form / wrong value, paired "
        "form contrasts over the same values, and the attached "
        "tokenizer's digit grouping measured at run time (a "
        "run-level label, never a within-run correlation). Monday "
        "morning: 'which written form should I normalise the "
        "figures in my prompts to, for this checkpoint, per task?' "
        "Costs candidates x 3 tasks x 16 values x 4 forms "
        "generations (4,608 at the legal maximum). Generated "
        "integer tasks only -- a probe, not a numeracy benchmark; "
        "no selection is derived from the synthetic set. "
        "Informational only; never touches selection. The stage "
        "never fails the run.",
    )
    run.add_argument(
        "--dateline",
        action="store_true",
        help="Date-line sensitivity probe (cycle 188, research wave-8 "
        "rank 1): a frozen '[Context] Today's date is ... "
        "[/Context]' block is prepended to each search fixture's "
        "last user message -- eight frozen date values (anchor "
        "2026-10-21, +/-1 day, +/-1 year, leap day 2028-02-29, a "
        "year-boundary pair) in three formats (ISO, long, "
        "weekday+ISO), a per-format no-date control sized to the "
        "anchor arm's rendered length, and one weekday-lie arm -- "
        "against the recorded native run, with per-pair discordant "
        "counts partitioned by pair type, date-attributable loss "
        "per date value, and the conservative selection re-applied "
        "per arm. If your serving stack prepends a 'Today is ...' "
        "line, this tells you -- per template family, on your own "
        "checkpoint -- how often changing only that date flips a "
        "task verdict, and whether the template you picked still "
        "wins afterwards, so a calendar rollover cannot silently "
        "change your eval results after you ship. Costs 28 extra "
        "generations per candidate per fixture (67,200 at the "
        "legal maximum: 24 date arms, 3 controls, 1 lie); the "
        "native arm is reused at zero cost. Dates are frozen "
        "protocol constants, never the wall clock. Meaningful "
        "readouts need a materially larger search set than the "
        "shipped example -- on two fixtures every classification "
        "is 'insufficient' by design. Date-derived fixtures are "
        "screened out and reported. Informational only; never "
        "touches selection. The stage never fails the run.",
    )
    run.add_argument(
        "--multiturn",
        action="store_true",
        help="Multi-turn drift probe (cycle 189, research wave-8 "
        "rank 2): generated shard tasks (ledger sums, shelf "
        "argmax; depths 2 and 3; 48 fixtures from the run seed) "
        "are delivered three ways per candidate -- all shards in "
        "one user turn, one shard per turn with the model's own "
        "boundary-cut replies as history, and one shard per turn "
        "with a frozen 'Noted.' history -- and only the final "
        "turn is scored. Monday morning: 'does the template I "
        "picked still win when the model's own earlier replies "
        "become the history?' Costs candidates x (24x4 + 24x5) "
        "generations (5,184 at the legal maximum); prompt cost "
        "grows every turn because each sharded turn re-renders "
        "the whole history. Intermediate replies are screened "
        "for special-token literals before they become history. "
        "The sharded-vs-scripted contrast is reported as a "
        "discordant pair (a bundle of authorship, content, and "
        "length), never a net-only difference. Informational "
        "only; never touches selection. The stage never fails "
        "the run.",
    )
    run.add_argument(
        "--needle",
        action="store_true",
        help="Needle-depth probe (cycle 191, research wave-8 "
        "rank 3): 20 generated cases from the run seed, each "
        "planting one fact sentence with a unique synthetic "
        "code inside neutral filler at five depths (0/25/50/75/"
        "100 percent) of two filler doses (1024 and 4096 "
        "rendered tokens), plus a no-filler control, a "
        "needle-absent false-recall floor, and a duplicate-"
        "needle arm per dose -- a recall-vs-depth curve per "
        "template family, with every depth offset verified on "
        "the candidate's own rendered tokens. Monday morning: "
        "'at which depth of the context I paste around a "
        "question does this template stop finding the fact?' "
        "Costs 15 generations per candidate per case (7,200 "
        "at the legal maximum: 24 candidates x 20 cases x 15). "
        "The 4096 dose needs --max-context above 4096 plus "
        "the needle, question, and generation budget; under "
        "the default 2048 cap every 4096 cell is reported "
        "context-exceeded, never truncated into existence. "
        "Informational only; never touches selection. The "
        "stage never fails the run.",
    )
    run.add_argument(
        "--placement",
        action="store_true",
        help="Instruction placement probe (cycle 193, research wave-8 "
        "rank 5): 24 generated cases from the run seed, each a "
        "three-sentence payload carrying exactly one answer-bearing "
        "fact in its middle sentence, asked in four arms per "
        "candidate -- question-first, payload-first, a sandwich "
        "control (question, payload, question repeated verbatim), "
        "and a closed-book screen (question alone; a pass excludes "
        "the case from that candidate's placement populations). "
        "Headline: the question-first vs payload-first discordant "
        "count split by direction, against the sandwich control's "
        "level on the same cases, with a pre-registered "
        "falsification rule. Monday morning: 'does putting the "
        "question before or after the passage change verdicts on "
        "this template, and which layout does the winner reward?' "
        "Costs 4 generations per candidate per case (2,304 at the "
        "legal maximum: 24 candidates x 24 cases x 4); each "
        "generation is one greedy decode over the arm's rendered "
        "prompt (the payload is three sentences by construction). "
        "Informational only; never touches selection. The stage "
        "never fails the run.",
    )
    run.add_argument(
        "--canonical-form",
        action="store_true",
        help="Canonical-form ladder probe (cycle 209, research wave-10 "
        "rank 1): 24 generated cases from the run seed, each a "
        "three-sentence payload carrying exactly one answer-bearing "
        "fact in its middle sentence, rendered in two arms per "
        "candidate -- NFC (composed) and NFD (decomposed) forms of "
        "the same generated message text. Accents live in carrier "
        "prose, topic, and question only; every expected value is "
        "pure ASCII, so the byte-exact scorer cannot manufacture "
        "discordance. Headline: per-case verdicts and the winner "
        "are identical across forms, conditioned on the "
        "id-divergent paired population (cases whose full render "
        "id sequences actually differ), with a pre-registered "
        "falsification rule and floors (6 paired, 2 discordant). "
        "Ids-identical cases are reported as the "
        "instrument-inapplicable share, never as robustness. "
        "Monday morning: 'must I NFC-normalize incoming text "
        "before evaluation, or does the winner I picked on NFC "
        "text silently change when the same text arrives "
        "decomposed?' Costs 2 generations per candidate per case "
        "(1,152 at the legal maximum: 24 candidates x 24 cases x "
        "2). Informational only; never touches selection. The "
        "stage never fails the run.",
    )
    run.add_argument(
        "--negation",
        action="store_true",
        help="Antonym-negated definite-description probe (cycle 212, "
        "research wave-10 rank 3): 16 generated numform compare "
        "pairs from the run seed, rendered under four instruction "
        "arms per candidate -- direct ('larger'), negated ('not "
        "the smaller'), flipped ('smaller'), and a descriptive "
        "negated_alt arm ('not smaller than the other', never a "
        "decision input). The negated arm differs from direct in "
        "two lexical moves (the negation token plus the "
        "larger-to-smaller antonym swap), so the claim is "
        "antonym-negated definite-description discordance, "
        "never negation simpliciter. Headline, per candidate "
        "over cases completed in all three headline arms: D = "
        "direct passes, J = cases in D where the negated cell "
        "is the wrong label AND the flipped cell passes on the "
        "same case; D < 8 is insufficient, D >= 8 and J >= 5 is "
        "negation_drop_observed, D >= 8 and J < 5 is "
        "below_threshold (an observed null, not proof of "
        "invariance). Flipped is a competence gate, and a "
        "flipped pass rate on D below 0.5 carries the "
        "gate-weak qualifier (negation-drop undiagnosed, not "
        "absent). No winner readout in v1. Monday morning: "
        "'does phrasing a comparison as not-the-smaller change "
        "verdicts on this template, or does the candidate hold "
        "the same label across the phrasing?' Costs 4 "
        "generations per candidate per case (1,536 at the "
        "legal maximum: 24 candidates x 16 cases x 4; 1,152 "
        "headline). Informational only; never touches "
        "selection. The stage never fails the run.",
    )
    run.add_argument(
        "--instruction-distance",
        action="store_true",
        help="Instruction-distance probe (cycle 218, research wave-11 "
        "rank 2): 24 generated cases from the run seed, each a fixed "
        "fact+question block governed by one frozen format directive "
        "('Reply with only the number: no units, no sentence, nothing "
        "else.'), rendered in six arms per candidate -- adjacent "
        "(directive, block), separated at two doses (directive, "
        "filler, block), displaced at the same two doses (directive, "
        "block, filler), and a descriptive order control (block, "
        "directive). Distance here means rendered tokens between "
        "the frozen directive and the question block, verified per "
        "candidate on its own rendered tokens (never edit "
        "distance): a dose is the exact rendered-token delta of the "
        "arm over the case's adjacent render, sized from the "
        "neutral corpus with a single-phase exact sizing rule; a "
        "cell whose delta cannot be sized exactly is "
        "arm_unmatchable, never approximated. Headline, per "
        "candidate per dose over cases completed in the adjacent, "
        "separated, and displaced arms of that dose: D = adjacent "
        "compliant-correct under the frozen digits grammar (the "
        "span matches [0-9]+ over the whole span, ASCII digits "
        "only), S = separated noncompliant in D, T = displaced "
        "noncompliant in D; the label "
        "instruction_distance_drop_observed fires iff D >= 10 and "
        "S >= 6 and S - T >= 5, and D < 10 is insufficient. The "
        "instrument is a per-dose large-drop falsifier: power is "
        "0.909 at a 50 percent conditional drop rate and only "
        "0.246 at a 25 percent drop rate, which is below this "
        "panel's resolution, so a below_threshold verdict "
        "excludes large "
        "separation-driven collapse and claims nothing smaller. "
        "The label is an excess-of-between-losses label, never a "
        "distance effect simpliciter: a pure constraint-to-"
        "generation effect yields S = T and reads below_threshold "
        "even at total collapse. No winner readout; discordance "
        "is never pooled across candidates. Monday morning: 'does "
        "placing 256 or 1024 tokens of filler between a format "
        "directive and the question it governs produce a large "
        "format-compliance collapse beyond what the same filler "
        "after the question produces, on this template family?' "
        "Costs 6 generations per candidate per case (3,456 at the "
        "legal maximum: 24 candidates x 24 cases x 6), plus "
        "sizing encodes (about 56K expected at the shipped "
        "estimator's mean, worst case about 1.18M at the frozen "
        "512-encode per-cell budget). Informational only; never "
        "touches selection. The stage never fails the run.",
    )
    run.add_argument(
        "--history-frame",
        action="store_true",
        help="History-frame probe (cycle 219, research wave-11 rank "
        "3): this probe prepends the frozen six-turn synthetic "
        "history to every search fixture and reframes ONLY that "
        "prepended segment -- the fixture's own messages are "
        "never reframed. Per candidate, the prepended segment is "
        "rendered in four arms: own (the candidate's own framing "
        "of the history segment), foreign_chatml and "
        "foreign_llama3 (the segment in the frozen chatml / "
        "llama3 framings), and a descriptive verbatim_chatml arm "
        "that differs from foreign_chatml only in the leading "
        "bos string, so its discordance against foreign_chatml "
        "is the bos-substitution share; when the candidate's bos "
        "equals chatml's, the verbatim arm is byte-identical to "
        "its twin and is recorded inapplicable for that "
        "candidate, never read as a bos share of zero. The tail, "
        "the prompt-initial bos, and the open suffix are always "
        "the candidate's own, and the own arm's render is "
        "byte-identical to the synth-history history_6 render of "
        "the same messages (verdict-identity with the "
        "synth-history stage is not claimed: that stage runs "
        "namespaced ids, hence different request seeds). "
        "Headline, per candidate per foreign framing over paired "
        "completed render-divergent ids-divergent fixtures: n "
        "paired, a = own-only passes, b = foreign-only passes; "
        "n below 10 is insufficient, no discordance is "
        "no_discordance_observed, fewer than 6 discordant is "
        "below_threshold, an even split is discordant_balanced, "
        "and otherwise a two-sided exact McNemar p below 0.05 is "
        "discordant_asymmetric -- the only falsification label. "
        "The exact random-labeller fire rate at the run's own n "
        "prints beside every classification (3.6 percent at "
        "n = 100), and the family-wise expectation is computed "
        "from those rates over the run's decisions (about 1.76 "
        "expected false labels over 48 decisions at n = 100 "
        "under the random null). Power at a 25-point drop is "
        "0.99 at n = 100 but a coin flip at n of 30 or below, so "
        "below_threshold is weak evidence, never invariance. "
        "Per-arm conservative selection is recomputed on the "
        "common completed set, with winner headlines per "
        "contrast; flips against the primary run pool history "
        "presence with frame and are labelled as such, never "
        "the headline. The label certifies string-level "
        "discordance on this run, not a mechanism: arm-level "
        "batch composition and numerics channels correlated "
        "with the foreign prompts' length are named residuals. "
        "Monday morning: 'can a history rendered in the old "
        "family's framing be trusted before the final turn, or "
        "must it be re-rendered first?' Costs 4 generations per "
        "candidate per search fixture (9,600 at the legal "
        "maximum: 24 candidates x 100 fixtures x 4; 7,200 "
        "headline). Informational only; never touches "
        "selection. The stage never fails the run.",
    )
    run.add_argument(
        "--prune-sweep",
        action="store_true",
        help="Prune-sweep probe (cycle 220, research wave-11 rank "
        "4): this probe prunes passage words -- seeded interior "
        "words of a frozen generated passage -- at exact word "
        "doses, and classifies the verdict drop per candidate. "
        "Each of the 24 generated cases is one user message: a "
        "frozen instruction, a 17-sentence passage (16 context "
        "sentences of exactly nine words carrying 144 eligible "
        "words, with the six-word answer-bearing fact sentence "
        "at passage position 9, never pruned), and a frozen "
        "question; the instruction and question are never "
        "pruned either. Arms per candidate: dose0 / dose18 / "
        "dose36 / dose72 (the first 0 / 18 / 36 / 72 words of "
        "the case's frozen sha256 drop order over the eligible "
        "positions, nested, so 72 words is half the eligible "
        "words), the descriptive tail72 displaced control "
        "(drops exactly the eligible words of the passage's "
        "last eight context sentences -- the dose-matched tail "
        "loss), and the closed_book screen arm (no passage; a "
        "case passed closed-book is screened out of the "
        "headline). Scoring is byte-exact with boundary, every "
        "arm a fixture with the same id, so request seeds are "
        "identical across arms. Headline, per candidate over "
        "cases completed in all four dose arms, not screened, "
        "and not ids-identical between dose0 and dose72: C = "
        "dose0 passes; A = cases in C whose dose72 answer "
        "digest is a wrong pool value or a non-value (a dose72 "
        "failure whose digest still equals the expected one is "
        "boundary attrition, printed beside A, never counted "
        "in it); B = reverse discordants. C below 12 is "
        "insufficient; otherwise A of at least 5 with a "
        "two-sided exact McNemar p below 0.05 is "
        "prune_drop_observed -- the only falsification label -- "
        "and anything else is below_threshold. The McNemar "
        "conjunct is load-bearing: (5, 0) gives p = 0.0625 and "
        "does not fire; (6, 0) fires. Exact-enumeration fire "
        "rates print beside every decision: under the stated "
        "conditional null the random labeller fires 0.0121 and "
        "the competent no-effect panel 0.0143 per decision "
        "(family-wise 0.34 expected false labels over a "
        "24-candidate panel), and power reaches only 0.40 at a "
        "25 percent top-dose drop, so below_threshold is weak "
        "evidence, never invariance. The label attributes the "
        "drop to the word-loss treatment as a bundle -- "
        "information loss, prompt shortening, fact-position "
        "shift (mean 36 words earlier at dose72, exactly 0 "
        "under tail72), and sentence breakage move together; "
        "when the tail72 discordance is within noise of "
        "dose72's, the section prints the word-count-loss "
        "reading in words. Shape is a readout, never a gate: "
        "the word sweep denotes the dose ladder, never a curve "
        "shape. No winner readout and no selection over the "
        "generated panel. Monday morning: 'may our trimming "
        "pipeline delete half the non-answer-bearing words of "
        "a retrieved passage before verdicts start dropping?' "
        "Costs 6 generations per candidate per case (3,456 at "
        "the legal maximum: 24 candidates x 24 cases x 6; "
        "2,304 headline). Informational only; never touches "
        "selection. The stage never fails the run.",
    )
    run.add_argument(
        "--record-notation",
        action="store_true",
        help="Record-notation probe (cycle 221, payload-"
        "serialisation line): this probe renders the same "
        "generated data table in two frozen notations -- prose "
        "sentences and canonical JSON -- inside the user "
        "message, and reports, per candidate, on how many of "
        "24 paired generated tables the two notations' "
        "byte-exact verdicts discord, in which direction, and "
        "whether the discordance is exact-only, boundary-only, "
        "or mixed. It is descriptive only: no classification, "
        "no winner readout. Each case is one generated code "
        "table (4 to 6 records of 3 to 4 fields by a frozen "
        "shape schedule; every value a single token of the form "
        "word-digits from a frozen 24-word list over a 216,000 "
        "code space, all values in a table distinct, so the "
        "prose rendering is injective -- it has an exhibited "
        "inverse -- and JSON escaping is unreachable). The "
        "generator's RNG call sequence is frozen (word draw, "
        "then number draw, duplicates consumed, row-major fill, "
        "the queried record is the next draw). Arms per "
        "candidate: prose and canonical_json (the record "
        "sequence held constant; only the notation moves, as a "
        "bundle of length, delimiters, punctuation, lexical "
        "framing, field order as a reading sequence, and output "
        "priming), plus a closed_book screen arm (question and "
        "contract only; a case passed closed-book is excluded "
        "from that candidate's live populations, and the "
        "unscreened discordance prints as a sensitivity). The "
        "screen arm costs 576 of the 1,728 generations at the "
        "legal maximum (plus 50 percent over the headline arms) "
        "for an expected 0.000111 exclusions per candidate -- "
        "it is retained as fail-closed hygiene against "
        "generator or code-space drift, and its pass count is "
        "printed as a leak readout. Scoring is byte-exact with "
        "boundary, every arm a fixture with the same id, so "
        "request seeds are identical across arms. The headline "
        "population is live cases with both headline cells "
        "completed, minus same-ids anomalies; token-ID "
        "divergence is reported, not gated. Beside the counts: "
        "per-arm echo and parsed-value rates (independent "
        "diagnostics -- JSON string escapes break the subset "
        "relation between them), R1 and fence regrade "
        "discordance, and the readout's resolution (one paired "
        "case is 4.2 points of discordance rate at the full "
        "panel). Why no classification: at 24 cases every "
        "count-based rule false-fires on competent panels (the "
        "canonical-form transplant fires 0.8897 under a "
        "competent independent null) and every family-wise "
        "controlled rule is near-inert (Bonferroni sign power "
        "0.0542 at a 25-point gap); the panel that would "
        "license a directional claim costs 6,912 generations "
        "for a claim this readout does not consume. Monday "
        "morning: 'for this checkpoint and this candidate's "
        "template family, on how many of 24 paired tables do "
        "prose sentences and canonical JSON discord -- and in "
        "which direction -- before we switch the frozen "
        "notation our prompt assembly uses for generated data "
        "tables?' Costs 3 generations per candidate per case "
        "(1,728 at the legal maximum: 24 candidates x 24 cases "
        "x 3; 1,152 headline). Informational only; never "
        "touches selection. The stage never fails the run.",
    )
    run.add_argument(
        "--rotate",
        action="store_true",
        help="Context rotation probe (cycle 203, research wave-9 "
        "rank 1): 20 generated cases from the run seed, each a "
        "span of five fact-bearing blocks (one asked fact, four "
        "fact-shaped distractors) with the question pinned after "
        "the span, run in eleven arms per candidate -- the five "
        "cyclic rotations of the block order; four mirror arms "
        "(rotations of the reversed order, the matched generic-"
        "reordering control whose asked-slot multiset equals the "
        "rotations'); a fact-absent arm (the false-recall floor); "
        "and a closed-book screen (question alone; a pass "
        "excludes the case from that candidate's rotation "
        "populations). A span arm counts as construction-exact "
        "only when its rendered token inventory digest equals "
        "the unrotated arm's, so paired readouts run over "
        "token-exact rotations only. Headline: verdict "
        "invariance across the five rotations, with a "
        "pre-registered falsification rule and a pass-by-"
        "absolute-slot curve presented as a cross-check against "
        "the needle probe's depth curve. Monday morning: 'if my "
        "retriever re-ranks the same five chunks into a "
        "different cyclic order, does this template keep the "
        "same verdict -- and the same winner?' Costs 11 "
        "generations per candidate per case (5,280 at the legal "
        "maximum: 24 candidates x 20 cases x 11); each "
        "generation is one greedy decode over the arm's rendered "
        "prompt. Informational only; never touches selection. "
        "The stage never fails the run.",
    )
    run.add_argument(
        "--density",
        action="store_true",
        help="Instruction-density ladder (cycle 204, research "
        "wave-9 rank 3): 20 generated cases from the run seed, "
        "each carrying 16 simultaneously-satisfiable constraints "
        "(four each of: include a word, start with a word, end "
        "with a word, stay within a word limit; start and end "
        "targets are shared per case, so a perfect follower "
        "passes every arm by construction). Per candidate, 21 "
        "arms: each constraint alone (solo), the first 2/4/8/16 "
        "constraints together, and a length-matched sham (the "
        "first constraint plus neutral text matched to the "
        "16-constraint block's length). Answers are committed "
        "as ordered per-word hash lists; every satisfaction "
        "verdict is derived from the commitment, never stored. "
        "Headline: the solo-vs-full-density contrast, decided "
        "per case by an exact sign test with the evidence "
        "withheld below six discordant cases (insufficient). "
        "Monday morning: 'my system prompt keeps growing -- "
        "which of its instructions does this template silently "
        "drop first, and at what stack depth?' Costs 21 "
        "generations per candidate per case (10,080 at the "
        "legal maximum: 24 candidates x 20 cases x 21); each "
        "generation is one greedy decode over the arm's "
        "rendered prompt. Informational only; never touches "
        "selection. The stage never fails the run.",
    )
    run.add_argument(
        "--field-order",
        action="store_true",
        help="Schema field-order probe (cycle 205, research "
        "wave-9 rank 4): 15 frozen flat-JSON specs (the shipped "
        "structured set plus six 3-4-field specs), each rendered "
        "in four field orders per candidate -- canonical "
        "(alphabetical), the designated answer field first, the "
        "answer field last, and fully reversed. Answers are "
        "committed as per-field canonical values; correctness, "
        "schema adherence, and the pairing are derived from the "
        "commitment, never stored. Headline: the answer field's "
        "first-vs-last accuracy over specs adherent in both "
        "endpoint arms, decided by an exact sign test with the "
        "evidence withheld below six discordant specs "
        "(insufficient); the decision's printed characteristics "
        "state that only very large effects can falsify. Monday "
        "morning: 'I am defining a JSON schema for extraction -- "
        "does putting the main answer field first change what "
        "this template produces for it?' Costs 60 generations "
        "per candidate (1,440 at the legal maximum: 24 "
        "candidates x 15 specs x 4 arms); each generation is one "
        "greedy decode over the arm's rendered prompt. "
        "Informational only; never touches selection. The stage "
        "never fails the run.",
    )
    run.add_argument(
        "--choice-audit",
        action="store_true",
        help="Choice-scoring audit (cycle 195, research wave-8 "
        "rank 6): a second opinion on the generation winner. "
        "For each search fixture a four-choice set is built from "
        "the run's own fixtures -- the expected answer plus three "
        "donor expecteds from other groups, screened so no donor "
        "is the same answer twice -- and every choice is "
        "teacher-forced under every candidate over the recorded "
        "search prompt (zero extra generations). Reports per "
        "candidate: choice accuracy under the total "
        "log-likelihood ranking (with per-byte and per-token-mean "
        "rankings beside it; the per-token mean is not lm-eval's "
        "acc_norm, which is per-byte), the generation x choice "
        "discordance table, and whether the choice method's top "
        "candidate agrees with the primary winner over the "
        "common evaluable population. Monday morning: 'does "
        "log-likelihood choice scoring pick the same answers and "
        "the same template as greedy generation on this "
        "checkpoint?' Costs 4 forward passes per candidate per "
        "fixture (9,600 at the legal maximum: 24 candidates x "
        "100 fixtures x 4). Informational only; never touches "
        "selection. The stage never fails the run.",
    )
    run.add_argument(
        "--precision",
        action="store_true",
        help="Load-precision probe (cycle 198, research wave-8 "
        "rank 7): loads the same checkpoint a second time in "
        "bfloat16 and re-evaluates every search candidate on the "
        "search split, against the recorded fp32 native run. "
        "Reports the per-candidate verdict-flip matrix over "
        "paired fixtures, the output-divergence rate, the "
        "teacher-forced expected-answer logprob deltas, and "
        "whether the fp32 winner survives as the selection under "
        "bf16 (withheld on mismatched completed sets). This is "
        "dtype precision, not quantization: the bf16 arm is a "
        "load-time cast of the same unquantized weights. Monday "
        "morning: 'I will serve this checkpoint in bf16 -- is "
        "the template winner I picked under fp32 still the "
        "winner there?' Costs a second full search evaluation "
        "plus a second checkpoint load, with both models "
        "resident at once; on CPU bf16 measured several times "
        "slower than fp32 in this lab's sandbox, so the stage "
        "can dominate run time. Informational only; never "
        "touches selection. The stage never fails the run.",
    )
    run.add_argument(
        "--confidence",
        action="store_true",
        help="Verbalized-confidence probe (cycle 199, research wave-8 "
        "rank 8): appends one frozen suffix to each search fixture's "
        "last user message asking the model to end its reply with "
        "'Confidence: <integer 0-100>', and re-evaluates every "
        "candidate. Reports per candidate: the compliance rate "
        "under the frozen line grammar, mean stated confidence on "
        "correct vs incorrect compliant answers (a descriptive gap, "
        "never a calibration verdict -- a stated number is model "
        "output text, not a probability), a frozen four-bin table, "
        "and the verdict flips the suffix itself caused against "
        "the recorded native run. A generated paired control "
        "(24 planted-fact cases whose value the prompt states, vs "
        "the same cases with the payload binding a different key) "
        "tests whether the stated number drops when the answer is "
        "not knowable from the prompt. Monday morning: 'when this "
        "checkpoint states a confidence under each template, does "
        "the number track whether it was right -- and does it drop "
        "when the answer is not in the prompt?' Costs one full "
        "extra search evaluation (2,400 generations at the legal "
        "maximum: 24 candidates x 100 fixtures) plus a generated "
        "paired control (2 arms x 24 cases x 24 candidates = "
        "1,152) -- 3,552 extra generations in total, each on a "
        "suffix-lengthened prompt. Informational only; never "
        "touches selection. The stage never fails the run.",
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
        "--token-depth-sweep",
        action="store_true",
        help="Re-run search with the prompt truncated by whole trailing "
        "tokens at each depth (1..--max-drop-tokens) to measure "
        "token-depth sensitivity; a prompt-format researcher checking "
        "whether a winner survives losing its trailing tokens (for "
        "example the multi-token assistant turn-closing header) uses "
        "this readout before trusting the verdict; multiplies search "
        "compute, combinable with --sensitivity-sweep",
    )
    run.add_argument(
        "--max-drop-tokens",
        type=int,
        default=4,
        help="Truncation depths for --token-depth-sweep (1-8, provisional)",
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
    "local_batch_cancelled",
    "local_batch_failed",
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


def _run_refusals_probe(result: Any) -> None:
    """Run the suspected-refusal profile probe; never raises.

    Post-hoc over the main run's recorded search observations (zero
    extra model calls): each candidate's recorded answers are
    classified against the frozen refusal-opener list (see
    :mod:`refusals`) and the per-candidate suspected-refusal rates
    are stored as the ``refusals`` profile on ``result``. Any
    failure degrades to an "unavailable" marker instead of failing
    the run.
    """
    try:
        from .experiments import refusal_profile

        result["refusals"] = refusal_profile(result)
    except Exception:  # noqa: BLE001 -- refusals probe must not fail the run
        result["refusals"] = {
            "available": False,
            "detail": "refusals_unavailable",
        }


def _run_scorerlens_probe(result: Any, *, search_fixtures: Any) -> None:
    """Run the scorer-lens regrade probe; never raises.

    Post-hoc over the main run's recorded search observations (zero
    extra model calls): each candidate's recorded answers are
    regraded under the frozen normalisation ladder (see
    :mod:`scorerlens`) against the expected texts joined from the
    run's own fixture set -- recorded results carry no ``expected``
    field by design, so the join is explicit here, exactly as in
    the verbosity probe. The profile is stored as the
    ``scorerlens`` payload on ``result``. Any failure degrades to
    an "unavailable" marker instead of failing the run.
    """
    try:
        from .experiments import scorerlens_profile

        expected_by_fixture = {f.id: f.expected for f in search_fixtures}
        result["scorerlens"] = scorerlens_profile(result, expected_by_fixture)
    except Exception:  # noqa: BLE001 -- scorer-lens probe must not fail the run
        result["scorerlens"] = {
            "available": False,
            "detail": "scorerlens_unavailable",
        }


def _run_aliasing_probe(result: Any) -> None:
    """Run the candidate aliasing audit probe; never raises.

    Post-hoc over the main run's recorded search observations (zero
    extra model calls): candidates are grouped into render and
    measurement equivalence classes over the recorded input IDs
    (see :mod:`aliasing`) and the audit is stored as the
    ``aliasing`` payload on ``result``. Unlike the scorer-lens the
    profile needs no extra join -- everything it consumes is in
    the recorded result. Any failure degrades to an "unavailable"
    marker instead of failing the run.
    """
    try:
        from .experiments import aliasing_profile

        result["aliasing"] = aliasing_profile(result)
    except Exception:  # noqa: BLE001 -- aliasing probe must not fail the run
        result["aliasing"] = {
            "available": False,
            "detail": "aliasing_unavailable",
        }


def _run_repeat_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the prompt-repetition probe; never raises.

    The x1 arm is the recorded native search, so a failed, cancelled,
    or truncated search gates the probe to zero generations (the
    treated arms would have nothing sound to compare against). The
    run's cancel event is threaded dispatch -> wrapper -> stage ->
    evaluate, so SIGINT/SIGTERM stop the stage cooperatively.
    """
    try:
        from .experiments import repetition_analysis, repetition_stage

        search = result.get("search")
        names = tuple(fmt.name for fmt in formats)
        if (
            result.get("decision") in ("failed", "cancelled")
            or type(search) is not list
            or len(search) != len(formats)
        ):
            result["repetition"] = {
                "available": False,
                "detail": "native_search_incomplete",
            }
            return
        stage = repetition_stage(
            backend,
            formats,
            cases,
            search,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["repetition"] = repetition_analysis(
            stage,
            result.get("decision"),
            result.get("selected_candidate"),
            names,
        )
    except Exception:  # noqa: BLE001 -- repetition must not fail the run
        result["repetition"] = {
            "available": False,
            "detail": "repetition_unavailable",
        }


def _run_distract_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the distractor-dilution probe; never raises.

    The native arm is the recorded native search, so a failed,
    cancelled, or truncated search gates the probe to zero
    generations. The run's cancel event is threaded dispatch ->
    wrapper -> stage -> evaluate, so SIGINT/SIGTERM stop the
    stage cooperatively.
    """
    try:
        from .experiments import distraction_analysis, distraction_stage

        search = result.get("search")
        names = tuple(fmt.name for fmt in formats)
        if (
            result.get("decision") in ("failed", "cancelled")
            or type(search) is not list
            or len(search) != len(formats)
        ):
            result["distraction"] = {
                "available": False,
                "detail": "native_search_incomplete",
            }
            return
        stage = distraction_stage(
            backend,
            formats,
            cases,
            search,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["distraction"] = distraction_analysis(
            stage,
            result.get("decision"),
            result.get("selected_candidate"),
            names,
        )
    except Exception:  # noqa: BLE001 -- distraction must not fail the run
        result["distraction"] = {
            "available": False,
            "detail": "distraction_unavailable",
        }


def _run_leading_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the leading-claim caving probe; never raises.

    The native arm is the recorded native search, so a failed,
    cancelled, or truncated search gates the probe to zero
    generations. The run's cancel event is threaded dispatch ->
    wrapper -> stage -> evaluate, so SIGINT/SIGTERM stop the
    stage cooperatively.
    """
    try:
        from .experiments import leading_analysis, leading_stage

        search = result.get("search")
        names = tuple(fmt.name for fmt in formats)
        if (
            result.get("decision") in ("failed", "cancelled")
            or type(search) is not list
            or len(search) != len(formats)
        ):
            result["leading"] = {
                "available": False,
                "detail": "native_search_incomplete",
            }
            return
        stage = leading_stage(
            backend,
            formats,
            cases,
            search,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["leading"] = leading_analysis(
            stage,
            result.get("decision"),
            result.get("selected_candidate"),
            names,
        )
    except Exception:  # noqa: BLE001 -- leading must not fail the run
        result["leading"] = {
            "available": False,
            "detail": "leading_unavailable",
        }


def _run_multiturn_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the multi-turn drift probe; never raises.

    The probe is self-contained: its fixtures are generated
    from the run seed, so -- unlike the native-arm probes -- it
    consults neither the recorded search nor the primary
    decision, and a failed user run cannot silently suppress
    it. The run's cancel event is threaded dispatch -> wrapper
    -> stage -> evaluate, so SIGINT/SIGTERM stop the stage
    cooperatively.
    """
    try:
        from .experiments import multiturn_analysis, multiturn_stage

        names = tuple(fmt.name for fmt in formats)
        stage = multiturn_stage(
            backend,
            formats,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["multiturn"] = multiturn_analysis(stage, names)
    except Exception:  # noqa: BLE001 -- multiturn must not fail the run
        result["multiturn"] = {
            "available": False,
            "detail": "multiturn_unavailable",
        }


def _run_placement_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the instruction placement probe; never raises.

    The probe is self-contained: its cases are generated from
    the run seed, so it consults neither the recorded search
    nor the primary decision, and a failed user run cannot
    silently suppress it. The run's cancel event is threaded
    dispatch -> wrapper -> stage -> evaluate, so SIGINT/
    SIGTERM stop the stage cooperatively.
    """
    try:
        from .experiments import placement_analysis, placement_stage

        names = tuple(fmt.name for fmt in formats)
        stage = placement_stage(
            backend,
            formats,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["placement"] = placement_analysis(stage, names)
    except Exception:  # noqa: BLE001 -- placement must not fail the run
        result["placement"] = {
            "available": False,
            "detail": "placement_unavailable",
        }


def _run_canonical_form_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the canonical-form ladder probe; never raises.

    The probe is self-contained: its cases are generated from
    the run seed, so it consults neither the recorded search
    nor the primary decision, and a failed user run cannot
    silently suppress it. The run's cancel event is threaded
    dispatch -> wrapper -> stage -> evaluate, so SIGINT/
    SIGTERM stop the stage cooperatively.
    """
    try:
        from .experiments import canonicalform_analysis, canonicalform_stage

        names = tuple(fmt.name for fmt in formats)
        stage = canonicalform_stage(
            backend,
            formats,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["canonical_form"] = canonicalform_analysis(stage, names)
    except Exception:  # noqa: BLE001 -- canonical-form must not fail the run
        result["canonical_form"] = {
            "available": False,
            "detail": "canonical_form_unavailable",
        }


def _run_negation_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the antonym-negated definite-description probe; never raises.

    The probe is self-contained: its cases are generated from
    the run seed, so it consults neither the recorded search
    nor the primary decision, and a failed user run cannot
    silently suppress it. The run's cancel event is threaded
    dispatch -> wrapper -> stage -> evaluate, so SIGINT/
    SIGTERM stop the stage cooperatively.
    """
    try:
        from .experiments import negation_analysis, negation_stage

        names = tuple(fmt.name for fmt in formats)
        stage = negation_stage(
            backend,
            formats,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["negation"] = negation_analysis(stage, names)
    except Exception:  # noqa: BLE001 -- negation must not fail the run
        result["negation"] = {
            "available": False,
            "detail": "negation_unavailable",
        }


def _run_instruction_distance_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the instruction-distance probe; never raises.

    The probe is self-contained: its cases are generated from
    the run seed, so it consults neither the recorded search
    nor the primary decision, and a failed user run cannot
    silently suppress it. The run's cancel event is threaded
    dispatch -> wrapper -> stage -> evaluate, so SIGINT/
    SIGTERM stop the stage cooperatively.
    """
    try:
        from .experiments import instruction_distance_analysis, instruction_distance_stage

        names = tuple(fmt.name for fmt in formats)
        stage = instruction_distance_stage(
            backend,
            formats,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["instruction_distance"] = instruction_distance_analysis(stage, names)
    except Exception:  # noqa: BLE001 -- instruction-distance must not fail the run
        result["instruction_distance"] = {
            "available": False,
            "detail": "instruction_distance_unavailable",
        }


def _run_history_frame_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    fixtures: Any,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the history-frame probe; never raises.

    The probe treats the run's own search fixtures (preceded
    by the frozen synthetic history), so it reads the recorded
    primary decision and winner for its separately-labelled
    flips readout, but it never writes to them: the primary
    decision, winner, and exit code are untouched whatever
    the stage does. The run's cancel event is threaded
    dispatch -> wrapper -> stage -> evaluate, so SIGINT/
    SIGTERM stop the stage cooperatively.
    """
    try:
        from .experiments import history_frame_analysis, history_frame_stage

        names = tuple(fmt.name for fmt in formats)
        stage = history_frame_stage(
            backend,
            formats,
            fixtures,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        primary = {
            "decision": result.get("decision"),
            "winner": result.get("selected_candidate"),
        }
        result["history_frame"] = history_frame_analysis(stage, names, primary)
    except Exception:  # noqa: BLE001 -- history-frame must not fail the run
        result["history_frame"] = {
            "available": False,
            "detail": "historyframe_unavailable",
        }


def _run_prune_sweep_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the prune-sweep probe; never raises.

    The probe is self-contained: its cases are generated from
    the run seed, so it consults neither the recorded search
    nor the primary decision, and a failed user run cannot
    silently suppress it. The run's cancel event is threaded
    dispatch -> wrapper -> stage -> evaluate, so SIGINT/
    SIGTERM stop the stage cooperatively.
    """
    try:
        from .experiments import prune_sweep_analysis, prune_sweep_stage

        names = tuple(fmt.name for fmt in formats)
        stage = prune_sweep_stage(
            backend,
            formats,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["prune_sweep"] = prune_sweep_analysis(stage, names)
    except Exception:  # noqa: BLE001 -- prune-sweep must not fail the run
        result["prune_sweep"] = {
            "available": False,
            "detail": "prune_unavailable",
        }


def _run_record_notation_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the record-notation probe; never raises.

    The probe is self-contained: its cases are generated from
    the run seed, so it consults neither the recorded search
    nor the primary decision, and a failed user run cannot
    silently suppress it. The run's cancel event is threaded
    dispatch -> wrapper -> stage -> evaluate, so SIGINT/SIGTERM
    stop the stage cooperatively.
    """
    try:
        from .experiments import record_notation_analysis, record_notation_stage

        names = tuple(fmt.name for fmt in formats)
        stage = record_notation_stage(
            backend,
            formats,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["record_notation"] = record_notation_analysis(stage, names)
    except Exception:  # noqa: BLE001 -- record-notation must not fail the run
        result["record_notation"] = {
            "available": False,
            "detail": "record_notation_unavailable",
        }


def _run_rotation_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the context rotation probe; never raises.

    The probe is self-contained: its cases are generated from
    the run seed, so it consults neither the recorded search
    nor the primary decision, and a failed user run cannot
    silently suppress it. The run's cancel event is threaded
    dispatch -> wrapper -> stage -> evaluate, so SIGINT/
    SIGTERM stop the stage cooperatively.
    """
    try:
        from .experiments import rotation_analysis, rotation_stage

        names = tuple(fmt.name for fmt in formats)
        stage = rotation_stage(
            backend,
            formats,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["rotation"] = rotation_analysis(stage, names)
    except Exception:  # noqa: BLE001 -- rotation must not fail the run
        result["rotation"] = {
            "available": False,
            "detail": "rotation_unavailable",
        }


def _run_density_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the instruction-density probe; never raises.

    The probe is self-contained: its cases are generated from
    the run seed, so it consults neither the recorded search
    nor the primary decision, and a failed user run cannot
    silently suppress it. The run's cancel event is threaded
    dispatch -> wrapper -> stage -> evaluate, so SIGINT/
    SIGTERM stop the stage cooperatively.
    """
    try:
        from .experiments import density_analysis, density_stage

        names = tuple(fmt.name for fmt in formats)
        stage = density_stage(
            backend,
            formats,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["density"] = density_analysis(stage, names)
    except Exception:  # noqa: BLE001 -- density must not fail the run
        result["density"] = {
            "available": False,
            "detail": "density_unavailable",
        }


def _run_fieldorder_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the schema field-order probe; never raises.

    The probe is self-contained: its specs are frozen in the
    fieldorder module, so it consults neither the recorded
    search nor the primary decision. The generation budget is
    the protocol's fixed 64 new tokens (the record caps are
    derived from it), not the run's --max-new-tokens. The run's
    cancel event is threaded dispatch -> wrapper -> stage ->
    evaluate, so SIGINT/SIGTERM stop the stage cooperatively.
    """
    try:
        from . import fieldorder as _fo
        from .experiments import fieldorder_analysis, fieldorder_stage

        names = tuple(fmt.name for fmt in formats)
        stage = fieldorder_stage(
            backend,
            formats,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=_fo.MAX_NEW_TOKENS,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["field_order"] = fieldorder_analysis(stage, names)
    except Exception:  # noqa: BLE001 -- field-order must not fail the run
        result["field_order"] = {
            "available": False,
            "detail": "field_order_unavailable",
        }


def _run_choiceaudit_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    cancel: Any,
) -> None:
    """Run the choice-scoring audit; never raises.

    The audit's native arm is the recorded native search, so a
    failed, cancelled, or truncated search gates it to zero
    forwards (leading's wrapper gate). Teacher forcing runs
    over the recorded prompts; the run's cancel event is
    threaded into the stage.
    """
    try:
        from .experiments import choiceaudit_analysis, choiceaudit_stage

        search = result.get("search")
        names = tuple(fmt.name for fmt in formats)
        if (
            result.get("decision") in ("failed", "cancelled")
            or type(search) is not list
            or len(search) != len(formats)
        ):
            result["choice_audit"] = {
                "available": False,
                "detail": "native_search_incomplete",
            }
            return
        stage = choiceaudit_stage(backend, formats, cases, search, cancel=cancel)
        result["choice_audit"] = choiceaudit_analysis(
            stage,
            result.get("decision"),
            result.get("selected_candidate"),
            names,
        )
    except Exception:  # noqa: BLE001 -- choice audit must not fail the run
        result["choice_audit"] = {
            "available": False,
            "detail": "choiceaudit_unavailable",
        }


def _run_precision_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    bf16_backend: Any,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the load-precision probe; never raises.

    The probe's fp32 arm is the recorded native search, so a
    failed, cancelled, or truncated search gates it to zero
    forwards (the choice-audit wrapper gate, verbatim). The
    bf16 arm is a second LocalHFBackend over the same checkpoint
    directory, constructed by the dispatch site in the stage
    phase (None here when that construction failed) with the
    full forwarding set (device, max_context, boundary and
    boundary_map, budget, seed), so the two arms differ in load
    dtype only. Informational only; never touches selection.
    """
    try:
        from .experiments import precision_stage

        search = result.get("search")
        if (
            result.get("decision") in ("failed", "cancelled")
            or type(search) is not list
            or len(search) != len(formats)
        ):
            result["precision"] = {
                "available": False,
                "detail": "native_search_incomplete",
            }
            return
        if bf16_backend is None:
            result["precision"] = {
                "available": False,
                "detail": "precision_unavailable",
            }
            return
        result["precision"] = precision_stage(
            backend,
            bf16_backend,
            formats,
            cases,
            result,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
            cancel=cancel,
        )
    except Exception:  # noqa: BLE001 -- precision probe must not fail the run
        result["precision"] = {
            "available": False,
            "detail": "precision_unavailable",
        }


def _run_confidence_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the verbalized-confidence probe; never raises.

    The flips' native arm is the recorded native search, so a
    failed, cancelled, or truncated search gates the probe to zero
    forwards (the choice-audit wrapper gate, verbatim). The stage
    re-evaluates the treated fixtures and the generated pair under
    the run's own backend, boundaries, budget, and seed.
    Informational only; never touches selection.
    """
    try:
        from .experiments import confidence_stage

        search = result.get("search")
        if (
            result.get("decision") in ("failed", "cancelled")
            or type(search) is not list
            or len(search) != len(formats)
        ):
            result["confidence"] = {
                "available": False,
                "detail": "native_search_incomplete",
            }
            return
        result["confidence"] = confidence_stage(
            backend,
            formats,
            cases,
            result,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
            cancel=cancel,
        )
    except Exception:  # noqa: BLE001 -- confidence probe must not fail the run
        result["confidence"] = {
            "available": False,
            "detail": "confidence_unavailable",
        }


def _run_needle_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the needle-depth probe; never raises.

    The probe is self-contained: its cases are generated from
    the run seed, so -- unlike the native-arm probes -- it
    consults neither the recorded search nor the primary
    decision, and a failed user run cannot silently suppress
    it. The run's cancel event is threaded dispatch -> wrapper
    -> stage -> evaluate, so SIGINT/SIGTERM stop the stage
    cooperatively.
    """
    try:
        from .experiments import needle_analysis, needle_stage

        names = tuple(fmt.name for fmt in formats)
        stage = needle_stage(
            backend,
            formats,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["needle"] = needle_analysis(stage, names)
    except Exception:  # noqa: BLE001 -- needle must not fail the run
        result["needle"] = {
            "available": False,
            "detail": "needle_unavailable",
        }


def _run_dateline_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    cases: Any,
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the date-line sensitivity probe; never raises.

    The native arm is the recorded native search, so a failed,
    cancelled, or truncated search gates the probe to zero
    generations. The run's cancel event is threaded dispatch ->
    wrapper -> stage -> evaluate, so SIGINT/SIGTERM stop the
    stage cooperatively.
    """
    try:
        from .experiments import dateline_analysis, dateline_stage

        search = result.get("search")
        names = tuple(fmt.name for fmt in formats)
        if (
            result.get("decision") in ("failed", "cancelled")
            or type(search) is not list
            or len(search) != len(formats)
        ):
            result["dateline"] = {
                "available": False,
                "detail": "native_search_incomplete",
            }
            return
        stage = dateline_stage(
            backend,
            formats,
            cases,
            search,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["dateline"] = dateline_analysis(
            stage,
            result.get("decision"),
            result.get("selected_candidate"),
            names,
        )
    except Exception:  # noqa: BLE001 -- dateline must not fail the run
        result["dateline"] = {
            "available": False,
            "detail": "dateline_unavailable",
        }


def _run_numform_probe(
    backend: Any,
    formats: Any,
    result: dict[str, Any],
    *,
    boundary: Any,
    boundary_map: Any,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Any,
) -> None:
    """Run the numeric surface-form probe; never raises.

    The probe is self-contained: its fixtures are generated from
    the run seed, so -- unlike the native-arm probes -- it
    consults neither the recorded search nor the primary
    decision, and a failed user run cannot silently suppress it.
    The run's cancel event is threaded dispatch -> wrapper ->
    stage -> evaluate, so SIGINT/SIGTERM stop the stage
    cooperatively.
    """
    try:
        from .experiments import numform_analysis, numform_stage

        names = tuple(fmt.name for fmt in formats)
        stage = numform_stage(
            backend,
            formats,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        result["numform"] = numform_analysis(stage, names)
    except Exception:  # noqa: BLE001 -- numform must not fail the run
        result["numform"] = {
            "available": False,
            "detail": "numform_unavailable",
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
                template_probe=args.template_probe,
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
        if not 1 <= args.max_drop_tokens <= 8:
            parser.error("--max-drop-tokens must be between 1 and 8")
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
            # skipcq: PY-W2000 - used on the next line; analyzer missed the use
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
            # skipcq: PY-W2000 - used two lines below; analyzer missed the use
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
                # Suspected-refusal profile (cycle 174, research
                # wave-6 idea 6): per-template suspected-refusal rate
                # on benign search fixtures from the recorded answers,
                # post-hoc (zero extra model calls). Informational
                # only -- never changes selection. Never fails the run.
                if args.refusals:
                    _run_refusals_probe(result)
                # Scorer-lens regrade (cycle 180, research wave-7
                # rank 1): per-candidate regrade of the recorded
                # search answers under the frozen normalisation
                # ladder, post-hoc (zero extra model calls).
                # Informational only -- never changes selection.
                # Never fails the run.
                if args.scorer_lens:
                    _run_scorerlens_probe(result, search_fixtures=tuple(split.search))
                # Candidate aliasing audit (cycle 181, research
                # wave-7 rank 2): token-identity equivalence classes
                # over the recorded search input_ids, post-hoc (zero
                # extra model calls). Informational only -- never
                # changes selection. Never fails the run.
                if args.aliasing:
                    _run_aliasing_probe(result)
                # Prompt-repetition probe (cycle 183, research wave-7
                # rank 4): x2 / pad generation arms against the
                # recorded native run, per-fixture degradation, the
                # run's cancel event threaded into the stage.
                # Informational only -- never changes selection.
                # Never fails the run.
                if args.repeat:
                    _run_repeat_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Distractor-dilution probe (cycle 184, research wave-7
                # rank 5): exact-dose filler arms against the recorded
                # native run, per-cell degradation, the run's cancel
                # event threaded into the stage. Informational only --
                # never changes selection. Never fails the run.
                if args.distract:
                    _run_distract_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Leading-claim caving probe (cycle 185, research
                # wave-7 rank 6): frozen claim arms against the
                # recorded native run, per-cell degradation, the
                # run's cancel event threaded into the stage.
                # Informational only -- never changes selection.
                # Never fails the run.
                if args.leading:
                    _run_leading_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Numeric surface-form probe (cycle 186, research
                # wave-7 rank 7): generated integer fixtures in
                # four written forms, self-contained (no native
                # arm, no consultation of the search or the
                # primary decision), the run's cancel event
                # threaded into the stage. Informational only --
                # never changes selection. Never fails the run.
                if args.numform:
                    _run_numform_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Date-line sensitivity probe (cycle 188, research
                # wave-8 rank 1): frozen date arms against the
                # recorded native run, per-cell degradation, the
                # run's cancel event threaded into the stage.
                # Informational only -- never changes selection.
                # Never fails the run.
                if args.dateline:
                    _run_dateline_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Multi-turn drift probe (cycle 189, research
                # wave-8 rank 2): generated shard fixtures in
                # full/sharded/scripted arms, self-generated
                # history screened before incorporation, the
                # run's cancel event threaded into the stage
                # loop. Informational only -- never changes
                # selection. Never fails the run.
                if args.multiturn:
                    _run_multiturn_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Needle-depth probe (cycle 191, research
                # wave-8 rank 3): generated code-recall cases
                # sized to exact rendered-token depths per
                # candidate, the run's cancel event threaded
                # into the stage loop. Informational only --
                # never changes selection. Never fails the run.
                if args.needle:
                    _run_needle_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Instruction placement probe (cycle 193, research
                # wave-8 rank 5): generated single-fact cases with
                # the question moved around a frozen short payload,
                # the run's cancel event threaded into the stage
                # loop. Informational only -- never changes
                # selection. Never fails the run.
                if args.placement:
                    _run_placement_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Canonical-form ladder probe (cycle 209, research
                # wave-10 rank 1): generated cases rendered under
                # NFC and NFD per candidate, the run's cancel event
                # threaded into the stage loop. Informational
                # only -- never changes selection. Never fails
                # the run.
                if args.canonical_form:
                    _run_canonical_form_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Negation probe (cycle 212, research wave-10
                # rank 3): generated compare pairs rendered
                # under direct, negated, flipped, and
                # descriptive negated_alt instructions per
                # candidate, the run's cancel event threaded
                # into the stage loop. Informational only --
                # never changes selection. Never fails the run.
                if args.negation:
                    _run_negation_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Instruction-distance probe (cycle 218,
                # research wave-11 rank 2): generated cases
                # rendered under adjacent, dose-separated,
                # dose-displaced, and order-control layouts of
                # one frozen format directive per candidate, the
                # run's cancel event threaded into the stage
                # loop. Informational only -- never changes
                # selection. Never fails the run.
                if args.instruction_distance:
                    _run_instruction_distance_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # History-frame probe (cycle 219, research
                # wave-11 rank 3): the run's search fixtures
                # preceded by the frozen synthetic history, with
                # only the prepended segment reframed in the
                # frozen chatml/llama3 framings per candidate,
                # the run's cancel event threaded into the stage
                # loop. Informational only -- never changes
                # selection. Never fails the run.
                if args.history_frame:
                    _run_history_frame_probe(
                        backend,
                        formats,
                        result,
                        fixtures=tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Prune-sweep probe (cycle 220, research
                # wave-11 rank 4): generated 24-case panel
                # whose passage context sentences lose seeded
                # interior words at exact doses 0/18/36/72,
                # with the dose-matched tail72 displaced
                # control and the closed_book screen arm, the
                # run's cancel event threaded into the stage
                # loop. Informational only -- never changes
                # selection. Never fails the run.
                if args.prune_sweep:
                    _run_prune_sweep_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Record-notation probe (cycle 221, payload-
                # serialisation line): generated 24-case code-
                # table panel rendered in two frozen notations
                # (prose sentences and canonical JSON) inside
                # the user message, plus the closed-book screen
                # arm, the run's cancel event threaded into the
                # stage loop. Descriptive only -- no
                # classification, no winner readout.
                # Informational only -- never changes selection.
                # Never fails the run.
                if args.record_notation:
                    _run_record_notation_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Context rotation probe (cycle 203, research
                # wave-9 rank 1): generated five-block cases whose
                # block span is cyclically rotated against a
                # matched mirror control, the run's cancel event
                # threaded into the stage loop. Informational
                # only -- never changes selection. Never fails
                # the run.
                if args.rotate:
                    _run_rotation_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Instruction-density ladder (cycle 204, research
                # wave-9 rank 3): generated cases whose constraint
                # prefix grows solo -> 2/4/8/16 against a length-
                # matched sham, the run's cancel event threaded
                # into the stage loop. Informational only -- never
                # changes selection. Never fails the run.
                if args.density:
                    _run_density_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Schema field-order probe (cycle 205, research
                # wave-9 rank 4): frozen flat-JSON specs in four
                # field orders, the answer field's endpoint
                # accuracy decided by an exact sign test over
                # adherent paired specs. Informational only --
                # never changes selection. Never fails the run.
                if args.field_order:
                    _run_fieldorder_probe(
                        backend,
                        formats,
                        result,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_context=args.max_context,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Choice-scoring audit (cycle 195, research
                # wave-8 rank 6): teacher-forced choice sets over
                # the recorded search prompts, audited against
                # the generation verdicts; a second opinion,
                # never a selection. Informational only -- never
                # changes selection. Never fails the run.
                if args.choice_audit:
                    _run_choiceaudit_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        cancel=cancel,
                    )
                # Load-precision probe (cycle 198, research
                # wave-8 rank 7): the same weights loaded in
                # bf16, every search candidate re-evaluated, the
                # selection re-applied over the paired
                # population. Informational only -- never
                # changes selection. Never fails the run.
                if args.precision:
                    try:
                        from .backends import LocalHFBackend as _PrecisionBackend

                        _bf16_backend: Any = _PrecisionBackend(
                            args.model,
                            device=args.device,
                            max_context=args.max_context,
                            dtype="bfloat16",
                        )
                    except Exception:  # noqa: BLE001 -- probe degrades, run stands
                        _bf16_backend = None
                    _run_precision_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        bf16_backend=_bf16_backend,
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                        cancel=cancel,
                    )
                # Verbalized-confidence probe (cycle 199, research
                # wave-8 rank 8): a frozen suffix elicits a stated
                # confidence line on every search fixture, parsed
                # under a frozen grammar and scored against the
                # exact-match verdicts, with a generated donor-key
                # control for knowability. Informational only --
                # never changes selection. Never fails the run.
                if args.confidence:
                    _run_confidence_probe(
                        backend,
                        formats,
                        result,
                        tuple(split.search),
                        boundary=args.boundary_ids,
                        boundary_map=boundary_map,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                        cancel=cancel,
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
                if args.sensitivity_sweep or args.token_depth_sweep:
                    from .experiments import (
                        prompt_sensitivity_sweep,
                        prompt_token_depth_sweep,
                    )
                    from .experiments import run_experiment as _diagnostics_run_experiment

                    # One native shared by both sweeps when both are
                    # requested: saving exactly one run_experiment
                    # versus two sweep calls (the CLI main run is a
                    # separate native and is not reused here). Each
                    # sweep alone recomputes its native, as before.
                    # Token mode inherits this block's
                    # diagnostics_unavailable fallback, documented in
                    # USAGE §8.53 -- a sweep failure degrades the
                    # diagnostics, never the run.
                    _shared_native = None
                    if args.sensitivity_sweep and args.token_depth_sweep:
                        _shared_native = _diagnostics_run_experiment(
                            backend,
                            formats,
                            split,
                            boundary=args.boundary_ids,
                            boundary_map=boundary_map,
                            max_new_tokens=args.max_new_tokens,
                            seed=args.seed,
                            cancel=cancel,
                        )
                    if args.sensitivity_sweep:
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
                            native=_shared_native,
                        )
                    if args.token_depth_sweep:
                        result["sensitivity_sweep_tokens"] = prompt_token_depth_sweep(
                            backend,
                            formats,
                            split,
                            boundary=args.boundary_ids,
                            boundary_map=boundary_map,
                            max_drop_tokens=args.max_drop_tokens,
                            max_new_tokens=args.max_new_tokens,
                            seed=args.seed,
                            cancel=cancel,
                            native=_shared_native,
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
