"""Docs accuracy audit (Cycle 46, 2026-10-15).

Audits every runnable command example in docs/USAGE.md and the command
sections of README.md against the CLI in model_protocol_lab/cli.py.

Method: every documented command name is checked against the argparse
surface; every documented flag is parsed; every representative example
invocation is run end-to-end. Commands that need a model run against a
scripted fake backend (char<->ord codec, deterministic; the same pattern
as tests/test_cycle17_sensitivity_sweep_verify.py) with
``model_protocol_lab.backends.LocalHFBackend`` patched out, so no real
weights, no network, no GPU are touched. Commands that need a checkpoint
directory run against a synthetic tokenizer-only checkpoint built in
setUpClass (config.json + a genuinely loadable tokenizer.json; a minimal
structurally-valid model.safetensors header for the doctor asset checks).

## Inaccuracies found (DOCUMENTED -> ACTUAL)

None material. Every runnable example verified runs as documented and
every documented flag/option behaves as described, except the two
precision notes below. This file does NOT fix the docs; the coordinator
decides.

## Precision notes (wording nits, not behavioral inaccuracies)

N1. docs/USAGE.md section 11 says "Input bounds (all enforced with fixed
    error codes, never silent truncation)". The 1 MB --prompt-text /
    --output-text caps are enforced through argparse (exit 2, fixed
    message "--prompt-text must be at most 1MB"), not through a
    fixed-vocabulary `*_too_long` code. The four parenthesized codes the
    docs name (output_ids_too_long, prompt_ids_too_long,
    stop_text_too_long, too_many_stop_texts) are all accurate -- they are
    raised verbatim by diffbackends._checked_ids/_checked_stop_texts.

N2. docs/USAGE.md section 10 shows the `.step 1,2,8,3` output elided as
    "...". The actual output is four lines, one per token:
    "token 3/4 id=8 overhang=1/1 boundary_at=2" etc. The example is
    abridged, not wrong; the pinned test asserts the real lines.

## Verified accurate (pinned below)

- Command names in USAGE.md sections 1-12 and README command sections:
  catalog, inspect, doctor, run, fixtures build, debug, diff-backends,
  diff-reports -- all present in the CLI.
- Documented flags for every command parse and round-trip to the parsed
  namespace (doctor --boundary-ids/--boundary-map/--reference-model/
  --portability-matrix; run --family/--mutate/--boundary-ids/
  --boundary-map/--max-new-tokens/--max-context/--seed/--device/--output/
  --include-evidence/--sensitivity-sweep/--max-drop-bytes/
  --logprob-diagnostic; fixtures build source/--output; debug
  --dataset/--model/--script; diff-backends --model/--prompt-text/
  --output-text/--stops/--eos-id/--min-tokens; diff-reports --all).
- USAGE section 5 exit codes: compatible -> 0, ambiguous/no_candidate ->
  2, failed -> 1, cancelled -> 130, argparse errors -> SystemExit(2).
- USAGE section 5 run template: defaults --max-new-tokens 64,
  --max-context 2048, --seed 0; --seed validated as unsigned 64-bit;
  --max-drop-bytes enforced 1-8; --boundary-ids grammar 1-16 ASCII IDs.
- USAGE section 4 --boundary-map: repeatable NAME=ID,ID; mistyped names
  fail fast listing valid names (chatml/prefix-newline,
  chatml/prefix-trim, kolibri/thinking-off, nemotron35/thinking-off,
  laguna/thinking-on confirmed in the mutate candidate set).
- USAGE section 2 doctor: exit 0/verdict ok on healthy inputs with a
  `suggestion` carrying the exact next `run` invocation (shell-quoted
  paths, FAMILY placeholder, --max-context from the declared window,
  --max-new-tokens 64); exit 1/verdict failed listing failing checks;
  --boundary-ids verified against the tokenizer vocabulary;
  --boundary-map checked per mapping; --reference-model reports
  `mismatch`; --portability-matrix JSON stays on stdout while the
  human-readable table goes to stderr.
- USAGE section 1 limits: 2 MB file, 100 fixtures/partition, 21
  messages, 16384 chars/field, regular-file requirement, empty-expected
  warning naming fixtures -- all verified.
- USAGE section 6 output shape: results.json carries decision,
  selected_candidate, search, budget_frontier (budget_robust,
  flip_ranges), selection_margin, token_overhead, template_conformance;
  report.html written; fresh output directory required.
- USAGE section 9 fixtures build: the exact documented CSV builds to a
  file that passes inspect; never overwrites; cross-partition group
  overlap and empty-cell-then-content are refused; audit sidecar written.
- USAGE section 10 debug: banner text, live prompt, .fixture by
  number/id, .render segment map (chars=91 on the documented fixture),
  .boundary/.step output lines, .families listing 29 families,
  abbreviations, # comments, .help PATTERN, .audit, .say/.render,
  .tokens with and without --model, .save, .read bounds (1 MB/10k
  lines/depth 5), EOF exit.
- USAGE section 11 diff-backends: the documented example reproduces the
  documented divergence record exactly
  (KEPT_VS_STRIPPED / "vllm_holdback_strip returned '42 ' while
  hf_stop_string kept '42 STOP'" / verdict "diverge"); top-level keys
  prompt_tokens/output_tokens/min_tokens/eos_id/stop_texts/cells/
  divergences/verdict; --eos-id default None, --min-tokens default 0.
- USAGE section 12 diff-reports: JSON to stdout with decision/winner/
  config_drift/candidate_verdicts/fixture_diffs/evidence/summary;
  human-readable summary to stderr; accepts report dirs or direct
  results.json paths; exit 0 on success, 1 on bad input.
- README: quick-start `catalog`/`inspect` invocations; the
  `model-protocol-lab` console script entry point; fixtures build never
  overwrites (README note); debug and diff-backends "Try" examples.

## Deliberately not verified (with reasons)

- `python -m pip install -e ".[dev]"` / `".[hf]"` (README, USAGE sec 7):
  installs packages; not re-verified in this offline environment.
- `python -m unittest discover -s tests -t .` (USAGE sec 7): the full
  suite; exceeds the ~5 minute budget for this audit file.
- `python examples/make_sample_report.py` (README): rewrites committed
  files under examples/sample-report/; running it would touch repo files
  beyond the single new test file this task allows. Existence and
  importability checked only.
- Real-model runs: no real checkpoint weights are available. `run`
  exercised a scripted fake backend; `doctor`/`diff-backends`/`debug
  --model` exercised synthetic tokenizer-only checkpoints.
- USAGE sec 5 exit 130 (cancellation) and Ctrl+C behavior: would need
  signal injection into a live run; not exercised.
- USAGE sec 8.2/8.5/8.9 research prose (e.g. the llama.cpp analysis,
  token-healing related work, lm-eval corroboration): literature claims,
  not runnable examples; out of scope.
- The `local_context_limit_exceeds_declared_window` fast-fail (USAGE
  sec 3): verified statically (backends.py raises the exact code); a
  real LocalHFBackend init needs torch + real weights.
"""

from __future__ import annotations

import io
import json
import re
import struct
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any
from unittest import mock

from model_protocol_lab import cli
from model_protocol_lab import core as real_core
from model_protocol_lab.catalog import CATALOG

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
FIXTURES_JSON = EXAMPLES / "fixtures.json"


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    """Run the CLI in-process; SystemExit becomes a return code."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
    return code, out.getvalue(), err.getvalue()


def write_minimal_safetensors(path: Path) -> None:
    """Structurally valid minimal safetensors (header only, no weights)."""
    header = {
        "__metadata__": {},
        "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00" * 4)


def write_word_tokenizer(root: Path, words: list[str]) -> None:
    """Genuinely loadable whitespace WordLevel tokenizer.json."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {word: idx for idx, word in enumerate(words)}
    vocab["[UNK]"] = len(vocab)
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["[UNK]", "[BOS]", "[EOS]"])
    tokenizer.save(str(root / "tokenizer.json"))


def make_synthetic_checkpoint(
    root: Path,
    *,
    weights: bool = True,
    n_positions: int | None = 1024,
    words: list[str] | None = None,
) -> Path:
    """Synthetic checkpoint dir: config.json + loadable tokenizer.json."""
    root.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"model_type": "gpt2"}
    if n_positions is not None:
        config["n_positions"] = n_positions
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if weights:
        write_minimal_safetensors(root / "model.safetensors")
    write_word_tokenizer(root, words or [f"tok{i}" for i in range(16)])
    return root


class ScriptedBackend:
    """Deterministic fake backend: char<->ord codec, scripted answers.

    generate() emits the fixture's expected answer (BLUE for the "Say
    BLUE" prompt, GREEN for "Say GREEN") followed by the <END> boundary
    marker, padded to the token cap -- a valid Generation with reason
    "limit". No model, no network.
    """

    fingerprint = "cycle46-scripted-backend"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        """Encode."""
        return tuple(ord(char) for char in text)

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return "".join(chr(token) for token in token_ids)

    @staticmethod
    def generate(
        input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Any:
        """Generate."""
        prompt = "".join(chr(token) for token in input_ids)
        answer = "BLUE" if "BLUE" in prompt else "GREEN"
        ids = tuple(ord(char) for char in answer + "<END>")
        padding = (ord("~"),) * (max_new_tokens - len(ids))
        return real_core.Generation(token_ids=ids + padding, reason="limit")


BOUNDARY_IDS = ",".join(str(ord(char)) for char in "<END>")

TINY_SPLIT = {
    "search": [
        {
            "id": "s1",
            "group": "g1",
            "messages": [{"role": "user", "content": "Say BLUE"}],
            "expected": "BLUE",
        }
    ],
    "held_out": [
        {
            "id": "h1",
            "group": "g2",
            "messages": [{"role": "user", "content": "Say GREEN"}],
            "expected": "GREEN",
        }
    ],
}


def patched_run(argv: list[str]) -> tuple[int, str, str]:
    """Run the `run` command with the scripted fake backend."""
    with mock.patch(
        "model_protocol_lab.backends.LocalHFBackend",
        lambda *args, **kwargs: ScriptedBackend(),
    ):
        return run_cli(argv)


class CommandNamesTest(unittest.TestCase):
    """Every command name in USAGE.md sections 1-12 + README exists."""

    DOCUMENTED_COMMANDS = [
        "catalog",  # USAGE sec 1, README quick start
        "inspect",  # USAGE sec 1, README quick start
        "doctor",  # USAGE sec 2, README quick start
        "run",  # USAGE sec 4-5
        "fixtures",  # USAGE sec 9, README
        "debug",  # USAGE sec 10, README
        "diff-backends",  # USAGE sec 11, README
        "diff-reports",  # USAGE sec 12
    ]

    def test_all_documented_commands_exist(self):
        """Test all documented commands exist."""
        parser = cli.build_parser()
        actions = parser._subparsers._group_actions[0]  # noqa: SLF001
        for command in self.DOCUMENTED_COMMANDS:
            with self.subTest(command=command):
                self.assertIn(command, actions.choices)

    def test_fixtures_build_subcommand_exists(self):
        """Test fixtures build subcommand exists."""
        args = cli.build_parser().parse_args(
            ["fixtures", "build", "cases.csv", "--output", "fixtures.json"]
        )
        self.assertEqual(args.fixtures_command, "build")
        self.assertEqual(str(args.source), "cases.csv")

    def test_console_script_entry_point(self):
        """Test console script entry point."""
        # README: "(a `model-protocol-lab` console script is installed as
        # an equivalent entry point.)"
        proc = subprocess.run(
            [sys.executable, "-m", "model_protocol_lab", "catalog"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=60,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(len(json.loads(proc.stdout)), len(CATALOG))


class DocumentedFlagsTest(unittest.TestCase):
    """Every documented flag parses and round-trips to the namespace."""

    def test_doctor_flags(self):
        """Test doctor flags."""
        args = cli.build_parser().parse_args(
            [
                "doctor",
                "--model",
                "path/to/checkpoint",
                "--dataset",
                "YOUR_FIXTURES.json",
                "--boundary-ids",
                "1,2,3",
                "--boundary-map",
                "chatml=1,2",
                "--reference-model",
                "REFERENCE",
                "--portability-matrix",
                "SECOND_CHECKPOINT",
            ]
        )
        self.assertEqual(str(args.model), "path/to/checkpoint")
        self.assertEqual(str(args.dataset), "YOUR_FIXTURES.json")
        self.assertEqual(args.boundary_ids, (1, 2, 3))
        self.assertEqual(args.boundary_map, [("chatml", (1, 2))])
        self.assertEqual(str(args.reference_model), "REFERENCE")
        self.assertEqual(str(args.portability_matrix), "SECOND_CHECKPOINT")

    def test_run_flags(self):
        """Test run flags."""
        # USAGE sec 5 template + sec 4/8.5/8.9 flags.
        args = cli.build_parser().parse_args(
            [
                "run",
                "--model",
                "LOCAL_MODEL_DIRECTORY",
                "--dataset",
                "YOUR_FIXTURES.json",
                "--family",
                "chatml",
                "--mutate",
                "--boundary-ids",
                "128001",
                "--boundary-map",
                "chatml/prefix-newline=128001",
                "--boundary-map",
                "chatml/prefix-trim=128002",
                "--max-new-tokens",
                "64",
                "--max-context",
                "2048",
                "--seed",
                "7",
                "--device",
                "cpu",
                "--output",
                "NEW_REPORT_DIRECTORY",
                "--include-evidence",
                "--sensitivity-sweep",
                "--max-drop-bytes",
                "3",
                "--logprob-diagnostic",
            ]
        )
        self.assertEqual(args.family, "chatml")
        self.assertTrue(args.mutate)
        self.assertEqual(args.boundary_ids, (128001,))
        self.assertEqual(
            args.boundary_map,
            [("chatml/prefix-newline", (128001,)), ("chatml/prefix-trim", (128002,))],
        )
        self.assertEqual(args.max_new_tokens, 64)
        self.assertEqual(args.max_context, 2048)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.include_evidence)
        self.assertTrue(args.sensitivity_sweep)
        self.assertEqual(args.max_drop_bytes, 3)
        self.assertTrue(args.logprob_diagnostic)

    def test_run_defaults_match_documented_template(self):
        """Test run defaults match documented template."""
        # USAGE sec 5: "--max-new-tokens 64 --max-context 2048"; "--seed
        # accepts an unsigned 64-bit integer (default 0)".
        args = cli.build_parser().parse_args(
            [
                "run",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                "o",
            ]
        )
        self.assertEqual(args.max_new_tokens, 64)
        self.assertEqual(args.max_context, 2048)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.device, "cpu")
        self.assertFalse(args.mutate)

    def test_debug_flags(self):
        """Test debug flags."""
        args = cli.build_parser().parse_args(
            [
                "debug",
                "--dataset",
                "examples/fixtures.json",
                "--model",
                "path/to/checkpoint",
                "--script",
                "session.txt",
            ]
        )
        self.assertEqual(str(args.dataset), "examples/fixtures.json")
        self.assertEqual(str(args.model), "path/to/checkpoint")
        self.assertEqual(str(args.script), "session.txt")

    def test_diff_backends_flags_and_defaults(self):
        """Test diff backends flags and defaults."""
        # USAGE sec 11: "--eos-id (default: none)", "--min-tokens (default 0)".
        args = cli.build_parser().parse_args(
            [
                "diff-backends",
                "--model",
                "path/to/checkpoint",
                "--prompt-text",
                "the answer is",
                "--output-text",
                "42 STOP hello world",
                "--stops",
                "STOP",
            ]
        )
        self.assertIsNone(args.eos_id)
        self.assertEqual(args.min_tokens, 0)
        args2 = cli.build_parser().parse_args(
            [
                "diff-backends",
                "--model",
                "m",
                "--prompt-text",
                "p",
                "--output-text",
                "o",
                "--stops",
                "STOP,</answer>",
                "--eos-id",
                "2",
                "--min-tokens",
                "1",
            ]
        )
        self.assertEqual(args2.eos_id, 2)
        self.assertEqual(args2.min_tokens, 1)

    def test_diff_reports_flags(self):
        """Test diff reports flags."""
        args = cli.build_parser().parse_args(
            ["diff-reports", "run-baseline", "run-fixed-template", "--all"]
        )
        self.assertEqual(str(args.old_results), "run-baseline")
        self.assertEqual(str(args.new_results), "run-fixed-template")
        self.assertTrue(args.all)

    def test_boundary_ids_grammar(self):
        """Test boundary ids grammar."""
        # USAGE sec 4/10: 1-16 ASCII token IDs.
        self.assertEqual(cli.parse_boundary("8"), (8,))
        self.assertEqual(cli.parse_boundary("1,2,3"), (1, 2, 3))
        self.assertEqual(cli.parse_boundary("2147483647"), (2147483647,))
        for bad in ("", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17", "12a", "-1"):
            with self.subTest(bad=bad), self.assertRaises(cli.argparse.ArgumentTypeError):
                cli.parse_boundary(bad)
        with self.assertRaises(cli.argparse.ArgumentTypeError):
            cli.parse_boundary("2147483648")  # 2**31, out of range


class CatalogInspectTest(unittest.TestCase):
    """USAGE sec 1 + README quick start: catalog / inspect."""

    def test_catalog_lists_29_families_with_framing_fields(self):
        """Test catalog lists 29 families with framing fields."""
        code, out, _ = run_cli(["catalog"])
        self.assertEqual(code, 0)
        families = json.loads(out)
        self.assertEqual(len(families), 29)
        self.assertEqual(len(families), len(CATALOG))
        names = {entry["name"] for entry in families}
        self.assertEqual(names, set(CATALOG))
        for entry in families:
            for field in (
                "name",
                "bos",
                "user_prefix",
                "assistant_prefix",
                "turn_end",
                "user_turn_end",
                "assistant_turn_end",
                "open_suffix",
            ):
                self.assertIn(field, entry)

    def test_inspect_examples_fixtures(self):
        """Test inspect examples fixtures."""
        code, out, _ = run_cli(["inspect", str(FIXTURES_JSON)])
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(
            report,
            {
                "search_cases": 2,
                "held_out_cases": 2,
                "valid": True,
                # The example fixtures are contamination-clean.
                "contamination": {"worst_tier": "clean", "pairs": []},
            },
        )

    def test_inspect_rejects_non_regular_file(self):
        """Test inspect rejects non regular file."""
        # USAGE sec 1: "The dataset path must be a regular file".
        with TemporaryDirectory() as tmp:
            code, _, err = run_cli(["inspect", tmp])
        self.assertEqual(code, 1)
        self.assertIn("not a regular file", err)

    def test_inspect_warns_on_empty_expected(self):
        """Test inspect warns on empty expected."""
        # USAGE sec 1: "`inspect` and `run` warn on stderr naming such
        # fixtures".
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "fx.json"
            path.write_text(
                json.dumps(
                    {
                        "search": [
                            {
                                "id": "vacuous-one",
                                "group": "g",
                                "messages": [{"role": "user", "content": "hi"}],
                                "expected": "",
                            }
                        ],
                        "held_out": [
                            {
                                "id": "h",
                                "group": "g2",
                                "messages": [{"role": "user", "content": "ho"}],
                                "expected": "x",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            code, _, err = run_cli(["inspect", str(path)])
        self.assertEqual(code, 0)
        self.assertIn("vacuous-one", err)
        self.assertIn("pass vacuously", err)

    def test_inspect_rejects_system_messages(self):
        """Test inspect rejects system messages."""
        # USAGE sec 1: "System messages, tools and assistant prefills are
        # not supported."
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "fx.json"
            path.write_text(
                json.dumps(
                    {
                        "search": [
                            {
                                "id": "s",
                                "group": "g",
                                "messages": [
                                    {"role": "system", "content": "be nice"},
                                    {"role": "user", "content": "hi"},
                                ],
                                "expected": "x",
                            }
                        ],
                        "held_out": [],
                    }
                ),
                encoding="utf-8",
            )
            code, _, err = run_cli(["inspect", str(path)])
        self.assertEqual(code, 1)
        self.assertIn("invalid fixture file", err)


# The exact authoring CSV from USAGE section 9.
DOCUMENTED_CSV = (
    "id,group,partition,expected,user_1,assistant_1,user_2\n"
    "search-copy-blue,copy-colors,search,BLUE,Reply with exactly BLUE and nothing else.,,\n"
    "held-number,held-integer,held_out,7,Return only the numeral 7.,,\n"
)


class FixturesBuildTest(unittest.TestCase):
    """USAGE sec 9 + README: `fixtures build`."""

    @staticmethod
    def _build(tmp: str, csv_text: str, name: str = "fixtures.json"):
        """Build."""
        csv_path = Path(tmp) / "cases.csv"
        csv_path.write_text(csv_text, encoding="utf-8")
        out_path = Path(tmp) / name
        code, out, err = run_cli(["fixtures", "build", str(csv_path), "--output", str(out_path)])
        return code, out, err, out_path

    def test_documented_csv_example_builds_and_passes_inspect(self):
        """Test documented csv example builds and passes inspect."""
        with TemporaryDirectory() as tmp:
            code, out, _, out_path = self._build(tmp, DOCUMENTED_CSV)
            self.assertEqual(code, 0)
            report = json.loads(out)
            self.assertEqual(report["search_cases"], 1)
            self.assertEqual(report["held_out_cases"], 1)
            self.assertTrue(out_path.exists())
            # The audit sidecar USAGE sec 9 implies (findings recorded in
            # a sidecar, never in the fixture file itself).
            self.assertTrue(Path(str(out_path) + ".audit.json").exists())
            code2, out2, _ = run_cli(["inspect", str(out_path)])
            self.assertEqual(code2, 0)
            self.assertEqual(
                json.loads(out2),
                {
                    "search_cases": 1,
                    "held_out_cases": 1,
                    "valid": True,
                    # The documented example fixtures are disjoint.
                    "contamination": {"worst_tier": "clean", "pairs": []},
                },
            )

    def test_build_never_overwrites(self):
        """Test build never overwrites."""
        # USAGE sec 9: "The output path is never overwritten"; README:
        # "(a fresh path: fixtures build never overwrites an existing file)".
        with TemporaryDirectory() as tmp:
            code, _, _, _ = self._build(tmp, DOCUMENTED_CSV)
            self.assertEqual(code, 0)
            code2, _, err2, _ = self._build(tmp, DOCUMENTED_CSV)
            self.assertEqual(code2, 1)
            self.assertIn("already exists", err2)

    def test_build_refuses_cross_partition_group_overlap(self):
        """Test build refuses cross partition group overlap."""
        # README: fixtures build "makes search/held-out leakage
        # structurally unbuildable".
        csv_text = (
            "id,group,partition,expected,user_1\n"
            "a,shared,search,BLUE,hi\n"
            "b,shared,held_out,7,hello\n"
        )
        with TemporaryDirectory() as tmp:
            code, _, err, out_path = self._build(tmp, csv_text)
            self.assertEqual(code, 1)
            self.assertIn("cross the split", err)
            self.assertFalse(out_path.exists())

    def test_build_rejects_empty_cell_followed_by_content(self):
        """Test build rejects empty cell followed by content."""
        # USAGE sec 9: "an empty cell followed by content is an error".
        csv_text = (
            "id,group,partition,expected,user_1,assistant_1,user_2\na,g,search,BLUE,hi,,later\n"
        )
        with TemporaryDirectory() as tmp:
            code, _, _err, out_path = self._build(tmp, csv_text)
            self.assertEqual(code, 1)
            self.assertFalse(out_path.exists())


class DoctorTest(unittest.TestCase):
    """USAGE sec 2 (+sec 8.4) and README: `doctor`."""

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        cls._tmp = TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.ckpt = make_synthetic_checkpoint(tmp / "ckpt")
        cls.ckpt2 = make_synthetic_checkpoint(tmp / "ckpt2")
        cls.ckpt_no_window = make_synthetic_checkpoint(tmp / "ckpt-nw", n_positions=None)
        # A second checkpoint with a genuinely different vocabulary, so the
        # cross-tokenizer boundary check has a real mismatch to report.
        cls.ckpt_other_vocab = make_synthetic_checkpoint(
            tmp / "ckpt-ov", words=["alpha", "beta", "gamma"]
        )

    @classmethod
    def tearDownClass(cls):
        """Tear down the test fixture."""
        cls._tmp.cleanup()

    @staticmethod
    def _doctor(*argv: str) -> tuple[int, dict, str]:
        """Doctor."""
        code, out, err = run_cli(["doctor", *argv])
        return code, json.loads(out), err

    def test_documented_invocation_passes_with_suggestion(self):
        """Test documented invocation passes with suggestion."""
        code, report, _ = self._doctor(
            "--model",
            str(self.ckpt),
            "--dataset",
            str(FIXTURES_JSON),
            "--boundary-ids",
            "1,2,3",
        )
        self.assertEqual(code, 0)
        self.assertEqual(report["verdict"], "ok")
        suggestion = report["suggestion"]
        self.assertIsNotNone(suggestion)
        # USAGE sec 2: the exact `run` invocation, shell-quoted paths,
        # explicit FAMILY placeholder, window-fitted limits.
        command = suggestion["next_command"]
        self.assertIn("python -m model_protocol_lab run", command)
        self.assertIn("--family FAMILY", command)
        self.assertIn("--max-context 1024", command)  # declared window
        self.assertIn("--max-new-tokens 64", command)
        self.assertIn("--boundary-ids 1,2,3", command)
        self.assertIn("--output NEW_REPORT_DIRECTORY", command)

    def test_suggestion_omitted_without_declared_window(self):
        """Test suggestion omitted without declared window."""
        # USAGE sec 2: "omitted whenever it could mislead ... or no
        # declared window".
        code, report, _ = self._doctor(
            "--model",
            str(self.ckpt_no_window),
            "--dataset",
            str(FIXTURES_JSON),
            "--boundary-ids",
            "1,2,3",
        )
        self.assertEqual(code, 0)
        self.assertEqual(report["verdict"], "ok")
        self.assertIsNone(report["suggestion"])

    def test_failing_checks_exit_1(self):
        """Test failing checks exit 1."""
        # USAGE sec 2: "Exit 0 means the run inputs look sane; exit 1
        # lists the failing checks."
        code, report, _ = self._doctor("--model", str(Path(self._tmp.name) / "missing"))
        self.assertEqual(code, 1)
        self.assertEqual(report["verdict"], "failed")
        failed = [
            name for name, check in report["model"]["checks"].items() if check["status"] == "failed"
        ]
        self.assertTrue(failed)

    def test_boundary_ids_checked_against_vocabulary(self):
        """Test boundary ids checked against vocabulary."""
        # USAGE sec 2: "verifies each ID against the tokenizer vocabulary".
        code, report, _ = self._doctor("--model", str(self.ckpt), "--boundary-ids", "1,999999")
        self.assertEqual(code, 1)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["model"]["checks"]["boundary_ids"]["status"], "failed")

    def test_boundary_map_checked_per_mapping(self):
        """Test boundary map checked per mapping."""
        # USAGE sec 4: "doctor --boundary-map NAME=ID,ID, which checks
        # each mapping".
        code, report, _ = self._doctor("--model", str(self.ckpt), "--boundary-map", "chatml=1,2")
        self.assertEqual(code, 0)
        self.assertEqual(report["model"]["checks"]["boundary_ids:chatml"]["status"], "ok")

    def test_reference_model_reports_mismatch(self):
        """Test reference model reports mismatch."""
        # USAGE sec 8.4: "Non-portability is reported as `mismatch`".
        _code, report, _ = self._doctor(
            "--model",
            str(self.ckpt),
            "--reference-model",
            str(self.ckpt_other_vocab),
            "--boundary-ids",
            "1,2,3",
        )
        self.assertEqual(report["model"]["checks"]["boundary_portability"]["status"], "mismatch")

    def test_portability_matrix_stdout_stays_parseable_json(self):
        """Test portability matrix stdout stays parseable json."""
        # USAGE sec 2: "the CLI also prints a readable table to stderr so
        # stdout stays parseable JSON".
        _code, out, err = run_cli(
            [
                "doctor",
                "--model",
                str(self.ckpt),
                "--portability-matrix",
                str(self.ckpt2),
            ]
        )
        report = json.loads(out)  # must not raise
        matrix = report["portability_matrix"]
        self.assertEqual(matrix["status"], "ok")
        self.assertTrue(len(matrix["rows"]) > 100)
        self.assertTrue(err.strip(), "expected the human-readable table on stderr")


class RunTest(unittest.TestCase):
    """USAGE sec 4-6 + 8.5/8.9: `run` against the scripted fake backend."""

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        cls._tmp = TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.ckpt = make_synthetic_checkpoint(tmp / "ckpt")
        cls.dataset = tmp / "tiny.json"
        cls.dataset.write_text(json.dumps(TINY_SPLIT), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        """Tear down the test fixture."""
        cls._tmp.cleanup()

    def _base_argv(self, output: Path, extra: list[str] | None = None) -> list[str]:
        """Base argv."""
        # USAGE sec 5 invocation template, placeholders filled.
        argv = [
            "run",
            "--model",
            str(self.ckpt),
            "--dataset",
            str(self.dataset),
            "--family",
            "chatml",
            "--mutate",
            "--boundary-ids",
            BOUNDARY_IDS,
            "--max-new-tokens",
            "16",
            "--max-context",
            "2048",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
        return argv + (extra or [])

    def test_documented_template_runs_end_to_end(self):
        """Test documented template runs end to end."""
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "report"
            code, out, _err = patched_run(self._base_argv(output))
            self.assertIn(code, (0, 2))
            summary = json.loads(out)
            self.assertIn("decision", summary)
            self.assertIn("selected_candidate", summary)
            self.assertTrue((output / "results.json").exists())
            self.assertTrue((output / "report.html").exists())
            result = json.loads((output / "results.json").read_text(encoding="utf-8"))
            # USAGE sec 6/8.x documented output shape.
            for key in (
                "decision",
                "search",
                "budget_frontier",
                "selection_margin",
                "token_overhead",
                "template_conformance",
            ):
                self.assertIn(key, result)
            # USAGE sec 4: the report records every candidate's boundary.
            for entry in result["search"]:
                self.assertIn("boundary_token_ids", entry)

    def test_exit_zero_on_compatible(self):
        """Test exit zero on compatible."""
        # USAGE sec 6: "0 for compatibility on the supplied fixture set".
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "report"
            argv = self._base_argv(output)
            argv.remove("--mutate")  # single candidate passes everything
            code, out, _ = patched_run(argv)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out)["decision"], "compatible_on_fixture_set")

    def test_exit_two_on_ambiguous(self):
        """Test exit two on ambiguous."""
        # USAGE sec 6: "2 for completed negative or ambiguous decisions".
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "report"
            code, out, _ = patched_run(self._base_argv(output))
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out)["decision"], "ambiguous")

    def test_diagnostic_flags_run_end_to_end(self):
        """Test diagnostic flags run end to end."""
        # USAGE sec 8.5 (--sensitivity-sweep/--max-drop-bytes), sec 8.9
        # (--logprob-diagnostic), sec 6 (--include-evidence).
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "report"
            code, _, _ = patched_run(
                self._base_argv(
                    output,
                    [
                        "--boundary-map",
                        f"chatml/prefix-newline={BOUNDARY_IDS}",
                        "--seed",
                        "7",
                        "--include-evidence",
                        "--sensitivity-sweep",
                        "--max-drop-bytes",
                        "1",
                        "--logprob-diagnostic",
                    ],
                )
            )
            self.assertIn(code, (0, 2))
            result = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertIsInstance(result["sensitivity_sweep"], dict)
            # The scripted backend offers no forward logits: the
            # diagnostic degrades honestly instead of failing the run.
            logprob = result["logprob_diagnostic"]
            self.assertFalse(logprob["available"])
            self.assertEqual(logprob["detail"], "logits_not_supported")
            for entry in result["search"]:
                self.assertIn("observations", entry)

    def test_argparse_errors_exit_2(self):
        """Test argparse errors exit 2."""
        # USAGE sec 6: "Argument-parser errors also return 2."
        code, _, _ = patched_run(["run"])
        self.assertEqual(code, 2)

    def test_seed_must_be_unsigned_64_bit(self):
        """Test seed must be unsigned 64 bit."""
        # USAGE sec 5: "--seed accepts an unsigned 64-bit integer ...
        # It is validated up front".
        with TemporaryDirectory() as tmp:
            argv = self._base_argv(Path(tmp) / "report", ["--seed", str(2**64)])
            code, _, err = patched_run(argv)
            self.assertEqual(code, 2)
            self.assertIn("Seed", err)

    def test_max_drop_bytes_range_enforced(self):
        """Test max drop bytes range enforced."""
        # USAGE sec 8.5 documents depths 1..--max-drop-bytes; the CLI
        # enforces 1-8.
        with TemporaryDirectory() as tmp:
            argv = self._base_argv(
                Path(tmp) / "report",
                ["--sensitivity-sweep", "--max-drop-bytes", "9"],
            )
            code, _, err = patched_run(argv)
            self.assertEqual(code, 2)
            self.assertIn("max-drop-bytes", err)

    def test_output_must_be_fresh(self):
        """Test output must be fresh."""
        # USAGE sec 6: "Output requires a new directory and will not
        # overwrite an existing one."
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "report"
            output.mkdir()
            code, _, err = patched_run(self._base_argv(output))
            self.assertEqual(code, 2)
            self.assertIn("already exists", err)

    def test_boundary_map_typo_fails_fast(self):
        """Test boundary map typo fails fast."""
        # USAGE sec 4: "A mistyped candidate name fails fast (the valid
        # names are listed)".
        with TemporaryDirectory() as tmp:
            argv = self._base_argv(Path(tmp) / "report", ["--boundary-map", "bogus=1"])
            code, _, err = patched_run(argv)
            self.assertEqual(code, 2)
            self.assertIn("chatml/prefix-newline", err)

    def test_run_warns_on_empty_expected(self):
        """Test run warns on empty expected."""
        # USAGE sec 1: "`inspect` and `run` warn on stderr naming such
        # fixtures".
        with TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "fx.json"
            dataset.write_text(
                json.dumps(
                    {
                        "search": [
                            {
                                "id": "vacuous-run",
                                "group": "g",
                                "messages": [{"role": "user", "content": "Say BLUE"}],
                                "expected": "",
                            }
                        ],
                        "held_out": [
                            {
                                "id": "h",
                                "group": "g2",
                                "messages": [{"role": "user", "content": "Say GREEN"}],
                                "expected": "GREEN",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            argv = self._base_argv(Path(tmp) / "report")
            argv[argv.index("--dataset") + 1] = str(dataset)
            code, _, err = patched_run(argv)
            self.assertIn(code, (0, 2))
            self.assertIn("vacuous-run", err)


class DebugTest(unittest.TestCase):
    """USAGE sec 10 + README: `debug`."""

    @staticmethod
    def _script_run(lines: list[str], model: Path | None = None):
        """Script run."""
        with TemporaryDirectory() as tmp:
            script = Path(tmp) / "session.txt"
            script.write_text("\n".join(lines) + "\n", encoding="utf-8")
            argv = ["debug", "--dataset", str(FIXTURES_JSON), "--script", str(script)]
            if model is not None:
                argv += ["--model", str(model)]
            return run_cli(argv)

    def test_documented_example_session(self):
        """Test documented example session."""
        # The verbatim session from USAGE sec 10 (the `.step` lines the
        # docs elide as "..." are asserted in full here).
        code, out, _ = self._script_run(
            [".fixture 1", ".render", ".boundary 8", ".step 1,2,8,3", ".quit"]
        )
        self.assertEqual(code, 0)
        self.assertIn("selected fixture 1: search-copy-blue", out)
        self.assertIn("render: family=chatml fixture=search-copy-blue messages=1 chars=91", out)
        self.assertIn("boundary set: [8]", out)
        self.assertIn("token 1/4 id=1 overhang=0/1 boundary_at=none", out)
        self.assertIn("token 2/4 id=2 overhang=0/1 boundary_at=none", out)
        self.assertIn("token 3/4 id=8 overhang=1/1 boundary_at=2", out)
        self.assertIn("token 4/4 id=3 overhang=0/1 boundary_at=2", out)

    def test_banner_and_live_prompt(self):
        """Test banner and live prompt."""
        # USAGE sec 10 documents the banner text and the
        # `mpl:<family> fixture-<n> >` prompt; feed via piped stdin.
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "model_protocol_lab",
                "debug",
                "--dataset",
                "examples/fixtures.json",
            ],
            input=".quit\n",
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=60,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        match = re.search(
            r'^version\s*=\s*"([^"]+)"', (REPO_ROOT / "pyproject.toml").read_text(), re.M
        )
        assert match is not None
        self.assertIn(f"model-protocol-lab debug {match.group(1)}", proc.stdout)
        self.assertIn(
            "dataset: examples/fixtures.json (4 fixtures) | family: chatml | "
            "tokenizer: none (rerun with --model PATH)",
            proc.stdout,
        )
        self.assertIn("Enter .help for usage hints.", proc.stdout)
        self.assertIn("mpl:chatml fixture-1 >", proc.stdout)

    def test_families_lists_29(self):
        """Test families lists 29."""
        # USAGE sec 10: "`.families` | List the 29 catalog families".
        code, out, _ = self._script_run([".families", ".quit"])
        self.assertEqual(code, 0)
        names = [line.strip() for line in out.splitlines() if line.strip()]
        self.assertEqual(len(names), 29)
        self.assertEqual(set(names), set(CATALOG))

    def test_abbreviations_comments_and_help_search(self):
        """Test abbreviations comments and help search."""
        # USAGE sec 10: "unambiguous abbreviations like `.help ren`,
        # `.help PATTERN` search, `#` comments".
        code, out, _ = self._script_run(["# a comment line", ".fam", ".ren", ".help ren", ".quit"])
        self.assertEqual(code, 0)
        self.assertIn("chatml", out)  # .fam abbreviation worked
        self.assertIn("chars=91", out)  # .ren abbreviation worked
        self.assertIn(".render: Render the hypothesis over messages", out)

    def test_fixture_by_id_audit_and_say(self):
        """Test fixture by id audit and say."""
        code, out, _ = self._script_run(
            [
                ".fixture search-copy-blue",
                ".audit",
                ".say",
                "user: hi",
                ".",
                ".render",
                ".quit",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("selected fixture 1: search-copy-blue", out)
        self.assertIn("audit: 0 finding(s)", out)
        self.assertIn("buffered 1 message(s)", out)
        self.assertIn("buffered messages=1", out)

    def test_tokens_with_and_without_model(self):
        """Test tokens with and without model."""
        # USAGE sec 10: "plus exact token IDs per segment when --model is
        # given"; without --model a reminder prints instead of IDs.
        code, out, _ = self._script_run([".render", ".tokens", ".quit"])
        self.assertEqual(code, 0)
        self.assertIn("no tokenizer attached", out)
        with TemporaryDirectory() as tmp:
            ckpt = make_synthetic_checkpoint(Path(tmp) / "ckpt")
            code2, out2, _ = self._script_run([".render", ".tokens", ".quit"], model=ckpt)
        self.assertEqual(code2, 0)
        self.assertIn("tokens=[", out2)
        self.assertIn("ids=[", out2)

    def test_save_writes_transcript(self):
        """Test save writes transcript."""
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "transcript.txt"
            code, _out, _ = self._script_run([".render", f".save {target}", ".quit"])
            self.assertEqual(code, 0)
            self.assertTrue(target.exists())
            self.assertIn("render: family=chatml", target.read_text(encoding="utf-8"))

    def test_script_bounds_refused(self):
        """Test script bounds refused."""
        # USAGE sec 10: ".read ... refused over 1 MB or 10k lines".
        with TemporaryDirectory() as tmp:
            script = Path(tmp) / "long.txt"
            script.write_text(".help\n" * 10001, encoding="utf-8")
            code, _, err = run_cli(
                [
                    "debug",
                    "--dataset",
                    str(FIXTURES_JSON),
                    "--script",
                    str(script),
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("10000", err)


class DiffBackendsTest(unittest.TestCase):
    """USAGE sec 11 + README: `diff-backends`."""

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        cls._tmp = TemporaryDirectory()
        # Tokenizer that really encodes the documented example words, so
        # the documented divergence is reproducible.
        cls.ckpt = make_synthetic_checkpoint(
            Path(cls._tmp.name) / "ckpt",
            words=["the", "answer", "is", "42", "STOP", "hello", "world", ",", "."],
        )

    @classmethod
    def tearDownClass(cls):
        """Tear down the test fixture."""
        cls._tmp.cleanup()

    @staticmethod
    def _run(*argv: str) -> tuple[int, dict, str]:
        """Run."""
        code, out, err = run_cli(["diff-backends", *argv])
        return code, (json.loads(out) if out.strip() else {}), err

    def test_documented_example_reproduces_documented_divergence(self):
        """Test documented example reproduces documented divergence."""
        # The compact real example from USAGE sec 11.
        code, report, _ = self._run(
            "--model",
            str(self.ckpt),
            "--prompt-text",
            "the answer is",
            "--output-text",
            "42 STOP hello world",
            "--stops",
            "STOP",
        )
        self.assertEqual(code, 0)
        # USAGE sec 11: "the top level carries `prompt_tokens`,
        # `output_tokens`, `min_tokens`, `eos_id`, `stop_texts`, `cells`,
        # `divergences` and the `verdict`".
        self.assertEqual(
            set(report),
            {
                "prompt_tokens",
                "output_tokens",
                "min_tokens",
                "eos_id",
                "stop_texts",
                "cells",
                "divergences",
                "verdict",
            },
        )
        self.assertEqual(report["stop_texts"], ["STOP"])
        self.assertEqual(report["min_tokens"], 0)
        self.assertIsNone(report["eos_id"])
        self.assertEqual(report["verdict"], "diverge")
        (divergence,) = report["divergences"]
        self.assertEqual(
            divergence,
            {
                "class": "KEPT_VS_STRIPPED",
                "stop_text": "STOP",
                "detail": (
                    "vllm_holdback_strip returned '42 ' while hf_stop_string kept '42 STOP'"
                ),
            },
        )

    def test_eos_id_and_min_tokens_flags(self):
        """Test eos id and min tokens flags."""
        code, report, _ = self._run(
            "--model",
            str(self.ckpt),
            "--prompt-text",
            "the answer is",
            "--output-text",
            "42 STOP",
            "--stops",
            "STOP",
            "--eos-id",
            "17",
            "--min-tokens",
            "1",
        )
        self.assertEqual(code, 0)
        self.assertEqual(report["eos_id"], 17)
        self.assertEqual(report["min_tokens"], 1)

    def test_prompt_text_capped_at_1mb(self):
        """Test prompt text capped at 1mb."""
        # USAGE sec 11: "`--prompt-text`/`--output-text` are capped at 1 MB
        # each ... never silent truncation".
        code, _, err = run_cli(
            [
                "diff-backends",
                "--model",
                str(self.ckpt),
                "--prompt-text",
                "x" * (1024 * 1024 + 1),
                "--output-text",
                "hi",
                "--stops",
                "STOP",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("--prompt-text must be at most 1MB", err)


class DiffReportsTest(unittest.TestCase):
    """USAGE sec 12: `diff-reports`."""

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        cls._tmp = TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        ckpt = make_synthetic_checkpoint(tmp / "ckpt")
        dataset = tmp / "tiny.json"
        dataset.write_text(json.dumps(TINY_SPLIT), encoding="utf-8")
        with mock.patch(
            "model_protocol_lab.backends.LocalHFBackend",
            lambda *args, **kwargs: ScriptedBackend(),
        ):
            code, _, _ = run_cli(
                [
                    "run",
                    "--model",
                    str(ckpt),
                    "--dataset",
                    str(dataset),
                    "--family",
                    "chatml",
                    "--boundary-ids",
                    BOUNDARY_IDS,
                    "--max-new-tokens",
                    "16",
                    "--output",
                    str(tmp / "run-baseline"),
                ]
            )
            assert code == 0
        # A second report with a deliberately changed decision, mimicking
        # the docs' baseline-vs-fixed narrative.
        import shutil

        shutil.copytree(tmp / "run-baseline", tmp / "run-fixed")
        results = tmp / "run-fixed" / "results.json"
        payload = json.loads(results.read_text(encoding="utf-8"))
        payload["decision"] = "no_candidate"
        payload["selected_candidate"] = None
        results.write_text(json.dumps(payload), encoding="utf-8")
        cls.baseline = tmp / "run-baseline"
        cls.fixed = tmp / "run-fixed"

    @classmethod
    def tearDownClass(cls):
        """Tear down the test fixture."""
        cls._tmp.cleanup()

    def test_stdout_json_stderr_summary_shape(self):
        """Test stdout json stderr summary shape."""
        # USAGE sec 12: "The JSON comparison goes to stdout; a
        # human-readable summary goes to stderr."
        code, out, err = run_cli(["diff-reports", str(self.fixed), str(self.baseline)])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        for key in (
            "decision",
            "winner",
            "config_drift",
            "candidate_verdicts",
            "fixture_diffs",
            "evidence",
            "summary",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["decision"]["old"], "no_candidate")
        self.assertEqual(payload["decision"]["new"], "compatible_on_fixture_set")
        self.assertIn("Decision: no_candidate -> compatible_on_fixture_set", err)
        self.assertIn("Summary:", err)

    def test_direct_results_json_paths_and_config_drift(self):
        """Test direct results json paths and config drift."""
        code, _, err = run_cli(
            [
                "diff-reports",
                str(self.fixed / "results.json"),
                str(self.baseline / "results.json"),
                "--all",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("diff-reports:", err)

    def test_identical_inputs_report_unchanged(self):
        """Test identical inputs report unchanged."""
        code, _, err = run_cli(["diff-reports", str(self.baseline), str(self.baseline)])
        self.assertEqual(code, 0)
        self.assertIn("No differences", err)
        self.assertIn("identical on every compared axis", err)

    def test_missing_input_exits_1(self):
        """Test missing input exits 1."""
        # USAGE sec 12: "Exit 1 means an input problem, with a fixed error
        # code on stderr".
        code, _, err = run_cli(["diff-reports", "/nonexistent-a", "/nonexistent-b"])
        self.assertEqual(code, 1)
        self.assertIn("results_path_missing", err)


class ReadmeArtifactsTest(unittest.TestCase):
    """README command-adjacent artifacts that must exist."""

    def test_make_sample_report_script_exists(self):
        """Test make sample report script exists."""
        # README: "python examples/make_sample_report.py". Not executed
        # here: it rewrites committed files under examples/sample-report/.
        script = EXAMPLES / "make_sample_report.py"
        self.assertTrue(script.exists())
        compile(script.read_text(encoding="utf-8"), str(script), "exec")

    def test_sample_report_committed(self):
        """Test sample report committed."""
        self.assertTrue((EXAMPLES / "sample-report" / "report.html").exists())

    def test_fixtures_csv_example_exists(self):
        """Test fixtures csv example exists."""
        # README: "python -m model_protocol_lab fixtures build
        # examples/fixtures.csv --output /tmp/mpl-fixtures.json".
        self.assertTrue((EXAMPLES / "fixtures.csv").exists())


if __name__ == "__main__":
    unittest.main()
