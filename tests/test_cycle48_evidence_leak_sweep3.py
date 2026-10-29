"""Cycle 48, sweep 3: canary evidence-leak sweep of every public output path.

Fresh canary: ``CANARY48-LEAKSWEEP3-ZK2`` (unique to this sweep; pure ASCII
with no HTML-significant characters, so ``escape()`` leaves it
byte-identical and absence in HTML is a meaningful negative). Cycles 24
and 43 swept clean; this third sweep re-pins every public report surface
after the catalog growth and reporting changes that landed since.

Methodology: the canary is planted in fixture message content, expected
answers, probe-backend generations, and prompt-decodable token-ID arrays.
Every public surface is exercised with ``include_evidence=False`` (or CLI
defaults) and the canary must appear in NONE of its four leak forms -- raw
text, JSON-escaped text, the standalone token-ID array, and the bracketless
token-ID comma sequence (the ID forms whitespace-squashed, so the canary is
caught however the serializer wraps it, including embedded inside a longer
ID array). Every ``assertNoCanary`` is paired with a guard proving the
scan is not vacuous (the canary provably sits in the carriers, or is kept
with ``include_evidence=True``).

Per-surface verdicts (each pinned below):

- ``save_report`` results.json with include_evidence=False: CLEAN.
- ``save_report`` report.html: CLEAN (``render_html`` withholds raw
  evidence from the HTML view even in evidence mode).
- ``mpl run --output`` (real CLI against a toy 1-layer GPT-2 checkpoint
  built in-test): results.json, report.html, stdout and stderr CLEAN with
  CLI defaults; the ``--include-evidence`` control run proves the canary
  was in the carriers (decoded input_ids contain it).
- ``diff_reports`` JSON + ``render_summary`` + ``mpl diff-reports`` CLI:
  CLEAN. There is no HTML render for diff-reports (stdout JSON + stderr
  summary only) -- documented here, not swept.
- ``mpl debug`` saved transcript: a content-free session's transcript is
  CLEAN; a session where the operator explicitly ran ``.render`` carries
  the canary BY DESIGN (the operator's own explicit render, faithfully
  echoed; USAGE.md section 10 warns not to paste transcripts into public
  issues).
- ``doctor()`` report dict + ``mpl doctor`` stdout: CLEAN.
- Committed ``examples/sample-report/`` results.json + report.html: CLEAN.

Mutation gate (run separately via /tmp/cycle48_mutants/run_mutations.py --
a /tmp copy of the package is mutated; the repo itself is never mutated):

- M1: ``_strip_raw_diagnostics`` bypassed -> the barrier tests fail.
- M2: ``DebugShell._cmd_save`` leaks prompt-decodable token IDs into the
  saved transcript -> the debug transcript test fails.
- M3: ``diff_reports`` leaks a decoded expected answer -> the diff-reports
  tests fail.
- Control: the unmutated copy passes the whole file.

Deterministic: no network, no model downloads. The only weights built are
a toy 1-layer GPT-2 created in-test with torch (CPU; milliseconds per
forward pass); the probe backend is pure Python.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Event

from model_protocol_lab.catalog import candidates
from model_protocol_lab.core import Generation
from model_protocol_lab.datasets import parse_dataset
from model_protocol_lab.debug import DebugShell
from model_protocol_lab.diffreports import diff_reports, render_summary
from model_protocol_lab.doctor import doctor
from model_protocol_lab.experiments import run_experiment
from model_protocol_lab.reporting import _public_view, save_report

# Under the mutation gate the test file runs from /tmp/cycle48_mutants, so
# the repo root (sample report, default CLI cwd) comes from the environment.
REPO_ROOT = Path(os.environ.get("MPL_REPO_ROOT", Path(__file__).resolve().parent.parent))

CANARY = "CANARY48-LEAKSWEEP3-ZK2"
# The probe backend encodes text as code points. CANARY_ID_ARRAY is the
# exact standalone array; CANARY_ID_SEQ is the bracketless comma sequence,
# which also catches the canary embedded inside a longer ID array (e.g.
# input_ids, where the prompt's framing IDs surround the canary's).
CANARY_ID_SEQ = ",".join(str(ord(char)) for char in CANARY)
CANARY_ID_ARRAY = "[" + CANARY_ID_SEQ + "]"
# Generations must be long enough to emit the whole 22-char canary, or an
# observations leak would carry only a truncated prefix and the content
# scan could not catch it.
MAX_NEW_TOKENS = 32
# The fixed replacement _strip_raw_diagnostics writes over free-text detail.
WITHHELD = "[withheld without --include-evidence]"
# The fixed paragraph render_html emits for the divergence section.
HTML_WITHHELD = "[stop text and diagnostic detail withheld from this HTML view]"


def _no_ws(text: str) -> str:
    """No ws."""
    return "".join(text.split())


# skipcq: PTC-W0046 - shared base: subclasses carry the test methods; the base runs none by design
class LeakAsserts(unittest.TestCase):
    """Shared canary scanners; not a test case itself (no test_ methods)."""

    def assertNoCanary(self, text: str, surface: str) -> None:  # noqa: N802 -- unittest-style name
        """The canary appears in none of its leak forms in ``text``."""
        self.assertNotIn(CANARY, text, f"raw canary leaked into {surface}")
        self.assertNotIn(
            json.dumps(CANARY)[1:-1],
            text,
            f"escaped canary leaked into {surface}",
        )
        squashed = _no_ws(text)
        self.assertNotIn(
            CANARY_ID_ARRAY,
            squashed,
            f"canary token-ID array leaked into {surface}",
        )
        self.assertNotIn(
            CANARY_ID_SEQ,
            squashed,
            f"canary token-ID sequence leaked into {surface}",
        )
        # Whitespace-normalized raw form: some decoders join tokens with
        # spaces (the toy BPE below does), so a decoded leak can surface as
        # e.g. "CANARY48 - LEAKSWEEP3 - ZK2". Stripping all whitespace first
        # catches the content however the decoder spaced it.
        self.assertNotIn(
            CANARY,
            squashed,
            f"whitespace-normalized canary leaked into {surface}",
        )

    def assertCanaryPresent(self, text: str, surface: str) -> None:
        """Guard: the scan is not vacuous -- the canary IS in ``text``."""
        self.assertIn(
            CANARY,
            text,
            f"expected canary in {surface} (vacuity guard)",
        )


class _CanaryBackend:
    """Ord-based probe backend; generations echo the canary plus boundary."""

    fingerprint = "canary48-probe-backend-v1"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        """Encode."""
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return "".join(map(chr, token_ids))

    @staticmethod
    def generate(
        input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        """Generate."""
        text = (CANARY + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


def _canary_ids() -> list[int]:
    """Canary ids."""
    return [ord(char) for char in CANARY]


def _hostile_document() -> dict:
    """A fixture document whose user content (not ids) carries the canary."""
    return {
        "search": [
            {
                "id": "c48-search-1",
                "group": "c48-g1",
                "messages": [{"role": "user", "content": f"Say the phrase {CANARY} out loud"}],
                "expected": f"the answer is {CANARY}",
            },
            {
                "id": "c48-search-2",
                "group": "c48-g1",
                "messages": [{"role": "user", "content": f"Repeat {CANARY} twice"}],
                "expected": "something else entirely",
            },
        ],
        "held_out": [
            {
                "id": "c48-held-1",
                "group": "c48-g2",
                "messages": [{"role": "user", "content": f"held-out says {CANARY}"}],
                "expected": f"held answer {CANARY}",
            },
        ],
    }


def _dataset_path(directory: Path, document: dict | None = None) -> Path:
    """Dataset path."""
    path = directory / "dataset.json"
    path.write_text(json.dumps(document or _hostile_document()), encoding="utf-8")
    return path


def _graft_carriers(result: dict) -> dict:
    """Attach every known raw-evidence carrier, all holding the canary.

    ``template_conformance`` hand/template ID arrays decode to the prompt;
    ``backend_divergences`` stop_texts/detail carry decoded answers and raw
    generations. The schema mirrors exactly what ``doctor`` and
    ``divergence_section_data`` emit, so the barrier is tested against the
    real shapes.
    """
    canary_ids = _canary_ids()
    result["template_conformance"] = {
        "status": "ok",
        "diffs": [
            {
                "format": "llama3",
                "fixture": "c48-search-1",
                "status": "ok",
                "hand_tokens": len(canary_ids),
                "template_tokens": len(canary_ids) + 2,
                "diff": [
                    {
                        "hand_span": [0, len(canary_ids)],
                        "template_span": [0, len(canary_ids) + 2],
                        "hand_ids": canary_ids,
                        "template_ids": canary_ids,
                    }
                ],
            }
        ],
    }
    result["backend_divergences"] = {
        "verdict": "diverge",
        "pairs_verdict": "HF vs vLLM vs SGLang stop semantics: 0/1 agree; 1 divergence",
        "per_fixture": [
            {
                "fixture": "c48-search-1",
                "verdict": "diverge",
                "divergences": [
                    {
                        "class": "KEPT_VS_STRIPPED",
                        "stop_text": CANARY,
                        "detail": f"raw generated text before the stop: {CANARY} tail",
                        "severity": "high",
                    }
                ],
            }
        ],
        "matrix": {"KEPT_VS_STRIPPED": {"hf_stop_string": 1}},
        "severities": {"KEPT_VS_STRIPPED": "high"},
        "stop_texts": [CANARY],
        "eos_id": None,
        "min_tokens": 0,
        "observations_truncated": False,
        "observations_total": 1,
        "observations_processed": 1,
    }
    return result


def _hostile_result() -> dict:
    """A real run_experiment result over the hostile document, carriers grafted."""
    split = parse_dataset(json.dumps(_hostile_document()))
    result = run_experiment(
        _CanaryBackend(),
        candidates("llama3", mutate=False)[:1],
        split,
        boundary=(124,),
        max_new_tokens=MAX_NEW_TOKENS,
        seed=7,
    )
    return _graft_carriers(result)


def _walk_keys(value: object, seen: set[str]) -> None:
    """Walk keys."""
    if isinstance(value, dict):
        for key, item in value.items():
            seen.add(str(key))
            _walk_keys(item, seen)
    elif isinstance(value, list):
        for item in value:
            _walk_keys(item, seen)


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run the CLI; the cwd defaults to the repo root but the mutation gate
    overrides it via MPL_TEST_CWD so CLI subprocesses import the mutated
    package copy through PYTHONPATH shadowing.
    """
    return subprocess.run(
        [sys.executable, "-m", "model_protocol_lab", *args],
        cwd=cwd or Path(os.environ.get("MPL_TEST_CWD", REPO_ROOT)),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


class PublicViewBarrier48(LeakAsserts):
    """The _public_view / _strip_raw_diagnostics barrier over every carrier."""

    def test_public_view_strips_every_carrier(self) -> None:
        """Test public view strips every carrier."""
        result = _hostile_result()
        public = _public_view(result, include_evidence=False)
        text = json.dumps(public, ensure_ascii=True)
        self.assertNoCanary(text, "public results.json (include_evidence=False)")
        self.assertFalse(public["raw_evidence_included"])
        # Observations are gone from both stages...
        for run in public["search"]:
            self.assertNotIn("observations", run)
        held = public.get("held_out")
        if type(held) is dict:
            self.assertNotIn("observations", held)
        # ...prompt-reconstructing token-ID arrays are gone...
        diffs = public["template_conformance"]["diffs"]
        for entry in diffs:
            for segment in entry["diff"]:
                with self.subTest(segment=segment):
                    self.assertNotIn("hand_ids", segment)
                    self.assertNotIn("template_ids", segment)
                    # Spans (positions, not content) stay on the record.
                    self.assertIn("hand_span", segment)
        # ...decoded stop texts are gone...
        divergences = public["backend_divergences"]
        self.assertNotIn("stop_texts", divergences)
        for entry in divergences["per_fixture"]:
            for finding in entry["divergences"]:
                with self.subTest(finding=finding):
                    self.assertNotIn("stop_text", finding)
                    # ...and free-text detail is replaced, never dropped
                    # silently (the fixed class/severity stay).
                    self.assertEqual(finding["detail"], WITHHELD)
                    self.assertEqual(finding["class"], "KEPT_VS_STRIPPED")
        # The fixed-vocabulary aggregates survive the strip.
        self.assertEqual(divergences["matrix"], {"KEPT_VS_STRIPPED": {"hf_stop_string": 1}})

    def test_public_view_true_keeps_evidence_by_design(self) -> None:
        """Non-vacuity: with include_evidence=True the canary IS present."""
        result = _hostile_result()
        public = _public_view(result, include_evidence=True)
        text = json.dumps(public, ensure_ascii=True)
        self.assertCanaryPresent(text, "public results.json (include_evidence=True)")
        self.assertTrue(public["raw_evidence_included"])
        self.assertIn("observations", public["search"][0])
        self.assertIn("stop_texts", public["backend_divergences"])
        # The prompt-decodable ID arrays are kept too, by design.
        segment = public["template_conformance"]["diffs"][0]["diff"][0]
        self.assertEqual(segment["hand_ids"], _canary_ids())

    def test_save_report_json_and_html_false_path(self) -> None:
        """Test save report json and html false path."""
        with tempfile.TemporaryDirectory() as tmp:
            json_path, html_path = save_report(
                _hostile_result(), Path(tmp) / "report", include_evidence=False
            )
            json_text = json_path.read_text(encoding="utf-8")
            html_text = html_path.read_text(encoding="utf-8")
        self.assertNoCanary(json_text, "results.json (include_evidence=False)")
        self.assertNoCanary(html_text, "report.html (include_evidence=False)")
        # Non-vacuous: the evidence-carrying sections did render, just
        # without content -- fixture ids appear in the outcome grid, the
        # divergence section shows its withheld paragraph, and the
        # conformance section shows its divergent-segment badge.
        self.assertIn("c48-search-1", html_text)
        self.assertIn(HTML_WITHHELD, html_text)
        self.assertIn("divergent segment(s)", html_text)
        self.assertIn("c48-search-1", json_text)

    def test_save_report_true_path_html_still_withholds(self) -> None:
        """The HTML view withholds raw evidence even in evidence mode: the
        flag only affects the JSON.
        """
        with tempfile.TemporaryDirectory() as tmp:
            json_path, html_path = save_report(
                _hostile_result(), Path(tmp) / "report", include_evidence=True
            )
            json_text = json_path.read_text(encoding="utf-8")
            html_text = html_path.read_text(encoding="utf-8")
        self.assertCanaryPresent(json_text, "results.json (include_evidence=True)")
        self.assertNoCanary(html_text, "report.html (include_evidence=True)")


def _write_tiny_checkpoint(root: Path):
    """Toy GPT-2 checkpoint for the real CLI run path.

    One layer, tiny hidden size, CPU-fast; the BPE is trained on synthetic
    text that includes the canary word, so the canary round-trips through
    the checkpoint's own tokenizer losslessly.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    root.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(
        ["hello world foo bar", CANARY, f"the answer is {CANARY}"],
        BpeTrainer(special_tokens=["[UNK]", "[BOS]", "[EOS]"], vocab_size=64),
    )
    tok.save(str(root / "tokenizer.json"))
    vocab = tok.get_vocab()

    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=64,
        n_positions=512,
        n_embd=32,
        n_layer=1,
        n_head=2,
        bos_token_id=vocab["[BOS]"],
        eos_token_id=vocab["[EOS]"],
    )
    (root / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    from safetensors.torch import save_model

    save_model(GPT2LMHeadModel(config), str(root / "model.safetensors"))
    return root, tok


def _canary_word_ids(tok) -> list[int]:
    """The canary word's exact BPE id sequence from the toy checkpoint."""
    return tok.encode(CANARY).ids


class CliRunSweep48(LeakAsserts):
    """``mpl run --output``: the real CLI end to end with CLI defaults."""

    def _cli_run(self, tmp: Path, include_evidence: bool):
        """Cli run."""
        ckpt, tok = _write_tiny_checkpoint(tmp / "ckpt")
        dataset = _dataset_path(tmp)
        out = tmp / ("out-true" if include_evidence else "out-false")
        args = [
            "run",
            "--model",
            str(ckpt),
            "--dataset",
            str(dataset),
            "--family",
            "llama3",
            "--boundary-ids",
            "5",
            "--max-new-tokens",
            "8",
            "--max-context",
            "512",
            "--seed",
            "0",
            "--output",
            str(out),
        ]
        if include_evidence:
            args.append("--include-evidence")
        proc = _run_cli(*args)
        # 0 = compatible_on_fixture_set, 2 = any other non-failed decision.
        self.assertIn(proc.returncode, (0, 2), proc.stderr[-3000:])
        return proc, out, tok

    def test_cli_run_false_path_outputs_clean(self) -> None:
        """Test cli run false path outputs clean."""
        with tempfile.TemporaryDirectory() as tmp:
            proc, out, tok = self._cli_run(Path(tmp), include_evidence=False)
            json_text = (out / "results.json").read_text(encoding="utf-8")
            html_text = (out / "report.html").read_text(encoding="utf-8")
        self.assertNoCanary(json_text, "CLI run results.json (CLI defaults)")
        self.assertNoCanary(html_text, "CLI run report.html (CLI defaults)")
        # Token-ID form from the run's own tokenizer: the canary word's
        # exact BPE id sequence must not appear, however wrapped.
        seq = ",".join(str(i) for i in _canary_word_ids(tok))
        self.assertNotIn(
            seq,
            _no_ws(json_text),
            "canary BPE token-ID sequence leaked into CLI results.json",
        )
        self.assertNoCanary(proc.stdout, "CLI run stdout")
        self.assertNoCanary(proc.stderr, "CLI run stderr")
        # Non-vacuous: a real report was written for the hostile fixtures.
        report = json.loads(json_text)
        self.assertFalse(report["raw_evidence_included"])
        self.assertEqual(len(report["search"]), 1)

    def test_cli_run_true_path_control(self) -> None:
        """Control: with --include-evidence the canary IS in the carriers."""
        with tempfile.TemporaryDirectory() as tmp:
            _, out, tok = self._cli_run(Path(tmp), include_evidence=True)
            report = json.loads((out / "results.json").read_text(encoding="utf-8"))
        self.assertTrue(report["raw_evidence_included"])
        observations = [obs for entry in report["search"] for obs in entry.get("observations", [])]
        self.assertTrue(observations, "the evidence run carried observations")
        decoded = "\n".join(tok.decode(obs["input_ids"]) for obs in observations)
        # The toy BPE joins decoded tokens with spaces; normalize before
        # comparing so the control is not hostage to decoder spacing.
        self.assertIn(CANARY, _no_ws(decoded), "control: decoded input_ids hold the canary")
        seq = ",".join(str(i) for i in _canary_word_ids(tok))
        self.assertIn(
            seq,
            _no_ws(json.dumps(report)),
            "control: canary BPE ids present in evidence JSON (scan not vacuous)",
        )


class DiffReportsSweep48(LeakAsserts):
    """diff-reports reduces observations to verdicts; drift carries IDs only.

    There is no HTML render for diff-reports (stdout JSON + stderr summary
    only), so there is no diff-reports HTML surface to sweep.
    """

    @staticmethod
    def _evidence_reports(tmp: Path) -> tuple[dict, dict, Path, Path]:
        """Evidence reports."""
        reports = []
        dirs = []
        for name in ("old", "new"):
            split = parse_dataset(json.dumps(_hostile_document()))
            result = run_experiment(
                _CanaryBackend(),
                candidates("llama3", mutate=False)[:1],
                split,
                boundary=(124,),
                max_new_tokens=MAX_NEW_TOKENS,
                seed=7,
            )
            json_path, _ = save_report(result, tmp / name, include_evidence=True)
            dirs.append(tmp / name)
            reports.append(json.loads(json_path.read_text(encoding="utf-8")))
        return reports[0], reports[1], dirs[0], dirs[1]

    def test_diff_reports_self_diff_clean(self) -> None:
        """Test diff reports self diff clean."""
        with tempfile.TemporaryDirectory() as tmp:
            old, new, _, _ = self._evidence_reports(Path(tmp))
        # Vacuity: the compared reports really carry the canary -- the
        # probe backend's decoded answers start with it.
        answers = [obs["answer"] for entry in old["search"] for obs in entry["observations"]]
        self.assertTrue(answers)
        self.assertTrue(all(CANARY in answer for answer in answers))
        self.assertTrue(old["raw_evidence_included"] and new["raw_evidence_included"])
        diff = diff_reports(old, new, all_fixtures=True)
        self.assertTrue(diff["summary"]["identical"])
        text = json.dumps(diff, ensure_ascii=True, sort_keys=True)
        self.assertNoCanary(text, "diff-reports JSON (self diff, all_fixtures)")
        self.assertNoCanary(render_summary(diff), "diff-reports stderr summary")

    def test_cli_diff_reports_clean(self) -> None:
        """The ``mpl diff-reports`` CLI: JSON on stdout, summary on stderr."""
        with tempfile.TemporaryDirectory() as tmp:
            _, _, old_dir, new_dir = self._evidence_reports(Path(tmp))
            proc = _run_cli("diff-reports", str(old_dir), str(new_dir), "--all")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        diff = json.loads(proc.stdout)
        self.assertTrue(diff["summary"]["identical"])
        self.assertNoCanary(proc.stdout, "mpl diff-reports stdout")
        self.assertNoCanary(proc.stderr, "mpl diff-reports stderr")


class _OrdTokenizer:
    """Stub tokenizer for the debug REPL: code-point encode/decode."""

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> tuple[int, ...]:
        """Encode."""
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids: list[int] | tuple[int, ...]) -> str:
        """Decode."""
        return "".join(map(chr, token_ids))


class DebugTranscriptSweep48(LeakAsserts):
    """The ``.save`` transcript is a faithful echo of the session.

    A content-free session's saved transcript must carry no fixture
    content; a session where the operator explicitly rendered content
    carries it BY DESIGN (the transcript echoes the operator's own
    commands and their output -- USAGE.md section 10 warns not to paste
    transcripts into public issues).
    """

    @staticmethod
    def _session(commands: list[str]) -> str:
        """Session."""
        split = parse_dataset(json.dumps(_hostile_document()))
        out = io.StringIO()
        shell = DebugShell(
            split, dataset_label="c48-fixtures", tokenizer=_OrdTokenizer(), stdout=out
        )
        shell.cmdqueue = list(commands)
        shell.cmdloop()
        return out.getvalue()

    def test_save_content_free_session_transcript_clean(self) -> None:
        """Test save content free session transcript clean."""
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "transcript.txt")
            self._session(
                [".fixtures", ".fixture c48-search-1", ".audit", f".save {target}", ".quit"]
            )
            saved = Path(target).read_text(encoding="utf-8")
        self.assertNoCanary(saved, "debug .save transcript (content-free session)")
        # Non-vacuous: the session really ran -- fixture ids and the audit
        # output are in the transcript. (The ".save ..." echo and its
        # confirmation land in the transcript only after the file is
        # written, so they are absent from the file itself, by design.)
        self.assertIn("c48-search-1", saved)
        self.assertIn("audit: 0 finding(s)", saved)
        self.assertGreater(len(saved), 100)

    def test_save_after_render_contains_canary_by_design(self) -> None:
        """Control: ``.save`` is a faithful echo -- after an explicit
        ``.render`` the canary IS in the transcript (operator's own
        content, shown on their explicit request).
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "transcript.txt")
            self._session([".render", f".save {target}", ".quit"])
            saved = Path(target).read_text(encoding="utf-8")
        self.assertCanaryPresent(saved, "debug .save transcript after .render (by design)")

    def test_cli_debug_script_content_free_clean(self) -> None:
        """Test cli debug script content free clean."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset_path(Path(tmp))
            script = Path(tmp) / "session.txt"
            script.write_text(".fixtures\n.audit\n.quit\n", encoding="utf-8")
            proc = _run_cli("debug", "--dataset", str(dataset), "--script", str(script))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNoCanary(proc.stdout, "mpl debug --script stdout (content-free session)")
        self.assertNoCanary(proc.stderr, "mpl debug --script stderr")
        self.assertIn("c48-search-1", proc.stdout)


class DoctorSweep48(LeakAsserts):
    """``mpl doctor``: checks and counts only, never fixture content."""

    def test_doctor_report_clean(self) -> None:
        """Test doctor report clean."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, _ = _write_tiny_checkpoint(Path(tmp) / "ckpt")
            report = doctor(ckpt, _dataset_path(Path(tmp)), boundary_ids=(1, 2))
        text = json.dumps(report, ensure_ascii=True)
        self.assertNoCanary(text, "doctor() report")
        # Non-vacuous: doctor really read the hostile dataset file.
        self.assertEqual(
            (report["dataset"]["search_cases"], report["dataset"]["held_out_cases"]),
            (2, 1),
        )

    def test_cli_doctor_clean(self) -> None:
        """Test cli doctor clean."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, _ = _write_tiny_checkpoint(Path(tmp) / "ckpt")
            dataset = _dataset_path(Path(tmp))
            proc = _run_cli(
                "doctor",
                "--model",
                str(ckpt),
                "--dataset",
                str(dataset),
                "--boundary-ids",
                "1,2",
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(
            (report["dataset"]["search_cases"], report["dataset"]["held_out_cases"]),
            (2, 1),
        )
        self.assertNoCanary(proc.stdout, "mpl doctor stdout")
        self.assertNoCanary(proc.stderr, "mpl doctor stderr")


class SampleReportHygiene48(LeakAsserts):
    """The committed examples/sample-report/ stays synthetic and evidence-free."""

    def test_committed_sample_report_clean(self) -> None:
        """Test committed sample report clean."""
        raw = (REPO_ROOT / "examples" / "sample-report" / "results.json").read_bytes()
        report = json.loads(raw)
        self.assertFalse(report["raw_evidence_included"])
        keys: set[str] = set()
        _walk_keys(report, keys)
        for banned in ("observations", "input_ids", "output_ids", "hand_ids", "stop_texts"):
            self.assertNotIn(banned, keys, f"{banned} must not ship in the sample report")
        html_text = (REPO_ROOT / "examples" / "sample-report" / "report.html").read_text(
            encoding="utf-8"
        )
        self.assertNoCanary(raw.decode("utf-8"), "sample results.json")
        self.assertNoCanary(html_text, "sample report.html")


if __name__ == "__main__":
    unittest.main()
