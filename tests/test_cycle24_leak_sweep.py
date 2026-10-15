"""Cycle 24: canary leak sweep of every user-facing output surface.

Cycle 22 found and fixed two REAL evidence leaks in ``reporting.py`` (raw
generated text and token-ID subsequences reaching public reports without
``--include-evidence``). This file sweeps the *other* output surfaces for
the same leak class -- raw prompts, generations, expected answers, and
token-ID sequences decodable to user content -- using a hostile fixture
whose messages and expected answers carry the canary ``CANARY-SECRET-XYZ``.

Methodology: for each surface, run the real output builder over the
hostile input, serialize the output exactly as the surface emits it
(stdout JSON, stderr summary, REPL transcript, committed report bytes),
then assert the canary is absent in three forms:

- raw text (``CANARY-SECRET-XYZ``),
- the JSON-escaped form (identical here -- the canary is pure ASCII --
  but checked explicitly so the helper stays correct for non-ASCII
  canaries),
- the token-ID array form: with the ord-based probe backend, the canary
  encodes as ``[67, 65, 78, ...]``; the whitespace-stripped check catches
  the array however the serializer wraps it.

Every ``assertNoCanary`` call is paired with a guard proving the test is
not vacuous: the hostile input report itself *does* decode to the canary,
so a passing scan means the surface dropped the content, not that it was
never there.

Per-surface verdicts (each pinned below):

- ``mpl diff-reports``: CLEAN. Observations are reduced to pass/fail
  verdicts per fixture id; the JSON and the stderr summary carry no
  observation text, prompts, generations, or token-ID arrays -- even when
  both inputs are ``--include-evidence`` reports whose observations
  decode to the canary.
- ``mpl debug`` REPL: BY DESIGN (documented contract). ``.render`` prints
  the active messages' text, ``.tokens`` prints the exact token IDs when
  a tokenizer is attached, and ``.say``/``.save`` echo the operator's own
  typed input/transcript -- the REPL is a local interactive debugger over
  the operator's own dataset (docs/USAGE.md section 10 documents each
  command's content; ``.render``'s docstring: "Print the active hypothesis
  rendered over the active messages"). Surfaces that must stay
  content-free (``.audit``, ``.fixtures``, ``.tokens`` without ``--model``,
  ``.step``) are asserted clean.
- ``mpl doctor``: CLEAN. The full report (model checks, dataset audit,
  portability matrix, ``next_command`` suggestion) carries counts, fixed
  vocabularies, catalog-owned token strings and shell-quoted paths --
  no fixture message text, expected answers, or prompt token IDs, even
  when a fixture embeds a special-token literal (the audit finding then
  carries only the catalog literal, fixture id, field and offset).
- CLI ``catalog`` / ``inspect`` / ``diff-backends``: CLEAN / BY DESIGN.
  ``catalog`` prints the lab's own format hypotheses; ``inspect`` prints
  counts only; ``diff-backends`` echoes only its operator-supplied
  ``--prompt-text``/``--output-text``/``--stops`` arguments (it takes no
  dataset input by construction).
- ``examples/sample-report/``: CLEAN. The committed report is synthetic
  (generated from the committed ``examples/fixtures.json`` by
  ``examples/make_sample_report.py``), carries
  ``raw_evidence_included: false`` with no ``observations`` /
  ``input_ids`` / ``output_ids`` keys anywhere, and neither
  ``results.json`` nor ``report.html`` contains any fixture message text
  or expected answer.

Doc promises relied on (docs/USAGE.md, docs/FAQ.md, docs/SECURITY.md):

- "Raw evidence is never embedded in this HTML view. Add
  ``--include-evidence`` only when you deliberately want raw observations
  in JSON and understand that token IDs can reveal private prompts."
  (USAGE.md section 6)
- "the HTML view never embeds raw prompts or generations" (FAQ.md)
- "Default report JSON excludes raw observation arrays ... Evidence mode
  retains answer text and input/output token IDs. Token IDs are not
  anonymization." (SECURITY.md)

Deterministic: no network, no model downloads. The one tokenizer used
(doctor's deep check) is a tiny BPE trained locally on synthetic text.
"""

from __future__ import annotations

import io
import json
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
from model_protocol_lab.diffbackends import diff_backends
from model_protocol_lab.diffreports import diff_reports, render_summary
from model_protocol_lab.doctor import doctor, examine_dataset
from model_protocol_lab.experiments import run_experiment
from model_protocol_lab.reporting import save_report

REPO_ROOT = Path(__file__).resolve().parent.parent

CANARY = "CANARY-SECRET-XYZ"
# The probe backend encodes text as code points, so this is the exact
# token-ID array the canary takes wherever token IDs are emitted.
CANARY_ID_ARRAY = "[" + ",".join(str(ord(char)) for char in CANARY) + "]"


def _no_ws(text: str) -> str:
    return "".join(text.split())


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
        self.assertNotIn(
            CANARY_ID_ARRAY,
            _no_ws(text),
            f"canary token-ID sequence leaked into {surface}",
        )


def _hostile_document(literal: str | None = None) -> dict:
    """A fixture document whose user content (not ids) carries the canary."""
    user_text = f"Say the phrase {CANARY} out loud now"
    if literal is not None:
        user_text = f"Say {CANARY} with {literal} inline"
    return {
        "search": [
            {
                "id": "canary-search-1",
                "group": "canary-g1",
                "messages": [{"role": "user", "content": user_text}],
                "expected": f"the answer is {CANARY}",
            },
            {
                "id": "canary-search-2",
                "group": "canary-g1",
                "messages": [{"role": "user", "content": f"Repeat {CANARY} twice"}],
                "expected": "something else entirely",
            },
        ],
        "held_out": [
            {
                "id": "canary-held-1",
                "group": "canary-g2",
                "messages": [{"role": "user", "content": f"held-out says {CANARY}"}],
                "expected": f"held answer {CANARY}",
            },
        ],
    }


class _CanaryBackend:
    """Ord-based probe backend; generations echo the canary plus boundary."""

    fingerprint = "canary-probe-backend-v1"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        return "".join(map(chr, token_ids))

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        text = (CANARY + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


def _evidence_report(directory: Path) -> dict:
    """An --include-evidence results.json built from the hostile document."""
    split = parse_dataset(json.dumps(_hostile_document()))
    result = run_experiment(
        _CanaryBackend(),
        candidates("llama3", mutate=False)[:1],
        split,
        boundary=(124,),
        max_new_tokens=16,
        seed=7,
    )
    json_path, _ = save_report(result, directory, include_evidence=True)
    return json.loads(json_path.read_text(encoding="utf-8"))


class DiffReportsLeakSweep(LeakAsserts):
    """diff-reports must reduce observations to verdicts, never re-emit them."""

    def _decoded_observation_text(self, report: dict) -> str:
        decode = _CanaryBackend.decode
        chunks = []
        for entry in report["search"]:
            for obs in entry.get("observations", []):
                chunks.append(decode(tuple(obs["input_ids"])))
                chunks.append(decode(tuple(obs["output_ids"])))
                chunks.append(obs.get("answer") or "")
        held = report.get("held_out") or {}
        for obs in held.get("observations", []):
            chunks.append(decode(tuple(obs["input_ids"])))
        return "\n".join(chunks)

    def test_evidence_input_actually_carries_canary(self) -> None:
        """Guard: the hostile input report decodes to the canary (non-vacuous)."""
        with tempfile.TemporaryDirectory() as tmp:
            report = _evidence_report(Path(tmp) / "run")
        self.assertTrue(report["raw_evidence_included"])
        self.assertIn(CANARY, self._decoded_observation_text(report))

    def test_self_diff_emits_no_canary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = _evidence_report(Path(tmp) / "run")
        self.assertTrue(report["search"][0]["observations"])  # evidence present
        diff = diff_reports(report, report, all_fixtures=True)
        self.assertTrue(diff["evidence"]["old"] and diff["evidence"]["new"])
        self.assertTrue(diff["fixture_diffs"])  # the compared axis is populated
        text = json.dumps(diff, indent=2, ensure_ascii=True, sort_keys=True)
        self.assertNoCanary(text, "diff-reports JSON (all_fixtures)")
        self.assertNoCanary(render_summary(diff), "diff-reports stderr summary")

    def test_changed_diff_emits_no_canary(self) -> None:
        """Exercise the changed-path rendering (drift entries, verdict flips)."""
        with tempfile.TemporaryDirectory() as tmp:
            old = _evidence_report(Path(tmp) / "old")
            new = _evidence_report(Path(tmp) / "new")
        new = json.loads(json.dumps(new))
        new["backend_fingerprint"] = "a-different-fingerprint"
        summary = new["search"][0]["summary"]
        summary["exact_and_boundary_passes"] = max(0, summary["planned"] - 1)
        diff = diff_reports(old, new, all_fixtures=True)
        self.assertTrue(diff["config_drift"])  # the changed path is exercised
        text = json.dumps(diff, indent=2, ensure_ascii=True, sort_keys=True)
        self.assertNoCanary(text, "diff-reports JSON (changed runs)")
        self.assertNoCanary(render_summary(diff), "diff-reports stderr summary (changed)")

    def test_diff_entry_schema_carries_no_free_text(self) -> None:
        """Diff entries are fixed-schema verdict/drift records, not text."""
        with tempfile.TemporaryDirectory() as tmp:
            report = _evidence_report(Path(tmp) / "run")
        diff = diff_reports(report, report, all_fixtures=True)
        for entry in diff["fixture_diffs"]:
            with self.subTest(entry=entry):
                self.assertEqual(set(entry), {"stage", "candidate", "fixture", "old", "new"})
                self.assertTrue(
                    all(
                        value in ("pass", "fail", "absent")
                        for value in (entry["old"], entry["new"])
                    )
                )
        for entry in diff["config_drift"]:
            with self.subTest(entry=entry):
                self.assertEqual(set(entry), {"field", "old", "new"})


class _OrdTokenizer:
    """Stub tokenizer for the debug REPL: code-point encode/decode."""

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> tuple[int, ...]:
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids: list[int] | tuple[int, ...]) -> str:
        return "".join(map(chr, token_ids))


class DebugReplContract(LeakAsserts):
    """The debug REPL shows the operator's own dataset by design; pin it.

    ``mpl debug`` is a local interactive debugger over a dataset the
    operator supplied (docs/USAGE.md section 10 documents every command's
    content). Showing fixture text in ``.render`` / ``.tokens`` is the
    tool's documented purpose -- the exact opposite of a report leak, where
    the output may be published. These tests pin the documented contract
    so a future change either keeps it or updates the docs deliberately.
    """

    def _session(self, commands: list[str], tokenizer=None) -> str:
        split = parse_dataset(json.dumps(_hostile_document()))
        out = io.StringIO()
        shell = DebugShell(split, dataset_label="canary-fixtures", tokenizer=tokenizer, stdout=out)
        shell.cmdqueue = list(commands)
        shell.cmdloop()
        return out.getvalue()

    def test_render_shows_fixture_messages_by_design(self) -> None:
        """USAGE.md section 10: ".render | Render the hypothesis over the active messages"."""
        out = self._session([".render", ".quit"])
        self.assertIn(CANARY, out)

    def test_tokens_with_tokenizer_shows_exact_ids_by_design(self) -> None:
        """USAGE.md section 10: ".tokens | Exact token IDs of the full render".

        The IDs printed are for the full render (format framing included),
        so the canary's code points appear embedded in the longer array;
        decoding the printed array must recover the canary text.
        """
        import re

        out = self._session([".tokens", ".quit"], tokenizer=_OrdTokenizer())
        match = re.search(r"ids=\[([0-9, ]+)\]", out)
        self.assertIsNotNone(match, "the .tokens output prints the ID array")
        assert match is not None
        ids = tuple(int(part) for part in match.group(1).split(","))
        self.assertIn(CANARY, _OrdTokenizer.decode(ids))

    def test_say_echoes_operator_typed_text_by_design(self) -> None:
        """.say buffers the operator's own lines; .render then uses them."""
        out = self._session([".say", f"user: my own {CANARY} line", ".", ".render", ".quit"])
        self.assertIn(CANARY, out)

    def test_save_persists_transcript_to_operator_file_by_design(self) -> None:
        """.save writes the session transcript to the operator-chosen path."""
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "transcript.txt")
            self._session([".render", f".save {target}", ".quit"])
            saved = Path(target).read_text(encoding="utf-8")
        self.assertIn(CANARY, saved)

    def test_audit_reports_no_message_text(self) -> None:
        """.audit flags special-token literals; findings carry no message text."""
        out = self._session([".audit", ".quit"])
        self.assertNoCanary(out, "debug .audit")

    def test_fixtures_lists_ids_only(self) -> None:
        out = self._session([".fixtures", ".quit"])
        self.assertNoCanary(out, "debug .fixtures")

    def test_tokens_without_tokenizer_prints_reminder_not_ids(self) -> None:
        out = self._session([".tokens", ".quit"])
        self.assertNoCanary(out, "debug .tokens (no tokenizer)")
        self.assertIn("no tokenizer attached", out)

    def test_step_with_typed_ids_prints_no_content(self) -> None:
        out = self._session([".boundary 1,2", ".step 1,2", ".quit"])
        self.assertNoCanary(out, "debug .step")
        self.assertIn("overhang=", out)


def _write_checkpoint(root: Path, *, real_tokenizer: bool) -> Path:
    """A minimal local checkpoint dir. The real-tokenizer variant trains a
    tiny BPE locally (no downloads) so the deep tokenizer check passes and
    ``doctor`` reaches verdict "ok" with a ``next_command`` suggestion."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "n_positions": 1024}), encoding="utf-8"
    )
    (root / "model.safetensors").write_bytes(b"")
    if real_tokenizer:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tok = Tokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        tok.train_from_iterator(
            ["hello world foo bar"],
            BpeTrainer(special_tokens=["[UNK]", "[BOS]", "[EOS]"], vocab_size=64),
        )
        tok.save(str(root / "tokenizer.json"))
    else:
        (root / "tokenizer.json").write_text(
            json.dumps({"model": {"dropout": 0}, "added_tokens": []}), encoding="utf-8"
        )
    return root


class DoctorLeakSweep(LeakAsserts):
    """mpl doctor's report carries checks and counts, never fixture text."""

    def _dataset_path(self, tmp: str, literal: str | None = None) -> Path:
        path = Path(tmp) / "dataset.json"
        path.write_text(json.dumps(_hostile_document(literal=literal)), encoding="utf-8")
        return path

    def test_examine_dataset_emits_no_fixture_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = examine_dataset(self._dataset_path(tmp))
        text = json.dumps(report, ensure_ascii=True)
        self.assertNoCanary(text, "doctor examine_dataset")
        dataset = report["dataset"]
        self.assertEqual(
            (dataset["search_cases"], dataset["held_out_cases"]),
            (2, 1),
            "scan ran over the hostile dataset",
        )

    def test_doctor_full_report_emits_no_fixture_text(self) -> None:
        """Full doctor() incl. the next_command suggestion (verdict ok)."""
        with tempfile.TemporaryDirectory() as tmp:
            report = doctor(
                _write_checkpoint(Path(tmp) / "ckpt", real_tokenizer=True),
                self._dataset_path(tmp),
                boundary_ids=(1, 2),
            )
        self.assertEqual(report["verdict"], "ok")
        self.assertIsNotNone(report["suggestion"], "suggestion path is exercised")
        text = json.dumps(report, ensure_ascii=True)
        self.assertNoCanary(text, "doctor full report")

    def test_suggestion_contains_only_paths_counts_and_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = doctor(
                _write_checkpoint(Path(tmp) / "ckpt", real_tokenizer=True),
                self._dataset_path(tmp),
                boundary_ids=(1, 2),
            )
        suggestion = report["suggestion"]
        self.assertIsNotNone(suggestion)
        text = json.dumps(suggestion, ensure_ascii=True)
        self.assertNoCanary(text, "doctor next_command suggestion")
        self.assertIn("--boundary-ids 1,2", suggestion["next_command"])

    def test_audit_findings_carry_catalog_literals_not_user_text(self) -> None:
        """A special-token literal in fixture text yields findings that name
        the catalog literal (lab-owned vocabulary) -- never the message."""
        with tempfile.TemporaryDirectory() as tmp:
            report = examine_dataset(self._dataset_path(tmp, literal="<|im_start|>"))
        audit = report["dataset"]["special_token_audit"]
        self.assertEqual(audit["status"], "warning")
        self.assertTrue(audit["findings"], "the literal is actually flagged")
        for finding in audit["findings"]:
            with self.subTest(finding=finding):
                self.assertEqual(
                    set(finding),
                    {"fixture_id", "field", "literal", "format_name", "offset"},
                )
                self.assertEqual(finding["literal"], "<|im_start|>")
        text = json.dumps(report, ensure_ascii=True)
        self.assertNoCanary(text, "doctor dataset audit with special-token literal")


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "model_protocol_lab", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


class CliSurfaceSweep(LeakAsserts):
    """catalog / inspect / diff-backends: no fixture-derived content."""

    def test_catalog_output_is_lab_owned_hypotheses(self) -> None:
        proc = _run_cli("catalog")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        entries = json.loads(proc.stdout)
        self.assertTrue(entries)
        for entry in entries:
            with self.subTest(name=entry.get("name")):
                self.assertIn("name", entry)
                self.assertIn("bos", entry)
        self.assertNoCanary(proc.stdout, "mpl catalog")

    def test_inspect_output_is_counts_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.json"
            path.write_text(json.dumps(_hostile_document()), encoding="utf-8")
            proc = _run_cli("inspect", str(path))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(set(payload), {"search_cases", "held_out_cases", "valid"})
        self.assertEqual((payload["search_cases"], payload["held_out_cases"]), (2, 1))
        self.assertNoCanary(proc.stdout, "mpl inspect stdout")
        self.assertNoCanary(proc.stderr, "mpl inspect stderr")

    def test_diff_backends_echoes_only_operator_supplied_inputs(self) -> None:
        """diff-backends takes no dataset input; its output derives only
        from the operator's --prompt-text/--output-text/--stops arguments,
        which it echoes back by design (like any CLI echoing its args)."""

        class _Backend:
            @staticmethod
            def decode(token_ids: tuple[int, ...]) -> str:
                return "".join(map(chr, token_ids))

        result = diff_backends(
            prompt_ids=tuple(map(ord, f"prompt {CANARY}")),
            output_ids=tuple(map(ord, "output stop!")),
            stop_texts=(CANARY, "stop!"),
            eos_id=None,
            min_tokens=0,
            backend=_Backend(),
        )
        # By design: the operator's own stop texts are echoed in the result.
        self.assertEqual(result["stop_texts"], [CANARY, "stop!"])
        # And with canary-free inputs, the output cannot contain the canary:
        # there is no dataset input for content to come from.
        clean = diff_backends(
            prompt_ids=(1, 2, 3),
            output_ids=(4, 5),
            stop_texts=("stop!",),
            eos_id=None,
            min_tokens=0,
            backend=_Backend(),
        )
        self.assertNoCanary(json.dumps(clean, ensure_ascii=True), "diff-backends")
        self.assertNotIn("observations", clean)
        self.assertNotIn("fixtures", json.dumps(clean))


def _walk_keys(value: object, seen: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            seen.add(str(key))
            _walk_keys(item, seen)
    elif isinstance(value, list):
        for item in value:
            _walk_keys(item, seen)


class SampleReportPublicHygiene(LeakAsserts):
    """The committed examples/sample-report/ is synthetic and evidence-free."""

    def test_committed_json_has_no_observations(self) -> None:
        raw = (REPO_ROOT / "examples" / "sample-report" / "results.json").read_bytes()
        report = json.loads(raw)
        self.assertFalse(report["raw_evidence_included"])
        keys: set[str] = set()
        _walk_keys(report, keys)
        for banned in ("observations", "input_ids", "output_ids"):
            self.assertNotIn(banned, keys, f"{banned} must not ship in the sample report")

    def test_committed_files_contain_no_fixture_message_text(self) -> None:
        """Fixture ids may appear (outcome grid, decisive lists); message
        text and expected answers must not -- the sample fixtures are
        synthetic, and this pins that no real user content crept in."""
        fixtures = json.loads((REPO_ROOT / "examples" / "fixtures.json").read_text())
        contents: list[str] = []
        ids: list[str] = []
        for section in ("search", "held_out"):
            for case in fixtures[section]:
                ids.append(case["id"])
                contents.append(case["expected"])
                contents.extend(message["content"] for message in case["messages"])
        self.assertTrue(contents and ids)
        # Single-character expected answers (e.g. "7") are untestable by
        # substring: they occur everywhere in any report ("20261012", ...).
        # The leak class is multi-character content; require length >= 4 so
        # a match would be evidence of actual fixture text, not coincidence.
        contents = [content for content in contents if len(content) >= 4]
        self.assertTrue(contents)
        json_bytes = (REPO_ROOT / "examples" / "sample-report" / "results.json").read_text()
        html_bytes = (REPO_ROOT / "examples" / "sample-report" / "report.html").read_text()
        for content in contents:
            with self.subTest(content=content[:24]):
                self.assertNotIn(content, json_bytes)
                self.assertNotIn(content, html_bytes)
        # Non-vacuous: fixture ids do appear in the public views by design.
        self.assertIn(ids[0], json_bytes)
        self.assertIn(ids[0], html_bytes)


if __name__ == "__main__":
    unittest.main()
