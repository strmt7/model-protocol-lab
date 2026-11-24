"""W4-3 (RR-14b) — crash-safe run progress + resume.

Before W4, a run that died hours in (host replacement, kill -9,
OOM) lost every completed cell: the report pair is written only at
the end, so the only record of finished work died with the process.
The progress journal fixes that without weakening determinism:

* every ``run`` writes ``progress.jsonl`` into its claimed output
  directory as cells complete — a header binding the run's identity
  (configuration digest + progress schema + package version, D-3),
  one hash-chained line per completed observation, and a footer
  once the report pair is saved;
* ``run --resume`` verifies the journal against the resuming
  invocation's own configuration before any generation, then
  re-executes only the missing cells. Per-cell seeds are
  position-independent (``request_seed_for(seed, case.id)``), so a
  resumed run's report equals the uninterrupted control's except
  for the ``resumed`` accounting block;
* refusals are fixed codes: a different configuration, schema or
  package version refuses with ``resume_configuration_mismatch``;
  a mid-file chain break refuses with ``resume_progress_corrupt``;
  held-out records ahead of a complete search stage refuse with
  ``resume_stage_order_violation``; a torn LAST line (the kill
  landed mid-write) is dropped and exactly that cell re-executes;
  a directory already holding the report pair refuses with
  ``resume_already_complete``.

The identity deliberately excludes the budget cap (D-3): the
resuming invocation's cap governs, enforced by the same full-run
preflight bound as a fresh run.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from model_protocol_lab import backends as backends_module
from model_protocol_lab import cli as cli_module
from model_protocol_lab import core
from model_protocol_lab.core import evaluate
from model_protocol_lab.experiments import configuration_digest, run_experiment
from model_protocol_lab.runprogress import (
    PROGRESS_SCHEMA,
    ProgressSink,
    load_progress,
)

BOUNDARY = (999,)
MAX_NEW_TOKENS = 40
PACKAGE_VERSION = "0.2.0-test"


class ScriptedBackend:
    """Deterministic backend: ord-encode; pass text + boundary, padded."""

    fingerprint = "w4-progress-resume"

    def __init__(self, expected, behavior=None):
        """Init."""
        self._expected = dict(expected)
        self._behavior = dict(behavior or {})
        self.generate_calls = 0

    def encode(self, text):
        """Encode."""
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids):
        """Decode."""
        return "".join(map(chr, token_ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        self.generate_calls += 1
        prompt = self.decode(input_ids)
        candidate = next((n for n in ("cand-A", "cand-B") if f"[{n}]" in prompt), None)
        fixture = next((fid for fid in self._expected if f"question-{fid}" in prompt), None)
        if candidate is None or fixture is None:
            return core.Generation((), "error", "unknown_request")
        mode = self._behavior.get((candidate, fixture), "pass")
        expected = self._expected[fixture]
        text = expected + chr(BOUNDARY[0]) if mode == "pass" else expected + "WRONG"
        ids = tuple(map(ord, text))
        assert len(ids) <= max_new_tokens
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


EXPECTED = {"S1": "GREEN", "S2": "BLUE", "H1": "RED", "H2": "CYAN"}
# cand-B fails S2, so cand-A is the unique winner and held-out runs.
BEHAVIOR = {("cand-B", "S2"): "fail"}
TOTAL_CELLS = 2 * 2 + 2


def _split():
    """Split."""

    def fixtures(ids):
        """Fixtures."""
        return tuple(
            core.Fixture(
                id=fid,
                messages=(core.Message(role="user", content=f"question-{fid}"),),
                expected=EXPECTED[fid],
                group=f"group-{fid}",
            )
            for fid in ids
        )

    return core.Split(search=fixtures(("S1", "S2")), held_out=fixtures(("H1", "H2")))


def _formats():
    """Formats."""
    return tuple(
        core.Format(
            name=name,
            bos="",
            user_prefix=f"[{name}]U:",
            assistant_prefix="A:",
            turn_end="\n",
        )
        for name in ("cand-A", "cand-B")
    )


def _digest(backend, **overrides):
    """Digest."""
    kwargs = {
        "boundary": BOUNDARY,
        "boundary_map": None,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": 7,
        "eos_id": None,
    }
    kwargs.update(overrides)
    return configuration_digest(backend, _formats(), _split(), **kwargs)


def _header(backend):
    """Header."""
    return {
        "schema": PROGRESS_SCHEMA,
        "package_version": PACKAGE_VERSION,
        "configuration_digest": _digest(backend),
        "max_total_tokens": None,
    }


def _run(backend, **kwargs):
    """Run."""
    return run_experiment(
        backend,
        _formats(),
        _split(),
        boundary=BOUNDARY,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=7,
        **kwargs,
    )


def _load(path, backend, **overrides):
    """Load."""
    kwargs = {
        "expected_digest": _digest(backend),
        "expected_package_version": PACKAGE_VERSION,
    }
    kwargs.update(overrides)
    return load_progress(path, formats=_formats(), split=_split(), **kwargs)


def _crash_run(path, backend, *, crash_after):
    """Run with a sink whose callback raises after ``crash_after`` records.

    Simulates kill -9: the journal holds exactly the records written
    before the crash, no footer, and the exception propagates.
    """
    sink = ProgressSink.create(path, header=_header(backend))
    written = 0

    def callback(stage, candidate, observation):
        """Callback."""
        nonlocal written
        sink.append(stage, candidate, observation)
        written += 1
        if written == crash_after:
            raise RuntimeError("simulated crash")

    with unittest.TestCase().assertRaises(RuntimeError):
        _run(backend, on_observation=callback)
    sink.close()
    return sink


class JournalTests(unittest.TestCase):
    """The journal verifies, chains, and refuses by fixed codes."""

    def test_journal_open_does_not_translate_newlines(self):
        """The journal writer must open with newline="".

        The journal is a byte format — load_progress splits the
        raw bytes and verifies a hash chain — so a text-mode
        open that translates newlines (CRLF on Windows) makes
        every journal unreadable to its own reader (RR-18
        follow-up). Pin the construction.
        """
        from unittest import mock

        real_open = Path.open
        calls: list[dict] = []

        def recording_open(self, *args, **kwargs):
            """Record the call's kwargs, then delegate."""
            calls.append(kwargs)
            return real_open(self, *args, **kwargs)

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            with mock.patch.object(Path, "open", recording_open):
                sink = ProgressSink.create(path, header=_header(backend))
                sink.close()
        self.assertTrue(calls)
        for kwargs in calls:
            self.assertEqual(kwargs.get("newline"), "")

    def test_roundtrip_full_run(self):
        """Test roundtrip full run."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            sink = ProgressSink.create(path, header=_header(backend))
            result = _run(backend, on_observation=sink.append)
            sink.finalize({"decision": result["decision"], "actual_total_tokens": 1})
            loaded = _load(path, ScriptedBackend(EXPECTED, BEHAVIOR))
            self.assertEqual(len(loaded.observations), TOTAL_CELLS)
            self.assertIsNotNone(loaded.footer)
            self.assertTrue(loaded.footer["complete"])
            self.assertEqual(loaded.torn_lines, 0)
            obs = loaded.observations[("search", "cand-A", "S1")]
            self.assertEqual(obs.candidate, "cand-A")
            self.assertEqual(obs.fixture, "S1")
            self.assertEqual(obs.status, "completed")
            self.assertIn(("held_out", "cand-A", "H2"), loaded.observations)

    def test_configuration_mismatch_refused(self):
        """Test configuration mismatch refused."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            _crash_run(path, backend, crash_after=2)
            for overrides in (
                {"expected_digest": _digest(backend, seed=99)},
                {"expected_package_version": "9.9.9-other"},
            ):
                with self.assertRaises(ValueError) as caught:
                    _load(path, ScriptedBackend(EXPECTED, BEHAVIOR), **overrides)
                self.assertEqual(str(caught.exception), "resume_configuration_mismatch", overrides)

    def test_mid_chain_corruption_refused(self):
        """Test mid chain corruption refused."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            _crash_run(path, backend, crash_after=3)
            lines = path.read_text(encoding="utf-8").splitlines()
            record = json.loads(lines[1])
            record["observation"]["answer"] = "TAMPERED"
            lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                _load(path, ScriptedBackend(EXPECTED, BEHAVIOR))
            self.assertEqual(str(caught.exception), "resume_progress_corrupt")

    def test_held_out_before_search_complete_refused(self):
        """Test held out before search complete refused."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            sink = ProgressSink.create(path, header=_header(backend))
            # One search record, then a held-out record: the original
            # run could never have written this order.
            probe = ScriptedBackend(EXPECTED, BEHAVIOR)
            captured = []
            _run(probe, on_observation=lambda s, c, o: captured.append((s, c, o)))
            sink.append(*captured[0])
            held = next(item for item in captured if item[0] == "held_out")
            sink.append(*held)
            sink.close()
            with self.assertRaises(ValueError) as caught:
                _load(path, ScriptedBackend(EXPECTED, BEHAVIOR))
            self.assertEqual(str(caught.exception), "resume_stage_order_violation")


class ResumeRunTests(unittest.TestCase):
    """A resumed run re-executes only the missing cells."""

    def test_crash_then_resume_matches_control(self):
        """Test crash then resume matches control."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            control = _run(ScriptedBackend(EXPECTED, BEHAVIOR))
            self.assertNotIn("resumed", control)
            crashed_backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            _crash_run(path, crashed_backend, crash_after=3)
            self.assertEqual(crashed_backend.generate_calls, 3)
            loaded = _load(path, ScriptedBackend(EXPECTED, BEHAVIOR))
            self.assertEqual(len(loaded.observations), 3)
            sink = ProgressSink.resume(path, loaded)
            resumed_backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            resumed = _run(
                resumed_backend,
                on_observation=sink.append,
                resume_observations=loaded.observations,
            )
            sink.finalize({"decision": resumed["decision"], "actual_total_tokens": 1})
            self.assertEqual(resumed_backend.generate_calls, TOTAL_CELLS - 3)
            self.assertEqual(
                resumed["resumed"], {"cells_reused": 3, "cells_generated": TOTAL_CELLS - 3}
            )
            stripped = {k: v for k, v in resumed.items() if k != "resumed"}
            self.assertEqual(stripped, control)
            # The journal is now complete and verifies end to end.
            reloaded = _load(path, ScriptedBackend(EXPECTED, BEHAVIOR))
            self.assertEqual(len(reloaded.observations), TOTAL_CELLS)
            self.assertIsNotNone(reloaded.footer)

    def test_torn_last_line_reexecutes_exactly_one_cell(self):
        """Test torn last line reexecutes exactly one cell."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            _crash_run(path, backend, crash_after=3)
            raw = path.read_bytes()
            path.write_bytes(raw[:-7])  # tear the third record line mid-write
            loaded = _load(path, ScriptedBackend(EXPECTED, BEHAVIOR))
            self.assertEqual(len(loaded.observations), 2)
            self.assertEqual(loaded.torn_lines, 1)
            sink = ProgressSink.resume(path, loaded)
            resumed_backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            resumed = _run(
                resumed_backend,
                on_observation=sink.append,
                resume_observations=loaded.observations,
            )
            sink.close()
            self.assertEqual(resumed_backend.generate_calls, TOTAL_CELLS - 2)
            self.assertEqual(resumed["resumed"]["cells_reused"], 2)
            reloaded = _load(path, ScriptedBackend(EXPECTED, BEHAVIOR))
            self.assertEqual(len(reloaded.observations), TOTAL_CELLS)

    def test_zero_generation_resume_from_complete_journal(self):
        """Test zero generation resume from complete journal."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            sink = ProgressSink.create(path, header=_header(backend))
            control = _run(backend, on_observation=sink.append)
            sink.close()  # died after the run, before the report was saved
            loaded = _load(path, ScriptedBackend(EXPECTED, BEHAVIOR))
            resumed_backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            resumed = _run(resumed_backend, resume_observations=loaded.observations)
            self.assertEqual(resumed_backend.generate_calls, 0)
            self.assertEqual(
                resumed["resumed"],
                {"cells_reused": TOTAL_CELLS, "cells_generated": 0},
            )
            stripped = {k: v for k, v in resumed.items() if k != "resumed"}
            self.assertEqual(stripped, control)

    def test_resume_cap_governs_via_full_preflight(self):
        """Test resume cap governs via full preflight."""
        # D-3: the cap is not part of the resume identity; the
        # resuming invocation's cap is enforced against the FULL
        # run's preflight bound, reused cells or not.
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            _crash_run(path, backend, crash_after=3)
            loaded = _load(path, ScriptedBackend(EXPECTED, BEHAVIOR))
            refusing = ScriptedBackend(EXPECTED, BEHAVIOR)
            with self.assertRaises(ValueError) as caught:
                _run(
                    refusing,
                    resume_observations=loaded.observations,
                    max_total_tokens=10,
                )
            self.assertEqual(str(caught.exception), "run_budget_exceeded_preflight")
            self.assertEqual(refusing.generate_calls, 0)

    def test_held_out_prior_for_loser_is_stage_order_violation(self):
        """Test held out prior for loser is stage order violation."""
        probe = ScriptedBackend(EXPECTED, BEHAVIOR)
        captured = {}
        _run(
            probe,
            on_observation=lambda s, c, o: captured.__setitem__((s, c, o.fixture), o),
        )
        priors = {key: obs for key, obs in captured.items() if key[0] == "search"}
        loser_held = asdict(captured[("held_out", "cand-A", "H1")])
        loser_held["candidate"] = "cand-B"
        priors[("held_out", "cand-B", "H1")] = core.Observation(
            **{
                **loser_held,
                "input_ids": tuple(loser_held["input_ids"]),
                "output_ids": tuple(loser_held["output_ids"]),
            }
        )
        with self.assertRaises(ValueError) as caught:
            _run(ScriptedBackend(EXPECTED, BEHAVIOR), resume_observations=priors)
        self.assertEqual(str(caught.exception), "resume_stage_order_violation")

    def test_resume_map_outside_plan_rejected(self):
        """Test resume map outside plan rejected."""
        probe = ScriptedBackend(EXPECTED, BEHAVIOR)
        captured = {}
        _run(
            probe,
            on_observation=lambda s, c, o: captured.__setitem__((s, c, o.fixture), o),
        )
        obs = captured[("search", "cand-A", "S1")]
        with self.assertRaises(ValueError):
            _run(
                ScriptedBackend(EXPECTED, BEHAVIOR),
                resume_observations={("search", "cand-A", "NOPE"): obs},
            )
        with self.assertRaises(ValueError):
            _run(
                ScriptedBackend(EXPECTED, BEHAVIOR),
                resume_observations={("search", "cand-B", "S1"): obs},
            )


REPO_ROOT = Path(__file__).resolve().parent.parent
MPL = [sys.executable, "-m", "model_protocol_lab"]


def _require_toolkit(testcase: unittest.TestCase) -> None:
    """Require toolkit."""
    try:
        import safetensors  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError as exc:
        testcase.skipTest(f"synthetic-checkpoint toolkit missing: {exc}")


def _build_checkpoint(root: Path) -> Path:
    """Build a tiny deterministic GPT-2 checkpoint (no network)."""
    import numpy as np
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as _BLDec
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    root.mkdir(parents=True, exist_ok=True)
    n_embd, n_layer, n_pos, vocab = 64, 2, 1024, 267
    rng = np.random.default_rng(7)
    tensors: dict = {}
    tensors["transformer.wte.weight"] = rng.normal(0, 0.02, (vocab, n_embd)).astype(np.float32)
    tensors["transformer.wpe.weight"] = rng.normal(0, 0.02, (n_pos, n_embd)).astype(np.float32)
    for i in range(n_layer):
        p = f"transformer.h.{i}"
        tensors[f"{p}.ln_1.weight"] = np.ones(n_embd, dtype=np.float32)
        tensors[f"{p}.ln_1.bias"] = np.zeros(n_embd, dtype=np.float32)
        tensors[f"{p}.attn.c_attn.weight"] = rng.normal(0, 0.02, (n_embd, 3 * n_embd)).astype(
            np.float32
        )
        tensors[f"{p}.attn.c_attn.bias"] = np.zeros(3 * n_embd, dtype=np.float32)
        tensors[f"{p}.attn.c_proj.weight"] = rng.normal(0, 0.02, (n_embd, n_embd)).astype(
            np.float32
        )
        tensors[f"{p}.attn.c_proj.bias"] = np.zeros(n_embd, dtype=np.float32)
        tensors[f"{p}.ln_2.weight"] = np.ones(n_embd, dtype=np.float32)
        tensors[f"{p}.ln_2.bias"] = np.zeros(n_embd, dtype=np.float32)
        tensors[f"{p}.mlp.c_fc.weight"] = rng.normal(0, 0.02, (n_embd, 4 * n_embd)).astype(
            np.float32
        )
        tensors[f"{p}.mlp.c_fc.bias"] = np.zeros(4 * n_embd, dtype=np.float32)
        tensors[f"{p}.mlp.c_proj.weight"] = rng.normal(0, 0.02, (4 * n_embd, n_embd)).astype(
            np.float32
        )
        tensors[f"{p}.mlp.c_proj.bias"] = np.zeros(n_embd, dtype=np.float32)
    tensors["transformer.ln_f.weight"] = np.ones(n_embd, dtype=np.float32)
    tensors["transformer.ln_f.bias"] = np.zeros(n_embd, dtype=np.float32)
    save_file(tensors, str(root / "model.safetensors"))
    vocab_map = {bytes([i]).decode("latin-1"): i for i in range(256)}
    for i in range(10):
        vocab_map[f"tok{i}"] = 256 + i
    vocab_map["<|eos|>"] = 266
    tok = Tokenizer(BPE(vocab=vocab_map, merges=[]))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = _BLDec()
    tok.add_special_tokens(["<|eos|>"])
    tok.save(str(root / "tokenizer.json"))
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GPT2LMHeadModel"],
                "model_type": "gpt2",
                "n_positions": n_pos,
                "n_ctx": n_pos,
                "n_embd": n_embd,
                "n_layer": n_layer,
                "n_head": 2,
                "vocab_size": vocab,
            }
        )
    )
    return root


def _write_dataset(path: Path, *, search: int = 20, held_out: int = 4) -> Path:
    """Write dataset."""

    def fixtures(prefix, count, word):
        """Fixtures."""
        return [
            {
                "id": f"{prefix}-{i}",
                "group": f"g-{prefix}",
                "messages": [
                    {"role": "user", "content": f"Reply with exactly {word}{i} and nothing else."}
                ],
                "expected": f"{word}{i}",
            }
            for i in range(count)
        ]

    path.write_text(
        json.dumps({"search": fixtures("s", search, "W"), "held_out": fixtures("h", held_out, "H")})
    )
    return path


class CliResumeTests(unittest.TestCase):
    """Subprocess acceptance: kill -9 mid-run, resume, compare.

    The synthetic checkpoint passes nothing (decision
    ``no_candidate``), so the CLI path exercises search-stage
    resume; held-out resume is covered at the run_experiment level
    above. The resumed report must equal the uninterrupted control
    except for the ``resumed`` block.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test fixture."""
        _require_toolkit(cls)
        cls._tmp = tempfile.mkdtemp(prefix="mpl_w4_resume_")
        cls.model = _build_checkpoint(Path(cls._tmp) / "model")
        cls.dataset = _write_dataset(Path(cls._tmp) / "data.json")

    def _run_args(self, output: Path, *extra: str) -> list[str]:
        """Run args."""
        return [
            *MPL,
            "run",
            "--model",
            str(self.model),
            "--dataset",
            str(self.dataset),
            "--family",
            "chatml",
            "--mutate",
            "--boundary-ids",
            "266",
            "--max-new-tokens",
            "48",
            "--max-context",
            "1024",
            "--output",
            str(output),
            *extra,
        ]

    def _run_to_completion(self, output: Path, *extra: str) -> subprocess.CompletedProcess:
        """Run to completion."""
        return subprocess.run(
            self._run_args(output, *extra),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )

    def _kill_mid_run(self, output: Path) -> int:
        """Start a run, SIGKILL it once >=3 cells are journaled."""
        proc = subprocess.Popen(
            self._run_args(output),
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        journal = output / "progress.jsonl"
        deadline = time.monotonic() + 600
        try:
            while time.monotonic() < deadline:
                if journal.is_file():
                    lines = journal.read_bytes().count(b"\n")
                    if lines >= 4:  # header + >=3 records
                        proc.kill()
                        proc.wait(timeout=60)
                        return lines - 1
                if proc.poll() is not None:
                    self.fail("run finished before the kill could land")
                time.sleep(0.1)
            self.fail("journal did not reach 3 records before the deadline")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=60)
        raise AssertionError("unreachable")

    def test_kill_resume_matches_control_and_refusals(self):
        """Test kill resume matches control and refusals."""
        with TemporaryDirectory() as tmp:
            control_dir = Path(tmp) / "control"
            victim_dir = Path(tmp) / "victim"
            control = self._run_to_completion(control_dir)
            # A completed run exits 0 only for a compatible verdict;
            # this checkpoint passes nothing, so the completed run
            # exits 2 with the report pair as its success artifact.
            self.assertEqual(control.returncode, 2, control.stderr[-2000:])
            self.assertTrue((control_dir / "results.json").is_file())
            control_result = json.loads((control_dir / "results.json").read_text())
            self.assertNotIn("resumed", control_result)
            self.assertTrue((control_dir / "progress.jsonl").is_file())

            journaled = self._kill_mid_run(victim_dir)
            self.assertGreaterEqual(journaled, 3)
            self.assertFalse((victim_dir / "results.json").exists())

            # A different seed is a different configuration: refuse
            # before any generation, journal untouched.
            mismatch = self._run_to_completion(victim_dir, "--resume", "--seed", "99")
            self.assertEqual(mismatch.returncode, 1, mismatch.stderr[-2000:])
            self.assertIn("resume_configuration_mismatch", mismatch.stderr)
            self.assertFalse((victim_dir / "results.json").exists())

            resumed = self._run_to_completion(victim_dir, "--resume")
            self.assertEqual(resumed.returncode, 2, resumed.stderr[-2000:])
            self.assertTrue((victim_dir / "results.json").is_file())
            resumed_result = json.loads((victim_dir / "results.json").read_text())
            block = resumed_result.pop("resumed")
            self.assertGreaterEqual(block["cells_reused"], 3)
            self.assertGreater(block["cells_generated"], 0)
            self.assertEqual(resumed_result, control_result)
            footer = json.loads(
                (victim_dir / "progress.jsonl").read_text().strip().splitlines()[-1]
            )
            self.assertTrue(footer["footer"]["complete"])

            # The completed directory refuses a second resume.
            again = self._run_to_completion(victim_dir, "--resume")
            self.assertEqual(again.returncode, 1, again.stderr[-2000:])
            self.assertIn("resume_already_complete", again.stderr)

    def test_resume_missing_directory_is_a_usage_error(self):
        """Test resume missing directory is a usage error."""
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope"
            proc = subprocess.run(
                self._run_args(missing, "--resume"),
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(proc.returncode, 2, proc.stderr[-2000:])
            self.assertIn("--resume", proc.stderr)


def _canon(obj):
    """Canon."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _chain(prev, body):
    """Chain."""
    return hashlib.sha256((prev + _canon(body)).encode("utf-8")).hexdigest()


def _obs_payload(candidate="cand-A", fixture="S1", status="completed", **over):
    """Obs payload."""
    payload = {
        "candidate": candidate,
        "fixture": fixture,
        "input_ids": [1, 2],
        "output_ids": [3],
        "answer": "a",
        "exact_match": True,
        "boundary_found": True,
        "boundary_text_match": True,
        "raw_reason": None,
        "error": None,
        "status": status,
        "input_text": "text",
    }
    payload.update(over)
    return payload


def _journal_bytes(records, header=None, footer=None):
    """Journal bytes."""
    header = header or {
        "schema": PROGRESS_SCHEMA,
        "package_version": PACKAGE_VERSION,
        "configuration_digest": "digest-x",
        "max_total_tokens": None,
    }
    lines = [_canon({"header": header})]
    prev = hashlib.sha256(lines[0].encode("utf-8")).hexdigest()
    for stage, candidate, payload in records:
        body = {"stage": stage, "candidate": candidate, "observation": payload}
        digest = _chain(prev, body)
        lines.append(_canon({**body, "cell_sha256": digest}))
        prev = digest
    if footer is not None:
        lines.append(_canon({"footer": footer}))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _load_crafted(path):
    """Load crafted."""
    return load_progress(
        path,
        expected_digest="digest-x",
        expected_package_version=PACKAGE_VERSION,
        formats=_formats(),
        split=_split(),
    )


_SEARCH_PREFIX = [
    ("search", "cand-A", _obs_payload("cand-A", "S1")),
    ("search", "cand-A", _obs_payload("cand-A", "S2")),
    ("search", "cand-B", _obs_payload("cand-B", "S1")),
    ("search", "cand-B", _obs_payload("cand-B", "S2")),
]


class JournalRefusalTests(unittest.TestCase):
    """Test journal refusal."""

    def _refuses(self, data, code="resume_progress_corrupt"):
        """Refuses."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            path.write_bytes(data)
            with self.assertRaises(ValueError) as caught:
                _load_crafted(path)
            self.assertEqual(str(caught.exception), code)

    def test_empty_journal_refused(self):
        """Test empty journal refused."""
        self._refuses(b"")

    def test_first_line_garbage_refused(self):
        """Test first line garbage refused."""
        self._refuses(b"not json{\n")

    def test_first_line_not_a_header_refused(self):
        """Test first line not a header refused."""
        record = {"stage": "search", "candidate": "cand-A"}
        self._refuses((_canon(record) + "\n").encode("utf-8"))

    def test_header_schema_mismatch_refused(self):
        """Test header schema mismatch refused."""
        header = {
            "schema": "mpl-progress-0",
            "package_version": PACKAGE_VERSION,
            "configuration_digest": "digest-x",
            "max_total_tokens": None,
        }
        self._refuses(_journal_bytes([], header=header), "resume_configuration_mismatch")

    def test_sink_resume_refuses_finalized_journal(self):
        """Test sink resume refuses finalized journal."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            sink = ProgressSink.create(path, header=_header(backend))
            sink.finalize({"decision": "no_candidate"})
            loaded = _load(path, backend)
            with self.assertRaises(ValueError) as caught:
                ProgressSink.resume(path, loaded)
            self.assertEqual(str(caught.exception), "resume_already_complete")

    def test_append_after_close_refused(self):
        """Test append after close refused."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            sink = ProgressSink.create(path, header=_header(backend))
            sink.close()
            sink.close()
            sink.finalize({"decision": "ignored"})
            observation = core.Observation(
                candidate="cand-A",
                fixture="S1",
                input_ids=(1,),
                output_ids=(2,),
                answer="a",
                exact_match=True,
                boundary_found=True,
                boundary_text_match=True,
                raw_reason=None,
                error=None,
                status="completed",
                input_text="text",
            )
            with self.assertRaises(ValueError) as caught:
                sink.append("search", "cand-A", observation)
            self.assertEqual(str(caught.exception), "progress_journal_closed")

    def test_record_field_set_refused(self):
        """Test record field set refused."""
        payload = _obs_payload()
        del payload["answer"]
        self._refuses(_journal_bytes([("search", "cand-A", payload)]))

    def test_record_token_id_type_refused(self):
        """Test record token id type refused."""
        payload = _obs_payload(input_ids=[1, "x"])
        self._refuses(_journal_bytes([("search", "cand-A", payload)]))

    def test_record_status_value_refused(self):
        """Test record status value refused."""
        payload = _obs_payload(status="bogus")
        self._refuses(_journal_bytes([("search", "cand-A", payload)]))

    def test_record_status_list_refused(self):
        """Test record status list refused (BH12-1: unhashable status)."""
        payload = _obs_payload(status=["completed"])
        self._refuses(_journal_bytes([("search", "cand-A", payload)]))

    def test_record_status_dict_refused(self):
        """Test record status dict refused (BH12-1: unhashable status)."""
        payload = _obs_payload(status={"value": "completed"})
        self._refuses(_journal_bytes([("search", "cand-A", payload)]))

    def test_torn_middle_line_refused(self):
        """Test torn middle line refused."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            sink = ProgressSink.create(path, header=_header(backend))
            for candidate, fixture in (
                ("cand-A", "S1"),
                ("cand-A", "S2"),
                ("cand-B", "S1"),
            ):
                sink.append(
                    "search",
                    candidate,
                    core.Observation(
                        candidate=candidate,
                        fixture=fixture,
                        input_ids=(1,),
                        output_ids=(2,),
                        answer="a",
                        exact_match=True,
                        boundary_found=True,
                        boundary_text_match=True,
                        raw_reason=None,
                        error=None,
                        status="completed",
                        input_text="text",
                    ),
                )
            sink.close()
            lines = path.read_text(encoding="utf-8").splitlines()
            lines[2] = "garbage{"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                _load(path, backend)
            self.assertEqual(str(caught.exception), "resume_progress_corrupt")

    def test_footer_not_last_refused(self):
        """Test footer not last refused."""
        data = _journal_bytes(_SEARCH_PREFIX[:1], footer={"complete": True})
        data += (_canon({"footer": {"complete": True}}) + "\n").encode("utf-8")
        self._refuses(data)

    def test_footer_not_a_dict_refused(self):
        """Test footer not a dict refused."""
        header_line = _canon(
            {
                "header": {
                    "schema": PROGRESS_SCHEMA,
                    "package_version": PACKAGE_VERSION,
                    "configuration_digest": "digest-x",
                    "max_total_tokens": None,
                }
            }
        )
        data = (header_line + "\n" + _canon({"footer": 3}) + "\n").encode("utf-8")
        self._refuses(data)

    def test_footer_incomplete_refused(self):
        """Test footer incomplete refused."""
        self._refuses(_journal_bytes([], footer={"complete": False}))

    def test_envelope_candidate_mismatch_refused(self):
        """Test envelope candidate mismatch refused."""
        record = ("search", "cand-B", _obs_payload("cand-A", "S1"))
        self._refuses(_journal_bytes([record]))

    def test_search_order_mismatch_refused(self):
        """Test search order mismatch refused."""
        records = [
            ("search", "cand-A", _obs_payload("cand-A", "S2")),
            ("search", "cand-A", _obs_payload("cand-A", "S1")),
        ]
        self._refuses(_journal_bytes(records))

    def test_record_after_terminal_search_refused(self):
        """Test record after terminal search refused."""
        records = [
            ("search", "cand-A", _obs_payload("cand-A", "S1", status="failed")),
            ("search", "cand-A", _obs_payload("cand-A", "S2")),
        ]
        self._refuses(_journal_bytes(records))

    def test_held_out_unknown_candidate_refused(self):
        """Test held out unknown candidate refused."""
        records = [*_SEARCH_PREFIX, ("held_out", "cand-Z", _obs_payload("cand-Z", "H1"))]
        self._refuses(_journal_bytes(records))

    def test_held_out_candidate_switch_refused(self):
        """Test held out candidate switch refused."""
        records = [
            *_SEARCH_PREFIX,
            ("held_out", "cand-A", _obs_payload("cand-A", "H1")),
            ("held_out", "cand-B", _obs_payload("cand-B", "H2")),
        ]
        self._refuses(_journal_bytes(records))

    def test_held_out_fixture_order_refused(self):
        """Test held out fixture order refused."""
        records = [*_SEARCH_PREFIX, ("held_out", "cand-A", _obs_payload("cand-A", "H2"))]
        self._refuses(_journal_bytes(records))

    def test_held_out_terminal_record_loads(self):
        """Test held out terminal record loads."""
        records = [
            *_SEARCH_PREFIX,
            ("held_out", "cand-A", _obs_payload("cand-A", "H1", status="failed")),
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            path.write_bytes(_journal_bytes(records))
            loaded = _load_crafted(path)
            self.assertEqual(len(loaded.observations), 5)
            terminal = loaded.observations[("held_out", "cand-A", "H1")]
            self.assertEqual(terminal.status, "failed")

    def test_unknown_stage_refused(self):
        """Test unknown stage refused."""
        record = ("diagnosis", "cand-A", _obs_payload("cand-A", "S1"))
        self._refuses(_journal_bytes([record]))


class EvaluatePriorTests(unittest.TestCase):
    """Test evaluate prior."""

    def test_evaluate_rejects_non_callable_callback(self):
        """Test evaluate rejects non callable callback."""
        backend = ScriptedBackend(EXPECTED, BEHAVIOR)
        with self.assertRaises(ValueError) as caught:
            evaluate(
                backend,
                _formats()[0],
                _split().search,
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                on_observation=42,
            )
        self.assertEqual(str(caught.exception), "on_observation must be callable")

    def test_evaluate_rejects_non_mapping_priors(self):
        """Test evaluate rejects non mapping priors."""
        backend = ScriptedBackend(EXPECTED, BEHAVIOR)
        with self.assertRaises(ValueError) as caught:
            evaluate(
                backend,
                _formats()[0],
                _split().search,
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                prior_observations=[("S1", None)],
            )
        self.assertIn("prior_observations must be a mapping", str(caught.exception))

    def test_evaluate_prior_mismatch_refused(self):
        """Test evaluate prior mismatch refused."""
        backend = ScriptedBackend(EXPECTED, BEHAVIOR)
        prior = core.Observation(
            candidate="cand-B",
            fixture="S1",
            input_ids=(1,),
            output_ids=(2,),
            answer="a",
            exact_match=False,
            boundary_found=True,
            boundary_text_match=True,
            raw_reason=None,
            error=None,
            status="completed",
            input_text="text",
        )
        with self.assertRaises(ValueError) as caught:
            evaluate(
                backend,
                _formats()[0],
                _split().search,
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                prior_observations={"S1": prior},
            )
        self.assertEqual(str(caught.exception), "prior observation does not match this evaluation")

    def test_evaluate_terminal_prior_short_circuits(self):
        """Test evaluate terminal prior short circuits."""
        backend = ScriptedBackend(EXPECTED, BEHAVIOR)
        prior = core.Observation(
            candidate="cand-A",
            fixture="S1",
            input_ids=(1,),
            output_ids=(2,),
            answer=None,
            exact_match=False,
            boundary_found=False,
            boundary_text_match=None,
            raw_reason=None,
            error="generate_failed",
            status="failed",
            input_text="text",
        )
        report = evaluate(
            backend,
            _formats()[0],
            _split().search,
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            prior_observations={"S1": prior},
        )
        self.assertEqual(report.status, "failed")
        self.assertEqual(report.observations, (prior,))
        self.assertEqual(backend.generate_calls, 0)


class _BadEncodeBackend(ScriptedBackend):
    """Bad encode backend."""

    def encode(self, text):
        """Encode."""
        return [-1]


class RunValidationTests(unittest.TestCase):
    """Test run validation."""

    def test_run_rejects_non_callable_callback(self):
        """Test run rejects non callable callback."""
        backend = ScriptedBackend(EXPECTED, BEHAVIOR)
        with self.assertRaises(ValueError) as caught:
            _run(backend, on_observation=42)
        self.assertEqual(str(caught.exception), "on_observation must be callable")

    def test_run_rejects_non_mapping_resume(self):
        """Test run rejects non mapping resume."""
        backend = ScriptedBackend(EXPECTED, BEHAVIOR)
        with self.assertRaises(ValueError) as caught:
            _run(backend, resume_observations=[("search", "cand-A", "S1")])
        self.assertEqual(str(caught.exception), "resume_observations must be a mapping")

    def test_preflight_invalid_token_ids_refused(self):
        """Test preflight invalid token ids refused."""
        backend = _BadEncodeBackend(EXPECTED, BEHAVIOR)
        with self.assertRaises(ValueError) as caught:
            _run(backend, max_total_tokens=1000)
        self.assertEqual(str(caught.exception), "run_budget_preflight_failed")


class _DispatchFakeBackend:
    """Dispatch fake backend."""

    fingerprint = "w4-dispatch-fake-backend"
    eos_token_id = None

    def __init__(self, model, device=None, max_context=None, dtype=None):
        """Init."""
        self.model = model


class CliResumeDispatchTests(unittest.TestCase):
    """Test cli resume dispatch."""

    def _fixture_file(self, directory):
        """Fixture file."""
        path = Path(directory) / "fixture.json"
        path.write_text(
            json.dumps(
                {
                    "search": [
                        {
                            "id": "s1",
                            "group": "g1",
                            "messages": [{"role": "user", "content": "hi"}],
                            "expected": "yo",
                        }
                    ],
                    "held_out": [
                        {
                            "id": "h1",
                            "group": "g2",
                            "messages": [{"role": "user", "content": "hey"}],
                            "expected": "yo",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def _argv(self, tmp, fixture, output, *extra):
        """Argv."""
        return [
            "run",
            "--model",
            str(tmp),
            "--dataset",
            str(fixture),
            "--output",
            str(output),
            "--family",
            "chatml",
            "--boundary-ids",
            "999",
            "--max-new-tokens",
            "40",
            "--seed",
            "7",
            *extra,
        ]

    def _invoke(self, argv):
        """Invoke."""
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = cli_module.main(argv)
        except SystemExit as exc:
            return ("exit", exc.code, err.getvalue())
        return ("return", code, err.getvalue())

    def test_resume_without_journal_is_usage_error(self):
        """Test resume without journal is usage error."""
        with TemporaryDirectory() as tmp:
            fixture = self._fixture_file(tmp)
            output = Path(tmp) / "out"
            output.mkdir()
            kind, code, err = self._invoke(self._argv(tmp, fixture, output, "--resume"))
            self.assertEqual((kind, code), ("exit", 2))
            self.assertIn("progress.jsonl", err)

    def _dispatch_resume_flow(self, tmp, *, version_patch):
        """Dispatch resume flow."""
        fixture = self._fixture_file(tmp)
        output = Path(tmp) / "out"
        canned = _run(ScriptedBackend(EXPECTED, BEHAVIOR))
        journaled = []

        def fresh_stub(*args, **kwargs):
            """Fresh stub."""
            kwargs["on_observation"](
                "search",
                "chatml",
                core.Observation(
                    candidate="chatml",
                    fixture="s1",
                    input_ids=(1, 2),
                    output_ids=(3,),
                    answer="yo",
                    exact_match=True,
                    boundary_found=True,
                    boundary_text_match=True,
                    raw_reason=None,
                    error=None,
                    status="completed",
                    input_text="hi",
                ),
            )
            return canned

        def resume_stub(*args, **kwargs):
            """Resume stub."""
            journaled.append(kwargs["resume_observations"])
            return canned

        patches = [
            mock.patch.object(backends_module, "LocalHFBackend", _DispatchFakeBackend),
        ]
        if version_patch:
            patches.append(
                mock.patch.object(
                    importlib.metadata,
                    "version",
                    side_effect=importlib.metadata.PackageNotFoundError("nope"),
                )
            )
        for patcher in patches:
            patcher.start()
        try:
            with mock.patch.object(cli_module, "run_experiment", fresh_stub):
                kind, code, _ = self._invoke(self._argv(tmp, fixture, output))
            self.assertEqual((kind, code), ("return", 0))
            journal = output / "progress.jsonl"
            lines = journal.read_text(encoding="utf-8").splitlines()
            self.assertIn('"footer"', lines[-1])
            (output / "results.json").unlink()
            (output / "report.html").unlink()
            journal.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8", newline="")
            with mock.patch.object(cli_module, "run_experiment", resume_stub):
                kind, code, _ = self._invoke(self._argv(tmp, fixture, output, "--resume"))
            self.assertEqual((kind, code), ("return", 0))
        finally:
            for patcher in patches:
                patcher.stop()
        self.assertEqual(len(journaled), 1)
        resumed = journaled[0]
        self.assertIsNotNone(resumed)
        self.assertIn(("search", "chatml", "s1"), resumed)
        lines = journal.read_text(encoding="utf-8").splitlines()
        self.assertIn('"footer"', lines[-1])
        self.assertIn('"complete":true', lines[-1])
        self.assertTrue((output / "results.json").is_file())

    def test_cli_resume_dispatch_in_process(self):
        """Test cli resume dispatch in process."""
        with TemporaryDirectory() as tmp:
            self._dispatch_resume_flow(tmp, version_patch=False)

    def test_cli_resume_with_unavailable_package_version(self):
        """Test cli resume with unavailable package version."""
        with TemporaryDirectory() as tmp:
            self._dispatch_resume_flow(tmp, version_patch=True)


if __name__ == "__main__":
    unittest.main()
