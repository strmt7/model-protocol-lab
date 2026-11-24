"""W4-1 (RR-10) — checkpoint-template rendering lives in an owned worker.

Before W4, ``doctor`` rendered checkpoint-controlled Jinja in-process
under a SIGALRM watchdog and a transient RLIMIT_AS cap. Both guards
degrade *silently* exactly where containment matters most: off the main
thread or off Unix the watchdog cannot arm and the render runs
unbounded in the host process (RR-10). The W4 design moves every render
(and every checkpoint-tokenizer encode) into a per-invocation worker
process (`model_protocol_lab.render_worker`): the parent enforces a
per-request deadline on every platform and thread, a memory bomb can
kill at most the worker, worker death degrades the pair to
``"unavailable"``, and a worker that cannot start — or dies three times
in a row — fails the section closed with the one new detail code
``render_worker_failed`` (worker-or-refuse; no in-process fallback).

These tests pin that contract:

* golden renders: worker output is byte-identical to a direct
  in-process ``apply_chat_template`` reference, across worker restarts;
* the hostile battery at section level: CPU-burn and memory-bomb
  templates degrade the pair, never the parent — including from a
  non-main thread, where the pre-W4 watchdog never armed (the RR-10
  regression proof);
* the render path consults no signal machinery in the parent;
* the wire protocol against scripted fake workers: single death ->
  respawn and continue; three consecutive deaths -> the circuit
  breaker; oversize / truncated / garbage frames and a sleeping worker
  all degrade in bounded time, never hang, never read unboundedly;
* the emitted detail vocabulary gains exactly one code:
  ``render_worker_failed``.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from model_protocol_lab import doctor
from model_protocol_lab.core import Fixture, Format, Message

CLEAN_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

LLAMA_STYLE_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message.role == 'user' %}"
    "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message.content }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)

# Each range() literal sits inside both bounds that apply to a real
# render — the firewall's 1_000_000 literal cap and the transformers
# sandbox's MAX_RANGE of 100_000; the nesting is the burn (10**10
# inner steps, ~1000 s of CPU — effectively unbounded).
BURN_TEMPLATE = (
    "{% for i in range(100000) %}"
    "{% for j in range(100000) %}"
    "{% set x = i + j %}"
    "{% endfor %}{% endfor %}done"
)

BOMB_TEMPLATE = "{{ 'x' * 1000000000 }}"

ERROR_TEMPLATE = "{{ 1 / 0 }}"

# The complete pre-W4 detail vocabulary of template_conformance, from
# the section's own return sites. W4 may add exactly one code.
PRE_W4_DETAILS = {
    "",
    "tokenizer_config_unreadable",
    "no_chat_template_in_config",
    "chat_template_too_large",
    "template_firewall_denied",
    "transformers_not_installed",
    "tokenizer_load_failed",
}


def _write_checkpoint(root: Path, template: str | None, bos: str | None = None) -> Path:
    """Write checkpoint."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    root.mkdir(parents=True, exist_ok=True)
    vocab = {f"tok{i}": i for i in range(15)}
    vocab["[UNK]"] = 15
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.add_special_tokens(["[UNK]", "[BOS]", "[EOS]"])
    tokenizer.save(str(root / "tokenizer.json"))
    config: dict[str, object] = {}
    if template is not None:
        config["chat_template"] = template
    if bos is not None:
        config["bos_token"] = bos
    (root / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


def _fmt() -> Format:
    """Fmt."""
    return Format(
        name="chatml",
        bos="",
        user_prefix="u:",
        assistant_prefix="a:",
        turn_end="\n",
        repeat_bos=False,
        user_turn_end="\n",
        assistant_turn_end="\n",
        open_suffix="a:",
    )


def _case(case_id: str = "f1", content: str = "hi") -> Fixture:
    """Case."""
    return Fixture(
        id=case_id,
        messages=(Message(role="user", content=content),),
        expected="",
        group="g",
    )


def _conformance(root: Path, cases: tuple[Fixture, ...]) -> dict:
    """Conformance."""
    return doctor.template_conformance(root, (_fmt(),), cases)


FAKE_WORKER = """
import importlib.metadata
import json
import struct
import sys
from pathlib import Path

mode = sys.argv[1]
counter = Path(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0

if mode == "exit_now":
    sys.exit(3)

def read_frame():
    header = sys.stdin.buffer.read(4)
    if len(header) < 4:
        return None
    (n,) = struct.unpack(">I", header)
    body = sys.stdin.buffer.read(n)
    if len(body) < n:
        return None
    return json.loads(body)

def send(obj):
    body = json.dumps(obj).encode()
    sys.stdout.buffer.write(struct.pack(">I", len(body)) + body)
    sys.stdout.buffer.flush()

def real_versions():
    return {
        "python": "fake",
        "transformers": importlib.metadata.version("transformers"),
        "jinja2": importlib.metadata.version("jinja2"),
    }

while True:
    frame = read_frame()
    if frame is None:
        break
    op = frame.get("op")
    if op == "init":
        if mode == "init_weird_code":
            send({"status": "init_error", "code": "weird_code"})
            continue
        if mode == "init_mystery":
            send({"status": "mystery"})
            continue
        if mode == "version_liar":
            versions = real_versions()
            versions["transformers"] = "0.0.0"
            send({"status": "ok", "versions": versions, "tokenizer_vocab_size": 16})
            continue
        if mode == "version_missing":
            send({"status": "ok", "tokenizer_vocab_size": 16})
            continue
        send({"status": "ok", "versions": real_versions(), "tokenizer_vocab_size": 16})
        if mode == "die_after_init":
            sys.exit(3)
        continue
    if mode == "die_after" and op == "render":
        seen = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(seen + 1))
        if seen + 1 > limit:
            sys.exit(3)
    if mode == "die_once" and not counter.exists():
        counter.write_text("1")
        sys.exit(3)
    if mode == "die_always":
        sys.exit(3)
    if mode == "error_replies":
        if op == "render":
            send({"status": "error", "code": "render_failed"})
        elif op == "encode":
            send({"status": "error", "code": "encode_failed"})
        else:
            send({"status": "error", "code": "unknown_op"})
        continue
    if mode == "bad_types":
        if op == "render":
            send({"status": "ok", "text": 42})
        elif op == "encode":
            send({"status": "ok", "ids": "nope"})
        else:
            send({"status": "error", "code": "unknown_op"})
        continue
    if mode == "oversize_ok":
        if op == "render":
            send({"status": "ok", "text": "x" * 1000001})
        elif op == "encode":
            send({"status": "ok", "ids": [1, 2, 3]})
        else:
            send({"status": "error", "code": "unknown_op"})
        continue
    if mode == "sleeper":
        import time
        time.sleep(30)
        send({"status": "ok", "text": "late"})
        continue
    if mode == "oversize":
        sys.stdout.buffer.write(struct.pack(">I", 64 * 1024 * 1024))
        sys.stdout.buffer.flush()
        import time
        time.sleep(30)
        continue
    if mode == "garbage":
        sys.stdout.buffer.write(b"\\xff\\xff\\xff\\xffjunkjunk")
        sys.stdout.buffer.flush()
        continue
    if mode == "truncate" and not counter.exists():
        counter.write_text("1")
        body = json.dumps({"status": "ok", "text": "half"}).encode()
        sys.stdout.buffer.write(struct.pack(">I", len(body)) + body[:3])
        sys.stdout.buffer.flush()
        sys.exit(3)
    if op == "render":
        send({"status": "ok", "text": "RENDERED"})
    elif op == "encode":
        send({"status": "ok", "ids": [1, 2, 3]})
    else:
        send({"status": "error", "code": "unknown_op"})
"""


def _fake_argv(mode: str, counter: Path, script: Path, limit: int = 0) -> tuple[str, ...]:
    """Fake argv."""
    return (sys.executable, str(script), mode, str(counter), str(limit))


class GoldenRenderTests(unittest.TestCase):
    """Worker renders are byte-identical to the in-process reference."""

    def _reference(self, root: Path, template: str, bos: str | None, conversations):
        """Reference."""
        from transformers import PreTrainedTokenizerFast

        factory = PreTrainedTokenizerFast
        tokenizer = factory(
            tokenizer_file=str(root / "tokenizer.json"),
            clean_up_tokenization_spaces=False,
        )
        tokenizer.chat_template = template
        if bos:
            tokenizer.bos_token = bos
        return [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations
        ]

    def test_worker_renders_match_reference_across_restarts(self):
        """Test worker renders match reference across restarts."""
        from model_protocol_lab.render_worker import RenderSession

        conversations = [
            [{"role": "user", "content": "hello world"}],
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
            [{"role": "user", "content": ""}],
        ]
        for template, bos in ((CLEAN_TEMPLATE, None), (LLAMA_STYLE_TEMPLATE, "[BOS]")):
            with tempfile.TemporaryDirectory() as tmp:
                root = _write_checkpoint(Path(tmp), template, bos=bos)
                expected = self._reference(root, template, bos, conversations)
                for _restart in range(2):
                    with RenderSession(str(root / "tokenizer.json"), template, bos) as session:
                        session.start()
                        got = [session.render(c) for c in conversations]
                    self.assertEqual(got, expected)

    def test_worker_encode_matches_reference(self):
        """Test worker encode matches reference."""
        from model_protocol_lab.render_worker import RenderSession

        with tempfile.TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), CLEAN_TEMPLATE)
            with RenderSession(str(root / "tokenizer.json"), CLEAN_TEMPLATE, None) as session:
                session.start()
                ids = session.encode("<|im_start|>user\nhi<|im_end|>\n")
            from transformers import PreTrainedTokenizerFast

            factory = PreTrainedTokenizerFast
            tokenizer = factory(
                tokenizer_file=str(root / "tokenizer.json"),
                clean_up_tokenization_spaces=False,
            )
            self.assertEqual(
                list(ids),
                tokenizer.encode("<|im_start|>user\nhi<|im_end|>\n", add_special_tokens=False),
            )


class SectionContainmentTests(unittest.TestCase):
    """Hostile templates degrade pairs, never the parent process."""

    def test_clean_template_pairs_ok(self):
        """Test clean template pairs ok."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), CLEAN_TEMPLATE)
            out = _conformance(root, (_case(),))
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["diffs"][0]["status"], "ok")

    def test_cpu_burn_pair_unavailable_in_bounded_time(self):
        """Test cpu burn pair unavailable in bounded time."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), BURN_TEMPLATE)
            start = time.monotonic()
            out = _conformance(root, (_case(),))
            elapsed = time.monotonic() - start
        self.assertEqual(out["diffs"][0]["status"], "unavailable")
        self.assertLess(elapsed, 40.0)

    def test_render_error_pair_unavailable(self):
        """Test render error pair unavailable."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), ERROR_TEMPLATE)
            out = _conformance(root, (_case(),))
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["diffs"][0]["status"], "unavailable")

    @unittest.skipUnless(os.name == "posix", "resource/RSS measurement is POSIX-only")
    def test_memory_bomb_pair_unavailable_parent_rss_bounded(self):
        """Test memory bomb pair unavailable parent rss bounded."""
        import resource

        def rss_mb() -> float:
            """Rss mb."""
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        with tempfile.TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), BOMB_TEMPLATE)
            before = rss_mb()
            out = _conformance(root, (_case(),))
            growth = rss_mb() - before
        self.assertEqual(out["diffs"][0]["status"], "unavailable")
        self.assertLess(growth, 100.0)

    def test_off_main_thread_deadline_enforced(self):
        """Test off main thread deadline enforced."""
        # The RR-10 regression: before W4 the SIGALRM watchdog could not
        # arm off the main thread, so this render ran unbounded.
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), BURN_TEMPLATE)
            holder: dict[str, object] = {}

            def run() -> None:
                """Run."""
                holder["out"] = _conformance(root, (_case(),))

            start = time.monotonic()
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            thread.join(timeout=45.0)
            elapsed = time.monotonic() - start
        self.assertFalse(thread.is_alive(), "conformance hung off the main thread")
        self.assertLess(elapsed, 45.0)
        out = holder["out"]
        self.assertEqual(out["diffs"][0]["status"], "unavailable")  # type: ignore[index]

    def test_render_path_uses_no_signal_machinery(self):
        """Test render path uses no signal machinery."""
        # The pre-W4 parent armed SIGALRM via platform_caps.alarm_deadline,
        # whose unguarded getsignal/getitimer save raises here and degrades
        # every pair. The worker path never consults signals in the parent.
        import signal

        def boom(*args: object, **kwargs: object) -> None:
            """Boom."""
            raise AssertionError("render path consulted signal machinery")

        with tempfile.TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), CLEAN_TEMPLATE)
            # Patch only the attributes the platform has: getitimer /
            # setitimer do not exist on Windows, where patch.object
            # raises AttributeError before the assertion can run
            # (RR-18 follow-up). The render path cannot consult
            # machinery that does not exist, so the proof is the
            # same: every signal entry point the platform offers is
            # armed to explode, and the render still succeeds.
            armed = [
                mock.patch.object(signal, name, boom)
                for name in ("getsignal", "getitimer", "setitimer")
                if hasattr(signal, name)
            ]
            with contextlib.ExitStack() as stack:
                for patcher in armed:
                    stack.enter_context(patcher)
                out = _conformance(root, (_case(),))
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["diffs"][0]["status"], "ok")

    def test_probe_section_served_by_worker(self):
        """Test probe section served by worker."""
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), CLEAN_TEMPLATE)
            section = doctor.template_probe_section(root)
        self.assertEqual(section["status"], "ok")
        self.assertGreater(section["renders_used"], 0)


class FakeWorkerProtocolTests(unittest.TestCase):
    """The wire protocol degrades in bounded time against hostile workers."""

    def setUp(self):
        """Set up."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.script = self.root / "fake_worker.py"
        self.script.write_text(FAKE_WORKER, encoding="utf-8")
        self.counter = self.root / "counter"
        _write_checkpoint(self.root / "ckpt", CLEAN_TEMPLATE)

    def _patched_argv(self, mode: str):
        """Patched argv."""
        from model_protocol_lab import render_worker

        return mock.patch.object(
            render_worker, "WORKER_ARGV", _fake_argv(mode, self.counter, self.script)
        )

    def _session(self, budget_s: float = 10.0):
        """Session."""
        from model_protocol_lab.render_worker import RenderSession

        session = RenderSession(
            str(self.root / "ckpt" / "tokenizer.json"),
            CLEAN_TEMPLATE,
            None,
            budget_s=budget_s,
        )
        session.start()
        return session

    def test_single_death_respawns_and_next_request_succeeds(self):
        """Test single death respawns and next request succeeds."""
        from model_protocol_lab.render_worker import WorkerDied

        with self._patched_argv("die_once"), self._session() as session:
            with self.assertRaises(WorkerDied):
                session.render([{"role": "user", "content": "hi"}])
            self.assertEqual(session.render([{"role": "user", "content": "hi"}]), "RENDERED")

    def test_single_death_degrades_one_pair_only(self):
        """Test single death degrades one pair only."""
        with self._patched_argv("die_once"):
            out = _conformance(self.root / "ckpt", (_case("f1"), _case("f2")))
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["diffs"][0]["status"], "unavailable")
        self.assertEqual(out["diffs"][1]["status"], "ok")

    def test_three_consecutive_deaths_trip_breaker(self):
        """Test three consecutive deaths trip breaker."""
        from model_protocol_lab.render_worker import WorkerDied, WorkerUnavailable

        with self._patched_argv("die_always"), self._session() as session:
            with self.assertRaises(WorkerDied):
                session.render([{"role": "user", "content": "hi"}])
            with self.assertRaises(WorkerDied):
                session.render([{"role": "user", "content": "hi"}])
            with self.assertRaises(WorkerUnavailable) as caught:
                session.render([{"role": "user", "content": "hi"}])
        self.assertEqual(caught.exception.code, "render_worker_failed")

    def test_breaker_trip_fails_section_with_new_detail(self):
        """Test breaker trip fails section with new detail."""
        # Deaths accumulate across pairs (each pair's first request
        # kills the fresh worker and degrades that pair); the third
        # consecutive death trips the breaker mid-sweep.
        with self._patched_argv("die_always"):
            out = _conformance(self.root / "ckpt", (_case("f1"), _case("f2"), _case("f3")))
        self.assertEqual(out["status"], "unavailable")
        self.assertEqual(out["detail"], "render_worker_failed")
        self.assertEqual(out["diffs"], [])

    def test_breaker_trip_fails_probe_with_new_detail(self):
        """Test breaker trip fails probe with new detail."""
        with self._patched_argv("die_always"):
            section = doctor.template_probe_section(self.root / "ckpt")
        self.assertEqual(section["status"], "unavailable")
        self.assertEqual(section["detail"], "render_worker_failed")

    def test_oversize_frame_announcement_is_a_bounded_death(self):
        """Test oversize frame announcement is a bounded death."""
        from model_protocol_lab.render_worker import WorkerDied

        with self._patched_argv("oversize"), self._session() as session:
            start = time.monotonic()
            with self.assertRaises(WorkerDied):
                session.render([{"role": "user", "content": "hi"}])
            self.assertLess(time.monotonic() - start, 10.0)

    def test_garbage_frame_is_a_death(self):
        """Test garbage frame is a death."""
        from model_protocol_lab.render_worker import WorkerDied

        with self._patched_argv("garbage"), self._session() as session:
            with self.assertRaises(WorkerDied):
                session.render([{"role": "user", "content": "hi"}])
            # A respawned garbage worker dies again — a second death,
            # still short of the breaker.
            with self.assertRaises(WorkerDied):
                session.render([{"role": "user", "content": "hi"}])

    def test_truncated_frame_is_a_death_then_respawn_serves(self):
        """Test truncated frame is a death then respawn serves."""
        from model_protocol_lab.render_worker import WorkerDied

        with self._patched_argv("truncate"), self._session() as session:
            with self.assertRaises(WorkerDied):
                session.render([{"role": "user", "content": "hi"}])
            self.assertEqual(session.render([{"role": "user", "content": "hi"}]), "RENDERED")

    def test_sleeping_worker_hits_request_deadline(self):
        """Test sleeping worker hits request deadline."""
        from model_protocol_lab.render_worker import WorkerDied

        with self._patched_argv("sleeper"), self._session(budget_s=1.0) as session:
            start = time.monotonic()
            with self.assertRaises(WorkerDied):
                session.render([{"role": "user", "content": "hi"}])
            self.assertLess(time.monotonic() - start, 10.0)


class WorkerUnavailableMappingTests(unittest.TestCase):
    """Worker-or-refuse: no in-process fallback, one new detail code."""

    def test_spawn_failure_is_render_worker_failed(self):
        """Test spawn failure is render worker failed."""
        from model_protocol_lab import render_worker

        with tempfile.TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), CLEAN_TEMPLATE)
            with mock.patch.object(
                render_worker, "WORKER_ARGV", ("/nonexistent/mpl-worker-binary",)
            ):
                out = _conformance(root, (_case(),))
        self.assertEqual(out["status"], "unavailable")
        self.assertEqual(out["detail"], "render_worker_failed")
        self.assertEqual(out["diffs"], [])

    def test_detail_vocabulary_gains_exactly_one_code(self):
        """Test detail vocabulary gains exactly one code."""
        from model_protocol_lab import render_worker

        seen: set[str] = set()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name, template in (
                ("clean", CLEAN_TEMPLATE),
                ("error", ERROR_TEMPLATE),
            ):
                out = _conformance(_write_checkpoint(base / name, template), (_case(),))
                seen.add(str(out.get("detail", "")))
                for entry in out["diffs"]:
                    self.assertIn(entry["status"], ("ok", "unavailable"))
            with mock.patch.object(
                render_worker, "WORKER_ARGV", ("/nonexistent/mpl-worker-binary",)
            ):
                out = _conformance(base / "clean", (_case(),))
                seen.add(str(out.get("detail", "")))
        self.assertLessEqual(seen, PRE_W4_DETAILS | {"render_worker_failed"})
        self.assertIn("render_worker_failed", seen)


class WorkerUnitTests(unittest.TestCase):
    """The worker's render helper keeps the pre-W4 render checks."""

    def test_render_once_returns_text(self):
        """Test render once returns text."""
        from model_protocol_lab.render_worker import _render_once

        class OkTokenizer:
            """Ok tokenizer."""

            @staticmethod
            def apply_chat_template(conversation, tokenize=False, add_generation_prompt=True):
                """Apply chat template."""
                return "<|user|>hi<|assistant|>"

        self.assertEqual(
            _render_once(OkTokenizer(), [{"role": "user", "content": "hi"}]),
            "<|user|>hi<|assistant|>",
        )

    def test_render_once_rejects_non_str(self):
        """Test render once rejects non str."""
        from model_protocol_lab.render_worker import _render_once

        class WeirdTokenizer:
            """Weird tokenizer."""

            @staticmethod
            def apply_chat_template(conversation, tokenize=False, add_generation_prompt=True):
                """Apply chat template."""
                return ["not", "a", "string"]

        with self.assertRaises(ValueError):
            _render_once(WeirdTokenizer(), [])

    @unittest.skipUnless(
        os.name == "posix",
        "POSIX-only: the address-space cap does not exist on win32",
    )
    def test_render_once_bomb_raises_memory_error(self):
        """Test render once bomb raises memory error."""
        from model_protocol_lab.render_worker import _render_once

        class BombTokenizer:
            """Bomb tokenizer."""

            @staticmethod
            def apply_chat_template(conversation, tokenize=False, add_generation_prompt=True):
                """Apply chat template."""
                return "x" * 500_000_000

        with self.assertRaises(MemoryError):
            _render_once(BombTokenizer(), [{"role": "user", "content": "hi"}])


class FrameCodecTests(unittest.TestCase):
    """The frame codec's boundary shapes, exercised in-process."""

    def test_read_frame_round_trip(self):
        """Test read frame round trip."""
        from model_protocol_lab import render_worker as rw

        stream = io.BytesIO(rw._encode_frame({"op": "render", "n": 3}))
        self.assertEqual(rw._read_frame(stream), {"op": "render", "n": 3})

    def test_read_frame_eof_is_none(self):
        """Test read frame eof is none."""
        from model_protocol_lab import render_worker as rw

        self.assertIsNone(rw._read_frame(io.BytesIO(b"")))

    def test_read_frame_ceiling_announcement_is_none(self):
        """Test read frame ceiling announcement is none."""
        from model_protocol_lab import render_worker as rw

        stream = io.BytesIO(struct.pack(">I", rw._MAX_FRAME_BYTES + 1))
        self.assertIsNone(rw._read_frame(stream))

    def test_read_frame_truncated_body_is_none(self):
        """Test read frame truncated body is none."""
        from model_protocol_lab import render_worker as rw

        stream = io.BytesIO(struct.pack(">I", 10) + b"abc")
        self.assertIsNone(rw._read_frame(stream))

    def test_read_frame_non_dict_is_none(self):
        """Test read frame non dict is none."""
        from model_protocol_lab import render_worker as rw

        body = b"[1, 2]"
        stream = io.BytesIO(struct.pack(">I", len(body)) + body)
        self.assertIsNone(rw._read_frame(stream))

    def test_read_frame_malformed_json_raises(self):
        """Test read frame malformed json raises."""
        from model_protocol_lab import render_worker as rw

        body = b"{not json"
        stream = io.BytesIO(struct.pack(">I", len(body)) + body)
        with self.assertRaises(json.JSONDecodeError):
            rw._read_frame(stream)

    def test_frame_reader_garbage_json_closes(self):
        """Test frame reader garbage json closes."""
        from model_protocol_lab import render_worker as rw

        body = b"{not json"
        stream = io.BytesIO(struct.pack(">I", len(body)) + body)
        inbox = queue.Queue()
        reader = rw._FrameReader(stream, inbox)
        reader.start()
        kind, _value = inbox.get(timeout=10)
        self.assertEqual(kind, "closed")

    def test_frame_reader_frame_then_eof(self):
        """Test frame reader frame then eof."""
        from model_protocol_lab import render_worker as rw

        stream = io.BytesIO(rw._encode_frame({"status": "ok"}))
        inbox = queue.Queue()
        reader = rw._FrameReader(stream, inbox)
        reader.start()
        kind, value = inbox.get(timeout=10)
        self.assertEqual((kind, value), ("frame", {"status": "ok"}))
        kind, _value = inbox.get(timeout=10)
        self.assertEqual(kind, "closed")


class SessionEdgeTests(unittest.TestCase):
    """Parent-side session edges: reply shapes, latches, dead pipes."""

    def setUp(self):
        """Set up."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.script = self.root / "fake_worker.py"
        self.script.write_text(FAKE_WORKER, encoding="utf-8")
        self.counter = self.root / "counter"
        _write_checkpoint(self.root / "ckpt", CLEAN_TEMPLATE)

    def _session(self, mode: str | None = None, **kwargs):
        """Session."""
        from model_protocol_lab.render_worker import RenderSession

        argv = (
            _fake_argv(mode, self.counter, self.script)
            if mode is not None
            else kwargs.pop("argv", None)
        )
        return RenderSession(
            str(self.root / "ckpt" / "tokenizer.json"),
            CLEAN_TEMPLATE,
            None,
            argv=argv,
            **kwargs,
        )

    def test_error_replies_raise_render_failed(self):
        """Test error replies raise render failed."""
        from model_protocol_lab.render_worker import RenderFailed

        with self._session("error_replies") as session:
            session.start()
            with self.assertRaises(RenderFailed) as caught:
                session.render([{"role": "user", "content": "hi"}])
            self.assertEqual(str(caught.exception), "render_failed")
            with self.assertRaises(RenderFailed) as caught:
                session.encode("hi")
            self.assertEqual(str(caught.exception), "encode_failed")

    def test_bad_reply_types_raise_render_failed(self):
        """Test bad reply types raise render failed."""
        from model_protocol_lab.render_worker import RenderFailed

        with self._session("bad_types") as session:
            session.start()
            with self.assertRaises(RenderFailed):
                session.render([{"role": "user", "content": "hi"}])
            with self.assertRaises(RenderFailed):
                session.encode("hi")

    def test_render_after_close_is_refused(self):
        """Test render after close is refused."""
        from model_protocol_lab.render_worker import RenderWorkerError

        with self._session("error_replies") as session:
            session.start()
        with self.assertRaises(RenderWorkerError) as caught:
            session.render([{"role": "user", "content": "hi"}])
        self.assertEqual(str(caught.exception), "render session is closed")

    def test_spawn_failure_latches(self):
        """Test spawn failure latches."""
        from model_protocol_lab.render_worker import WorkerUnavailable

        session = self._session(argv=("/nonexistent/mpl-worker-binary",))
        with self.assertRaises(WorkerUnavailable) as caught:
            session.start()
        self.assertEqual(caught.exception.code, "render_worker_failed")
        with self.assertRaises(WorkerUnavailable) as caught:
            session.render([{"role": "user", "content": "hi"}])
        self.assertEqual(caught.exception.code, "render_worker_failed")

    def test_pipeless_spawn_is_render_worker_failed(self):
        """Test pipeless spawn is render worker failed."""
        from model_protocol_lab import render_worker as rw

        class StubProc:
            """Stub proc."""

            stdout = None
            stdin = None

            def kill(self) -> None:
                """Kill."""

            def wait(self, timeout: float | None = None) -> int:
                """Wait."""
                return 0

        session = self._session()
        with (
            mock.patch.object(rw.subprocess, "Popen", return_value=StubProc()),
            self.assertRaises(rw.WorkerUnavailable) as caught,
        ):
            session.start()
        self.assertEqual(caught.exception.code, "render_worker_failed")

    def test_init_send_failure_is_render_worker_failed(self):
        """Test init send failure is render worker failed."""
        from model_protocol_lab import render_worker as rw

        class RaisingStdin:
            """Raising stdin."""

            @staticmethod
            def write(data: bytes) -> int:
                """Write."""
                raise OSError("pipe gone")

            @staticmethod
            def flush() -> None:
                """Flush."""

        class StubProc:
            """Stub proc."""

            def __init__(self) -> None:
                """Init."""
                self.stdin = RaisingStdin()
                self.stdout = io.BytesIO()

            def kill(self) -> None:
                """Kill."""

            def wait(self, timeout: float | None = None) -> int:
                """Wait."""
                return 0

        session = self._session()
        with (
            mock.patch.object(rw.subprocess, "Popen", return_value=StubProc()),
            self.assertRaises(rw.WorkerUnavailable) as caught,
        ):
            session.start()
        self.assertEqual(caught.exception.code, "render_worker_failed")

    def test_init_eof_is_render_worker_failed(self):
        """Test init eof is render worker failed."""
        from model_protocol_lab.render_worker import WorkerUnavailable

        session = self._session("exit_now")
        with self.assertRaises(WorkerUnavailable) as caught:
            session.start()
        self.assertEqual(caught.exception.code, "render_worker_failed")

    def test_init_error_unknown_code_maps_to_load_failed(self):
        """Test init error unknown code maps to load failed."""
        from model_protocol_lab.render_worker import WorkerUnavailable

        session = self._session("init_weird_code")
        with self.assertRaises(WorkerUnavailable) as caught:
            session.start()
        self.assertEqual(caught.exception.code, "tokenizer_load_failed")

    def test_init_mystery_status_is_render_worker_failed(self):
        """Test init mystery status is render worker failed."""
        from model_protocol_lab.render_worker import WorkerUnavailable

        session = self._session("init_mystery")
        with self.assertRaises(WorkerUnavailable) as caught:
            session.start()
        self.assertEqual(caught.exception.code, "render_worker_failed")

    def test_version_liar_is_render_worker_failed(self):
        """Test version liar is render worker failed."""
        from model_protocol_lab.render_worker import WorkerUnavailable

        session = self._session("version_liar")
        with self.assertRaises(WorkerUnavailable) as caught:
            session.start()
        self.assertEqual(caught.exception.code, "render_worker_failed")

    def test_version_missing_is_render_worker_failed(self):
        """Test version missing is render worker failed."""
        from model_protocol_lab.render_worker import WorkerUnavailable

        session = self._session("version_missing")
        with self.assertRaises(WorkerUnavailable) as caught:
            session.start()
        self.assertEqual(caught.exception.code, "render_worker_failed")

    def test_parent_unreadable_version_is_not_checked(self):
        """Test parent unreadable version is not checked."""
        from model_protocol_lab import render_worker as rw

        real_version = rw.metadata.version

        def flaky(name: str) -> str:
            """Flaky."""
            if name == "jinja2":
                raise rw.metadata.PackageNotFoundError(name)
            return real_version(name)

        with (
            mock.patch.object(rw.metadata, "version", flaky),
            self._session("error_replies") as session,
        ):
            session.start()
            with self.assertRaises(rw.RenderFailed):
                session.render([{"role": "user", "content": "hi"}])

    def test_oversize_request_frame_refused_before_send(self):
        """Test oversize request frame refused before send."""
        from model_protocol_lab.render_worker import RenderFailed

        with self._session("error_replies") as session:
            session.start()
            with self.assertRaises(RenderFailed) as caught:
                session.render([{"role": "user", "content": "x" * 40_000_000}])
        self.assertEqual(str(caught.exception), "request_frame_too_large")

    def test_write_to_dead_worker_is_a_death(self):
        """Test write to dead worker is a death."""
        from model_protocol_lab.render_worker import WorkerDied

        with self._session("error_replies") as session:
            session.start()
            proc = session._proc
            assert proc is not None
            proc.kill()
            proc.wait()
            with self.assertRaises(WorkerDied):
                session.render([{"role": "user", "content": "hi"}])

    def test_dead_worker_degrades_under_sigpipe_dfl(self):
        """Test dead worker degrades under sigpipe dfl."""
        # cli.main() resets SIGPIPE to SIG_DFL (its stdout pipeline
        # convention); under that disposition, writing a request into
        # a dead worker's stdin would kill the whole process with
        # SIGPIPE instead of raising BrokenPipeError. The session must
        # notice the death before writing. (Gate evidence: the first
        # full-suite run of this file died with exit 141 at the plain
        # dead-worker write test, after an in-process cli.main() test
        # had latched SIG_DFL.)
        import signal

        if not hasattr(signal, "SIGPIPE"):
            self.skipTest("SIGPIPE does not exist on this platform")
        from model_protocol_lab.render_worker import WorkerDied

        previous = signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        try:
            with self._session("error_replies") as session:
                session.start()
                proc = session._proc
                assert proc is not None
                proc.kill()
                proc.wait()
                with self.assertRaises(WorkerDied):
                    session.render([{"role": "user", "content": "hi"}])
        finally:
            signal.signal(signal.SIGPIPE, previous)

    def test_request_without_worker_is_refused(self):
        """Test request without worker is refused."""
        from model_protocol_lab.render_worker import RenderWorkerError

        session = self._session("error_replies")
        with (
            mock.patch.object(session, "_ensure_worker", lambda: None),
            self.assertRaises(RenderWorkerError) as caught,
        ):
            session.render([{"role": "user", "content": "hi"}])
        self.assertEqual(str(caught.exception), "render worker is not running")


class WorkerSideUnitTests(unittest.TestCase):
    """The worker's init/reply/main paths, driven in-process."""

    def test_init_worker_transformers_missing(self):
        """Test init worker transformers missing."""
        from model_protocol_lab import render_worker as rw

        with mock.patch.dict(sys.modules, {"transformers": None}):
            tokenizer, reply = rw._init_worker(
                {"tokenizer_file": "x", "template": "t", "bos_token": None}
            )
        self.assertIsNone(tokenizer)
        self.assertEqual(reply, {"status": "init_error", "code": "transformers_not_installed"})

    def test_init_worker_bad_file_is_load_failed(self):
        """Test init worker bad file is load failed."""
        from model_protocol_lab import render_worker as rw

        tokenizer, reply = rw._init_worker(
            {"tokenizer_file": "/nonexistent/tokenizer.json", "template": "t", "bos_token": None}
        )
        self.assertIsNone(tokenizer)
        self.assertEqual(reply, {"status": "init_error", "code": "tokenizer_load_failed"})

    def test_init_worker_success(self):
        """Test init worker success."""
        from model_protocol_lab import render_worker as rw

        with tempfile.TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), CLEAN_TEMPLATE)
            tokenizer, reply = rw._init_worker(
                {
                    "tokenizer_file": str(root / "tokenizer.json"),
                    "template": CLEAN_TEMPLATE,
                    "bos_token": "[BOS]",
                }
            )
        self.assertIsNotNone(tokenizer)
        self.assertEqual(reply["status"], "ok")
        self.assertEqual(reply["tokenizer_vocab_size"], len(tokenizer))
        self.assertIn("transformers", reply["versions"])
        self.assertIn("jinja2", reply["versions"])

    def test_init_worker_bos_setter_failure_suppressed(self):
        """Test init worker bos setter failure suppressed."""
        import transformers

        from model_protocol_lab import render_worker as rw

        class BosBomb:
            """Bos bomb."""

            def __init__(self, **kwargs: object) -> None:
                """Init."""
                self.chat_template: str | None = None

            def __len__(self) -> int:
                """Len."""
                return 7

            @property
            def bos_token(self) -> None:
                """Bos token."""
                return None

            @bos_token.setter
            def bos_token(self, value: str) -> None:
                """Bos token."""
                raise RuntimeError("setter boom")

        with mock.patch.object(transformers, "PreTrainedTokenizerFast", BosBomb):
            tokenizer, reply = rw._init_worker(
                {"tokenizer_file": "x", "template": "t", "bos_token": "[BOS]"}
            )
        self.assertIsNotNone(tokenizer)
        self.assertEqual(reply["status"], "ok")
        self.assertEqual(reply["tokenizer_vocab_size"], 7)

    def test_init_worker_versions_unavailable_when_metadata_missing(self):
        """Test init worker versions unavailable when metadata missing."""
        from model_protocol_lab import render_worker as rw

        def missing(name: str) -> str:
            """Missing."""
            raise rw.metadata.PackageNotFoundError(name)

        with tempfile.TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), CLEAN_TEMPLATE)
            with mock.patch.object(rw.metadata, "version", missing):
                _tokenizer, reply = rw._init_worker(
                    {
                        "tokenizer_file": str(root / "tokenizer.json"),
                        "template": CLEAN_TEMPLATE,
                        "bos_token": None,
                    }
                )
        self.assertEqual(reply["status"], "ok")
        self.assertEqual(reply["versions"]["transformers"], "unavailable")
        self.assertEqual(reply["versions"]["jinja2"], "unavailable")

    def test_render_reply_shapes(self):
        """Test render reply shapes."""
        from model_protocol_lab import render_worker as rw

        class Tok:
            """Tok."""

            @staticmethod
            def apply_chat_template(conversation, tokenize=False, add_generation_prompt=True):
                """Apply chat template."""
                return "TEXT"

            @staticmethod
            def encode(text, add_special_tokens=False):
                """Encode."""
                return [5, 6]

        self.assertEqual(
            rw._render_reply(Tok(), {"conversation": "not-a-list"}),
            {"status": "error", "code": "render_failed"},
        )
        self.assertEqual(
            rw._render_reply(Tok(), {"conversation": []}),
            {"status": "ok", "text": "TEXT"},
        )

        class Raising:
            """Raising."""

            @staticmethod
            def apply_chat_template(conversation, tokenize=False, add_generation_prompt=True):
                """Apply chat template."""
                raise RuntimeError("boom")

            @staticmethod
            def encode(text, add_special_tokens=False):
                """Encode."""
                raise RuntimeError("boom")

        self.assertEqual(
            rw._render_reply(Raising(), {"conversation": []}),
            {"status": "error", "code": "render_failed"},
        )
        self.assertEqual(
            rw._encode_reply(Raising(), {"text": "hi"}),
            {"status": "error", "code": "encode_failed"},
        )
        self.assertEqual(
            rw._encode_reply(Tok(), {"text": 42}),
            {"status": "error", "code": "encode_failed"},
        )
        self.assertEqual(
            rw._encode_reply(Tok(), {"text": "hi"}),
            {"status": "ok", "ids": [5, 6]},
        )

    def test_render_reply_oversize(self):
        """Test render reply oversize."""
        from model_protocol_lab import render_worker as rw

        class Huge:
            """Huge."""

            @staticmethod
            def apply_chat_template(conversation, tokenize=False, add_generation_prompt=True):
                """Apply chat template."""
                return "x" * (rw.MAX_RENDERED_CHARS + 1)

        self.assertEqual(
            rw._render_reply(Huge(), {"conversation": []}),
            {"status": "oversize"},
        )


class _FakeStd:
    """A sys.stdin/sys.stdout stand-in carrying a .buffer stream."""

    def __init__(self, buffer: object) -> None:
        """Init."""
        self.buffer = buffer


class WorkerMainLoopTests(unittest.TestCase):
    """main() serves the protocol over stdio until EOF or a dead pipe."""

    def _run_main(self, payload: bytes, out_buffer: object) -> tuple[int, list]:
        """Run main."""
        from model_protocol_lab import render_worker as rw

        class Tok:
            """Tok."""

            @staticmethod
            def apply_chat_template(conversation, tokenize=False, add_generation_prompt=True):
                """Apply chat template."""
                return "RENDERED"

            @staticmethod
            def encode(text, add_special_tokens=False):
                """Encode."""
                return [9, 8]

        def fake_init(frame: dict) -> tuple[object, dict]:
            """Fake init."""
            return Tok(), {"status": "ok", "versions": {}, "tokenizer_vocab_size": 3}

        fake_in = _FakeStd(io.BytesIO(payload))
        fake_out = _FakeStd(out_buffer)
        with (
            mock.patch.object(rw, "_init_worker", fake_init),
            mock.patch.object(sys, "stdin", fake_in),
            mock.patch.object(sys, "stdout", fake_out),
        ):
            code = rw.main()
        replies: list = []
        if isinstance(out_buffer, io.BytesIO):
            out_buffer.seek(0)
            while True:
                frame = rw._read_frame(out_buffer)
                if frame is None:
                    break
                replies.append(frame)
        return code, replies

    def test_main_serves_until_eof(self):
        """Test main serves until eof."""
        from model_protocol_lab import render_worker as rw

        payload = b"".join(
            [
                rw._encode_frame({"op": "render", "conversation": []}),
                rw._encode_frame(
                    {"op": "init", "tokenizer_file": "x", "template": "t", "bos_token": None}
                ),
                rw._encode_frame({"op": "render", "conversation": []}),
                rw._encode_frame({"op": "encode", "text": "hi"}),
                rw._encode_frame({"op": "bogus"}),
            ]
        )
        code, replies = self._run_main(payload, io.BytesIO())
        self.assertEqual(code, 0)
        self.assertEqual(
            [r.get("status") for r in replies],
            ["error", "ok", "ok", "ok", "error"],
        )
        self.assertEqual(replies[0].get("code"), "not_initialized")
        self.assertEqual(replies[2].get("text"), "RENDERED")
        self.assertEqual(replies[3].get("ids"), [9, 8])
        self.assertEqual(replies[4].get("code"), "unknown_op")

    def test_main_malformed_frame_returns(self):
        """Test main malformed frame returns."""
        body = b"{not json"
        payload = struct.pack(">I", len(body)) + body
        code, replies = self._run_main(payload, io.BytesIO())
        self.assertEqual(code, 0)
        self.assertEqual(replies, [])

    def test_module_main_guard_exits_zero_on_eof(self):
        """Test module main guard exits zero on eof."""
        import runpy

        fake_in = _FakeStd(io.BytesIO(b""))
        fake_out = _FakeStd(io.BytesIO())
        with (
            mock.patch.object(sys, "stdin", fake_in),
            mock.patch.object(sys, "stdout", fake_out),
            self.assertRaises(SystemExit) as caught,
        ):
            runpy.run_module("model_protocol_lab.render_worker", run_name="__main__")
        self.assertEqual(caught.exception.code, 0)

    def test_main_broken_pipe_returns(self):
        """Test main broken pipe returns."""
        from model_protocol_lab import render_worker as rw

        class RaisingStream:
            """Raising stream."""

            @staticmethod
            def write(data: bytes) -> int:
                """Write."""
                raise OSError("pipe gone")

            @staticmethod
            def flush() -> None:
                """Flush."""

        payload = rw._encode_frame(
            {"op": "init", "tokenizer_file": "x", "template": "t", "bos_token": None}
        )
        code, _replies = self._run_main(payload, RaisingStream())
        self.assertEqual(code, 0)


class DoctorWorkerFailureTests(unittest.TestCase):
    """Section-level worker failures past the first render phase."""

    def setUp(self):
        """Set up."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.script = self.root / "fake_worker.py"
        self.script.write_text(FAKE_WORKER, encoding="utf-8")
        self.counter = self.root / "counter"
        _write_checkpoint(self.root / "ckpt", CLEAN_TEMPLATE)

    def test_probe_breaker_during_role_arms_fails_closed(self):
        """Test probe breaker during role arms fails closed."""
        # The pair loop is 8 pairs x 4 renders = 32 renders. The fake
        # serves 31, then dies on every request: death 1 lands on the
        # last pair render, death 2 on the first role arm, death 3 on
        # the second — tripping the breaker inside the role-arm phase,
        # whose worker_failure return the section must honor.
        from model_protocol_lab import render_worker

        argv = _fake_argv("die_after", self.counter, self.script, 31)
        with mock.patch.object(render_worker, "WORKER_ARGV", argv):
            section = doctor.template_probe_section(self.root / "ckpt")
        self.assertEqual(section["detail"], "render_worker_failed")
        self.assertEqual(section["renders_used"], 33)

    def test_conformance_parent_cap_catches_misbehaving_worker(self):
        """Test conformance parent cap catches misbehaving worker."""
        # A worker that answers ok with over-cap text (a real worker
        # enforces the cap before the wire) is caught by the parent's
        # own post-render check.
        from model_protocol_lab import render_worker

        argv = _fake_argv("oversize_ok", self.counter, self.script)
        with mock.patch.object(render_worker, "WORKER_ARGV", argv):
            out = _conformance(self.root / "ckpt", (_case(),))
        self.assertEqual(out["diffs"][0]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
