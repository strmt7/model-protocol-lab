"""Cycle 243 -- R3 build cycle 1: serving-stack backends.

Spec: docs/SERVING_BACKENDS_DESIGN.md (the design record as
adjudicated: candidate + A1-A11 + Q1 as amended) and the cycle-243
build brief section 2.4, which names the regression tests below.
Pattern per gate (brief F11): the suite test runs against a hostile
fake implementing the documented b12182 surface and proves the gate
logic fires / fails closed; the executed b12182 result is cited
from cycle evidence (the spike + this cycle's C-1 probes and
end-to-end smoke, recorded in the cycle work log) -- the suite test
never claims the real-stack result, because CI has no real stack.

The fake tokenizer is concatenative (decode is string
concatenation over a fixed vocab), which is what makes the
mirror-oracle brute-force agreement provable rather than sampled.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import random
import struct
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

from model_protocol_lab import (
    diffbackends,
    diffreports,
    experiments,
    reporting,
    servingbackends,
)
from model_protocol_lab.servingbackends import (
    LlamacppHttpBackend,
    RecordSink,
    StackRequest,
    load_records_artifact,
    predict_stop_outcome,
    read_gguf_declared_tokens,
    run_cache_sequence,
    tier_a_sampler_params,
)

API_KEY = "test-key"
EOS = 0

_VOCAB = {
    0: "<|endoftext|>",
    1: "The",
    2: " capital",
    3: " of",
    4: " France",
    5: " is",
    6: "abc",
    7: "STOP",
    8: "xyz",
    9: " Paris",
    10: "a",
    11: "b",
    12: "c",
    13: "ab",
    14: "bc",
    999: "<|pad|>",
}


class FakeTokenizer:
    """Concatenative tokenizer: encode is greedy longest-match,
    decode is concatenation. Deterministic and merge-free."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        pieces = sorted(_VOCAB.items(), key=lambda item: -len(item[1]))
        out: list[int] = []
        rest = text
        while rest:
            for token_id, piece in pieces:
                if piece and rest.startswith(piece):
                    out.append(token_id)
                    rest = rest[len(piece) :]
                    break
            else:
                raise ValueError(f"unencodable: {rest[:1]!r}")
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(_VOCAB[token] for token in ids)


def fake_decode(ids: tuple[int, ...]) -> str:
    return "".join(_VOCAB[token] for token in ids)


def _gguf_bytes(
    eos_ids: list[int] | None, *, array_form: bool = False, bad_magic: bool = False
) -> bytes:
    def key_bytes(key: str) -> bytes:
        raw = key.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    entries: list[bytes] = []
    entries.append(key_bytes("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", 0))
    entries.append(
        key_bytes("general.name") + struct.pack("<I", 8) + struct.pack("<Q", 4) + b"fake"
    )
    if eos_ids is None:
        pass  # no declared-EOS key at all
    elif array_form:
        value = struct.pack("<I", 9) + struct.pack("<I", 4) + struct.pack("<Q", len(eos_ids))
        value += b"".join(struct.pack("<I", item) for item in eos_ids)
        entries.append(key_bytes("tokenizer.ggml.eos_token_id") + value)
    else:
        entries.append(
            key_bytes("tokenizer.ggml.eos_token_id")
            + struct.pack("<I", 4)
            + struct.pack("<I", eos_ids[0])
        )
    entries.append(
        key_bytes("tokenizer.ggml.bos_token_id") + struct.pack("<I", 4) + struct.pack("<I", 0)
    )
    strings = struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", 2)
    for word in (b"a", b"bb"):
        strings += struct.pack("<Q", len(word)) + word
    entries.append(key_bytes("some.strings") + strings)
    magic = b"GGUF" if not bad_magic else b"XXXX"
    return magic + struct.pack("<IQQ", 3, 0, len(entries)) + b"".join(entries)


class _FakeState:
    """Mutable fake-server behaviour, read per request."""

    def __init__(self, gguf_path: Path) -> None:
        self.gguf_path = str(gguf_path)
        self.api_key = API_KEY
        self.auth_required = True
        self.build_info = "b12182-fd27b20"
        self.total_slots = 1
        self.model_path: str | None = None  # defaults to gguf_path
        self.erase_enabled = True
        self.mode = "limit"
        self.echo_override: dict[str, Any] = {}
        self.prompt_override: str | None = None
        self.prompt_as_ids = False
        self.detok_override: Any = None
        self.tokenize_override: Any = None
        self.flat_cache = False
        self.no_cache_fields = False
        self.tokens_cached_value: int | None = None
        self.cache_n_override: int | None = None
        self.props_override: dict[str, Any] | None = None
        self.props_status = 200
        self.redirect_props = False
        self.detok_status = 200
        self.tokenize_status = 200
        self.flip_build_info_after: int | None = None
        self.drop_paths: set[str] = set()
        self.fail_completions_after: int | None = None
        self.seen_prompts: list[tuple[int, ...]] = []
        self.completion_calls = 0
        self.last_body: dict[str, Any] | None = None

    def cache_n_for(self, prompt: tuple[int, ...]) -> int:
        if self.cache_n_override is not None:
            return self.cache_n_override
        if self.flat_cache:
            return 0
        best = 0
        for seen in self.seen_prompts:
            shared = 0
            for left, right in zip(prompt, seen, strict=False):
                if left != right:
                    break
                shared += 1
            best = max(best, shared)
        self.seen_prompts.append(prompt)
        return best - 1 if best >= 1 else 0


def _make_handler(state: _FakeState):
    class Handler(BaseHTTPRequestHandler):
        """Fake llama.cpp HTTP surface at the documented b12182 shape."""

        def log_message(self, *_args: Any) -> None:
            pass

        def _send(self, status: int, payload: Any) -> None:
            raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _authorized(self) -> bool:
            if not state.auth_required:
                return True
            return self.headers.get("Authorization") == f"Bearer {state.api_key}"

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            parsed = json.loads(raw.decode("utf-8"))
            return parsed if type(parsed) is dict else {}

        def _dropped(self, path: str) -> bool:
            if path not in state.drop_paths:
                return False
            import socket

            self.close_connection = True
            with contextlib.suppress(OSError):
                self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return True

        def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler API
            path = urllib.parse.urlsplit(self.path).path
            if self._dropped(path):
                return
            if path == "/health":
                self._send(200, {"status": "ok"})
                return
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return
            if path == "/props":
                if state.redirect_props:
                    self.send_response(302)
                    self.send_header("Location", "/props")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if state.props_status != 200:
                    self._send(state.props_status, {"error": "props down"})
                    return
                if state.props_override is not None:
                    self._send(200, state.props_override)
                    return
                self._send(
                    200,
                    {
                        "build_info": state.build_info,
                        "total_slots": state.total_slots,
                        "model_path": state.model_path or state.gguf_path,
                        "endpoint_slots": True,
                        "endpoint_props": False,
                        "endpoint_metrics": False,
                        "ui": False,
                        "default_generation_settings": {
                            "params": {"temperature": 0.8},
                            "n_ctx": 1024,
                        },
                    },
                )
                return
            if path == "/slots":
                self._send(200, [{"id": 0, "n_ctx": 4096}])
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler API
            parts = urllib.parse.urlsplit(self.path)
            path = parts.path
            if self._dropped(path):
                return
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return
            if path == "/slots/0":
                if not state.erase_enabled:
                    self._send(501, {"error": "slots endpoint disabled"})
                    return
                state.seen_prompts.clear()
                self._send(200, {"id_slot": 0, "n_erased": 15})
                return
            body = self._body()
            if path == "/detokenize":
                if state.detok_status != 200:
                    self._send(state.detok_status, {"error": "detokenize down"})
                    return
                tokens = tuple(body.get("tokens") or [])
                if state.detok_override is not None:
                    self._send(200, {"content": state.detok_override(tokens)})
                    return
                self._send(200, {"content": fake_decode(tokens)})
                return
            if path == "/tokenize":
                if state.tokenize_status != 200:
                    self._send(state.tokenize_status, {"error": "tokenize down"})
                    return
                if state.tokenize_override is not None:
                    self._send(200, {"tokens": state.tokenize_override(body.get("content"))})
                    return
                self._send(200, {"tokens": FakeTokenizer().encode(str(body.get("content")))})
                return
            if path == "/completion":
                state.completion_calls += 1
                state.last_body = body
                if (
                    state.flip_build_info_after is not None
                    and state.completion_calls >= state.flip_build_info_after
                ):
                    state.build_info = "b12182-deadbeef"
                self._completion(body)
                return
            self._send(404, {"error": "not found"})

        def _completion(self, body: dict[str, Any]) -> None:
            if state.mode == "http500":
                self._send(500, {"error": "boom"})
                return
            if (
                state.fail_completions_after is not None
                and state.completion_calls > state.fail_completions_after
            ):
                self._send(500, {"error": "late failure"})
                return
            if state.mode == "huge":
                self._send(200, b"x" * 16_777_217)
                return
            if state.mode == "broken_json":
                self._send(200, b"this is not json")
                return
            prompt = tuple(body.get("prompt") or [])
            n_predict = int(body.get("n_predict") or 1)
            stops = body.get("stop") or []
            cache_n = state.cache_n_for(prompt) if body.get("cache_prompt") else 0
            mode = state.mode
            tokens: list[int]
            stop_type = "limit"
            stopping_word: str | None = None
            if mode == "eos" and not body.get("ignore_eos"):
                tokens, stop_type = [6, EOS], "eos"
            elif mode in ("word", "word_kept", "word_weird") and "STOP" in stops:
                tokens, stop_type, stopping_word = [6, 7], "word", "STOP"
            elif mode == "stop_unfired":
                tokens, stop_type = [6, 7], "limit"
            elif mode == "spurious":
                tokens, stop_type, stopping_word = [6], "word", "STOP"
            elif mode == "step_differs":
                tokens, stop_type, stopping_word = [6, 7, 6], "word", "STOP"
            elif mode == "eos_other":
                tokens, stop_type = [6, 999], "eos"
            elif mode == "str_over_eos":
                tokens, stop_type, stopping_word = [6, EOS], "word", "STOP"
            elif mode == "tie_stop":
                tokens, stop_type, stopping_word = [6, EOS], "word", "<|endoftext|>"
            elif mode == "tie_stops":
                tokens, stop_type, stopping_word = [13], "word", "b"
            elif mode == "none":
                tokens, stop_type = [6], "none"
            elif mode == "unclassified":
                tokens, stop_type = [6], "mystery"
            elif mode == "overshoot":
                tokens = [6] * (n_predict + 3)
            elif mode == "bad_tokens":
                tokens = "abc"  # type: ignore[assignment]
            elif mode == "eos_empty":
                tokens, stop_type = [], "eos"
            elif mode == "word_missing":
                tokens, stop_type = [6, 7], "word"
            else:
                tokens = [6] * n_predict
            settings = {
                "samplers": body.get("samplers"),
                "temperature": body.get("temperature"),
                "top_k": body.get("top_k"),
                "top_p": body.get("top_p"),
                "min_p": body.get("min_p"),
                "repeat_penalty": body.get("repeat_penalty"),
                "dynatemp_range": body.get("dynatemp_range"),
                "n_probs": body.get("n_probs"),
                "seed": body.get("seed"),
            }
            settings.update(state.echo_override)
            payload: dict[str, Any] = {
                "content": fake_decode(tuple(tokens)) if mode != "bad_tokens" else "abc",
                "tokens": tokens,
                "stop": True,
                "stop_type": stop_type,
                "prompt": list(prompt)
                if state.prompt_as_ids
                else (
                    state.prompt_override
                    if state.prompt_override is not None
                    else fake_decode(prompt)
                ),
                "generation_settings": settings,
                "timings": {} if state.no_cache_fields else {"cache_n": cache_n},
            }
            if stopping_word is not None:
                payload["stopping_word"] = stopping_word
                if mode != "word_kept":
                    payload["content"] = "abc"
            if mode == "word_weird":
                payload["content"] = "zzz"
            if not state.no_cache_fields:
                payload["tokens_cached"] = (
                    state.tokens_cached_value if state.tokens_cached_value is not None else cache_n
                )
                payload["tokens_evaluated"] = len(prompt)
            self._send(200, payload)

    return Handler


class _FakeServer:
    def __init__(self, gguf_path: Path) -> None:
        self.state = _FakeState(gguf_path)
        self.httpd = HTTPServer(("127.0.0.1", 0), _make_handler(self.state))
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> _FakeServer:
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class ServingBackendsTestBase(unittest.TestCase):
    """Shared fixtures: fake GGUF on disk + a backend factory."""

    def setUp(self) -> None:
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.gguf = Path(self._tmpdir.name) / "fake.gguf"
        self.gguf.write_bytes(_gguf_bytes([EOS]))

    def make_backend(self, server: _FakeServer, **overrides: Any) -> LlamacppHttpBackend:
        kwargs: dict[str, Any] = {
            "api_key": API_KEY,
            "gguf_path": self.gguf,
            "tokenizer": FakeTokenizer(),
            "tokenizer_fingerprint": "sha256:" + "0" * 64,
            "timeout_s": 10.0,
        }
        kwargs.update(overrides)
        return LlamacppHttpBackend(server.url, **kwargs)

    def probed_backend(self, server: _FakeServer, **overrides: Any) -> LlamacppHttpBackend:
        backend = self.make_backend(server, **overrides)
        backend.probe()
        return backend

    def _args(self, server: _FakeServer | None, out: Path, **overrides: Any) -> Any:
        import argparse

        values: dict[str, Any] = {
            "command": "exec-backends",
            "base_url": server.url if server is not None else "http://127.0.0.1:9",
            "api_key": API_KEY,
            "gguf": self.gguf,
            "model": Path(self._tmpdir.name),
            "out": out,
            "prompt_ids": "1,2,3,4,5",
            "stops": None,
            "eos_id": None,
            "min_tokens": 0,
            "max_new_tokens": 8,
            "results": None,
            "cache_cold_ids": None,
            "cache_shared_ids": None,
            "cache_divergent_ids": None,
            "seed": 0,
            "timeout": 5.0,
            "hardware_class": "cpu",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def sentinel(self) -> tuple[int, ...]:
        return tuple(FakeTokenizer().encode(servingbackends.SENTINEL_TEXT))


class ProbeGateTests(ServingBackendsTestBase):
    """The probe gates, against the hostile fake (brief section 2.4)."""

    def test_g1_identity_window_llama(self) -> None:
        """Spec: design D-8 + section 8.6 -- identity binds (tag
        b12182, commit fd27b20..., resolved model path); the build
        counter is an acquisition artefact, so a shallow clone's
        b1-fd27b20 is the same window member as b12182-fd27b20."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            self.assertEqual(backend.identity.stack_version, "b12182-fd27b20")
        with _FakeServer(self.gguf) as server:
            server.state.build_info = "b1-fd27b20"
            backend = self.probed_backend(server)
            self.assertEqual(backend.identity.stack_version, "b1-fd27b20")
        with _FakeServer(self.gguf) as server:
            server.state.build_info = "b12182-aaaaaaa"
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "stack_version_unsupported")
        with _FakeServer(self.gguf) as server:
            server.state.build_info = "not-a-build"
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "stack_identity_unavailable")
        with _FakeServer(self.gguf) as server:
            server.state.model_path = "/somewhere/else.gguf"
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "stack_identity_unavailable")
        with _FakeServer(self.gguf) as server:
            server.state.total_slots = 2
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "stack_topology_mismatch")

    def test_a3_auth_gate(self) -> None:
        """Spec: adjudication C-1(a) -- the API key must guard the
        surface (unauthenticated 401, authenticated 200); an
        unguarded server is refused, never measured."""
        with _FakeServer(self.gguf) as server:
            server.state.auth_required = False
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "stack_endpoint_refused")
        with _FakeServer(self.gguf) as server:
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server, api_key="wrong-key").probe()
            self.assertEqual(str(caught.exception), "stack_endpoint_refused")

    def test_g2_full_tuple_echo_exact(self) -> None:
        """Spec: section 8.2 G2 + 7.2 -- the generation_settings
        echo must match the full Tier-A tuple exactly, at the probe
        and on every executed request."""
        with _FakeServer(self.gguf) as server:
            server.state.echo_override = {"temperature": 0.8}
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "config_echo_mismatch")
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.state.echo_override = {"seed": 999}
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(prompt_ids=self.sentinel(), seed=7, max_new_tokens=4),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
            )
            self.assertEqual(record["cell_detail"], "config_echo_mismatch")
            self.assertEqual(record["mirror"]["fidelity_verdict"], "unobservable")

    def test_a5_repeat_penalty_neutral_pin(self) -> None:
        """Spec: amendment R-1 (spike: three executed reads) -- the
        neutral repeat_penalty at b12182 is exactly 1.0; the pin is
        in the sampler tuple and on the wire."""
        self.assertEqual(tier_a_sampler_params(5)["repeat_penalty"], 1.0)
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            servingbackends.execute_cell(
                backend,
                request=StackRequest(prompt_ids=self.sentinel(), seed=7, max_new_tokens=4),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
            )
            self.assertEqual(server.state.last_body["repeat_penalty"], 1.0)
            self.assertEqual(
                server.state.last_body["samplers"],
                ["top_k", "top_p", "min_p", "temperature"],
            )

    def test_g3_prompt_echo_text_form(self) -> None:
        """Spec: amendment R-5 + section 8.2 G3 -- the prompt echo
        is compared as TEXT (local detokenization and the server's
        /detokenize); there is no ids-form prompt assertion: a
        server returning the prompt as a list of ids fails the gate
        as a text mismatch, it is not reinterpreted."""
        with _FakeServer(self.gguf) as server:
            server.state.prompt_override = "some other text"
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "prompt_echo_mismatch")
        with _FakeServer(self.gguf) as server:
            # The negative half: a server answering with an
            # ids-valued prompt gets a text-form mismatch, never a
            # reinterpretation as ids.
            server.state.prompt_as_ids = True
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "prompt_echo_mismatch")

    def test_g4_vocab_roundtrip_incl_eot(self) -> None:
        """Spec: section 8.2 G4 -- detokenize(sentinel + EOT) must
        re-tokenize to the same ids on the server and locally."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            self.assertTrue(backend.capabilities.raw_token_ids)
        with _FakeServer(self.gguf) as server:
            server.state.tokenize_override = lambda _text: [1, 2]
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "token_vocab_mismatch")
        with _FakeServer(self.gguf) as server:
            server.state.detok_override = lambda _ids: "different text"
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "prompt_echo_mismatch")


class StopArmTests(ServingBackendsTestBase):
    """The section 5.2 arms and the classification discipline."""

    def _cell(
        self, server: _FakeServer, request: StackRequest, mode: str | None = None
    ) -> dict[str, Any]:
        backend = self.probed_backend(server)
        if mode is not None:
            server.state.mode = mode
        return servingbackends.execute_cell(
            backend,
            request=request,
            decode=backend.decode_ids,
            candidate="t",
            fixture="t1",
        )

    def test_s52_forced_eos_arm(self) -> None:
        """Spec: section 5.2 + spike E10 -- a stream ending on the
        EOS id classifies eos, the firing id is in the returned
        tokens, and the predictor agrees."""
        with _FakeServer(self.gguf) as server:
            record = self._cell(
                server,
                StackRequest(prompt_ids=self.sentinel(), seed=7, max_new_tokens=8, eos_id=EOS),
                mode="eos",
            )
            self.assertEqual(record["finish_class"], "eos")
            self.assertEqual(record["stop_winner"], {"kind": "eos", "id": EOS})
            self.assertEqual(record["returned_ids"][-1], EOS)
            self.assertEqual(record["mirror"]["fidelity_verdict"], "agree")

    def test_s52_ignore_eos_arm(self) -> None:
        """Spec: section 5.2 -- with ignore_eos the same stack runs
        to the cap: finish length, no winner, predictor suppresses
        the EOS identically."""
        with _FakeServer(self.gguf) as server:
            record = self._cell(
                server,
                StackRequest(
                    prompt_ids=self.sentinel(),
                    seed=7,
                    max_new_tokens=6,
                    eos_id=EOS,
                    ignore_eos=True,
                ),
                mode="eos",
            )
            self.assertEqual(record["finish_class"], "length")
            self.assertIsNone(record["stop_winner"])
            self.assertEqual(len(record["returned_ids"]), 6)
            self.assertEqual(record["mirror"]["fidelity_verdict"], "agree")

    def test_a7_detokenize_crosscheck(self) -> None:
        """Spec: adjudication A7 -- the stack's own /detokenize
        disagreeing with the predictor's detokenization is a
        predictor-side finding in its own row, never a divergence."""
        with _FakeServer(self.gguf) as server:
            server.state.mode = "eos"
            backend = self.probed_backend(server)
            server.state.detok_override = lambda ids: (
                "different" if len(ids) < 5 else fake_decode(ids)
            )
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(
                    prompt_ids=self.sentinel(), seed=7, max_new_tokens=8, eos_id=EOS
                ),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
            )
            self.assertEqual(
                record["mirror"]["fidelity_verdict"],
                "predictor_detokenization_mismatch",
            )
            aggregate = servingbackends.aggregate_fidelity([record])
            self.assertEqual(aggregate["counts"]["predictor_detokenization_mismatch"], 1)
            self.assertEqual(aggregate["counts"]["diverge"], 0)
            self.assertEqual(aggregate["verdict"], "fidelity_unavailable")

    def test_a11_eos_attribution(self) -> None:
        """Spec: adjudication A11 -- EOS attribution binds the
        GGUF-declared ids: a stream ending on an undeclared id
        fails attribution for that cell."""
        other = Path(self._tmpdir.name) / "other.gguf"
        other.write_bytes(_gguf_bytes([999]))
        with _FakeServer(self.gguf) as server:
            server.state.mode = "eos"
            server.state.gguf_path = str(other)
            backend = LlamacppHttpBackend(
                server.url,
                api_key=API_KEY,
                gguf_path=other,
                tokenizer=FakeTokenizer(),
                tokenizer_fingerprint="sha256:" + "0" * 64,
                timeout_s=10.0,
            )
            backend.probe()
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(
                    prompt_ids=self.sentinel(), seed=7, max_new_tokens=8, eos_id=EOS
                ),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
            )
            self.assertEqual(record["cell_detail"], "eos_attribution_mismatch")
            self.assertEqual(record["mirror"]["fidelity_verdict"], "diverge")

    def test_finish_allowlist_classification(self) -> None:
        """Spec: adjudication A9(i) -- a finish outside the
        allowlist is stack_response_unclassified, never exported
        raw; every classified finish stays inside the allowlist."""
        with _FakeServer(self.gguf) as server:
            record = self._cell(
                server,
                StackRequest(prompt_ids=self.sentinel(), seed=7, max_new_tokens=4),
                mode="unclassified",
            )
            self.assertEqual(record["finish_class"], "unclassified")
            self.assertEqual(record["cell_detail"], "stack_response_unclassified")
            self.assertIn(record["finish_class"], servingbackends.FINISH_CLASSES)
        with _FakeServer(self.gguf) as server:
            record = self._cell(
                server,
                StackRequest(prompt_ids=self.sentinel(), seed=7, max_new_tokens=4),
                mode="none",
            )
            self.assertEqual(record["finish_class"], "none")
            self.assertEqual(record["mirror"]["fidelity_verdict"], "unobservable")


class CacheProtocolTests(ServingBackendsTestBase):
    """The section 5.4 cache companion and the A2/A8 identity rules."""

    def test_g5_cache_quantity_is_cache_n(self) -> None:
        """Spec: amendment R-2 -- the G5 quantity is timings.cache_n;
        tokens_cached is recorded as reported and never decides:
        here tokens_cached is a flat 999 while cache_n discriminates,
        and reporting is still judged present from cache_n."""
        with _FakeServer(self.gguf) as server:
            server.state.tokens_cached_value = 999
            backend = self.probed_backend(server)
            self.assertTrue(backend.capabilities.cached_token_reporting)
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(
                    prompt_ids=(8, 12),  # fresh prompt: genuinely cold
                    seed=7,
                    max_new_tokens=4,
                    cache_mode="cold",
                ),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
                cache_position="cold",
            )
            self.assertEqual(record["tokens_cached_reported"], 999)
            self.assertEqual(record["cache_n_reported"], 0)
        with _FakeServer(self.gguf) as server:
            server.state.flat_cache = True
            backend = self.probed_backend(server)
            self.assertFalse(backend.capabilities.cached_token_reporting)
        with _FakeServer(self.gguf) as server:
            server.state.no_cache_fields = True
            backend = self.probed_backend(server)
            self.assertFalse(backend.capabilities.cached_token_reporting)

    def test_g5_divergent_control_threshold(self) -> None:
        """Spec: section 5.4 + C-3 + A6 -- a divergent control
        reporting reuse above one quantum (here a 3-token shared
        prefix, cache_n 2) marks the sequence contaminated."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            sequence = run_cache_sequence(
                backend,
                cold_ids=self.sentinel(),
                shared_ids=self.sentinel() + (9,),
                divergent_ids=(1, 2, 3, 8),  # shares "The capital of" prefix
                seed=3,
                decode=backend.decode_ids,
            )
            self.assertEqual(sequence["detail"], "cache_protocol_contaminated")
            positions = sequence["positions"]
            self.assertEqual(positions["cold"]["cache_n"], 0)
            self.assertEqual(positions["warm_repeat"]["cache_n"], 4)
            self.assertEqual(positions["divergent_control"]["cache_n"], 2)
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            sequence = run_cache_sequence(
                backend,
                cold_ids=self.sentinel(),
                shared_ids=self.sentinel() + (9,),
                divergent_ids=(8, 12),  # shares nothing
                seed=3,
                decode=backend.decode_ids,
            )
            self.assertEqual(sequence["detail"], "")
            self.assertEqual(sequence["positions"]["divergent_control"]["cache_n"], 0)

    def test_a2_cache_machine_fingerprinted(self) -> None:
        """Spec: adjudication A2 -- the identity fingerprint binds
        the cache machine as probed (posture + measured cache
        probe): same server, same fingerprint; a server whose
        cache behaviour differs fingerprints differently."""
        with _FakeServer(self.gguf) as server:
            first = self.probed_backend(server)
            second = self.probed_backend(server)
            self.assertEqual(first.identity.config_fingerprint, second.identity.config_fingerprint)
            self.assertEqual(first.identity.config["cache_probe"]["cold_cache_n"], 0)
            self.assertEqual(first.identity.config["cache_probe"]["warm_cache_n"], 4)
            self.assertTrue(first.identity.config["posture"]["slot_erase_available"])
        with _FakeServer(self.gguf) as server:
            server.state.flat_cache = True
            flat = self.probed_backend(server)
            self.assertNotEqual(flat.identity.config_fingerprint, first.identity.config_fingerprint)

    def test_a8_identity_reread(self) -> None:
        """Spec: adjudication A8(i) -- identity is re-read at a
        sequence's start and end; a mid-run change is detected and
        carried as the sequence's detail, never silently absorbed."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            self.assertTrue(backend.verify_identity())
            server.state.build_info = "b12182-deadbeef"
            self.assertFalse(backend.verify_identity())
            sequence = run_cache_sequence(
                backend,
                cold_ids=self.sentinel(),
                shared_ids=self.sentinel() + (9,),
                divergent_ids=(8, 12),
                seed=3,
                decode=backend.decode_ids,
            )
            self.assertEqual(sequence["detail"], "stack_identity_changed_during_run")
            self.assertEqual(sequence["records"], [])

    def test_a8_incremental_persistence(self) -> None:
        """Spec: adjudication A8(iv) -- records persist per cell as
        executions complete; a run that dies mid-plan leaves a valid
        prefix whose planned-but-absent cells read back as
        cell_not_executed."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(prompt_ids=self.sentinel(), seed=7, max_new_tokens=4),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
            )
        path = Path(self._tmpdir.name) / "records.jsonl"
        plan = [
            {"candidate": "t", "fixture": "t1", "cache_position": "standalone"},
            {"candidate": "t", "fixture": "t2", "cache_position": "standalone"},
            {"candidate": "cache-probe", "fixture": "cache-cold", "cache_position": "cold"},
        ]
        sink = RecordSink(path, header={"kind": "exec-backends-records", "plan": plan})
        self.assertTrue(sink.append(record))
        sink.close()  # the run dies here: no footer
        loaded = load_records_artifact(path)
        self.assertFalse(loaded["complete"])
        self.assertFalse(loaded["finalized"])
        self.assertEqual(len(loaded["records"]), 1)
        self.assertEqual({entry["fixture"] for entry in loaded["missing"]}, {"t2", "cache-cold"})
        self.assertTrue(
            all(entry["cell_detail"] == "cell_not_executed" for entry in loaded["missing"])
        )
        # The finalized form loads too.
        final_path = Path(self._tmpdir.name) / "final.json"
        section = experiments.serving_backends_analysis(
            {
                "available": True,
                "detail": "",
                "identity": backend.identity.as_dict(),
                "capabilities": backend.capabilities.as_dict(),
                "records": [record],
                "cache_sequence": None,
                "fidelity": None,
                "audit": {"n_records": 1},
            }
        )
        final_path.write_text(
            json.dumps({"schema_version": 1, "serving_backends": section}),
            encoding="utf-8",
        )
        loaded_final = load_records_artifact(final_path)
        self.assertTrue(loaded_final["finalized"])
        self.assertEqual(len(loaded_final["records"]), 1)


class CapabilityAndRecordTests(ServingBackendsTestBase):
    """Capability-as-data, no-silent-fallback, ids+digests records."""

    def test_capability_absent_cells_recorded(self) -> None:
        """Spec: section 5.2/A2 -- each unsupported exercised
        capability records a capability_absent CELL; nothing is
        executed for it and no row is omitted."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            calls_before = server.state.completion_calls
            variants = [
                StackRequest(prompt_ids=self.sentinel(), seed=7, stop_token_ids=(13,)),
                StackRequest(prompt_ids=self.sentinel(), seed=7, min_tokens=3),
                StackRequest(
                    prompt_ids=self.sentinel(),
                    seed=7,
                    include_stop_str_in_output=True,
                ),
                StackRequest(prompt_ids=self.sentinel(), seed=7, cache_salt="pepper"),
            ]
            for index, request in enumerate(variants):
                record = servingbackends.execute_cell(
                    backend,
                    request=request,
                    decode=backend.decode_ids,
                    candidate="t",
                    fixture=f"t{index}",
                )
                self.assertEqual(record["cell_detail"], "capability_absent")
                self.assertIsNone(record["returned_ids"])
                self.assertEqual(record["mirror"]["fidelity_verdict"], "capability_absent")
            self.assertEqual(server.state.completion_calls, calls_before)

    def test_no_silent_fallback(self) -> None:
        """Spec: section 5.4 fallback + capability-as-data -- a
        server without slot erase yields capability_absent cache
        cells (no restart, no substitution, no freshness claim);
        a non-loopback or credentialed endpoint is refused at
        construction."""
        with _FakeServer(self.gguf) as server:
            server.state.erase_enabled = False
            backend = self.probed_backend(server)
            sequence = run_cache_sequence(
                backend,
                cold_ids=self.sentinel(),
                shared_ids=self.sentinel() + (9,),
                divergent_ids=(8, 12),
                seed=3,
                decode=backend.decode_ids,
            )
            self.assertFalse(sequence["available"])
            self.assertEqual(sequence["detail"], "capability_absent")
            self.assertEqual(len(sequence["records"]), 4)
            self.assertTrue(
                all(record["cell_detail"] == "capability_absent" for record in sequence["records"])
            )
        for bad_url in (
            "https://127.0.0.1:8080",
            "http://192.168.1.10:8080",
            "http://user@127.0.0.1:8080",
            "http://127.0.0.1:8080/?x=1",
        ):
            with self.assertRaises(ValueError) as caught:
                LlamacppHttpBackend(
                    bad_url,
                    api_key=API_KEY,
                    gguf_path=self.gguf,
                    tokenizer=FakeTokenizer(),
                    tokenizer_fingerprint="sha256:" + "0" * 64,
                )
            self.assertEqual(str(caught.exception), "stack_endpoint_refused")

    def test_records_ids_and_digests_only(self) -> None:
        """Spec: section 6.3 -- records carry full prompt ids and
        SHA-256 digests of text; raw stop text appears nowhere in
        the serialized record."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.state.mode = "word"
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(
                    prompt_ids=self.sentinel(),
                    seed=7,
                    max_new_tokens=8,
                    stop_texts=("STOP",),
                    eos_id=EOS,
                ),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
            )
            blob = json.dumps(record)
            # "STOP" cannot hide in a hex digest (uppercase, non-hex
            # letters), so its absence is a proof, not a sample.
            self.assertNotIn("STOP", blob)
            self.assertNotIn("content", record)
            self.assertEqual(record["prompt_ids"], list(self.sentinel()))
            self.assertEqual(record["stop_winner"]["kind"], "stop_text")
            self.assertEqual(record["stop_winner"]["index"], 0)
            digest = record["returned_text_digest"]
            self.assertEqual(len(digest), 64)
            self.assertTrue(all(char in "0123456789abcdef" for char in digest))
            self.assertIsNone(record["returned_text_digest_no_special"])
            self.assertIsNone(record["cache_creation_tokens_reported"])

    def test_request_validation(self) -> None:
        """Spec: section 5.2/8.5 -- malformed requests fail with
        the fixed code, never a partial execution."""
        for kwargs in (
            {"prompt_ids": ()},
            {"prompt_ids": (1, True)},
            {"prompt_ids": (1, -2)},
            {"prompt_ids": (1,), "max_new_tokens": 0},
            {"prompt_ids": (1,), "max_new_tokens": 4097},
            {"prompt_ids": (1,), "min_tokens": -1},
            {"prompt_ids": (1,), "cache_mode": "tepid"},
        ):
            with self.assertRaises(ValueError) as caught:
                StackRequest(seed=1, **kwargs)
            self.assertEqual(str(caught.exception), "stack_request_invalid")


def _naive_predict(
    *,
    returned_ids: tuple[int, ...],
    stop_texts: tuple[str, ...],
    eos_id: int | None,
    ignore_eos: bool,
) -> tuple[int | None, dict[str, Any] | None]:
    """Independent oracle for the predictor: find stop spans in the
    single full decode, then map the span end back to a step by
    prefix lengths. Written from the section 5.5 spec, not from the
    implementation (which scans prefix decodes)."""
    full = fake_decode(returned_ids)
    best: tuple[int, int, dict[str, Any]] | None = None
    for index, text in enumerate(stop_texts):
        start = full.find(text)
        if start < 0:
            continue
        end = start + len(text)
        step = None
        for candidate_step in range(1, len(returned_ids) + 1):
            if len(fake_decode(returned_ids[:candidate_step])) >= end:
                step = candidate_step
                break
        if step is not None and (best is None or step < best[0]):
            best = (step, 0, {"kind": "stop_text", "index": index})
    if eos_id is not None and not ignore_eos and eos_id in returned_ids:
        step = returned_ids.index(eos_id) + 1
        if best is None or step < best[0]:
            best = (step, 1, {"kind": "eos", "id": eos_id})
    if best is None:
        return None, None
    return best[0], best[2]


class PredictorAndFidelityTests(ServingBackendsTestBase):
    """The mirror predictor against its oracle; section 9.3 fidelity."""

    def test_mirror_oracle_bruteforce(self) -> None:
        """Spec: section 5.5 + brief 2.4 -- random streams x random
        stop configs: the production predictor and the naive oracle
        agree on firing step and winner (the fake's decode is
        concatenative, so agreement is exact, not sampled)."""
        rng = random.Random(243)  # noqa: S311 -- deterministic test seed
        pool = [6, 7, 8, 10, 11, 12, 13, 14, EOS]
        stop_pool = ["STOP", "abc", "bc", "xyz", "a", "ab"]
        for trial in range(300):
            ids = tuple(rng.choice(pool) for _ in range(rng.randrange(0, 9)))
            stops = tuple(rng.sample(stop_pool, rng.randrange(0, 3)))
            ignore = rng.random() < 0.25
            predicted = predict_stop_outcome(
                decode=fake_decode,
                returned_ids=ids,
                stop_texts=stops,
                eos_id=EOS,
                ignore_eos=ignore,
            )
            step, winner = _naive_predict(
                returned_ids=ids, stop_texts=stops, eos_id=EOS, ignore_eos=ignore
            )
            self.assertEqual(predicted["predicted_firing_step"], step, f"trial {trial}")
            self.assertEqual(predicted["predicted_winner"], winner, f"trial {trial}")

    def test_determinism_rate_shape(self) -> None:
        """Spec: spike determinism discipline -- determinism is a
        rate with an n, never a property claim: five identical
        executions of one pinned request produce five identical
        records, reported as 5/5."""
        with _FakeServer(self.gguf) as server:
            server.state.mode = "eos"
            backend = self.probed_backend(server)
            request = StackRequest(prompt_ids=self.sentinel(), seed=7, max_new_tokens=8, eos_id=EOS)
            digests = []
            for _ in range(5):
                record = servingbackends.execute_cell(
                    backend,
                    request=request,
                    decode=backend.decode_ids,
                    candidate="t",
                    fixture="t1",
                )
                digests.append(record["returned_id_digest"])
            report = {
                "n": len(digests),
                "identical": sum(1 for digest in digests if digest == digests[0]),
            }
            report["rate"] = report["identical"] / report["n"]
            self.assertEqual(report, {"n": 5, "identical": 5, "rate": 1.0})

    def test_fidelity_stream_conditioned(self) -> None:
        """Spec: section 9.3 -- fidelity verdicts are
        stream-conditioned: the recorded HF stream here differs
        from the stack's stream (GENERATION_STREAM_DIVERGED with
        the first-divergence index) while the stop behaviour
        agrees; both facts are reported, neither replaces the
        other. The keep arm is capability_absent against llama
        (U1), so the section aggregate is fidelity_unavailable --
        the absence is data, not a silent skip."""
        with _FakeServer(self.gguf) as server:
            server.state.mode = "eos"
            backend = self.probed_backend(server)
            observations = [
                {
                    "fixture": "fx1",
                    "input_ids": list(self.sentinel()),
                    "output_ids": [6, 11, EOS],  # recorded stream differs at index 1
                }
            ]
            section = diffbackends.fidelity_section_data(
                observations=observations,
                stop_texts=("STOP",),
                eos_id=EOS,
                min_tokens=0,
                backend=backend,
                decode=backend.decode_ids,
            )
            self.assertIsNotNone(section)
            row = section["per_fixture"][0]
            self.assertEqual(row["stream"]["class"], "GENERATION_STREAM_DIVERGED")
            self.assertEqual(row["stream"]["first_divergence"], 1)
            self.assertEqual(row["verdict"], "agree")
            self.assertEqual(section["verdict"], "fidelity_unavailable")
            self.assertEqual(section["counts"]["capability_absent"], 1)
            self.assertEqual(section["counts"]["agree"], 1)


class HostileFakeBatteryTests(ServingBackendsTestBase):
    """Hostile fakes: every stack misbehaviour degrades to a
    labelled cell or a closed gate, never a crash or a silent pass."""

    def _record_for_mode(self, mode: str) -> dict[str, Any]:
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.state.mode = mode
            return servingbackends.execute_cell(
                backend,
                request=StackRequest(
                    prompt_ids=self.sentinel(), seed=7, max_new_tokens=4, eos_id=EOS
                ),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
            )

    def test_hostile_fake_battery(self) -> None:
        """Spec: sections 8.1/8.5 -- HTTP 500, malformed JSON,
        wrong-typed tokens, and cap overshoot each land in the
        fixed vocabulary."""
        record = self._record_for_mode("http500")
        self.assertEqual(record["cell_detail"], "stack_execution_failed")
        self.assertIsNone(record["returned_ids"])
        record = self._record_for_mode("broken_json")
        self.assertEqual(record["cell_detail"], "stack_execution_failed")
        record = self._record_for_mode("bad_tokens")
        self.assertEqual(record["cell_detail"], "stack_response_invalid")
        record = self._record_for_mode("overshoot")
        self.assertEqual(record["cell_detail"], "stack_response_invalid")
        self.assertEqual(len(record["returned_ids"]), 7)  # cap 4, tolerance 1
        self.assertEqual(record["mirror"]["fidelity_verdict"], "unobservable")

    def test_gguf_reader_variants(self) -> None:
        """Spec: A11 supply path -- the stdlib GGUF reader extracts
        scalar and array declared ids and fails closed on a
        malformed file."""
        scalar = Path(self._tmpdir.name) / "scalar.gguf"
        scalar.write_bytes(_gguf_bytes([50256]))
        declared = read_gguf_declared_tokens(scalar)
        self.assertEqual(declared["eos_token_ids"], [50256])
        self.assertEqual(declared["bos_token_id"], 0)
        array = Path(self._tmpdir.name) / "array.gguf"
        array.write_bytes(_gguf_bytes([50256, 50257], array_form=True))
        self.assertEqual(read_gguf_declared_tokens(array)["eos_token_ids"], [50256, 50257])
        bad = Path(self._tmpdir.name) / "bad.gguf"
        bad.write_bytes(_gguf_bytes([1], bad_magic=True))
        with self.assertRaises(ValueError) as caught:
            read_gguf_declared_tokens(bad)
        self.assertEqual(str(caught.exception), "stack_checkpoint_unavailable")
        with self.assertRaises(ValueError) as caught:
            LlamacppHttpBackend(
                "http://127.0.0.1:9",
                api_key=API_KEY,
                gguf_path=bad,
                tokenizer=FakeTokenizer(),
                tokenizer_fingerprint="sha256:" + "0" * 64,
            )
        self.assertEqual(str(caught.exception), "stack_checkpoint_unavailable")


class MirrorHostileLeafTests(ServingBackendsTestBase):
    """S-1 (post-cycle-241 discipline): the diffreports mirror for
    serving_backends type-gates every stored leaf before use."""

    def _genuine_section(self, server: _FakeServer) -> dict[str, Any]:
        backend = self.probed_backend(server)
        server.state.mode = "word"
        record = servingbackends.execute_cell(
            backend,
            request=StackRequest(
                prompt_ids=self.sentinel(),
                seed=7,
                max_new_tokens=8,
                stop_texts=("STOP",),
                eos_id=EOS,
            ),
            decode=backend.decode_ids,
            candidate="t",
            fixture="t1",
        )
        return experiments.serving_backends_analysis(
            {
                "available": True,
                "detail": "",
                "identity": backend.identity.as_dict(),
                "capabilities": backend.capabilities.as_dict(),
                "records": [record],
                "cache_sequence": None,
                "fidelity": None,
                "audit": {
                    "n_records": 1,
                    "n_planned_cells": 1,
                    "n_executions": 1,
                    "budget_spent": False,
                },
            }
        )

    def test_mirror_oracle_bruteforce(self) -> None:
        """Spec: brief 2.4 (mirror oracle) -- a genuine section
        validates; every single-leaf hostile mutation is judged
        identically by the production mirror and cycle 67's
        independent oracle, and the corruption-class mutations are
        rejected by both."""
        from tests.test_cycle67_diffreports_audit import _o_serving_backends_valid

        with _FakeServer(self.gguf) as server:
            section = self._genuine_section(server)
        self.assertTrue(diffreports._valid_serving_backends(section))
        self.assertTrue(_o_serving_backends_valid(section))
        junk_values = [[], {}, 7, True, "x" * 65, -1, None, "ab\n"]
        paths: list[tuple[str, ...]] = []

        def walk(value: Any, prefix: tuple[str, ...]) -> None:
            if type(value) is dict:
                for key, item in value.items():
                    walk(item, (*prefix, key))
            elif type(value) is list:
                for index, item in enumerate(value):
                    walk(item, (*prefix, str(index)))
            else:
                paths.append(prefix)

        walk(section, ())
        rejected = 0
        checked = 0
        for path in paths:
            for junk in junk_values:
                mutated = copy.deepcopy(section)
                node: Any = mutated
                for part in path[:-1]:
                    node = node[int(part)] if type(node) is list else node[part]
                last = path[-1]
                if type(node) is list:
                    node[int(last)] = junk
                else:
                    node[last] = junk
                production = diffreports._valid_serving_backends(mutated)
                oracle = _o_serving_backends_valid(mutated)
                self.assertEqual(production, oracle, f"path {path} junk {junk!r}")
                checked += 1
                if not production:
                    rejected += 1
        # The battery is non-vacuous: most leaf corruptions reject.
        self.assertGreater(rejected, checked // 2)

    def test_diff_serving_backends_drift(self) -> None:
        """Spec: section 7.3 -- identical sections do not change;
        a different stack build is identity drift (not comparable),
        listed, and marks the comparison changed."""
        with _FakeServer(self.gguf) as server:
            section = self._genuine_section(server)
        same = diffreports.diff_serving_backends(section, copy.deepcopy(section))
        self.assertEqual(same["available"], "both")
        self.assertFalse(same["changed"])
        self.assertTrue(same["comparable"])
        drifted = copy.deepcopy(section)
        drifted["identity"]["stack_version"] = "b12182-deadbeef"
        drifted["identity"]["config"]["build_info"] = "b12182-deadbeef"
        drifted["identity"]["config_fingerprint"] = hashlib.sha256(
            json.dumps(
                drifted["identity"]["config"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        diff = diffreports.diff_serving_backends(section, drifted)
        self.assertTrue(diff["changed"])
        self.assertFalse(diff["comparable"])
        self.assertEqual(
            [entry["field"] for entry in diff["identity_drift"]],
            ["stack_version", "config_fingerprint"],
        )
        neither = diffreports.diff_serving_backends(None, None)
        self.assertEqual(neither, {"available": "neither", "changed": False})


class CliHandlerTests(ServingBackendsTestBase):
    """The exec-backends handler: dispatch, exit codes, artifacts."""

    def test_cli_exec_backends_end_to_end(self) -> None:
        """Spec: brief 2.1/2.4 -- the handler probes, executes,
        writes a finalized artifact that the mirror validates, and
        exits 0."""
        with _FakeServer(self.gguf) as server:
            server.state.mode = "eos"
            out = Path(self._tmpdir.name) / "artifact.json"
            with mock.patch(
                "model_protocol_lab.doctor._build_primary_tokenizer",
                return_value=FakeTokenizer(),
            ):
                code = servingbackends.run_exec_backends(self._args(server, out), mock.MagicMock())
            self.assertEqual(code, 0)
            doc = json.loads(out.read_text(encoding="utf-8"))
            section = doc["serving_backends"]
            self.assertTrue(section["available"])
            self.assertTrue(diffreports._valid_serving_backends(section))
            rendered = reporting.render_html({"serving_backends": section})
            self.assertIn('id="serving-backends"', rendered)

    def test_cli_exec_backends_unavailable_report(self) -> None:
        """Spec: handler exit discipline -- a probe gate failure is
        a produced report (exit 0) recording the fail-closed
        detail; invalid input exits 2; a tokenizer failure exits 1."""
        out = Path(self._tmpdir.name) / "unavailable.json"
        with mock.patch(
            "model_protocol_lab.doctor._build_primary_tokenizer",
            return_value=FakeTokenizer(),
        ):
            code = servingbackends.run_exec_backends(
                self._args(None, out, timeout=0.5), mock.MagicMock()
            )
        self.assertEqual(code, 0)
        doc = json.loads(out.read_text(encoding="utf-8"))
        self.assertFalse(doc["serving_backends"]["available"])
        self.assertEqual(doc["serving_backends"]["detail"], "stack_unreachable")
        out2 = Path(self._tmpdir.name) / "bad.json"
        with mock.patch(
            "model_protocol_lab.doctor._build_primary_tokenizer",
            side_effect=ValueError("tokenizer_load_failed"),
        ):
            code = servingbackends.run_exec_backends(self._args(None, out2), mock.MagicMock())
        self.assertEqual(code, 1)

    def test_cli_dispatch(self) -> None:
        """Spec: cli ANCHOR pattern -- main() dispatches
        exec-backends to servingbackends.run_exec_backends."""
        from model_protocol_lab import cli

        with mock.patch(
            "model_protocol_lab.servingbackends.run_exec_backends", return_value=0
        ) as handler:
            code = cli.main(
                [
                    "exec-backends",
                    "--base-url",
                    "http://127.0.0.1:8080",
                    "--api-key",
                    "k",
                    "--gguf",
                    "m.gguf",
                    "--model",
                    "m",
                    "--out",
                    "o.json",
                ]
            )
        self.assertEqual(code, 0)
        self.assertTrue(handler.called)
        args = handler.call_args[0][0]
        self.assertEqual(args.command, "exec-backends")
        self.assertEqual(args.hardware_class, "cpu")
        self.assertEqual(args.timeout, 120.0)


if __name__ == "__main__":
    unittest.main()


class DivergenceClassTests(ServingBackendsTestBase):
    """Each section 5.5/9.1 divergence class, produced by a fake
    whose behaviour differs from the predictor in exactly one way."""

    def _record(self, server: _FakeServer, mode: str, **request_kwargs: Any) -> dict[str, Any]:
        backend = self.probed_backend(server)
        server.state.mode = mode
        kwargs: dict[str, Any] = {
            "prompt_ids": self.sentinel(),
            "seed": 7,
            "max_new_tokens": 8,
            "eos_id": EOS,
        }
        kwargs.update(request_kwargs)
        return servingbackends.execute_cell(
            backend,
            request=StackRequest(**kwargs),
            decode=backend.decode_ids,
            candidate="t",
            fixture="t1",
        )

    def test_missing_stop_at_cap(self) -> None:
        """The stop text completes in the returned stream but the
        stack ran to the cap: MISSING_STOP_AT_CAP."""
        with _FakeServer(self.gguf) as server:
            record = self._record(server, "stop_unfired", stop_texts=("STOP",))
            self.assertEqual(record["mirror"]["fidelity_verdict"], "diverge")
            self.assertEqual(record["mirror"]["divergence_class"], "MISSING_STOP_AT_CAP")

    def test_spurious_stop(self) -> None:
        """The stack claims a stop the text never completes:
        SPURIOUS_STOP."""
        with _FakeServer(self.gguf) as server:
            record = self._record(server, "spurious", stop_texts=("STOP",))
            self.assertEqual(record["mirror"]["divergence_class"], "SPURIOUS_STOP")

    def test_firing_step_differs(self) -> None:
        """The stack's stream continues past the step at which the
        stop completes: FIRING_STEP_DIFFERS."""
        with _FakeServer(self.gguf) as server:
            record = self._record(server, "step_differs", stop_texts=("STOP",))
            self.assertEqual(record["mirror"]["divergence_class"], "FIRING_STEP_DIFFERS")

    def test_eos_over_str_precedence(self) -> None:
        """A declared non-primary EOS id wins where the predictor
        has a stop completing at the same step:
        EOS_OVER_STR_PRECEDENCE (attribution passes: 999 is
        declared in this checkpoint)."""
        dual = Path(self._tmpdir.name) / "dual.gguf"
        dual.write_bytes(_gguf_bytes([EOS, 999], array_form=True))
        with _FakeServer(self.gguf) as server:
            server.state.gguf_path = str(dual)
            backend = LlamacppHttpBackend(
                server.url,
                api_key=API_KEY,
                gguf_path=dual,
                tokenizer=FakeTokenizer(),
                tokenizer_fingerprint="sha256:" + "0" * 64,
                timeout_s=10.0,
            )
            backend.probe()
            server.state.mode = "eos_other"
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(
                    prompt_ids=self.sentinel(),
                    seed=7,
                    max_new_tokens=8,
                    stop_texts=("<|pad|>",),
                    eos_id=EOS,
                ),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
            )
            self.assertEqual(record["mirror"]["divergence_class"], "EOS_OVER_STR_PRECEDENCE")

    def test_str_over_eos_precedence(self) -> None:
        """The stack reports a word stop on the step the EOS id
        lands: STR_OVER_EOS_PRECEDENCE."""
        with _FakeServer(self.gguf) as server:
            record = self._record(server, "str_over_eos", stop_texts=("STOP",))
            self.assertEqual(record["mirror"]["divergence_class"], "STR_OVER_EOS_PRECEDENCE")

    def test_kept_vs_stripped(self) -> None:
        """Same winner, same step, but the stack's returned text
        keeps the stop text the predictor strips:
        KEPT_VS_STRIPPED."""
        with _FakeServer(self.gguf) as server:
            record = self._record(server, "word_kept", stop_texts=("STOP",))
            self.assertEqual(record["mirror"]["divergence_class"], "KEPT_VS_STRIPPED")
            self.assertEqual(record["mirror"]["observed_disposition"], "kept")

    def test_disposition_unobservable(self) -> None:
        """The stack's returned text matches neither the full nor
        the stripped digest: the disposition is unobservable, not
        a divergence."""
        with _FakeServer(self.gguf) as server:
            record = self._record(server, "word_weird", stop_texts=("STOP",))
            self.assertEqual(record["mirror"]["fidelity_verdict"], "unobservable")
            self.assertEqual(record["mirror"]["divergence_detail"], "disposition_unobservable")

    def test_precedence_tie_agrees(self) -> None:
        """A stop and the EOS completing on the same step is a
        precedence tie; the stack resolving it to the stop agrees."""
        with _FakeServer(self.gguf) as server:
            record = self._record(server, "tie_stop", stop_texts=("<|endoftext|>",))
            self.assertTrue(record["mirror"]["precedence_tie"])
            self.assertEqual(record["mirror"]["fidelity_verdict"], "agree")

    def test_aggregate_verdict_shapes(self) -> None:
        """Spec: section 9.3 -- the aggregate is a verdict over
        counts: empty is unavailable, any diverge dominates."""
        empty = servingbackends.aggregate_fidelity([])
        self.assertEqual(empty["verdict"], "fidelity_unavailable")
        with _FakeServer(self.gguf) as server:
            record = self._record(server, "spurious", stop_texts=("STOP",))
        aggregate = servingbackends.aggregate_fidelity([record])
        self.assertEqual(aggregate["verdict"], "fidelity_diverge")
        self.assertEqual(aggregate["divergence_classes"], {"SPURIOUS_STOP": 1})


class ProbeFailureTests(ServingBackendsTestBase):
    """Fail-closed probe and execution paths beyond the gates."""

    def test_probe_unreachable(self) -> None:
        """A dead endpoint fails the probe closed as
        stack_unreachable (health deadline honoured)."""
        backend = LlamacppHttpBackend(
            "http://127.0.0.1:9",
            api_key=API_KEY,
            gguf_path=self.gguf,
            tokenizer=FakeTokenizer(),
            tokenizer_fingerprint="sha256:" + "0" * 64,
            timeout_s=1.0,
        )
        with self.assertRaises(ValueError) as caught:
            backend.probe()
        self.assertEqual(str(caught.exception), "stack_unreachable")

    def test_probe_props_failures(self) -> None:
        """A 500 from /props is endpoint-refused (unreachable is
        the connection-level class); a /props whose total_slots
        is not exactly 1 is a topology mismatch; neither is
        measured through."""
        with _FakeServer(self.gguf) as server:
            server.state.props_status = 500
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "stack_endpoint_refused")
        with _FakeServer(self.gguf) as server:
            server.state.props_override = {
                "build_info": "b12182-fd27b20",
                "total_slots": "one",
                "model_path": str(self.gguf),
                "endpoint_slots": True,
                "ui": False,
                "default_generation_settings": {"params": {}, "n_ctx": 1},
            }
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "stack_topology_mismatch")

    def test_probe_vocab_endpoint_failures(self) -> None:
        """A failing /tokenize is a vocab mismatch; a non-text
        /detokenize answer is a prompt-echo mismatch."""
        with _FakeServer(self.gguf) as server:
            server.state.tokenize_status = 500
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "token_vocab_mismatch")
        with _FakeServer(self.gguf) as server:
            server.state.detok_override = lambda _ids: 42
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "prompt_echo_mismatch")

    def test_checkpoint_without_declared_eos(self) -> None:
        """A GGUF declaring no EOS id cannot be attributed:
        construction fails closed; a truncated file likewise."""
        bare = Path(self._tmpdir.name) / "bare.gguf"
        bare.write_bytes(_gguf_bytes(None))
        with self.assertRaises(ValueError) as caught:
            LlamacppHttpBackend(
                "http://127.0.0.1:9",
                api_key=API_KEY,
                gguf_path=bare,
                tokenizer=FakeTokenizer(),
                tokenizer_fingerprint="sha256:" + "0" * 64,
            )
        self.assertEqual(str(caught.exception), "stack_checkpoint_unavailable")
        truncated = Path(self._tmpdir.name) / "truncated.gguf"
        truncated.write_bytes(_gguf_bytes([EOS])[:24])
        with self.assertRaises(ValueError) as caught:
            read_gguf_declared_tokens(truncated)
        self.assertEqual(str(caught.exception), "stack_checkpoint_unavailable")

    def test_execute_requires_probe_and_honours_cancel(self) -> None:
        """Executing before a probe is an identity failure; a set
        cancel event raises ExecutionCancelled, never a record."""
        import threading

        with _FakeServer(self.gguf) as server:
            backend = self.make_backend(server)
            request = StackRequest(prompt_ids=self.sentinel(), seed=7)
            with self.assertRaises(ValueError) as caught:
                backend.execute(request)
            self.assertEqual(str(caught.exception), "stack_identity_unavailable")
            backend.probe()
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(servingbackends.ExecutionCancelled):
                backend.execute(request, cancel=cancel)

    def test_execution_after_server_death(self) -> None:
        """Death mid-run degrades cells to stack_execution_failed
        and identity/erase checks to False -- never an exception
        escaping execute_cell, never a fabricated record."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.httpd.shutdown()
            server.httpd.server_close()
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(prompt_ids=self.sentinel(), seed=7),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
            )
            self.assertEqual(record["cell_detail"], "stack_execution_failed")
            self.assertFalse(backend.verify_identity())
            self.assertFalse(backend.erase_slot())

    def test_record_without_cache_fields(self) -> None:
        """A stack reporting no cache quantities records None for
        each -- recorded-as-reported, never back-filled."""
        with _FakeServer(self.gguf) as server:
            server.state.no_cache_fields = True
            backend = self.probed_backend(server)
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(prompt_ids=self.sentinel(), seed=7, cache_mode="cold"),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
                cache_position="cold",
            )
            self.assertIsNone(record["cache_n_reported"])
            self.assertIsNone(record["tokens_cached_reported"])
            self.assertIsNone(record["tokens_evaluated_reported"])

    def test_request_validation_extra(self) -> None:
        """Spec: sections 5.2/8.5 -- typed-field gates: bool seed,
        negative eos, bool stop-token id, non-text cache salt, and
        the prompt cap all fail with the fixed code."""
        for kwargs in (
            {"prompt_ids": (1,), "seed": True},
            {"prompt_ids": (1,), "seed": 1, "eos_id": -1},
            {"prompt_ids": (1,), "seed": 1, "stop_token_ids": (True,)},
            {"prompt_ids": (1,), "seed": 1, "cache_salt": 5},
            {"prompt_ids": tuple(range(131_073)), "seed": 1},
        ):
            with self.assertRaises(ValueError) as caught:
                StackRequest(**kwargs)
            self.assertEqual(str(caught.exception), "stack_request_invalid")


class CacheEdgeTests(ServingBackendsTestBase):
    """Cache-sequence edges: cold-arm reuse, mid-run identity
    change, unreported quantities."""

    def test_cold_arm_reuse_contaminates(self) -> None:
        """C-3: a cold arm reporting reuse above one quantum is
        contamination whatever the erase returned."""
        with _FakeServer(self.gguf) as server:
            server.state.cache_n_override = 5
            backend = self.probed_backend(server)
            sequence = run_cache_sequence(
                backend,
                cold_ids=self.sentinel(),
                shared_ids=self.sentinel() + (9,),
                divergent_ids=(8, 12),
                seed=3,
                decode=backend.decode_ids,
            )
            self.assertEqual(sequence["detail"], "cache_protocol_contaminated")
            self.assertEqual(sequence["positions"]["cold"]["cache_n"], 5)

    def test_identity_change_mid_sequence(self) -> None:
        """A8(i): identity verified again at the end -- a change
        during the arms keeps the records and labels the
        sequence."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.state.flip_build_info_after = server.state.completion_calls + 2
            sequence = run_cache_sequence(
                backend,
                cold_ids=self.sentinel(),
                shared_ids=self.sentinel() + (9,),
                divergent_ids=(8, 12),
                seed=3,
                decode=backend.decode_ids,
            )
            self.assertEqual(sequence["detail"], "stack_identity_changed_during_run")
            self.assertEqual(len(sequence["records"]), 4)
            self.assertTrue(sequence["available"])

    def test_unreported_cache_quantities(self) -> None:
        """A stack reporting no cache_n leaves the positions
        unreported: the sequence detail is capability_absent, the
        cells stand."""
        with _FakeServer(self.gguf) as server:
            server.state.no_cache_fields = True
            backend = self.probed_backend(server)
            sequence = run_cache_sequence(
                backend,
                cold_ids=self.sentinel(),
                shared_ids=self.sentinel() + (9,),
                divergent_ids=(8, 12),
                seed=3,
                decode=backend.decode_ids,
            )
            self.assertEqual(sequence["detail"], "capability_absent")
            self.assertIsNone(sequence["positions"]["cold"]["cache_n"])


class SinkLoaderStageTests(ServingBackendsTestBase):
    """A8(iv) sink budget + loader failure shapes; run_stage
    budget accounting."""

    def test_sink_budget_marker_and_loader(self) -> None:
        """Over-budget appends write one marker and return False;
        the loader then reads the planned cells back as
        budget-exceeded, and appending after finalize raises."""
        path = Path(self._tmpdir.name) / "budget.jsonl"
        plan = [
            {"candidate": "t", "fixture": "t1", "cache_position": "standalone"},
            {"candidate": "t", "fixture": "t2", "cache_position": "standalone"},
        ]
        record = {"candidate": "t", "fixture": "t1", "cache_position": "standalone"}
        with mock.patch.object(servingbackends, "MAX_RECORD_BYTES", 10):
            sink = RecordSink(path, header={"kind": "exec-backends-records", "plan": plan})
            self.assertFalse(sink.append(record))
            self.assertFalse(sink.append(record))
            self.assertTrue(sink.budget_tripped)
            sink.finalize({})
        with self.assertRaises(ValueError):
            sink.append(record)
        loaded = load_records_artifact(path)
        self.assertEqual(len(loaded["missing"]), 2)
        self.assertTrue(
            all(
                entry["cell_detail"] == servingbackends.EXECUTION_BUDGET_DETAIL
                for entry in loaded["missing"]
            )
        )

    def test_loader_invalid_shapes(self) -> None:
        """Unreadable, shapeless, and headerless artifacts are
        invalid with the fixed code; a finalized doc with a
        wrong-typed records field reads back empty, not crashing."""
        with self.assertRaises(ValueError):
            load_records_artifact(Path(self._tmpdir.name) / "absent.jsonl")
        shapeless = Path(self._tmpdir.name) / "shapeless.json"
        shapeless.write_text('{"foo": 1}', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_records_artifact(shapeless)
        headerless = Path(self._tmpdir.name) / "headerless.jsonl"
        headerless.write_text('{"record": {}}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_records_artifact(headerless)
        wrong = Path(self._tmpdir.name) / "wrong.json"
        wrong.write_text('{"serving_backends": {"records": 5}}', encoding="utf-8")
        loaded = load_records_artifact(wrong)
        self.assertTrue(loaded["finalized"])
        self.assertEqual(loaded["records"], [])

    def test_run_stage_execution_budget(self) -> None:
        """Spec: section 8.5 -- past the execution cap, remaining
        cells are budget records, counted in the audit."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            cells = [
                {
                    "candidate": "t",
                    "fixture": f"t{i}",
                    "request": StackRequest(prompt_ids=self.sentinel(), seed=7),
                    "cache_position": "standalone",
                }
                for i in range(3)
            ]
            with mock.patch.object(servingbackends, "MAX_EXECUTIONS_PER_RUN", 1):
                stage = servingbackends.run_stage(
                    backend,
                    plan_cells=cells,
                    fidelity_input=None,
                    cache_prompts=None,
                    seed=0,
                    decode=backend.decode_ids,
                    capabilities=backend.capabilities,
                )
            self.assertEqual(len(stage["records"]), 3)
            self.assertEqual(stage["audit"]["n_executions"], 1)
            self.assertTrue(stage["audit"]["budget_spent"])
            self.assertEqual(stage["detail"], servingbackends.EXECUTION_BUDGET_DETAIL)
            self.assertEqual(
                stage["records"][1]["cell_detail"],
                servingbackends.EXECUTION_BUDGET_DETAIL,
            )

    def test_run_stage_cache_budget_shape(self) -> None:
        """With the budget spent before the cache sequence, the
        cache cells are budget records too -- labelled, counted,
        never silently dropped."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            cells = [
                {
                    "candidate": "t",
                    "fixture": "t0",
                    "request": StackRequest(prompt_ids=self.sentinel(), seed=7),
                    "cache_position": "standalone",
                }
            ]
            with mock.patch.object(servingbackends, "MAX_EXECUTIONS_PER_RUN", 0):
                stage = servingbackends.run_stage(
                    backend,
                    plan_cells=cells,
                    fidelity_input=None,
                    cache_prompts={
                        "cold_ids": self.sentinel(),
                        "shared_prefix_ids": self.sentinel() + (9,),
                        "divergent_control_ids": (8, 12),
                    },
                    seed=0,
                    decode=backend.decode_ids,
                    capabilities=backend.capabilities,
                )
            self.assertEqual(len(stage["records"]), 5)
            self.assertEqual(
                stage["cache_sequence"]["detail"],
                servingbackends.EXECUTION_BUDGET_DETAIL,
            )


class HandlerEdgeTests(ServingBackendsTestBase):
    """run_exec_backends input discipline: parser errors are
    SystemExit via parser.error; value failures map to the
    documented exit codes."""

    def _parser(self) -> Any:
        import argparse

        return argparse.ArgumentParser(prog="exec-backends")

    def _run(self, server: _FakeServer | None, out: Path, **overrides: Any) -> int:
        with mock.patch(
            "model_protocol_lab.doctor._build_primary_tokenizer",
            return_value=FakeTokenizer(),
        ):
            return servingbackends.run_exec_backends(
                self._args(server, out, **overrides), self._parser()
            )

    def test_handler_out_exists_and_bad_inputs(self) -> None:
        out = Path(self._tmpdir.name) / "taken.json"
        out.write_text("{}", encoding="utf-8")
        with self.assertRaises(SystemExit):
            self._run(None, out)
        with _FakeServer(self.gguf) as server:
            fresh = Path(self._tmpdir.name) / "a.json"
            with self.assertRaises(SystemExit):
                self._run(server, fresh, stops="STOP,,X")
            fresh2 = Path(self._tmpdir.name) / "b.json"
            # The shared stop-text checker rejects over-long stops.
            self.assertEqual(self._run(server, fresh2, stops="x" * 100_000), 2)
            fresh3 = Path(self._tmpdir.name) / "c.json"
            with self.assertRaises(SystemExit):
                self._run(server, fresh3, prompt_ids="1,x")
            fresh4 = Path(self._tmpdir.name) / "d.json"
            self.assertEqual(self._run(server, fresh4, max_new_tokens=0), 2)
            fresh5 = Path(self._tmpdir.name) / "e.json"
            self.assertEqual(self._run(server, fresh5, base_url="https://127.0.0.1:8080"), 2)
            fresh6 = Path(self._tmpdir.name) / "f.json"
            self.assertEqual(self._run(server, fresh6, gguf=out), 1)

    def test_handler_results_discipline(self) -> None:
        """--results needs stops and observations; a good results
        file drives the fidelity arm end to end."""
        with _FakeServer(self.gguf) as server:
            server.state.mode = "eos"
            out = Path(self._tmpdir.name) / "r1.json"
            results = Path(self._tmpdir.name) / "results.json"
            results.write_text(
                json.dumps(
                    {
                        "search": [
                            {
                                "observations": [
                                    {
                                        "fixture": "fx",
                                        "input_ids": list(self.sentinel()),
                                        "output_ids": [6, EOS],
                                    }
                                ]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                self._run(server, out, results=results)  # no --stops
            out2 = Path(self._tmpdir.name) / "r2.json"
            self.assertEqual(
                self._run(
                    server,
                    out2,
                    results=Path(self._tmpdir.name) / "nope.json",
                    stops="STOP",
                ),
                2,
            )
            out3 = Path(self._tmpdir.name) / "r3.json"
            empty = Path(self._tmpdir.name) / "empty.json"
            empty.write_text('{"search": []}', encoding="utf-8")
            with self.assertRaises(SystemExit):
                self._run(server, out3, results=empty, stops="STOP")
            out4 = Path(self._tmpdir.name) / "r4.json"
            code = self._run(server, out4, results=results, stops="STOP")
            self.assertEqual(code, 0)
            section = json.loads(out4.read_text(encoding="utf-8"))["serving_backends"]
            self.assertTrue(section["fidelity"]["available"])

    def test_handler_cache_flags(self) -> None:
        """Cache flags are all-or-none; together they run the
        sequence into the artifact."""
        with _FakeServer(self.gguf) as server:
            out = Path(self._tmpdir.name) / "k1.json"
            with self.assertRaises(SystemExit):
                self._run(server, out, cache_cold_ids="1,2,3,4,5")
            out2 = Path(self._tmpdir.name) / "k2.json"
            code = self._run(
                server,
                out2,
                cache_cold_ids="1,2,3,4,5",
                cache_shared_ids="1,2,3,4,5,9",
                cache_divergent_ids="8,12",
            )
            self.assertEqual(code, 0)
            section = json.loads(out2.read_text(encoding="utf-8"))["serving_backends"]
            self.assertTrue(section["cache"]["available"])
            self.assertEqual(section["cache"]["positions"]["cold"]["cache_n"], 0)


class GgufWalkerTests(unittest.TestCase):
    """The GGUF reader's bounds: every malformation fails closed
    with the fixed code (hostile-checkpoint supply path)."""

    def _write(self, data: bytes) -> Path:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as handle:
            handle.write(data)
            path = Path(handle.name)
        self.addCleanup(path.unlink)
        return path

    def _header(self, kv_count: int) -> bytes:
        return b"GGUF" + struct.pack("<IQQ", 3, 0, kv_count)

    def _kv(self, key: str, value_type: int, payload: bytes) -> bytes:
        raw = key.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw + struct.pack("<I", value_type) + payload

    def test_walker_malformations(self) -> None:
        """Oversized key counts, oversized strings, unknown value
        types, unknown array element types, oversized arrays, and
        short files are all stack_checkpoint_unavailable."""
        cases = [
            self._header(10**9),  # metadata count over the cap
            b"GG",  # short header
            self._header(1) + struct.pack("<Q", 10**9),  # key string over cap
            self._header(1) + self._kv("x.y", 99, b""),  # unknown value type
            self._header(1)
            + self._kv("x.y", 9, struct.pack("<I", 99) + struct.pack("<Q", 1)),  # bad elem
            self._header(1)
            + self._kv("x.y", 9, struct.pack("<I", 4) + struct.pack("<Q", 10**9)),  # huge array
            self._header(1)
            + self._kv(
                "tokenizer.ggml.eos_token_id",
                9,
                struct.pack("<I", 4) + struct.pack("<Q", 5000) + b"\x00" * 20000,
            ),  # declared array over the value cap
        ]
        for data in cases:
            with self.assertRaises(ValueError) as caught:
                read_gguf_declared_tokens(self._write(data))
            self.assertEqual(str(caught.exception), "stack_checkpoint_unavailable")
        with self.assertRaises(ValueError):
            read_gguf_declared_tokens(Path("/nonexistent/nope.gguf"))

    def test_walker_int32_declared_eos(self) -> None:
        """An INT32-typed declared id reads as a signed value."""
        data = self._header(1) + self._kv(
            "tokenizer.ggml.eos_token_id", 5, struct.pack("<I", 50256)
        )
        declared = read_gguf_declared_tokens(self._write(data))
        self.assertEqual(declared["eos_token_ids"], [50256])
        negative = self._header(1) + self._kv(
            "tokenizer.ggml.eos_token_id", 5, struct.pack("<I", 0xFFFFFFFF)
        )
        self.assertEqual(read_gguf_declared_tokens(self._write(negative))["eos_token_ids"], [-1])


class CoverageBranchTests(ServingBackendsTestBase):
    """Remaining behavioural branches, each tied to its contract."""

    def test_detok_none_and_g4_failure(self) -> None:
        """A /detokenize answering null content fails G3 as a
        prompt-echo mismatch; one failing only on the EOT-extended
        ids fails G4 as a vocab mismatch."""
        with _FakeServer(self.gguf) as server:
            server.state.detok_override = lambda _ids: None
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "prompt_echo_mismatch")
        with _FakeServer(self.gguf) as server:
            server.state.detok_override = lambda ids: None if EOS in ids else fake_decode(ids)
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "token_vocab_mismatch")
        with _FakeServer(self.gguf) as server:
            server.state.detok_override = lambda _ids: "different text"
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "prompt_echo_mismatch")

    def test_erase_slot_direct(self) -> None:
        """erase_slot returns the server's answer as data: False
        when the slots endpoint refuses (the fallback posture)."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            self.assertTrue(backend.erase_slot())
            server.state.erase_enabled = False
            self.assertFalse(backend.erase_slot())

    def test_sequence_erase_fails_at_start(self) -> None:
        """Erase available at probe time but refused when the
        sequence starts: the four cells are capability_absent
        records, not an exception."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.state.erase_enabled = False
            sequence = run_cache_sequence(
                backend,
                cold_ids=self.sentinel(),
                shared_ids=self.sentinel() + (9,),
                divergent_ids=(8, 12),
                seed=3,
                decode=backend.decode_ids,
            )
            self.assertFalse(sequence["available"])
            self.assertEqual(sequence["detail"], "capability_absent")
            self.assertEqual(len(sequence["records"]), 4)

    def test_sequence_cancel(self) -> None:
        """A set cancel event aborts the cache sequence with
        ExecutionCancelled."""
        import threading

        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(servingbackends.ExecutionCancelled):
                run_cache_sequence(
                    backend,
                    cold_ids=self.sentinel(),
                    shared_ids=self.sentinel() + (9,),
                    divergent_ids=(8, 12),
                    seed=3,
                    decode=backend.decode_ids,
                    cancel=cancel,
                )

    def test_winner_without_stop_index(self) -> None:
        """A stopping_word the request never listed records a
        winner with index None (the stack's claim, unbound)."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.state.mode = "spurious"
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(prompt_ids=self.sentinel(), seed=7, max_new_tokens=4),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
            )
            self.assertEqual(
                record["stop_winner"],
                {
                    "kind": "stop_text",
                    "index": None,
                    "digest": hashlib.sha256(b"STOP").hexdigest(),
                },
            )
            self.assertEqual(record["mirror"]["divergence_class"], "SPURIOUS_STOP")

    def test_same_step_other_stop_agrees(self) -> None:
        """Two stops completing on the same step: the stack
        reporting the other index is a tie resolved the other
        way -- the firing event agrees."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.state.mode = "tie_stops"
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(
                    prompt_ids=self.sentinel(),
                    seed=7,
                    max_new_tokens=4,
                    stop_texts=("ab", "b"),
                ),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
            )
            self.assertEqual(record["mirror"]["fidelity_verdict"], "agree")
            self.assertEqual(record["stop_winner"]["index"], 1)

    def test_aggregate_unobservable_only(self) -> None:
        """An aggregate of only-unobservable cells is
        fidelity_unavailable with the unobservable count."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.state.mode = "word_weird"
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(
                    prompt_ids=self.sentinel(),
                    seed=7,
                    max_new_tokens=8,
                    stop_texts=("STOP",),
                    eos_id=EOS,
                ),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
            )
        aggregate = servingbackends.aggregate_fidelity([record])
        self.assertEqual(aggregate["verdict"], "fidelity_unavailable")
        self.assertEqual(aggregate["counts"]["unobservable"], 1)

    def test_sink_finalize_idempotent_and_complete(self) -> None:
        """A finalized sink reads back complete with no missing
        cells; finalize/close are idempotent."""
        path = Path(self._tmpdir.name) / "done.jsonl"
        plan = [{"candidate": "t", "fixture": "t1", "cache_position": "standalone"}]
        record = {"candidate": "t", "fixture": "t1", "cache_position": "standalone"}
        sink = RecordSink(path, header={"kind": "exec-backends-records", "plan": plan})
        self.assertTrue(sink.append(record))
        sink.finalize({"n_records": 1})
        sink.finalize({"n_records": 1})
        sink.close()
        sink.close()
        loaded = load_records_artifact(path)
        self.assertTrue(loaded["complete"])
        self.assertEqual(loaded["missing"], [])
        self.assertEqual(len(loaded["records"]), 1)

    def test_loader_non_dict_line(self) -> None:
        """A JSONL line that is not an object is an invalid
        artifact, not a skipped line."""
        path = Path(self._tmpdir.name) / "bad.jsonl"
        path.write_text('{"header": {"plan": []}}\n[1, 2]\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_records_artifact(path)

    def test_run_stage_full_shape(self) -> None:
        """run_stage happy path with fidelity and cache arms:
        records from all three parts, cache result carried."""
        with _FakeServer(self.gguf) as server:
            server.state.mode = "eos"
            backend = self.probed_backend(server)
            stage = servingbackends.run_stage(
                backend,
                plan_cells=[
                    {
                        "candidate": "t",
                        "fixture": "t0",
                        "request": StackRequest(prompt_ids=self.sentinel(), seed=7, eos_id=EOS),
                        "cache_position": "standalone",
                    }
                ],
                fidelity_input={
                    "observations": [
                        {
                            "fixture": "fx",
                            "input_ids": list(self.sentinel()),
                            "output_ids": [6, EOS],
                        }
                    ],
                    "stop_texts": ("STOP",),
                    "eos_id": EOS,
                    "min_tokens": 0,
                },
                cache_prompts={
                    "cold_ids": self.sentinel(),
                    "shared_prefix_ids": self.sentinel() + (9,),
                    "divergent_control_ids": (8, 12),
                },
                seed=0,
                decode=backend.decode_ids,
                capabilities=backend.capabilities,
            )
            self.assertTrue(stage["available"])
            self.assertTrue(stage["cache_sequence"]["available"])
            self.assertIsNotNone(stage["fidelity"])
            self.assertEqual(stage["audit"]["n_records"], len(stage["records"]))
            section = experiments.serving_backends_analysis(stage)
            self.assertTrue(diffreports._valid_serving_backends(section))

    def test_run_stage_requires_identity_and_honours_cancel(self) -> None:
        """run_stage on an unprobed backend is an identity
        failure; a preset cancel aborts the plan."""
        import threading

        with _FakeServer(self.gguf) as server:
            backend = self.make_backend(server)
            with self.assertRaises(ValueError):
                servingbackends.run_stage(
                    backend,
                    plan_cells=[],
                    fidelity_input=None,
                    cache_prompts=None,
                    seed=0,
                    decode=backend.decode_ids,
                    capabilities=backend.capabilities,
                )
            backend.probe()
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(servingbackends.ExecutionCancelled):
                servingbackends.run_stage(
                    backend,
                    plan_cells=[
                        {
                            "candidate": "t",
                            "fixture": "t0",
                            "request": StackRequest(prompt_ids=self.sentinel(), seed=7),
                            "cache_position": "standalone",
                        }
                    ],
                    fidelity_input=None,
                    cache_prompts=None,
                    seed=0,
                    decode=backend.decode_ids,
                    capabilities=backend.capabilities,
                    cancel=cancel,
                )

    def test_tokenizer_fingerprint_assets(self) -> None:
        """The tokenizer fingerprint binds the asset manifest:
        an empty directory and one carrying tokenizer.json
        fingerprint differently."""
        empty = servingbackends._tokenizer_fingerprint(Path(self._tmpdir.name))
        assets = Path(self._tmpdir.name) / "with-assets"
        assets.mkdir()
        (assets / "tokenizer.json").write_text("{}", encoding="utf-8")
        self.assertNotEqual(servingbackends._tokenizer_fingerprint(assets), empty)
        self.assertEqual(servingbackends._parse_id_list("1, 2,3"), (1, 2, 3))
        self.assertIsNone(servingbackends._parse_id_list("1,,2"))
        self.assertIsNone(servingbackends._parse_id_list("1,-2"))
        self.assertIsNone(servingbackends._parse_id_list(""))

    def test_handler_sink_unwritable(self) -> None:
        """An artifact path in a nonexistent directory cannot be
        opened: exit 1, no partial artifact."""
        with _FakeServer(self.gguf) as server:
            out = Path(self._tmpdir.name) / "no-dir" / "x.json"
            code = self._run_handler(server, out)
            self.assertEqual(code, 1)

    def _run_handler(self, server: _FakeServer, out: Path) -> int:
        with mock.patch(
            "model_protocol_lab.doctor._build_primary_tokenizer",
            return_value=FakeTokenizer(),
        ):
            return servingbackends.run_exec_backends(self._args(server, out), mock.MagicMock())


class CoverageRecoveryTests(ServingBackendsTestBase):
    """Coverage-recovery battery (gate-floor discipline, cycles
    167/233/237 precedent): the first full coverage leg measured
    the new code's reachable error/edge branches below the cycle
    floor; each test below pins one such branch to its contract —
    a fail-closed code, a recorded-as-data outcome, or a documented
    exit code — against the same hostile fake."""

    # -- pure helpers and request validation --

    def test_pure_helper_branches(self) -> None:
        """Spec: G1/G2 helper contracts -- the build-info parser
        rejects over-long/non-hex tails; the Tier-A echo comparison
        is exact per field type; invalid seeds and request fields
        fail with the fixed code."""
        commit = servingbackends._build_info_commit
        self.assertIsNone(commit("b12182-" + "f" * 41))
        self.assertIsNone(commit("b12182-zzzzzzz"))
        self.assertIsNone(commit("x12182-fd27b20"))
        self.assertIsNone(commit(7))
        self.assertEqual(commit("b12182-fd27b20"), "fd27b20")
        echo = servingbackends._echo_matches
        params = servingbackends.tier_a_sampler_params(3)
        good = dict(params)
        self.assertTrue(echo(good, params))
        self.assertFalse(echo("nope", params))
        missing = dict(good)
        del missing["seed"]
        self.assertFalse(echo(missing, params))
        reordered = dict(good, samplers=["top_p", "top_k", "min_p", "temperature"])
        self.assertFalse(echo(reordered, params))
        self.assertFalse(echo(dict(good, temperature=True), params))
        self.assertFalse(echo(dict(good, temperature=0.5), params))
        self.assertFalse(echo(dict(good, top_k=0.0), params))
        self.assertFalse(echo(dict(good, top_k=1), params))
        self.assertFalse(echo({"a": "x"}, {"a": "y"}))
        for bad_seed in (True, -1, "3"):
            with self.assertRaises(ValueError):
                servingbackends.tier_a_sampler_params(bad_seed)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            StackRequest(prompt_ids=(1, 2), seed=1, cache_salt=7)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            StackRequest(prompt_ids=(1, 2), seed=1, stop_token_ids=(True,))
        observation = servingbackends.StackObservation(
            returned_ids=(6,),
            finish_class="length",
            stop_winner=None,
            cache_n_reported=None,
            tokens_cached_reported=None,
            tokens_evaluated_reported=None,
            returned_text_digest=None,
            server_detok_digest=None,
            config_echo_digest=None,
        )
        self.assertEqual(observation.as_dict()["returned_ids"], [6])
        caps = servingbackends.StackCapabilities(
            raw_token_ids=True,
            stop_token_ids=False,
            cached_token_reporting=False,
            token_id_echo=True,
            prompt_echo=True,
            min_tokens=False,
            include_stop_str_in_output=False,
        )
        self.assertTrue(caps.as_dict()["raw_token_ids"])

    # -- GGUF walker skip/rewind shapes --

    def _gguf_kv(self, key: str, value_type: int, payload: bytes) -> bytes:
        raw = key.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw + struct.pack("<I", value_type) + payload

    def _gguf_doc(self, kvs: list[bytes]) -> Path:
        path = Path(self._tmpdir.name) / "shaped.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs))
        return path

    def test_gguf_skip_and_rewind_shapes(self) -> None:
        """Spec: A11 reader bounds -- uninteresting scalar and
        array values are skipped; a wanted key carrying an
        unexpected array element type is rewound and skipped; a
        wanted key carrying a float reads as absent, not a crash."""
        eos = self._gguf_kv("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", EOS))
        scalar = self._gguf_kv("general.name", 4, struct.pack("<I", 7))
        array = self._gguf_kv(
            "x.arr", 9, struct.pack("<I", 4) + struct.pack("<Q", 2) + struct.pack("<II", 1, 2)
        )
        declared = read_gguf_declared_tokens(self._gguf_doc([scalar, array, eos]))
        self.assertEqual(declared["eos_token_ids"], [EOS])
        typed_array = self._gguf_kv(
            "general.file_type", 9, struct.pack("<I", 0) + struct.pack("<Q", 3) + b"abc"
        )
        declared = read_gguf_declared_tokens(self._gguf_doc([typed_array, eos]))
        self.assertIsNone(declared["file_type"])
        floated = self._gguf_kv("tokenizer.ggml.eos_token_id", 6, struct.pack("<f", 1.5))
        declared = read_gguf_declared_tokens(self._gguf_doc([floated]))
        self.assertEqual(declared["eos_token_ids"], [])

    # -- constructor and tokenizer faults --

    def test_constructor_and_tokenizer_faults(self) -> None:
        """Spec: section 5.2/A11 -- an unreadable checkpoint fails
        construction closed; a tokenizer whose decode is not text
        or whose encode is not ids fails the probe as a vocabulary
        mismatch, at the gate that consumes it."""

        class BadDecode(FakeTokenizer):
            def decode(self, ids: Any, **kwargs: Any) -> Any:
                return b"bytes"

        class BadEncode(FakeTokenizer):
            def encode(self, text: str, **kwargs: Any) -> Any:
                return ["x"]

        class FlakyEncode(FakeTokenizer):
            def encode(self, text: str, **kwargs: Any) -> Any:
                if "endoftext" in text:
                    return ["x"]
                return super().encode(text, **kwargs)

        with _FakeServer(self.gguf) as server:
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server, gguf_path=Path(self._tmpdir.name))
            self.assertEqual(str(caught.exception), "stack_checkpoint_unavailable")
            for tokenizer in (BadDecode(), BadEncode(), FlakyEncode()):
                with self.assertRaises(ValueError) as caught:
                    self.make_backend(server, tokenizer=tokenizer).probe()
                self.assertEqual(str(caught.exception), "token_vocab_mismatch")

    # -- probe transport faults --

    def test_probe_transport_faults(self) -> None:
        """Spec: section 8.1 -- a transport failure at each probe
        stage surfaces as its fixed code, never a hang or a raw
        urllib error: props (stack_unreachable), completions
        (stack_unreachable via the G2 observation), tokenize
        (vocabulary mismatch), a late completion failure (execution
        failed at G3), and an unparseable build tail (identity
        unavailable)."""
        with _FakeServer(self.gguf) as server:
            server.state.drop_paths = {"/props"}
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "stack_unreachable")
        with _FakeServer(self.gguf) as server:
            server.state.drop_paths = {"/completion"}
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "stack_unreachable")
        with _FakeServer(self.gguf) as server:
            server.state.drop_paths = {"/tokenize"}
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "token_vocab_mismatch")
        with _FakeServer(self.gguf) as server:
            server.state.tokenize_override = lambda _text: "abc"  # noqa: E731
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "token_vocab_mismatch")
        with _FakeServer(self.gguf) as server:
            server.state.fail_completions_after = 1
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "stack_execution_failed")
        with _FakeServer(self.gguf) as server:
            server.state.build_info = "b12182-" + "f" * 41
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "stack_identity_unavailable")

    # -- execute transport and classification faults --

    def test_execute_transport_and_classify_faults(self) -> None:
        """Spec: sections 5.2/8.1/A8 -- identity re-verification is
        False before the probe and on an unreachable re-read; a
        dropped detokenize is recorded as a missing server detok
        digest; an EOS finish with no tokens and a word finish
        with no stopping word are response-invalid; an oversized
        body is an execution failure, not a parse crash."""
        with _FakeServer(self.gguf) as server:
            backend = self.make_backend(server)
            self.assertFalse(backend.verify_identity())
            backend.probe()
            self.assertTrue(backend.verify_identity())
            server.state.drop_paths = {"/props"}
            self.assertFalse(backend.verify_identity())
            server.state.drop_paths = {"/detokenize"}
            observation = backend.execute(
                StackRequest(prompt_ids=self.sentinel(), seed=5, max_new_tokens=4)
            )
            self.assertIsNone(observation.server_detok_digest)
            server.state.drop_paths = set()
            server.state.mode = "eos_empty"
            observation = backend.execute(
                StackRequest(prompt_ids=self.sentinel(), seed=5, max_new_tokens=4)
            )
            self.assertEqual(observation.detail, "stack_response_invalid")
            server.state.mode = "word_missing"
            observation = backend.execute(
                StackRequest(
                    prompt_ids=self.sentinel(),
                    seed=5,
                    max_new_tokens=4,
                    stop_texts=("STOP",),
                )
            )
            self.assertEqual(observation.detail, "stack_response_invalid")
            server.state.mode = "huge"
            observation = backend.execute(
                StackRequest(prompt_ids=self.sentinel(), seed=5, max_new_tokens=4)
            )
            self.assertEqual(observation.detail, "stack_execution_failed")

    # -- fidelity verdict edges --

    def test_fidelity_verdict_edges(self) -> None:
        """Spec: section 5.5 -- a stack text matching neither the
        local nor the stripped digest is disposition-unobservable;
        a same-step stop/EOS tie the stack resolved for EOS still
        agrees; unknown aggregate verdicts count as unobservable;
        the verdict function's defensive guards answer
        unobservable on inconsistent predictions; unprobed drivers
        fail with the identity code."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.state.mode = "word_weird"
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(
                    prompt_ids=self.sentinel(),
                    seed=7,
                    max_new_tokens=8,
                    stop_texts=("STOP",),
                    eos_id=EOS,
                ),
                decode=backend.decode_ids,
                candidate="t",
                fixture="weird",
            )
            self.assertEqual(record["mirror"]["fidelity_verdict"], "unobservable")
            self.assertEqual(record["mirror"]["divergence_detail"], "disposition_unobservable")
            server.state.mode = "eos"
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(
                    prompt_ids=self.sentinel(),
                    seed=7,
                    max_new_tokens=8,
                    stop_texts=("<|endoftext|>",),
                    eos_id=EOS,
                ),
                decode=backend.decode_ids,
                candidate="t",
                fixture="tie",
            )
            self.assertEqual(record["mirror"]["fidelity_verdict"], "agree")
        aggregate = servingbackends.aggregate_fidelity([{}])
        self.assertEqual(aggregate["counts"]["unobservable"], 1)
        request = StackRequest(prompt_ids=(1, 2), seed=1, eos_id=EOS)
        base_observation = servingbackends.StackObservation(
            returned_ids=(6,),
            finish_class="length",
            stop_winner={"kind": "eos", "id": EOS},
            cache_n_reported=None,
            tokens_cached_reported=None,
            tokens_evaluated_reported=None,
            returned_text_digest=None,
            server_detok_digest=None,
            config_echo_digest=None,
        )
        verdict = servingbackends._fidelity_verdict(
            request=request,
            observation=base_observation,
            prediction={
                "predicted_winner": {"kind": "eos", "id": EOS},
                "predicted_firing_step": None,
                "stop_steps": {},
                "precedence_tie": False,
                "local_text_digest": "x",
                "stripped_text_digest": None,
            },
            declared_eos_ids=(EOS,),
        )
        self.assertEqual(verdict[0], "unobservable")
        self.assertEqual(verdict[2], "predictor_step_missing")
        word_observation = servingbackends.StackObservation(
            returned_ids=(6, 7),
            finish_class="word",
            stop_winner={"kind": "stop_text", "index": 1, "digest": "d"},
            cache_n_reported=None,
            tokens_cached_reported=None,
            tokens_evaluated_reported=None,
            returned_text_digest="x",
            server_detok_digest=None,
            config_echo_digest=None,
        )
        verdict = servingbackends._fidelity_verdict(
            request=request,
            observation=word_observation,
            prediction={
                "predicted_winner": {"kind": "stop_text", "index": 0},
                "predicted_firing_step": 2,
                "stop_steps": {"0": 2, "1": 1},
                "precedence_tie": False,
                "local_text_digest": "x",
                "stripped_text_digest": None,
            },
            declared_eos_ids=(EOS,),
        )
        self.assertEqual(verdict[0], "diverge")
        self.assertEqual(verdict[1], "FIRING_STEP_DIFFERS")
        with _FakeServer(self.gguf) as server:
            unprobed = self.make_backend(server)
            with self.assertRaises(ValueError):
                servingbackends.execute_cell(
                    unprobed,
                    request=request,
                    decode=FakeTokenizer().decode,
                    candidate="t",
                    fixture="t",
                )
            with self.assertRaises(ValueError):
                run_cache_sequence(
                    unprobed,
                    cold_ids=(1,),
                    shared_ids=(1,),
                    divergent_ids=(2,),
                    seed=0,
                    decode=FakeTokenizer().decode,
                )
            with self.assertRaises(ValueError):
                servingbackends.run_stage(
                    unprobed,
                    plan_cells=[],
                    fidelity_input=None,
                    cache_prompts=None,
                    seed=0,
                    decode=FakeTokenizer().decode,
                    capabilities=servingbackends.StackCapabilities(
                        raw_token_ids=True,
                        stop_token_ids=False,
                        cached_token_reporting=False,
                        token_id_echo=True,
                        prompt_echo=True,
                        min_tokens=False,
                        include_stop_str_in_output=False,
                    ),
                )

    # -- stage budget paths and loader line shapes --

    def test_stage_budget_paths(self) -> None:
        """Spec: A8(iv)/section 8.5 -- when the sink reports the
        byte budget spent mid-plan, the remaining planned cells are
        recorded as budget omissions; when it trips inside the
        cache sequence, the sequence reports the budget detail and
        no arm result is invented."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            cells = [
                {
                    "candidate": "t",
                    "fixture": f"t{i}",
                    "request": StackRequest(prompt_ids=self.sentinel(), seed=7),
                    "cache_position": "standalone",
                }
                for i in range(2)
            ]
            stage = servingbackends.run_stage(
                backend,
                plan_cells=cells,
                fidelity_input=None,
                cache_prompts=None,
                seed=0,
                decode=backend.decode_ids,
                capabilities=backend.capabilities,
                on_record=lambda _record: False,
            )
            self.assertTrue(stage["audit"]["budget_spent"])
            self.assertEqual(len(stage["records"]), 2)
            self.assertEqual(
                stage["records"][1]["cell_detail"],
                servingbackends.EXECUTION_BUDGET_DETAIL,
            )
            calls = {"n": 0}

            def fail_on_cache(_record: dict[str, Any]) -> bool:
                calls["n"] += 1
                return False

            stage = servingbackends.run_stage(
                backend,
                plan_cells=[],
                fidelity_input=None,
                cache_prompts={
                    "cold_ids": self.sentinel(),
                    "shared_prefix_ids": self.sentinel() + (9,),
                    "divergent_control_ids": (8, 12),
                },
                seed=0,
                decode=backend.decode_ids,
                capabilities=backend.capabilities,
                on_record=fail_on_cache,
            )
            self.assertTrue(stage["audit"]["budget_spent"])
            self.assertEqual(
                stage["cache_sequence"]["detail"],
                servingbackends.EXECUTION_BUDGET_DETAIL,
            )
            self.assertFalse(stage["cache_sequence"]["available"])

    def test_loader_line_shapes(self) -> None:
        """Spec: A8(iv) -- a single-line artifact that is not JSON,
        a JSONL line that is not JSON, and a JSONL line that is not
        an object are all invalid with the fixed code; blank lines
        are skipped and non-object plan descriptors are ignored,
        with the planned cell read back as not executed."""
        broken = Path(self._tmpdir.name) / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_records_artifact(broken)
        bad_line = Path(self._tmpdir.name) / "bad.jsonl"
        bad_line.write_text('{"header": {}}\n{oops\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_records_artifact(bad_line)
        non_dict = Path(self._tmpdir.name) / "nondict.jsonl"
        non_dict.write_text('{"header": {}}\n[1, 2]\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_records_artifact(non_dict)
        sparse = Path(self._tmpdir.name) / "sparse.jsonl"
        sparse.write_text(
            '{"header": {"plan": [{"candidate": "t", "fixture": "t1",'
            ' "cache_position": "standalone"}, "junk"]}}\n\n{"footer": {"complete": true}}\n',
            encoding="utf-8",
        )
        loaded = load_records_artifact(sparse)
        self.assertTrue(loaded["complete"])
        self.assertEqual(len(loaded["missing"]), 1)
        self.assertEqual(loaded["missing"][0]["cell_detail"], "cell_not_executed")

    # -- fidelity entry validation (diffbackends) --

    def test_fidelity_entry_validation(self) -> None:
        """Spec: section 9.3 entry discipline -- the observation
        container, EOS, min_tokens, backend, and decode are
        validated before any execution; truncation caps the
        candidate list; unusable observations are skipped, a set
        cancel raises, and a diverging fixture records its finding
        in the per-fixture row."""
        valid = {
            "fixture": "fx",
            "input_ids": list(self.sentinel()),
            "output_ids": [6, EOS],
        }
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            section = diffbackends.fidelity_section_data(
                observations=(valid,),
                stop_texts=("STOP",),
                eos_id=EOS,
                min_tokens=0,
                backend=backend,
                decode=backend.decode_ids,
            )
            self.assertIsNotNone(section)
            with self.assertRaises(ValueError):
                diffbackends.fidelity_section_data(
                    observations="nope",  # type: ignore[arg-type]
                    stop_texts=("STOP",),
                    eos_id=EOS,
                    min_tokens=0,
                    backend=backend,
                    decode=backend.decode_ids,
                )
            self.assertIsNone(
                diffbackends.fidelity_section_data(
                    observations=[{}] * 513,
                    stop_texts=("STOP",),
                    eos_id=EOS,
                    min_tokens=0,
                    backend=backend,
                    decode=backend.decode_ids,
                )
            )
            with self.assertRaises(ValueError):
                diffbackends.fidelity_section_data(
                    observations=[valid],
                    stop_texts=("STOP",),
                    eos_id=-1,
                    min_tokens=0,
                    backend=backend,
                    decode=backend.decode_ids,
                )
            with self.assertRaises(ValueError):
                diffbackends.fidelity_section_data(
                    observations=[valid],
                    stop_texts=("STOP",),
                    eos_id=EOS,
                    min_tokens=True,  # type: ignore[arg-type]
                    backend=backend,
                    decode=backend.decode_ids,
                )
            with self.assertRaises(ValueError):
                diffbackends.fidelity_section_data(
                    observations=[valid],
                    stop_texts=("STOP",),
                    eos_id=EOS,
                    min_tokens=0,
                    backend=object(),
                    decode=backend.decode_ids,
                )
            with self.assertRaises(ValueError):
                diffbackends.fidelity_section_data(
                    observations=[valid],
                    stop_texts=("STOP",),
                    eos_id=EOS,
                    min_tokens=0,
                    backend=backend,
                    decode=None,
                )
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(servingbackends.ExecutionCancelled):
                diffbackends.fidelity_section_data(
                    observations=[valid],
                    stop_texts=("STOP",),
                    eos_id=EOS,
                    min_tokens=0,
                    backend=backend,
                    decode=backend.decode_ids,
                    cancel=cancel,
                )
            section = diffbackends.fidelity_section_data(
                observations=["junk", valid],
                stop_texts=("STOP",),
                eos_id=EOS,
                min_tokens=0,
                backend=backend,
                decode=backend.decode_ids,
            )
            self.assertEqual(len(section["per_fixture"]), 1)
            server.state.mode = "spurious"
            section = diffbackends.fidelity_section_data(
                observations=[valid],
                stop_texts=("STOP",),
                eos_id=EOS,
                min_tokens=0,
                backend=backend,
                decode=backend.decode_ids,
            )
            self.assertEqual(section["per_fixture"][0]["verdict"], "diverge")
            self.assertEqual(
                section["per_fixture"][0]["divergences"][0]["class"],
                "SPURIOUS_STOP",
            )

    # -- mirror deep branches --

    def _diverse_section(self, server: _FakeServer) -> dict[str, Any]:
        backend = self.probed_backend(server)
        records = []
        plan = [
            ("word", {"stop_texts": ("STOP",), "eos_id": EOS}),
            ("word_kept", {"stop_texts": ("STOP",), "eos_id": EOS}),
            ("eos", {"eos_id": EOS}),
            ("limit", {"stop_token_ids": (5,), "eos_id": EOS}),
            ("spurious", {"stop_texts": ("STOP",), "eos_id": EOS}),
            ("http500", {"eos_id": EOS}),
        ]
        for index, (mode, extra) in enumerate(plan):
            server.state.mode = mode
            records.append(
                servingbackends.execute_cell(
                    backend,
                    request=StackRequest(
                        prompt_ids=self.sentinel(),
                        seed=7,
                        max_new_tokens=8,
                        **extra,
                    ),
                    decode=backend.decode_ids,
                    candidate="t",
                    fixture=f"f{index}",
                )
            )
        return experiments.serving_backends_analysis(
            {
                "available": True,
                "detail": "",
                "identity": backend.identity.as_dict(),
                "capabilities": backend.capabilities.as_dict(),
                "records": records,
                "cache_sequence": None,
                "fidelity": None,
                "audit": {
                    "n_records": len(records),
                    "n_planned_cells": len(records),
                    "n_executions": len(records),
                    "budget_spent": False,
                },
            }
        )

    def _fidelity_section(self, server: _FakeServer) -> dict[str, Any]:
        backend = self.probed_backend(server)
        server.state.mode = "eos"
        fidelity = diffbackends.fidelity_section_data(
            observations=[
                {
                    "fixture": "fx",
                    "input_ids": list(self.sentinel()),
                    "output_ids": [6, EOS],
                }
            ],
            stop_texts=("STOP",),
            eos_id=EOS,
            min_tokens=0,
            backend=backend,
            decode=backend.decode_ids,
        )
        return experiments.serving_backends_analysis(
            {
                "available": True,
                "detail": "",
                "identity": backend.identity.as_dict(),
                "capabilities": backend.capabilities.as_dict(),
                "records": fidelity["records"],
                "cache_sequence": None,
                "fidelity": fidelity,
                "audit": {
                    "n_records": len(fidelity["records"]),
                    "n_planned_cells": 1,
                    "n_executions": 2,
                    "budget_spent": False,
                },
            }
        )

    def _cache_section(self, server: _FakeServer, **prompts: Any) -> dict[str, Any]:
        backend = self.probed_backend(server)
        stage = servingbackends.run_stage(
            backend,
            plan_cells=[],
            fidelity_input=None,
            cache_prompts=prompts
            or {
                "cold_ids": self.sentinel(),
                "shared_prefix_ids": self.sentinel() + (9,),
                "divergent_control_ids": (8, 12),
            },
            seed=0,
            decode=backend.decode_ids,
            capabilities=backend.capabilities,
        )
        return experiments.serving_backends_analysis(stage)

    def test_mirror_deep_reject_branches(self) -> None:
        """Spec: section 7.3.1 recompute standard -- a genuine
        section with every verdict flavour validates (kept
        dispositions and diverge/unobservable recomputes included);
        each deep corruption below passes the mirror's earlier
        gates and is rejected by the gate it names."""
        with _FakeServer(self.gguf) as server:
            section = self._diverse_section(server)
        self.assertTrue(diffreports._valid_serving_backends(section))
        counts = section["fidelity"]["counts"]
        self.assertEqual(counts["diverge"], 2)  # kept-vs-stripped + spurious
        self.assertEqual(counts["unobservable"], 1)

        def rejected(edit: Any) -> dict[str, Any]:
            mutated = copy.deepcopy(section)
            edit(mutated)
            self.assertFalse(diffreports._valid_serving_backends(mutated))
            return mutated

        rejected(lambda s: s["identity"]["config"].pop("tier_a"))
        rejected(lambda s: s["identity"]["config"]["posture"].update(api_key_required=False))
        rejected(lambda s: s["identity"]["config"]["cache_probe"].update(cold_cache_n=-1))
        rejected(lambda s: s["identity"]["config"]["tier_a"].update(top_k=5))
        rejected(lambda s: s["capabilities"].update(min_tokens="no"))
        rejected(lambda s: s.update(records={}))
        rejected(lambda s: s["records"][0].update(finish_class="bogus"))
        rejected(lambda s: s["stop_matrix"].update(available=False))
        rejected(lambda s: s["fidelity"].update(verdict="fidelity_agree"))
        rejected(lambda s: s["fidelity"].update(detail="not_attempted"))
        rejected(lambda s: s["fidelity"].update(per_fixture={}))
        rejected(lambda s: s["fidelity"]["per_fixture"].append({"x": 1}))
        rejected(lambda s: s["audit"].update(n_records=99))
        # Record-level cross-checks.
        rejected(lambda s: s["records"][3].update(returned_ids=[6]))
        rejected(lambda s: s["records"][3].update(cell_detail=""))
        rejected(lambda s: s["records"][0]["mirror"].update(divergence_class="SPURIOUS_STOP"))
        rejected(lambda s: s["records"][0]["mirror"].update(observed_disposition="kept"))
        rejected(lambda s: s["records"][2]["stop_winner"].update(id=999))
        rejected(
            lambda s: s["records"][1].update(
                returned_ids=[6] * 12, mirror=s["records"][1]["mirror"]
            )
        )
        # Drift on a shared record is reported, not hidden. (A
        # single-record section: request_digest binds the request
        # config, so same-config cells share a digest by design.)
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.state.mode = "word"
            solo = experiments.serving_backends_analysis(
                {
                    "available": True,
                    "detail": "",
                    "identity": backend.identity.as_dict(),
                    "capabilities": backend.capabilities.as_dict(),
                    "records": [
                        servingbackends.execute_cell(
                            backend,
                            request=StackRequest(
                                prompt_ids=self.sentinel(),
                                seed=7,
                                max_new_tokens=8,
                                stop_texts=("STOP",),
                                eos_id=EOS,
                            ),
                            decode=backend.decode_ids,
                            candidate="t",
                            fixture="solo",
                        )
                    ],
                    "cache_sequence": None,
                    "fidelity": None,
                    "audit": {
                        "n_records": 1,
                        "n_planned_cells": 1,
                        "n_executions": 1,
                        "budget_spent": False,
                    },
                }
            )
        drifted = copy.deepcopy(solo)
        drifted["records"][0]["cache_n_reported"] = 777
        self.assertTrue(diffreports._valid_serving_backends(drifted))
        diff = diffreports.diff_serving_backends(solo, drifted)
        self.assertTrue(diff["changed"])
        self.assertEqual(
            [entry["metric"] for entry in diff["record_drift"]],
            ["cache_n_reported"],
        )

    def test_mirror_fidelity_and_cache_branches(self) -> None:
        """Spec: sections 7.3.1/9.1.3 -- per-fixture finding and
        stream shapes are gated field by field; cache positions
        recompute from the records, the contamination detail must
        match the measured arms in both directions, and
        capability-absent cache records are skipped by the
        positions recompute without invalidating the section."""
        with _FakeServer(self.gguf) as server:
            section = self._fidelity_section(server)
        self.assertTrue(diffreports._valid_serving_backends(section))

        def rejected(edit: Any) -> None:
            mutated = copy.deepcopy(section)
            edit(mutated)
            self.assertFalse(diffreports._valid_serving_backends(mutated))

        def with_finding(edit: Any) -> Any:
            def apply(s: dict[str, Any]) -> None:
                s["fidelity"]["per_fixture"][0]["divergences"] = [
                    {"class": None, "detail": "d", "severity": "low"}
                ]
                edit(s)

            return apply

        def finding0(s: dict[str, Any]) -> dict[str, Any]:
            return s["fidelity"]["per_fixture"][0]["divergences"][0]

        rejected(with_finding(lambda s: finding0(s).update(extra=1)))
        rejected(with_finding(lambda s: finding0(s).update(**{"class": "NOPE"})))
        rejected(with_finding(lambda s: finding0(s).update(severity="critical")))
        rejected(lambda s: s["records"][0]["mirror"]["hf_stream"].update(match="yes"))
        rejected(lambda s: s["records"][0]["mirror"]["hf_stream"].update(first_divergence=-1))
        rejected(lambda s: s["records"][0]["mirror"]["hf_stream"].update(shared_prefix=-1))
        rejected(
            lambda s: s["records"][0]["mirror"]["hf_stream"].update(
                **{"class": "GENERATION_STREAM_DIVERGED"}
            )
        )
        rejected(
            lambda s: s["records"][0]["mirror"]["hf_stream"].update(match=False, **{"class": None})
        )
        with _FakeServer(self.gguf) as server:
            cache_section = self._cache_section(server)
        self.assertTrue(diffreports._valid_serving_backends(cache_section))

        def cache_rejected(edit: Any) -> None:
            mutated = copy.deepcopy(cache_section)
            edit(mutated)
            self.assertFalse(diffreports._valid_serving_backends(mutated))

        cache_rejected(lambda s: s["cache"]["positions"]["cold"].update(cache_n=999))
        cache_rejected(lambda s: s["cache"].update(available=False))
        cache_rejected(lambda s: s["cache"].update(detail="cache_protocol_contaminated"))
        with _FakeServer(self.gguf) as server:
            server.state.cache_n_override = 5
            hot = self._cache_section(server)
        self.assertEqual(hot["cache"]["detail"], "cache_protocol_contaminated")
        self.assertTrue(diffreports._valid_serving_backends(hot))
        hot_bad = copy.deepcopy(hot)
        hot_bad["cache"]["detail"] = ""
        self.assertFalse(diffreports._valid_serving_backends(hot_bad))
        with _FakeServer(self.gguf) as server:
            server.state.erase_enabled = False
            absent = self._cache_section(server)
        self.assertTrue(diffreports._valid_serving_backends(absent))
        self.assertEqual(absent["cache"]["positions"], {})

    # -- reporting section shapes --

    def test_reporting_section_shapes(self) -> None:
        """Spec: section 7.3 rendering -- a full stage section
        renders its cache position rows; missing identity or
        capabilities render nothing; junk stop-matrix rows are
        skipped row by row, never crashing the report."""
        with _FakeServer(self.gguf) as server:
            section = self._cache_section(server)
        rendered = reporting.render_html({"serving_backends": section})
        self.assertIn("warm_repeat", rendered)
        self.assertIn("Measured cache companion", rendered)
        self.assertEqual(
            reporting._serving_backends_section(
                {"serving_backends": {"available": True, "identity": "x"}}
            ),
            "",
        )
        self.assertEqual(
            reporting._serving_backends_section(
                {
                    "serving_backends": {
                        "available": True,
                        "identity": {},
                        "capabilities": "x",
                    }
                }
            ),
            "",
        )
        rendered = reporting._serving_backends_section(
            {
                "serving_backends": {
                    "available": True,
                    "identity": {},
                    "capabilities": {},
                    "stop_matrix": {
                        "rows": [
                            "junk",
                            {"cells": {}},
                            {"cells": {"llamacpp": None}},
                            {"cells": {"llamacpp": {"fired": True}}},
                        ]
                    },
                }
            }
        )
        self.assertIn("fired", rendered)

    # -- handler recovery paths --

    def test_handler_recovery_paths(self) -> None:
        """Spec: handler exit discipline -- an unwritable
        unavailable-report path exits 1; a malformed cache id list
        is a parser error; a stage failure after the sink opens is
        an interrupted run (exit 1, sink closed); a blocked final
        rename exits 1; the results extractor reads both the search
        and held-out observation lists."""
        import argparse

        parser = argparse.ArgumentParser(prog="exec-backends")
        with mock.patch(
            "model_protocol_lab.doctor._build_primary_tokenizer",
            return_value=FakeTokenizer(),
        ):
            out = Path(self._tmpdir.name) / "nope" / "o.json"
            code = servingbackends.run_exec_backends(self._args(None, out, timeout=0.5), parser)
            self.assertEqual(code, 1)
            with _FakeServer(self.gguf) as server:
                out2 = Path(self._tmpdir.name) / "c.json"
                with self.assertRaises(SystemExit):
                    servingbackends.run_exec_backends(
                        self._args(
                            server,
                            out2,
                            cache_cold_ids="1,2",
                            cache_shared_ids="1,2",
                            cache_divergent_ids="1,x",
                        ),
                        parser,
                    )
                out3 = Path(self._tmpdir.name) / "i.json"
                with mock.patch.object(
                    servingbackends,
                    "run_cache_sequence",
                    side_effect=ValueError("stack_identity_unavailable"),
                ):
                    code = servingbackends.run_exec_backends(
                        self._args(
                            server,
                            out3,
                            prompt_ids=None,
                            cache_cold_ids="1,2,3,4,5",
                            cache_shared_ids="1,2,3,4,5,9",
                            cache_divergent_ids="8,12",
                        ),
                        parser,
                    )
                self.assertEqual(code, 1)
                out4 = Path(self._tmpdir.name) / "final.json"
                (Path(self._tmpdir.name) / "final.json.tmp").mkdir()
                code = servingbackends.run_exec_backends(self._args(server, out4), parser)
                self.assertEqual(code, 1)
        observations = servingbackends._extract_observations(
            {
                "search": [{"observations": [{"a": 1}, "junk"]}],
                "held_out": {"observations": [{"b": 2}]},
            }
        )
        self.assertEqual(observations, [{"a": 1}, {"b": 2}])
        self.assertEqual(servingbackends._extract_observations({"x": 1}), [])
        self.assertEqual(servingbackends._extract_observations("nope"), [])

    # -- mop-up: the last reachable branches --

    def test_mopup_branches(self) -> None:
        """Spec: remaining fail-closed branches -- a non-bool
        request flag; a sentinel encoding shorter than two tokens;
        a redirect refused as a different endpoint; identity
        re-verification against a failing /props; a detokenize
        HTTP failure recorded as a missing digest; a stop winner
        with no stack text digest is disposition-unobservable; a
        headerless multi-line artifact is invalid; a sink budget
        trip during the fidelity phase marks the stage spent."""
        with self.assertRaises(ValueError):
            StackRequest(
                prompt_ids=(1, 2),
                seed=1,
                include_stop_str_in_output="yes",  # type: ignore[arg-type]
            )

        class ShortEncode(FakeTokenizer):
            def encode(self, text: str, **kwargs: Any) -> Any:
                return (1,)

        with _FakeServer(self.gguf) as server:
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server, tokenizer=ShortEncode()).probe()
            self.assertEqual(str(caught.exception), "token_vocab_mismatch")
            server.state.redirect_props = True
            with self.assertRaises(ValueError) as caught:
                self.make_backend(server).probe()
            self.assertEqual(str(caught.exception), "stack_endpoint_refused")
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.state.props_status = 500
            self.assertFalse(backend.verify_identity())
            server.state.props_status = 200
            server.state.detok_status = 500
            observation = backend.execute(
                StackRequest(prompt_ids=self.sentinel(), seed=5, max_new_tokens=4)
            )
            self.assertIsNone(observation.server_detok_digest)
        disposition = servingbackends._observed_disposition(
            observation=servingbackends.StackObservation(
                returned_ids=(6, 7),
                finish_class="word",
                stop_winner={"kind": "stop_text", "index": 0, "digest": "d"},
                cache_n_reported=None,
                tokens_cached_reported=None,
                tokens_evaluated_reported=None,
                returned_text_digest=None,
                server_detok_digest=None,
                config_echo_digest=None,
            ),
            prediction={"local_text_digest": "x", "stripped_text_digest": None},
        )
        self.assertEqual(disposition, "unobservable")
        headerless = Path(self._tmpdir.name) / "headerless2.jsonl"
        headerless.write_text('{"record": {}}\n{"record": {}}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_records_artifact(headerless)
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.state.mode = "eos"
            stage = servingbackends.run_stage(
                backend,
                plan_cells=[],
                fidelity_input={
                    "observations": [
                        {
                            "fixture": "fx",
                            "input_ids": list(self.sentinel()),
                            "output_ids": [6, EOS],
                        }
                    ],
                    "stop_texts": ("STOP",),
                    "eos_id": EOS,
                    "min_tokens": 0,
                },
                cache_prompts=None,
                seed=0,
                decode=backend.decode_ids,
                capabilities=backend.capabilities,
                on_record=lambda _record: False,
            )
            self.assertTrue(stage["audit"]["budget_spent"])
            self.assertEqual(stage["detail"], servingbackends.EXECUTION_BUDGET_DETAIL)
