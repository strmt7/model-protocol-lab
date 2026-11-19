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
import importlib.metadata
import json
import os
import random
import struct
import sys
import tempfile
import threading
import types
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

from model_protocol_lab import (
    backends,
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
            "stack": "llamacpp",
            "max_model_len": None,
            "kv_cache_mib": None,
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
        # --timeout defaults to None at parse time; the handler
        # applies the 120-second default on the llamacpp arm (the
        # None default is what makes an explicit --timeout on the
        # vllm arm detectable and rejectable).
        self.assertIsNone(args.timeout)
        self.assertEqual(args.stack, "llamacpp")
        self.assertIsNone(args.kv_cache_mib)
        self.assertIsNone(args.max_model_len)


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


# ---------------------------------------------------------------------------
# W6 Push 1: the vLLM Tier-A tuple, the two-model predictor, and the
# observation plumbing the vLLM-shaped record fields ride on. The
# llama expectations above stay byte-identical (B1); everything here
# is additive.
# ---------------------------------------------------------------------------


class VllmTierATupleTests(unittest.TestCase):
    """Plan C13: the vLLM tuple is the post-normalization canonical
    surface (top_k 0, repetition_penalty), pinned whole."""

    def test_exact_tuple(self) -> None:
        self.assertEqual(
            servingbackends.tier_a_sampling_params_vllm(7),
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "seed": 7,
                "n": 1,
                "detokenize": True,
                "skip_special_tokens": True,
                "logprobs": None,
            },
        )

    def test_seed_validation(self) -> None:
        for bad in (-1, True, "3"):
            with self.assertRaises(ValueError) as caught:
                servingbackends.tier_a_sampling_params_vllm(bad)  # type: ignore[arg-type]
            self.assertEqual(str(caught.exception), "stack_request_invalid")

    def test_sampler_dispatch(self) -> None:
        self.assertEqual(
            servingbackends._sampler_for(servingbackends.VLLM_STACK, 3),
            servingbackends.tier_a_sampling_params_vllm(3),
        )
        self.assertEqual(
            servingbackends._sampler_for(servingbackends.LLAMACPP_STACK, 3),
            servingbackends.tier_a_sampler_params(3),
        )
        # An unknown stack falls back to the llama tuple (the mirror
        # owns stack admission; the producer never invents a tuple).
        self.assertEqual(
            servingbackends._sampler_for("other", 3),
            servingbackends.tier_a_sampler_params(3),
        )

    def test_request_config_default_is_the_llama_tuple(self) -> None:
        request = servingbackends.StackRequest(prompt_ids=(10,), max_new_tokens=4, seed=5)
        self.assertEqual(
            request.request_config()["sampler"],
            servingbackends.tier_a_sampler_params(5),
        )
        self.assertEqual(
            request.request_config(servingbackends.tier_a_sampling_params_vllm(5))["sampler"],
            servingbackends.tier_a_sampling_params_vllm(5),
        )


class PredictorVllmModelTests(unittest.TestCase):
    """Plan Step 2 + PF-2/PF-3/PF-4: the vllm stop model is the
    executed detokenizer port for texts and sampler masking for
    token ids and the EOS."""

    def test_gate_crossing_span_fires_at_completion_step(self) -> None:
        # "abc" begins at step 1 (token 13 = "ab") and completes at
        # step 2 (token 12 = "c"); min_tokens=1: the completion step
        # is beyond the gate, so the port fires (W6 PF-2 shape).
        predicted = servingbackends.predict_stop_outcome(
            decode=fake_decode,
            returned_ids=(13, 12),
            stop_texts=("abc",),
            eos_id=None,
            min_tokens=1,
            stop_model="vllm",
        )
        self.assertEqual(predicted["stop_steps"], {"0": 2})
        self.assertEqual(predicted["predicted_firing_step"], 2)
        self.assertEqual(predicted["predicted_winner"], {"kind": "stop_text", "index": 0})

    def test_gated_completion_is_never_refound(self) -> None:
        # The stop completes exactly at the gated step 2 and the
        # tail window never sees it again (R3 S52b; PF-2's other
        # arm): the vllm model predicts no fire, the generic model
        # fires at 2 — the pair discriminates the two models.
        ids = (13, 12, 10)
        vllm = servingbackends.predict_stop_outcome(
            decode=fake_decode,
            returned_ids=ids,
            stop_texts=("abc",),
            eos_id=None,
            min_tokens=2,
            stop_model="vllm",
        )
        self.assertEqual(vllm["stop_steps"], {})
        self.assertIsNone(vllm["predicted_firing_step"])
        self.assertIsNone(vllm["predicted_winner"])
        generic = servingbackends.predict_stop_outcome(
            decode=fake_decode,
            returned_ids=ids,
            stop_texts=("abc",),
            eos_id=None,
            min_tokens=2,
        )
        self.assertEqual(generic["predicted_firing_step"], 2)

    def test_stop_token_ids_gated_by_min_tokens(self) -> None:
        # PF-3 shape: token 12 occurs at steps 3 and 6; the gated
        # occurrence is masked out of the stream, so only the later
        # one can win.
        ids = (10, 11, 12, 10, 11, 12)
        floored = servingbackends.predict_stop_outcome(
            decode=fake_decode,
            returned_ids=ids,
            stop_texts=(),
            eos_id=None,
            min_tokens=4,
            stop_token_ids=(12,),
            stop_model="vllm",
        )
        self.assertEqual(floored["predicted_firing_step"], 6)
        self.assertEqual(floored["predicted_winner"], {"kind": "stop_token", "id": 12})
        unfloored = servingbackends.predict_stop_outcome(
            decode=fake_decode,
            returned_ids=ids,
            stop_texts=(),
            eos_id=None,
            min_tokens=0,
            stop_token_ids=(12,),
            stop_model="vllm",
        )
        self.assertEqual(unfloored["predicted_firing_step"], 3)

    def test_eos_gated_by_min_tokens(self) -> None:
        # PF-4 shape: the EOS at step 2 is masked when the floor
        # covers it; ignore_eos removes the candidate entirely.
        ids = (10, EOS, 11)
        floored = servingbackends.predict_stop_outcome(
            decode=fake_decode,
            returned_ids=ids,
            stop_texts=(),
            eos_id=EOS,
            min_tokens=4,
            stop_model="vllm",
        )
        self.assertIsNone(floored["predicted_winner"])
        unfloored = servingbackends.predict_stop_outcome(
            decode=fake_decode,
            returned_ids=ids,
            stop_texts=(),
            eos_id=EOS,
            min_tokens=0,
            stop_model="vllm",
        )
        self.assertEqual(unfloored["predicted_firing_step"], 2)
        self.assertEqual(unfloored["predicted_winner"], {"kind": "eos", "id": EOS})
        ignored = servingbackends.predict_stop_outcome(
            decode=fake_decode,
            returned_ids=ids,
            stop_texts=(),
            eos_id=EOS,
            ignore_eos=True,
            min_tokens=0,
            stop_model="vllm",
        )
        self.assertIsNone(ignored["predicted_winner"])

    def test_text_and_token_tie_at_one_step(self) -> None:
        # "abc" completes at step 2 and token 12 is a stop token at
        # the same step: the step is certain, the winner is a tie,
        # and the text kind takes documented precedence.
        predicted = servingbackends.predict_stop_outcome(
            decode=fake_decode,
            returned_ids=(13, 12),
            stop_texts=("abc",),
            eos_id=None,
            min_tokens=0,
            stop_token_ids=(12,),
            stop_model="vllm",
        )
        self.assertEqual(predicted["predicted_firing_step"], 2)
        self.assertTrue(predicted["precedence_tie"])
        self.assertEqual(predicted["predicted_winner"], {"kind": "stop_text", "index": 0})

    def test_differential_against_the_executed_port(self) -> None:
        # The predictor's text steps must equal the port's firing
        # step on every stream — the port is the executed model of
        # the engine, so agreement here is the Step-2 acceptance.
        rng = random.Random(2436)  # noqa: S311 -- deterministic test seed
        tokenizer = FakeTokenizer()
        pool = [6, 7, 10, 11, 12, 13, 14]
        stop_pool = ["abc", "bc", "ab", "a", "STOP", "abcabc"]
        for trial in range(300):
            ids = tuple(rng.choice(pool) for _ in range(rng.randrange(0, 9)))
            text = rng.choice(stop_pool)
            floor = rng.choice([0, 1, 2, 4])
            outcome = backends.emulate_vllm_holdback(tokenizer, ids, text, min_tokens=floor)
            expected = len(outcome.truncated_ids) if outcome.available and outcome.fired else None
            predicted = servingbackends.predict_stop_outcome(
                decode=fake_decode,
                returned_ids=ids,
                stop_texts=(text,),
                eos_id=None,
                min_tokens=floor,
                stop_model="vllm",
            )
            self.assertEqual(
                predicted["stop_steps"],
                {"0": expected} if expected is not None else {},
                f"trial {trial}: ids={ids} stop={text!r} floor={floor}",
            )
            self.assertEqual(predicted["predicted_firing_step"], expected)

    def test_generic_model_ignores_the_vllm_parameters(self) -> None:
        # The generic scan has no min_tokens and no token-stop
        # candidates: a step-1 completion fires at 1 under any floor.
        predicted = servingbackends.predict_stop_outcome(
            decode=fake_decode,
            returned_ids=(12,),
            stop_texts=("c",),
            eos_id=None,
            min_tokens=9,
            stop_token_ids=(12,),
        )
        self.assertEqual(predicted["predicted_firing_step"], 1)
        self.assertEqual(predicted["predicted_winner"], {"kind": "stop_text", "index": 0})

    def test_invalid_model_and_parameters(self) -> None:
        with self.assertRaises(ValueError) as caught:
            servingbackends.predict_stop_outcome(
                decode=fake_decode,
                returned_ids=(10,),
                stop_texts=(),
                eos_id=None,
                stop_model="polled",  # type: ignore[arg-type]
            )
        self.assertEqual(str(caught.exception), "invalid_stop_model")
        with self.assertRaises(ValueError) as caught:
            servingbackends.predict_stop_outcome(
                decode=fake_decode,
                returned_ids=(10,),
                stop_texts=(),
                eos_id=None,
                min_tokens=-1,
                stop_model="vllm",
            )
        self.assertEqual(str(caught.exception), "stack_request_invalid")
        with self.assertRaises(ValueError) as caught:
            servingbackends.predict_stop_outcome(
                decode=fake_decode,
                returned_ids=(10,),
                stop_texts=(),
                eos_id=None,
                stop_token_ids=(-1,),
                stop_model="vllm",
            )
        self.assertEqual(str(caught.exception), "stack_request_invalid")


def _identity_for(stack: str) -> servingbackends.StackIdentity:
    return servingbackends.StackIdentity(
        stack=stack,
        stack_version="0.33.0" if stack == servingbackends.VLLM_STACK else "test",
        checkpoint_digest="0" * 64,
        checkpoint_bytes=1,
        tokenizer_fingerprint="sha256:" + "0" * 64,
        hardware_class="cpu",
        declared_eos_ids=(EOS,),
        config={},
        config_fingerprint="0" * 64,
    )


class _StubBackend:
    """Minimal ExecutionBackend double for execute_cell wiring."""

    def __init__(self, identity: servingbackends.StackIdentity, observation: Any) -> None:
        self.identity = identity
        self._observation = observation

    def execute(self, request: servingbackends.StackRequest) -> Any:
        return self._observation


class ExecuteCellStackDerivationTests(unittest.TestCase):
    """execute_cell derives the sampler tuple and the stop model
    from the producing identity's stack."""

    def _run(self, stack: str, min_tokens: int) -> dict[str, Any]:
        observation = servingbackends.StackObservation(
            returned_ids=(13, 12),
            finish_class="stop",
            stop_winner={
                "kind": "stop_text",
                "index": 0,
                "digest": servingbackends._text_digest("abc"),
            },
            cache_n_reported=None,
            tokens_cached_reported=None,
            tokens_evaluated_reported=None,
            # The stack trims the stop string: its returned text is
            # the empty prefix before "abc".
            returned_text_digest=servingbackends._text_digest(""),
            server_detok_digest=None,
            config_echo_digest=None,
        )
        backend = _StubBackend(_identity_for(stack), observation)
        request = servingbackends.StackRequest(
            prompt_ids=(10,),
            max_new_tokens=8,
            stop_texts=("abc",),
            min_tokens=min_tokens,
            seed=11,
        )
        return servingbackends.execute_cell(
            backend,  # type: ignore[arg-type]
            request=request,
            decode=fake_decode,
            candidate="c",
            fixture="f",
        )

    def test_vllm_identity_records_vllm_sampler_and_model(self) -> None:
        record = self._run(servingbackends.VLLM_STACK, min_tokens=2)
        self.assertEqual(
            record["request_config"]["sampler"],
            servingbackends.tier_a_sampling_params_vllm(11),
        )
        # Under the vllm model the gated completion never refires:
        # the mirror prediction is empty while the stack claims a
        # stop — a spurious-stop divergence, by design.
        self.assertIsNone(record["mirror"]["predicted_winner"])
        self.assertEqual(record["mirror"]["fidelity_verdict"], "diverge")
        self.assertEqual(record["mirror"]["divergence_class"], "SPURIOUS_STOP")

    def test_llama_identity_records_llama_sampler_and_generic_model(self) -> None:
        record = self._run(servingbackends.LLAMACPP_STACK, min_tokens=2)
        self.assertEqual(
            record["request_config"]["sampler"],
            servingbackends.tier_a_sampler_params(11),
        )
        # The generic scan ignores the floor: the completion at
        # step 2 is the prediction, and the record agrees.
        self.assertEqual(record["mirror"]["predicted_firing_step"], 2)
        self.assertEqual(record["mirror"]["fidelity_verdict"], "agree")

    def test_absent_record_uses_the_identity_stack_sampler(self) -> None:
        request = servingbackends.StackRequest(prompt_ids=(10,), max_new_tokens=4, seed=3)
        record = servingbackends._absent_record(
            request=request,
            identity=_identity_for(servingbackends.VLLM_STACK),
            candidate="c",
            fixture="f",
            cache_position="standalone",
            detail="capability_absent: test",
        )
        self.assertEqual(
            record["request_config"]["sampler"],
            servingbackends.tier_a_sampling_params_vllm(3),
        )


class ObservationPlumbingTests(unittest.TestCase):
    """Plan C4: the two vLLM-shaped record fields are observation
    fields now — None unless a backend reports them."""

    def test_defaults_are_none_and_serialize(self) -> None:
        observation = servingbackends.StackObservation(
            returned_ids=(10,),
            finish_class="length",
            stop_winner=None,
            cache_n_reported=None,
            tokens_cached_reported=None,
            tokens_evaluated_reported=None,
            returned_text_digest=None,
            server_detok_digest=None,
            config_echo_digest=None,
        )
        self.assertIsNone(observation.cache_creation_tokens_reported)
        self.assertIsNone(observation.returned_text_digest_no_special)
        as_dict = observation.as_dict()
        self.assertIn("cache_creation_tokens_reported", as_dict)
        self.assertIn("returned_text_digest_no_special", as_dict)
        self.assertIsNone(as_dict["cache_creation_tokens_reported"])
        self.assertIsNone(as_dict["returned_text_digest_no_special"])

    def test_record_threads_observation_values(self) -> None:
        observation = servingbackends.StackObservation(
            returned_ids=(10,),
            finish_class="length",
            stop_winner=None,
            cache_n_reported=None,
            tokens_cached_reported=64,
            tokens_evaluated_reported=None,
            returned_text_digest=servingbackends._text_digest("a"),
            server_detok_digest=None,
            config_echo_digest=None,
            cache_creation_tokens_reported=128,
            returned_text_digest_no_special="ab" * 32,
        )
        backend = _StubBackend(_identity_for(servingbackends.VLLM_STACK), observation)
        request = servingbackends.StackRequest(prompt_ids=(10,), max_new_tokens=4, seed=1)
        record = servingbackends.execute_cell(
            backend,  # type: ignore[arg-type]
            request=request,
            decode=fake_decode,
            candidate="c",
            fixture="f",
        )
        self.assertEqual(record["cache_creation_tokens_reported"], 128)
        self.assertEqual(record["returned_text_digest_no_special"], "ab" * 32)


class FidelityStopTokenTests(unittest.TestCase):
    """_fidelity_verdict on an observed stop_token winner: the id
    must be the token the stream holds at the firing step."""

    def _run(self, observed_id: int) -> dict[str, Any]:
        observation = servingbackends.StackObservation(
            returned_ids=(10, 12),
            finish_class="stop",
            stop_winner={"kind": "stop_token", "id": observed_id},
            cache_n_reported=None,
            tokens_cached_reported=None,
            tokens_evaluated_reported=None,
            returned_text_digest=servingbackends._text_digest("ac"),
            server_detok_digest=None,
            config_echo_digest=None,
        )
        backend = _StubBackend(_identity_for(servingbackends.VLLM_STACK), observation)
        request = servingbackends.StackRequest(
            prompt_ids=(10,),
            max_new_tokens=8,
            stop_token_ids=(12,),
            seed=11,
        )
        return servingbackends.execute_cell(
            backend,  # type: ignore[arg-type]
            request=request,
            decode=fake_decode,
            candidate="c",
            fixture="f",
        )

    def test_matching_stop_token_agrees(self) -> None:
        record = self._run(12)
        self.assertEqual(record["mirror"]["predicted_winner"], {"kind": "stop_token", "id": 12})
        self.assertEqual(record["mirror"]["fidelity_verdict"], "agree")

    def test_different_stop_token_diverges(self) -> None:
        record = self._run(11)
        self.assertEqual(record["mirror"]["fidelity_verdict"], "diverge")
        self.assertEqual(record["mirror"]["divergence_class"], "FIRING_STEP_DIFFERS")


# ---------------------------------------------------------------------------
# W6 Push 2: VllmOfflineBackend against a hostile fake `vllm` module
# (Steps 3+4). The fake is injected into sys.modules — the backend's
# lazy import is the seam — and importlib.metadata.version is patched
# for the three version reads. CI never installs vllm.
# ---------------------------------------------------------------------------


class _VllmFakeTokenizer:
    """Longest-match concatenative tokenizer over a fixed vocab,
    shared by the lab side and the engine side of the fake."""

    PIECES = {
        0: "<|endoftext|>",
        1: "The",
        2: " capital",
        3: " of",
        4: " France",
        5: " is",
        6: " Paris",
        7: "abc",
        8: "a",
        9: "b",
        10: "c",
        11: "ab",
        12: "bc",
        13: "STOP",
    }
    all_special_ids = [0]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        rest = text
        while rest:
            best: tuple[int, str] | None = None
            for token, piece in self.PIECES.items():
                if rest.startswith(piece) and (best is None or len(piece) > len(best[1])):
                    best = (token, piece)
            if best is None:
                raise ValueError("unencodable")
            ids.append(best[0])
            rest = rest[len(best[1]) :]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        parts = []
        for token in ids:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            parts.append(self.PIECES[token])
        return "".join(parts)


class _FakeSamplingParams:
    """Stores the constructor kwargs as attributes verbatim (the
    adapter reads them back for verification). ``drift`` rewrites
    temperature on read — a sampler that cannot be made canonical."""

    drift = False
    raise_always = False
    raise_on_paired = False

    def __init__(self, **kwargs: Any) -> None:
        if type(self).raise_always:
            raise RuntimeError("params construction refused")
        if type(self).raise_on_paired and kwargs.get("skip_special_tokens") is False:
            raise RuntimeError("paired params refused")
        self.__dict__.update(kwargs)
        if type(self).drift:
            self.__dict__["temperature"] = 0.5


class _FakeTokensPrompt(dict):
    def __init__(self, prompt_token_ids: Any = None, cache_salt: Any = None) -> None:
        ids = list(prompt_token_ids or [])
        super().__init__(prompt_token_ids=ids, cache_salt=cache_salt)
        self.prompt_token_ids = ids
        self.cache_salt = cache_salt


class _FakeCompletion:
    def __init__(
        self,
        token_ids: Any,
        text: Any,
        finish_reason: Any,
        stop_reason: Any = None,
    ) -> None:
        self.token_ids = token_ids
        self.text = text
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason


class _FakeRequestOutput:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: Any,
        completion: _FakeCompletion,
        *,
        finished: bool = True,
        num_cached_tokens: Any = 0,
        num_cache_creation_tokens: Any = 0,
    ) -> None:
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [completion]
        self.finished = finished
        self.num_cached_tokens = num_cached_tokens
        self.num_cache_creation_tokens = num_cache_creation_tokens


class _FakeEngineHandle:
    def __init__(self, llm: _FakeLLM) -> None:
        self._llm = llm
        self.vllm_config = llm.vllm_config
        self.output_processor = types.SimpleNamespace(tokenizer=llm.engine_tokenizer)

    def abort_request(self, rids: Any) -> None:
        self._llm.aborted.extend(rids)
        self._llm.abort_event.set()


class _FakeLLM:
    """A scripted vllm.LLM. Class attributes carry the script so the
    backend's constructor call (which passes only kwargs) builds the
    instance the test configured."""

    script: dict[str, Any] = {}
    instances: list[_FakeLLM] = []

    def __init__(self, **kwargs: Any) -> None:
        script = type(self).script
        if script.get("llm_raises"):
            raise RuntimeError("engine construction refused")
        type(self).instances.append(self)
        self.kwargs = kwargs
        self.engine_tokenizer = script.get("engine_tokenizer") or _VllmFakeTokenizer()
        if script.get("config_broken"):
            self.vllm_config = types.SimpleNamespace()
        else:
            self.vllm_config = types.SimpleNamespace(
                model_config=types.SimpleNamespace(
                    max_model_len=script.get("readback_max_model_len", kwargs["max_model_len"]),
                    dtype=script.get("readback_dtype", "torch.bfloat16"),
                ),
                cache_config=types.SimpleNamespace(
                    block_size=script.get("block_size", 128),
                    num_gpu_blocks=script.get("num_gpu_blocks", 56),
                    kv_cache_memory_bytes=script.get(
                        "readback_kv_bytes", kwargs["kv_cache_memory_bytes"]
                    ),
                ),
            )
        self.llm_engine = _FakeEngineHandle(self)
        if script.get("processor_broken"):
            self.llm_engine.output_processor = types.SimpleNamespace()
        self.aborted: list[str] = []
        self.abort_event = threading.Event()
        self.generate_calls: list[tuple[list[int], Any]] = []
        self.generate_handler = script.get("generate_handler")
        self.abort_mode = script.get("abort_mode", "terminal")
        self._rid_counter = 0

    def get_default_sampling_params(self) -> _FakeSamplingParams:
        if type(self).script.get("default_params_raises"):
            raise RuntimeError("no default params")
        return _FakeSamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            seed=None,
            detokenize=True,
            skip_special_tokens=True,
            logprobs=None,
            max_tokens=16,
            min_tokens=0,
            stop=[],
            stop_token_ids=[],
            ignore_eos=False,
            include_stop_str_in_output=False,
        )

    def get_tokenizer(self) -> Any:
        return self.engine_tokenizer

    def generate(
        self, prompts: Any, params: Any, use_tqdm: bool = True
    ) -> list[_FakeRequestOutput]:
        prompt = prompts[0]
        prompt_ids = list(prompt.prompt_token_ids)
        self.generate_calls.append((prompt_ids, params, prompt))
        if self.generate_handler is not None:
            return self.generate_handler(prompt_ids, params)
        completion = _FakeCompletion([8], "a", "length")
        return [_FakeRequestOutput("fake-req", prompt_ids, completion)]

    def enqueue(self, prompts: Any, params: Any) -> list[str]:
        if self.abort_mode == "enqueue_raises":
            raise RuntimeError("enqueue refused")
        self._rid_counter += 1
        rid = f"fake-req-{self._rid_counter}"
        self._pending = (rid, list(prompts[0].prompt_token_ids))
        return [rid]

    def wait_for_completion(self, use_tqdm: bool = True) -> list[_FakeRequestOutput]:
        if self.abort_mode == "wait_raises":
            raise RuntimeError("wait failed")
        rid, prompt_ids = self._pending
        self.abort_event.wait(timeout=5)
        if self.abort_mode == "no_terminal":
            return []
        if self.abort_mode == "wrong_finish":
            completion = _FakeCompletion([8, 9], "ab", "length")
        else:
            completion = _FakeCompletion([8, 9, 10], "abc", "abort")
        return [_FakeRequestOutput(rid, prompt_ids, completion)]

    def reset_prefix_cache(self) -> bool:
        if type(self).script.get("reset_raises"):
            raise RuntimeError("reset refused")
        return type(self).script.get("reset_result", True)


class _UnspecifiedPlatform:
    device_type = ""


class _CpuPlatform:
    device_type = "cpu"


def _fake_vllm_module(script: dict[str, Any] | None = None) -> dict[str, Any]:
    """Install the fake vllm module tree into sys.modules; returns
    the saved entries for restoration."""
    _FakeLLM.script = dict(script or {})
    _FakeLLM.instances = []
    _FakeSamplingParams.drift = bool((script or {}).get("params_drift"))
    module = types.ModuleType("vllm")
    module.__version__ = (script or {}).get("lib_version", "0.33.0")
    module.__file__ = "/fake/vllm/__init__.py"
    module.LLM = _FakeLLM
    module.SamplingParams = _FakeSamplingParams
    module.TokensPrompt = _FakeTokensPrompt
    platforms = types.ModuleType("vllm.platforms")
    platform = (script or {}).get("platform") or _CpuPlatform()
    platforms.current_platform = platform
    module.platforms = platforms
    detokenizer = types.ModuleType("vllm.v1.engine.detokenizer")
    detokenizer.uses_fast_detokenizer = lambda tokenizer: bool(_FakeLLM.script.get("detok_fast"))
    v1 = types.ModuleType("vllm.v1")
    engine_pkg = types.ModuleType("vllm.v1.engine")
    engine_pkg.detokenizer = detokenizer
    v1.engine = engine_pkg
    module.v1 = v1
    saved: dict[str, Any] = {}
    for name, fake in {
        "vllm": module,
        "vllm.platforms": platforms,
        "vllm.v1": v1,
        "vllm.v1.engine": engine_pkg,
        "vllm.v1.engine.detokenizer": detokenizer,
    }.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = fake
    return saved


def _restore_modules(saved: dict[str, Any]) -> None:
    for name, previous in saved.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


_FAKE_VERSIONS = {"vllm": "0.33.0+cpu", "torch": "2.14.0+cpu", "transformers": "5.20.2"}


def _fake_metadata_version(name: str) -> str:
    if name in _FAKE_VERSIONS:
        return _FAKE_VERSIONS[name]
    return _REAL_METADATA_VERSION(name)


_REAL_METADATA_VERSION = importlib.metadata.version


class _VllmBackendCase(unittest.TestCase):
    """Shared fixture: checkpoint dir + env/module hygiene."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model_dir = Path(self._tmp.name) / "ckpt"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text(
            json.dumps({"eos_token_id": 0, "max_position_embeddings": 512}),
            encoding="utf-8",
        )
        (self.model_dir / "generation_config.json").write_text(
            json.dumps({"eos_token_id": 0}), encoding="utf-8"
        )
        (self.model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (self.model_dir / "model.safetensors").write_bytes(b"weights")
        self._saved_env = {
            key: os.environ.get(key)
            for key in list(servingbackends._VLLM_ENV_PINS) + ["VLLM_CPU_KVCACHE_SPACE"]
        }
        self.addCleanup(self._restore_env)
        servingbackends._VLLM_ENV_DIGEST = None
        self.addCleanup(setattr, servingbackends, "_VLLM_ENV_DIGEST", None)

    def _restore_env(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def install_fake(self, script: dict[str, Any] | None = None) -> None:
        saved = _fake_vllm_module(script)
        self.addCleanup(_restore_modules, saved)
        patcher = mock.patch.object(importlib.metadata, "version", _fake_metadata_version)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_backend(self, **overrides: Any) -> servingbackends.VllmOfflineBackend:
        kwargs: dict[str, Any] = {
            "tokenizer": _VllmFakeTokenizer(),
            "kv_cache_memory_bytes": 512 * 1024 * 1024,
            "sleep": lambda _seconds: None,
        }
        kwargs.update(overrides)
        return servingbackends.VllmOfflineBackend(self.model_dir, **kwargs)


class VllmConstructionTests(_VllmBackendCase):
    def test_missing_checkpoint_dir(self) -> None:
        with self.assertRaises(ValueError) as caught:
            servingbackends.VllmOfflineBackend(
                Path(self._tmp.name) / "nope",
                tokenizer=_VllmFakeTokenizer(),
                kv_cache_memory_bytes=1024,
            )
        self.assertEqual(str(caught.exception), "stack_checkpoint_unavailable")

    def test_empty_eos_union_fails(self) -> None:
        (self.model_dir / "config.json").write_text(
            json.dumps({"max_position_embeddings": 512}), encoding="utf-8"
        )
        (self.model_dir / "generation_config.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            self.make_backend()
        self.assertEqual(str(caught.exception), "stack_checkpoint_unavailable")

    def test_window_resolution(self) -> None:
        backend = self.make_backend()
        self.assertEqual(backend._max_model_len, 512)
        explicit = self.make_backend(max_model_len=256)
        self.assertEqual(explicit._max_model_len, 256)
        (self.model_dir / "config.json").write_text(
            json.dumps({"eos_token_id": 0}), encoding="utf-8"
        )
        with self.assertRaises(ValueError) as caught:
            self.make_backend()
        self.assertEqual(str(caught.exception), "stack_checkpoint_unavailable")
        rescued = self.make_backend(max_model_len=128)
        self.assertEqual(rescued._max_model_len, 128)

    def test_env_drift_second_construction_refused(self) -> None:
        self.make_backend()
        os.environ["VLLM_CPU_KVCACHE_SPACE"] = "8"
        with self.assertRaises(ValueError) as caught:
            self.make_backend()
        self.assertEqual(str(caught.exception), "stack_config_unpinned")

    def test_import_failure_is_stack_unavailable(self) -> None:
        backend = self.make_backend()
        saved = sys.modules.get("vllm")
        sys.modules["vllm"] = None  # type: ignore[assignment]
        self.addCleanup(
            lambda: (
                sys.modules.__setitem__("vllm", saved)
                if saved is not None
                else sys.modules.pop("vllm", None)
            )
        )
        with self.assertRaises(ValueError) as caught:
            backend.probe()
        self.assertEqual(str(caught.exception), "stack_unavailable")


class VllmProbeGateTests(_VllmBackendCase):
    def test_identity_config_closed_key_set(self) -> None:
        self.install_fake()
        backend = self.make_backend()
        identity = backend.probe()
        self.assertEqual(identity.stack, "vllm")
        self.assertEqual(identity.stack_version, "0.33.0+cpu")
        self.assertEqual(identity.declared_eos_ids, (0,))
        self.assertEqual(
            set(identity.config),
            {
                "protocol",
                "stack",
                "vllm_version",
                "torch_version",
                "transformers_version",
                "vllm_module_digest",
                "platform",
                "env",
                "engine",
                "default_sampling_params_digest",
                "detokenizer_variant",
                "declared_eos_ids",
                "tier_a_vllm",
                "abort_probe",
            },
        )
        engine = identity.config["engine"]
        self.assertEqual(
            set(engine),
            {
                "checkpoint_digest",
                "seed",
                "enforce_eager",
                "max_num_seqs",
                "max_model_len",
                "enable_prefix_caching",
                "dtype_resolved",
                "kv_cache_memory_bytes",
                "kv_cache_sizing_input",
                "block_size",
                "num_gpu_blocks",
            },
        )
        self.assertEqual(engine["block_size"], 128)
        self.assertEqual(engine["num_gpu_blocks"], 56)
        self.assertEqual(engine["kv_cache_memory_bytes"], 512 * 1024 * 1024)
        self.assertEqual(engine["dtype_resolved"], "bfloat16")
        self.assertEqual(engine["max_model_len"], 512)
        self.assertEqual(identity.config["env"], dict(servingbackends._VLLM_ENV_PINS))
        self.assertEqual(
            identity.config["tier_a_vllm"],
            servingbackends.tier_a_sampling_params_vllm(1234),
        )
        self.assertEqual(identity.config["detokenizer_variant"], "SlowIncrementalDetokenizer")
        self.assertEqual(
            identity.config["platform"], {"device_type": "cpu", "platform_class": "_CpuPlatform"}
        )
        self.assertEqual(
            identity.config_fingerprint,
            servingbackends.config_fingerprint_for(identity.config),
        )
        abort = identity.config["abort_probe"]
        self.assertTrue(abort["terminal_observed"])
        self.assertEqual(abort["finish_reason"], "abort")
        self.assertEqual(abort["returned_tokens"], 3)
        caps = backend.capabilities
        self.assertTrue(all(caps.as_dict().values()))
        self.assertTrue(backend.verify_identity())
        self.assertTrue(backend.reset_cache())

    def test_g1_version_window(self) -> None:
        self.install_fake({"lib_version": "0.34.0"})
        backend = self.make_backend()
        with self.assertRaises(ValueError) as caught:
            backend.probe()
        self.assertEqual(str(caught.exception), "stack_identity_unavailable")

    def test_g1_platform_lie(self) -> None:
        self.install_fake({"platform": _UnspecifiedPlatform()})
        backend = self.make_backend()
        with self.assertRaises(ValueError) as caught:
            backend.probe()
        self.assertEqual(str(caught.exception), "stack_identity_unavailable")

    def test_g2_readback_lie_is_unpinned(self) -> None:
        self.install_fake({"block_size": -1})
        backend = self.make_backend()
        with self.assertRaises(ValueError) as caught:
            backend.probe()
        self.assertEqual(str(caught.exception), "stack_config_unpinned")

    def test_g3_prompt_echo_lie(self) -> None:
        def lying_handler(prompt_ids: list[int], params: Any) -> list[Any]:
            completion = _FakeCompletion([8], "a", "length")
            return [_FakeRequestOutput("fake-req", [999], completion)]

        self.install_fake({"generate_handler": lying_handler})
        backend = self.make_backend()
        with self.assertRaises(ValueError) as caught:
            backend.probe()
        self.assertEqual(str(caught.exception), "stack_identity_unavailable")

    def test_abort_no_terminal_degrades_claim_only(self) -> None:
        self.install_fake({"abort_mode": "no_terminal"})
        backend = self.make_backend()
        identity = backend.probe()
        abort = identity.config["abort_probe"]
        self.assertFalse(abort["terminal_observed"])
        self.assertIsNone(abort["finish_reason"])
        self.assertIsNone(abort["returned_tokens"])

    def test_abort_wrong_finish_degrades_claim_only(self) -> None:
        self.install_fake({"abort_mode": "wrong_finish"})
        backend = self.make_backend()
        identity = backend.probe()
        self.assertFalse(identity.config["abort_probe"]["terminal_observed"])


class VllmExecuteTests(_VllmBackendCase):
    def setUp(self) -> None:
        super().setUp()
        self.install_fake()
        self.backend = self.make_backend()
        self.backend.probe()
        self.llm = _FakeLLM.instances[-1]

    def _execute(self, handler: Any, **request_kwargs: Any) -> Any:
        self.llm.generate_handler = handler
        request = servingbackends.StackRequest(
            prompt_ids=(1, 2), max_new_tokens=8, seed=7, **request_kwargs
        )
        return self.backend.execute(request)

    def test_execute_before_probe_refused(self) -> None:
        fresh = self.make_backend()
        with self.assertRaises(ValueError) as caught:
            fresh.execute(servingbackends.StackRequest(prompt_ids=(1,), max_new_tokens=4, seed=0))
        self.assertEqual(str(caught.exception), "stack_identity_unavailable")

    def test_stop_text_winner(self) -> None:
        def handler(prompt_ids: list[int], params: Any) -> list[Any]:
            completion = _FakeCompletion([7], "abc", "stop", "abc")
            return [
                _FakeRequestOutput(
                    "r",
                    prompt_ids,
                    completion,
                    num_cached_tokens=128,
                    num_cache_creation_tokens=64,
                )
            ]

        observation = self._execute(handler, stop_texts=("abc",))
        self.assertEqual(observation.finish_class, "stop")
        self.assertEqual(
            observation.stop_winner,
            {
                "kind": "stop_text",
                "index": 0,
                "digest": servingbackends._text_digest("abc"),
            },
        )
        self.assertEqual(observation.tokens_cached_reported, 128)
        self.assertEqual(observation.cache_creation_tokens_reported, 64)
        self.assertIsNone(observation.cache_n_reported)
        self.assertIsNotNone(observation.config_echo_digest)
        self.assertEqual(observation.detail, "")

    def test_eos_winner_from_null_stop_reason(self) -> None:
        def handler(prompt_ids: list[int], params: Any) -> list[Any]:
            completion = _FakeCompletion([8, 0], "a<|endoftext|>", "stop", None)
            return [_FakeRequestOutput("r", prompt_ids, completion)]

        observation = self._execute(handler)
        self.assertEqual(observation.stop_winner, {"kind": "eos", "id": 0})

    def test_null_stop_reason_without_eos_has_no_winner(self) -> None:
        def handler(prompt_ids: list[int], params: Any) -> list[Any]:
            completion = _FakeCompletion([8, 9], "ab", "stop", None)
            return [_FakeRequestOutput("r", prompt_ids, completion)]

        observation = self._execute(handler)
        self.assertEqual(observation.finish_class, "stop")
        self.assertIsNone(observation.stop_winner)

    def test_stop_token_winner(self) -> None:
        def handler(prompt_ids: list[int], params: Any) -> list[Any]:
            completion = _FakeCompletion([8, 9], "ab", "stop", 9)
            return [_FakeRequestOutput("r", prompt_ids, completion)]

        observation = self._execute(handler, stop_token_ids=(9,))
        self.assertEqual(observation.stop_winner, {"kind": "stop_token", "id": 9})

    def test_unknown_stop_string_is_invalid(self) -> None:
        def handler(prompt_ids: list[int], params: Any) -> list[Any]:
            completion = _FakeCompletion([7], "abc", "stop", "not-submitted")
            return [_FakeRequestOutput("r", prompt_ids, completion)]

        observation = self._execute(handler, stop_texts=("abc",))
        self.assertEqual(observation.detail, "stack_response_invalid")
        self.assertIsNone(observation.returned_ids)

    def test_overshoot_is_invalid(self) -> None:
        def handler(prompt_ids: list[int], params: Any) -> list[Any]:
            completion = _FakeCompletion([8] * 9, "a" * 9, "length")
            return [_FakeRequestOutput("r", prompt_ids, completion)]

        observation = self._execute(handler)
        self.assertEqual(observation.detail, "stack_response_invalid")

    def test_error_and_unfinished_and_raise(self) -> None:
        def error_handler(prompt_ids: list[int], params: Any) -> list[Any]:
            return [_FakeRequestOutput("r", prompt_ids, _FakeCompletion([], "", "error"))]

        self.assertEqual(self._execute(error_handler).detail, "stack_execution_failed")

        def unfinished_handler(prompt_ids: list[int], params: Any) -> list[Any]:
            return [
                _FakeRequestOutput("r", prompt_ids, _FakeCompletion([8], "a", None), finished=False)
            ]

        self.assertEqual(self._execute(unfinished_handler).detail, "stack_response_unclassified")

        def raising_handler(prompt_ids: list[int], params: Any) -> list[Any]:
            raise RuntimeError("engine died")

        self.assertEqual(self._execute(raising_handler).detail, "stack_execution_failed")

    def test_sampler_drift_is_unpinned(self) -> None:
        _FakeSamplingParams.drift = True
        self.addCleanup(setattr, _FakeSamplingParams, "drift", False)
        observation = self._execute(
            lambda p, params: [_FakeRequestOutput("r", p, _FakeCompletion([8], "a", "length"))]
        )
        self.assertEqual(observation.detail, "stack_config_unpinned")

    def test_paired_arm_only_for_special_streams(self) -> None:
        calls: list[bool] = []

        def handler(prompt_ids: list[int], params: Any) -> list[Any]:
            calls.append(bool(params.skip_special_tokens))
            text = "a" if params.skip_special_tokens else "a<|endoftext|>"
            completion = _FakeCompletion([8, 0], text, "length")
            return [_FakeRequestOutput("r", prompt_ids, completion)]

        observation = self._execute(handler)
        self.assertEqual(calls, [True, False])
        self.assertEqual(
            observation.returned_text_digest_no_special,
            servingbackends._text_digest("a<|endoftext|>"),
        )

        calls.clear()

        def plain_handler(prompt_ids: list[int], params: Any) -> list[Any]:
            calls.append(bool(params.skip_special_tokens))
            return [_FakeRequestOutput("r", prompt_ids, _FakeCompletion([8], "a", "length"))]

        observation = self._execute(plain_handler)
        self.assertEqual(calls, [True])
        self.assertIsNone(observation.returned_text_digest_no_special)

    def test_cancel_raises(self) -> None:
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(servingbackends.ExecutionCancelled):
            self.backend.execute(
                servingbackends.StackRequest(prompt_ids=(1,), max_new_tokens=4, seed=0),
                cancel=cancel,
            )

    def test_verify_identity_detects_drift(self) -> None:
        self.llm.vllm_config.cache_config.block_size = 64
        self.assertFalse(self.backend.verify_identity())


class _ExplodingPlatform:
    @property
    def device_type(self) -> str:
        raise RuntimeError("platform unreadable")


class _LyingEngineTokenizer(_VllmFakeTokenizer):
    def encode(self, text: str) -> list[int]:
        return [1, 2]


class _RaisingEngineTokenizer(_VllmFakeTokenizer):
    def encode(self, text: str) -> list[int]:
        raise RuntimeError("tokenizer dead")


class _StreamDecodeTokenizer(_VllmFakeTokenizer):
    """Decodes the sentinel (so G4 passes) but refuses any other
    stream (so execute's detokenization cross-check is unobservable)."""

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        if list(ids) != [1, 2, 3, 4, 5]:
            raise RuntimeError("stream undecodable")
        return super().decode(ids, skip_special_tokens=skip_special_tokens)


class VllmFailureBranchTests(_VllmBackendCase):
    """Every defensive branch of the vLLM backend, driven through
    the hostile fake: construction validation, probe gate failures,
    abort-cell failures, and classification edges."""

    # -- construction / local validation --------------------------------

    def test_bad_hardware_class_and_kv_bytes(self) -> None:
        with self.assertRaises(ValueError):
            self.make_backend(hardware_class="tpu")
        with self.assertRaises(ValueError):
            self.make_backend(kv_cache_memory_bytes=0)
        with self.assertRaises(ValueError):
            self.make_backend(kv_cache_memory_bytes=True)

    def test_empty_directory(self) -> None:
        empty = Path(self._tmp.name) / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError) as caught:
            servingbackends.VllmOfflineBackend(
                empty,
                tokenizer=_VllmFakeTokenizer(),
                kv_cache_memory_bytes=1024,
            )
        self.assertEqual(str(caught.exception), "stack_checkpoint_unavailable")

    def test_eos_union_forms(self) -> None:
        # List form + bool junk ignored + malformed sibling file.
        (self.model_dir / "config.json").write_text(
            json.dumps({"eos_token_id": [0, 50256, True], "n_positions": 256}),
            encoding="utf-8",
        )
        (self.model_dir / "generation_config.json").write_text("{not json", encoding="utf-8")
        backend = self.make_backend()
        self.assertEqual(backend._declared_eos_ids, (0, 50256))
        self.assertEqual(backend._max_model_len, 256)

    def test_window_explicit_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.make_backend(max_model_len=0)
        with self.assertRaises(ValueError):
            self.make_backend(max_model_len=True)
        (self.model_dir / "config.json").unlink()
        with self.assertRaises(ValueError):
            self.make_backend()
        rescued = self.make_backend(max_model_len=64)
        self.assertEqual(rescued._max_model_len, 64)

    def test_manifest_failure(self) -> None:
        patcher = mock.patch.object(backends, "_manifest", side_effect=OSError("unreadable member"))
        patcher.start()
        self.addCleanup(patcher.stop)
        with self.assertRaises(ValueError) as caught:
            self.make_backend()
        self.assertEqual(str(caught.exception), "stack_checkpoint_unavailable")

    def test_non_dict_config_payload_ignored(self) -> None:
        (self.model_dir / "config.json").write_text("[1, 2]", encoding="utf-8")
        (self.model_dir / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [3]}), encoding="utf-8"
        )
        backend = self.make_backend(max_model_len=64)
        self.assertEqual(backend._declared_eos_ids, (3,))

    # -- probe gates ------------------------------------------------------

    def test_probe_is_idempotent_and_capabilities_gate(self) -> None:
        self.install_fake()
        backend = self.make_backend()
        with self.assertRaises(ValueError):
            _ = backend.capabilities
        first = backend.probe()
        self.assertIs(backend.probe(), first)

    def test_metadata_failure(self) -> None:
        self.install_fake()
        real = _fake_metadata_version

        def broken(name: str) -> str:
            if name == "torch":
                raise RuntimeError("no metadata")
            return real(name)

        patcher = mock.patch.object(importlib.metadata, "version", broken)
        patcher.start()
        self.addCleanup(patcher.stop)
        with self.assertRaises(ValueError) as caught:
            self.make_backend().probe()
        self.assertEqual(str(caught.exception), "stack_identity_unavailable")

    def test_platform_unreadable(self) -> None:
        self.install_fake({"platform": _ExplodingPlatform()})
        with self.assertRaises(ValueError) as caught:
            self.make_backend().probe()
        self.assertEqual(str(caught.exception), "stack_identity_unavailable")

    def test_engine_construction_failure(self) -> None:
        self.install_fake({"llm_raises": True})
        with self.assertRaises(ValueError) as caught:
            self.make_backend().probe()
        self.assertEqual(str(caught.exception), "stack_unavailable")

    def test_env_mutation_between_construction_and_probe(self) -> None:
        self.install_fake()
        backend = self.make_backend()
        os.environ["HF_HUB_OFFLINE"] = "0"
        with self.assertRaises(ValueError) as caught:
            backend.probe()
        self.assertEqual(str(caught.exception), "stack_config_unpinned")

    def test_config_broken_readback(self) -> None:
        self.install_fake({"config_broken": True})
        with self.assertRaises(ValueError) as caught:
            self.make_backend().probe()
        self.assertEqual(str(caught.exception), "stack_config_unpinned")

    def test_default_params_failure(self) -> None:
        self.install_fake({"default_params_raises": True})
        with self.assertRaises(ValueError) as caught:
            self.make_backend().probe()
        self.assertEqual(str(caught.exception), "stack_config_unpinned")

    def test_fast_detokenizer_variant(self) -> None:
        self.install_fake({"detok_fast": True})
        identity = self.make_backend().probe()
        self.assertEqual(identity.config["detokenizer_variant"], "FastIncrementalDetokenizer")

    def test_processor_broken(self) -> None:
        self.install_fake({"processor_broken": True})
        with self.assertRaises(ValueError) as caught:
            self.make_backend().probe()
        self.assertEqual(str(caught.exception), "stack_config_unpinned")

    def test_sentinel_encode_failures(self) -> None:
        self.install_fake()

        class _RaisingTokenizer(_VllmFakeTokenizer):
            def encode(self, text: str) -> list[int]:
                raise RuntimeError("no encode")

        with self.assertRaises(ValueError) as caught:
            self.make_backend(tokenizer=_RaisingTokenizer()).probe()
        self.assertEqual(str(caught.exception), "stack_identity_unavailable")

        class _ShortTokenizer(_VllmFakeTokenizer):
            def encode(self, text: str) -> list[int]:
                return [1]

        with self.assertRaises(ValueError) as caught:
            self.make_backend(tokenizer=_ShortTokenizer()).probe()
        self.assertEqual(str(caught.exception), "stack_identity_unavailable")

    def test_g3_params_drift_fails_probe(self) -> None:
        self.install_fake({"params_drift": True})
        with self.assertRaises(ValueError) as caught:
            self.make_backend().probe()
        self.assertEqual(str(caught.exception), "stack_config_unpinned")

    def test_g3_generate_raises(self) -> None:
        def raising_handler(prompt_ids: list[int], params: Any) -> list[Any]:
            raise RuntimeError("generate dead")

        self.install_fake({"generate_handler": raising_handler})
        with self.assertRaises(ValueError) as caught:
            self.make_backend().probe()
        self.assertEqual(str(caught.exception), "stack_identity_unavailable")

    def test_g4_engine_tokenizer_lies_and_raises(self) -> None:
        self.install_fake({"engine_tokenizer": _LyingEngineTokenizer()})
        with self.assertRaises(ValueError) as caught:
            self.make_backend().probe()
        self.assertEqual(str(caught.exception), "stack_identity_unavailable")
        self.install_fake({"engine_tokenizer": _RaisingEngineTokenizer()})
        with self.assertRaises(ValueError) as caught:
            self.make_backend().probe()
        self.assertEqual(str(caught.exception), "stack_identity_unavailable")

    def test_abort_wait_and_enqueue_failures(self) -> None:
        self.install_fake({"abort_mode": "wait_raises"})
        identity = self.make_backend().probe()
        self.assertFalse(identity.config["abort_probe"]["terminal_observed"])
        self.install_fake({"abort_mode": "enqueue_raises"})
        identity = self.make_backend().probe()
        self.assertFalse(identity.config["abort_probe"]["terminal_observed"])

    # -- execute branches ---------------------------------------------------

    def _probed(self, script: dict[str, Any] | None = None) -> Any:
        self.install_fake(script)
        backend = self.make_backend()
        backend.probe()
        return backend, _FakeLLM.instances[-1]

    def test_params_construction_failure_is_unpinned(self) -> None:
        backend, _llm = self._probed()
        _FakeSamplingParams.raise_always = True
        self.addCleanup(setattr, _FakeSamplingParams, "raise_always", False)
        observation = backend.execute(
            servingbackends.StackRequest(prompt_ids=(1, 2), max_new_tokens=4, seed=1)
        )
        self.assertEqual(observation.detail, "stack_config_unpinned")

    def test_classify_malformed_outputs(self) -> None:
        backend, llm = self._probed()
        request = servingbackends.StackRequest(prompt_ids=(1, 2), max_new_tokens=4, seed=1)
        llm.generate_handler = lambda p, params: [object()]
        self.assertEqual(backend.execute(request).detail, "stack_response_unclassified")
        llm.generate_handler = lambda p, params: [
            _FakeRequestOutput("r", p, _FakeCompletion([8], "a", "banana"))
        ]
        self.assertEqual(backend.execute(request).detail, "stack_response_unclassified")
        llm.generate_handler = lambda p, params: [
            _FakeRequestOutput("r", p, _FakeCompletion(["x"], "a", "length"))
        ]
        self.assertEqual(backend.execute(request).detail, "stack_response_invalid")
        llm.generate_handler = lambda p, params: [
            _FakeRequestOutput("r", p, _FakeCompletion([8], None, "length"))
        ]
        self.assertEqual(backend.execute(request).detail, "stack_response_invalid")
        llm.generate_handler = lambda p, params: [
            _FakeRequestOutput("r", p, _FakeCompletion([8], "a", "stop", 1.5))
        ]
        self.assertEqual(backend.execute(request).detail, "stack_response_invalid")

    def test_paired_arm_failures_leave_field_unmeasured(self) -> None:
        backend, llm = self._probed()
        request = servingbackends.StackRequest(prompt_ids=(1, 2), max_new_tokens=4, seed=1)
        calls = {"n": 0}

        def flaky(prompt_ids: list[int], params: Any) -> list[Any]:
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("paired died")
            return [_FakeRequestOutput("r", prompt_ids, _FakeCompletion([8, 0], "a", "length"))]

        llm.generate_handler = flaky
        observation = backend.execute(request)
        self.assertEqual(observation.returned_ids, (8, 0))
        self.assertIsNone(observation.returned_text_digest_no_special)

        def bad_text(prompt_ids: list[int], params: Any) -> list[Any]:
            text = "a" if params.skip_special_tokens else None
            return [_FakeRequestOutput("r", prompt_ids, _FakeCompletion([8, 0], text, "length"))]

        llm.generate_handler = bad_text
        observation = backend.execute(request)
        self.assertIsNone(observation.returned_text_digest_no_special)

        _FakeSamplingParams.raise_on_paired = True
        self.addCleanup(setattr, _FakeSamplingParams, "raise_on_paired", False)
        llm.generate_handler = lambda p, params: [
            _FakeRequestOutput("r", p, _FakeCompletion([8, 0], "a", "length"))
        ]
        observation = backend.execute(request)
        self.assertIsNone(observation.returned_text_digest_no_special)

    def test_execute_detok_unobservable(self) -> None:
        backend, llm = self._probed({"engine_tokenizer": _StreamDecodeTokenizer()})
        llm.generate_handler = lambda p, params: [
            _FakeRequestOutput("r", p, _FakeCompletion([8, 9], "ab", "length"))
        ]
        observation = backend.execute(
            servingbackends.StackRequest(prompt_ids=(1, 2), max_new_tokens=4, seed=1)
        )
        self.assertEqual(observation.returned_ids, (8, 9))
        self.assertIsNone(observation.server_detok_digest)

    def test_mistyped_cache_reports_are_unreported(self) -> None:
        backend, llm = self._probed()
        llm.generate_handler = lambda p, params: [
            _FakeRequestOutput(
                "r",
                p,
                _FakeCompletion([8], "a", "length"),
                num_cached_tokens=-5,
                num_cache_creation_tokens="junk",
            )
        ]
        observation = backend.execute(
            servingbackends.StackRequest(prompt_ids=(1, 2), max_new_tokens=4, seed=1)
        )
        self.assertIsNone(observation.tokens_cached_reported)
        self.assertIsNone(observation.cache_creation_tokens_reported)

    def test_empty_stream_stop_has_no_winner(self) -> None:
        backend, llm = self._probed()
        llm.generate_handler = lambda p, params: [
            _FakeRequestOutput("r", p, _FakeCompletion([], "", "stop", None))
        ]
        observation = backend.execute(
            servingbackends.StackRequest(prompt_ids=(1, 2), max_new_tokens=4, seed=1)
        )
        self.assertEqual(observation.returned_ids, ())
        self.assertIsNone(observation.stop_winner)

    def test_special_set_fallback_to_declared_eos(self) -> None:
        class _NoSpecialsTokenizer(_VllmFakeTokenizer):
            all_special_ids: list[int] = []

        backend, llm = self._probed({"engine_tokenizer": _NoSpecialsTokenizer()})
        calls: list[bool] = []

        def handler(prompt_ids: list[int], params: Any) -> list[Any]:
            calls.append(bool(params.skip_special_tokens))
            return [_FakeRequestOutput("r", prompt_ids, _FakeCompletion([8, 0], "a", "length"))]

        llm.generate_handler = handler
        observation = backend.execute(
            servingbackends.StackRequest(prompt_ids=(1, 2), max_new_tokens=4, seed=1)
        )
        # The declared-EOS fallback set still sees token 0 as special.
        self.assertEqual(calls, [True, False])
        self.assertIsNotNone(observation.returned_text_digest_no_special)

    def test_missing_engine_tokenizer_degrades_gracefully(self) -> None:
        backend, llm = self._probed()
        backend._engine_tokenizer = None
        llm.generate_handler = lambda p, params: [
            _FakeRequestOutput("r", p, _FakeCompletion([8, 0], "a", "length"))
        ]
        observation = backend.execute(
            servingbackends.StackRequest(prompt_ids=(1, 2), max_new_tokens=4, seed=1)
        )
        self.assertEqual(observation.returned_ids, (8, 0))
        self.assertIsNone(observation.server_detok_digest)
        # The declared-EOS fallback still triggers the paired arm.
        self.assertIsNotNone(observation.returned_text_digest_no_special)

    def test_reset_and_verify_failure_branches(self) -> None:
        fresh = self.make_backend()
        self.assertFalse(fresh.reset_cache())
        self.assertFalse(fresh.verify_identity())
        backend, llm = self._probed({"reset_raises": True})
        self.assertFalse(backend.reset_cache())
        llm.llm_engine.vllm_config = types.SimpleNamespace()
        self.assertFalse(backend.verify_identity())


# ---------------------------------------------------------------------------
# W6 Push 3: Step 5 — the cache driver generalized to the
# ExecutionBackend protocol (per-stack reset seam, contamination
# quantum, and position shapes). Llama expectations above stay
# byte-identical (B1).
# ---------------------------------------------------------------------------


class LlamaResetCacheSeamTests(ServingBackendsTestBase):
    """LlamacppHttpBackend.reset_cache() is the slot erase under the
    cache driver's protocol name."""

    def test_reset_cache_delegates_to_slot_erase(self) -> None:
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            self.assertTrue(backend.reset_cache())
        with _FakeServer(self.gguf) as server:
            server.state.erase_enabled = False
            backend = self.probed_backend(server)
            self.assertFalse(backend.reset_cache())


class VllmCacheSequenceTests(_VllmBackendCase):
    """run_cache_sequence on the vLLM arm: positions are
    {tokens_cached, cache_creation_tokens}; the contamination
    quantum is one full block (128 on the fake)."""

    COLD = (1, 2, 3, 4)
    SHARED = (1, 2, 9, 9)
    DIVERGENT = (13, 10)

    def _probed(self, script: dict[str, Any] | None = None) -> Any:
        self.install_fake(script)
        backend = self.make_backend()
        backend.probe()
        return backend, _FakeLLM.instances[-1]

    def _run(self, backend: Any) -> dict[str, Any]:
        tokenizer = _VllmFakeTokenizer()
        return servingbackends.run_cache_sequence(
            backend,
            cold_ids=self.COLD,
            shared_ids=self.SHARED,
            divergent_ids=self.DIVERGENT,
            seed=5,
            decode=lambda ids: tokenizer.decode(list(ids)),
        )

    def _script(self, llm: Any, table: dict[tuple[int, ...], list[tuple[int, int]]]) -> None:
        def handler(prompt_ids: list[int], params: Any) -> list[Any]:
            key = tuple(prompt_ids)
            cached, creation = table[key].pop(0)
            return [
                _FakeRequestOutput(
                    "r",
                    prompt_ids,
                    _FakeCompletion([8], "a", "length"),
                    num_cached_tokens=cached,
                    num_cache_creation_tokens=creation,
                )
            ]

        llm.generate_handler = handler

    def test_g5l_shapes(self) -> None:
        backend, llm = self._probed()
        self._script(
            llm,
            {
                self.COLD: [(0, 512), (384, 128)],
                self.SHARED: [(128, 256)],
                self.DIVERGENT: [(0, 384)],
            },
        )
        result = self._run(backend)
        self.assertTrue(result["available"])
        self.assertEqual(result["detail"], "")
        self.assertEqual(
            result["positions"],
            {
                "cold": {"tokens_cached": 0, "cache_creation_tokens": 512},
                "warm_repeat": {"tokens_cached": 384, "cache_creation_tokens": 128},
                "shared_prefix": {"tokens_cached": 128, "cache_creation_tokens": 256},
                "divergent_control": {"tokens_cached": 0, "cache_creation_tokens": 384},
            },
        )
        self.assertEqual(len(result["records"]), 4)

    def test_contamination_at_one_block(self) -> None:
        backend, llm = self._probed()
        self._script(
            llm,
            {
                self.COLD: [(0, 512), (384, 128)],
                self.SHARED: [(128, 256)],
                self.DIVERGENT: [(128, 256)],
            },
        )
        result = self._run(backend)
        self.assertEqual(result["detail"], "cache_protocol_contaminated")
        backend2, llm2 = self._probed()
        self._script(
            llm2,
            {
                self.COLD: [(128, 384), (384, 128)],
                self.SHARED: [(128, 256)],
                self.DIVERGENT: [(0, 384)],
            },
        )
        result2 = self._run(backend2)
        self.assertEqual(result2["detail"], "cache_protocol_contaminated")

    def test_reset_failure_records_absent_cells(self) -> None:
        backend, _llm = self._probed({"reset_result": False})
        result = self._run(backend)
        self.assertFalse(result["available"])
        self.assertEqual(result["detail"], servingbackends.CAPABILITY_ABSENT)
        self.assertEqual(len(result["records"]), 4)
        self.assertEqual(result["positions"], {})

    def test_unreported_quantities_are_capability_absent(self) -> None:
        backend, llm = self._probed()
        self._script(
            llm,
            {
                self.COLD: [(None, 512), (384, 128)],
                self.SHARED: [(128, 256)],
                self.DIVERGENT: [(0, 384)],
            },
        )
        result = self._run(backend)
        self.assertEqual(result["detail"], servingbackends.CAPABILITY_ABSENT)

    def test_cache_salt_passed_through_tokens_prompt(self) -> None:
        backend, llm = self._probed()
        llm.generate_calls.clear()
        backend.execute(servingbackends.StackRequest(prompt_ids=(1, 2), max_new_tokens=4, seed=3))
        backend.execute(
            servingbackends.StackRequest(
                prompt_ids=(1, 2), max_new_tokens=4, seed=3, cache_salt="salt-1"
            )
        )
        salts = [call[2].cache_salt for call in llm.generate_calls]
        self.assertEqual(salts, [None, "salt-1"])


# ---------------------------------------------------------------------------
# W6 Push 4: Steps 6+7 — CLI --stack dispatch, per-stack argument
# discipline, the SA-02-shaped conservation pin on both stacks, and
# the serve extra.
# ---------------------------------------------------------------------------


class CliStackValidationTests(ServingBackendsTestBase):
    """Per-stack CLI requirements: each stack's options are required
    on their own arm and rejected on the other (parser.error, exit 2
    via SystemExit), before any backend is constructed."""

    def _parser(self) -> Any:
        import argparse

        return argparse.ArgumentParser(prog="exec-backends")

    def _run(self, out: Path, **overrides: Any) -> int:
        with (
            mock.patch(
                "model_protocol_lab.doctor._build_primary_tokenizer",
                return_value=FakeTokenizer(),
            ),
            mock.patch.object(
                servingbackends, "_load_hf_tokenizer", return_value=_VllmFakeTokenizer()
            ),
        ):
            return servingbackends.run_exec_backends(
                self._args(None, out, **overrides), self._parser()
            )

    def _vllm_overrides(self, **extra: Any) -> dict[str, Any]:
        values: dict[str, Any] = {
            "stack": "vllm",
            "base_url": None,
            "api_key": None,
            "gguf": None,
            "timeout": None,
            "kv_cache_mib": 512,
        }
        values.update(extra)
        return values

    def test_llama_arm_requirements(self) -> None:
        out = Path(self._tmpdir.name)
        with self.assertRaises(SystemExit):
            self._run(out / "a.json", base_url=None)
        with self.assertRaises(SystemExit):
            self._run(out / "b.json", gguf=None)
        with self.assertRaises(SystemExit):
            self._run(out / "c.json", kv_cache_mib=512)
        with self.assertRaises(SystemExit):
            self._run(out / "d.json", max_model_len=1024)

    def test_unknown_stack_rejected(self) -> None:
        out = Path(self._tmpdir.name) / "u.json"
        with self.assertRaises(SystemExit):
            self._run(out, stack="bogus")

    def test_load_hf_tokenizer_is_local_only(self) -> None:
        calls: dict[str, Any] = {}
        fake = types.ModuleType("transformers")

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(path: str, **kwargs: Any) -> str:
                calls["path"] = path
                calls["kwargs"] = kwargs
                return "tok"

        fake.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"transformers": fake}):
            result = servingbackends._load_hf_tokenizer(Path("/ckpt"))
        self.assertEqual(result, "tok")
        self.assertEqual(calls["path"], "/ckpt")
        self.assertEqual(calls["kwargs"], {"local_files_only": True})

    def test_vllm_arm_rejections_construct_nothing(self) -> None:
        saved = _fake_vllm_module({})
        self.addCleanup(_restore_modules, saved)
        out = Path(self._tmpdir.name)
        for name, extra in (
            ("gguf", {"gguf": self.gguf}),
            ("base_url", {"base_url": "http://127.0.0.1:8080"}),
            ("api_key", {"api_key": "k"}),
            ("timeout", {"timeout": 5.0}),
            ("missing_kv", {"kv_cache_mib": None}),
            ("zero_kv", {"kv_cache_mib": 0}),
            ("bad_len", {"max_model_len": 0}),
        ):
            with self.assertRaises(SystemExit, msg=name):
                self._run(out / f"{name}.json", **self._vllm_overrides(**extra))
        self.assertEqual(_FakeLLM.instances, [])


class VllmHandlerEndToEndTests(_VllmBackendCase):
    """The full CLI handler on the vllm arm against the fake engine:
    exit 0 and an available section whose identity names the stack."""

    def test_handler_vllm_end_to_end(self) -> None:
        self.install_fake()
        model_dir = self.model_dir
        out = Path(self._tmp.name) / "vllm-artifact.json"
        import argparse

        args = argparse.Namespace(
            command="exec-backends",
            stack="vllm",
            max_model_len=None,
            kv_cache_mib=512,
            base_url=None,
            api_key=None,
            gguf=None,
            model=model_dir,
            out=out,
            prompt_ids="1,2,3",
            stops=None,
            eos_id=None,
            min_tokens=0,
            max_new_tokens=8,
            results=None,
            cache_cold_ids=None,
            cache_shared_ids=None,
            cache_divergent_ids=None,
            seed=0,
            timeout=None,
            hardware_class="cpu",
        )
        with mock.patch.object(
            servingbackends, "_load_hf_tokenizer", return_value=_VllmFakeTokenizer()
        ):
            code = servingbackends.run_exec_backends(
                args, argparse.ArgumentParser(prog="exec-backends")
            )
        self.assertEqual(code, 0)
        doc = json.loads(out.read_text(encoding="utf-8"))
        section = doc["serving_backends"]
        self.assertTrue(section["available"])
        self.assertEqual(section["identity"]["stack"], "vllm")


class ConservationPinTests(ServingBackendsTestBase):
    """SA-02-shaped pin (derived counts; the audit key set is frozen
    by the mirror): for a budget-degraded stage,
    n_planned == n_executions + n_degraded + n_not_executed, with
    every count derived independently from the records."""

    def _assert_conservation(self, stage: dict[str, Any], n_planned: int) -> None:
        records = stage["records"]
        audit = stage["audit"]
        standalone = [r for r in records if r["cache_position"] == "standalone"]
        executed = [r for r in standalone if r["cell_detail"] == ""]
        degraded = [
            r for r in standalone if r["cell_detail"] == servingbackends.EXECUTION_BUDGET_DETAIL
        ]
        not_executed = [
            r
            for r in standalone
            if r["cell_detail"] not in ("", servingbackends.EXECUTION_BUDGET_DETAIL)
        ]
        # The audit's universe is the plan cells: its planned count
        # reconciles exactly against independently derived record
        # counts, and its execution counter equals the derived
        # executed count.
        self.assertEqual(audit["n_planned_cells"], n_planned)
        self.assertEqual(audit["n_executions"], len(executed))
        self.assertEqual(n_planned, len(executed) + len(degraded) + len(not_executed))
        self.assertEqual(audit["n_records"], len(records))
        # The cache tail is present and every arm is accounted for:
        # executed, or budget-degraded once the budget is spent.
        cache_records = [r for r in records if r["cache_position"] != "standalone"]
        self.assertEqual(len(cache_records), 4)
        self.assertTrue(
            all(
                r["cell_detail"] in ("", servingbackends.EXECUTION_BUDGET_DETAIL)
                for r in cache_records
            )
        )

    def _cells(self, backend_decode: Any) -> list[dict[str, Any]]:
        return [
            {
                "candidate": "t",
                "fixture": f"t{i}",
                "request": servingbackends.StackRequest(
                    prompt_ids=(1, 2, 3), seed=7, max_new_tokens=4, eos_id=0
                ),
                "cache_position": "standalone",
            }
            for i in range(3)
        ]

    def test_conservation_llama_budget_degraded(self) -> None:
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            seen = {"n": 0}

            def on_record(record: dict[str, Any]) -> bool:
                seen["n"] += 1
                return seen["n"] < 2

            stage = servingbackends.run_stage(
                backend,
                plan_cells=self._cells(backend.decode_ids),
                fidelity_input=None,
                cache_prompts={
                    "cold_ids": (8, 12),
                    "shared_prefix_ids": (8, 12, 9),
                    "divergent_control_ids": (1, 2),
                },
                seed=0,
                decode=backend.decode_ids,
                capabilities=backend.capabilities,
                on_record=on_record,
            )
            self._assert_conservation(stage, 3)

    def test_conservation_vllm_budget_degraded(self) -> None:
        saved_env = dict(os.environ)
        saved_digest = servingbackends._VLLM_ENV_DIGEST
        try:
            self._conservation_vllm_budget_degraded()
        finally:
            os.environ.clear()
            os.environ.update(saved_env)
            servingbackends._VLLM_ENV_DIGEST = saved_digest

    def _conservation_vllm_budget_degraded(self) -> None:
        saved = _fake_vllm_module({})
        self.addCleanup(_restore_modules, saved)
        patcher = mock.patch.object(importlib.metadata, "version", _fake_metadata_version)
        patcher.start()
        self.addCleanup(patcher.stop)
        tokenizer = _VllmFakeTokenizer()
        model_dir = Path(self._tmpdir.name) / "hf-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
        )
        (model_dir / "generation_config.json").write_text(
            json.dumps({"eos_token_id": 0}), encoding="utf-8"
        )
        backend = servingbackends.VllmOfflineBackend(
            model_dir, tokenizer=tokenizer, kv_cache_memory_bytes=512 * 1024 * 1024
        )
        backend.probe()
        seen = {"n": 0}

        def on_record(record: dict[str, Any]) -> bool:
            seen["n"] += 1
            return seen["n"] < 2

        stage = servingbackends.run_stage(
            backend,
            plan_cells=self._cells(None),
            fidelity_input=None,
            cache_prompts={
                "cold_ids": (1, 2, 3, 4),
                "shared_prefix_ids": (1, 2, 9, 9),
                "divergent_control_ids": (13, 10),
            },
            seed=0,
            decode=lambda ids: tokenizer.decode(list(ids)),
            capabilities=backend.capabilities,
            on_record=on_record,
        )
        self._assert_conservation(stage, 3)


class ServeExtraTests(unittest.TestCase):
    """Step 7: the serve extra pins the admitted vLLM release
    exactly (extras are alternatives, not a union)."""

    def test_serve_extra_exact_pin(self) -> None:
        import tomllib

        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["project"]["optional-dependencies"]["serve"], ["vllm==0.33.0"])
