"""Cycle 165: surface-noise MCP server contract tests (real stdio).

Spawns model_protocol_lab.mcp_server as a subprocess over stdio and
exercises the JSON-RPC 2.0 contract: tools/list, server/discover,
initialize-compat, tools/call for each tool, and the error codes.
"""

import json
import subprocess
import sys
import unittest

PYTHON = sys.executable
# The server ships as model_protocol_lab.mcp_server (cycle 208);
# spawn it as a module so the test exercises the shipped form.
SERVER_CMD = [PYTHON, "-m", "model_protocol_lab.mcp_server"]


def _run_session(requests, timeout=120):
    """Send newline-delimited JSON-RPC requests; return response dicts."""
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    proc = subprocess.run(
        SERVER_CMD,
        input=payload,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, f"server exited {proc.returncode}: {proc.stderr[-2000:]}"
    responses = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            responses.append(json.loads(line))
    return responses


def _req(id_, method, params=None):
    request = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        request["params"] = params
    return request


class MCPContractTests(unittest.TestCase):
    def test_self_test_flag(self):
        proc = subprocess.run(
            SERVER_CMD + ["--self-test"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("self-test: OK", proc.stderr)

    def test_tools_list(self):
        (resp,) = _run_session([_req(1, "tools/list")])
        self.assertEqual(resp["id"], 1)
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        self.assertEqual(
            names,
            {
                "probe_surface_noise_stability",
                "surface_noise_hardening_notes",
                "surface_noise_protocol_info",
                "surface_noise_controls",
            },
        )
        for tool in tools:
            self.assertIn("inputSchema", tool)
            self.assertIn("description", tool)

    def test_server_discover(self):
        (resp,) = _run_session([_req(1, "server/discover")])
        self.assertIn("tools", resp["result"]["capabilities"])
        self.assertIn("serverInfo", resp["result"])
        self.assertIn("2026-07-28", resp["result"]["protocolVersions"])

    def test_initialize_compat(self):
        # The server accepts initialize without a prior handshake (liberal
        # acceptance); it must not require the 2026-07-28 stateless
        # framing nor the older stateful one.
        (resp,) = _run_session(
            [_req(1, "initialize", {"protocolVersion": "2026-07-28", "clientInfo": {}})]
        )
        self.assertIn("result", resp)
        self.assertIn("protocolVersion", resp["result"])

    def test_probe_tool_end_to_end(self):
        (resp,) = _run_session(
            [
                _req(
                    1,
                    "tools/call",
                    {
                        "name": "probe_surface_noise_stability",
                        "arguments": {
                            "prompt": "MAGICWORD please answer now",
                            "fixtures": [
                                {
                                    "id": "f1",
                                    "messages": [{"role": "user", "content": "x"}],
                                    "expected": "BLUE",
                                    "group": "probe",
                                }
                            ],
                            "boundary": [7],
                        },
                    },
                )
            ]
        )
        self.assertIn("result", resp)
        structured = resp["result"]["structuredContent"]
        self.assertEqual(structured["status"], "ok")
        self.assertEqual(structured["instrument"], "surface-noise-probe")
        self.assertIn("typo", structured["operators"])
        # The Adapter Rule travels with every result.
        text = resp["result"]["content"][0]["text"]
        self.assertIn("surface-form", text.lower())

    def test_controls_tool(self):
        (resp,) = _run_session(
            [
                _req(
                    1,
                    "tools/call",
                    {
                        "name": "surface_noise_controls",
                        "arguments": {"boundary": [7]},
                    },
                )
            ]
        )
        structured = resp["result"]["structuredContent"]
        self.assertTrue(structured["discriminates"])
        self.assertEqual(len(structured["controls"]), 2)

    def test_protocol_info_tool(self):
        (resp,) = _run_session(
            [_req(1, "tools/call", {"name": "surface_noise_protocol_info", "arguments": {}})]
        )
        structured = resp["result"]["structuredContent"]
        self.assertEqual(structured["instrument"], "surface-noise-probe")
        self.assertIn("adapter_rule", structured)

    def test_hardening_notes_tool(self):
        # hardening_notes surfaces counterfactuals from an existing probe
        # report: run the probe first, then pass its structured content.
        responses = _run_session(
            [
                _req(
                    1,
                    "tools/call",
                    {
                        "name": "probe_surface_noise_stability",
                        "arguments": {
                            "prompt": "MAGICWORD please answer now",
                            "fixtures": [
                                {
                                    "id": "f1",
                                    "messages": [{"role": "user", "content": "x"}],
                                    "expected": "BLUE",
                                    "group": "probe",
                                }
                            ],
                            "boundary": [7],
                        },
                    },
                ),
                _req(
                    2,
                    "tools/call",
                    {"name": "surface_noise_hardening_notes", "arguments": {}},
                ),
            ]
        )
        probe_report = responses[0]["result"]["structuredContent"]
        # Re-send with the report attached (the tool takes a report object).
        (resp,) = _run_session(
            [
                _req(
                    1,
                    "tools/call",
                    {
                        "name": "surface_noise_hardening_notes",
                        "arguments": {"report": probe_report},
                    },
                )
            ]
        )
        structured = resp["result"]["structuredContent"]
        self.assertIn("counterfactuals", structured)
        self.assertIn("loop_guidance", structured)
        # A missing report is a param error, not a crash.
        (err,) = _run_session(
            [
                _req(
                    1,
                    "tools/call",
                    {"name": "surface_noise_hardening_notes", "arguments": {}},
                )
            ]
        )
        self.assertEqual(err["error"]["code"], -32602)

    def test_unknown_tool_error(self):
        # tools/call exists, so an unknown tool NAME is an invalid param.
        (resp,) = _run_session([_req(1, "tools/call", {"name": "nope", "arguments": {}})])
        self.assertEqual(resp["error"]["code"], -32602)

    def test_unknown_method_error(self):
        (resp,) = _run_session([_req(1, "frobnicate")])
        self.assertEqual(resp["error"]["code"], -32601)

    def test_invalid_params_error(self):
        (resp,) = _run_session(
            [
                _req(
                    1,
                    "tools/call",
                    {
                        "name": "probe_surface_noise_stability",
                        "arguments": {"prompt": "x"},  # missing fixtures/boundary
                    },
                )
            ]
        )
        self.assertEqual(resp["error"]["code"], -32602)

    def test_parse_error(self):
        (resp,) = _run_session([{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])
        # Sanity: a well-formed request still works through _run_session.
        self.assertIn("result", resp)
        proc = subprocess.run(
            SERVER_CMD,
            input="{not json}\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        resp = json.loads(proc.stdout.strip().splitlines()[0])
        self.assertEqual(resp["error"]["code"], -32700)
        self.assertIsNone(resp["id"])

    def test_prompt_size_cap(self):
        (resp,) = _run_session(
            [
                _req(
                    1,
                    "tools/call",
                    {
                        "name": "probe_surface_noise_stability",
                        "arguments": {
                            "prompt": "x" * (1 << 20 + 1),
                            "fixtures": [
                                {
                                    "id": "f1",
                                    "messages": [{"role": "user", "content": "x"}],
                                    "expected": "BLUE",
                                    "group": "probe",
                                }
                            ],
                            "boundary": [7],
                        },
                    },
                )
            ]
        )
        self.assertEqual(resp["error"]["code"], -32602)

    def test_fixture_group_defaults(self):
        # Fixtures without an explicit group are accepted (defaulted).
        (resp,) = _run_session(
            [
                _req(
                    1,
                    "tools/call",
                    {
                        "name": "probe_surface_noise_stability",
                        "arguments": {
                            "prompt": "MAGICWORD",
                            "fixtures": [
                                {
                                    "id": "f1",
                                    "messages": [{"role": "user", "content": "x"}],
                                    "expected": "BLUE",
                                }
                            ],
                            "boundary": [7],
                        },
                    },
                )
            ]
        )
        self.assertIn("result", resp)
        self.assertEqual(resp["result"]["structuredContent"]["status"], "ok")

    def test_batch_requests(self):
        responses = _run_session(
            [
                _req(1, "tools/list"),
                _req(2, "server/discover"),
                _req(3, "tools/call", {"name": "surface_noise_protocol_info", "arguments": {}}),
            ]
        )
        self.assertEqual([r["id"] for r in responses], [1, 2, 3])
        self.assertTrue(all("result" in r for r in responses))

    def test_finding8_falsy_backend_rejected(self):
        # "backend": [] (or "", 0, {}) must be -32602, not silent selftest.
        for bad in ([], "", 0, {}):
            responses = _run_session(
                [
                    _req(
                        1,
                        "tools/call",
                        {
                            "name": "probe_surface_noise_stability",
                            "arguments": {
                                "prompt": "x",
                                "fixtures": [
                                    {
                                        "id": "t0",
                                        "messages": [{"role": "user", "content": "x"}],
                                        "expected": "BLUE",
                                    }
                                ],
                                "boundary_ids": [7],
                                "backend": bad,
                            },
                        },
                    ),
                ]
            )
            self.assertIn("error", responses[0], f"backend={bad!r} not rejected")
            self.assertEqual(responses[0]["error"]["code"], -32602)


if __name__ == "__main__":
    unittest.main()


class MCPPosixGuardTests(unittest.TestCase):
    """Wave-6 red-team A finding 7: the SIGALRM timeout is POSIX-only."""

    def _load_server(self):
        # The shipped module, shared with the rest of the process:
        # callers that patch its globals must restore them (see the
        # SIGALRM test's addCleanup below).
        import importlib

        return importlib.import_module("model_protocol_lab.mcp_server")

    def test_probe_without_sigalrm_degrades_to_best_effort(self):
        # Simulate a platform without SIGALRM (Windows): the probe must
        # still run instead of raising AttributeError.
        import signal as real_signal

        server = self._load_server()
        # _load_server returns the shared module now, not a fresh
        # spec-loaded copy: restore the real signal module so later
        # tests in this process see unpatched globals.
        self.addCleanup(setattr, server, "signal", real_signal)

        class NoAlarmSignal:
            ITIMER_REAL = real_signal.ITIMER_REAL

            def __getattr__(self, name):
                if name == "SIGALRM":
                    raise AttributeError("no SIGALRM on this platform")
                return getattr(real_signal, name)

        server.signal = NoAlarmSignal()
        result = server.tool_probe(
            {
                "prompt": "MAGICWORD please",
                "fixtures": [
                    {
                        "id": "f1",
                        "messages": [{"role": "user", "content": "x"}],
                        "expected": "BLUE",
                        "group": "g",
                    }
                ],
                "boundary": [7],
            }
        )
        self.assertEqual(result["structuredContent"]["status"], "ok")

    def test_timeout_doc_mentions_posix_only(self):
        server = self._load_server()
        self.assertIn("POSIX-only", server.__doc__)
