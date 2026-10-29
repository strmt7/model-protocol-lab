"""Cycle 196: oversight-audit correction regressions.

Every test here pins a confirmed 2026-10-22 oversight finding
(security S-3..S-14, edge-case F2/F3, plus the needle HTML leak
check the auditor's own attempt left vacuous). Each fix was
re-verified by execution before landing; these tests keep the
guards engaged — a passing test whose guard never fired would be
vacuous, so several assert the pre-fix failure mode is now a
fixed-vocabulary outcome instead.
"""

import copy
import io
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

from model_protocol_lab import diffreports, fixtures, verify
from model_protocol_lab.cli import _display_id as _cli_display_id
from model_protocol_lab.pack import build_bundle_bytes, pack_bundle, verify_bundle_bytes
from model_protocol_lab.reporting import render_html
from tests.test_cycle188_dateline import _base_payload as _dateline_payload
from tests.test_cycle191_needle import _full_payload as _needle_full_payload
from tests.test_cycle191_needle import _payload as _needle_payload
from tests.test_cycle193_placement import _payload as _placement_payload

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
# The server ships as model_protocol_lab.mcp_server (cycle 208).
SERVER_CMD = [PYTHON, "-m", "model_protocol_lab.mcp_server"]


def _sealed_files():
    """Sealed files."""
    data, _manifest, _comment = pack_bundle(ROOT / "examples" / "sample-report", [])
    archive = zipfile.ZipFile(io.BytesIO(data))
    files = [(name, archive.read(name)) for name in archive.namelist() if name != "manifest.json"]
    manifest_text = archive.read("manifest.json").decode("utf-8")
    return data, manifest_text, files


def _schema_detail(report):
    """Schema detail."""
    for check in report["checks"]:
        if check["id"] == "seal_manifest_schema":
            return check["detail"]
    raise AssertionError("seal_manifest_schema check missing")


def _container_detail(report):
    """Container detail."""
    for check in report["checks"]:
        if check["id"] == "seal_container":
            return check["detail"]
    raise AssertionError("seal_container check missing")


class PackExceptionNormalisationTests(unittest.TestCase):
    """S-4 / S-6 / L-C: the verify boundary returns the documented
    manifest_unparseable report instead of leaking interpreter
    exceptions, and the container boundary names invalid UTF-8
    entry names in the fixed vocabulary.
    """

    def test_control_bundle_still_ok(self):
        """Test control bundle still ok."""
        data, _text, _files = _sealed_files()
        self.assertEqual(verify_bundle_bytes(data)["verdict"], "ok")

    def test_deeply_nested_manifest_is_unparseable_not_recursion(self):
        """Test deeply nested manifest is unparseable not recursion."""
        _data, text, files = _sealed_files()
        deep = "[" * 20000 + "]" * 20000
        tampered = (text.rstrip()[:-1] + ', "extra": ' + deep + "}").encode("utf-8")
        report = verify_bundle_bytes(build_bundle_bytes(tampered, files))
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_schema_detail(report), "manifest_unparseable")

    def test_over_limit_integer_is_unparseable(self):
        """Test over limit integer is unparseable."""
        _data, text, files = _sealed_files()
        tampered = (text.rstrip()[:-1] + ', "bignum": ' + "9" * 5000 + "}").encode("utf-8")
        report = verify_bundle_bytes(build_bundle_bytes(tampered, files))
        self.assertEqual(_schema_detail(report), "manifest_unparseable")

    def test_nonfinite_constant_is_unparseable(self):
        """Test nonfinite constant is unparseable."""
        _data, text, files = _sealed_files()
        tampered = (text.rstrip()[:-1] + ', "extra": NaN}').encode("utf-8")
        report = verify_bundle_bytes(build_bundle_bytes(tampered, files))
        self.assertEqual(_schema_detail(report), "manifest_unparseable")

    def test_invalid_utf8_entry_name_is_container_invalid(self):
        """Test invalid utf8 entry name is container invalid."""
        name = b"\xff\xfe"
        payload = b"x"
        crc = zlib.crc32(payload)
        local = (
            struct.pack(
                "<IHHHHHIIIHH",
                0x04034B50,
                20,
                0x800,
                0,
                0,
                0,
                crc,
                len(payload),
                len(payload),
                len(name),
                0,
            )
            + name
            + payload
        )
        central = (
            struct.pack(
                "<IHHHHHHIIIHHHHHII",
                0x02014B50,
                20,
                20,
                0x800,
                0,
                0,
                0,
                crc,
                len(payload),
                len(payload),
                len(name),
                0,
                0,
                0,
                0,
                0,
                0,
            )
            + name
        )
        eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(central), len(local), 0)
        with self.assertRaises(ValueError) as ctx:
            verify_bundle_bytes(local + central + eocd)
        self.assertEqual(str(ctx.exception), "bundle_container_invalid")


class PackEntryTypeTests(unittest.TestCase):
    """S-8: an entry whose stored Unix file type is explicitly not
    a regular file fails seal_container; pack's own shape (no type
    bits claimed) and explicitly regular entries still pass.
    """

    @staticmethod
    def _retyped(target, mode_bits):
        """Retyped."""
        data, _text, _files = _sealed_files()
        source = zipfile.ZipFile(io.BytesIO(data))
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as bundle:
            for info in source.infolist():
                content = source.read(info.filename)
                entry = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                entry.compress_type = zipfile.ZIP_DEFLATED
                if info.filename == target:
                    entry.create_system = 3
                    entry.external_attr = (mode_bits | 0o777) << 16
                bundle.writestr(entry, content)
            bundle.comment = source.comment
        return out.getvalue()

    def test_symlink_typed_entry_fails_container(self):
        """Test symlink typed entry fails container."""
        report = verify_bundle_bytes(self._retyped("report.html", 0o120000))
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_container_detail(report), "entry_type_not_regular")

    def test_fifo_typed_entry_fails_container(self):
        """Test fifo typed entry fails container."""
        report = verify_bundle_bytes(self._retyped("report.html", 0o010000))
        self.assertEqual(_container_detail(report), "entry_type_not_regular")

    def test_regular_typed_entry_passes(self):
        """Test regular typed entry passes."""
        report = verify_bundle_bytes(self._retyped("report.html", 0o100000))
        self.assertEqual(report["verdict"], "ok")


class PackSurrogateNameTests(unittest.TestCase):
    """S-10: a surrogateescape filename in a report directory is a
    pack_entry_name_invalid ValueError, not a raw UnicodeEncodeError
    during serialisation.
    """

    def test_surrogateescape_filename_rejected_in_vocabulary(self):
        """Test surrogateescape filename rejected in vocabulary."""
        with tempfile.TemporaryDirectory() as tmp:
            fd = os.open(os.fsencode(tmp) + b"/bad\xffname.json", os.O_CREAT | os.O_WRONLY)
            os.close(fd)
            Path(tmp, "results.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                pack_bundle(Path(tmp), [])
            self.assertEqual(str(ctx.exception), "pack_entry_name_invalid")


class DisplayCapTests(unittest.TestCase):
    """S-9: the R4-7 200-character preview cap now lives in every
    display sanitizer (verify/cli/diffreports _display_id and
    fixtures._safe), not just at the sites the earlier round fixed.
    """

    def test_caps(self):
        """Test caps."""
        long_value = "x" * 300
        for helper in (
            verify._display_id,
            _cli_display_id,
            diffreports._display_id,
            fixtures._safe,
        ):
            capped = helper(long_value)
            self.assertLessEqual(len(capped), 203, helper.__module__)
            self.assertTrue(capped.endswith("..."), helper.__module__)
            self.assertEqual(helper("short"), "short", helper.__module__)


def _run_server(raw_input, timeout=60):
    """Run server."""
    proc = subprocess.run(
        SERVER_CMD, input=raw_input, capture_output=True, timeout=timeout, check=False
    )
    responses = [
        json.loads(line) for line in proc.stdout.decode("utf-8").splitlines() if line.strip()
    ]
    return proc.returncode, responses


_PING = json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping"})
_FIXTURES = [
    {
        "id": "f1",
        "group": "g",
        "messages": [{"role": "user", "content": "Capital of France?"}],
        "expected": "Paris",
    }
]


def _tool_call(arguments):
    """Tool call."""
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "probe_surface_noise_stability", "arguments": arguments},
        }
    )


class MCPFrameHardeningTests(unittest.TestCase):
    """S-3 / S-5 / S-11 / S-12 / S-13: the MCP frame boundary fails
    closed in the fixed JSON-RPC vocabulary and the process
    survives every malformed frame class.
    """

    def test_nested_bomb_is_parse_error_and_server_survives(self):
        """Test nested bomb is parse error and server survives."""
        rc, out = _run_server(("[" * 10000 + "]" * 10000 + "\n" + _PING + "\n").encode())
        self.assertEqual(rc, 0)
        self.assertEqual(out[0]["error"]["code"], -32700)
        self.assertTrue(any(entry.get("id") == 99 for entry in out))

    def test_overlong_line_is_parse_error_and_framing_resyncs(self):
        """Test overlong line is parse error and framing resyncs."""
        big = "x" * (16 * 1024 * 1024 + 100)
        rc, out = _run_server((big + "\n" + _PING + "\n").encode())
        self.assertEqual(rc, 0)
        self.assertEqual(out[0]["error"]["code"], -32700)
        self.assertTrue(any(entry.get("id") == 99 for entry in out))

    def test_duplicate_keys_are_parse_error(self):
        """Test duplicate keys are parse error."""
        rc, out = _run_server(b'{"jsonrpc":"2.0","id":1,"id":2,"method":"ping"}\n')
        self.assertEqual(rc, 0)
        self.assertEqual(out[0]["error"]["code"], -32700)

    def test_notification_never_answered(self):
        """Test notification never answered."""
        notification = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "surface_noise_protocol_info", "arguments": {}},
            }
        )
        rc, out = _run_server((notification + "\n" + _PING + "\n").encode())
        self.assertEqual(rc, 0)
        self.assertEqual([entry.get("id") for entry in out], [99])

    def test_null_id_errors_are_reported(self):
        """Test null id errors are reported."""
        frame = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": None,
                "method": "tools/call",
                "params": {"name": "nope", "arguments": {}},
            }
        )
        _rc, out = _run_server((frame + "\n").encode())
        self.assertEqual(out[0]["error"]["code"], -32602)

    def test_params_as_list_is_invalid_params(self):
        """Test params as list is invalid params."""
        frame = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": [1, 2]})
        _rc, out = _run_server((frame + "\n").encode())
        self.assertEqual(out[0]["error"]["code"], -32602)

    def test_boundary_parity_with_cli(self):
        """Test boundary parity with cli."""
        for boundary in ([7] * 17, [2**100], [7] * 200000):
            frame = _tool_call({"prompt": "x", "fixtures": _FIXTURES, "boundary": boundary})
            _rc, out = _run_server((frame + "\n").encode())
            self.assertEqual(out[0]["error"]["code"], -32602, boundary[:2])
            self.assertIn("boundary", out[0]["error"]["message"])

    def test_tool_failure_is_fixed_internal_error(self):
        """Test tool failure is fixed internal error."""
        frame = _tool_call({"prompt": "x\udcff", "fixtures": _FIXTURES, "boundary": [7]})
        _rc, out = _run_server((frame + "\n").encode())
        self.assertEqual(out[0]["error"]["code"], -32603)
        self.assertEqual(out[0]["error"]["message"], "internal error")


class ProbeOutputPermissionTests(unittest.TestCase):
    """S-7: surface-noise probe output follows the save_report
    permission discipline (0700 directory, 0600 files) under a
    permissive umask.
    """

    def test_probe_output_permissions(self):
        """Test probe output permissions."""
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp, "prompt.txt")
            prompt.write_text("MAGICWORD please", encoding="utf-8")
            out_dir = Path(tmp, "probe-out")
            old_umask = os.umask(0o022)
            try:
                proc = subprocess.run(
                    [
                        PYTHON,
                        "-m",
                        "model_protocol_lab",
                        "surface-noise",
                        "probe",
                        "--prompt",
                        str(prompt),
                        "--fixtures",
                        str(ROOT / "examples" / "fixtures.json"),
                        "--boundary-ids",
                        "7",
                        "--output",
                        str(out_dir),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=ROOT,
                    check=False,
                )
            finally:
                os.umask(old_umask)
            self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
            self.assertEqual(stat.S_IMODE(os.stat(out_dir).st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(os.stat(out_dir / "probe.json").st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(os.stat(out_dir / "dashboard.html").st_mode), 0o600)


class ReportControlStripTests(unittest.TestCase):
    """S-14: C0 controls planted in candidate names / fixture IDs
    do not reach report.html verbatim through the main rendering
    paths.
    """

    def test_main_paths_strip_controls(self):
        """Test main paths strip controls."""
        result = json.loads(
            (ROOT / "examples" / "sample-report" / "results.json").read_text(encoding="utf-8")
        )
        evil_name = "cand\x00\x1b[31m\rX"
        evil_fid = "fix\x00\x07Y"

        def plant(obj):
            """Plant."""
            if isinstance(obj, dict):
                return {
                    (
                        evil_name
                        if key == "chatml"
                        else evil_fid
                        if isinstance(key, str) and key.startswith("demo-")
                        else key
                    ): plant(value)
                    for key, value in obj.items()
                }
            if isinstance(obj, list):
                return [plant(value) for value in obj]
            if isinstance(obj, str):
                return obj.replace("chatml", evil_name)
            return obj

        html = render_html(plant(copy.deepcopy(result)))
        self.assertNotIn("\x00", html)
        self.assertNotIn("\x1b", html)
        self.assertNotIn("\r", html)
        self.assertIn("cand", html)


class BoolForIntGuardTests(unittest.TestCase):
    """Edge F2: a JSON true must not validate as the integer 1 in
    equality-compared validator fields.
    """

    def test_needle_protocol_version_bool_rejected(self):
        """Test needle protocol version bool rejected."""
        payload = _needle_payload()
        self.assertTrue(diffreports._valid_needle(payload))
        payload["protocol"]["version"] = True
        self.assertFalse(diffreports._valid_needle(payload))

    def test_needle_n_cases_bool_rejected(self):
        """Test needle n cases bool rejected."""
        payload = _needle_payload()
        payload["n_cases"] = True
        self.assertFalse(diffreports._valid_needle(payload))

    def test_placement_protocol_version_bool_rejected(self):
        """Test placement protocol version bool rejected."""
        payload = _placement_payload()
        self.assertTrue(diffreports._valid_placement(payload))
        payload["protocol"]["version"] = True
        self.assertFalse(diffreports._valid_placement(payload))

    def test_dateline_n_multi_turn_bool_rejected(self):
        """Test dateline n multi turn bool rejected."""
        payload = _dateline_payload()
        self.assertTrue(diffreports._valid_dateline(payload))
        payload["n_multi_turn_fixtures"] = True
        self.assertFalse(diffreports._valid_dateline(payload))


class PrimaryPairBindingTests(unittest.TestCase):
    """Edge F3: primary_decision/primary_winner are bound to the
    run-decision vocabulary and winner consistency.
    """

    def test_genuine_payload_valid(self):
        """Test genuine payload valid."""
        self.assertTrue(diffreports._valid_dateline(_dateline_payload()))

    def test_garbage_decision_rejected(self):
        """Test garbage decision rejected."""
        payload = _dateline_payload()
        payload["primary_decision"] = "selected"
        self.assertFalse(diffreports._valid_dateline(payload))

    def test_surrogate_suffixed_decision_rejected(self):
        """Test surrogate suffixed decision rejected."""
        payload = _dateline_payload()
        payload["primary_decision"] = "compatible_on_fixture_set\ud800"
        self.assertFalse(diffreports._valid_dateline(payload))

    def test_abstention_with_winner_rejected(self):
        """Test abstention with winner rejected."""
        payload = _dateline_payload()
        payload["primary_decision"] = "ambiguous"
        self.assertFalse(diffreports._valid_dateline(payload))

    def test_unknown_winner_rejected(self):
        """Test unknown winner rejected."""
        payload = _dateline_payload()
        payload["primary_winner"] = "not-a-candidate"
        self.assertFalse(diffreports._valid_dateline(payload))

    def test_winner_class_without_winner_rejected(self):
        """Test winner class without winner rejected."""
        payload = _dateline_payload()
        payload["primary_winner"] = None
        self.assertFalse(diffreports._valid_dateline(payload))


class NeedleHtmlLeakTests(unittest.TestCase):
    """The needle-section leak check the oversight auditor's
    attempt left vacuous: render a genuine full payload through
    render_html (section provably present), with a marker planted
    in free-text fields a leaky renderer might emit.
    """

    MARKER = "XYZZY-NEEDLE-VALUE-7749"

    def test_planted_needle_value_never_rendered(self):
        """Test planted needle value never rendered."""
        payload = _needle_full_payload()
        html = render_html({"needle": payload})
        self.assertIn("Dose-0", html)  # the section actually rendered
        planted = copy.deepcopy(payload)
        for entry in planted["candidates"].values():
            if isinstance(entry, dict):
                entry["needle_value"] = self.MARKER
                entry["expected_answer"] = self.MARKER
        planted["protocol"]["needle_text"] = self.MARKER
        self.assertNotIn(self.MARKER, render_html({"needle": planted}))

    def test_hostile_candidate_name_only_escaped(self):
        """Test hostile candidate name only escaped."""
        payload = _needle_full_payload()
        name = next(iter(payload["candidates"]))
        payload["candidates"]["evil<script>" + self.MARKER] = payload["candidates"].pop(name)
        html = render_html({"needle": payload})
        self.assertNotIn("evil<script>", html)
        self.assertIn("evil&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
