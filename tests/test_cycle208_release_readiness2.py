"""Cycle 208: release-readiness-2 pins (wave-10 enterprise MUSTs/SHOULDs).

Every assertion here pins a cycle-208 change at the level the tree
already pins such things: workflow text (cf. test_cycle201/
test_cycle202 reading release.yml as data), console-script and
module presence, README quickstart order, template text, and doc
sections. Behavioral pins (the MCP server actually serving) live in
the cycle-165/196/202 suites, repointed this cycle at the shipped
module; this file pins the packaging and documentation surface.
"""

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
PYPROJECT = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
SECURITY = (REPO_ROOT / "docs" / "SECURITY.md").read_text(encoding="utf-8")
USAGE = (REPO_ROOT / "docs" / "USAGE.md").read_text(encoding="utf-8")
VALIDATION = (REPO_ROOT / "VALIDATION.md").read_text(encoding="utf-8")
BUG_TEMPLATE = (REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").read_text(
    encoding="utf-8"
)
TOOLS_SERVER = REPO_ROOT / "tools" / "surface_noise_mcp_server.py"


def _quickstart_section() -> str:
    start = README.index("## Quick start")
    rest = README[start:]
    next_heading = rest.index("\n## ", 1)
    return rest[:next_heading]


class ReleaseWorkflowTests(unittest.TestCase):
    def test_build_job_smoke_tests_installed_wheel(self) -> None:
        # M1: the build job installs the wheel into a fresh venv and
        # runs the zero-dependency commands plus the MCP self-test.
        self.assertIn("python -m venv", WORKFLOW)
        self.assertIn("dist/*.whl", WORKFLOW)
        self.assertIn("model-protocol-lab catalog", WORKFLOW)
        self.assertIn("model-protocol-lab inspect", WORKFLOW)
        self.assertIn("model-protocol-lab-mcp --self-test", WORKFLOW)

    def test_epoch_derived_from_tagged_commit(self) -> None:
        # S4: no fixed epoch in the workflow; it comes from git.
        self.assertIn("git log -1 --format=%ct", WORKFLOW)
        self.assertNotIn("1700000000", WORKFLOW)

    def test_release_notes_file_wired_with_fallback(self) -> None:
        # S6: versioned notes file used when present.
        self.assertIn("--notes-file", WORKFLOW)
        self.assertIn("RELEASE_NOTES_", WORKFLOW)

    def test_workflow_stays_tag_gated(self) -> None:
        # The trigger must not have widened: release only on tags.
        self.assertIn("tags:", WORKFLOW)
        self.assertNotIn("workflow_dispatch", WORKFLOW)
        self.assertNotIn("branches:", WORKFLOW)


class McpServerPackagingTests(unittest.TestCase):
    def test_console_script_entry_point_declared(self) -> None:
        self.assertIn("model-protocol-lab-mcp", PYPROJECT)
        self.assertIn("model_protocol_lab.mcp_server:serve", PYPROJECT)

    def test_shipped_module_defines_server_entry_points(self) -> None:
        import model_protocol_lab.mcp_server as server

        for name in ("serve", "self_test", "handle_request", "tool_probe"):
            self.assertTrue(callable(getattr(server, name, None)), name)

    def test_module_self_test_exits_zero(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "model_protocol_lab.mcp_server", "--self-test"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-1000:])
        self.assertIn("self-test: OK", proc.stderr)

    def test_checkout_launcher_still_works(self) -> None:
        # The tools/ launcher must keep working from a checkout.
        proc = subprocess.run(
            [sys.executable, str(TOOLS_SERVER), "--self-test"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-1000:])
        self.assertIn("self-test: OK", proc.stderr)


class QuickstartAndTemplateTests(unittest.TestCase):
    def test_quickstart_installs_base_first(self) -> None:
        # M3: the first install in Quick start is the base install;
        # the heavy stacks come only after the zero-dep commands.
        section = _quickstart_section()
        installs = [
            line.strip()
            for line in section.splitlines()
            if line.strip().startswith("python -m pip install")
        ]
        self.assertTrue(installs, "no pip install line in Quick start")
        self.assertEqual(installs[0], "python -m pip install -e .")
        self.assertIn('python -m pip install -e ".[hf]"', installs)
        self.assertIn("CONTRIBUTING", section)

    def test_bug_template_prints_real_version(self) -> None:
        # S10: the version command works and the placeholder is real.
        self.assertIn("print(model_protocol_lab.__version__)", BUG_TEMPLATE)
        self.assertNotIn("0.1.0a0", BUG_TEMPLATE)


class DocsPolicyTests(unittest.TestCase):
    def test_security_supported_versions_section(self) -> None:
        # S1: latest release line only, no invented SLA.
        self.assertIn("## Supported versions", SECURITY)
        self.assertIn("latest release line only", SECURITY)

    def test_security_verifying_release_section(self) -> None:
        # S5: attestation verification + double-build reproduction.
        self.assertIn("## Verifying a release", SECURITY)
        self.assertIn("gh attestation verify", SECURITY)
        self.assertIn("release_build.py", SECURITY)
        # The base-wheel claim is precise: Requires-Dist lines exist
        # but are all extra-gated, so a plain install pulls nothing.
        self.assertIn("every `Requires-Dist` line is gated on an extra", SECURITY)

    def test_platform_statement_in_readme_and_validation(self) -> None:
        # S2: CI tests Linux x86-64; other platforms untested.
        self.assertIn("CI tests Linux x86-64", README)
        top = VALIDATION.split("\n## ")[1]
        self.assertIn("CI tests Linux x86-64", top)

    def test_usage_exit_code_table_and_contracts(self) -> None:
        # S7/S8/S9: one table, the output contract, schema_version.
        section6 = USAGE.split("## 6. Read the outputs")[1].split("## 7.")[0]
        self.assertIn("Exit codes for every subcommand", section6)
        self.assertIn("surface-noise probe", section6)
        self.assertIn("2 cancelled; 3", section6)
        self.assertIn("stdout stays parseable", section6)
        self.assertIn('"schema_version": 1', section6)
        self.assertIn("unsupported_schema_version", section6)


if __name__ == "__main__":
    unittest.main()
