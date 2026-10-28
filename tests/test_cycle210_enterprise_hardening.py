"""Cycle 210: enterprise-hardening surface.

Every assertion reads the shipped file, so a test passes only when the
hardening artifact itself is present with the required content. No
test here executes a workflow; the workflows are verified by their own
first runs on main after the push (CodeQL, Scorecard, gitleaks, smoke),
and their results are reported in the cycle closure, not asserted here.
"""

import re
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _load_yaml(path):
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _checkout_steps(doc):
    steps = []
    for job in doc["jobs"].values():
        for step in job.get("steps", []):
            uses = step.get("uses", "")
            if isinstance(uses, str) and uses.startswith("actions/checkout@"):
                steps.append(step)
    return steps


class TestWorkflowHygiene(unittest.TestCase):
    def test_ci_top_level_permissions(self):
        doc = _load_yaml(WORKFLOWS / "ci.yml")
        self.assertEqual(doc.get("permissions"), {"contents": "read"})
        # The test job's own block replaces the top-level one; its
        # Codecov OIDC grant must survive the tightening.
        self.assertEqual(doc["jobs"]["test"]["permissions"].get("id-token"), "write")

    def test_ci_still_four_jobs(self):
        doc = _load_yaml(WORKFLOWS / "ci.yml")
        self.assertEqual(len(doc["jobs"]), 4)

    def test_all_checkouts_persist_credentials_false(self):
        for name in ("ci.yml", "release.yml"):
            doc = _load_yaml(WORKFLOWS / name)
            steps = _checkout_steps(doc)
            self.assertGreater(len(steps), 0, name)
            for step in steps:
                self.assertIs(
                    step.get("with", {}).get("persist-credentials"),
                    False,
                    f"{name}: {step.get('uses')}",
                )

    def test_every_job_has_timeout(self):
        for name in ("ci.yml", "release.yml"):
            doc = _load_yaml(WORKFLOWS / name)
            for job_name, job in doc["jobs"].items():
                timeout = job.get("timeout-minutes")
                self.assertIsInstance(timeout, int, f"{name}/{job_name}")
                self.assertGreater(timeout, 0, f"{name}/{job_name}")

    def test_new_workflows_parse(self):
        for name in ("codeql.yml", "scorecard.yml", "gitleaks.yml", "smoke.yml"):
            doc = _load_yaml(WORKFLOWS / name)
            self.assertIn("jobs", doc, name)
            for job in doc["jobs"].values():
                self.assertIn("timeout-minutes", job, name)

    def test_codeql_python_and_schedule(self):
        doc = _load_yaml(WORKFLOWS / "codeql.yml")
        text = (WORKFLOWS / "codeql.yml").read_text(encoding="utf-8")
        self.assertIn("languages: python", text)
        self.assertIn("schedule:", text)
        # Cycle 215: the grant moved into the analyze job (zizmor
        # excessive-permissions — a workflow-level write grant is
        # inherited by every future job added to the file). Through
        # cycle 214 this pin asserted the top-level block.
        self.assertEqual(doc["jobs"]["analyze"]["permissions"].get("security-events"), "write")
        self.assertNotEqual(doc["permissions"].get("security-events"), "write")

    def test_scorecard_publishes_results(self):
        doc = _load_yaml(WORKFLOWS / "scorecard.yml")
        steps = doc["jobs"]["analysis"]["steps"]
        scorecard = [
            s for s in steps if str(s.get("uses", "")).startswith("ossf/scorecard-action@")
        ]
        self.assertEqual(len(scorecard), 1)
        self.assertIs(scorecard[0]["with"].get("publish_results"), True)

    def test_gitleaks_pinned_binary(self):
        text = (WORKFLOWS / "gitleaks.yml").read_text(encoding="utf-8")
        self.assertIn('GITLEAKS_VERSION: "8.30.1"', text)
        self.assertIn(
            "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb",
            text,
        )
        self.assertIn("sha256sum -c -", text)

    def test_smoke_matrix_and_scope(self):
        doc = _load_yaml(WORKFLOWS / "smoke.yml")
        job = doc["jobs"]["smoke"]
        self.assertEqual(
            sorted(job["strategy"]["matrix"]["os"]),
            ["macos-latest", "ubuntu-latest", "windows-latest"],
        )
        runs = " ".join(str(s.get("run", "")) for s in job["steps"])
        self.assertIn("model-protocol-lab catalog", runs)
        self.assertIn("model-protocol-lab inspect", runs)
        self.assertNotIn("unittest", runs)
        self.assertNotIn("pytest", runs)
        self.assertNotIn("--self-test", runs)


class TestCommunityFiles(unittest.TestCase):
    def test_codeowners(self):
        text = (REPO_ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
        self.assertIn("* @strmt7", text)
        for path in (
            "backends.py",
            "reporting.py",
            ".github/workflows/",
            "release_build.py",
            "docs/SECURITY.md",
            "pyproject.toml",
        ):
            self.assertIn(path, text)
        # Documentary, not enforcement: the header must say so.
        self.assertIn("documentary", text.lower())

    def test_issue_template_config(self):
        doc = _load_yaml(REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml")
        self.assertIs(doc.get("blank_issues_enabled"), False)
        links = doc.get("contact_links", [])
        self.assertGreater(len(links), 0)
        self.assertTrue(any("advisories" in str(link.get("url", "")) for link in links))

    def test_support_policy(self):
        text = (REPO_ROOT / "SUPPORT.md").read_text(encoding="utf-8")
        self.assertIn("best-effort", text)
        self.assertIn("no SLA", text)
        self.assertIn("docs/SECURITY.md", text)

    def test_root_security_is_a_pointer(self):
        text = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
        self.assertIn("docs/SECURITY.md", text)
        self.assertIn("pointer", text.lower())
        self.assertTrue((REPO_ROOT / "docs" / "SECURITY.md").exists())

    def test_editorconfig_and_gitattributes(self):
        editor = (REPO_ROOT / ".editorconfig").read_text(encoding="utf-8")
        self.assertIn("indent_size = 4", editor)
        self.assertIn("end_of_line = lf", editor)
        attrs = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("linguist-generated", attrs)
        self.assertIn("examples/sample-report", attrs)

    def test_typing_classifier(self):
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"Typing :: Typed"', text)

    def test_dependabot_cooldown_and_groups(self):
        doc = _load_yaml(REPO_ROOT / ".github" / "dependabot.yml")
        # Cycle 217 added the pip entry for tools/requirements
        # (hash-pinned workflow installs); the cooldown/groups
        # discipline below applies to all four updates.
        self.assertEqual(len(doc["updates"]), 4)
        for update in doc["updates"]:
            self.assertEqual(update.get("cooldown", {}).get("default-days"), 7)
            self.assertTrue(update.get("groups"))

    def test_gitleaks_config(self):
        with (REPO_ROOT / ".gitleaks.toml").open("rb") as handle:
            doc = tomllib.load(handle)
        patterns = doc.get("allowlist", {}).get("regexes", [])
        self.assertEqual(len(patterns), 2)
        for pattern in patterns:
            re.compile(pattern)


class TestReadmeSurface(unittest.TestCase):
    def test_dev_tool_badges_dropped_and_scorecard_added(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        for badge in ("[![cocoindex-code]", "[![caveman]", "[![crawl4ai]"):
            self.assertNotIn(badge, readme)
        self.assertIn("api.securityscorecards.dev", readme)
        # The Codecov badge pin survives: still directly after CI.
        self.assertLess(readme.index("[![CI]"), readme.index("[![Codecov]"))

    def test_toc_and_expected_output(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("## Contents", readme)
        self.assertIn("[Quick start](#quick-start)", readme)
        self.assertIn("Expected output", readme)
        self.assertIn(
            '{"search_cases": 2, "held_out_cases": 2, "valid": true, '
            '"contamination": {"worst_tier": "clean", "pairs": []}}',
            readme,
        )
        self.assertIn('"name": "chatml"', readme)

    def test_contributing_link(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("[CONTRIBUTING.md](CONTRIBUTING.md)", readme)

    def test_sbom_decision_documented(self):
        text = (REPO_ROOT / "docs" / "SECURITY.md").read_text(encoding="utf-8")
        self.assertIn("Why no SBOM is shipped (cycle 210 adjudication)", text)


if __name__ == "__main__":
    unittest.main()
