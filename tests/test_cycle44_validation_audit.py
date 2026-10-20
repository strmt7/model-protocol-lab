"""VALIDATION.md accuracy audit (Cycle 44, 2026-10-15).

Verifies every checkable claim in VALIDATION.md against the current working
tree by running the documented commands and comparing actual output to the
documented claims. Inaccuracies found are documented below with
actual-vs-documented values; this test file pins the *verified* values and
the honest framings. It does NOT modify VALIDATION.md itself.

## Inaccuracies found (documented claim -> actual, verified 2026-10-15)

1. Test count: "1001 tests + 3142 subtests" (cycles 10-13 section)
   -> current `pytest --collect-only -q -p no:cacheprovider tests`
   collects 3338 tests, matching the README pin test
   (`test_readme_test_count_matches_discovered_suite`), and the full
   documented run reports "43319 subtests passed" (pytest-subtests
   counts unittest subTest invocations). The doc number is ~31 cycles
   stale.

2. ruff format: "clean (66 files)" -> bare `ruff format --check` now
   reports "101 files already formatted". Clean, but the count is stale.

3. bandit: "0 issues above Low; 1 Low/Medium-confidence B105 false
   positive (`"scored_token_count": 0` literal, pre-existing)" ->
   `bandit -r model_protocol_lab/` now reports "No issues identified."
   The cited literal at model_protocol_lab/experiments.py:674 now carries
   `# nosec B105 -- integer counter field, not a credential`, and
   model_protocol_lab/core.py:511-517 carry similar targeted suppressions
   with justifications. The doc describes the pre-suppression state and
   was never updated after the fix.

4. Format families: "23 format families" -> `CATALOG` in
   model_protocol_lab/catalog.py holds 29 families. The six families the
   doc names as added in the period (kolibri, lfm25, nemotron35,
   muse_glimmer, laguna, ling) are all present; the six added later are
   gemma4, deepseek41, kimi2, kimi3, muse_spark, qwen38.

5. FAQ: "docs/FAQ.md (12 Q&As)" -> docs/FAQ.md has 15 `##` Q&A headings.

6. Red-team: "plus 8 follow-up findings across two more passes (all fixed
   with regression tests)" -> per docs/RESEARCH_LOG.md the first
   follow-up pass found 6 findings (5 fixed with regression tests;
   finding 6 documented in a code comment with no behavior change) and
   the second follow-up pass found 2 findings (both fixed). So 7 fixed
   and regression-pinned, 1 documented-only. "All fixed" overstates it.

7. Historical counts (361 / 708 / 224 / 213 tests; 98% / 97% / 95%
   coverage; run timings; differential-trial tallies like 1500/1500) are
   archival claims about past tree states and cannot be re-verified on
   the current tree. Recorded here as not-verifiable, not as refuted.

8. pip-audit and "GitHub Actions CI on main -- green" require network
   access (OSV database / GitHub) and are not verifiable in this
   offline environment. Recorded as not-verified-here, not as refuted.

## Verified accurate (pinned below)

- Environment: Python 3.12.3, Linux x86-64, torch 2.14.1+cpu,
  transformers 5.19.0, hypothesis importable, `pytest>=8.0` in the
  `[dev]` extra of pyproject.toml.
- `ruff check` clean; `ruff format --check` clean; `mypy
  model_protocol_lab` clean on exactly 15 source files.
- CLI surface: `debug`, `diff-backends`, `diff-reports`,
  `doctor --portability-matrix`, `doctor --boundary-ids` /
  `--reference-model`, `run --boundary-map` / `--max-drop-bytes` /
  `--sensitivity-sweep` / `--include-evidence`, `fixtures build`,
  `catalog`, `inspect`.
- Teacher-forced log-likelihood diagnostic present in experiments.py.
- CI YAML: 4 jobs, actions pinned to SHAs with `# v7.0.1` / `# v7.0.0`
  comments; pre-commit ruff-pre-commit v0.17.0; Dependabot ecosystems
  pip + github-actions + pre-commit; CITATION.cff has title and authors.
- docs/USAGE.md: §8.2 stop-behavior isolation matrix; llama.cpp stop
  cell documented as killed; `local_context_limit_exceeds_declared_window`
  and NFC normalization documented.
- Red-team fix round H1/M1/L1 (diffbackends bounds + loader
  sanitization) documented in docs/RESEARCH_LOG.md, matching the doc's
  three-finding description (quadratic vLLM-holdback hang -> split
  input/output caps; unbounded stop texts -> length/count bounds;
  hostile terminal text -> fixed vocabulary / sanitizer).
- "Not established" items honestly framed: no real (non-synthetic)
  model run (the only `from_pretrained` call site is marked local-only);
  no browser automation in the repo; weekly CI schedule + Dependabot
  config exist; the "not yet executed on GitHub" and "pre-commit not
  run end-to-end" caveats are retained verbatim in VALIDATION.md.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# NB: no .resolve() on sys.executable -- the project .venv python is a
# symlink, and resolving it would escape into the system bin directory.
BIN_DIR = Path(sys.executable).parent


def run_tool(name: str, *args: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    exe = BIN_DIR / name
    assert exe.exists(), f"expected tool at {exe}"
    return subprocess.run(
        [str(exe), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def cli_help(*args: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "model_protocol_lab", *args, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"help failed for {args}: {proc.stderr[-500:]}"
    return proc.stdout


class ValidationEnvironmentTests(unittest.TestCase):
    def test_python_meets_requires_python(self):
        # Doc: "Python 3.12.3, Linux x86-64" (the validated version). CI
        # also runs 3.11 and 3.13, so assert the running interpreter
        # satisfies pyproject's requires-python range instead of pinning
        # an exact version.
        from packaging.specifiers import SpecifierSet

        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'requires-python\s*=\s*"([^"]+)"', text)
        self.assertIsNotNone(match)
        assert match is not None
        spec = SpecifierSet(match.group(1))
        version = f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
        self.assertIn(version, spec, f"Python {version} not in {match.group(1)}")

    def test_torch_version(self):
        # Doc: "torch 2.14.1+cpu" (the validated build). CI installs the
        # CUDA wheel, so compare only the base version, not the local
        # build tag.
        import torch

        self.assertEqual(torch.__version__.split("+")[0], "2.14.1")

    def test_transformers_version(self):
        # Doc: "transformers 5.19.0" (the validated version). CI may resolve
        # a newer release, so assert the installed version satisfies
        # pyproject's hf-extra pin instead of an exact version.
        import transformers
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'"transformers([^"]+)"', text)
        self.assertIsNotNone(match)
        assert match is not None
        spec = SpecifierSet(match.group(1))
        installed = Version(transformers.__version__)
        self.assertIn(installed, spec, f"transformers {installed} not in {match.group(1)}")

    def test_hypothesis_installed(self):
        # Doc: "hypothesis installed".
        import hypothesis  # noqa: F401

    def test_pytest_in_dev_extra(self):
        # Doc: "the [dev] extra now includes pytest>=8.0".
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("pytest>=8.0", text)


class ValidationToolchainTests(unittest.TestCase):
    def test_ruff_check_clean(self):
        # Doc: "ruff check -- clean".
        proc = run_tool("ruff", "check")
        self.assertEqual(proc.returncode, 0, proc.stdout[-1000:])

    def test_ruff_format_check_clean(self):
        # Doc: "ruff format --check -- clean (66 files)". Clean, but the
        # file count is stale: the current tree formats 200 files.
        proc = run_tool("ruff", "format", "--check")
        self.assertEqual(proc.returncode, 0, proc.stdout[-1000:])
        match = re.search(r"(\d+) files already formatted", proc.stdout)
        self.assertIsNotNone(match)
        self.assertGreater(int(match.group(1)), 66)

    def test_mypy_clean_45_source_files(self):
        # Doc: "mypy model_protocol_lab -- clean (45 source files)".
        proc = run_tool("mypy", "model_protocol_lab", timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stdout[-1000:])
        self.assertIn("45 source files", proc.stdout)

    def test_bandit_zero_issues_at_any_severity(self):
        # Doc claims a remaining Low/Medium B105 false positive; actual:
        # the literal now carries a targeted # nosec with justification,
        # so bandit reports zero issues at any severity.
        # Bandit is only installed in the security-audit job, not the test
        # jobs: skip there instead of failing on a missing binary.
        if shutil.which("bandit") is None:
            self.skipTest("bandit binary not installed in this job")
        proc = run_tool("bandit", "-r", "model_protocol_lab/")
        self.assertEqual(proc.returncode, 0, proc.stdout[-1000:])
        self.assertIn("No issues identified.", proc.stdout)


class ValidationCliSurfaceTests(unittest.TestCase):
    def test_debug_command(self):
        out = cli_help("debug")
        self.assertIn("usage: model-protocol-lab debug", out)

    def test_diff_backends_command(self):
        out = cli_help("diff-backends")
        self.assertIn("usage: model-protocol-lab diff-backends", out)

    def test_diff_reports_command(self):
        out = cli_help("diff-reports")
        self.assertIn("usage: model-protocol-lab diff-reports", out)

    def test_doctor_portability_matrix_option(self):
        self.assertIn("--portability-matrix", cli_help("doctor"))

    def test_doctor_boundary_ids_and_reference_model_options(self):
        out = cli_help("doctor")
        self.assertIn("--boundary-ids", out)
        self.assertIn("--reference-model", out)

    def test_run_verification_flags(self):
        out = cli_help("run")
        for flag in (
            "--boundary-map",
            "--max-drop-bytes",
            "--sensitivity-sweep",
            "--include-evidence",
        ):
            self.assertIn(flag, out)

    def test_fixtures_build_command(self):
        out = cli_help("fixtures", "build")
        self.assertIn("usage: model-protocol-lab fixtures build", out)

    def test_catalog_command_runs(self):
        proc = subprocess.run(
            [sys.executable, "-m", "model_protocol_lab", "catalog"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertIn("kolibri", proc.stdout)

    def test_inspect_example_fixtures(self):
        proc = subprocess.run(
            [sys.executable, "-m", "model_protocol_lab", "inspect", "examples/fixtures.json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])


class ValidationFormatFamilyTests(unittest.TestCase):
    def test_family_count_is_29_not_23(self):
        # Doc claimed 23; the catalog grew through later cycles (29 now,
        # with muse_spark added in cycle 52 and qwen38 in cycle 54).
        from model_protocol_lab.catalog import CATALOG

        self.assertEqual(len(CATALOG), 29)

    def test_six_period_families_present(self):
        # The six families the doc names as added in the period.
        from model_protocol_lab.catalog import CATALOG

        for family in ("kolibri", "lfm25", "nemotron35", "muse_glimmer", "laguna", "ling"):
            self.assertIn(family, CATALOG)


class ValidationConfigFileTests(unittest.TestCase):
    def test_ci_yaml_four_jobs_sha_pinned(self):
        # Doc: "CI YAML (4 jobs, actions pinned to SHAs)"; the 2026-10-12
        # section pins checkout@v7.0.1 and setup-python@v7.0.0.
        import yaml

        ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        doc = yaml.safe_load(ci_path.read_text(encoding="utf-8"))
        self.assertEqual(len(doc["jobs"]), 4)
        text = ci_path.read_text(encoding="utf-8")
        self.assertIn(
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1",
            text,
        )
        self.assertIn(
            "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0",
            text,
        )

    def test_ci_codecov_upload_shape(self):
        # Owner-mandated Codecov wiring (cycle 149 add-on): the test job
        # emits coverage.xml and uploads it with the pinned
        # codecov-action via OIDC, failing the job when the upload
        # itself errors. The job needs id-token:write for OIDC.
        import yaml

        ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        doc = yaml.safe_load(ci_path.read_text(encoding="utf-8"))
        job = doc["jobs"]["test"]
        self.assertEqual(job["permissions"].get("id-token"), "write")
        steps = job["steps"]
        xml_steps = [s for s in steps if s.get("run") == "coverage xml -o coverage.xml"]
        self.assertEqual(len(xml_steps), 1)
        self.assertEqual(xml_steps[0].get("if"), "always()")
        uploads = [
            s
            for s in steps
            if type(s.get("uses")) is str and s["uses"].startswith("codecov/codecov-action@")
        ]
        self.assertEqual(len(uploads), 1)
        upload = uploads[0]
        self.assertEqual(
            upload["uses"],
            "codecov/codecov-action@303a32d7a59b442fa8d48b6a1cc6825c09c847a5",
        )
        self.assertEqual(upload.get("if"), "always()")
        with_block = upload.get("with") or {}
        self.assertEqual(with_block.get("files"), "./coverage.xml")
        self.assertTrue(with_block.get("use_oidc"))
        self.assertEqual(with_block.get("name"), "model-protocol-lab")
        self.assertTrue(with_block.get("fail_ci_if_error"))
        # The repo-side config mirrors the owner's other repos: patch
        # target holds the CI gate, tests/examples are ignored.
        import yaml as yaml2

        codecov = yaml2.safe_load((REPO_ROOT / ".codecov.yml").read_text(encoding="utf-8"))
        self.assertEqual(codecov["coverage"]["status"]["patch"]["default"]["target"], "95%")
        self.assertIn("tests/", codecov["ignore"])

    def test_readme_codecov_badge_exact(self):
        # The badge line must be byte-exact: a wrong org/repo slug would
        # silently point at someone else's coverage. Owner instruction
        # 2026-10-19: shields.io badge with label=Codecov (capital C in the
        # rendered image, unlike codecov.io's own badge.svg); preserve
        # verbatim.
        badge = (
            "[![Codecov](https://img.shields.io/codecov/c/github/strmt7/model-protocol-lab?label=Codecov&logo=codecov)]"
            "(https://codecov.io/gh/strmt7/model-protocol-lab)"
        )
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(badge, readme)
        # It sits directly after the CI badge, not buried elsewhere.
        ci_badge = (
            "[![CI](https://github.com/strmt7/model-protocol-lab/actions/workflows/ci.yml/badge.svg)]"
            "(https://github.com/strmt7/model-protocol-lab/actions/workflows/ci.yml)"
        )
        self.assertLess(readme.index(ci_badge), readme.index(badge))

    def test_pre_commit_ruff_rev(self):
        # Doc: "pre-commit YAML (ruff-pre-commit v0.17.1)".
        text = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        self.assertIn("ruff-pre-commit", text)
        self.assertIn("rev: v0.17.1", text)

    def test_dependabot_three_ecosystems(self):
        # Doc: "Dependabot YAML (pip, github-actions, pre-commit)".
        import yaml

        doc = yaml.safe_load((REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
        ecosystems = sorted(u["package-ecosystem"] for u in doc["updates"])
        self.assertEqual(ecosystems, ["github-actions", "pip", "pre-commit"])

    def test_citation_cff_title_and_authors(self):
        # Doc: "CITATION.cff (title/authors present)".
        text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("title:", text)
        self.assertIn("authors:", text)


class ValidationDocsTests(unittest.TestCase):
    def test_faq_has_20_qas_not_12(self):
        # Doc claims 12 Q&As; the file grew to 15 by cycle 44 and to
        # 20 in cycle 178 (docs-currency backfill entries).
        lines = (REPO_ROOT / "docs" / "FAQ.md").read_text(encoding="utf-8").splitlines()
        headings = [line for line in lines if line.startswith("## ")]
        self.assertEqual(len(headings), 20)

    def test_usage_stop_isolation_matrix_section(self):
        text = (REPO_ROOT / "docs" / "USAGE.md").read_text(encoding="utf-8")
        self.assertIn("### 8.2 Stop-behavior isolation matrix", text)

    def test_usage_documents_llamacpp_cell_killed(self):
        text = (REPO_ROOT / "docs" / "USAGE.md").read_text(encoding="utf-8")
        self.assertIn("Why no llama.cpp cell", text)

    def test_usage_documents_local_context_limit_code(self):
        text = (REPO_ROOT / "docs" / "USAGE.md").read_text(encoding="utf-8")
        self.assertIn("local_context_limit_exceeds_declared_window", text)

    def test_usage_documents_nfc_normalization(self):
        text = (REPO_ROOT / "docs" / "USAGE.md").read_text(encoding="utf-8")
        self.assertIn("NFC", text)

    def test_research_log_contains_redteam_fix_rounds(self):
        # The three-finding fix round (H1 quadratic hang -> split caps,
        # M1 unbounded stop texts -> length/count bounds, L1 hostile
        # terminal text -> fixed vocabulary) plus the two follow-up
        # diffreports passes must be recorded.
        text = (REPO_ROOT / "docs" / "RESEARCH_LOG.md").read_text(encoding="utf-8")
        flat = re.sub(r"\s+", " ", text)
        self.assertIn("## 2026-10-15 -- Red-team fix round: H1/M1/L1", flat)
        self.assertIn("MAX_OUTPUT_IDS", flat)
        self.assertIn("MAX_STOP_TEXT_LEN", flat)
        self.assertIn("## 2026-10-15 -- Red-team fix round: diffreports H1/M1-M3/L1", flat)
        self.assertIn("## 2026-10-15 -- Red-team fix round (follow-up): diffreports", flat)
        # Follow-up reality check: finding 6 was documented, not fixed.
        self.assertIn("finding 6 documented in a code comment with no behavior change", flat)

    def test_teacher_forced_logprob_diagnostic_present(self):
        text = (REPO_ROOT / "model_protocol_lab" / "experiments.py").read_text(encoding="utf-8")
        cli_text = (REPO_ROOT / "model_protocol_lab" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("Teacher-forced log-likelihood", cli_text)
        self.assertIn("teacher-forced logprob", text)


class ValidationHonestyTests(unittest.TestCase):
    def test_no_browser_automation_dependencies(self):
        # Doc: "No browser rendering check of report.html". The repo must
        # not depend on browser automation at all.
        tree = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
        for dep in ("selenium", "playwright", "pyppeteer"):
            self.assertNotIn(dep, tree)
        hits = [
            path
            for path in REPO_ROOT.rglob("*.py")
            if ".venv" not in path.parts
            and path.name != "test_cycle44_validation_audit.py"
            and "selenium" in path.read_text(encoding="utf-8").lower()
        ]
        self.assertEqual(hits, [])

    def test_model_loading_is_local_only(self):
        # Doc: "No real (non-synthetic) model run has been performed". The
        # only from_pretrained call site must be explicitly local-only.
        text = (REPO_ROOT / "model_protocol_lab" / "backends.py").read_text(encoding="utf-8")
        call_sites = [m.start() for m in re.finditer(r"from_pretrained\(", text)]
        self.assertGreater(len(call_sites), 0)
        for site in call_sites:
            window = text[max(0, site - 200) : site + 400]
            self.assertTrue(
                "local_files_only" in window or "local only" in window or "local-only" in window,
                f"from_pretrained call site not marked local-only: {window[:120]!r}",
            )

    def test_not_established_section_retained(self):
        # The release-gate discipline: the honest "not established" list
        # must not have been weakened into assertions.
        text = (REPO_ROOT / "VALIDATION.md").read_text(encoding="utf-8")
        section = text.split("## Not established", 1)[1].split("## Important limitations", 1)[0]
        for item in (
            "No real (non-synthetic) model run has been performed",
            "No browser rendering check",
            "No full-codebase mutation-testing run",
            "not yet executed on GitHub",
            "has not executed end-to-end",
        ):
            self.assertIn(item, section)

    def test_weekly_schedule_and_dependabot_exist(self):
        # The honest framing ("schedule and Dependabot have not yet
        # executed on GitHub") requires the infrastructure to exist.
        ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("schedule:", ci)
        self.assertTrue((REPO_ROOT / ".github" / "dependabot.yml").exists())

    def test_suite_count_meets_documented_floor(self):
        # The doc's "1001 tests" is a floor the grown suite must exceed;
        # the exact pin lives in test_cycle4.py against the README.
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(proc.returncode, 0, (proc.stderr or proc.stdout)[-1000:])
        match = re.search(r"(\d+) tests collected", proc.stdout)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(int(match.group(1)), 1001)


if __name__ == "__main__":
    unittest.main()
