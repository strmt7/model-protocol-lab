"""Cycle 215: enterprise polish 2 — zizmor lane, dependency review,
ADR index.

Every assertion reads the shipped file, so a test passes only when
the artifact itself is present with the required content (the
cycle-210/214 pattern). No test here executes a workflow; the zizmor
lane and the docs deployment are verified by their own first runs
on main after the push, and the dependency-review action by its
first pull request — those results are reported in the cycle
closure, not asserted here.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
DOCS = REPO_ROOT / "docs"

_MPL_SHORTHAND = re.compile(r"`mpl (?!:)")
_SHA_PIN = re.compile(r"^[0-9a-f]{40}$")
_MD_LINK = re.compile(r"\]\(([^)\s]+\.md)\)")

DEP_REVIEW_SHA = "a1d282b36b6f3519aa1f3fc636f609c47dddb294"


def _load_yaml(path):
    """Load a YAML file (the workflow under audit) with yaml.safe_load."""
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(doc):
    # PyYAML (YAML 1.1) parses the `on:` key as boolean True.
    """Return the workflow's trigger map (YAML parses the on key as True)."""
    return doc.get(True, doc.get("on", {}))


def _nav_targets(node):
    """Yield every string leaf of a parsed mkdocs nav structure."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _nav_targets(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _nav_targets(value)


def _all_uses(doc):
    """Collect every step's uses string across all jobs in the workflow."""
    for job in doc["jobs"].values():
        for step in job.get("steps", []):
            uses = step.get("uses")
            if isinstance(uses, str):
                yield step, uses


class TestZizmorLane(unittest.TestCase):
    """The zizmor workflow lane: pinned, hardened, suppression-free."""

    def setUp(self):
        """Load the zizmor workflow and its text."""
        self.path = WORKFLOWS / "zizmor.yml"
        self.doc = _load_yaml(self.path)
        self.text = self.path.read_text(encoding="utf-8")

    def test_lane_shape(self):
        """The zizmor lane has the frozen name, read-only permissions and job shape."""
        self.assertEqual(self.doc["name"], "Zizmor")
        self.assertEqual(self.doc.get("permissions"), {"contents": "read"})
        job = self.doc["jobs"]["zizmor"]
        self.assertEqual(job.get("name"), "zizmor (workflow security lint)")
        self.assertIsInstance(job.get("timeout-minutes"), int)
        triggers = _triggers(self.doc)
        self.assertEqual(triggers["push"]["branches"], ["main"])
        self.assertEqual(triggers["push"]["paths"], [".github/workflows/**"])
        self.assertEqual(triggers["pull_request"]["paths"], [".github/workflows/**"])

    def test_pinned_version_and_adjudicated_floor(self):
        # The exact command the lane runs; locally verified to exit 0
        # on the fixed tree (cycle-215 build log). Pedantic persona,
        # lowest severity/confidence floor, offline for determinism.
        # Cycle 217 moved the install to the hash-pinned closure, so
        # the pin lives in tools/requirements/zizmor.txt, not in the
        # workflow text.
        """Zizmor is pinned at 1.31.1 with the adjudicated findings floor."""
        self.assertIn(
            "python -m pip install --require-hashes -r tools/requirements/zizmor.txt",
            self.text,
        )
        zizmor_txt = (REPO_ROOT / "tools" / "requirements" / "zizmor.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("zizmor==1.31.1", zizmor_txt)
        self.assertIn(
            "zizmor --persona pedantic --min-severity low "
            "--min-confidence low --offline .github/workflows/",
            self.text,
        )

    def test_actions_sha_pinned_and_checkout_hardened(self):
        """Every action in the lane is SHA-pinned and checkout is hardened."""
        for step, uses in _all_uses(self.doc):
            self.assertTrue(_SHA_PIN.match(uses.rsplit("@", 1)[1]), uses)
            if uses.startswith("actions/checkout@"):
                self.assertIs(step.get("with", {}).get("persist-credentials"), False)

    def test_no_suppressions_anywhere(self):
        # docs/SECURITY.md states the no-suppression posture: no
        # zizmor config file and no inline ignore in any workflow.
        """No zizmor suppression config exists anywhere in the repo."""
        self.assertFalse((REPO_ROOT / "zizmor.yml").exists())
        self.assertFalse((REPO_ROOT / ".github" / "zizmor.yml").exists())
        for path in sorted(WORKFLOWS.glob("*.yml")):
            self.assertNotIn("zizmor: ignore", path.read_text(encoding="utf-8"), path.name)


class TestWorkflowPermissionScoping(unittest.TestCase):
    """The cycle-215 zizmor fixes, pinned so they cannot regress."""

    def test_no_top_level_write_grants_anywhere(self):
        """No workflow grants write permissions at the top level."""
        for path in sorted(WORKFLOWS.glob("*.yml")):
            doc = _load_yaml(path)
            for scope, level in (doc.get("permissions") or {}).items():
                self.assertEqual(level, "read", f"{path.name}: {scope}")

    def test_docs_pages_grants_are_job_level(self):
        # Docs-publishing migration: branch publishing retired the
        # deploy job and the Pages grants this pin used to assert
        # (pages/id-token write on build and deploy). The surviving
        # cycle-215 rule is the scoping itself: grants live at job
        # level, the build job stays read-only, and the publish
        # job's contents write — the grant a git push to gh-pages
        # needs — exists nowhere else and at no higher level.
        """The docs workflow's Pages grants are job-level only."""
        doc = _load_yaml(WORKFLOWS / "docs.yml")
        build_perms = doc["jobs"]["build"]["permissions"]
        self.assertEqual(build_perms.get("contents"), "read")
        self.assertIsNone(build_perms.get("pages"))
        self.assertIsNone(build_perms.get("id-token"))
        publish_perms = doc["jobs"]["publish"]["permissions"]
        self.assertEqual(publish_perms.get("contents"), "write")
        self.assertIsNone(publish_perms.get("pages"))
        self.assertIsNone(publish_perms.get("id-token"))

    def test_codeql_and_scorecard_grants_are_job_level(self):
        """CodeQL and Scorecard grants are job-level, with security-events write only where
        required.
        """
        codeql = _load_yaml(WORKFLOWS / "codeql.yml")
        self.assertEqual(codeql["jobs"]["analyze"]["permissions"].get("security-events"), "write")
        scorecard = _load_yaml(WORKFLOWS / "scorecard.yml")
        perms = scorecard["jobs"]["analysis"]["permissions"]
        self.assertEqual(perms.get("security-events"), "write")
        self.assertEqual(perms.get("id-token"), "write")

    def test_ci_test_step_has_no_template_expansion_in_script(self):
        """The CI test step's script contains no template expansion."""
        doc = _load_yaml(WORKFLOWS / "ci.yml")
        steps = doc["jobs"]["test"]["steps"]
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        step = next(s for s in steps if s.get("name") == "run tests (output captured)")
        self.assertNotIn("${{", step["run"])
        self.assertEqual(step["env"]["PYTHON_VERSION"], "${{ matrix.python-version }}")
        self.assertIn("unittest-py$PYTHON_VERSION.log", step["run"])

    def test_release_named_jobs_and_concurrency(self):
        """The release workflow's jobs carry their frozen names and concurrency settings."""
        doc = _load_yaml(WORKFLOWS / "release.yml")
        self.assertEqual(doc["jobs"]["build"].get("name"), "build release artifacts")
        self.assertEqual(doc["jobs"]["publish"].get("name"), "publish to PyPI and GitHub")
        self.assertIs(doc["concurrency"]["cancel-in-progress"], False)


class TestDependencyReview(unittest.TestCase):
    """The dependency-review workflow: pull-request-only, pinned, read-only."""

    def setUp(self):
        """Load the dependency-review workflow."""
        self.path = WORKFLOWS / "dependency-review.yml"
        self.doc = _load_yaml(self.path)

    def test_pull_request_only(self):
        # Never on push: the autonomous push pipeline is structurally
        # out of this gate's reach.
        """Dependency review triggers on pull requests only, with read-only permissions."""
        self.assertEqual(set(_triggers(self.doc)), {"pull_request"})
        self.assertEqual(self.doc.get("permissions"), {"contents": "read"})

    def test_action_pin_and_posture(self):
        """The review action is pinned and the job carries the frozen posture."""
        job = self.doc["jobs"]["review"]
        self.assertEqual(job.get("name"), "dependency review")
        self.assertIsInstance(job.get("timeout-minutes"), int)
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        review = next(
            s
            for s in job["steps"]
            if str(s.get("uses", "")).startswith("actions/dependency-review-action@")
        )
        self.assertEqual(review["uses"], f"actions/dependency-review-action@{DEP_REVIEW_SHA}")
        posture = review["with"]
        self.assertEqual(posture.get("fail-on-severity"), "high")
        scopes = {s.strip() for s in posture["fail-on-scopes"].split(",")}
        self.assertEqual(scopes, {"runtime", "development"})
        denied = {s.strip() for s in posture["deny-licenses"].split(",")}
        for family in ("AGPL-3.0", "GPL-2.0", "GPL-3.0", "LGPL-2.1", "LGPL-3.0"):
            self.assertIn(family, denied)


class TestAdrIndex(unittest.TestCase):
    """The ADR index page matches the shipped design set and every link resolves."""

    def setUp(self):
        """Load the ADR index page, its text and the mkdocs nav."""
        self.page = DOCS / "explanation" / "adr-index.md"
        self.text = self.page.read_text(encoding="utf-8")
        mkdocs = _load_yaml(REPO_ROOT / "mkdocs.yml")
        self.nav = sorted(set(_nav_targets(mkdocs["nav"])))
        self.linked = sorted({Path(t).name for t in _MD_LINK.findall(self.text)})

    def test_page_is_in_nav(self):
        """The ADR index page is in the mkdocs nav."""
        self.assertIn("explanation/adr-index.md", self.nav)

    def test_index_matches_shipped_design_set(self):
        # Both directions: every design document in the navigation
        # is indexed, and nothing is indexed that is not in the
        # navigation (the navigation is the shipped set — cycle
        # 214's lesson: local-only design documents must not be
        # linked, and the strict docs build enforces the same on
        # the CI checkout).
        """The index links exactly the shipped design-document set."""
        nav_design = {Path(t).name for t in self.nav if t.endswith("_DESIGN.md")}
        indexed_design = {name for name in self.linked if name.endswith("_DESIGN.md")}
        self.assertEqual(indexed_design, nav_design)
        self.assertGreater(len(nav_design), 20)

    def test_every_link_resolves(self):
        """Every link on the ADR index page resolves."""
        for target in _MD_LINK.findall(self.text):
            resolved = (self.page.parent / target).resolve()
            self.assertTrue(resolved.exists(), target)

    def test_standing_decisions_linked(self):
        """The standing-decision documents are linked from the index."""
        self.assertIn("SECURITY.md", self.linked)
        self.assertIn("RELEASE_READINESS_0.1.0.md", self.linked)

    def test_explanation_index_points_here(self):
        """The explanation index points at the ADR index."""
        text = (DOCS / "explanation" / "index.md").read_text(encoding="utf-8")
        self.assertIn("adr-index.md", text)

    def test_no_mpl_shorthand(self):
        """The ADR index never uses the banned shorthand for the project name."""
        offenders = [
            i for i, line in enumerate(self.text.splitlines(), 1) if _MPL_SHORTHAND.search(line)
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
