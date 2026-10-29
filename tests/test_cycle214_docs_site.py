"""Cycle 214: documentation site / Diataxis structure.

Every assertion reads the shipped file, so a test passes only when
the docs-site artifact itself is present with the required content
(the cycle-210 pattern). No test here builds the site; the build is
verified by the docs workflow's own runs on main and by the local
`mkdocs build --strict` recorded in the cycle's build log.
"""

import re
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

_MPL_SHORTHAND = re.compile(r"`mpl (?!:)")


def _load_yaml(path):
    """Load a YAML file (the config under audit) with yaml.safe_load."""
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


class TestMkdocsConfig(unittest.TestCase):
    """MkDocs config: parses with the Material theme; every nav target exists."""

    def test_mkdocs_config_parses_with_material_theme(self):
        """mkdocs.yml parses with the Material theme and docs as the docs dir."""
        doc = _load_yaml(REPO_ROOT / "mkdocs.yml")
        self.assertEqual(doc["docs_dir"], "docs")
        self.assertEqual(doc["theme"]["name"], "material")
        self.assertEqual(doc["site_url"], "https://strmt7.github.io/model-protocol-lab/")

    def test_every_nav_target_exists(self):
        """Every mkdocs nav target exists on disk."""
        doc = _load_yaml(REPO_ROOT / "mkdocs.yml")
        targets = sorted(set(_nav_targets(doc["nav"])))
        # Anti-stub floor only: completeness against the pushed tree
        # is enforced by the docs workflow's `mkdocs build --strict`
        # on the CI checkout (a nav entry for a file that exists only
        # in a local working tree passes here but fails there — the
        # first docs run failed exactly that way).
        self.assertGreater(len(targets), 35)
        missing = [t for t in targets if not (DOCS / t).exists()]
        self.assertEqual(missing, [], f"nav targets missing under docs/: {missing}")

    def test_quadrant_index_pages_exist(self):
        """All four quadrant index pages exist."""
        for rel in (
            "index.md",
            "tutorials/index.md",
            "how-to/index.md",
            "reference/index.md",
            "explanation/index.md",
        ):
            self.assertTrue((DOCS / rel).exists(), rel)

    def test_landing_page_routes_to_all_quadrants(self):
        """The landing page routes to all four documentation quadrants."""
        text = (DOCS / "index.md").read_text(encoding="utf-8")
        for rel in (
            "tutorials/index.md",
            "how-to/index.md",
            "reference/index.md",
            "explanation/index.md",
        ):
            self.assertIn(rel, text, rel)

    def test_new_pages_have_no_mpl_shorthand(self):
        # Mirrors the cycle-133 rule for the pages this cycle adds
        # (the subdirectories sit outside that test's top-level glob).
        """New docs pages never use the banned shorthand for the project name."""
        offenders = []
        for rel in (
            "index.md",
            "tutorials/index.md",
            "how-to/index.md",
            "reference/index.md",
            "explanation/index.md",
        ):
            for i, line in enumerate((DOCS / rel).read_text(encoding="utf-8").splitlines(), 1):
                if _MPL_SHORTHAND.search(line):
                    offenders.append(f"{rel}:{i}")
        self.assertEqual(offenders, [])


class TestDocsToolchain(unittest.TestCase):
    """The docs toolchain is pinned and the wheel stays dependency-free."""

    def test_requirements_docs_pins_the_toolchain(self):
        """requirements-docs.txt pins mkdocs 1.6.1 and mkdocs-material 9.8.0."""
        text = (REPO_ROOT / "requirements-docs.txt").read_text(encoding="utf-8")
        self.assertIn("mkdocs==1.6.1", text)
        self.assertIn("mkdocs-material==9.8.0", text)

    def test_wheel_runtime_dependencies_stay_zero(self):
        # The docs toolchain must never leak into the shipped package.
        """The wheel's runtime dependencies stay zero."""
        with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
            project = tomllib.load(fh)["project"]
        self.assertEqual(project["dependencies"], [])

    def test_site_output_is_gitignored(self):
        """The built site output directory is gitignored."""
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("site/", text.splitlines())


class TestDocsWorkflow(unittest.TestCase):
    """The docs workflow builds strict and publishes the site."""

    def test_docs_workflow_builds_strict_and_publishes_to_branch(self):
        """The docs workflow builds with mkdocs strict mode and publishes the site."""
        path = REPO_ROOT / ".github" / "workflows" / "docs.yml"
        doc = _load_yaml(path)
        # Cycle 215: the Pages grants moved into the build and
        # deploy jobs (zizmor excessive-permissions — workflow-level
        # write grants are inherited by every job). Through cycle
        # 214 this pin asserted the top-level block.
        # Docs-publishing migration: the deploy job and its Pages
        # grants are gone. Publishing is a git push to the gh-pages
        # branch (Pages build_type=legacy), so no job may hold the
        # pages or id-token grants the deployment API required, and
        # the only write grant in the workflow is the publish job's
        # contents grant for the push itself.
        self.assertNotEqual(doc["permissions"].get("pages"), "write")
        for job_name, job in doc["jobs"].items():
            perms = job.get("permissions") or {}
            self.assertNotEqual(perms.get("pages"), "write", job_name)
            self.assertNotEqual(perms.get("id-token"), "write", job_name)
        self.assertEqual(doc["jobs"]["build"]["permissions"].get("contents"), "read")
        self.assertEqual(doc["jobs"]["publish"]["permissions"].get("contents"), "write")
        self.assertIsNone(doc["jobs"]["publish"].get("environment"))
        for job_name, job in doc["jobs"].items():
            self.assertIsInstance(job.get("timeout-minutes"), int, job_name)
        text = path.read_text(encoding="utf-8")
        self.assertIn("mkdocs build --strict", text)
        self.assertNotIn("actions/upload-pages-artifact@", text)
        self.assertNotIn("actions/deploy-pages@", text)
        self.assertIn("git push", text)
        self.assertIn("gh-pages", text)
        self.assertIn("requirements-docs.txt", text)


if __name__ == "__main__":
    unittest.main()
