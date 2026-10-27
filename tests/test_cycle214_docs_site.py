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
    def test_mkdocs_config_parses_with_material_theme(self):
        doc = _load_yaml(REPO_ROOT / "mkdocs.yml")
        self.assertEqual(doc["docs_dir"], "docs")
        self.assertEqual(doc["theme"]["name"], "material")
        self.assertEqual(doc["site_url"], "https://strmt7.github.io/model-protocol-lab/")

    def test_every_nav_target_exists(self):
        doc = _load_yaml(REPO_ROOT / "mkdocs.yml")
        targets = sorted(set(_nav_targets(doc["nav"])))
        self.assertGreater(len(targets), 40)
        missing = [t for t in targets if not (DOCS / t).exists()]
        self.assertEqual(missing, [], f"nav targets missing under docs/: {missing}")

    def test_quadrant_index_pages_exist(self):
        for rel in (
            "index.md",
            "tutorials/index.md",
            "how-to/index.md",
            "reference/index.md",
            "explanation/index.md",
        ):
            self.assertTrue((DOCS / rel).exists(), rel)

    def test_landing_page_routes_to_all_quadrants(self):
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
    def test_requirements_docs_pins_the_toolchain(self):
        text = (REPO_ROOT / "requirements-docs.txt").read_text(encoding="utf-8")
        self.assertIn("mkdocs==1.6.1", text)
        self.assertIn("mkdocs-material==9.8.0", text)

    def test_wheel_runtime_dependencies_stay_zero(self):
        # The docs toolchain must never leak into the shipped package.
        with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
            project = tomllib.load(fh)["project"]
        self.assertEqual(project["dependencies"], [])

    def test_site_output_is_gitignored(self):
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("site/", text.splitlines())


class TestDocsWorkflow(unittest.TestCase):
    def test_docs_workflow_builds_strict_and_deploys(self):
        path = REPO_ROOT / ".github" / "workflows" / "docs.yml"
        doc = _load_yaml(path)
        self.assertEqual(doc["permissions"].get("pages"), "write")
        for job_name, job in doc["jobs"].items():
            self.assertIsInstance(job.get("timeout-minutes"), int, job_name)
        text = path.read_text(encoding="utf-8")
        self.assertIn("mkdocs build --strict", text)
        self.assertIn("actions/upload-pages-artifact@", text)
        self.assertIn("actions/deploy-pages@", text)
        self.assertIn("requirements-docs.txt", text)


if __name__ == "__main__":
    unittest.main()
