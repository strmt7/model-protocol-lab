"""Cycle 41: public API surface + documented-example accuracy audit.

Scope: the package's public surface (``model_protocol_lab/__init__.py``),
version consistency, and every CLI example/behavior claimed in README.md and
docs/USAGE.md that can be checked without a model or network.  Model-backed
commands are exercised against a tiny synthetic GPT-2 checkpoint built locally
in /tmp (seeded RNG, 2 layers, 64-wide, 267-token byte-level tokenizer; never
downloads anything, never touches the network).  Model-REQUIRED examples are
verified to fail in the documented way (fixed error codes, no tracebacks).

Doc/code discrepancies found (all four FIXED in cycle 41 by the
coordinator -- docs only, no behavior changes):

1. (FIXED.) README.md Quick start used ``-o`` for ``fixtures build`` --
   not a valid flag; corrected to ``--output``.
2. (FIXED.) docs/USAGE.md section 10 said ``.boundary`` uses "the same
   grammar as ``run --boundary``" -- no such flag; corrected to
   ``run --boundary-ids``.
3. (FIXED.) README.md's "27-family catalog" parenthetical listed only 26
   display names (``gemma4`` missing); corrected.
4. (FIXED.) README.md Capabilities claimed "``catalog`` lists every
   family with its source" -- ``mpl catalog`` emits no ``source`` field;
   corrected to framing fields + code-comment sources.

Everything else checked matched the docs: 27 families; ``inspect`` output
shape; ``fixtures build`` round-trip byte-identical to examples/fixtures.json;
overwrite refusal; the ``debug`` scripted session's "real output" lines
verbatim; dot-command errors; ``doctor`` checks/suggestion/reference-model/
portability-matrix shapes; ``diff-reports`` JSON + stderr contract and fixed
error codes; ``diff-backends`` JSON contract and divergence classes; ``run``
exit codes 0/1/2 semantics (completed ``no_candidate`` -> 2); the
``local_context_limit_exceeds_declared_window`` fast-fail; console-script
entry point; ``make_sample_report.py`` byte-determinism; version consistency
across ``__init__.py``, ``pyproject.toml`` and ``CITATION.cff``.

Sensitivity controls (mutation-style, run against copies in /tmp -- the repo
is never mutated) live in TestSensitivityControls and prove the pins catch
real regressions.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MPL = [sys.executable, "-m", "model_protocol_lab"]
EXAMPLE_FIXTURES = REPO_ROOT / "examples" / "fixtures.json"
EXAMPLE_CSV = REPO_ROOT / "examples" / "fixtures.csv"

# Fixed nonexistent-path fixtures for CLI error paths (never created).
MISSING_MODEL = Path("/tmp/mpl_cycle41_no_such_model")  # noqa: S108 -- task-mandated /tmp scratch

# The 27 catalog families, in catalog order (pinned against `mpl catalog`).
EXPECTED_FAMILIES = [
    "chatml",
    "llama3",
    "gemma",
    "qwen3",
    "qwen38",
    "deepseek",
    "mistral",
    "mistral_large",
    "phi4",
    "gemma3",
    "gemma4",
    "llama4",
    "glm4",
    "kimi2",
    "kimi3",
    "deepseek41",
    "glm45",
    "glm53",
    "kolibri",
    "lfm25",
    "nemotron35",
    "muse_glimmer",
    "laguna",
    "ling",
    "k2horizon",
    "hy4",
    "minimax3",
    "nanbeige42",
    "muse_spark",
]
EXPECTED_CATALOG_KEYS = {
    "name",
    "bos",
    "user_prefix",
    "assistant_prefix",
    "turn_end",
    "repeat_bos",
    "user_turn_end",
    "assistant_turn_end",
    "open_suffix",
}


def run_mpl(
    *args: str, cwd: Path = REPO_ROOT, input_text: str | None = None, timeout: int = 120
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*MPL, *args],
        cwd=cwd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


_MUTANT_DRIVER = (
    "import runpy, sys\n"
    "mutant_dir, *cli_args = sys.argv[1:]\n"
    "sys.path.insert(0, mutant_dir)\n"
    "# Drop the editable-install meta-path finder so the /tmp copy wins.\n"
    "sys.meta_path = [f for f in sys.meta_path\n"
    "                 if type(f).__name__ != '_EditableFinder']\n"
    "sys.argv = ['model-protocol-lab', *cli_args]\n"
    "runpy.run_module('model_protocol_lab', run_name='__main__')\n"
)


def run_mutant_cli(mutant_dir: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run the CLI from a mutated package copy under /tmp.

    The venv installs model_protocol_lab via an editable meta-path finder,
    which outranks PYTHONPATH; the driver removes that finder so the copy
    in ``mutant_dir`` is the one imported.  The repo itself is never
    mutated.
    """
    return subprocess.run(
        [sys.executable, "-c", _MUTANT_DRIVER, str(mutant_dir), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def assert_no_traceback(testcase: unittest.TestCase, stderr: str) -> None:
    testcase.assertNotIn("Traceback", stderr)


class TestPackageSurface(unittest.TestCase):
    """The Python package's public surface: __init__.py exports nothing.

    The documented interface is the CLI (``python -m model_protocol_lab``
    / the ``model-protocol-lab`` console script); the package itself must
    expose only ``__version__`` -- no __all__, no leaked private names.

    The probes load ``__init__.py`` as a fresh standalone module so the
    result is independent of which submodules the import system has
    already attached to the installed package in this process.
    """

    @staticmethod
    def _load_init_fresh(tag: str):
        path = REPO_ROOT / "model_protocol_lab" / "__init__.py"
        name = f"mpl_init_surface_probe_{tag}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module, name

    @staticmethod
    def _drop_probe(name: str) -> None:
        sys.modules.pop(name, None)

    def test_init_exports_only_version(self) -> None:
        module, name = self._load_init_fresh("exports")
        try:
            self.assertFalse(hasattr(module, "__all__"), "__init__.py should not define __all__")
            public = [n for n in dir(module) if not n.startswith("_")]
            self.assertEqual(public, [], f"unexpected public names in __init__.py: {public}")
        finally:
            self._drop_probe(name)

    def test_import_star_leaks_nothing(self) -> None:
        _, name = self._load_init_fresh("star")
        try:
            namespace: dict = {}
            # skipcq: PYL-W0122 - import-* audit execs this repo's own modules only
            exec(f"from {name} import *", namespace)  # noqa: S102
            leaked = sorted(n for n in namespace if not n.startswith("__"))
            self.assertEqual(leaked, [], f"import * leaked names: {leaked}")
        finally:
            self._drop_probe(name)

    def test_version_consistent_with_pyproject(self) -> None:
        import model_protocol_lab as pkg

        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
        self.assertIsNotNone(match, "no version in pyproject.toml")
        self.assertEqual(pkg.__version__, match.group(1))

    def test_version_consistent_with_citation(self) -> None:
        import model_protocol_lab as pkg

        cff = (REPO_ROOT / "CITATION.cff").read_text()
        match = re.search(r"^version:\s*(\S+)", cff, re.M)
        self.assertIsNotNone(match, "no version in CITATION.cff")
        self.assertEqual(pkg.__version__, match.group(1))

    def test_version_format(self) -> None:
        import model_protocol_lab as pkg

        self.assertRegex(
            pkg.__version__, r"^\d+\.\d+\.\d+", "version should start with MAJOR.MINOR.PATCH"
        )


class TestCatalog(unittest.TestCase):
    def test_catalog_29_families_and_keys(self) -> None:
        proc = run_mpl("catalog")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual([d["name"] for d in data], EXPECTED_FAMILIES)
        with self.subTest("entry key sets"):
            for entry in data:
                self.assertEqual(
                    set(entry.keys()), EXPECTED_CATALOG_KEYS, f"unexpected keys for {entry['name']}"
                )
        # Documented discrepancy: README claims catalog lists each family's
        # source; the JSON carries no source field.
        self.assertFalse(any("source" in d for d in data))

    def test_catalog_deterministic(self) -> None:
        first = run_mpl("catalog").stdout
        second = run_mpl("catalog").stdout
        self.assertEqual(first, second)


class TestTopLevelCLI(unittest.TestCase):
    def test_help_lists_all_subcommands(self) -> None:
        proc = run_mpl("--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for cmd in (
            "catalog",
            "inspect",
            "doctor",
            "run",
            "fixtures",
            "diff-reports",
            "diff-backends",
            "debug",
        ):
            with self.subTest(cmd=cmd):
                self.assertIn(cmd, proc.stdout)

    def test_console_script_equivalent(self) -> None:
        script = Path(sys.executable).parent / "model-protocol-lab"
        if not script.exists():
            script = shutil.which("model-protocol-lab")
        self.assertIsNotNone(script, "console script not installed")
        proc = subprocess.run([script, "catalog"], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('"chatml"', proc.stdout)

    def test_unknown_subcommand_exit_2(self) -> None:
        proc = run_mpl("nope")
        self.assertEqual(proc.returncode, 2)
        assert_no_traceback(self, proc.stderr)


class TestInspect(unittest.TestCase):
    def test_inspect_examples_shape(self) -> None:
        proc = run_mpl("inspect", str(EXAMPLE_FIXTURES))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            json.loads(proc.stdout),
            {
                "search_cases": 2,
                "held_out_cases": 2,
                "valid": True,
                # The example fixtures are contamination-clean.
                "contamination": {"worst_tier": "clean", "pairs": []},
            },
        )

    def test_empty_expected_warns_on_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fx.json"
            path.write_text(
                json.dumps(
                    {
                        "search": [
                            {
                                "id": "e1",
                                "group": "g",
                                "messages": [{"role": "user", "content": "hi"}],
                                "expected": "",
                            }
                        ],
                        "held_out": [
                            {
                                "id": "h1",
                                "group": "gh",
                                "messages": [{"role": "user", "content": "yo"}],
                                "expected": "x",
                            }
                        ],
                    }
                )
            )
            proc = run_mpl("inspect", str(path))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("empty expected answer", proc.stderr)
        self.assertIn("e1", proc.stderr)

    def test_rejects_directory(self) -> None:
        proc = run_mpl("inspect", tempfile.gettempdir())
        self.assertEqual(proc.returncode, 1)
        self.assertIn("not a regular file", proc.stderr)
        assert_no_traceback(self, proc.stderr)

    def test_rejects_22_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fx.json"
            msgs = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": "x"} for i in range(22)
            ]
            path.write_text(
                json.dumps(
                    {
                        "search": [{"id": "m1", "group": "g", "messages": msgs, "expected": "y"}],
                        "held_out": [
                            {
                                "id": "h1",
                                "group": "gh",
                                "messages": [{"role": "user", "content": "y"}],
                                "expected": "z",
                            }
                        ],
                    }
                )
            )
            proc = run_mpl("inspect", str(path))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("1-21 messages", proc.stderr)
        assert_no_traceback(self, proc.stderr)


class TestFixturesBuild(unittest.TestCase):
    def test_build_csv_matches_committed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fixtures.json"
            proc = run_mpl("fixtures", "build", str(EXAMPLE_CSV), "--output", str(out))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            built = json.loads(out.read_text())
            committed = json.loads(EXAMPLE_FIXTURES.read_text())
            self.assertEqual(built, committed)

    def test_build_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fixtures.json"
            out.write_text("{}")
            proc = run_mpl("fixtures", "build", str(EXAMPLE_CSV), "--output", str(out))
            self.assertEqual(proc.returncode, 1)
            self.assertIn("already exists", proc.stderr)
            assert_no_traceback(self, proc.stderr)

    def test_readme_short_flag_is_invalid(self) -> None:
        """The ``-o`` shorthand does not exist (README used to show it;
        fixed in cycle 41 to ``--output``). Pinning the rejection.
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fixtures.json"
            proc = run_mpl("fixtures", "build", str(EXAMPLE_CSV), "-o", str(out))
            self.assertEqual(proc.returncode, 2)
            self.assertIn("--output", proc.stderr)
            self.assertFalse(out.exists())


class TestDebugRepl(unittest.TestCase):
    def _script(self, commands: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text(commands)
            return run_mpl("debug", "--dataset", str(EXAMPLE_FIXTURES), "--script", str(script))

    def test_banner_matches_doc(self) -> None:
        proc = run_mpl("debug", "--dataset", "examples/fixtures.json", input_text=".quit\n")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        match = re.search(
            r'^version\s*=\s*"([^"]+)"', (REPO_ROOT / "pyproject.toml").read_text(), re.M
        )
        assert match is not None
        self.assertIn(f"model-protocol-lab debug {match.group(1)}", proc.stdout)
        self.assertIn(
            "dataset: examples/fixtures.json (4 fixtures) | family: chatml | tokenizer: none",
            proc.stdout,
        )

    def test_scripted_render_matches_doc_real_output(self) -> None:
        """The USAGE.md section 10 'real output' example, verbatim."""
        proc = self._script(".fixture 1\n.render\n.boundary 8\n.quit\n")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for line in (
            "selected fixture 1: search-copy-blue",
            "render: family=chatml fixture=search-copy-blue messages=1 chars=91",
            "[0] bos              chars 0:0",
            "[1] user_prefix      chars 0:17",
            "[2] user_content     chars 17:58",
            "[3] user_turn_end    chars 58:69",
            "[4] open_suffix      chars 69:91",
            "boundary set: [8]",
        ):
            with self.subTest(line=line):
                self.assertIn(line, proc.stdout)

    def test_step_walks_boundary_detection(self) -> None:
        proc = self._script(".boundary 8\n.step 1,2,8,3\n.quit\n")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("token 3/4 id=8 overhang=1/1 boundary_at=2", proc.stdout)

    def test_families_lists_29(self) -> None:
        proc = self._script(".families\n.quit\n")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip().splitlines(), EXPECTED_FAMILIES)

    def test_unknown_family_error(self) -> None:
        proc = self._script(".use notafamily\n.quit\n")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("error: unknown family 'notafamily'", proc.stdout)

    def test_boundary_grammar(self) -> None:
        cases = {
            "8": True,
            "1,2,3": True,
            ",".join(["7"] * 16): True,
            ",".join(["7"] * 17): False,
            "abc": False,
            "": False,
            "1,,2": False,
        }
        for spec, ok in cases.items():
            with self.subTest(spec=spec[:12]):
                proc = self._script(f".boundary {spec}\n.quit\n")
                if ok:
                    self.assertIn("boundary set:", proc.stdout)
                else:
                    self.assertNotIn("boundary set:", proc.stdout)

    def test_step_without_boundary_errors(self) -> None:
        proc = self._script(".step 1,2\n.quit\n")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("error: no boundary set", proc.stdout)

    def test_unknown_dot_command_errors(self) -> None:
        proc = self._script(".bogus\n.quit\n")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("error: unknown dot-command '.bogus'", proc.stdout)

    def test_tokens_without_model_prints_reminder(self) -> None:
        proc = self._script(".tokens\n.quit\n")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("no tokenizer attached", proc.stdout)

    def test_audit_reports_zero_findings(self) -> None:
        proc = self._script(".audit\n.quit\n")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(
            "audit: 0 finding(s) on fixture search-copy-blue with family chatml", proc.stdout
        )

    def test_save_appends_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            saved = Path(tmp) / "t.txt"
            script.write_text(f".tokens\n.save {saved}\n.quit\n")
            proc = run_mpl("debug", "--dataset", str(EXAMPLE_FIXTURES), "--script", str(script))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            content = saved.read_text()
            self.assertIn(".tokens", content)
            self.assertIn("no tokenizer attached", content)


def _require_toolkit(testcase: unittest.TestCase) -> None:
    try:
        import safetensors  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError as exc:
        testcase.skipTest(f"synthetic-checkpoint toolkit missing: {exc}")


class SyntheticCheckpointMixin:
    """Build a tiny deterministic GPT-2 checkpoint in /tmp (no network)."""

    _model_dir: Path | None = None

    @classmethod
    def build_checkpoint(cls) -> Path:
        if cls._model_dir is not None and cls._model_dir.exists():
            return cls._model_dir
        import numpy as np
        from safetensors.numpy import save_file
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as _BLDec
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel

        root = Path(tempfile.mkdtemp(prefix="mpl_synth_model_"))
        n_embd, n_layer, n_pos, vocab = 64, 2, 1024, 267
        rng = np.random.default_rng(7)
        tensors: dict = {}
        tensors["transformer.wte.weight"] = rng.normal(0, 0.02, (vocab, n_embd)).astype(np.float32)
        tensors["transformer.wpe.weight"] = rng.normal(0, 0.02, (n_pos, n_embd)).astype(np.float32)
        for i in range(n_layer):
            p = f"transformer.h.{i}"
            tensors[f"{p}.ln_1.weight"] = np.ones(n_embd, dtype=np.float32)
            tensors[f"{p}.ln_1.bias"] = np.zeros(n_embd, dtype=np.float32)
            tensors[f"{p}.attn.c_attn.weight"] = rng.normal(0, 0.02, (n_embd, 3 * n_embd)).astype(
                np.float32
            )
            tensors[f"{p}.attn.c_attn.bias"] = np.zeros(3 * n_embd, dtype=np.float32)
            tensors[f"{p}.attn.c_proj.weight"] = rng.normal(0, 0.02, (n_embd, n_embd)).astype(
                np.float32
            )
            tensors[f"{p}.attn.c_proj.bias"] = np.zeros(n_embd, dtype=np.float32)
            tensors[f"{p}.ln_2.weight"] = np.ones(n_embd, dtype=np.float32)
            tensors[f"{p}.ln_2.bias"] = np.zeros(n_embd, dtype=np.float32)
            tensors[f"{p}.mlp.c_fc.weight"] = rng.normal(0, 0.02, (n_embd, 4 * n_embd)).astype(
                np.float32
            )
            tensors[f"{p}.mlp.c_fc.bias"] = np.zeros(4 * n_embd, dtype=np.float32)
            tensors[f"{p}.mlp.c_proj.weight"] = rng.normal(0, 0.02, (4 * n_embd, n_embd)).astype(
                np.float32
            )
            tensors[f"{p}.mlp.c_proj.bias"] = np.zeros(n_embd, dtype=np.float32)
        tensors["transformer.ln_f.weight"] = np.ones(n_embd, dtype=np.float32)
        tensors["transformer.ln_f.bias"] = np.zeros(n_embd, dtype=np.float32)
        save_file(tensors, str(root / "model.safetensors"))

        vocab_map = {bytes([i]).decode("latin-1"): i for i in range(256)}
        for i in range(10):
            vocab_map[f"tok{i}"] = 256 + i
        vocab_map["<|eos|>"] = 266
        tok = Tokenizer(BPE(vocab=vocab_map, merges=[]))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = _BLDec()
        tok.add_special_tokens(["<|eos|>"])
        tok.save(str(root / "tokenizer.json"))

        (root / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["GPT2LMHeadModel"],
                    "model_type": "gpt2",
                    "n_positions": n_pos,
                    "n_ctx": n_pos,
                    "n_embd": n_embd,
                    "n_layer": n_layer,
                    "n_head": 2,
                    "vocab_size": vocab,
                }
            )
        )
        cls._model_dir = root
        return root


class TestDebugWithTokenizer(unittest.TestCase, SyntheticCheckpointMixin):
    """USAGE.md section 10: --model attaches the tokenizer only."""

    @classmethod
    def setUpClass(cls) -> None:
        _require_toolkit(cls)
        cls.model = cls.build_checkpoint()

    def test_tokens_exact_ids_with_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text(".tokens\n.quit\n")
            proc = run_mpl(
                "debug",
                "--dataset",
                str(EXAMPLE_FIXTURES),
                "--model",
                str(self.model),
                "--script",
                str(script),
                timeout=180,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # Byte-level synthetic tokenizer: '<' is 60, '|' is 124, matching
        # the chatml render "<|im_start|>user...".
        self.assertIn("tokens: family=chatml fixture=search-copy-blue", proc.stdout)
        self.assertIn("ids=[60, 124, 105, 109", proc.stdout)

    def test_render_shows_per_segment_ids_with_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text(".render\n.quit\n")
            proc = run_mpl(
                "debug",
                "--dataset",
                str(EXAMPLE_FIXTURES),
                "--model",
                str(self.model),
                "--script",
                str(script),
                timeout=180,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("tokens=[60, 124", proc.stdout)


class TestDoctor(unittest.TestCase, SyntheticCheckpointMixin):
    @classmethod
    def setUpClass(cls) -> None:
        _require_toolkit(cls)
        cls.model = cls.build_checkpoint()

    def test_ok_verdict(self) -> None:
        proc = run_mpl(
            "doctor", "--model", str(self.model), "--dataset", str(EXAMPLE_FIXTURES), timeout=180
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["verdict"], "ok")
        with self.subTest("all checks ok"):
            for name, check in report["model"]["checks"].items():
                self.assertEqual(check["status"], "ok", f"check {name} not ok: {check}")
        # USAGE.md section 2: stdout stays parseable JSON (table-free).
        self.assertNotIn("portability matrix", proc.stdout)

    def test_suggestion_matches_doc_template(self) -> None:
        proc = run_mpl(
            "doctor",
            "--model",
            str(self.model),
            "--dataset",
            str(EXAMPLE_FIXTURES),
            "--boundary-ids",
            "266",
            timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        suggestion = json.loads(proc.stdout)["suggestion"]
        self.assertIsNotNone(suggestion, "suggestion should be present")
        cmd = suggestion["next_command"]
        # The doc: --max-context from the checkpoint's declared window,
        # shell-quoted paths, explicit FAMILY placeholder.
        self.assertIn("--max-context 1024", cmd)
        self.assertIn("--family FAMILY", cmd)
        self.assertIn("--boundary-ids 266", cmd)
        self.assertIn("--mutate", cmd)
        self.assertIn("--output NEW_REPORT_DIRECTORY", cmd)
        self.assertIn(str(self.model), cmd)

    def test_missing_model_dir_fails(self) -> None:
        proc = run_mpl("doctor", "--model", str(MISSING_MODEL), "--dataset", str(EXAMPLE_FIXTURES))
        self.assertEqual(proc.returncode, 1)
        report = json.loads(proc.stdout)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["model"]["checks"]["directory"]["status"], "failed")
        self.assertIn(
            "not a readable local directory", report["model"]["checks"]["directory"]["detail"]
        )
        assert_no_traceback(self, proc.stderr)

    def test_broken_model_fails_with_fixed_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text("{}")
            proc = run_mpl("doctor", "--model", tmp, "--dataset", str(EXAMPLE_FIXTURES))
        self.assertEqual(proc.returncode, 1)
        report = json.loads(proc.stdout)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(
            report["model"]["checks"]["asset_layout"]["detail"],
            "require_one_standard_safetensors_layout",
        )
        assert_no_traceback(self, proc.stderr)

    def test_reference_model_portability(self) -> None:
        proc = run_mpl(
            "doctor",
            "--model",
            str(self.model),
            "--reference-model",
            str(self.model),
            "--boundary-ids",
            "65",
            timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        checks = json.loads(proc.stdout)["model"]["checks"]
        self.assertEqual(checks["boundary_portability"]["status"], "ok")
        self.assertIn(
            "identical under reference tokenizer", checks["boundary_portability"]["detail"]
        )

    def test_portability_matrix_shape(self) -> None:
        proc = run_mpl(
            "doctor",
            "--model",
            str(self.model),
            "--portability-matrix",
            str(self.model),
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        # The matrix never fails the doctor verdict.
        self.assertEqual(report["verdict"], "ok")
        matrix = report["portability_matrix"]
        rows = matrix["rows"]
        self.assertGreater(len(rows), 100)
        row = rows[0]
        self.assertEqual(
            set(row.keys()), {"bos_skipped", "detail", "format", "reason", "tokens", "verdict"}
        )
        # The readable table goes to stderr so stdout stays parseable JSON.
        self.assertIn("verdict", proc.stderr)
        self.assertIn("format", proc.stderr)


class TestDiffReports(unittest.TestCase):
    @staticmethod
    def _write_results(
        path: Path, *, decision: str, winner: str | None, search: list, held_out: dict
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "decision": decision,
                    "selected_candidate": winner,
                    "search": search,
                    "held_out": held_out,
                }
            )
        )

    @staticmethod
    def _entry(
        name: str, planned: int, passes: int, eligible: bool, observations: list | None = None
    ) -> dict:
        entry: dict = {
            "format": {"name": name},
            "summary": {
                "planned": planned,
                "exact_and_boundary_passes": passes,
                "aggregate_eligible": eligible,
            },
        }
        if observations is not None:
            entry["observations"] = observations
        return entry

    @staticmethod
    def _obs(fixture: str, passed: bool) -> dict:
        return {"fixture": fixture, "exact_match": passed, "boundary_found": passed}

    def _pair(self, tmp: Path) -> tuple[Path, Path]:
        base, fixed = tmp / "base", tmp / "fixed"
        base.mkdir()
        fixed.mkdir()
        self._write_results(
            base / "results.json",
            decision="no_candidate",
            winner=None,
            search=[
                self._entry("chatml", 2, 1, True, [self._obs("f1", True), self._obs("f2", False)]),
                self._entry("llama3", 2, 1, True, [self._obs("f1", False), self._obs("f2", True)]),
            ],
            held_out={"observations": []},
        )
        self._write_results(
            fixed / "results.json",
            decision="compatible_on_fixture_set",
            winner="llama3",
            search=[
                self._entry("chatml", 2, 1, True, [self._obs("f1", True), self._obs("f2", False)]),
                self._entry("llama3", 2, 2, True, [self._obs("f1", True), self._obs("f2", True)]),
            ],
            held_out={
                "observations": [self._obs("h1", True), self._obs("h2", True)],
                "summary": {
                    "planned": 2,
                    "exact_and_boundary_passes": 2,
                    "aggregate_eligible": True,
                },
            },
        )
        return base, fixed

    def test_compare_two_runs_matches_doc_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base, fixed = self._pair(Path(tmp))
            proc = run_mpl("diff-reports", str(base), str(fixed))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        # Documented stdout keys.
        for key in (
            "decision",
            "winner",
            "config_drift",
            "candidate_verdicts",
            "fixture_diffs",
            "evidence",
            "summary",
        ):
            with self.subTest(key=key):
                self.assertIn(key, payload)
        self.assertEqual(
            payload["decision"],
            {"old": "no_candidate", "new": "compatible_on_fixture_set", "changed": True},
        )
        self.assertEqual(payload["winner"], {"old": None, "new": "llama3", "changed": True})
        for key in (
            "fixtures_flipped",
            "candidates_changed_status",
            "config_drift_warnings",
            "identical",
        ):
            with self.subTest(summary_key=key):
                self.assertIn(key, payload["summary"])
        # Documented stderr summary lines.
        for line in (
            "Decision: no_candidate -> compatible_on_fixture_set",
            "Winner: (none) -> llama3",
            "search/llama3: fail -> pass",
            "held_out/llama3: absent -> pass",
            "Summary: decision CHANGED, winner CHANGED",
        ):
            with self.subTest(stderr_line=line[:30]):
                self.assertIn(line, proc.stderr)

    def test_identical_inputs_compare_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base, _ = self._pair(Path(tmp))
            proc = run_mpl("diff-reports", str(base), str(base))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["summary"]["identical"])
        self.assertIn("No differences", proc.stderr)

    def test_missing_input_fixed_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base, _ = self._pair(Path(tmp))
            proc = run_mpl("diff-reports", str(base), str(Path(tmp) / "missing"))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("results_path_missing", proc.stderr)
        assert_no_traceback(self, proc.stderr)

    def test_wrong_shape_fixed_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"nope": 1}))
            proc = run_mpl("diff-reports", str(bad), str(bad))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("results_wrong_shape", proc.stderr)
        assert_no_traceback(self, proc.stderr)


class TestDiffBackends(unittest.TestCase, SyntheticCheckpointMixin):
    @classmethod
    def setUpClass(cls) -> None:
        _require_toolkit(cls)
        cls.model = cls.build_checkpoint()

    def test_json_contract_and_divergence(self) -> None:
        proc = run_mpl(
            "diff-backends",
            "--model",
            str(self.model),
            "--prompt-text",
            "the answer is",
            "--output-text",
            "42 STOP hello",
            "--stops",
            "STOP",
            timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        for key in (
            "prompt_tokens",
            "output_tokens",
            "min_tokens",
            "eos_id",
            "stop_texts",
            "cells",
            "divergences",
            "verdict",
        ):
            with self.subTest(key=key):
                self.assertIn(key, payload)
        self.assertEqual(payload["stop_texts"], ["STOP"])
        cell = payload["cells"][0]
        self.assertEqual(cell["stop_text"], "STOP")
        self.assertEqual(
            set(cell["semantics"].keys()),
            {
                "hf_stop_string",
                "vllm_holdback_strip",
                "vllm_holdback_keep",
                "sglang_stop",
                "eos_appended",
            },
        )
        for name, record in cell["semantics"].items():
            with self.subTest(semantic=name):
                self.assertEqual(
                    set(record.keys()),
                    {
                        "available",
                        "fired",
                        "firing_step",
                        "truncated_ids",
                        "returned_text",
                        "disposition",
                    },
                )
        self.assertIn(payload["verdict"], ("agree", "diverge"))
        self.assertEqual(payload["verdict"], "diverge")
        div = payload["divergences"][0]
        self.assertEqual(set(div.keys()), {"class", "stop_text", "detail"})
        self.assertEqual(div["class"], "KEPT_VS_STRIPPED")

    def test_missing_model_fixed_code(self) -> None:
        proc = run_mpl(
            "diff-backends",
            "--model",
            str(MISSING_MODEL),
            "--prompt-text",
            "p",
            "--output-text",
            "o",
            "--stops",
            "STOP",
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("tokenizer_load_failed", proc.stderr)
        assert_no_traceback(self, proc.stderr)


class TestRunExitCodes(unittest.TestCase, SyntheticCheckpointMixin):
    @classmethod
    def setUpClass(cls) -> None:
        _require_toolkit(cls)
        cls.model = cls.build_checkpoint()

    def _base_args(self, out: Path) -> list:
        return [
            "run",
            "--model",
            str(self.model),
            "--dataset",
            str(EXAMPLE_FIXTURES),
            "--family",
            "chatml",
            "--boundary-ids",
            "266",
            "--max-new-tokens",
            "8",
            "--max-context",
            "1024",
            "--output",
            str(out),
        ]

    def test_missing_model_fixed_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_mpl(
                "run",
                "--model",
                str(MISSING_MODEL),
                "--dataset",
                str(EXAMPLE_FIXTURES),
                "--family",
                "chatml",
                "--boundary-ids",
                "266",
                "--output",
                str(Path(tmp) / "r"),
                timeout=120,
            )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("[local_backend_initialization_failed]", proc.stdout + proc.stderr)
        assert_no_traceback(self, proc.stderr)

    def test_context_exceeds_declared_window(self) -> None:
        """USAGE.md section 3: the distinct fast-fail code."""
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_mpl(
                "run",
                "--model",
                str(self.model),
                "--dataset",
                str(EXAMPLE_FIXTURES),
                "--family",
                "chatml",
                "--boundary-ids",
                "266",
                "--max-new-tokens",
                "8",
                "--max-context",
                "2048",
                "--output",
                str(Path(tmp) / "r"),
                timeout=180,
            )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("[local_context_limit_exceeds_declared_window]", proc.stdout + proc.stderr)
        assert_no_traceback(self, proc.stderr)

    def test_completed_negative_decision_exit_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report"
            proc = run_mpl(*self._base_args(out), timeout=300)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            line = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertEqual(line["decision"], "no_candidate")
            self.assertIsNone(line["selected_candidate"])
            results = json.loads((out / "results.json").read_text())
            self.assertEqual(results["decision"], "no_candidate")

    def test_existing_output_dir_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report"
            out.mkdir()
            proc = run_mpl(*self._base_args(out), timeout=120)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("Output path already exists", proc.stderr)
        assert_no_traceback(self, proc.stderr)

    def test_boundary_map_unknown_candidate_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_mpl(
                "run",
                "--model",
                str(self.model),
                "--dataset",
                str(EXAMPLE_FIXTURES),
                "--family",
                "chatml",
                "--boundary-ids",
                "266",
                "--boundary-map",
                "nope/wrong=1",
                "--output",
                str(Path(tmp) / "r"),
                timeout=120,
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("unknown candidate 'nope/wrong'", proc.stderr)
        self.assertIn("valid names: chatml", proc.stderr)

    def test_mutate_variant_names_and_per_candidate_boundary(self) -> None:
        """USAGE.md section 4: BOS-less chatml collapses toggle/no-bos."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report"
            proc = run_mpl(
                "run",
                "--model",
                str(self.model),
                "--dataset",
                str(EXAMPLE_FIXTURES),
                "--family",
                "chatml",
                "--mutate",
                "--boundary-ids",
                "266",
                "--boundary-map",
                "chatml/prefix-newline=65",
                "--max-new-tokens",
                "8",
                "--max-context",
                "1024",
                "--output",
                str(out),
                timeout=300,
            )
            self.assertEqual(proc.returncode, 2, proc.stderr)
            results = json.loads((out / "results.json").read_text())
        names = [e["format"]["name"] for e in results["search"]]
        self.assertEqual(names, ["chatml", "chatml/prefix-newline", "chatml/prefix-trim"])
        per_candidate = {e["format"]["name"]: e["boundary_token_ids"] for e in results["search"]}
        self.assertEqual(per_candidate["chatml/prefix-newline"], [65])
        self.assertEqual(per_candidate["chatml"], [266])

    def test_bad_boundary_ids_argparse_exit_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_mpl(
                "run",
                "--model",
                str(self.model),
                "--dataset",
                str(EXAMPLE_FIXTURES),
                "--family",
                "chatml",
                "--boundary-ids",
                "abc",
                "--output",
                str(Path(tmp) / "r"),
                timeout=60,
            )
        self.assertEqual(proc.returncode, 2)
        assert_no_traceback(self, proc.stderr)

    def test_missing_required_args_exit_2(self) -> None:
        proc = run_mpl("run", "--family", "chatml", timeout=60)
        self.assertEqual(proc.returncode, 2)
        assert_no_traceback(self, proc.stderr)


class TestSensitivityControls(unittest.TestCase):
    """Mutation-style controls: each pin must catch a real regression.

    Mutants are built on COPIES under /tmp; the repo is never mutated.
    Every mutant here is expected to make its corresponding pin FAIL --
    the test asserts the pin fires (AssertionError), proving sensitivity.
    """

    @staticmethod
    def _load_mutant_init(source: str, tag: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "__init__.py"
            path.write_text(source)
            spec = importlib.util.spec_from_file_location(f"mpl_mutant_{tag}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    def _assert_surface(self, module) -> None:
        public = [n for n in dir(module) if not n.startswith("_")]
        assert public == [], f"unexpected public names: {public}"
        assert not hasattr(module, "__all__"), "__all__ must not exist"

    def _assert_version(self, module) -> None:
        match = re.search(
            r'^version\s*=\s*"([^"]+)"', (REPO_ROOT / "pyproject.toml").read_text(), re.M
        )
        assert match is not None
        assert module.__version__ == match.group(1), (
            f"version {module.__version__!r} != pyproject {match.group(1)!r}"
        )

    def test_mutant_bumped_version_detected(self) -> None:
        source = (
            (REPO_ROOT / "model_protocol_lab" / "__init__.py").read_text().replace("0.1.0", "9.9.9")
        )
        mutant = self._load_mutant_init(source, "version")
        with self.assertRaises(AssertionError):
            self._assert_version(mutant)

    def test_mutant_leaked_public_name_detected(self) -> None:
        source = (
            REPO_ROOT / "model_protocol_lab" / "__init__.py"
        ).read_text() + "\nLEAKED_CONSTANT = 1\n"
        mutant = self._load_mutant_init(source, "leak")
        with self.assertRaises(AssertionError):
            self._assert_surface(mutant)

    def test_mutant_added_dunder_all_detected(self) -> None:
        source = (
            REPO_ROOT / "model_protocol_lab" / "__init__.py"
        ).read_text() + '\n__all__ = ["__version__"]\n'
        mutant = self._load_mutant_init(source, "all")
        with self.assertRaises(AssertionError):
            self._assert_surface(mutant)

    def test_mutant_boundary_grammar_widened_detected(self) -> None:
        """Copy the package to /tmp, widen parse_boundary to 17 IDs, and
        show the CLI pin (17 IDs must exit 2) no longer fires.
        """
        pkg_src = REPO_ROOT / "model_protocol_lab"
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model_protocol_lab"
            shutil.copytree(pkg_src, dest, ignore=shutil.ignore_patterns("__pycache__"))
            cli_py = dest / "cli.py"
            original = cli_py.read_text()
            widened = original.replace("1 <= len(parts) <= 16", "1 <= len(parts) <= 17", 1)
            self.assertNotEqual(widened, original, "anchor text for mutant not found")
            cli_py.write_text(widened)
            proc = run_mutant_cli(
                Path(tmp),
                "run",
                "--model",
                str(MISSING_MODEL),
                "--dataset",
                str(EXAMPLE_FIXTURES),
                "--family",
                "chatml",
                "--boundary-ids",
                ",".join(["7"] * 17),
                "--output",
                str(Path(tmp) / "r"),
                timeout=60,
            )
        # The pin requires exit 2 for 17 IDs; the mutant must NOT exit 2.
        self.assertNotEqual(
            proc.returncode,
            2,
            "mutant unexpectedly rejected 17 boundary IDs; sensitivity control is vacuous",
        )

    def test_mutant_catalog_family_removed_detected(self) -> None:
        """Remove one family from a /tmp package copy: the catalog pin
        must stop matching EXPECTED_FAMILIES.
        """
        pkg_src = REPO_ROOT / "model_protocol_lab"
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model_protocol_lab"
            shutil.copytree(pkg_src, dest, ignore=shutil.ignore_patterns("__pycache__"))
            cat_py = dest / "catalog.py"
            original = cat_py.read_text()
            pruned = original.replace(
                '"gemma4": Format(\n        "gemma4",',
                '"gemma4": Format(\n        "gemma4-MUTANT",',
                1,
            )
            self.assertNotEqual(pruned, original, "anchor text for mutant not found")
            cat_py.write_text(pruned)
            proc = run_mutant_cli(Path(tmp), "catalog", timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            names = [d["name"] for d in json.loads(proc.stdout)]
        self.assertNotEqual(names, EXPECTED_FAMILIES, "mutant catalog unexpectedly matched pins")


class TestSampleReportDeterminism(unittest.TestCase):
    def test_make_sample_report_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "repo"
            shutil.copytree(
                REPO_ROOT / "examples",
                dest / "examples",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            shutil.copytree(
                REPO_ROOT / "model_protocol_lab",
                dest / "model_protocol_lab",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            proc = subprocess.run(
                [sys.executable, "examples/make_sample_report.py"],
                cwd=dest,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            for name in ("results.json", "report.html"):
                fresh = (dest / "examples" / "sample-report" / name).read_bytes()
                committed = (REPO_ROOT / "examples" / "sample-report" / name).read_bytes()
                with self.subTest(file=name):
                    self.assertEqual(fresh, committed)


if __name__ == "__main__":
    unittest.main()
