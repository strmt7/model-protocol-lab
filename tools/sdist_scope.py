"""Sdist-scope test runner: the shipped suite's declared scope, executed.

Monday morning: whoever cuts or verifies a release runs this against
the built sdist and learns exactly which part of the shipped test
suite the tarball can honestly verify — and that the rest is named,
with reasons, instead of silently failing.

Why this exists (RR-12): the sdist ships the full test suite, but
part of it audits the checkout itself — modules that import the
checkout's ``tools/`` package, read ``docs/`` or ``.github/`` at
import time, or read ``examples/`` and other checkout content at run
time. In an sdist-only extraction those modules error at collection
or fail (measured on the 0.2.0 tree: 14 collection errors caused by
checkout content; the remaining collection errors in a bare
environment are the [hf] stack, an environment scope the test-layer
manifest already declares separately). The honest contract is a
declared, narrower scope, not shipping the whole tree in the
tarball:

* ``tests/CHECKOUT_ONLY`` (shipped in the sdist) names every test
  module that requires checkout content, with the reason per module.
* This runner extracts the sdist (tarfile ``data`` filter, the
  pinned method), builds the project's own test environment in a
  venv — the hashed ``tools/requirements`` locks CI's suite jobs
  use (pip, the pinned build stack, dev-hf) plus the sdist itself
  installed non-editable, exactly as a recipient installs it — and,
  with the checkout unreachable (scrubbed environment, the
  extraction as the working directory), first asserts the imported
  package resolves inside the extraction, then collects the
  declared scope with zero errors, and with ``--phase run`` also
  runs it green.

The declared scope is the dev test layers (base + posix from
``tests/test_layers.json``) minus ``tests/CHECKOUT_ONLY``. The hf
layer is the [hf] extra's environment scope, declared in the layer
manifest, not an sdist defect, and stays out of this runner's scope
by that declaration. A module that gains a checkout dependency
without joining ``tests/CHECKOUT_ONLY`` fails this runner — a
collection error or a red run — which is the point: the manifest
cannot silently rot.

Usage: python tools/sdist_scope.py --sdist dist/<sdist file> \
    --source-tree . [--phase collect|run] [--work DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

_COMMAND_TIMEOUT = 3600


class ScopeFailure(Exception):
    """The sdist-scope verification ran and failed."""


def _clean_env(tmp: Path) -> dict[str, str]:
    """The only environment the sdist-scope subprocesses may see.

    No PYTHONPATH/PYTHONHOME (the checkout must be unreachable), and
    TMPDIR points inside the work directory — a small /tmp tmpfs
    cannot hold the dev-hf stack's unpack or the suite's tempdirs.
    The host's network configuration (proxy and index variables)
    passes through: it configures the machine's egress, not the
    checkout, and the hashed installs need it to reach an index.
    """
    env: dict[str, str] = {}
    passthrough = (
        "PATH",
        "HOME",
        "USERPROFILE",
        "SystemRoot",
        "windir",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
    )
    for key in passthrough:
        if key in os.environ:
            env[key] = os.environ[key]
    env["TMPDIR"] = str(tmp)
    env["TEMP"] = str(tmp)
    env["TMP"] = str(tmp)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def parse_checkout_only(text: str) -> list[tuple[str, str]]:
    """Parse tests/CHECKOUT_ONLY: ``module | reason`` per line.

    Blank lines and ``#`` comments are skipped. A line without a
    reason, a duplicate module, or a name that is not a test module
    file is a ScopeFailure — the manifest is a contract, not notes.
    """
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        module, sep, reason = line.partition("|")
        module = module.strip()
        reason = reason.strip()
        if not sep or not reason:
            raise ScopeFailure(f"CHECKOUT_ONLY line {lineno} carries no reason: {raw!r}")
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.py", module):
            raise ScopeFailure(f"CHECKOUT_ONLY line {lineno} is not a test module: {raw!r}")
        if module in seen:
            raise ScopeFailure(f"CHECKOUT_ONLY lists {module} twice")
        seen.add(module)
        entries.append((module, reason))
    if not entries:
        raise ScopeFailure("CHECKOUT_ONLY names no modules")
    return entries


def layer_scope(layers_text: str) -> tuple[list[str], list[str]]:
    """(in-scope, hf) module lists from tests/test_layers.json.

    In scope for the sdist contract = base + posix layers (the dev
    suite). The hf layer needs the [hf] extra's environment and is
    excluded by layer, never by CHECKOUT_ONLY.

    The shape is validated, not merely navigated: each layer must
    be a list of module-name strings. A layer given as an object or
    a string would otherwise be silently reinterpreted (keys, or
    single characters) into a garbage module selection.
    """
    try:
        data = json.loads(layers_text)
    except ValueError as error:
        raise ScopeFailure(f"test_layers.json is malformed: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("layers"), dict):
        raise ScopeFailure("test_layers.json is malformed: no 'layers' object")
    layers = data["layers"]

    def modules(name: str) -> list[str]:
        """Modules."""
        value = layers.get(name)
        if not isinstance(value, list) or not all(isinstance(m, str) for m in value):
            raise ScopeFailure(
                f"test_layers.json is malformed: layer {name!r} is not a list of module names"
            )
        return sorted(value)

    return modules("base") + modules("posix"), modules("hf")


def _run(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    label: str,
    timeout: int = _COMMAND_TIMEOUT,
) -> subprocess.CompletedProcess[bytes]:
    """Run one subprocess; any failure is a ScopeFailure."""
    try:
        result = subprocess.run(
            argv, cwd=cwd, env=env, capture_output=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ScopeFailure(f"{label}: could not run {argv[0]}: {error}") from error
    if result.returncode != 0:
        tail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        detail = tail[-1] if tail else f"no stderr, exit {result.returncode}"
        raise ScopeFailure(f"{label}: exited {result.returncode}: {detail}")
    return result


def _venv_python(venv: Path) -> Path:
    """Return the venv python."""
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def run_scope(sdist: Path, source_tree: Path, work: Path, phase: str) -> list[str]:
    """Execute the sdist-scope contract; return report lines."""
    if not sdist.is_file():
        raise ScopeFailure(f"sdist not found: {sdist}")
    requirements = source_tree / "tools" / "requirements"
    lock_names = ("pip.txt", "pip-setuptools.txt", "build.txt", "dev-hf.txt")
    for name in lock_names:
        if not (requirements / name).is_file():
            raise ScopeFailure(f"source tree is missing a runner input: {requirements / name}")
    work.mkdir(parents=True, exist_ok=True)
    tmp = work / "tmp"
    tmp.mkdir(exist_ok=True)
    env = _clean_env(tmp)
    try:
        with tarfile.open(sdist, "r:gz") as archive:
            archive.extractall(work / "sdist", filter="data")
    except (tarfile.TarError, OSError, EOFError) as error:
        # The data filter fails closed on traversal members, escaping
        # links and non-gzip input; the refusal is a ScopeFailure with
        # a named message (as in artifact_manifest.rebuild_check),
        # never a raw tarfile traceback out of the CLI.
        raise ScopeFailure(f"sdist extraction refused (data filter): {error}") from error
    candidates = [path for path in (work / "sdist").iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise ScopeFailure(f"sdist extracted to an unexpected layout: {candidates}")
    root = candidates[0]
    manifest_path = root / "tests" / "CHECKOUT_ONLY"
    if not manifest_path.is_file():
        raise ScopeFailure("the sdist does not ship tests/CHECKOUT_ONLY")
    checkout_only = parse_checkout_only(manifest_path.read_text(encoding="utf-8"))
    layers_path = root / "tests" / "test_layers.json"
    in_scope, hf = layer_scope(layers_path.read_text(encoding="utf-8"))
    shipped = {path.name for path in (root / "tests").glob("test_*.py")}
    for module, _reason in checkout_only:
        if module not in shipped:
            raise ScopeFailure(f"CHECKOUT_ONLY names a module the sdist does not ship: {module}")
        if module not in in_scope and module not in hf:
            raise ScopeFailure(f"CHECKOUT_ONLY names a module in no test layer: {module}")
    # A listed module may also sit in the hf layer (checkout-bound
    # AND heavy): the layer declaration excludes it from this
    # runner's scope either way, and the manifest stays the complete
    # census of checkout dependencies.
    excluded = {module for module, _reason in checkout_only} | set(hf)
    selected = [module for module in in_scope if module not in excluded]
    if not selected:
        raise ScopeFailure("the sdist scope selects no modules")
    venv = work / "venv"
    created = subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        capture_output=True,
        timeout=_COMMAND_TIMEOUT,
        check=False,
    )
    if created.returncode != 0:
        raise ScopeFailure("venv creation failed")
    python = _venv_python(venv)
    pip = [str(python), "-m", "pip", "install", "--quiet", "--no-input", "--require-hashes"]
    _run([*pip, "-r", str(requirements / "pip.txt")], cwd=work, env=env, label="install pip lock")
    _run(
        [
            *pip,
            "-r",
            str(requirements / "pip-setuptools.txt"),
            "-r",
            str(requirements / "build.txt"),
        ],
        cwd=work,
        env=env,
        label="install build stack",
    )
    _run(
        [*pip, "-r", str(requirements / "dev-hf.txt")],
        cwd=work,
        env=env,
        label="install dev-hf lock",
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-input",
            "--no-deps",
            "--no-build-isolation",
            str(sdist),
        ],
        cwd=work,
        env=env,
        label="install the sdist",
    )
    probe = _run(
        [str(python), "-c", "import model_protocol_lab; print(model_protocol_lab.__file__)"],
        cwd=root,
        env=env,
        label="package origin probe",
    )
    origin = probe.stdout.decode("utf-8").strip().splitlines()[-1]
    if not Path(origin).is_relative_to(root):
        raise ScopeFailure(f"package resolved outside the extraction: {origin}")
    report = [
        f"scope: {len(selected)} of {len(in_scope)} dev-layer modules "
        f"({len(checkout_only)} checkout-only, {len(hf)} hf-layer excluded by layer)",
        f"origin: {origin}",
    ]
    ignores = [f"--ignore=tests/{module}" for module in sorted(excluded)]
    _run(
        [
            str(python),
            "-m",
            "pytest",
            "tests/",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *ignores,
        ],
        cwd=root,
        env=env,
        label="sdist-only collection",
    )
    report.append("collection: zero errors")
    if phase == "run":
        _run(
            [str(python), "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider", *ignores],
            cwd=root,
            env=env,
            label="sdist-only run",
            timeout=4 * _COMMAND_TIMEOUT,
        )
        report.append("run: green")
    return report


def main(argv: list[str] | None = None) -> int:
    """Return the main."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdist", type=Path, required=True, help="the sdist to verify")
    parser.add_argument(
        "--source-tree",
        type=Path,
        default=Path("."),
        help="checkout carrying tools/requirements (the runner's own inputs)",
    )
    parser.add_argument("--phase", choices=("collect", "run"), default="run")
    parser.add_argument(
        "--work",
        type=Path,
        default=None,
        help="scratch directory (default: a fresh temporary directory)",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="on success, write a JSON result bound to the sdist's "
        "SHA-256 (consumed by tools/artifact_manifest.py)",
    )
    args = parser.parse_args(argv)
    work = args.work
    if work is None:
        work = Path(tempfile.mkdtemp(prefix="mpl_sdist_scope_"))
    try:
        report = run_scope(args.sdist.resolve(), args.source_tree.resolve(), work, args.phase)
    except ScopeFailure as error:
        print(f"sdist scope FAILED: {error}", file=sys.stderr)
        return 1
    for line in report:
        print(f"  {line}")
    print("sdist scope: OK")
    if args.result_json is not None:
        sdist = args.sdist.resolve()
        digest = hashlib.sha256()
        with sdist.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result = {
            "tool": "sdist_scope",
            "artifact": {"name": sdist.name, "sha256": digest.hexdigest()},
            "report": report,
            "result": "passed",
        }
        args.result_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
