"""Origin-asserting smoke test for a built wheel (W5-1, RR-11).

Monday morning: the release engineer runs this on the exact wheel a
release would publish and learns that the wheel itself works — not
that the commands can run against the checkout that happens to sit
next to it.

Why this exists: the release pipeline's previous smoke installed the
wheel into a fresh venv and then ran the venv's console scripts from
the checkout, with the checkout reachable. The W5 readiness lane
proved by sabotage that the smoke could not tell the artifact from
the tree: with no artifact installed at all and the checkout on
PYTHONPATH, every smoke command exited 0 (arm A); with the wheel
installed but the checkout on PYTHONPATH, the console scripts
silently imported the checkout instead of the wheel (arm B). A smoke
that passes in both of those worlds proves nothing about the
artifact.

This tool therefore controls the whole environment itself:

1. It creates its own venv and installs the wheel into it with
   --no-index (the wheel declares no runtime dependencies; an
   install that wants the network is a failed install).
2. Every subprocess runs from an empty external directory with a
   scrubbed environment — no PYTHONPATH, no PYTHONHOME, no user
   site — so the checkout is unreachable by construction.
3. Before any command output is trusted, an origin probe imports
   the package in the venv interpreter and asserts that the package,
   the CLI module and the MCP server module all resolve under that
   venv's site-packages, and that the imported version equals the
   version in the wheel's own dist-info METADATA.
4. Only then does it exercise the artifact: every subcommand's
   --help, the catalog (byte-compared against the same command run
   from the source tree), a fixtures build + inspect round-trip, a
   diff of the shipped sample report against a copy of itself, a
   pack + verify --bundle round-trip, and the MCP self-test — all
   on inputs copied into the external directory.

Any failed step raises SmokeFailure and the tool exits 1 with the
reason on stderr; nothing is reported as passed that was not run.

Usage: python tools/artifact_smoke.py --wheel dist/<wheel file> --source-tree .
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

_PROBE = """
import argparse
import json
import sysconfig

import model_protocol_lab
import model_protocol_lab.cli
import model_protocol_lab.mcp_server

commands = []
parser = model_protocol_lab.cli.build_parser()
for action in parser._actions:
    if isinstance(action, argparse._SubParsersAction) and action.dest == "command":
        commands = sorted(action.choices)
print(
    json.dumps(
        {
            "version": model_protocol_lab.__version__,
            "purelib": sysconfig.get_paths()["purelib"],
            "files": {
                "package": model_protocol_lab.__file__,
                "cli": model_protocol_lab.cli.__file__,
                "mcp_server": model_protocol_lab.mcp_server.__file__,
            },
            "commands": commands,
        }
    )
)
"""

_SOURCE_CATALOG = "from model_protocol_lab.cli import main; raise SystemExit(main(['catalog']))"

_COMMAND_TIMEOUT = 300
_INSTALL_TIMEOUT = 900


class SmokeFailure(Exception):
    """The smoke ran and the artifact (or the setup) failed it."""


def smoke_env() -> dict[str, str]:
    """The only environment a smoke subprocess may see.

    Deliberately an allowlist, not a denylist: PYTHONPATH, PYTHONHOME
    and PYTHONSTARTUP from the caller's shell must not reach the
    artifact (readiness arm B is exactly that leak), and neither may
    anything the next caller adds to their shell. PATH survives
    because console-script shebangs and venv plumbing expect it; the
    Windows system variables survive because the interpreter needs
    them to start at all.
    """
    env: dict[str, str] = {}
    for key in ("PATH", "HOME", "USERPROFILE", "SystemRoot", "windir", "TEMP", "TMP"):
        if key in os.environ:
            env[key] = os.environ[key]
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def check_origin(files: dict[str, str], purelib: str) -> None:
    """Refuse any probed module that resolved outside the venv.

    This assertion is the whole point of the tool: it is what fails
    in both readiness sabotage arms, where the import resolves to
    the checkout while every command still exits 0.
    """
    pure = Path(purelib)
    for key, file in files.items():
        resolved = Path(file)
        if pure not in resolved.parents:
            raise SmokeFailure(
                f"origin assertion failed: {key} imported from {file}, "
                f"which is outside the smoke venv site-packages {purelib}"
            )


def check_version(imported: str, expected: str) -> None:
    """Refuse an installed package whose version is not the wheel's."""
    if imported != expected:
        raise SmokeFailure(
            f"version assertion failed: installed package reports {imported}, "
            f"wheel METADATA carries {expected}"
        )


def wheel_version(wheel: Path) -> str:
    """The version the wheel itself declares, from its METADATA."""
    if not wheel.is_file():
        raise SmokeFailure(f"wheel not found: {wheel}")
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata_names = [
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise SmokeFailure(
                    f"wheel carries {len(metadata_names)} dist-info METADATA files: {wheel}"
                )
            metadata = archive.read(metadata_names[0]).decode("utf-8")
    except zipfile.BadZipFile as error:
        raise SmokeFailure(f"not a readable wheel (zip) file: {wheel}: {error}") from error
    for line in metadata.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    raise SmokeFailure(f"wheel METADATA carries no Version line: {wheel}")


def _venv_python(venv: Path) -> Path:
    """Return the venv python."""
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _venv_script(venv: Path, name: str) -> Path:
    """Return the venv script."""
    if os.name == "nt":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def _run(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    label: str,
    timeout: int = _COMMAND_TIMEOUT,
) -> subprocess.CompletedProcess[bytes]:
    """Run one smoke subprocess; any failure is a SmokeFailure."""
    try:
        result = subprocess.run(
            argv, cwd=cwd, env=env, capture_output=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SmokeFailure(f"{label}: could not run {argv[0]}: {error}") from error
    if result.returncode != 0:
        tail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        detail = tail[-1] if tail else f"no stderr, exit {result.returncode}"
        raise SmokeFailure(f"{label}: exited {result.returncode}: {detail}")
    return result


def create_venv(venv: Path) -> Path:
    """Create the smoke venv and return its interpreter path."""
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        capture_output=True,
        timeout=_INSTALL_TIMEOUT,
        check=False,
    )
    if result.returncode != 0:
        tail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        detail = tail[-1] if tail else f"exit {result.returncode}"
        raise SmokeFailure(f"venv creation failed: {detail}")
    return _venv_python(venv)


def install_wheel(python: Path, wheel: Path, env: dict[str, str]) -> None:
    """Install the wheel into the smoke venv, offline by construction."""
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-input",
            "--no-index",
            str(wheel),
        ],
        cwd=Path(tempfile.gettempdir()),
        env=env,
        label="wheel install",
        timeout=_INSTALL_TIMEOUT,
    )


def probe_installed(python: Path, *, cwd: Path, env: dict[str, str]) -> dict[str, object]:
    """Import the package in the venv interpreter and report origin.

    Returns the probe's JSON: version, purelib, the resolved module
    files and the CLI's subcommand names. Raises SmokeFailure when
    the import itself fails (the nothing-installed arm without a
    PYTHONPATH crutch lands here).
    """
    result = _run([str(python), "-c", _PROBE], cwd=cwd, env=env, label="origin probe")
    try:
        probe = json.loads(result.stdout.decode("utf-8").strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise SmokeFailure(f"origin probe printed no JSON verdict: {error}") from error
    if not isinstance(probe, dict) or "files" not in probe or "purelib" not in probe:
        raise SmokeFailure(f"origin probe verdict is malformed: {probe!r}")
    return probe


def run_smoke(wheel: Path, source_tree: Path, work: Path) -> list[str]:
    """Run the whole smoke; return the report lines on success."""
    expected_version = wheel_version(wheel)
    examples = source_tree / "examples"
    for needed in (examples / "fixtures.csv", examples / "sample-report"):
        if not needed.exists():
            raise SmokeFailure(f"source tree is missing a smoke input: {needed}")
    work.mkdir(parents=True, exist_ok=True)
    external = work / "external"
    external.mkdir(exist_ok=True)
    env = smoke_env()
    venv = work / "venv"
    python = create_venv(venv)
    install_wheel(python, wheel, env)
    report: list[str] = []
    probe = probe_installed(python, cwd=external, env=env)
    files = probe["files"]
    purelib = probe["purelib"]
    if not isinstance(files, dict) or not isinstance(purelib, str):
        raise SmokeFailure(f"origin probe verdict is malformed: {probe!r}")
    check_origin(files, purelib)
    check_version(str(probe["version"]), expected_version)
    report.append(f"origin: package, cli and mcp_server resolve under {purelib}")
    report.append(f"version: installed {probe['version']} == wheel METADATA")
    commands = probe["commands"]
    if not isinstance(commands, list) or not commands:
        raise SmokeFailure("installed CLI exposes no subcommands")
    cli = _venv_script(venv, "model-protocol-lab")
    mcp = _venv_script(venv, "model-protocol-lab-mcp")
    _run([str(cli), "--help"], cwd=external, env=env, label="model-protocol-lab --help")
    for name in commands:
        _run(
            [str(cli), str(name), "--help"],
            cwd=external,
            env=env,
            label=f"model-protocol-lab {name} --help",
        )
    report.append(f"help: top level plus {len(commands)} subcommands render")
    installed_catalog = _run(
        [str(cli), "catalog"], cwd=external, env=env, label="installed catalog"
    ).stdout
    if not installed_catalog.strip():
        raise SmokeFailure("installed catalog printed nothing")
    source_catalog = _run(
        [sys.executable, "-c", _SOURCE_CATALOG],
        cwd=source_tree,
        env={**env, "PYTHONPATH": str(source_tree)},
        label="source-tree catalog",
    ).stdout
    if installed_catalog != source_catalog:
        raise SmokeFailure(
            "installed catalog output differs from the source tree's "
            f"({len(installed_catalog)} bytes vs {len(source_catalog)})"
        )
    report.append(f"catalog: {len(installed_catalog)} bytes, byte-identical to the source tree")
    csv_copy = external / "fixtures.csv"
    shutil.copyfile(examples / "fixtures.csv", csv_copy)
    built = external / "fixtures.json"
    _run(
        [str(cli), "fixtures", "build", str(csv_copy), "--output", str(built)],
        cwd=external,
        env=env,
        label="fixtures build",
    )
    inspected = _run(
        [str(cli), "inspect", str(built)], cwd=external, env=env, label="inspect"
    ).stdout
    if not inspected.strip():
        raise SmokeFailure("inspect printed no verdict for the built fixtures")
    report.append("fixtures build + inspect: round-trip on external inputs")
    report_a = external / "report_a"
    report_b = external / "report_b"
    shutil.copytree(examples / "sample-report", report_a)
    shutil.copytree(examples / "sample-report", report_b)
    _run(
        [str(cli), "diff-reports", str(report_a), str(report_b)],
        cwd=external,
        env=env,
        label="diff-reports on the sample report",
    )
    report.append("diff-reports: the sample report compares identical to its copy")
    _run(
        [str(cli), "pack", str(report_a)],
        cwd=external,
        env=env,
        label="pack the sample report",
    )
    bundle = external / "report_a.bundle.zip"
    if not bundle.is_file():
        raise SmokeFailure(f"pack did not produce the expected bundle: {bundle}")
    _run(
        [str(cli), "verify", "--bundle", str(bundle)],
        cwd=external,
        env=env,
        label="verify --bundle",
    )
    report.append("pack + verify --bundle: the sealed sample report verifies")
    _run([str(mcp), "--self-test"], cwd=external, env=env, label="MCP self-test")
    report.append("mcp: model-protocol-lab-mcp --self-test passes")
    return report


def main(argv: list[str] | None = None) -> int:
    """Return the main."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True, help="the wheel to smoke")
    parser.add_argument(
        "--source-tree",
        type=Path,
        default=Path("."),
        help="checkout carrying the smoke inputs (examples/) and the catalog comparison baseline",
    )
    parser.add_argument(
        "--work",
        type=Path,
        default=None,
        help="scratch directory for the venv and external inputs "
        "(default: a fresh temporary directory)",
    )
    args = parser.parse_args(argv)
    work = args.work
    if work is None:
        work = Path(tempfile.mkdtemp(prefix="mpl_artifact_smoke_"))
    try:
        report = run_smoke(args.wheel.resolve(), args.source_tree.resolve(), work)
    except SmokeFailure as error:
        print(f"artifact smoke FAILED: {error}", file=sys.stderr)
        return 1
    for line in report:
        print(f"  {line}")
    print("artifact smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
