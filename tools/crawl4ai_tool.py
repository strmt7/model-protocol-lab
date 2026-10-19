"""Install and invoke the lab's isolated, self-hosted development crawler.

This product includes software developed by UncleCode (https://x.com/unclecode)
as part of the Crawl4AI project (https://github.com/unclecode/crawl4ai).

Adapted from strmt7/FlitzZip tools/crawl4ai_tool.py (MIT). FlitzZip-specific
machinery (Windows process-ownership contracts, NLTK security source builds,
repaired-wheel admission) is intentionally not ported: this repo installs the
upstream release directly with a version-pinned lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "tools" / "requirements" / "crawl4ai.txt"
VERSION = "0.9.4"
ATTRIBUTION = (
    "This product includes software developed by UncleCode (https://x.com/unclecode) "
    "as part of the Crawl4AI project (https://github.com/unclecode/crawl4ai)."
)


def cache_home(override: str | None = None) -> Path:
    """Purpose: Locate a portable tool cache
    Inputs: Optional absolute override
    Outputs: Absolute path."""
    selected = override or os.environ.get("MPL_CRAWL4AI_HOME")
    if selected:
        result = Path(selected).expanduser()
        if not result.is_absolute():
            raise ValueError("Crawler home must be an absolute path")
    elif sys.platform == "win32":
        result = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
        result = result / "model-protocol-lab/crawl4ai"
    elif sys.platform == "darwin":
        result = Path.home() / "Library/Caches/model-protocol-lab/crawl4ai"
    else:
        result = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        result = result / "model-protocol-lab/crawl4ai"
    result = result.resolve()
    if result == ROOT or ROOT in result.parents:
        raise ValueError("Crawler environment must stay outside the source checkout")
    return result


def text_identity(path: Path) -> str:
    """Purpose: Match Git's portable text identity
    Inputs: Text file
    Outputs: LF-normalized SHA-256 digest."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def environment_paths(home: Path) -> tuple[Path, Path, str]:
    """Purpose: Isolate dependency revisions
    Inputs: Cache root
    Outputs: Environment, Python and lock identity."""
    identity = text_identity(LOCK)
    name = f"{VERSION}-py{sys.version_info.major}{sys.version_info.minor}-{identity[:16]}"
    directory = home / name
    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return directory, python, identity


def tool_environment(home: Path) -> dict[str, str]:
    """Purpose: Keep browser/cache writes scoped
    Inputs: Tool cache
    Outputs: Child-only environment."""
    env = dict(os.environ)
    env.update(
        PLAYWRIGHT_BROWSERS_PATH=str(home / "browsers"),
        CRAWL4AI_BASE_DIRECTORY=str(home / "data"),
        PYTHONUTF8="1",
        PYTHONNOUSERSITE="1",
        DO_NOT_TRACK="1",
        LITELLM_TELEMETRY="False",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    return env


def _run(args: list[str], env: dict[str, str], timeout: float = 900) -> None:
    """Purpose: Run a bounded child
    Inputs: Command/environment/deadline
    Outputs: None or failure."""
    subprocess.run(
        args,
        env=env,
        timeout=timeout,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
    )


def verify_versions(python: Path, env: dict[str, str]) -> None:
    """Purpose: Confirm the pinned crawler
    Inputs: Environment interpreter
    Outputs: None or version failure."""
    output = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('crawl4ai'))",
        ],
        env=env,
        timeout=60,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()
    if output != VERSION:
        raise ValueError(
            f"crawler version {output} does not match the reviewed integration {VERSION}"
        )


def install(home: Path) -> Path:
    """Purpose: Provision the pinned package/browser once
    Inputs: Portable cache
    Outputs: Isolated interpreter."""
    directory, python, identity = environment_paths(home)
    receipt = directory / "install-receipt.json"
    if receipt.is_file():
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            recorded.get("version") == VERSION
            and recorded.get("lock_sha256") == identity
            and python.exists()
        ):
            return python
    home.mkdir(parents=True, exist_ok=True)
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(directory)
    env = tool_environment(home)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--no-input",
            "-r",
            str(LOCK),
        ],
        env,
        timeout=1800,
    )
    _run([str(python), "-m", "pip", "check"], env, timeout=120)
    verify_versions(python, env)
    _run([str(python), "-m", "playwright", "install", "chromium"], env, timeout=1800)
    receipt.write_text(
        json.dumps(
            {
                "schema": 1,
                "version": VERSION,
                "lock_sha256": identity,
                "platform": platform.system(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return python


def doctor(home: Path) -> None:
    """Purpose: Diagnose the installation
    Inputs: Portable cache
    Outputs: Diagnostic report or failure."""
    directory, python, identity = environment_paths(home)
    report = {
        "version": VERSION,
        "lock_sha256": identity,
        "home": str(home),
        "directory": str(directory),
        "python_exists": python.exists(),
    }
    if python.exists():
        env = tool_environment(home)
        try:
            _run([str(python), "-m", "pip", "check"], env, timeout=120)
            report["pip_check"] = "ok"
        except subprocess.CalledProcessError as exc:
            report["pip_check"] = f"failed: {exc}"
        try:
            verify_versions(python, env)
            report["version_check"] = "ok"
        except (ValueError, subprocess.CalledProcessError) as exc:
            report["version_check"] = f"failed: {exc}"
    print(json.dumps(report, indent=2))


def check_lock(home: Path, requirements: Path = LOCK) -> None:
    """Purpose: Reject incompatible dependencies
    Inputs: Cache/lock
    Outputs: Resolver result; no runtime install."""
    home.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lock-validation-", dir=home) as owned:
        directory = Path(owned)
        venv.EnvBuilder(with_pip=True).create(directory)
        python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--disable-pip-version-check",
                "--no-input",
                "-r",
                str(requirements),
            ],
            tool_environment(home),
            timeout=900,
        )
    print(json.dumps({"dependency_lock": "resolved", "lock_sha256": text_identity(requirements)}))


def main(argv: list[str] | None = None) -> int:
    """Purpose: Expose installer and crawler entry points
    Inputs: CLI options
    Outputs: Child exit status."""
    parser = argparse.ArgumentParser(description=__doc__, epilog=ATTRIBUTION)
    parser.add_argument(
        "--home",
        help="Absolute external cache root; environment override: MPL_CRAWL4AI_HOME",
    )
    parser.add_argument("command", choices=("install", "doctor", "crawl", "qualify", "check-lock"))
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected tool",
    )
    args = parser.parse_args(argv)
    remaining = args.arguments
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    home = cache_home(args.home)
    env = tool_environment(home)
    if args.command == "check-lock":
        if remaining:
            parser.error("check-lock takes no additional arguments")
        check_lock(home)
        return 0
    if args.command == "install":
        if remaining:
            parser.error("install takes no additional arguments")
        python = install(home)
        print(json.dumps({"installed": VERSION, "python": str(python)}))
        return 0
    if args.command == "doctor":
        if remaining:
            parser.error("doctor takes no additional arguments")
        doctor(home)
        return 0
    python = install(home)
    if args.command == "crawl":
        command = [str(python), "-m", "crawl4ai.cli", "crawl", *remaining]
    else:
        command = [str(python), "-m", "tools.crawl4ai_research", args.command, *remaining]
    completed = subprocess.run(command, env=env, timeout=1800)
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"crawl4ai: {error}", file=sys.stderr)
        raise SystemExit(1) from error
