"""Pinned reproducible release build for model-protocol-lab.

Builds the sdist and the wheel twice, from two independent copies of the
source tree, and requires identical SHA-256 hashes for both artifacts.
Filesystem timestamps otherwise leak into wheel entry dates and sdist tar
metadata (verified in the cycle-201 release-readiness audit: two copies
of the same tree produced different artifacts; under SOURCE_DATE_EPOCH
the wheel matched but the sdist still carried raw member mtimes and the
build user's name). The procedure here removes both leaks:

1. Copy the tree (no .git, venvs, caches, or prior build output).
2. Normalize every file and directory mtime to the epoch.
3. Build the sdist with SOURCE_DATE_EPOCH set, then rewrite the tar
   deterministically (member mtime = epoch, uid/gid = 0, empty
   uname/gname, gzip header mtime = 0).
4. Build the wheel from the normalized sdist, so one canonical input
   produces the wheel.
5. Repeat from a second copy; compare hashes. On a mismatch, name the
   members that differ instead of rebuilding until it happens to match.

Usage: python tools/release_build.py --source . --output dist \
    --epoch 1700000000
The epoch for a real release is the tagged commit's timestamp
(`SOURCE_DATE_EPOCH` convention); any fixed value demonstrates the
reproducibility property.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

_EXCLUDED_NAMES = {
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    ".tox",
}


def _ignore(directory: str, names: list[str]) -> list[str]:
    ignored = []
    for name in names:
        if name in _EXCLUDED_NAMES or name.endswith(".egg-info"):
            ignored.append(name)
    return ignored


def _copy_tree(source: Path, dest: Path) -> None:
    shutil.copytree(source, dest, ignore=_ignore)


def _normalize_mtimes(root: Path, epoch: int) -> None:
    for path in sorted(root.rglob("*")):
        os.utime(path, (epoch, epoch))
    os.utime(root, (epoch, epoch))


def normalize_sdist_bytes(data: bytes, epoch: int) -> bytes:
    """Rewrite a .tar.gz sdist deterministically; pure function of the input.

    Every member gets mtime=epoch, uid=gid=0 and empty uname/gname; the
    gzip wrapper gets mtime=0. Member order and contents are preserved.
    Applying it twice yields the same bytes as applying it once.
    """
    raw = gzip.decompress(data)
    out_buffer = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as source,
        tarfile.open(fileobj=out_buffer, mode="w", format=tarfile.PAX_FORMAT) as target,
    ):
        for member in source.getmembers():
            member.mtime = epoch
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            # Files setuptools generates during the build carry a PAX
            # extended header with the build-time float mtime, which
            # addfile would re-emit over the normalized header field —
            # drop every time key so the epoch above is the only one.
            member.pax_headers = {
                key: value
                for key, value in member.pax_headers.items()
                if key not in ("mtime", "atime", "ctime")
            }
            extracted = source.extractfile(member) if member.isfile() else None
            target.addfile(member, extracted)
    gzip_buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=gzip_buffer, mtime=0) as handle:
        handle.write(out_buffer.getvalue())
    return gzip_buffer.getvalue()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_digests(path: Path) -> dict[str, str]:
    """Per-member content digests, for naming what differs on a mismatch."""
    members: dict[str, str] = {}
    if path.suffix == ".whl" or path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                members[info.filename] = hashlib.sha256(archive.read(info.filename)).hexdigest()
    else:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        members[member.name] = hashlib.sha256(extracted.read()).hexdigest()
    return members


def _report_mismatch(label: str, first: Path, second: Path) -> None:
    print(f"MISMATCH: {label} artifacts differ", file=sys.stderr)
    print(f"  first : {_sha256(first)}  {first}", file=sys.stderr)
    print(f"  second: {_sha256(second)}  {second}", file=sys.stderr)
    first_members = _member_digests(first)
    second_members = _member_digests(second)
    for name in sorted(set(first_members) | set(second_members)):
        if first_members.get(name) != second_members.get(name):
            print(f"  differing member: {name}", file=sys.stderr)


def _build_once(source: Path, work: Path, epoch: int) -> tuple[Path, Path]:
    tree = work / "tree"
    _copy_tree(source, tree)
    _normalize_mtimes(tree, epoch)
    dist = work / "dist"
    dist.mkdir(parents=True)
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = str(epoch)
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(dist), str(tree)],
        check=True,
        env=env,
    )
    sdist = next(dist.glob("*.tar.gz"))
    sdist.write_bytes(normalize_sdist_bytes(sdist.read_bytes(), epoch))
    unpacked = work / "unpacked"
    unpacked.mkdir()
    with tarfile.open(sdist, "r:gz") as archive:
        archive.extractall(unpacked, filter="data")
    package_dir = next(unpacked.iterdir())
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist), str(package_dir)],
        check=True,
        env=env,
    )
    wheel = next(dist.glob("*.whl"))
    return sdist, wheel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args(argv)

    source = args.source.resolve()
    results: list[tuple[Path, Path]] = []
    with tempfile.TemporaryDirectory(prefix="mpl_release_") as tmp:
        for run in (1, 2):
            results.append(_build_once(source, Path(tmp) / f"run{run}", args.epoch))
        (sdist_a, wheel_a), (sdist_b, wheel_b) = results
        ok = True
        if _sha256(sdist_a) != _sha256(sdist_b):
            _report_mismatch("sdist", sdist_a, sdist_b)
            ok = False
        if _sha256(wheel_a) != _sha256(wheel_b):
            _report_mismatch("wheel", wheel_a, wheel_b)
            ok = False
        if not ok:
            return 1
        output = args.output
        output.mkdir(parents=True, exist_ok=True)
        for artifact in (sdist_a, wheel_a):
            target = output / artifact.name
            shutil.copyfile(artifact, target)
            print(f"{_sha256(target)}  {target}")
    print("reproducible: both builds produced identical artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
