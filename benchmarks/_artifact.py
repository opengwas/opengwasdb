"""Shared plumbing for the benchmarks that write a JSON artifact.

CONTRIBUTING asks for benchmark numbers to be re-run, never hand-edited, which
means every one of these scripts records the same three things about its own
run -- which commit produced it, when, and where the numbers went. Keeping that
in one place is what stops two artifacts disagreeing about what a field means.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def commit() -> str:
    """The short SHA the measurement was taken at, or empty outside a checkout."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def tree_fingerprint(root: str | Path) -> str:
    """A sha256 over an `opengwasdb` package's Python sources, in path order.

    `commit()` reads the worktree's `HEAD`, which is not the code a run
    imported when another tree is injected ahead of it on `sys.path`: #253's
    base harness record said `328f536` while it ran `5cf7f78`. The fingerprint
    names the code itself, so an artifact cannot claim a revision it did not
    run. It is stable under a re-run and changes with any source edit, in a
    checkout or not.
    """
    package = Path(root) / "opengwasdb"
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _digest(files: list[tuple[tuple[str, ...], bytes]]) -> str:
    digest = hashlib.sha256()
    for parts, content in sorted(files, key=lambda item: item[0]):
        digest.update("/".join(parts).encode())
        digest.update(content)
    return digest.hexdigest()


def is_ancestor(ancestor: str, descendant: str, *, repo: str | Path | None = None) -> bool | None:
    """Whether `ancestor` is in `descendant`'s history; `None` when git cannot say.

    `git merge-base --is-ancestor` exits 0 for yes and 1 for no, and anything
    else (an unknown revision, a shallow clone that lacks it) is not a "no".
    """
    found = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    if found.returncode == 0:
        return True
    return False if found.returncode == 1 else None


def export_revision(
    revision: str, destination: Path, *, repo: str | Path | None = None
) -> dict[str, Any]:
    """Export `revision`'s `opengwasdb` with `git archive`, fingerprinted from the archive.

    The fingerprint is computed from the archive's own bytes, in memory, before
    anything is written, so it names the commit's code even if the exported
    tree is later edited or a working tree holds uncommitted changes. The
    exported tree is re-read and must agree with it.
    """
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode != 0:
        raise SystemExit(f"--control-revision {revision!r} is not a commit in this repository")
    sha = resolved.stdout.strip()
    archive = subprocess.run(
        ["git", "archive", sha, "opengwasdb"], cwd=repo, capture_output=True, check=False
    )
    if archive.returncode != 0:
        raise SystemExit(
            f"git archive {sha} opengwasdb failed: {archive.stderr.decode(errors='replace')[-300:]}"
        )
    files: list[tuple[tuple[str, ...], bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".py"):
                extracted = tar.extractfile(member)
                assert extracted is not None
                files.append((tuple(member.name.split("/")[1:]), extracted.read()))
        if destination.exists():
            raise SystemExit(f"{destination}: already exists; refusing to overwrite")
        destination.mkdir(parents=True)
        tar.extractall(destination, filter="data")
    fingerprint = _digest(files)
    if tree_fingerprint(destination) != fingerprint:
        raise SystemExit(f"{destination}: the exported tree does not match git archive {sha}")
    return {
        "module_path": destination,
        "revision": sha,
        "opengwasdb_fingerprint": fingerprint,
        "fingerprint_basis": f"git archive {sha}",
    }


def package_fingerprint() -> tuple[str, str]:
    """(the path of the `opengwasdb` actually imported, a fingerprint of it)."""
    import opengwasdb

    package = Path(opengwasdb.__file__).resolve().parent
    return str(package), tree_fingerprint(package.parent)


def provenance() -> dict[str, str]:
    """The measured commit and wall-clock time an artifact records, so an older
    JSON cannot be mistaken for a current measurement, plus the path and
    fingerprint of the `opengwasdb` this process actually imported."""
    path, fingerprint = package_fingerprint()
    return {
        "commit": commit(),
        "measured_at": datetime.now(UTC).isoformat(),
        "opengwasdb_path": path,
        "opengwasdb_fingerprint": fingerprint,
    }


def scratch_copy(source: Path, destination: Path) -> str:
    """Copy a release into scratch, returning the copy method actually used.

    A reflink makes the copy near-free until one side is written, which is what
    makes "measure against a copy, keep the original" affordable for a store
    measured in tens of gigabytes. The reflink is **verified**: the first attempt
    is ``cp --reflink=always``, which fails rather than silently falling back on
    a filesystem that cannot share extents; the retry is a documented full copy.
    The return value names which happened (``"reflink"`` or ``"full_copy"``), so
    an artifact can record a verified method instead of an assumed one. Refuses
    an existing destination rather than merging into it.
    """
    if destination.exists():
        raise SystemExit(f"{destination}: already exists; refusing to overwrite")
    reflink = subprocess.run(
        ["cp", "-a", "--reflink=always", str(source), str(destination)],
        capture_output=True,
        text=True,
    )
    if reflink.returncode == 0:
        return "reflink"
    shutil.rmtree(destination, ignore_errors=True)
    subprocess.run(["cp", "-a", str(source), str(destination)], check=True)
    return "full_copy"


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def add_labelled_store_option(parser: Any, *, help: str = "a store to measure; repeatable") -> None:
    """Add the `--store LABEL=PATH` option the per-store harnesses share."""
    parser.add_argument(
        "--store", action="append", required=True, metavar="LABEL=PATH", help=help
    )


def labelled_stores(entries: list[str]) -> list[tuple[str, Path]]:
    """Parse `LABEL=PATH` entries, refusing one without a path."""
    stores: list[tuple[str, Path]] = []
    for entry in entries:
        label, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--store needs LABEL=PATH, got {entry!r}")
        stores.append((label, Path(path)))
    return stores
