"""Shared plumbing for the benchmarks that write a JSON artifact.

CONTRIBUTING asks for benchmark numbers to be re-run, never hand-edited, which
means every one of these scripts records the same three things about its own
run -- which commit produced it, when, and where the numbers went. Keeping that
in one place is what stops two artifacts disagreeing about what a field means.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
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


def reflink_copy(source: Path, destination: Path) -> None:
    """Copy a release, sharing extents where the filesystem can.

    A reflink makes the copy near-free until one side is written, which is what
    makes "measure against a copy, keep the original" affordable for a store
    measured in tens of gigabytes. Refuses an existing destination rather than
    merging into it, and falls back to a full copy where the filesystem cannot
    share extents.
    """
    if destination.exists():
        raise SystemExit(f"{destination}: already exists; refusing to overwrite")
    subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(destination)], check=True)


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
