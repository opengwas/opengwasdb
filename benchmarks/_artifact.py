"""Shared plumbing for the benchmarks that write a JSON artifact.

CONTRIBUTING asks for benchmark numbers to be re-run, never hand-edited, which
means every one of these scripts records the same three things about its own
run -- which commit produced it, when, and where the numbers went. Keeping that
in one place is what stops two artifacts disagreeing about what a field means.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
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


def host_load() -> dict[str, Any]:
    """The node's load averages and core count, as one reading."""
    one, five, fifteen = os.getloadavg()
    return {"load_1m": one, "load_5m": five, "load_15m": fifteen, "cpus": os.cpu_count()}


def tree_is_clean() -> bool:
    """True when the checkout has no uncommitted change, so `commit()` names the code run."""
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    return not status.stdout.strip()


def artifact_header(tool: str, argv: list[str]) -> dict[str, Any]:
    """What an artifact says about the run that made it, taken before it starts:
    the tool, its argv, the host load, and `provenance()` plus whether the tree
    was clean."""
    return {
        "tool": tool,
        "argv": argv,
        "host_load_start": host_load(),
        "worktree_clean": tree_is_clean(),
        **provenance(),
    }


class LoadSampler:
    """Samples `host_load()` every `period_s` seconds on a thread while a run is in progress.

    A load read only at the start and the end says nothing about a run that
    overlapped another job in between. Use as a context manager; `samples`
    holds each reading with its seconds since the start.
    """

    def __init__(self, period_s: float = 30.0) -> None:
        self.period_s = period_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = 0.0

    def _sample(self) -> None:
        self.samples.append({"seconds": round(time.monotonic() - self._started, 1), **host_load()})

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            self._sample()

    def __enter__(self) -> LoadSampler:
        self._started = time.monotonic()
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


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
