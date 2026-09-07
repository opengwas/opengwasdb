"""Shared plumbing for the benchmarks that write a JSON artifact.

CONTRIBUTING asks for benchmark numbers to be re-run, never hand-edited, which
means every one of these scripts records the same three things about its own
run -- which commit produced it, when, and where the numbers went. Keeping that
in one place is what stops two artifacts disagreeing about what a field means.
"""

from __future__ import annotations

import json
import subprocess
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
