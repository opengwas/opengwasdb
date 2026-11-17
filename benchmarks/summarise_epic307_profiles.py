#!/usr/bin/env python3
"""Summarise #322's existing py-spy raw recordings; never run a Store operation.

Non-native samples name caller frames, not interpreter time. Only --native
recordings distinguish compiled leaves from Python frames. Pool native counts
before taking shares: equal-length recordings need not contain equal samples.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from benchmarks._artifact import provenance, write_artifact


def _samples(path: Path) -> Iterator[tuple[list[str], int]]:
    for line in path.read_text(errors="replace").splitlines():
        stack, _, count = line.rpartition(" ")
        if count.isdigit():
            yield stack.split(";"), int(count)


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _native_kind(leaf: str) -> str:
    if re.search(r"\([^()]*\.py(?::\d+)?\)$", leaf):
        return "interpreted Python frame"
    if "openblas" in leaf.lower():
        return "OpenBLAS"
    if "_cd_fast" in leaf or "_cython_blas" in leaf:
        return "scikit-learn Cython"
    if "numpy" in leaf:
        return "numpy C"
    return "other native"


def native_summary(paths: Sequence[Path]) -> dict[str, Any]:
    total: Counter[str] = Counter()
    recordings = []
    for path in paths:
        counts: Counter[str] = Counter()
        for frames, count in _samples(path):
            counts[_native_kind(frames[-1])] += count
        if not counts:
            raise ValueError(f"{path}: no samples")
        total.update(counts)
        recordings.append({**_identity(path), "samples": dict(counts)})
    size = sum(total.values())
    if not size:
        raise ValueError("no samples in native recordings")
    return {
        "recordings": recordings,
        "total_samples": size,
        "samples": dict(total),
        "shares_percent": {key: round(100 * count / size, 1) for key, count in total.items()},
    }


def python_summary(path: Path) -> dict[str, Any]:
    roles: dict[str, Counter[str]] = defaultdict(Counter)
    for stack, count in _samples(path):
        depth = sum(frame.startswith("process ") for frame in stack)
        role = {1: "parent", 2: "worker"}.get(depth, f"depth{depth}")
        frames = [frame for frame in stack if not frame.startswith(("process ", "thread "))]
        if frames:
            leaf = re.sub(r":\d+\)$", ")", frames[-1])
            roles[role][leaf] += count
    if not roles:
        raise ValueError(f"{path}: no samples")
    return {
        **_identity(path),
        "roles": {
            role: {"samples": sum(counts.values()), "leaf_frames": dict(counts.most_common(12))}
            for role, counts in roles.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="existing run/e2e directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshots = []
    index = args.run_dir / "prof/index.tsv"
    for line in index.read_text().splitlines():
        stamp, step, code, name, phase, *_ = line.split("\t")
        if code == "0":
            snapshots.append(
                {
                    "stamp": stamp,
                    "step": step,
                    "last_phase_line": phase,
                    **python_summary(index.parent / name),
                }
            )
    write_artifact(
        args.output,
        {
            "generator": "benchmarks/summarise_epic307_profiles.py",
            **provenance(),
            "method": "40 s non-native whole-tree snapshots at 50 Hz every 20 min; three 10 s "
            "native single-worker snapshots at 20 Hz. Active samples, not wall-time shares; "
            "non-native leaves may be compiled calls. Native classification uses the "
            "leaf's file; other native includes interpreter internals and gc.",
            "index": _identity(index),
            "python_snapshots": snapshots,
            "native_block_workers": native_summary(
                sorted((args.run_dir / "prof-native").glob("*.txt"))
            ),
        },
    )


if __name__ == "__main__":
    main()
