"""Shared RSS probing for the query benchmarks.

Lives here rather than in one harness because the probe method is the
contract the reports rest on: each query shape runs in a fresh interpreter,
`baseline` is RSS with the store open before the query, `peak` is sampled on a
background thread *during* it (a query's intermediate buffers are freed before
it returns, so a before/after read misses the peak entirely), and RSS comes
from `/proc/self/statm` rather than `ru_maxrss`, which is a lifetime
high-water mark that cannot be reset. Two harnesses measure the same way or
their numbers are not comparable — so they share this one implementation.

The harness-specific half stays with the harness: which store to open, which
query shapes to build, and what extra argv the re-invocation needs.
`run_probe` re-invokes the *current* script with `--rss-shape <name>` and the
caller's extra argv, and reads back the single JSON line it prints.
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

_PAGE_KB = resource.getpagesize() / 1024.0


def rss_mb() -> float:
    """Current (not high-water) RSS of this process in MB."""
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * _PAGE_KB / 1024.0


class RssSampler:
    """Poll RSS on a background thread and keep the maximum seen.

    The 5 ms interval trades resolution against perturbing the measurement;
    it resolves the sub-second shapes without measurably slowing them.
    """

    def __init__(self, interval: float = 0.005) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self.peak_mb = 0.0

    def __enter__(self) -> RssSampler:
        self.peak_mb = rss_mb()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, rss_mb())
            self._stop.wait(self._interval)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.peak_mb = max(self.peak_mb, rss_mb())


def sample_query(fn: Callable[[], dict[str, Any]]) -> dict[str, float]:
    """Run one query callable, returning its baseline/peak/delta RSS and count."""
    baseline = rss_mb()
    with RssSampler() as sampler:
        result = fn()
    peak = max(sampler.peak_mb, rss_mb())
    return {
        "baseline_mb": round(baseline, 1),
        "peak_mb": round(peak, 1),
        "delta_mb": round(peak - baseline, 1),
        "result_count": len(result["z"]),
    }


def run_probe(shape: str, extra_argv: list[str]) -> dict[str, Any]:
    """Re-invoke the current script for one shape's RSS record."""
    out = subprocess.run(
        [sys.executable, str(Path(sys.argv[0]).resolve()), "--rss-shape", shape, *extra_argv],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"RSS probe for {shape!r} failed:\n{out.stdout}\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])
