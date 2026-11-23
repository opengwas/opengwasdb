#!/usr/bin/env python3
"""#40's reads, using #8's runner with result fingerprints, distinct chunks and uptime.

Use latency.py's arguments, including `step1 --target local http10 http50`. Set
`--zarr-threads 1` and BLAS 1: seven workers share local I/O and numeric decoding,
with one fallback codec helper. Use an eight-CPU affinity cap inherited by HTTP children.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from collections import Counter
from typing import Any

import latency

from opengwasdb.ld.store.location import ChunkCache

CHUNKS: Counter[tuple[str, int]] = Counter()


def fingerprint(clumps: Any) -> str:
    """All result columns, including order, dtype and shape, not just the lead count."""
    digest = hashlib.sha256()
    for name, value in clumps.columns().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    digest.update(repr((clumps.absent, clumps.r2, clumps.window_bp)).encode())
    return digest.hexdigest()


def query(alids: list[str], z: Any, r2: float) -> Any:
    def run(store: Any) -> dict[str, Any]:
        clumps = store.clump(alids, z, r2=r2, kb=3000)
        return {"hits": len(alids), "leads": len(clumps.leads()), "sha256": fingerprint(clumps)}

    return run


def install() -> None:
    kept = ChunkCache._keep_run
    timed = latency.Bench.timed

    def keeping(self: Any, path: str, first: int, stop: int, size: int, data: Any) -> Any:
        CHUNKS.update((path, k) for k in range(first, stop))
        return kept(self, path, first, stop, size, data)

    def timing(self: Any, *args: Any, **kwargs: Any) -> Any:
        CHUNKS.clear()
        before = subprocess.check_output(["uptime"], text=True).strip()
        out = self.out
        self.out = io.StringIO()
        try:
            record = timed(self, *args, **kwargs)
        finally:
            self.out = out
        near = {k: n for k, n in CHUNKS.items() if k[0].startswith("near/") and k[0].endswith("/r")}
        record.update(
            uptime_before=before,
            uptime_after=subprocess.check_output(["uptime"], text=True).strip(),
            chunk_fetches=sum(CHUNKS.values()),
            distinct_chunks=len(CHUNKS),
            near_fetches=sum(near.values()),
            distinct_near_chunks=len(near),
        )
        out.write(json.dumps(record) + "\n")
        out.flush()
        return record

    ChunkCache._keep_run = keeping
    latency.Bench.timed = timing
    latency.clump_query = query


if __name__ == "__main__":
    install()
    latency.main()
