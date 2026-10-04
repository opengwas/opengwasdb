"""How long one real `[1000, 1000]` chunk takes to decode, with and without Blosc's threads.

zarr 3 sets `numcodecs.blosc.use_threads = False` for the whole process when it
is imported, so every chunk decodes single-threaded; #244 turns the threads back
on (ADR 0056). This decodes one stored OGS-00009 `z` chunk (int16, 2 MB
uncompressed, 16 Blosc blocks) `--reps` times per setting, in this process,
with zarr not imported, and reports the median in microseconds:

  default_us                  numcodecs' own default for the main thread
  use_threads=<flag>,nthreads=<n>_us  for use_threads False/True and 1, 4, 8 threads

    pixi run -e dev python benchmarks/zarr3_blosc_decode.py \\
        --chunk /data/opengwasdb/stores/OGS-00009/store.opengwasdb/data.zarr/z/6543.1 \\
        --output docs/benchmark-output/opengwasdb_zarr3_read_levers/blosc_decode.json
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numcodecs
from numcodecs import blosc

from benchmarks._artifact import provenance, write_artifact


def median_us(codec: Any, raw: bytes, reps: int) -> float:
    codec.decode(raw)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        codec.decode(raw)
        samples.append((time.perf_counter() - t0) * 1e6)
    return round(statistics.median(samples), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chunk", type=Path, required=True, help="one stored Blosc chunk file")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    raw = args.chunk.read_bytes()
    codec = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)
    out: dict[str, Any] = {
        **provenance(),
        "chunk": str(args.chunk),
        "reps": args.reps,
        "numcodecs": numcodecs.__version__,
        "blosc_version": getattr(blosc, "VERSION_STRING", "?"),
        "compressed_bytes": len(raw),
        "uncompressed_bytes": len(codec.decode(raw)),
        "nthreads_default": blosc.get_nthreads(),
        "use_threads_default": blosc.use_threads,
        "cpu_count": os.cpu_count(),
        "load_1m_before": os.getloadavg()[0],
        "BLOSC_NTHREADS": os.environ.get("BLOSC_NTHREADS"),
    }
    out["default_us"] = median_us(codec, raw, args.reps)
    for flag in (False, True):
        blosc.use_threads = flag
        for nthreads in (1, 4, 8):
            blosc.set_nthreads(nthreads)
            out[f"use_threads={flag},nthreads={nthreads}_us"] = median_us(codec, raw, args.reps)
    out["load_1m_after"] = os.getloadavg()[0]
    if args.output is not None:
        write_artifact(args.output, out)
    else:
        print(out)


if __name__ == "__main__":
    main()
