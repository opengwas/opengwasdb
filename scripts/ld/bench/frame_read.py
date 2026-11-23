#!/usr/bin/env python3
"""What one Partner Rows frame read costs, local and through the store (partner-rows §7; ogld#70).

A frame is read one of two ways: `PreadSource` on a descriptor the reader keeps, for a store on
this disk, and `RangeSource` through the array's own store path, which over HTTP is one Range
request. The difference is the whole of a local read's cost, so this measures both on the same
file, warmed:

    pixi run -e dev python scripts/ld/bench/frame_read.py [--bytes 64M] [--reads 4000] [--uptime]

It builds a temporary one-chunk `uint8` array whose chunk file is `--bytes` long and whose frames
are `--frame-bytes` each (25,600 by default: 20 KiB compressed is EUR chr22's median frame), then
times `--reads` random frame reads through `PreadSource` and through `location.RangeSource`, which
is the store path a remote store uses, and the one a local store paid before ADR 0095's direct
decoder. Interleave the
two, report the median of `--runs`, and print µs/read and the projection onto 3,850 frames (the
reads GCST90476460's clump made, #65). Only the I/O is timed: nothing is decompressed and no frame
is decoded. Nothing is committed.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Final

import zarr

from opengwasdb.ld.store import layout
from opengwasdb.ld.store.location import PreadSource, RangeSource

#: The frames GCST90476460's clump read (#65): what a per-read cost is projected onto.
FRAMES: Final = 3_850
#: EUR chr22's median frame's compressed size, in bytes (§5.2: about 20 KiB).
FRAME_BYTES: Final = 25_600


def built(root: Path, length: int) -> zarr.Array[object]:
    """One chunk of `length` bytes, at the path the array's metadata names (§5.1): the file a
    frame is read out of, filled with bytes no frame's header begins."""
    group = zarr.open_group(str(root), mode="w", zarr_format=3)
    array = layout.new_frames_array(group, length)
    path = layout.chunk_file(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    block = zlib.compress(os.urandom(1 << 20), 1)
    with path.open("wb") as held:
        written = 0
        while written < length:
            piece = block[: length - written]
            held.write(piece)
            written += len(piece)
    return array


def timed(source: PreadSource | RangeSource, offsets: list[int], frame_bytes: int) -> float:
    """Seconds one run of `offsets`' reads took, refused unless every read was whole."""
    start = time.perf_counter()
    for at in offsets:
        held = source.read(at, at + frame_bytes)
        if len(held) != frame_bytes:
            raise SystemExit(f"a short read: {len(held)} of {frame_bytes} bytes at {at}")
    return time.perf_counter() - start


def measured(
    frame_bytes: int, length: int, offsets: list[int], runs: int
) -> dict[str, list[float]]:
    """The seconds each source's `runs` runs of `offsets` took, interleaved, on one file."""
    with tempfile.TemporaryDirectory(prefix="ogld-frame-read-") as made:
        array = built(Path(made), length)
        # Warm the page cache with one sequential pass, so both paths are timed warm.
        with layout.chunk_file(array).open("rb") as held:
            while held.read(1 << 22):
                pass
        sources = {"pread": PreadSource(layout.chunk_file(array)), "range": RangeSource(array)}
        seconds: dict[str, list[float]] = {name: [] for name in sources}
        try:
            for _ in range(runs):
                for name, source in sources.items():
                    seconds[name].append(timed(source, offsets, frame_bytes))
        finally:
            for source in sources.values():
                source.close()
    return seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=64 << 20)
    parser.add_argument("--frame-bytes", type=int, default=FRAME_BYTES)
    parser.add_argument("--reads", type=int, default=4_000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=70)
    parser.add_argument("--uptime", action="store_true", help="print the load averages")
    args = parser.parse_args()

    if args.uptime:
        print("uptime:", Path("/proc/loadavg").read_text().strip())
    rng = random.Random(args.seed)
    offsets = [rng.randrange(0, args.bytes - args.frame_bytes - 1) for _ in range(args.reads)]
    seconds = measured(args.frame_bytes, args.bytes, offsets, args.runs)
    print(f"{args.reads} reads of {args.frame_bytes} bytes, median of {args.runs} interleaved runs")
    for name, runs in seconds.items():
        each = statistics.median(runs) / args.reads * 1e6
        print(
            f"  {name:5}  {each:8.1f} µs/read  {each * FRAMES / 1e6:6.2f} s for {FRAMES} frames"
            f"   runs: {' '.join(f'{run:.3f}' for run in runs)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
