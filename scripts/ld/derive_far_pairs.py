#!/usr/bin/env python3
"""Give an LD Store written before ADR 0006 its Far Pairs, derived from its own Proxy Index.

    pixi run -e dev python scripts/derive_far_pairs.py STORE [STORE ...] [--threads 8]

ogld#7 wrote the five HGDP+1kGP stores before the owner brought the Far Pairs back beside the
Proxy Index (ADR 0006, 12 November 2026). A store's Far Pairs are its index's entries beyond
near_bp in each pair's earlier row (spec §5), so each chromosome's are derived from its own
index by the store assembly (`ogld.store.assembly.derive_far_pairs`), which writes them through
the stream a build writes them through. On HGDP+1kGP EUR chr22 the derived `far/` was
byte-identical, file for file, to a rebuild's (ogld#31).

Nothing else in a store is written: no array, and not the provenance, which counts the Far Pairs
already (`counts.far_pairs`). A store that holds `far/` is refused, and so is one whose derived
Far Pairs do not number its `counts.far_pairs`; that one's `far/` is removed again. Validate each
store afterwards: `ogld validate`, or `pixi run -e dev acceptance-stores` for the five. The
store's other files are only read, so a reader may read it while this runs; `far/` appears
only as it is written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zarr

from ogld.format import REncoding
from ogld.store.assembly import derive_far_pairs
from ogld.store.layout import chromosomes_listed


def derive(store: Path) -> int:
    """Write `store`'s Far Pairs, chromosome by chromosome, from its Proxy Index; return how
    many. The count must be the one its provenance records."""
    if (store / "far").exists():
        raise SystemExit(f"{store} holds far/ already: its Far Pairs are not derived again")
    provenance = json.loads((store / "zarr.json").read_text())["attributes"]["ld_store"]
    encoding = REncoding.of(provenance["encoding"])
    root = zarr.open_group(store, mode="r+")
    position = root["variants/position"]
    assert isinstance(position, zarr.Array)
    written = 0
    for name, start, stop in chromosomes_listed(root["variants"].attrs["chromosomes"]):
        here = np.asarray(position[start:stop]).astype(np.int64)
        written += derive_far_pairs(root, name, here, provenance["near_bp"], encoding)
    recorded = provenance["counts"]["far_pairs"]
    if written != recorded:
        shutil.rmtree(store / "far")
        raise SystemExit(
            f"{store}: its index holds {written:,} Far Pairs, but its provenance counts "
            f"{recorded:,}; far/ is removed again"
        )
    return written


def uptime() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stores", nargs="+", type=Path)
    parser.add_argument("--threads", type=int, default=8, help="zarr's threads (8)")
    args = parser.parse_args(argv)
    zarr.config.set({"threading.max_workers": args.threads})
    for store in args.stores:
        print(f"{store}: start  {uptime()}", flush=True)
        started = time.perf_counter()
        count = derive(store)
        seconds = time.perf_counter() - started
        print(f"{store}: {count:,} Far Pairs in {seconds:.1f} s  {uptime()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
