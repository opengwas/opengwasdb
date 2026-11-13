#!/usr/bin/env python3
"""Give an LD Store written before ADR 0007 its Far Pairs, derived from its own Proxy Index.

    pixi run -e dev python scripts/derive_far_pairs.py STORE [STORE ...] [--threads 8]

ogld#7 wrote the five HGDP+1kGP stores before the owner brought the Far Pairs back beside the
Proxy Index (ADR 0007, 12 November 2026). A store's Far Pairs are its index's entries beyond
near_bp in each pair's earlier row (spec §5), so each chromosome's are derived from its own
index by the store assembly (`ogld.store.assembly.derive_far_pairs`), which writes them through
the stream a build writes them through. On HGDP+1kGP EUR chr22 the derived `far/` was
byte-identical, file for file, to a rebuild's (ogld#31).

Nothing else in a store is written: no array, and not the provenance, which counts the Far Pairs
already (`counts.far_pairs`). They are written under `PARTIAL` and renamed to `far/` only when
every chromosome's are written and number the store's `counts.far_pairs`. So a run that fails,
or finds another count, leaves the store as it was, with nothing to clean up, and a `PARTIAL` a
killed run left is this script's own and is replaced. A store that holds `far/` is refused.
Validate each store afterwards: `ogld validate`, or `pixi run -e dev acceptance-stores` for the
five. The store's other files are only read, so a reader may read it while this runs: a reader
ignores a group it does not know (spec §1), and `far/` appears whole.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zarr

from ogld.store import StoreReader
from ogld.store.assembly import derive_far_pairs
from ogld.store.layout import chromosomes_listed

#: Where the Far Pairs are written until every chromosome's are, and counted: then it is renamed
#: `far/`. A reader ignores it (spec §1), and no other writer makes it.
PARTIAL = "far-partial"


def derive(store: Path) -> int:
    """Write `store`'s Far Pairs, chromosome by chromosome, from its Proxy Index; return how
    many. They take the name `far/` only when complete and numbering what its provenance
    records; otherwise nothing is left."""
    if (store / "far").exists():
        raise SystemExit(f"{store} holds far/ already: its Far Pairs are not derived again")
    partial = store / PARTIAL
    if partial.exists():  # a killed run's: complete runs and failures alike leave none
        shutil.rmtree(partial)
    reader = StoreReader.open(store)
    try:
        written = _write(store, reader)
        recorded = reader.provenance["counts"]["far_pairs"]
        if written != recorded:
            raise SystemExit(
                f"{store}: its index holds {written:,} Far Pairs, but its provenance counts "
                f"{recorded:,}; far/ is not written"
            )
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    partial.rename(store / "far")
    return written


def _write(store: Path, reader: StoreReader) -> int:
    """Every chromosome's Far Pairs, under `PARTIAL`, at the near_bp and in the encoding the
    store records."""
    root = zarr.open_group(store, mode="r+")
    position = root["variants/position"]
    assert isinstance(position, zarr.Array)
    written = 0
    for name, start, stop in chromosomes_listed(root["variants"].attrs["chromosomes"]):
        here = np.asarray(position[start:stop]).astype(np.int64)
        index = root[f"index/{name}"]
        assert isinstance(index, zarr.Group)
        far = root.require_group(f"{PARTIAL}/{name}")
        near_bp = reader.provenance["near_bp"]
        written += derive_far_pairs(index, far, here, near_bp, reader.encoding)
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
