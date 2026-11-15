#!/usr/bin/env python3
"""Give an LD Store written before ADR 0010 its ALID search index, from its own variant table.

    pixi run -e dev python scripts/derive_alid_index.py STORE [STORE ...]

ogld#7 wrote the five HGDP+1kGP stores before the owner's 15 November 2026 decision to carry the
engine's ALID index beside the variant table (ADR 0010). A store's index is its own `variants/alid`
column, padded and sorted (`ogld.store.assembly.derive_alid_index`, the function
`StoreWriter.finish` calls), so it is derived rather than rebuilt. On HGDP+1kGP EUR chr22 the
derived index was byte-identical, file for file, to a rebuild's (ogld#42).

Nothing else in a store is written: no array, and not the provenance, which counts the ALIDs the
index leaves out already (`counts.alids_left_out`). The arrays are written under `PARTIAL` and the
group is renamed to `alid_index` only when every entry is written and the count matches the
provenance. So a run that fails, or finds another count, leaves the store as it was, with nothing
to clean up, and a `PARTIAL` a killed run left is this script's own and is replaced. A store that
holds `alid_index` is refused. Validate each store afterwards: `ogld validate`, or
`pixi run -e dev acceptance-stores` for the five. The store's other files are only read, so a
reader may read it while this runs: a reader ignores a group it does not know (spec §1), and
`alid_index` appears whole.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import zarr

from ogld.store import StoreReader
from ogld.store.assembly import derive_alid_index

#: Where the index is written until every entry is, and counted: then it is renamed `alid_index`.
#: A reader ignores it (spec §1), and no other writer makes it.
PARTIAL = "alid_index-partial"


def derive(store: Path) -> int:
    """Write `store`'s ALID index from its own variant table; return how many ALIDs it leaves out.
    It takes the name `alid_index` only when complete and numbering what its provenance records;
    otherwise nothing is left."""
    if (store / "variants" / "alid_index").exists():
        raise SystemExit(f"{store} holds an ALID index already: it is not derived again")
    partial = store / "variants" / PARTIAL
    if partial.exists():  # a killed run's: complete runs and failures alike leave none
        shutil.rmtree(partial)
    reader = StoreReader.open(store)
    recorded = reader.provenance["counts"]["alids_left_out"]
    try:
        left_out, entries = _write(store)
        if left_out != recorded:
            raise SystemExit(
                f"{store}: its table has {left_out:,} ALID(s) wider than the index's slot, but "
                f"its provenance records {recorded:,}; the index is not written"
            )
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    partial.rename(store / "variants" / "alid_index")
    return entries


def _write(store: Path) -> tuple[int, int]:
    """The index, under `partial`, from the store's table: how many ALIDs it leaves out, and how
    many entries it holds."""
    root = zarr.open_group(store, mode="r+")
    variants = root["variants"]
    assert isinstance(variants, zarr.Group)
    partial = variants.create_group(PARTIAL)
    left_out = derive_alid_index(variants, partial)
    entries = int(partial["row"].shape[0])
    return left_out, entries


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
        entries = derive(store)
        seconds = time.perf_counter() - started
        print(f"{store}: {entries:,} entries in {seconds:.1f} s  {uptime()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
