#!/usr/bin/env python3
"""Give an LD Store written before ADR 0010 its ALID search index, from its own variant table.

    pixi run -e dev python scripts/derive_alid_index.py STORE [STORE ...]

ogld#7 wrote the five HGDP+1kGP stores before the owner's 15 November 2026 decision to carry the
engine's ALID index beside the variant table (ADR 0010). A store's index is its own `variants/alid`
column, padded and sorted (`opengwasdb.ld.store.assembly.derive_alid_index`, the function
`StoreWriter.finish` calls), so it is derived rather than rebuilt. On HGDP+1kGP EUR chr22 the
derived index was byte-identical, file for file, to a rebuild's (ogld#42).

**It is idempotent, and a half-derived store is repaired by running it again.** The steps are
ordered so that the provenance never counts an index the store lacks: the arrays are written under
`alid_index-partial/`, renamed to `alid_index` once every entry is written and their number is what
the table gives, and only then does the provenance record the count, as a new document renamed over
the old one. So a run that fails anywhere leaves a store validation refuses — a `partial` group, or
an index whose count is unrecorded — and the next run finishes it: a `partial` is replaced, and an
index already in place has its count recorded, once that count is shown to be this table's. A store
that is fully derived is left as it is. This is not one transaction across two renames: a reader
may open the store between them, so run this when no reader does. No array is written but the
index's two, and no other file but the root metadata document.

Validate each store afterwards: `ogld validate`, or `pixi run -e dev acceptance-stores` for the
five.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from opengwasdb.ld.store import layout
from opengwasdb.ld.store.assembly import ALID_ROWS, derive_alid_index

#: Where the index is written until every entry is, and counted: then it is renamed `alid_index`.
#: A reader ignores it (spec §1), and no other writer makes it.
PARTIAL = "alid_index-partial"


def derive(store: Path) -> int:
    """Give `store` the ALID index its table defines, or finish a half-derived one; return how many
    entries the index holds.

    A store whose index is already in place keeps it: its count is recorded if the provenance lacks
    it, and refused if the index holds a number of entries this table does not give. A `partial`
    left by a failed run is replaced. Nothing else in the store is written.
    """
    variants = store / "variants"
    index, partial = variants / "alid_index", variants / PARTIAL
    rows, wide = table_counts(store)
    recorded = recorded_count(store)
    if recorded is not None and recorded != wide:
        raise SystemExit(
            f"{store}: its provenance records {recorded:,} ALIDs left out of the index, but its "
            f"table has {wide:,} wider than the slot: this is not derived over that record"
        )
    if index.exists():
        entries = entry_count(index)
        if entries != rows - wide:
            raise SystemExit(
                f"{store}: its index holds {entries:,} entries, but its table has {rows:,} ALIDs "
                f"of which {wide:,} are too wide for the slot, so {rows - wide:,} belong in it"
            )
        if recorded is None:
            record_count(store, wide)
        return entries
    if partial.exists():  # a killed run's: complete runs and failures alike leave one to replace
        shutil.rmtree(partial)
    left_out, entries = write_partial(store)
    if (left_out, entries) != (wide, rows - wide):
        shutil.rmtree(partial, ignore_errors=True)
        raise SystemExit(
            f"{store}: its table gives {wide:,} ALIDs too wide for the slot and {rows - wide:,} "
            f"entries, but the index written has {left_out:,} and {entries:,}"
        )
    partial.rename(index)
    record_count(store, wide)
    return entries


def table_counts(store: Path) -> tuple[int, int]:
    """The table's rows, and how many of its ALIDs are wider than the index's slot: counted from
    `variants/alid` itself, block by block, never inferred from an index's length (spec §7)."""
    alid = zarr.open_array(store / "variants" / "alid", mode="r")
    rows = int(alid.shape[0])
    wide = 0
    for lo in range(0, rows, ALID_ROWS):
        wide += int((~layout.alid_fits(np.asarray(alid[lo : lo + ALID_ROWS]))).sum())
    return rows, wide


def entry_count(index: Path) -> int:
    """How many entries the index at `index` holds."""
    return int(zarr.open_array(index / "row", mode="r").shape[0])


def recorded_count(store: Path) -> int | None:
    """The ALIDs `store`'s provenance records the index leaving out, or None if it records none."""
    recorded = root_provenance(store)["counts"].get("alids_left_out")
    return None if recorded is None else int(recorded)


def root_provenance(store: Path) -> dict[str, Any]:
    """The root group's `ld_store` attribute, as recorded (spec §7)."""
    document: dict[str, Any] = json.loads((store / "zarr.json").read_text())
    return dict(document["attributes"]["ld_store"])


def record_count(store: Path, left_out: int) -> None:
    """Rewrite the store's provenance with `counts.alids_left_out` (spec §7), the one key this
    derivation adds. A new document renamed over the old one, so no reader sees a half one."""
    meta = store / "zarr.json"
    document = json.loads(meta.read_text())
    document["attributes"]["ld_store"]["counts"]["alids_left_out"] = left_out
    written = meta.with_suffix(".json.renamed")
    written.write_text(json.dumps(document))
    os.replace(written, meta)


def write_partial(store: Path) -> tuple[int, int]:
    """The index, under `partial`, from the store's table: how many ALIDs it leaves out and how
    many entries it holds."""
    root = zarr.open_group(store, mode="r+")
    variants = root["variants"]
    assert isinstance(variants, zarr.Group)
    group = variants.create_group(PARTIAL)
    left_out = derive_alid_index(variants, group)
    return left_out, int(group["row"].shape[0])


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
        print(
            f"{store}: {entries:,} entries in {time.perf_counter() - started:.1f} s  {uptime()}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
