#!/usr/bin/env python3
"""Copy the arrays a Dense lookup reads at a smaller variant-axis inner chunk.

The destination is a benchmark overlay, not a new publishable Store Release:
unchanged envelope files and unrelated Zarr groups are symlinks, and the copied
manifest's historical layout record is deliberately left untouched.  Every
copied cell is read back and compared with its source before publication.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.store.arrays import (
    ArrayRole,
    create_array,
    open_group,
    open_group_for_write,
    sharded_compressor,
)

PLANES = ("z", "se", "eaf", "imputed")
SIDE_ARRAYS = ("eaf_baseline", "eaf_reference")
UNCHANGED_ARRAYS = (
    "eaf_exception_index",
    "eaf_exception_value",
    "se_coefficients",
    "se_exception_index",
    "se_exception_value",
    "z_overflow_index",
    "z_overflow_value",
)


def _link(source: Path, destination: Path) -> None:
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _copy_chunked_array(
    source: Any,
    destination_group: Any,
    name: str,
    *,
    role: ArrayRole,
    variant_chunk: int,
    block_cells: tuple[int, int] | int,
) -> dict[str, object]:
    shape = tuple(int(x) for x in source.shape)
    shards = tuple(int(x) for x in source.shards)
    inner = (variant_chunk, *(int(x) for x in source.chunks[1:]))
    started = time.perf_counter()
    destination = create_array(
        destination_group,
        name,
        role,
        shape=shape,
        dtype=source.dtype,
        fill_value=source.fill_value,
        compressor=sharded_compressor(),
        inner_chunk=inner,
        shards=shards,
    )
    blocks = 0
    if len(shape) == 2:
        assert isinstance(block_cells, tuple)
        row_step, col_step = block_cells
        ranges = (
            (
                slice(r0, min(r0 + row_step, shape[0])),
                slice(c0, min(c0 + col_step, shape[1])),
            )
            for r0 in range(0, shape[0], row_step)
            for c0 in range(0, shape[1], col_step)
        )
    else:
        assert isinstance(block_cells, int)
        ranges = (
            (slice(r0, min(r0 + block_cells, shape[0])),)
            for r0 in range(0, shape[0], block_cells)
        )
    for selection in ranges:
        raw = np.asarray(source[selection])
        destination[selection] = raw
        observed = np.asarray(destination[selection])
        if observed.dtype != raw.dtype or observed.shape != raw.shape or not np.array_equal(
            observed.view(np.uint8), raw.view(np.uint8)
        ):
            raise RuntimeError(f"{name}: copied codes differ in {selection!r}; refusing publication")
        blocks += 1
        print(
            f"{name}: block {blocks}, rows {selection[0].start}:{selection[0].stop}, "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return {
        "name": name,
        "shape": list(shape),
        "source_inner_chunk": list(source.chunks),
        "inner_chunk": list(inner),
        "shard": list(shards),
        "blocks_verified": blocks,
        "seconds": time.perf_counter() - started,
    }


def rechunk(source_path: Path, destination_path: Path, variant_chunk: int) -> None:
    if variant_chunk <= 0 or 100_000 % variant_chunk:
        raise ValueError("the variant chunk must be positive and divide the 100,000-row shard")
    if destination_path.exists():
        raise FileExistsError(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging = destination_path.with_name(f".{destination_path.name}.tmp.{os.getpid()}")
    started = time.perf_counter()
    records: list[dict[str, object]] = []
    try:
        staging.mkdir()
        data_source = source_path / "data.zarr"
        data_destination = staging / "data.zarr"
        data_destination.mkdir()
        for entry in source_path.iterdir():
            if entry.name != "data.zarr":
                _link(entry, staging / entry.name)
        shutil.copy2(data_source / "zarr.json", data_destination / "zarr.json")
        copied = {*PLANES, *SIDE_ARRAYS}
        for entry in data_source.iterdir():
            if entry.name != "zarr.json" and entry.name not in copied:
                _link(entry, data_destination / entry.name)
            elif entry.name in UNCHANGED_ARRAYS:
                _link(entry, data_destination / entry.name)
        source_group = open_group(data_source, mode="r")
        destination_group = open_group_for_write(data_destination, "a", zarr_format=3)
        for name in PLANES:
            array = source_group[name]
            role = ArrayRole.DENSE_IMPUTED_MASK if name == "imputed" else ArrayRole.DENSE_STATISTIC_PLANE
            records.append(
                _copy_chunked_array(
                    array,
                    destination_group,
                    name,
                    role=role,
                    variant_chunk=variant_chunk,
                    block_cells=tuple(int(x) for x in array.shards),
                )
            )
        for name in SIDE_ARRAYS:
            array = source_group[name]
            records.append(
                _copy_chunked_array(
                    array,
                    destination_group,
                    name,
                    role=ArrayRole.PER_VARIANT,
                    variant_chunk=variant_chunk,
                    block_cells=int(array.shards[0]),
                )
            )
        report = {
            "source": str(source_path),
            "destination": str(destination_path),
            "scope": "full variant and Analysis axes; lookup arrays only",
            "variant_inner_chunk": variant_chunk,
            "analysis_inner_chunk": records[0]["inner_chunk"][1],
            "arrays": records,
            "seconds": time.perf_counter() - started,
            "publication": "every copied block read back and equal before staging rename",
        }
        report_path = destination_path.with_name(f"{destination_path.name}.json")
        if report_path.exists():
            raise FileExistsError(report_path)
        staging.rename(destination_path)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--variant-chunk", type=int, default=100)
    args = parser.parse_args()
    rechunk(args.source, args.destination, args.variant_chunk)


if __name__ == "__main__":
    main()
