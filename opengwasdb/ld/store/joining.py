"""Join immutable chromosome outputs behind StoreWriter (ADR 0015)."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import zarr

from ogld.format import REncoding, Sourced
from ogld.format.geometry import chromosome_ranges, chromosome_ranges_from_counts
from ogld.identity import chromosome_rank
from ogld.store import layout
from ogld.store.assembly import Cutter, Tally
from ogld.store.reader import StoreReader


def _compatible(first: dict[str, Any], record: dict[str, Any]) -> None:
    for key in ("encoding", "near_bp", "far_bp", "t_far", "t_partner", "reference_assembly"):
        if record[key] != first[key]:
            raise ValueError(f"chromosome stores disagree on {key}")
    for key, field in (("panel", "n"), ("adjustment", "k")):
        if record[key][field] != first[key][field]:
            raise ValueError(f"chromosome stores disagree on {key}.{field}")


def joined(
    location: Path,
    paths: Sequence[Path],
    sourced: Sourced,
) -> tuple[zarr.Group, Cutter, dict[str, tuple[int, int]], Tally]:
    """Retain chromosome-local component coordinates; only variant row ranges shift."""
    if location.exists():
        raise FileExistsError(f"{location} exists: a store is never written over another")
    if not paths:
        raise ValueError("no closed chromosomes to join")
    records = _records(paths)
    first = records[0][3]
    _source_matches(sourced, records)
    # Import here avoids a writer/joining import cycle; all public entry is StoreWriter.join.
    from ogld.store.writer import write_variant_table

    with StoreReader.open(records[0][1]) as reader:
        variants = reader.variants(np.arange(reader.provenance["n_variants"], dtype=np.int64))
    ranges_out = chromosome_ranges_from_counts(
        (name, record["n_variants"]) for name, _, _, record in records
    )
    out = write_variant_table(location, variants, chromosome_ranges(variants.chrom))
    table = out["variants"]
    assert isinstance(table, zarr.Group)
    tally = Tally()
    for name, path, root, record in records:
        start, stop = ranges_out[name]
        if start:
            source = root["variants"]
            assert isinstance(source, zarr.Group)
            _append(table, source, start, stop - start)
        for component in ("near", "far", layout.PARTNERS):
            out.require_group(component)
            shutil.copytree(
                path / component / name, location / component / name, copy_function=os.link
            )
        for key in ("near_pairs", "far_candidates", "far_pairs", "partner_entries"):
            setattr(tally, key, getattr(tally, key) + record["counts"][key])
    table.attrs["chromosomes"] = layout.chromosomes_record(ranges_out)
    position = table["position"]
    assert isinstance(position, zarr.Array)
    cutter = Cutter(
        np.asarray(position[:], dtype=np.int64),
        ranges_out,
        (first["near_bp"], first["far_bp"]),
        REncoding.of(first["encoding"]),
        n=first["panel"]["n"],
        k=first["adjustment"]["k"],
        t_partner=first["t_partner"],
    )
    return out, cutter, ranges_out, tally


def _source_matches(
    sourced: Sourced, records: list[tuple[str, Path, zarr.Group, dict[str, Any]]]
) -> None:
    for _, path, _, record in records:
        for key in ("panel", "adjustment", "variant_filters", "reference_assembly"):
            if record[key] != dict(sourced)[key]:
                raise ValueError(f"{path}: closed chromosome disagrees with source {key}")
        if record["source"]["kind"] != sourced["source"]["kind"]:
            raise ValueError(f"{path}: closed chromosome disagrees with source kind")


def _records(paths: Sequence[Path]) -> list[tuple[str, Path, zarr.Group, dict[str, Any]]]:
    records = []
    seen = set()
    for path in paths:
        root = zarr.open_group(str(path), mode="r")
        record = cast(dict[str, Any], root.attrs["ld_store"])
        table = root["variants"]
        assert isinstance(table, zarr.Group)
        ranges = layout.chromosomes_listed(table.attrs["chromosomes"])
        if len(ranges) != 1 or ranges[0][0] in seen:
            raise ValueError(f"{path}: expected one distinct closed chromosome")
        seen.add(ranges[0][0])
        records.append((ranges[0][0], path, root, record))
    records.sort(key=lambda entry: chromosome_rank(entry[0]))
    for _, _, _, record in records:
        _compatible(records[0][3], record)
    return records


def _append(target: zarr.Group, source: zarr.Group, first: int, count: int) -> None:
    """Copy strings a bounded slice at a time, not a genome-wide fixed-width array."""
    for name in target.array_keys():
        array = target[name]
        original = source[name]
        assert isinstance(array, zarr.Array)
        assert isinstance(original, zarr.Array)
        array.resize((first + count,))
        for start in range(0, count, layout.SIDE[1]):
            stop = min(count, start + layout.SIDE[1])
            array[first + start : first + stop] = original[start:stop]
