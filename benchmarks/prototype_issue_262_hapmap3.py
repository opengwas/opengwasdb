#!/usr/bin/env python3
"""PROTOTYPE ONLY — test the storage/latency trade-off proposed in issue #262.

Question: can OGS-00009 extract its HapMap3 associations for one Analysis in
under one second without substantially increasing the Store Release's size?
This scratch benchmark distinguishes a membership index from the materialised,
Analysis-major Z-score projection needed to avoid the Dense Layout's 1000 x
1000 source chunks. It never modifies the immutable Store Release.

One-command run:
    pixi run -e dev prototype-issue-262
"""

from __future__ import annotations

import csv
import gzip
import json
import shutil
import time
import urllib.request
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc

from opengwasdb.variants.axis import VariantAxis

STORE = Path("/data/opengwasdb/stores/OGS-00009/store.opengwasdb")
WORK = Path("/data/opengwasdb/work/262-hapmap3-prototype")
ANALYSIS_ID = "ukb-b-17805"
HM3_URL = "https://zenodo.org/api/records/7773502/files/w_hm3.snplist.gz/content"
HM3_GZIP_MD5 = "153ecc2bcfa740afafe656e6a384d769"
CACHE_ROWS = 65_536
MISSING_CODE = -32_768
OVERFLOW_CODE = -32_767
Z_SCALE = 1024.0


@dataclass(frozen=True)
class HapMap3Reference:
    rsid: np.ndarray
    a1: np.ndarray
    a2: np.ndarray


@dataclass(frozen=True)
class Projection:
    variant_index: np.ndarray
    hm3_ordinal: np.ndarray
    effect_sign: np.ndarray
    diagnostics: dict[str, int]


def _md5(path: Path) -> str:
    import hashlib

    # MD5 is the checksum the publisher records; this verifies identity, not security.
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_reference(work: Path) -> Path:
    destination = work / "w_hm3.snplist.gz"
    if not destination.exists():
        print(f"Downloading canonical LDSC HapMap3 list to {destination} ...")
        temporary = destination.with_suffix(".tmp")
        urllib.request.urlretrieve(HM3_URL, temporary)
        temporary.replace(destination)
    observed = _md5(destination)
    if observed != HM3_GZIP_MD5:
        raise SystemExit(
            f"HapMap3 checksum mismatch: expected {HM3_GZIP_MD5}, observed {observed}"
        )
    return destination


def _read_reference(path: Path) -> HapMap3Reference:
    rows: list[tuple[str, str, str]] = []
    with gzip.open(path, "rt") as handle:
        header = handle.readline().split()
        if header != ["SNP", "A1", "A2"]:
            raise SystemExit(f"unexpected HapMap3 header: {header}")
        for line in handle:
            fields = line.split()
            if len(fields) != 3:
                raise SystemExit(f"malformed HapMap3 row: {line.rstrip()}")
            rows.append((fields[0], fields[1], fields[2]))
    return HapMap3Reference(
        rsid=np.asarray([row[0] for row in rows], dtype="S24"),
        a1=np.asarray([row[1] for row in rows], dtype=object),
        a2=np.asarray([row[2] for row in rows], dtype=object),
    )


def _candidate_arrays(
    lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    counts = upper - lower
    positions: list[np.ndarray] = []
    ordinals: list[np.ndarray] = []
    for offset in range(int(counts.max(initial=0))):
        eligible = np.flatnonzero(counts > offset)
        positions.append(lower[eligible] + offset)
        ordinals.append(eligible)
    if not positions:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(positions), np.concatenate(ordinals)


def _resolve_projection(store: Path, reference: HapMap3Reference) -> Projection:
    keys = np.load(store / "variant_rsid_bytes.npy", mmap_mode="r")
    rows = np.load(store / "variant_rsid_rows.npy", mmap_mode="r")
    started = time.perf_counter()
    lower = np.searchsorted(keys, reference.rsid, side="left")
    upper = np.searchsorted(keys, reference.rsid, side="right")
    candidate_positions, ordinals = _candidate_arrays(lower, upper)
    candidates = np.asarray(rows[candidate_positions], dtype=np.int32)

    axis = VariantAxis(store)
    try:
        identity = axis.identity_by_indices(candidates)
    finally:
        axis.close()
    if identity is None:
        raise SystemExit("OGS-00009 has no ALID sidecar; allele-safe resolution is impossible")

    stored_a1 = identity["effect_allele"]
    stored_a2 = identity["other_allele"]
    wanted_a1 = reference.a1[ordinals]
    wanted_a2 = reference.a2[ordinals]
    forward = (stored_a1 == wanted_a1) & (stored_a2 == wanted_a2)
    reverse = (stored_a1 == wanted_a2) & (stored_a2 == wanted_a1)
    compatible = forward | reverse
    compatible_counts = np.bincount(ordinals[compatible], minlength=len(reference.rsid))
    if np.any(compatible_counts > 1):
        count = int(np.count_nonzero(compatible_counts > 1))
        raise SystemExit(f"{count} HapMap3 rsids resolve to multiple allele-compatible variants")

    chosen = np.flatnonzero(compatible)
    selected_rows = candidates[chosen]
    selected_ordinals = ordinals[chosen]
    selected_sign = np.where(forward[chosen], 1, -1).astype(np.int8)
    order = np.argsort(selected_rows, kind="stable")
    counts = upper - lower
    diagnostics = {
        "n_hm3": len(reference.rsid),
        "n_rsid_absent_from_store_axis": int(np.count_nonzero(counts == 0)),
        "n_rsid_with_multiple_store_rows": int(np.count_nonzero(counts > 1)),
        "n_rsid_present_but_alleles_incompatible": int(
            np.count_nonzero((counts > 0) & (compatible_counts == 0))
        ),
        "n_resolved": len(selected_rows),
        "resolution_ms": round((time.perf_counter() - started) * 1000),
    }
    return Projection(
        variant_index=selected_rows[order],
        hm3_ordinal=selected_ordinals[order].astype(np.int32),
        effect_sign=selected_sign[order],
        diagnostics=diagnostics,
    )


def _analysis_index(store: Path, analysis_id: str) -> int:
    with (store / "analyses.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    unsupported = [
        row["analysis_id"] for row in rows if row["sample_size_scope"] != "analysis_level"
    ]
    if unsupported:
        raise SystemExit(
            f"{len(unsupported)} Analyses lack Analysis-level sample size; "
            "a Z-only LDSC cache is insufficient"
        )
    for row in rows:
        if row["analysis_id"] == analysis_id:
            return int(row["analysis_index"])
    raise SystemExit(f"Analysis {analysis_id!r} is absent from {store}")


def _project_overflow(
    source: zarr.Group, projection: Projection, n_analyses: int
) -> tuple[np.ndarray, np.ndarray]:
    source_positions = np.asarray(source["z_overflow_index"][:], dtype=np.int64)
    source_values = np.asarray(source["z_overflow_value"][:], dtype=np.float32)
    source_rows = source_positions // n_analyses
    source_columns = source_positions % n_analyses
    slots = np.searchsorted(projection.variant_index, source_rows)
    bounded = np.minimum(slots, len(projection.variant_index) - 1)
    keep = (slots < len(projection.variant_index)) & (
        projection.variant_index[bounded] == source_rows
    )
    cache_positions = source_columns[keep] * len(projection.variant_index) + slots[keep]
    order = np.argsort(cache_positions, kind="stable")
    return cache_positions[order], source_values[keep][order]


def _cache_matches(
    cache: zarr.Group, store: Path, reference_path: Path, projection: Projection
) -> bool:
    attrs = cache.attrs
    return (
        attrs.get("prototype") == "issue-262-hapmap3"
        and attrs.get("source_store") == str(store.resolve())
        and attrs.get("hm3_gzip_md5") == _md5(reference_path)
        and tuple(cache["z"].shape) == (attrs.get("n_analyses"), len(projection.variant_index))
        and np.array_equal(cache["variant_index"][:], projection.variant_index)
    )


def _build_cache(
    store: Path, cache_path: Path, reference_path: Path, projection: Projection
) -> tuple[zarr.Group, float]:
    if cache_path.exists():
        existing = zarr.open_group(str(cache_path), mode="r")
        if _cache_matches(existing, store, reference_path, projection):
            print(f"Reusing scratch projection at {cache_path}")
            return existing, float(existing.attrs.get("build_seconds", 0.0))
        raise SystemExit(f"refusing to overwrite incompatible scratch cache {cache_path}")

    source = zarr.open_group(str(store / "data.zarr"), mode="r")
    n_analyses = int(source["z"].shape[1])
    overflow_index, overflow_value = _project_overflow(source, projection, n_analyses)
    temporary = cache_path.with_name(f".{cache_path.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    cache = zarr.open_group(str(temporary), mode="w")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    target = cache.create_dataset(
        "z",
        shape=(n_analyses, len(projection.variant_index)),
        chunks=(1, CACHE_ROWS),
        dtype="int16",
        fill_value=MISSING_CODE,
        compressor=compressor,
    )
    cache.create_dataset("variant_index", data=projection.variant_index, compressor=compressor)
    cache.create_dataset("hm3_ordinal", data=projection.hm3_ordinal, compressor=compressor)
    cache.create_dataset("effect_sign", data=projection.effect_sign, compressor=compressor)
    cache.create_dataset("z_overflow_index", data=overflow_index, compressor=compressor)
    cache.create_dataset("z_overflow_value", data=overflow_value, compressor=compressor)
    cache.attrs.update(
        prototype="issue-262-hapmap3",
        source_store=str(store.resolve()),
        hm3_gzip_md5=_md5(reference_path),
        n_analyses=n_analyses,
        z_scale=int(Z_SCALE),
        orientation="HM3 A1 via effect_sign",
    )

    print(
        f"Building full {n_analyses:,} Analysis x "
        f"{len(projection.variant_index):,} projection ..."
    )
    started = time.perf_counter()
    for start in range(0, len(projection.variant_index), CACHE_ROWS):
        stop = min(start + CACHE_ROWS, len(projection.variant_index))
        source_block = np.asarray(source["z"].oindex[projection.variant_index[start:stop], :])
        target[:, start:stop] = source_block.T
        print(f"  {stop:,}/{len(projection.variant_index):,} variants", flush=True)
    build_seconds = time.perf_counter() - started
    cache.attrs["build_seconds"] = build_seconds
    cache.store.close() if hasattr(cache.store, "close") else None
    temporary.replace(cache_path)
    return zarr.open_group(str(cache_path), mode="r"), build_seconds


def _decode_cache_row(cache: zarr.Group, analysis_index: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(cache["z"][analysis_index, :], dtype=np.int16)
    z_value = raw.astype(np.float32) / Z_SCALE
    present = raw != MISSING_CODE
    overflow_index = np.asarray(cache["z_overflow_index"][:], dtype=np.int64)
    overflow_value = np.asarray(cache["z_overflow_value"][:], dtype=np.float32)
    width = raw.size
    lower = np.searchsorted(overflow_index, analysis_index * width, side="left")
    upper = np.searchsorted(overflow_index, (analysis_index + 1) * width, side="left")
    if lower != upper:
        local = overflow_index[lower:upper] - analysis_index * width
        z_value[local] = overflow_value[lower:upper]
    z_value *= np.asarray(cache["effect_sign"][:], dtype=np.int8)
    return np.flatnonzero(present).astype(np.int32), z_value[present]


def _timings(function: Any, repetitions: int) -> dict[str, Any]:
    values = []
    count = 0
    for _ in range(repetitions):
        started = time.perf_counter()
        result = function()
        values.append((time.perf_counter() - started) * 1000)
        count = len(result[0])
    return {
        "first_ms": round(values[0], 3),
        "median_ms": round(float(np.median(values)), 3),
        "max_ms": round(max(values), 3),
        "repetitions": repetitions,
        "result_count": count,
    }


def _source_projection_once(
    source: zarr.Group, projection: Projection, analysis_index: int
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(source["z"].oindex[projection.variant_index, analysis_index])
    z_value = raw.astype(np.float32) / Z_SCALE
    source_index = np.asarray(source["z_overflow_index"][:], dtype=np.int64)
    source_value = np.asarray(source["z_overflow_value"][:], dtype=np.float32)
    n_analyses = int(source["z"].shape[1])
    for source_position, exact in zip(source_index, source_value, strict=True):
        if source_position % n_analyses != analysis_index:
            continue
        slot = int(np.searchsorted(projection.variant_index, source_position // n_analyses))
        if slot < len(projection.variant_index) and (
            projection.variant_index[slot] == source_position // n_analyses
        ):
            z_value[slot] = exact
    z_value *= projection.effect_sign
    present = raw != MISSING_CODE
    return np.flatnonzero(present).astype(np.int32), z_value[present]


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_result(
    work: Path,
    store: Path,
    projection: Projection,
    cache: zarr.Group,
    build_seconds: float,
    analysis_id: str,
    analysis_index: int,
    repetitions: int,
) -> Path:
    source = zarr.open_group(str(store / "data.zarr"), mode="r")
    source_chunks = np.unique(projection.variant_index // int(source["z"].chunks[0]))
    n_source_chunks = (int(source["z"].shape[0]) + int(source["z"].chunks[0]) - 1) // int(
        source["z"].chunks[0]
    )
    original = _timings(
        lambda: _source_projection_once(source, projection, analysis_index), 2
    )
    cached = _timings(lambda: _decode_cache_row(cache, analysis_index), repetitions)
    expected = _source_projection_once(source, projection, analysis_index)
    observed = _decode_cache_row(cache, analysis_index)
    if not (np.array_equal(expected[0], observed[0]) and np.array_equal(expected[1], observed[1])):
        raise SystemExit("cache result differs from OGS-00009 source codes")

    store_bytes = _tree_bytes(store)
    cache_bytes = _tree_bytes(work / "hapmap3-z-projection.zarr")
    result = {
        "prototype": "issue-262-hapmap3",
        "question": (
            "Can OGS-00009 extract HapMap3 Z-scores for one Analysis in under one second "
            "without substantially increasing storage?"
        ),
        "store": str(store),
        "analysis_id": analysis_id,
        "analysis_index": analysis_index,
        "mapping": projection.diagnostics,
        "source_chunk_overlap": {
            "chunks_touched": len(source_chunks),
            "chunks_total": n_source_chunks,
            "percent": round(len(source_chunks) / n_source_chunks * 100, 3),
        },
        "timing": {"membership_index_only": original, "materialised_projection": cached},
        "storage": {
            "store_bytes": store_bytes,
            "projection_bytes": cache_bytes,
            "increase_percent": round(cache_bytes / store_bytes * 100, 3),
            "membership_arrays_uncompressed_bytes": sum(
                cache[name].nbytes for name in ("variant_index", "hm3_ordinal", "effect_sign")
            ),
        },
        "cache": {
            "shape": list(cache["z"].shape),
            "chunks": list(cache["z"].chunks),
            "build_seconds": round(build_seconds, 3),
            "z_overflow_count": len(cache["z_overflow_index"]),
            "result_exactly_matches_source": True,
        },
        "boundary": (
            "Timing returns in-memory (HM3 ordinal, A1-oriented Z) arrays and excludes text "
            "serialisation. OGS-00009 has Analysis-level sample sizes, so LDSC N comes from "
            "analyses.tsv and need not be duplicated per variant."
        ),
    }
    destination = work / "result.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return destination


def _parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--analysis-id", default=ANALYSIS_ID)
    parser.add_argument("--repetitions", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    reference_path = _download_reference(args.work)
    reference = _read_reference(reference_path)
    projection = _resolve_projection(args.store, reference)
    print(json.dumps(projection.diagnostics, indent=2))
    cache_path = args.work / "hapmap3-z-projection.zarr"
    cache, build_seconds = _build_cache(
        args.store, cache_path, reference_path, projection
    )
    analysis_index = _analysis_index(args.store, args.analysis_id)
    result = _write_result(
        args.work,
        args.store,
        projection,
        cache,
        build_seconds,
        args.analysis_id,
        analysis_index,
        args.repetitions,
    )
    print(result.read_text())
    print(f"Wrote {result}")


if __name__ == "__main__":
    main()
