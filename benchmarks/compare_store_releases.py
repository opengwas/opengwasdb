#!/usr/bin/env python3
"""Compare two Store Releases of the same data array by array, and the envelope.

#249's comparison is a **rebuild from source** (this branch's 0.2.0 builders)
against the **conversion** (#248) of the registered 0.1.0 release. The pilots
differ in ways that must be explained, not waved through, so this script reports
every difference it can see and lets the caller explain each one:

* array metadata -- shape, dtype, inner chunk, shard, compressor, zarr_format;
* decoded values, shard by shard, NaN-aware, with the count of differing cells,
  the widest difference and a few positions;
* `analyses.tsv`, column by column (differing rows and an example);
* `variants.tsv.gz` -- sha256, and if it differs, the number of differing lines;
* the encoding plan each release's `manifest.json` records.

Both release roots are walked whole, so a Hybrid's nested `dense/data.zarr` is
compared beside the outer `data.zarr`.  A difference is data, not a verdict: the
caller names the build-time fix or source fact behind each one.

  pixi run -e dev python benchmarks/compare_store_releases.py \
      /data/opengwasdb/work/epic240/249/OGS-00001-rebuild \
      /data/opengwasdb/work/epic240/248/OGS-00001 \
      --output /tmp/epic240/249/compare-OGS-00001.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections.abc import Iterator
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import zarr

#: Cells read from each side at a time. 4,194,304 float32 cells is 16 MB a side,
#: so both trees plus the comparison never dominate memory on a 3 GB release.
BLOCK_CELLS = 1 << 22


def _codec_record(metadata: dict[str, Any]) -> dict[str, Any]:
    """The compressor configuration an array's metadata names, or a note.

    A v2 `.zarray` names one `compressor` and cannot shard; a v3 `zarr.json`
    lists `codecs`, the compressor nested inside `sharding_indexed` when sharded.
    """
    if metadata.get("zarr_format") == 2:
        compressor = metadata.get("compressor")
        return {"sharding": None, **compressor} if compressor else {"sharding": None, "codecs": []}
    for codec in metadata.get("codecs", []):
        if codec.get("name") == "sharding_indexed":
            inner = codec["configuration"]["codecs"]
            blosc = next((c for c in inner if c.get("name") == "blosc"), None)
            if blosc is not None:
                return {"sharding": "sharding_indexed", **blosc["configuration"]}
        if codec.get("name") == "blosc":
            return {"sharding": None, **codec["configuration"]}
    return {"sharding": None, "codecs": [c.get("name") for c in metadata.get("codecs", [])]}


def discover_arrays(root: Path) -> dict[str, dict[str, Any]]:
    """Every Zarr array under `root`, keyed by its path relative to `root`.

    Found by `zarr.json` (v3) **or** `.zarray` (v2), so a format 0.1.0 release
    is described by the same record as a 0.2.0 one: the OGS-00009/OGS-00010
    comparison (#312) sets the two formats side by side.
    """
    found: dict[str, dict[str, Any]] = {}
    for meta_path in sorted([*root.rglob("zarr.json"), *root.rglob(".zarray")]):
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.name == "zarr.json" and raw.get("node_type") != "array":
            continue
        rel = meta_path.parent.relative_to(root).as_posix()
        array = zarr.open_array(meta_path.parent)
        found[rel] = {
            "shape": [int(size) for size in array.shape],
            "dtype": str(array.dtype),
            "inner_chunk": [int(size) for size in array.chunks],
            "shard": [int(size) for size in array.shards] if array.shards else None,
            "zarr_format": int(array.metadata.zarr_format),
            "compressor": _codec_record(raw),
            "nbytes_on_disk": sum(
                f.stat().st_size for f in meta_path.parent.rglob("*") if f.is_file()
            ),
        }
    return found


def _iter_blocks(shape: tuple[int, ...], block_cells: int) -> Iterator[tuple[slice, ...]]:
    """Yield one `tuple[slice, ...]` per block of at most `block_cells` cells."""
    per_row = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    rows = max(1, block_cells // max(1, per_row))
    for start in range(0, shape[0], rows):
        yield (slice(start, min(start + rows, shape[0])),) + (slice(None),) * (len(shape) - 1)


def compare_values(left: Path, right: Path, block_cells: int = BLOCK_CELLS) -> dict[str, Any]:
    """Decoded comparison of one array, block by block, NaN-aware."""
    a = zarr.open_array(left)
    b = zarr.open_array(right)
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return {"comparable": False, "reason": "shape or dtype differs"}
    is_float = np.issubdtype(a.dtype, np.floating)
    differing = 0
    max_abs = 0.0
    examples: list[dict[str, Any]] = []
    for selection in _iter_blocks(tuple(a.shape), block_cells):
        va = np.asarray(a[selection])
        vb = np.asarray(b[selection])
        if is_float:
            both_nan = np.isnan(va) & np.isnan(vb)
            diff = ~((va == vb) | both_nan)
        else:
            diff = va != vb
        n = int(np.count_nonzero(diff))
        if n == 0:
            continue
        differing += n
        if is_float:
            pair = np.abs(va[diff].astype(np.float64) - vb[diff].astype(np.float64))
            finite = np.isfinite(pair)
            if finite.any():
                max_abs = max(max_abs, float(pair[finite].max()))
        if len(examples) < 5:
            positions = np.argwhere(diff)[: 5 - len(examples)]
            for pos in positions:
                index = tuple(int(p) for p in pos)
                offset = np.ravel_multi_index(
                    tuple(p + (selection[axis].start or 0) for axis, p in enumerate(index)),
                    tuple(int(s) for s in a.shape),
                )
                examples.append(
                    {
                        "flat_index": int(offset),
                        "left": _scalar(va[index]),
                        "right": _scalar(vb[index]),
                    }
                )
    return {
        "comparable": True,
        "differing_cells": differing,
        "max_abs_diff": max_abs if differing else 0.0,
        "examples": examples,
    }


def _scalar(value: Any) -> Any:
    """A JSON-able Python scalar, preserving integer exactness; NaN is `"nan"`."""
    if np.issubdtype(np.asarray(value).dtype, np.integer):
        return int(value)
    as_float = float(value)
    return "nan" if np.isnan(as_float) else as_float


def compare_tsv(left: Path, right: Path) -> dict[str, Any]:
    """Compare two `analyses.tsv`s: columns, differing cells, one example each."""
    left_rows = _read_tsv(left)
    right_rows = _read_tsv(right)
    if not left_rows or not right_rows:
        return {
            "left_rows": len(left_rows),
            "right_rows": len(right_rows),
            "note": "one side empty",
        }
    left_columns, right_columns = left_rows[0], right_rows[0]
    shared = [name for name in left_columns if name in right_columns]
    differing: dict[str, int] = {}
    examples: dict[str, Any] = {}
    n = min(len(left_rows), len(right_rows))
    for name in shared:
        i, j = left_columns.index(name), right_columns.index(name)
        count = 0
        for row in range(1, n):
            if left_rows[row][i] != right_rows[row][j]:
                count += 1
                if name not in examples:
                    examples[name] = {
                        "row": row,
                        "left": left_rows[row][i],
                        "right": right_rows[row][j],
                    }
        if count:
            differing[name] = count
    return {
        "columns_only_left": sorted(set(left_columns) - set(right_columns)),
        "columns_only_right": sorted(set(right_columns) - set(left_columns)),
        "rows": {"left": len(left_rows) - 1, "right": len(right_rows) - 1},
        "differing_cells_by_column": differing,
        "examples": examples,
    }


def _read_tsv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [row for row in csv.reader(fh, delimiter="\t")]


def compare_gzip_text(left: Path, right: Path) -> dict[str, Any]:
    """sha256, and if the files differ, how many lines do."""
    left_sha, right_sha = _sha256(left), _sha256(right)
    result: dict[str, Any] = {
        "left_sha256": left_sha,
        "right_sha256": right_sha,
        "identical": left_sha == right_sha,
    }
    if left_sha == right_sha:
        return result
    differing = 0
    examples: list[dict[str, Any]] = []
    with (
        gzip.open(left, "rt", encoding="utf-8") as fa,
        gzip.open(right, "rt", encoding="utf-8") as fb,
    ):
        for index, (a, b) in enumerate(zip_longest(fa, fb, fillvalue="")):
            if a != b:
                differing += 1
                if len(examples) < 5:
                    examples.append({"line": index, "left": a[:200], "right": b[:200]})
    result["differing_lines"] = differing
    result["examples"] = examples
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _encoding_plan(root: Path) -> Any:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return manifest.get("encoding", manifest.get("provenance", {}).get("encoding"))


def _component_encodings(root: Path) -> dict[str, Any]:
    plans = {"": _encoding_plan(root)}
    nested = root / "dense"
    if (nested / "manifest.json").is_file():
        plans["dense"] = _encoding_plan(nested)
    return plans


def compare_releases(left: Path, right: Path, block_cells: int = BLOCK_CELLS) -> dict[str, Any]:
    """The whole comparison for two release roots, as a JSON-able report."""
    left_arrays = discover_arrays(left)
    right_arrays = discover_arrays(right)
    if not left_arrays or not right_arrays:
        raise SystemExit(
            f"found {len(left_arrays)} arrays under {left} and {len(right_arrays)} under "
            f"{right}; the comparison would prove nothing"
        )
    only_left = sorted(set(left_arrays) - set(right_arrays))
    only_right = sorted(set(right_arrays) - set(left_arrays))
    arrays: dict[str, Any] = {}
    for rel in sorted(set(left_arrays) & set(right_arrays)):
        metadata_left, metadata_right = left_arrays[rel], right_arrays[rel]
        entry = {
            "metadata_left": metadata_left,
            "metadata_right": metadata_right,
            "metadata_equal": metadata_left == metadata_right,
        }
        entry["values"] = compare_values(left / rel, right / rel, block_cells)
        arrays[rel] = entry
    report: dict[str, Any] = {
        "left": str(left),
        "right": str(right),
        "n_arrays_left": len(left_arrays),
        "n_arrays_right": len(right_arrays),
        "arrays_only_left": only_left,
        "arrays_only_right": only_right,
        "arrays_with_metadata_differences": [
            rel for rel, entry in arrays.items() if not entry["metadata_equal"]
        ],
        "arrays_with_value_differences": [
            rel
            for rel, entry in arrays.items()
            if entry["values"].get("comparable") and entry["values"]["differing_cells"]
        ],
        "arrays": arrays,
        "encoding_plan": {"left": _component_encodings(left), "right": _component_encodings(right)},
    }
    if (left / "analyses.tsv").is_file() and (right / "analyses.tsv").is_file():
        report["analyses_tsv"] = compare_tsv(left / "analyses.tsv", right / "analyses.tsv")
    if (left / "variants.tsv.gz").is_file() and (right / "variants.tsv.gz").is_file():
        report["variants_tsv_gz"] = compare_gzip_text(
            left / "variants.tsv.gz", right / "variants.tsv.gz"
        )
    return report


def _summarise(report: dict[str, Any]) -> None:
    print(f"arrays: {report['n_arrays_left']} left, {report['n_arrays_right']} right")
    print(f"only left: {report['arrays_only_left']}")
    print(f"only right: {report['arrays_only_right']}")
    print(f"metadata differences: {report['arrays_with_metadata_differences']}")
    print(f"value differences: {report['arrays_with_value_differences']}")
    for rel in report["arrays_with_value_differences"]:
        values = report["arrays"][rel]["values"]
        print(f"  {rel}: {values['differing_cells']:,} cells, max |diff| {values['max_abs_diff']}")
    if report["encoding_plan"]["left"] != report["encoding_plan"]["right"]:
        print(f"encoding plan differs: {report['encoding_plan']}")
    if "analyses_tsv" in report:
        print(f"analyses.tsv: {report['analyses_tsv'].get('differing_cells_by_column', {})}")
    if "variants_tsv_gz" in report:
        print(f"variants.tsv.gz identical: {report['variants_tsv_gz']['identical']}")


# --- Dense releases whose variant axes differ (#322) --------------------------
#
# A rebuild that drops rows (#376's off-primary lifts) or spells an indel's ALID
# another way (#381) has a different variant axis, so the arrays above are not
# comparable position by position. The aligned comparison decodes both grids
# through the reader every query uses and compares them cell by cell on the rows
# they share by ALID. Rows only one side holds are listed whole, and a pair that
# names one site and one allele pair in two spellings is compared with its
# orientation taken into account. Nothing is classified as expected here: the
# caller names the change behind each kind.

#: Rows of each side decoded at a time. A band of 20,000 rows of a 2,514-Analysis
#: release is three planes of 201 MB a side as float32.
ALIGNED_BAND_ROWS = 20_000

#: Rows kept per band as examples of a difference; the counts are always whole.
EXAMPLE_ROWS = 20

PLANES = ("z", "se", "eaf")

_OPEN: dict[str, Any] = {}


def _dense_reader(path: str) -> Any:
    """A Dense grid reader for `path`, opened once per process (after any fork)."""
    if path not in _OPEN:
        from opengwasdb.associations import DenseGridReader
        from opengwasdb.store.open import open_store

        store = open_store(path)
        root = store.arrays(mode="r")
        _OPEN[path] = (
            DenseGridReader.open(root, store.manifest.encoding, source=f"Store Release {path}"),
            int(root["z"].shape[1]),
        )
    return _OPEN[path]


def _decoded(path: str, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Every cell of `rows` (ascending) by every Analysis, decoded, as `rows × analyses`."""
    from opengwasdb.associations import Cells

    reader, n_analyses = _dense_reader(path)
    found = reader.inspect(Cells.variants(rows))
    slot = np.searchsorted(rows, np.asarray(found.variant_index, dtype="int64"))
    column = np.asarray(found.analysis_index, dtype="int64")
    grids = {}
    for plane in PLANES:
        grid = np.full((rows.size, n_analyses), np.nan, dtype=np.float32)
        grid[slot, column] = np.asarray(getattr(found, plane), dtype=np.float32)
        grids[plane] = grid
    return grids


def _float16_rounding(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Where one side is the other's `float16` rounding (a `float16`-coded SE, #355)."""
    with np.errstate(over="ignore"):
        a16 = a.astype(np.float16).astype(np.float32)
        b16 = b.astype(np.float16).astype(np.float32)
    return (a16 == b) | (b16 == a)


#: A row's difference code: one bit per kind of difference any of its cells shows.
ROW_CODES = {
    "z_value": 1,
    "se_value": 2,
    "se_float16_rounding": 4,
    "eaf_value": 8,
    "presence": 16,
    "se_within_codec": 32,
}

#: `float16`'s relative precision, as a log-space half step.
_FLOAT16_LOG_HALF_STEP = float(np.log1p(2.0**-11))

#: Slack for the decode's own `float32` arithmetic, in log space.
_FLOAT32_LOG_SLACK = 1e-6


def se_log_half_step(encoding: dict[str, Any] | None) -> float:
    """How far, in log space, a decoded SE can sit from the value it encodes.

    An `int8_residual` code decodes to within half a step, `residual_range / 127`,
    of the source's `log(SE)`, whatever the fitted model predicts (spec §6a);
    `float16` to within its relative precision. An exact exception is exact.
    """
    se = (encoding or {}).get("se") or {}
    if se.get("kind") == "int8_residual":
        return float(se["residual_range"]) / 127 / 2
    if se.get("kind") == "float16":
        return _FLOAT16_LOG_HALF_STEP
    raise SystemExit(f"no SE codec bound for encoding {se!r}")


def _plane_difference(
    plane: str,
    a: np.ndarray,
    b: np.ndarray,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    se_log_bound: float = 0.0,
) -> tuple[dict[str, Any], np.ndarray]:
    """Counts of one plane's differing cells, and each row's difference code.

    An SE difference no larger than `se_log_bound` in log space is within what the
    two releases' codecs allow for one source value (a refitted residual model
    re-quantises every coded cell); it is counted apart from the rest.
    """
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    only_left = ~nan_a & nan_b
    only_right = nan_a & ~nan_b
    value = ~nan_a & ~nan_b & (a != b)
    record: dict[str, Any] = {
        "only_left": int(only_left.sum()),
        "only_right": int(only_right.sum()),
        "value_differs": int(value.sum()),
        "max_abs_diff": float(np.abs(a[value] - b[value]).max()) if value.any() else 0.0,
    }
    code = np.where((only_left | only_right).any(axis=1), ROW_CODES["presence"], 0)
    if plane == "se":
        rounding = np.zeros_like(value)
        rounding[value] = _float16_rounding(a[value], b[value])
        within = np.zeros_like(value)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.abs(np.log(a[value].astype(np.float64) / b[value].astype(np.float64)))
        within[value] = log_ratio <= se_log_bound + _FLOAT32_LOG_SLACK
        within &= ~rounding
        beyond = value & ~rounding & ~within
        record["value_differs_by_float16_rounding"] = int(rounding.sum())
        record["value_differs_within_codec"] = int(within.sum())
        record["value_differs_beyond_codec"] = int(beyond.sum())
        record["max_abs_log_ratio"] = float(np.nanmax(log_ratio)) if log_ratio.size else 0.0
        record["se_log_bound"] = se_log_bound
        code |= np.where(beyond.any(axis=1), ROW_CODES["se_value"], 0)
        code |= np.where(rounding.any(axis=1), ROW_CODES["se_float16_rounding"], 0)
        code |= np.where(within.any(axis=1), ROW_CODES["se_within_codec"], 0)
    else:
        code |= np.where(value.any(axis=1), ROW_CODES[f"{plane}_value"], 0)
    differs = only_left | only_right | value
    examples = []
    for index in np.flatnonzero(differs.any(axis=1))[:EXAMPLE_ROWS]:
        column = int(np.flatnonzero(differs[index])[0])
        examples.append(
            {
                "left_row": int(left_rows[index]),
                "right_row": int(right_rows[index]),
                "analysis_index": column,
                "left": _scalar(a[index, column]),
                "right": _scalar(b[index, column]),
                "cells_in_row": int(differs[index].sum()),
            }
        )
    record["examples"] = examples
    return record, code.astype(np.uint8)


def _aligned_unit(
    shared: tuple[str, str, float], unit: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> dict[str, Any]:
    """Compare one unit of row pairs. `flip` marks a pair whose effect alleles are swapped:
    its right-hand Z is negated and its EAF taken as `1 - EAF` before the comparison."""
    left, right, se_log_bound = shared
    left_rows, right_rows, flip = unit
    a = _decoded(left, left_rows)
    order = np.argsort(right_rows, kind="stable")
    b_sorted = _decoded(right, right_rows[order])
    b = {plane: np.empty_like(grid) for plane, grid in b_sorted.items()}
    for plane, grid in b_sorted.items():
        b[plane][order] = grid
    if flip.any():
        b["z"][flip] = -b["z"][flip]
        b["eaf"][flip] = np.float32(1.0) - b["eaf"][flip]
    planes: dict[str, Any] = {}
    codes = np.zeros(left_rows.size, dtype=np.uint8)
    for plane in PLANES:
        planes[plane], code = _plane_difference(
            plane, a[plane], b[plane], left_rows, right_rows, se_log_bound
        )
        codes |= code
    return {"pairs": int(left_rows.size), "cells": int(a["z"].size), "codes": codes,
            "planes": planes}


def _merge_units(results: Iterator[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray]:
    """The units' counts added up, and every row's difference code in input order."""
    total: dict[str, Any] = {"pairs": 0, "cells": 0, "planes": {}}
    codes: list[np.ndarray] = []
    for result in results:
        total["pairs"] += result["pairs"]
        total["cells"] += result["cells"]
        codes.append(result["codes"])
        for plane, record in result["planes"].items():
            into = total["planes"].setdefault(plane, {"examples": []})
            for key, value in record.items():
                if key == "examples":
                    into["examples"].extend(value[: max(0, EXAMPLE_ROWS - len(into["examples"]))])
                elif key in ("max_abs_diff", "max_abs_log_ratio", "se_log_bound"):
                    into[key] = max(into.get(key, 0.0), value)
                else:
                    into[key] = into.get(key, 0) + value
    every = np.concatenate(codes) if codes else np.zeros(0, dtype=np.uint8)
    total["rows_differing"] = int(np.count_nonzero(every))
    total["rows_by_code"] = {
        "+".join(name for name, bit in ROW_CODES.items() if code & bit): int(count)
        for code, count in zip(*np.unique(every[every > 0], return_counts=True), strict=True)
    }
    return total, every


def _units(
    left_rows: np.ndarray, right_rows: np.ndarray, flip: np.ndarray, band_rows: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for start in range(0, left_rows.size, band_rows):
        stop = start + band_rows
        yield left_rows[start:stop], right_rows[start:stop], flip[start:stop]


def _variant_rows(path: Path, wanted: np.ndarray) -> dict[int, dict[str, Any]]:
    """The Store Variant Table rows `wanted`, read in one pass over the table."""
    from opengwasdb.variants import VariantAxis

    if not wanted.size:
        return {}
    axis = VariantAxis(path)
    try:
        return {
            row: {
                "alid": record.alid,
                "chromosome": record.chromosome,
                "position": record.position,
                "effect_allele": record.effect_allele,
                "other_allele": record.other_allele,
                "rsid": record.rsid,
                "source_alid": record.source_alid,
            }
            for row, record in axis.by_indices(sorted(int(r) for r in wanted)).items()
        }
    finally:
        axis.close()


def _respelled_pairs(
    left_only: dict[int, dict[str, Any]], right_only: dict[int, dict[str, Any]]
) -> list[tuple[int, int, bool]]:
    """Unmatched rows that name one site and one allele pair: `(left, right, flipped)`.

    A key held by more than one row on either side is not paired: which row would
    match is then a guess.
    """

    def key(record: dict[str, Any]) -> tuple[str, int, frozenset[str]]:
        return (
            record["chromosome"],
            record["position"],
            frozenset((record["effect_allele"], record["other_allele"])),
        )

    def unique(rows: dict[int, dict[str, Any]]) -> dict[tuple[str, int, frozenset[str]], int]:
        seen: dict[tuple[str, int, frozenset[str]], list[int]] = {}
        for row, record in rows.items():
            seen.setdefault(key(record), []).append(row)
        return {k: v[0] for k, v in seen.items() if len(v) == 1}

    left_keys, right_keys = unique(left_only), unique(right_only)
    pairs = []
    for k in sorted(set(left_keys) & set(right_keys), key=lambda k: left_keys[k]):
        left_row, right_row = left_keys[k], right_keys[k]
        flipped = left_only[left_row]["effect_allele"] != right_only[right_row]["effect_allele"]
        pairs.append((left_row, right_row, flipped))
    return pairs


def compare_aligned(
    left: Path, right: Path, *, n_workers: int = 1, band_rows: int = ALIGNED_BAND_ROWS
) -> tuple[dict[str, Any], list[tuple[Any, ...]], dict[str, np.ndarray]]:
    """Two Dense releases compared cell by cell on the rows they share by ALID.

    Their Analyses must be the same, in the same order. Every row is accounted
    for: matched by ALID, paired as one site in two spellings, or held by one side
    only. Returns the report, the unmatched rows with their Store Variant Table
    rows, and every compared row's difference code (`ROW_CODES`).
    """
    from opengwasdb.build.ordered_pool import ordered_map
    from opengwasdb.validation.observed_view import _map_axis

    left_ids = [row[1] for row in _read_tsv(left / "analyses.tsv")[1:]]
    right_ids = [row[1] for row in _read_tsv(right / "analyses.tsv")[1:]]
    if left_ids != right_ids:
        raise SystemExit("the two releases do not hold the same Analyses in the same order")
    n_left = int(zarr.open_array(left / "data.zarr" / "z").shape[0])
    n_right = int(zarr.open_array(right / "data.zarr" / "z").shape[0])
    axis = _map_axis(left, right, n_left, n_right)
    mapped = axis.source_to_target
    hit = mapped >= 0
    right_hit = np.zeros(n_right, dtype=bool)
    right_hit[mapped[hit]] = True
    left_only = _variant_rows(left, np.flatnonzero(~hit))
    right_only = _variant_rows(right, np.flatnonzero(~right_hit))
    pairs = _respelled_pairs(left_only, right_only)
    paired_left = {pair[0] for pair in pairs}
    paired_right = {pair[1] for pair in pairs}

    se_log_bound = se_log_half_step(_encoding_plan(left)) + se_log_half_step(_encoding_plan(right))
    shared = (str(left), str(right), se_log_bound)
    left_rows = np.flatnonzero(hit)
    matched, matched_codes = _merge_units(
        ordered_map(
            _aligned_unit,
            _units(left_rows, mapped[hit], np.zeros(left_rows.size, dtype=bool), band_rows),
            n_workers,
            shared=shared,
        )
    )
    pair_array = np.array([(a, b, f) for a, b, f in pairs], dtype="int64").reshape(-1, 3)
    respelled, respelled_codes = _merge_units(
        ordered_map(
            _aligned_unit,
            _units(pair_array[:, 0], pair_array[:, 1], pair_array[:, 2].astype(bool), band_rows),
            n_workers,
            shared=shared,
        )
    )
    respelled["flipped"] = int(pair_array[:, 2].sum())
    report = {
        "left": str(left),
        "right": str(right),
        "n_analyses": len(left_ids),
        "n_rows": {"left": n_left, "right": n_right},
        "row_codes": ROW_CODES,
        "matched_by_alid": {"rows": int(hit.sum()), "order_preserved": axis.order_preserved,
                            **matched},
        "respelled": {"rows": len(pairs), **respelled},
        "left_only": {"rows": len(left_only) - len(paired_left)},
        "right_only": {"rows": len(right_only) - len(paired_right)},
    }
    table = [
        ("left_only", row, "", record, None, "")
        for row, record in left_only.items() if row not in paired_left
    ] + [
        ("right_only", "", row, record, None, "")
        for row, record in right_only.items() if row not in paired_right
    ] + [
        ("respelled", a, b, left_only[a], right_only[b], int(f))
        for a, b, f in pairs
    ]
    codes = {
        "matched_left_rows": left_rows,
        "matched_right_rows": mapped[hit],
        "matched_codes": matched_codes,
        "respelled_left_rows": pair_array[:, 0],
        "respelled_right_rows": pair_array[:, 1],
        "respelled_flipped": pair_array[:, 2].astype(bool),
        "respelled_codes": respelled_codes,
    }
    return report, table, codes


UNMATCHED_COLUMNS = (
    "kind", "left_row", "right_row", "alid", "right_alid", "flipped", "chromosome", "position",
    "effect_allele", "other_allele", "rsid", "source_alid",
)


def write_unmatched(path: Path, table: list[tuple[Any, ...]]) -> None:
    """Every row not matched by ALID, with its Store Variant Table row, as a TSV."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(UNMATCHED_COLUMNS)
        for kind, left_row, right_row, record, other, flipped in table:
            writer.writerow(
                (kind, left_row, right_row, record["alid"], other["alid"] if other else "",
                 flipped, record["chromosome"], record["position"], record["effect_allele"],
                 record["other_allele"], record["rsid"] or "", record["source_alid"] or "")
            )


def _summarise_aligned(report: dict[str, Any]) -> None:
    matched, respelled = report["matched_by_alid"], report["respelled"]
    print(f"rows: {report['n_rows']}; matched by ALID {matched['rows']:,}, "
          f"respelled {respelled['rows']:,} ({respelled['flipped']:,} flipped), "
          f"left only {report['left_only']['rows']:,}, right only {report['right_only']['rows']:,}")
    for label, part in (("matched", matched), ("respelled", respelled)):
        print(f"{label}: {part['rows_differing']:,} rows differ: {part['rows_by_code']}")
        for plane, record in part["planes"].items():
            counts = {k: v for k, v in record.items() if k != "examples"}
            print(f"  {plane}: {counts}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("left", type=Path, help="the Rebuild from source")
    parser.add_argument("right", type=Path, help="the conversion of the registered release")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-cells", type=int, default=BLOCK_CELLS)
    parser.add_argument(
        "--aligned",
        action="store_true",
        help="compare two Dense releases' decoded cells on the rows they share by ALID, "
        "for releases whose variant axes differ",
    )
    parser.add_argument("--n-workers", type=int, default=1, help="fork pool for --aligned")
    parser.add_argument("--band-rows", type=int, default=ALIGNED_BAND_ROWS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.aligned:
        report, table, codes = compare_aligned(
            args.left, args.right, n_workers=args.n_workers, band_rows=args.band_rows
        )
        write_unmatched(args.output.with_suffix(".unmatched.tsv"), table)
        np.savez_compressed(args.output.with_suffix(".codes.npz"), **codes)
    else:
        report = compare_releases(args.left, args.right, args.block_cells)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.aligned:
        _summarise_aligned(report)
    else:
        _summarise(report)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
