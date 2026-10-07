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
    """The compressor configuration a v3 array's `zarr.json` names, or a note."""
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
    """Every Zarr v3 array under `root`, keyed by its path relative to `root`."""
    found: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(root.rglob("zarr.json")):
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if raw.get("node_type") != "array":
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("left", type=Path, help="the Rebuild from source")
    parser.add_argument("right", type=Path, help="the conversion of the registered release")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-cells", type=int, default=BLOCK_CELLS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = compare_releases(args.left, args.right, args.block_cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _summarise(report)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
