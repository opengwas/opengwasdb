#!/usr/bin/env python3
"""Compare several Store Releases holding the SAME data in different physical shapes.

This is the instrument epic #240 measures with: tickets convert one source
release into Zarr v3 sharded copies ([1000, 1000], [1000, 128], [1000, 64]
inner chunks) and compare them against the pre-upgrade baseline this script
records for OGS-00009 on zarr 2.18. For every labelled store it records the
on-disk footprint, the seven query shapes of the OGS-00009/OGS-00016 reports
(timings and fresh-interpreter peak RSS), and the run environment; and it
refuses to publish an artifact unless every store returns IDENTICAL results for
every shape.

A fast wrong answer must not reach the comparison table. The selections
(exposure Analysis, PheWAS variant, region, seeded random draws) are resolved
ONCE against the first store and applied unchanged to every store, so the
timings compare the same query rather than two similar ones. A shape whose
results differ between stores -- different values, order, dtype, length or NaN
positions -- fails the run with a non-zero exit and writes no artifact; results
are never sorted or normalised to make them agree.

Usage:
  pixi run -e dev python benchmarks/benchmark_store_comparison.py \
      --store v2-c1000=/data/opengwasdb/stores/OGS-00009/store.opengwasdb \
      --reps 5 \
      --output docs/benchmark-output/opengwasdb_store_comparison_ogs00009_zarr2.json

  # quick comparison of two stores that genuinely differ; skips the RSS probes
  pixi run -e dev python benchmarks/benchmark_store_comparison.py \
      --store OGS00009=/data/opengwasdb/stores/OGS-00009/store.opengwasdb \
      --store OGS00010=/data/opengwasdb/stores/OGS-00010/store.opengwasdb \
      --reps 1 --skip-rss --output /tmp/epic240/242/identity-check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numcodecs
import numpy as np
import zarr

from benchmarks import _query_shapes
from benchmarks._artifact import provenance, write_artifact
from benchmarks._rss import run_probe
from benchmarks.benchmark_ukbb_dense import _median_ms
from opengwasdb.query.facade import _empty_result

# Selection anchors, matching the OGS-00009 report (benchmark_ukbb_dense.py):
# the statin-use exposure Analysis, the chr19 APOE/APOC region, and the seeded
# random draws `_query_shapes` fixes. The PheWAS variant is derived once from
# this exposure's strongest genome-wide hit.
EXPOSURE = "ukb-b-17805"
REGION = ("19", 44_500_000, 45_500_000)
DEFAULT_OUTPUT = Path("docs/benchmark-output/opengwasdb_store_comparison.json")

# Every index-keyed query result carries these six parallel arrays. Read from
# the query API's own empty result rather than restated here, so the identity
# check cannot drift from the arrays the queries actually return; hashing all
# six covers values, order, dtype, length and NaN positions at once.
RESULT_ARRAY_NAMES: tuple[str, ...] = tuple(_empty_result())

_ZARR_METADATA_NAMES = frozenset({".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"})

CACHE_NOTE = (
    "warm page cache: the node holds about 1 TB of page cache and the stores are "
    "warm. A cold-cache run cannot be forced without root, so these timings are "
    "warm-cache numbers, not first-touch disk numbers."
)


@dataclass(frozen=True)
class StoreSpec:
    """One labelled store on the command line."""

    label: str
    path: Path


def _parse_store(text: str) -> StoreSpec:
    label, separator, path = text.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(f"--store wants LABEL=PATH, got {text!r}")
    return StoreSpec(label=label, path=Path(path))


def _single_store(args: argparse.Namespace) -> StoreSpec:
    if len(args.store) != 1:
        raise SystemExit(
            f"an RSS probe measures exactly one store, got {len(args.store)}; "
            "the parent re-invokes the script per store"
        )
    return args.store[0]


def _validated_stores(args: argparse.Namespace) -> list[StoreSpec]:
    seen: set[str] = set()
    for spec in args.store:
        if spec.label in seen:
            raise SystemExit(f"--store label {spec.label!r} is used twice; labels must be unique")
        seen.add(spec.label)
    missing = [str(spec.path) for spec in args.store if not spec.path.is_dir()]
    if missing:
        raise SystemExit(f"store path is not a directory: {missing[0]}")
    return list(args.store)


def _region_tuple(selection: dict[str, Any]) -> tuple[str, int, int]:
    region = selection["region"]
    return (str(region["chrom"]), int(region["start"]), int(region["end"]))


def _patterns_for_store(
    q: Any, selection: dict[str, Any]
) -> dict[str, Callable[[], dict[str, np.ndarray]]]:
    """The seven shapes, with the once-resolved selection pinned to this store.

    The random ALIDs and Analysis ids are passed in rather than drawn here, so
    every store is timed on the identical identifiers. The region is resolved
    against this store's axis by the shared builder; any resulting difference
    between stores is caught by the identity check rather than hidden.
    """
    return _query_shapes.common_query_patterns(
        q,
        exposure=selection["exposure_analysis_id"],
        phewas_alid=selection["phewas_alid"],
        region=_region_tuple(selection),
        random_alids=selection["random_alids"],
        random_analyses=selection["random_analyses"],
    )


def _resolve_selection(spec: StoreSpec) -> dict[str, Any]:
    """Resolve the selection once, against the first store, and record it."""
    q, _plan = _query_shapes.open_benchmark_store(spec.path)
    try:
        analyses = q.analyses_table()
        by_id = {row["analysis_id"]: index for index, row in analyses.items()}
        if EXPOSURE not in by_id:
            raise SystemExit(
                f"{spec.path}: exposure Analysis {EXPOSURE!r} is absent, so the "
                "shared selection cannot be resolved. Refusing to guess one."
            )
        top_hits = q.top_hits(threshold=_query_shapes.GENOME_WIDE)
        keep = top_hits["analysis_index"] == by_id[EXPOSURE]
        if not keep.any():
            raise SystemExit(
                f"{spec.path}: exposure Analysis {EXPOSURE!r} has no genome-wide "
                "significant hits; there is no top hit to derive the PheWAS variant from."
            )
        hit_z = np.abs(top_hits["z"][keep])
        strongest = int(top_hits["variant_index"][keep][int(np.argmax(hit_z))])
        record = q._variant_axis.by_index(strongest)
        if record is None:
            raise SystemExit(f"{spec.path}: top-hit variant index {strongest} does not resolve")
        random_alids, random_analyses = _query_shapes.resolve_axis_selections(
            q._variant_axis,
            analyses,
            int(q._root["z"].shape[0]),
            len(analyses),
        )
        if len(random_alids) != _query_shapes.RANDOM_AXIS_SIZE:
            raise SystemExit(
                f"{spec.path}: resolved {len(random_alids)} of "
                f"{_query_shapes.RANDOM_AXIS_SIZE} random-lookup variants; a partial "
                "selection is not the selection the report names."
            )
        return {
            "exposure_analysis_id": EXPOSURE,
            "phewas_alid": record.alid,
            "region": {"chrom": REGION[0], "start": REGION[1], "end": REGION[2]},
            "random_lookup_shapes": _query_shapes.RANDOM_LOOKUP_SHAPES,
            "random_alids": random_alids,
            "random_analyses": random_analyses,
            "resolved_from": {
                "label": spec.label,
                "path": str(spec.path),
                "n_variants": int(q._root["z"].shape[0]),
                "n_analyses": len(analyses),
            },
        }
    finally:
        q.close()


def digest_array(values: np.ndarray) -> str:
    """A sha256 over one result array's dtype, shape, order and values.

    NaN is canonicalised as a position: a NaN may carry a different payload in
    two stores and still mean the same missing cell, so the digest records
    where the NaNs are and hashes only the non-NaN bytes.
    """
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    if values.dtype.kind == "O":
        unique, inverse = np.unique(values, return_inverse=True)
        digest.update(np.asarray(inverse, dtype="int64").tobytes())
        for item in unique.tolist():
            digest.update(repr(item).encode())
            digest.update(b"\x1f")
    elif values.dtype.kind == "f":
        missing = np.isnan(values)
        digest.update(b"nan\x1f")
        digest.update(np.packbits(missing).tobytes())
        digest.update(b"values\x1f")
        digest.update(values[~missing].tobytes())
    else:
        digest.update(values.tobytes())
    return digest.hexdigest()


def result_digests(result: dict[str, np.ndarray]) -> dict[str, str]:
    missing = [name for name in RESULT_ARRAY_NAMES if name not in result]
    if missing:
        raise SystemExit(
            f"query result is missing array(s) {missing}; refusing an identity check "
            "that cannot see every array it claims to compare"
        )
    return {name: digest_array(result[name]) for name in RESULT_ARRAY_NAMES}


def differing_arrays(reference: dict[str, str], candidate: dict[str, str]) -> list[str]:
    names = sorted(set(reference) | set(candidate))
    return [name for name in names if reference.get(name) != candidate.get(name)]


def differing_shapes(
    reference: dict[str, dict[str, str]], candidate: dict[str, dict[str, str]]
) -> dict[str, list[str]]:
    """Shapes whose result arrays differ, mapping each to the arrays that differ."""
    out: dict[str, list[str]] = {}
    for shape in sorted(set(reference) | set(candidate)):
        arrays = differing_arrays(reference.get(shape, {}), candidate.get(shape, {}))
        if arrays:
            out[shape] = arrays
    return out


def assert_identical(
    reference_label: str,
    reference: dict[str, dict[str, str]],
    candidate_label: str,
    candidate: dict[str, dict[str, str]],
) -> None:
    """Fail loudly, naming the shape, the stores and the differing arrays."""
    differing = differing_shapes(reference, candidate)
    if not differing:
        return
    first_shape = next(iter(differing))
    detail = "; ".join(f"{shape}: {', '.join(arrays)}" for shape, arrays in differing.items())
    raise SystemExit(
        f"identity check FAILED for shape {first_shape!r}: store {candidate_label!r} "
        f"differs from reference {reference_label!r}. Differing arrays -- {detail}. "
        "Stores holding the same data in different physical shapes must return identical "
        "results; refusing to write an artifact."
    )


def _timed_shape(
    fn: Callable[[], dict[str, np.ndarray]], reps: int
) -> tuple[float, float, int, dict[str, str]]:
    """Time one shape over `reps` after the shared warm-up, digesting that warm-up.

    `_median_ms` runs one untimed warm-up call, then `reps` timed calls; the
    digest and the per-rep result counts are captured on the untimed call, so
    the identity check costs no timed run.
    """
    digests: dict[str, str] = {}
    counts: list[int] = []
    calls = 0

    def instrumented() -> dict[str, np.ndarray]:
        nonlocal calls
        result = fn()
        calls += 1
        if calls == 1:
            digests.update(result_digests(result))
        counts.append(len(result["z"]))
        return result

    median_ms, p95_ms, count = _median_ms(instrumented, reps)
    if len(set(counts)) != 1:
        raise SystemExit(
            f"a shape returned {sorted(set(counts))} rows across its runs; "
            "a query whose result size is not stable cannot be timed or compared."
        )
    return median_ms, p95_ms, count, digests


def _dataset_block(q: Any, plan: Any) -> dict[str, Any]:
    return {
        "release_id": plan.release_id,
        "store_id": plan.store_id,
        "n_variants": int(q._root["z"].shape[0]),
        "n_analyses": len(q.analyses_table()),
        "completion_state": str(plan.completion_state),
        "reference_assembly": getattr(plan, "reference_assembly", None),
        "format_version": plan.format_version,
        "encoding": plan.encoding.to_manifest(),
    }


def _measure_store(
    spec: StoreSpec, selection: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Time and digest every shape for one store; run its RSS probes if asked."""
    load_before = os.getloadavg()[0]
    q, plan = _query_shapes.open_benchmark_store(spec.path)
    try:
        dataset = _dataset_block(q, plan)
        patterns = _patterns_for_store(q, selection)
        timings: list[dict[str, Any]] = []
        digests: dict[str, dict[str, str]] = {}
        for name, fn in patterns.items():
            median_ms, p95_ms, count, shape_digests = _timed_shape(fn, args.reps)
            timings.append(
                {
                    "query": name,
                    "median_ms": round(median_ms, 3),
                    "p95_ms": round(p95_ms, 3),
                    "result_count": count,
                }
            )
            digests[name] = shape_digests
            print(
                f"[{spec.label}] {name:38s} median={median_ms:9.2f} ms  "
                f"p95={p95_ms:9.2f} ms  count={count:,}",
                flush=True,
            )
    finally:
        q.close()

    memory = [] if args.skip_rss else _probe_store(spec, selection, list(digests))
    _check_memory_counts(memory, timings)
    load_after = os.getloadavg()[0]
    record = {
        "label": spec.label,
        "path": str(spec.path),
        "dataset": dataset,
        "footprint": footprint(spec.path),
        "timings": timings,
        "memory": memory,
        "load_average_1m_before": round(load_before, 2),
        "load_average_1m_after": round(load_after, 2),
    }
    return record, digests


def _probe_store(
    spec: StoreSpec, selection: dict[str, Any], shapes: list[str]
) -> list[dict[str, Any]]:
    """One fresh interpreter per shape, because a shape's peak is not readable after it."""
    extra = ["--store", f"{spec.label}={spec.path}", "--selection-json", json.dumps(selection)]
    out = []
    for name in shapes:
        record = run_probe(name, extra)
        out.append(record)
        print(
            f"[{spec.label}] {name:38s} baseline={record['baseline_mb']:9.1f} MB  "
            f"peak={record['peak_mb']:9.1f} MB  delta={record['delta_mb']:9.1f} MB",
            flush=True,
        )
    return out


def _check_memory_counts(memory: list[dict[str, Any]], timings: list[dict[str, Any]]) -> None:
    by_shape = {row["query"]: row["result_count"] for row in timings}
    for record in memory:
        expected = by_shape.get(record["query"])
        if expected is not None and record["result_count"] != expected:
            raise SystemExit(
                f"RSS probe for {record['query']!r} returned {record['result_count']} rows "
                f"but the timed run returned {expected}; the probe is not measuring the same query."
            )


def _zarr_node_type(directory: Path) -> str | None:
    if (directory / ".zarray").is_file():
        return "array"
    marker = directory / "zarr.json"
    if not marker.is_file():
        return None
    try:
        node_type = json.loads(marker.read_text(encoding="utf-8")).get("node_type")
    except (OSError, json.JSONDecodeError):
        return "unknown"
    return str(node_type) if node_type else "unknown"


def _walk_entries(root: Path) -> tuple[list[tuple[Path, bool, int, int]], dict[Path, str]]:
    entries: list[tuple[Path, bool, int, int]] = []
    nodes: dict[Path, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        directory = Path(dirpath)
        relative = directory.relative_to(root)
        stat = os.lstat(directory)
        entries.append((relative, True, stat.st_size, stat.st_blocks * 512))
        node_type = _zarr_node_type(directory)
        if node_type is not None:
            nodes[relative] = node_type
        for name in filenames:
            path = directory / name
            file_stat = os.lstat(path)
            entries.append(
                (path.relative_to(root), False, file_stat.st_size, file_stat.st_blocks * 512)
            )
    return entries, nodes


def _bucket_for(
    relative: Path, nodes: dict[Path, str]
) -> tuple[str, Path | None]:
    array_roots = {path for path, node_type in nodes.items() if node_type == "array"}
    for candidate in (relative, *relative.parents):
        if candidate in array_roots:
            return "array", candidate
    if relative in nodes or relative.name in _ZARR_METADATA_NAMES:
        return "zarr_metadata", None
    return "envelope", None


def _empty_bucket() -> dict[str, Any]:
    return {"n_files": 0, "apparent_bytes": 0, "allocated_bytes": 0}


def _du_bytes(path: Path, *flags: str) -> int:
    result = subprocess.run(
        ["du", *flags, str(path)], capture_output=True, text=True, check=True
    )
    return int(result.stdout.split()[0])


def footprint(root: Path) -> dict[str, Any]:
    """File count, apparent/allocated bytes, and a per-array breakdown.

    Detection is on `.zarray` OR `zarr.json` so the same walker covers the Zarr
    v2 baseline and the sharded Zarr v3 copies epic #240 builds. Every array
    node owns its whole subtree (v3 chunk directories included); group metadata
    and the non-Zarr envelope (variant table, SQLite index, npy indexes, manifest)
    are totalled separately. The walked totals are checked against `du -sb` and
    `du -s --block-size=1`, so a footprint that disagrees with itself fails
    rather than being published.
    """
    entries, nodes = _walk_entries(root)
    arrays: dict[str, dict[str, Any]] = defaultdict(_empty_bucket)
    zarr_metadata = _empty_bucket()
    envelope = _empty_bucket()
    envelope["files"] = []
    for relative, is_dir, apparent, allocated in entries:
        kind, owner = _bucket_for(relative, nodes)
        if kind == "array":
            assert owner is not None
            bucket = arrays[str(owner)]
        elif kind == "zarr_metadata":
            bucket = zarr_metadata
        else:
            bucket = envelope
        bucket["apparent_bytes"] += apparent
        bucket["allocated_bytes"] += allocated
        if not is_dir:
            bucket["n_files"] += 1
            if kind == "envelope":
                envelope["files"].append(
                    {
                        "path": str(relative),
                        "apparent_bytes": apparent,
                        "allocated_bytes": allocated,
                    }
                )
    total_apparent = sum(entry[2] for entry in entries)
    total_allocated = sum(entry[3] for entry in entries)
    du_apparent = _du_bytes(root, "-sb")
    du_allocated = _du_bytes(root, "-s", "--block-size=1")
    if (du_apparent, du_allocated) != (total_apparent, total_allocated):
        raise SystemExit(
            f"{root}: du reports {du_apparent}/{du_allocated} apparent/allocated bytes but the "
            f"walked breakdown sums to {total_apparent}/{total_allocated}; refusing to publish "
            "a footprint that disagrees with itself."
        )
    return {
        "n_files": sum(1 for entry in entries if not entry[1]),
        "apparent_bytes": total_apparent,
        "allocated_bytes": total_allocated,
        "du_apparent_bytes": du_apparent,
        "du_allocated_bytes": du_allocated,
        "arrays": [
            {"node": node, **arrays[node]} for node in sorted(arrays)
        ],
        "zarr_metadata": zarr_metadata,
        "envelope": envelope | {"files": sorted(envelope["files"], key=lambda row: row["path"])},
    }


def _environment_block() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "numcodecs": numcodecs.__version__,
        "opengwasdb_commit": provenance()["commit"],
        "hostname": socket.gethostname(),
        "nproc": os.cpu_count(),
        "cache": CACHE_NOTE,
    }


def _selection_from_json(text: str | None) -> dict[str, Any]:
    if not text:
        raise SystemExit("--selection-json is required for an RSS probe")
    try:
        selection = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--selection-json is not valid JSON: {exc}") from exc
    required = {"exposure_analysis_id", "phewas_alid", "region", "random_alids", "random_analyses"}
    missing = sorted(required - set(selection))
    if missing:
        raise SystemExit(f"--selection-json is missing {missing}")
    return selection


def _measure_shape_rss(args: argparse.Namespace, shape: str) -> dict[str, float]:
    """Measure one shape's RSS in this fresh interpreter.

    The parent passes the once-resolved selection, so the probe runs the exact
    query the timed run ran. The pattern mapping is built inside
    `measure_shape_rss`'s own frame (a factory, not a mapping) so it is dropped
    and collected before the baseline is sampled.
    """
    spec = _single_store(args)
    selection = _selection_from_json(args.selection_json)
    q, _plan = _query_shapes.open_benchmark_store(spec.path)
    try:
        return _query_shapes.measure_shape_rss(lambda: _patterns_for_store(q, selection), shape)
    finally:
        q.close()


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--store",
        action="append",
        type=_parse_store,
        required=True,
        metavar="LABEL=PATH",
        help="a labelled Store Release; repeat for every store being compared",
    )
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--selection-json",
        default=None,
        help="internal: the once-resolved selection an RSS probe must reuse",
    )
    _query_shapes.add_common_args(ap)
    return ap


def main() -> None:
    args = _parser().parse_args()
    if _query_shapes.emit_rss_probe(args, _measure_shape_rss):
        return

    stores = _validated_stores(args)
    selection = _resolve_selection(stores[0])
    print(f"selection resolved from {stores[0].label}: {json.dumps(selection, sort_keys=True)}")

    reference_label = stores[0].label
    reference_digests: dict[str, dict[str, str]] = {}
    records: list[dict[str, Any]] = []
    for index, spec in enumerate(stores):
        record, digests = _measure_store(spec, selection, args)
        if index == 0:
            reference_digests = digests
        else:
            assert_identical(reference_label, reference_digests, spec.label, digests)
            print(f"identity check passed: {spec.label} matches {reference_label}", flush=True)
        record["result_digests"] = digests
        records.append(record)

    result = {
        "harness": "benchmark_store_comparison",
        "selection": selection,
        "stores": records,
        "identity": {
            "reference_store": reference_label,
            "stores": [spec.label for spec in stores],
            "shapes": sorted(reference_digests),
            "arrays_per_shape": list(RESULT_ARRAY_NAMES),
            "identical": True,
        },
        "environment": _environment_block(),
        **provenance(),
    }
    write_artifact(args.output, result)


if __name__ == "__main__":
    main()
