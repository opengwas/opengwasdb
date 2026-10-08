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
import signal
import socket
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
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
#
# These are DEFAULTS, not constants: a Store Release whose analyses do not carry
# `ukb-b-17805` (OGS-00016's FinnGen R13, OGS-00011's GWAS Catalog Hybrid) cannot
# resolve them, so `--exposure`, `--phewas-alid` and `--region` let the caller
# name the anchors that store actually has. The defaults reproduce the committed
# OGS-00009 baseline byte for byte when no override is given (#250).
DEFAULT_EXPOSURE = "ukb-b-17805"
DEFAULT_PHEWAS_ALID: str | None = None
DEFAULT_REGION = ("19", 44_500_000, 45_500_000)
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


def _parse_region(text: str) -> tuple[str, int, int]:
    """``"19:44500000-45500000"`` -> ``("19", 44500000, 45500000)``.

    A malformed region is a hard error: a silently empty window would make the
    regional shape time an empty read and still agree across stores.
    """
    chrom, separator, span = text.partition(":")
    start_text, dash, end_text = span.partition("-")
    if not (chrom and separator and dash and start_text.isdigit() and end_text.isdigit()):
        raise argparse.ArgumentTypeError(
            f"--region must be CHROM:START-END with integer coordinates, got {text!r}"
        )
    start, end = int(start_text), int(end_text)
    if end <= start:
        raise argparse.ArgumentTypeError(f"--region end must exceed start, got {text!r}")
    return (chrom, start, end)


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


def _resolve_selection(
    spec: StoreSpec,
    *,
    exposure: str = DEFAULT_EXPOSURE,
    phewas_alid: str | None = DEFAULT_PHEWAS_ALID,
    region: tuple[str, int, int] = DEFAULT_REGION,
) -> dict[str, Any]:
    """Resolve the selection once, against the first store, and record it.

    `exposure` must be an Analysis the store carries; the PheWAS variant is
    derived from that Analysis's strongest genome-wide hit unless the caller
    pinned `phewas_alid` (a store whose top hits cannot resolve one). The region
    is the caller's, defaulting to the OGS-00009 chr19 window.
    """
    q, _plan = _query_shapes.open_benchmark_store(spec.path)
    try:
        analyses = q.analyses_table()
        by_id = {row["analysis_id"]: index for index, row in analyses.items()}
        if exposure not in by_id:
            raise SystemExit(
                f"{spec.path}: exposure Analysis {exposure!r} is absent, so the "
                "shared selection cannot be resolved. Refusing to guess one; pass "
                "--exposure with an Analysis this store carries."
            )
        if phewas_alid is not None:
            record_alid = phewas_alid
        else:
            top_hits = q.top_hits(threshold=_query_shapes.GENOME_WIDE)
            keep = top_hits["analysis_index"] == by_id[exposure]
            if not keep.any():
                raise SystemExit(
                    f"{spec.path}: exposure Analysis {exposure!r} has no genome-wide "
                    "significant hits; there is no top hit to derive the PheWAS variant "
                    "from. Pass --phewas-alid to pin one."
                )
            hit_z = np.abs(top_hits["z"][keep])
            strongest = int(top_hits["variant_index"][keep][int(np.argmax(hit_z))])
            record = q._variant_axis.by_index(strongest)
            if record is None:
                raise SystemExit(
                    f"{spec.path}: top-hit variant index {strongest} does not resolve"
                )
            record_alid = record.alid
        random_alids, random_analyses = _query_shapes.resolve_axis_selections(
            q._variant_axis,
            analyses,
            int(q._variant_axis.n_variants),
            len(analyses),
        )
        if len(random_alids) != _query_shapes.RANDOM_AXIS_SIZE:
            raise SystemExit(
                f"{spec.path}: resolved {len(random_alids)} of "
                f"{_query_shapes.RANDOM_AXIS_SIZE} random-lookup variants; a partial "
                "selection is not the selection the report names."
            )
        return {
            "exposure_analysis_id": exposure,
            "phewas_alid": record_alid,
            "region": {"chrom": region[0], "start": region[1], "end": region[2]},
            "random_lookup_shapes": _query_shapes.RANDOM_LOOKUP_SHAPES,
            "random_alids": random_alids,
            "random_analyses": random_analyses,
            "resolved_from": {
                "label": spec.label,
                "path": str(spec.path),
                "n_variants": int(q._variant_axis.n_variants),
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
    """Digest EVERY array the query returned, not only the expected six.

    Hashing just the known contract would silently discard an array one store
    returns and another does not, so the comparison would call two different
    results identical. The expected six must still be present; the digest map's
    keys are otherwise the result's own keys, so `differing_arrays` reports a
    one-sided extra array by name.
    """
    missing = [name for name in RESULT_ARRAY_NAMES if name not in result]
    if missing:
        raise SystemExit(
            f"query result is missing array(s) {missing}; refusing an identity check "
            "that cannot see every array it claims to compare"
        )
    return {name: digest_array(values) for name, values in result.items()}


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


def _check_counts(counts: list[int]) -> None:
    if len(set(counts)) != 1:
        raise SystemExit(
            f"a shape returned {sorted(set(counts))} rows across its runs; "
            "a query whose result size is not stable cannot be timed or compared."
        )


class _ShapeTimeout(Exception):
    """A single query call exceeded `--shape-limit-s`."""


@contextmanager
def _time_limit(seconds: float):
    """Abort the enclosed call after `seconds`, or do nothing when it is 0.

    #252's harness used a 1500 s SIGALRM limit for the slow variant-side shapes
    """
    if not seconds or seconds <= 0:
        yield
        return

    def _raise(_signum: int, _frame: Any) -> None:
        raise _ShapeTimeout

    previous = signal.signal(signal.SIGALRM, _raise)
    try:
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        yield
    finally:
        # Restore the handler even if arming or clearing the timer raises, so a
        # failed `setitimer` cannot leave our handler installed for the rest of
        # the process (#250 review r1, nit 12).
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        finally:
            signal.signal(signal.SIGALRM, previous)


def _wait_for_quiet(max_load: float, timeout_s: float = 3600.0, poll_s: float = 15.0) -> float:
    """Block until the 1-minute load average is below `max_load`; return seconds waited.

    Every *column* waits, not only every pair: a run that times store B after
    store A's timings and its seven RSS probes can otherwise start contended,
    which is what #250 review round 1 found on the 0.2.0 columns. A `max_load`
    of 0 disables the wait; the timeout stops a permanently busy node from
    stalling the run for ever.
    """
    if max_load <= 0:
        return 0.0
    started = time.perf_counter()
    while os.getloadavg()[0] >= max_load:
        if time.perf_counter() - started > timeout_s:
            break
        time.sleep(poll_s)
    return time.perf_counter() - started


def _harness_fingerprint() -> dict[str, str]:
    """The harness file that ran, so an artifact can name it and its digest.

    `opengwasdb_fingerprint` names the package; a 2.18 column runs this harness
    under a *different* checkout, so the artifact must also say which harness
    revision produced the numbers (#250 review r1, minor 9).
    """
    path = Path(__file__).resolve()
    return {
        "harness_path": str(path),
        "harness_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _common_shapes(
    all_digests: list[dict[str, dict[str, str]]],
) -> tuple[set[str], set[str]]:
    """The shapes every store measured, and every shape any store measured.

    A shape that hit the time limit on one store carries no digest and must not
    be compared as if it were equal; the difference is what
    `identity.shapes_not_compared` records (#250 review r1, minor 11).
    """
    measured = set().union(*(set(d) for d in all_digests)) if all_digests else set()
    common = set.intersection(*(set(d) for d in all_digests)) if all_digests else set()
    return common, measured


def _timed_shape(
    fn: Callable[[], dict[str, np.ndarray]],
    reps: int,
    *,
    limit_s: float = 0.0,
    slow_shape_s: float = 0.0,
) -> dict[str, Any]:
    """Time one shape after a shared warm-up, digesting that warm-up.

    With no limit and no slow threshold this is `_median_ms` exactly (one
    untimed warm-up, then `reps` timed calls), which is what the committed
    OGS-00009 baseline used. With `limit_s` set it aborts a call that runs
    longer and returns a `timed_out` record instead of a timing (#252's
    1500 s Hybrid limit); with `slow_shape_s` set a shape whose warm-up is
    slower than the threshold is timed once, as #252's harness did for the
    O(overflow) 2.18 shapes. A timed-out shape carries no digest, so the
    identity check cannot silently compare an empty result.
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

    if not limit_s and not slow_shape_s:
        median_ms, p95_ms, count = _median_ms(instrumented, reps)
        _check_counts(counts)
        return {
            "timed_out": False,
            "median_ms": median_ms,
            "p95_ms": p95_ms,
            "result_count": count,
            "digests": digests,
            "repetitions": reps,
            "warmup_ms": None,
        }

    warmup_start = time.perf_counter()
    try:
        with _time_limit(limit_s):
            instrumented()
    except _ShapeTimeout:
        return {"timed_out": True, "limit_s": limit_s, "digests": {}, "result_count": None}
    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
    repetitions = 1 if (slow_shape_s and warmup_ms / 1000.0 > slow_shape_s) else reps
    times: list[float] = []
    for _ in range(repetitions):
        try:
            started = time.perf_counter()
            with _time_limit(limit_s):
                instrumented()
            times.append((time.perf_counter() - started) * 1000.0)
        except _ShapeTimeout:
            return {
                "timed_out": True,
                "limit_s": limit_s,
                "digests": {},
                "result_count": None,
            }
    _check_counts(counts)
    times.sort()
    return {
        "timed_out": False,
        "median_ms": times[len(times) // 2],
        "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
        "result_count": counts[0],
        "digests": digests,
        "repetitions": repetitions,
        "warmup_ms": round(warmup_ms, 3),
    }


def _dense_plane_root(q: Any) -> Any:
    """The Zarr root that holds the `z`/`se`/`eaf` planes for this store.

    A Dense or Reference-Completed release keeps them at the release root. A
    Hybrid release keeps them in the nested Dense Component
    (`dense/data.zarr`); its outer root holds only `ragged` and `top_hits`, and
    the Hybrid facade has no `_root` at all, so a plane lookup must name the
    component explicitly (#250).
    """
    root = getattr(q, "_root", None)
    if root is not None and "z" in root:
        return root
    dense = getattr(q, "_dense", None)
    if dense is not None and "z" in dense._root:
        return dense._root
    raise SystemExit(
        f"{type(q).__name__} exposes no Dense plane root (z/se/eaf); cannot read its layout"
    )


def _layout_block(q: Any) -> dict[str, Any]:
    """The physical layout of every array a query reads, by component.

    #246 compares shapes, so the artifact must say which shape each store is,
    read back from the arrays rather than from the manifest: a 0.1.0 plane has
    no shard, and a converted one has the shard the conversion wrote. The
    inner chunk is zarr's `chunks` (the read unit), the shard `shards` (the
    file unit). A Hybrid reads *two* components, so its Dense planes are
    prefixed `dense/` and its Overflow's `ragged/*` arrays and outer
    `top_hits` tiers are recorded too (#250 review r1, minor 8).
    """

    def shape_of(array: Any) -> dict[str, Any]:
        shards = getattr(array, "shards", None)
        return {
            # `Array.chunks` is the inner chunk (the read unit) for a v2 array and
            # a v3 sharded array alike; `inner_chunk_of` is the seam's spelling of
            # this, but inlined so the harness also runs against the 2.18 tree,
            # which predates the seam helper (#250).
            "chunk_shape": [int(size) for size in array.chunks],
            "shard_shape": None if shards is None else [int(size) for size in shards],
            "dtype": str(array.dtype),
        }

    def top_hit_tiers(group: Any, prefix: str, layout: dict[str, Any]) -> None:
        if "top_hits" not in group:
            return
        for tier in sorted(group["top_hits"].group_keys()):
            if "z" in group["top_hits"][tier]:
                layout[f"{prefix}top_hits/{tier}/z"] = shape_of(group["top_hits"][tier]["z"])

    outer = getattr(q, "_root", None)
    is_hybrid = outer is None or "z" not in outer
    dense = getattr(q, "_dense", None)._root if is_hybrid else outer
    prefix = "dense/" if is_hybrid else ""

    layout: dict[str, Any] = {}
    for name in ("z", "se", "eaf"):
        if name in dense:
            layout[f"{prefix}{name}"] = shape_of(dense[name])
    top_hit_tiers(dense, prefix, layout)
    if is_hybrid:
        outer = q.store.arrays(mode="r")
        if "ragged" in outer:
            for name in ("z", "se", "eaf", "variant_index", "offsets"):
                if name in outer["ragged"]:
                    layout[f"ragged/{name}"] = shape_of(outer["ragged"][name])
        top_hit_tiers(outer, "", layout)
    return layout


def _dataset_block(q: Any, plan: Any) -> dict[str, Any]:
    return {
        "release_id": plan.release_id,
        "store_id": plan.store_id,
        "n_variants": int(q._variant_axis.n_variants),
        "n_analyses": len(q.analyses_table()),
        "completion_state": str(plan.completion_state),
        "reference_assembly": getattr(plan, "reference_assembly", None),
        "format_version": plan.format_version,
        "encoding": plan.encoding.to_manifest(),
        "layout": _layout_block(q),
    }


def _measure_store(
    spec: StoreSpec, selection: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Time and digest every shape for one store; run its RSS probes if asked."""
    waited_for_load_s = _wait_for_quiet(args.max_start_load)
    load_before = os.getloadavg()[0]
    q, plan = _query_shapes.open_benchmark_store(spec.path)
    try:
        dataset = _dataset_block(q, plan)
        effective_reader = effective_reader_settings(_dense_plane_root(q))
        patterns = _patterns_for_store(q, selection)
        timings: list[dict[str, Any]] = []
        digests: dict[str, dict[str, str]] = {}
        limit_hits: list[str] = []
        after_limit_hit = False
        for name, fn in patterns.items():
            timed = _timed_shape(
                fn,
                args.reps,
                limit_s=args.shape_limit_s,
                slow_shape_s=args.slow_shape_s,
            )
            if timed["timed_out"]:
                limit_hits.append(name)
                after_limit_hit = True
                timings.append(
                    {"query": name, "timed_out": True, "limit_s": args.shape_limit_s}
                )
                print(
                    f"[{spec.label}] {name:38s} HIT the {args.shape_limit_s:g}s limit; "
                    "recorded as a limit hit",
                    flush=True,
                )
                continue
            row: dict[str, Any] = {
                "query": name,
                "timed_out": False,
                "median_ms": round(timed["median_ms"], 3),
                "p95_ms": round(timed["p95_ms"], 3),
                "repetitions": timed["repetitions"],
                "warmup_ms": timed["warmup_ms"],
                "result_count": timed["result_count"],
            }
            if after_limit_hit:
                # A SIGALRM abandons an in-flight Zarr 3 read on its event-loop
                # thread; a shape timed after that shares the CPU with it
                # (#250 review r1, nit 12).
                row["after_limit_hit"] = True
            timings.append(row)
            digests[name] = timed["digests"]
            print(
                f"[{spec.label}] {name:38s} median={timed['median_ms']:9.2f} ms  "
                f"p95={timed['p95_ms']:9.2f} ms  count={timed['result_count']:,}",
                flush=True,
            )
    finally:
        q.close()

    memory = [] if args.skip_rss else _probe_store(spec, selection, list(digests), args)
    _check_memory_counts(memory, timings)
    load_after = os.getloadavg()[0]
    record = {
        "label": spec.label,
        "path": str(spec.path),
        "dataset": dataset,
        "footprint": footprint(spec.path),
        "timings": timings,
        "memory": memory,
        "effective_reader": effective_reader,
        "limit_hits": limit_hits,
        "waited_for_load_s": round(waited_for_load_s, 1),
        "load_average_1m_before": round(load_before, 2),
        "load_average_1m_after": round(load_after, 2),
    }
    return record, digests


def _probe_store(
    spec: StoreSpec, selection: dict[str, Any], shapes: list[str], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """One fresh interpreter per shape, because a shape's peak is not readable after it."""
    extra = [
        "--store",
        f"{spec.label}={spec.path}",
        "--selection-json",
        json.dumps(selection),
        "--shape-limit-s",
        str(args.shape_limit_s),
    ]
    out = []
    for name in shapes:
        record = run_probe(name, extra)
        out.append(record)
        if record.get("timed_out"):
            print(
                f"[{spec.label}] {name:38s} RSS probe HIT the {args.shape_limit_s:g}s limit",
                flush=True,
            )
            continue
        print(
            f"[{spec.label}] {name:38s} baseline={record['baseline_mb']:9.1f} MB  "
            f"peak={record['peak_mb']:9.1f} MB  delta={record['delta_mb']:9.1f} MB",
            flush=True,
        )
    return out


def _check_memory_counts(memory: list[dict[str, Any]], timings: list[dict[str, Any]]) -> None:
    by_shape = {
        row["query"]: row.get("result_count")
        for row in timings
        if not row.get("timed_out")
    }
    for record in memory:
        if record.get("timed_out"):
            continue
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
    # `largest_file_bytes` is the shard size a 0.2.0 array actually stores: a v3
    # array's files are its shards plus one `zarr.json`, so the largest file is
    # its largest shard.  #246 reports shard file sizes for copying, hosting and
    # HTTP range requests; a v2 array's largest file is its largest chunk.
    return {
        "n_files": 0,
        "apparent_bytes": 0,
        "allocated_bytes": 0,
        "largest_file_bytes": 0,
    }


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
            bucket["largest_file_bytes"] = max(bucket["largest_file_bytes"], apparent)
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


def effective_reader_settings(root: Any) -> dict[str, Any]:
    """The reader configuration actually in force, read back from the process.

    The pinned configuration (#244's Blosc threads and one-worker fused
    pipeline, #253's single EAF read) is what makes two runs comparable, so the
    artifact records what ran rather than what the code was meant to set.  The
    thread flag and the worker count are process-wide `zarr.config`/numcodecs
    state; the pipeline is per-array, so its class is read from the plane a
    query actually reads.  `provenance()` separately records the commit and the
    fingerprint of the `opengwasdb` this process imported (#253).
    """
    plane = root["z"]
    # zarr-python 3 dispatches every read through an async array; zarr-python 2
    # has no such wrapper, so its `pipeline`/`max_workers` do not exist and are
    # reported as None. The 2.18 baseline column (#250) runs this same harness
    # under the 2.18 environment, so the absent wrapper must not crash the run.
    async_array = getattr(plane, "_async_array", None)
    codec_pipeline = getattr(async_array, "codec_pipeline", None)
    return {
        # The raw value, not `bool(...)`: numcodecs 0.12 (the 2.18 environment)
        # uses `None` to mean "decide at decode time" (effectively threaded),
        # and coercing that to False would record the opposite of what ran. The
        # 2.18 baseline column (#250) therefore records `null`, and the v3
        # environment records `true`.
        "use_threads": numcodecs.blosc.use_threads,
        "pipeline": None if codec_pipeline is None else type(codec_pipeline).__name__,
        "max_workers": (
            None if codec_pipeline is None else zarr.config.get("codec_pipeline.max_workers", None)
        ),
    }


def _environment_block(effective_reader: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "numcodecs": numcodecs.__version__,
        "opengwasdb_commit": provenance()["commit"],
        "hostname": socket.gethostname(),
        "nproc": os.cpu_count(),
        "cache": CACHE_NOTE,
        "effective_reader": effective_reader,
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
        with _time_limit(args.shape_limit_s):
            return _query_shapes.measure_shape_rss(
                lambda: _patterns_for_store(q, selection), shape
            )
    except _ShapeTimeout:
        return {"query": shape, "timed_out": True, "limit_s": args.shape_limit_s}
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
        "--exposure",
        default=DEFAULT_EXPOSURE,
        metavar="ANALYSIS_ID",
        help=f"the Analysis the shapes are anchored on (default {DEFAULT_EXPOSURE!r}, "
        "the OGS-00009 exposure); must exist in the first store",
    )
    ap.add_argument(
        "--phewas-alid",
        default=DEFAULT_PHEWAS_ALID,
        metavar="ALID",
        help="pin the PheWAS variant instead of deriving it from the exposure's "
        "strongest genome-wide top hit",
    )
    ap.add_argument(
        "--region",
        type=_parse_region,
        default=DEFAULT_REGION,
        metavar="CHROM:START-END",
        help=f"the regional shape's window (default {DEFAULT_REGION[0]}:"
        f"{DEFAULT_REGION[1]}-{DEFAULT_REGION[2]})",
    )
    ap.add_argument(
        "--max-start-load",
        type=float,
        default=3.0,
        metavar="LOAD",
        help="wait until the 1-minute load before each column's timing is below LOAD "
        "(0 disables); every column waits, not only each pair (#250 review r1)",
    )
    ap.add_argument(
        "--selection-json",
        default=None,
        help="internal: the once-resolved selection an RSS probe must reuse",
    )
    ap.add_argument(
        "--shape-limit-s",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="abort one query call after SECONDS and record it as a limit hit "
        "(0 disables it); #252's harness used 1500 s for the slow O(overflow) 2.18 "
        "Hybrid shapes",
    )
    ap.add_argument(
        "--slow-shape-s",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="after a warm-up slower than SECONDS, time the shape once instead of "
        "--reps times (0 disables it), as #252's harness did for the slow shapes",
    )
    _query_shapes.add_common_args(ap)
    return ap


def main() -> None:
    args = _parser().parse_args()
    if _query_shapes.emit_rss_probe(args, _measure_shape_rss):
        return

    stores = _validated_stores(args)
    selection = _resolve_selection(
        stores[0], exposure=args.exposure, phewas_alid=args.phewas_alid, region=args.region
    )
    print(f"selection resolved from {stores[0].label}: {json.dumps(selection, sort_keys=True)}")

    reference_label = stores[0].label
    all_digests: list[dict[str, dict[str, str]]] = []
    records: list[dict[str, Any]] = []
    for spec in stores:
        record, digests = _measure_store(spec, selection, args)
        record["result_digests"] = digests
        all_digests.append(digests)
        records.append(record)

    # A shape that hit the time limit on any store carries no digest, so it is
    # recorded per store but excluded from the identity comparison -- and named
    # in the artifact, so a missing shape cannot hide as "identical".
    common, measured = _common_shapes(all_digests)
    if not common:
        raise SystemExit(
            "no shape was measured by every store; there is nothing to compare and "
            "the run is not evidence"
        )
    reference_digests = {shape: all_digests[0][shape] for shape in common}
    for spec, digests in zip(stores[1:], all_digests[1:], strict=True):
        assert_identical(
            reference_label,
            reference_digests,
            spec.label,
            {shape: digests[shape] for shape in common},
        )
        print(f"identity check passed: {spec.label} matches {reference_label}", flush=True)

    result = {
        "harness": "benchmark_store_comparison",
        "selection": selection,
        "stores": records,
        "identity": {
            "reference_store": reference_label,
            "stores": [spec.label for spec in stores],
            "shapes": sorted(common),
            "shapes_not_compared": sorted(measured - common),
            "arrays_per_shape": list(RESULT_ARRAY_NAMES),
            "identical": True,
        },
        "environment": _environment_block(records[0]["effective_reader"]),
        "harness_file": _harness_fingerprint(),
        **provenance(),
    }
    write_artifact(args.output, result)


if __name__ == "__main__":
    main()
