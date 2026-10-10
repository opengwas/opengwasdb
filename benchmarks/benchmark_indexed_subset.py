#!/usr/bin/env python3
"""Benchmark the full-statistic HapMap3 Indexed Variant Subset on OGS-00009 (#267).

Promotes the issue-262 prototype into a reproducible harness with one rule: it
refuses to publish a result unless the indexed full-result query decodes
*exactly* to the ordinary filtered result. The measurement is taken against a
reflinked copy of the Store Release, so the authoritative release is never
written, and the original's recursive metadata fingerprint is compared before
and after the run to prove it.

The artifact records every field #267 names -- commit and UTC time; Store, release,
format and encoding identity; input path/checksum and Reference Assembly;
requested/resolved/absent counts; total and per-plane physical bytes; build phase
times and peak RSS; first-read/warm-media/p95 full-result timings; result counts
and the exact-equivalence outcome; and ordinary timings before and after index
generation. ``docs/benchmark-output/opengwasdb_267_indexed_subset_benchmark.qmd``
reads it; the numbers are re-run, never hand-edited.

Usage:
  pixi run -e dev python benchmarks/benchmark_indexed_subset.py \
      [--store PATH] [--work DIR] [--analysis-id ID] [--reps N] \
      [--output PATH] [--reuse-copy] [--skip-build]
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._artifact import provenance, scratch_copy, write_artifact
from benchmarks.measure_build_cost import parse_time_v
from opengwasdb.layouts.dense.indexed_subsets import open_indexed_subset, remove_indexed_subset
from opengwasdb.query import query_store
from opengwasdb.variants.axis import VariantAxis

STORE = Path("/data/opengwasdb/stores/OGS-00009/store.opengwasdb")
WORK = Path("/data/opengwasdb/work/267-indexed")
ANALYSIS_ID = "ukb-b-17805"
SUBSET_NAME = "hm3"
HM3_GZIP = Path("/data/opengwasdb/work/262-hapmap3-prototype/w_hm3.snplist.gz")
HM3_URL = "https://zenodo.org/api/records/7773502/files/w_hm3.snplist.gz/content"
#: The publisher's MD5 for the canonical LDSC HapMap3 list (issue #262 downloaded
#: and verified it). MD5 verifies the published bytes, not security.
HM3_GZIP_MD5 = "153ecc2bcfa740afafe656e6a384d769"
HM3_HEADER = ("SNP", "A1", "A2")

#: The writer's default band; the RSS bound is band-derived, not matrix-derived.
BAND_CELLS = 4_000_000
#: Generous per-band-cell allowance for the index planes plus decode scratch.
BYTES_PER_BAND_CELL = 32
FIXED_OVERHEAD_MB = 1024.0

DEFAULT_REPS = 5
#: Ordinary timings are compared before/after; the envelope is the change the
#: run's own spread can hide (the OGS-00009 report notes ~13% run-to-run).
NOISE_ENVELOPE_PERCENT = 25.0
SUB_SECOND_TARGET_MS = 1_000.0

OUTPUT = Path("docs/benchmark-output/opengwasdb_267_indexed_subset_benchmark.json")

#: Bumped when the required artifact shape changes, so a committed artifact from
#: an older harness is detectable rather than silently trusted.
ARTIFACT_SCHEMA_VERSION = 2

#: The six parallel arrays every selected-Analysis result carries (spec 10b).
#: The six parallel arrays every selected-Analysis result carries (spec 10b).
#: Exact equivalence must check all six; a field silently absent from both sides
#: is not agreement, it is an unmeasured field.
REQUIRED_RESULT_FIELDS: tuple[str, ...] = (
    "variant_index", "analysis_index", "z", "se", "eaf", "association_status",
)
#: Every field #267 names, as dotted paths. ``assert_artifact_complete`` refuses
#: to write an artifact that cannot answer the issue.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "artifact_schema_version",
    "commit",
    "measured_at",
    "opengwasdb_path",
    "opengwasdb_fingerprint",
    "analysis_id",
    "subset_name",
    "store.path",
    "store.authoritative_path",
    "store.copy_kind",
    "store.store_id",
    "store.release_id",
    "store.format_version",
    "store.reference_assembly",
    "store.completion_state",
    "store.layout",
    "store.n_analyses",
    "store.n_variants",
    "store.encoding",
    "input.variant_list_path",
    "input.variant_list_sha256",
    "input.hapmap3_source_path",
    "input.hapmap3_source_md5",
    "input.reference_assembly",
    "input.requested",
    "input.resolved",
    "input.absent",
    "input.alids_derived",
    "input.counts_reconciled",
    "input.counts_reconciliation.writer_requested_equals_alids",
    "input.counts_reconciliation.writer_resolved_equals_alids",
    "input.counts_reconciliation.writer_absent_zero",
    "input.counts_reconciliation.resolution_resolved_equals_alids",
    "input.counts_reconciliation.rsid_budget_balances",
    "input.counts_reconciliation.variant_list_sha256_equals_writer",
    "input.hapmap3_resolution.rsids_requested",
    "input.hapmap3_resolution.resolved_rsids",
    "input.hapmap3_resolution.absent_from_store_axis",
    "input.hapmap3_resolution.allele_incompatible",
    "input.hapmap3_resolution.multiple_store_rows",
    "input.hapmap3_resolution.duplicate_alids",
    "storage.baseline_logical_bytes",
    "storage.baseline_physical_bytes",
    "storage.index_physical_bytes",
    "storage.index_logical_bytes",
    "storage.increase_percent",
    "storage.planes",
    "build.run_mode",
    "build.reused_from",
    "build.total_seconds",
    "build.peak_rss_mb",
    "build.band_cells",
    "build.rss_bound_mb",
    "build.rss_within_bound",
    "build.phases.prepare_seconds",
    "build.phases.publish_seconds",
    "build.peak_rss_source",
    "build.requested",
    "build.resolved",
    "build.absent",
    "build.input_sha256",
    "timings.method.first_read",
    "timings.method.p95",
    "timings.ordinary_before.first_ms",
    "timings.ordinary_before.median_ms",
    "timings.ordinary_before.p95_ms",
    "timings.ordinary_before.result_count",
    "timings.indexed.first_ms",
    "timings.indexed.median_ms",
    "timings.indexed.p95_ms",
    "timings.indexed.result_count",
    "timings.ordinary_after.first_ms",
    "timings.ordinary_after.median_ms",
    "timings.ordinary_after.p95_ms",
    "timings.ordinary_after.result_count",
    "equivalence.required_fields",
    "equivalence.required_fields_present",
    "equivalence.dtypes_match",
    "equivalence.lengths_match",
    "equivalence.non_empty",
    "equivalence.exact",
    "equivalence.fields",
    "equivalence.expected_count",
    "equivalence.indexed_count",
    "run.mode",
    "run.started_at",
    "run.finished_at",
    "run.subset_present_during_ordinary_before",
    "run.subset_published_by_this_run",
    "run.ordinary_bracketed",
    "targets.warm_median_under_1s",
    "targets.exact_equivalence",
    "targets.ordinary_unchanged_within_envelope",
    "targets.original_store_unchanged",
    "targets.peak_rss_within_bound",
    "targets.ordinary_bracketed",
    "publication.published",
    "publication.reason",
)

#: The build subprocess composes the same two functions ``build_indexed_subset``
#: does, so the reach into privates measures the real write/validate phases
#: rather than a re-implementation. It prints its phase timings as one JSON line.
_BUILD_RUNNER = (
    "import json, sys, time\n"
    "from opengwasdb.layouts.dense.indexed_subsets import _prepare_subset, _publish_subset\n"
    "store, name, vlist, assembly, band = sys.argv[1], sys.argv[2], sys.argv[3], "
    "sys.argv[4], int(sys.argv[5])\n"
    "t0 = time.perf_counter()\n"
    "plan = _prepare_subset(store, name, vlist, assembly, band)\n"
    "t1 = time.perf_counter()\n"
    "result = _publish_subset(plan, overwrite=False, band_cells=band)\n"
    "t2 = time.perf_counter()\n"
    "print(json.dumps({'prepare_seconds': t1 - t0, 'publish_seconds': t2 - t1, "
    "'requested': result.requested_count, 'resolved': result.resolved_count, "
    "'absent': result.absent_count, 'n_subset_variants': result.n_subset_variants, "
    "'input_sha256': result.input_sha256}))\n"
)


# ── Input: canonical GRCh38 HapMap3 ALIDs ───────────────────────────────────


@dataclass(frozen=True)
class HapMap3Reference:
    rsid: np.ndarray
    a1: np.ndarray
    a2: np.ndarray


@dataclass(frozen=True)
class HapMap3Resolution:
    variant_index: np.ndarray
    alids: tuple[str, ...]
    diagnostics: dict[str, int]


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_hapmap3(path: Path) -> HapMap3Reference:
    rows: list[tuple[str, str, str]] = []
    with gzip.open(path, "rt") as handle:
        header = tuple(handle.readline().split())
        if header != HM3_HEADER:
            raise SystemExit(f"unexpected HapMap3 header {header!r} in {path}")
        for line in handle:
            fields = line.split()
            if len(fields) != 3:
                raise SystemExit(f"malformed HapMap3 row in {path}: {line.rstrip()!r}")
            rows.append((fields[0], fields[1], fields[2]))
    return HapMap3Reference(
        rsid=np.asarray([row[0] for row in rows], dtype="S24"),
        a1=np.asarray([row[1] for row in rows], dtype=object),
        a2=np.asarray([row[2] for row in rows], dtype=object),
    )


def _candidate_arrays(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def resolve_hapmap3(store: Path, reference: HapMap3Reference) -> HapMap3Resolution:
    """Resolve every HapMap3 rsid to one allele-compatible Store variant ALID.

    A rsid matching several allele-compatible Store variants is refused rather
    than guessed: either row would be a plausible wrong variant (issue #262
    prototype behaviour). rsids absent from the Store axis and rsids whose
    alleles do not match any Store row are counted, never substituted.
    """
    started = time.perf_counter()
    keys = np.load(store / "variant_rsid_bytes.npy", mmap_mode="r")
    rows = np.load(store / "variant_rsid_rows.npy", mmap_mode="r")
    lower = np.searchsorted(keys, reference.rsid, side="left")
    upper = np.searchsorted(keys, reference.rsid, side="right")
    counts = upper - lower
    positions, ordinals = _candidate_arrays(lower, upper)
    candidates = np.asarray(rows[positions], dtype=np.int64)

    axis = VariantAxis(store)
    try:
        identity = axis.identity_by_indices(candidates)
    finally:
        axis.close()
    if identity is None:
        raise SystemExit(f"{store} has no ALID sidecar; allele-safe resolution is impossible")
    stored_a1 = identity["effect_allele"]
    stored_a2 = identity["other_allele"]
    wanted_a1 = reference.a1[ordinals]
    wanted_a2 = reference.a2[ordinals]
    forward = (stored_a1 == wanted_a1) & (stored_a2 == wanted_a2)
    reverse = (stored_a1 == wanted_a2) & (stored_a2 == wanted_a1)
    compatible = forward | reverse
    compatible_counts = np.bincount(ordinals[compatible], minlength=len(reference.rsid))
    ambiguous = int(np.count_nonzero(compatible_counts > 1))
    if ambiguous:
        raise SystemExit(
            f"{ambiguous} HapMap3 rsids resolve to multiple allele-compatible Store "
            "variants; refusing to guess which variant the index should carry"
        )

    chosen = np.flatnonzero(compatible)
    selected_rows = candidates[chosen]
    selected_alids = [str(value) for value in identity["alid"][chosen]]
    by_index: dict[int, str] = {}
    duplicates = 0
    for variant_index, alid in zip(selected_rows, selected_alids, strict=True):
        if int(variant_index) in by_index:
            duplicates += 1
        else:
            by_index[int(variant_index)] = alid
    ordered = sorted(by_index.items())
    diagnostics = {
        "rsids_requested": int(len(reference.rsid)),
        "resolved_rsids": int(np.count_nonzero(compatible_counts > 0)),
        "absent_from_store_axis": int(np.count_nonzero(counts == 0)),
        "allele_incompatible": int(np.count_nonzero((counts > 0) & (compatible_counts == 0))),
        "multiple_store_rows": int(np.count_nonzero(counts > 1)),
        "duplicate_alids": duplicates,
        "resolution_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    return HapMap3Resolution(
        variant_index=np.asarray([row for row, _ in ordered], dtype=np.int64),
        alids=tuple(alid for _, alid in ordered),
        diagnostics=diagnostics,
    )


def write_alid_list(path: Path, alids: tuple[str, ...]) -> str:
    """Write one canonical ALID per line and return the file's sha256."""
    raw = ("\n".join(alids) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


# ── Byte accounting ─────────────────────────────────────────────────────────


def logical_bytes(path: Path) -> int:
    """Sum of file sizes under `path` (the release's apparent size)."""
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def physical_bytes(path: Path) -> int:
    """Allocated bytes for `path` and everything under it, matching `du -s -B1`.

    ``st_blocks`` is in 512-byte units and counts files *and* directories; a
    store is tens of thousands of small chunk files inside deep directories, so
    omitting the directory blocks undercounts the real footprint.
    """
    total = path.stat().st_blocks * 512
    for entry in path.rglob("*"):
        total += entry.stat().st_blocks * 512
    return total


def plane_bytes(group: Path) -> dict[str, int]:
    """Physical bytes per top-level entry of an index group, plus ``total``.

    Bucketing by the group's own top-level names keeps the per-plane split the
    issue asks for while naming whatever arrays the release's encoding defined,
    rather than a hard-coded plane list a future encoding would fall outside.
    ``total`` is the whole group (including its own directory blocks), so it can
    exceed the sum of the child buckets by that directory overhead.
    """
    buckets = {entry.name: physical_bytes(entry) for entry in group.iterdir()}
    buckets["total"] = physical_bytes(group)
    return buckets


def _metadata_fingerprint(store: Path) -> dict[str, Any]:
    """A cheap recursive identity of every file under `store`: path, size, mtime.

    Hashing 34 GB of chunks per run would dominate the benchmark; size and
    mtime catch every mutation a build could make, including one that rewrites a
    chunk with different bytes of the same length. The manifest and analyses.tsv
    are additionally content-hashed because they are the release's identity.
    """
    digest = hashlib.sha256()
    entries: list[tuple[str, int, int, int]] = []
    for entry in store.rglob("*"):
        if not entry.is_file():
            continue
        info = entry.stat()
        entries.append(
            (str(entry.relative_to(store)), info.st_size, info.st_mtime_ns, info.st_blocks * 512)
        )
    entries.sort()
    for relative, size, mtime_ns, _allocated in entries:
        digest.update(f"{relative}\0{size}\0{mtime_ns}\n".encode())
    contents = hashlib.sha256()
    for name in ("manifest.json", "analyses.tsv"):
        contents.update((store / name).read_bytes())
    return {
        "n_files": len(entries),
        "logical_bytes": sum(size for _, size, _, _ in entries),
        "physical_bytes": sum(allocated for _, _, _, allocated in entries),
        "listing_sha256": digest.hexdigest(),
        "identity_sha256": contents.hexdigest(),
    }


# ── Equivalence ─────────────────────────────────────────────────────────────


def subset_ordinary_rows(
    ordinary: dict[str, np.ndarray], subset_variant_index: np.ndarray
) -> dict[str, np.ndarray]:
    """The ordinary full-result rows restricted to a subset's Variant Indices.

    The subset axis is sorted and the ordinary result is in ascending Store
    order, so a searchsorted membership mask reproduces the exact rows an index
    must carry, in the same order.
    """
    subset = np.asarray(subset_variant_index, dtype="int64")
    values = np.asarray(ordinary["variant_index"], dtype="int64")
    positions = np.searchsorted(subset, values)
    in_range = positions < len(subset)
    keep = np.zeros(len(values), dtype=bool)
    keep[in_range] = subset[positions[in_range]] == values[in_range]
    return {name: column[keep] for name, column in ordinary.items()}


def _arrays_equal(expected: np.ndarray, observed: np.ndarray) -> bool:
    if expected.dtype.kind == "f" or observed.dtype.kind == "f":
        return bool(np.array_equal(expected, observed, equal_nan=True))
    return bool(np.array_equal(expected, observed))


def compare_indexed_to_ordinary(
    ordinary: dict[str, np.ndarray],
    indexed: dict[str, np.ndarray],
    subset_variant_index: np.ndarray,
) -> dict[str, Any]:
    """Exact decoded equality of the indexed result and the ordinary subset.

    Publication requires all six ``REQUIRED_RESULT_FIELDS`` to be present on
    both sides, non-empty, of equal length and dtype, and value-equal including
    ``association_status``. A field missing from both sides is an *unmeasured*
    field, not agreement, so it fails: a store format that dropped a column
    would otherwise compare two dicts that simply lack it.
    """
    expected = subset_ordinary_rows(ordinary, subset_variant_index)
    fields = sorted(set(expected) | set(indexed) | set(REQUIRED_RESULT_FIELDS))
    per_field: dict[str, bool] = {}
    dtype_match: dict[str, bool] = {}
    length_match: dict[str, bool] = {}
    for name in fields:
        left = expected.get(name)
        right = indexed.get(name)
        present = left is not None and right is not None
        dtype_match[name] = present and left.dtype == right.dtype
        length_match[name] = present and left.shape == right.shape
        per_field[name] = (
            present and dtype_match[name] and length_match[name] and _arrays_equal(left, right)
        )
    required_present = all(
        name in expected and name in indexed for name in REQUIRED_RESULT_FIELDS
    )
    expected_count = int(expected["variant_index"].size) if "variant_index" in expected else 0
    indexed_count = int(indexed["variant_index"].size) if "variant_index" in indexed else 0
    non_empty = expected_count > 0 and indexed_count > 0
    exact_fields = set(expected) == set(indexed) == set(REQUIRED_RESULT_FIELDS)
    return {
        "exact": bool(
            exact_fields
            and required_present
            and non_empty
            and expected_count == indexed_count
            and all(per_field.values())
        ),
        "required_fields": list(REQUIRED_RESULT_FIELDS),
        "required_fields_present": required_present,
        "dtypes_match": bool(all(dtype_match.values())),
        "lengths_match": bool(all(length_match.values())),
        "non_empty": non_empty,
        "fields": per_field,
        "expected_count": expected_count,
        "indexed_count": indexed_count,
    }


# ── Timing ──────────────────────────────────────────────────────────────────


# ── Timing ──────────────────────────────────────────────────────────────────

#: Disclosed once and reused by every timing block and by ``timings.method``.
#: The OS page cache is never dropped, so ``first_ms`` is not a cold figure.
FIRST_READ_SEMANTICS = (
    "first full-result read of the block in the same process; the OS page cache is "
    "not dropped, so it is not a cold-cache figure, and ordinary_after runs after "
    "ordinary_before has already warmed the column"
)


def p95_method(repetitions: int) -> str:
    """The exact p95 rule, so a reader is not left to assume interpolation."""
    return (
        f"nearest-rank over {repetitions} sorted warm repetitions: "
        f"samples[min(N-1, int(0.95*N))] (max for N<20; N={repetitions})"
    )


def timed(query: Any, repetitions: int) -> dict[str, Any]:
    """First-read plus warm median/p95 milliseconds for one full-result query.

    ``first_ms`` is the first read of this block in a long-lived process; the
    OS page cache is **not** dropped, and an earlier block may already have
    warmed the same column, so it is not a cold-cache measurement. ``p95_ms`` is
    the nearest-rank order statistic over the sorted warm repetitions -- the
    maximum for the five repetitions this harness runs -- and is disclosed so a
    reader does not mistake it for an interpolated percentile.
    """
    started = time.perf_counter()
    first = query()
    first_ms = (time.perf_counter() - started) * 1000.0
    count = int(first["variant_index"].size)
    del first
    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        result = query()
        samples.append((time.perf_counter() - started) * 1000.0)
        del result
    samples.sort()
    median = samples[len(samples) // 2]
    p95_rank = min(len(samples) - 1, int(0.95 * len(samples)))
    p95 = samples[p95_rank]
    return {
        "first_ms": round(first_ms, 3),
        "first_read_semantics": FIRST_READ_SEMANTICS,
        "median_ms": round(median, 3),
        "p95_ms": round(p95, 3),
        "p95_method": p95_method(repetitions),
        "repetitions": repetitions,
        "result_count": count,
    }


# ── Build ───────────────────────────────────────────────────────────────────


def run_build(store: Path, name: str, variant_list: Path, assembly: str, band_cells: int) -> dict:
    """Run the index build under ``/usr/bin/time -v`` for wall time and peak RSS.

    A subprocess rather than in-process: the build's peak must not include the
    harness's own query caches, and ``/usr/bin/time -v`` reports the child's
    high-water mark directly.
    """
    completed = subprocess.run(
        [
            "/usr/bin/time",
            "-v",
            sys.executable,
            "-c",
            _BUILD_RUNNER,
            str(store),
            name,
            str(variant_list),
            assembly,
            str(band_cells),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(f"indexed-subset build failed:\n{completed.stdout}\n{completed.stderr}")
    seconds, maxrss_kib = parse_time_v(completed.stderr)
    phases = json.loads(completed.stdout.strip().splitlines()[-1])
    return {
        "run_mode": "fresh",
        "reused_from": "",
        "total_seconds": round(seconds, 3),
        "peak_rss_mb": round(maxrss_kib / 1024.0, 1),
        "phases": {
            "prepare_seconds": round(float(phases["prepare_seconds"]), 3),
            "publish_seconds": round(float(phases["publish_seconds"]), 3),
        },
        "requested": int(phases["requested"]),
        "resolved": int(phases["resolved"]),
        "absent": int(phases["absent"]),
        "n_subset_variants": int(phases["n_subset_variants"]),
        "input_sha256": str(phases["input_sha256"]),
    }


def reuse_requires_build_stats(
    subset_present_before: bool, build_stats: Path | None
) -> Path | None:
    """Refuse a reuse-only run that would have no genuine build measurement.

    A subset already in the scratch copy means this run will not build it, so
    the build block can only come from an earlier artifact the caller names.
    Without it the harness would write exactly the fabricated zeros this
    refusal exists to prevent.
    """
    if subset_present_before and build_stats is None:
        raise SystemExit(
            "scratch copy already carries the subset; refusing to re-measure it "
            "without --build-stats, because the build block would be a fabricated "
            "zero. Pass --build-stats <artifact> or --reset-subset."
        )
    return build_stats


def load_prior_build(path: Path) -> dict:
    """The genuine build block of an earlier artifact, for a query-only re-run.

    Only used when the caller explicitly passes ``--build-stats`` for a copy
    that already carries the subset. Every field is required and strictly
    validated: a missing or zero build measurement must fail here rather than be
    written into the artifact as a fabricated zero.
    """
    prior = json.loads(path.read_text(encoding="utf-8"))
    build = prior.get("build")
    if not isinstance(build, dict):
        raise SystemExit(f"{path} has no build block to reuse")
    required = (
        "total_seconds",
        "peak_rss_mb",
        "requested",
        "resolved",
        "absent",
        "input_sha256",
    )
    missing = [key for key in required if key not in build]
    phases = build.get("phases") or {}
    missing += [
        f"phases.{key}"
        for key in ("prepare_seconds", "publish_seconds")
        if key not in phases
    ]
    if missing:
        raise SystemExit(f"{path} build block is missing: {', '.join(sorted(missing))}")
    if float(build["total_seconds"]) <= 0.0 or float(build["peak_rss_mb"]) <= 0.0:
        raise SystemExit(f"{path} build block has non-positive measured totals; refusing")
    if int(build["requested"]) <= 0:
        raise SystemExit(f"{path} build block has no requested variant count; refusing")
    return {
        "run_mode": "reused",
        "reused_from": str(path),
        "total_seconds": float(build["total_seconds"]),
        "in_process_seconds": float(
            build.get(
                "in_process_seconds",
                float(phases["prepare_seconds"]) + float(phases["publish_seconds"]),
            )
        ),
        "peak_rss_mb": float(build["peak_rss_mb"]),
        "phases": {
            "prepare_seconds": float(phases["prepare_seconds"]),
            "publish_seconds": float(phases["publish_seconds"]),
        },
        "requested": int(build["requested"]),
        "resolved": int(build["resolved"]),
        "absent": int(build["absent"]),
        "input_sha256": str(build["input_sha256"]),
    }


# ── Artifact ────────────────────────────────────────────────────────────────


def _has(artifact: dict[str, Any], dotted: str) -> bool:
    node: Any = artifact
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def assert_artifact_complete(artifact: dict[str, Any]) -> None:
    """Refuse to publish an artifact missing any #267 field or self-inconsistent."""
    missing = [field for field in _REQUIRED_FIELDS if not _has(artifact, field)]
    if missing:
        raise SystemExit(
            "benchmark artifact is missing #267 fields: " + ", ".join(sorted(missing))
        )
    problems = _semantic_problems(artifact)
    if problems:
        raise SystemExit(
            "benchmark artifact is semantically invalid: " + "; ".join(problems)
        )


def _positive(artifact: dict[str, Any], dotted: str, problems: list[str]) -> None:
    value = artifact
    for part in dotted.split("."):
        value = value[part]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        problems.append(f"{dotted} must be a positive number, got {value!r}")


def _semantic_problems(artifact: dict[str, Any]) -> list[str]:
    """Reasons a structurally complete artifact still cannot be believed."""
    problems: list[str] = []
    if artifact.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        problems.append(
            f"artifact_schema_version is {artifact.get('artifact_schema_version')!r}, "
            f"expected {ARTIFACT_SCHEMA_VERSION}"
        )
    for field in ("analysis_id", "subset_name"):
        if not isinstance(artifact.get(field), str) or not artifact[field]:
            problems.append(f"{field} must be a non-empty string")
    build = artifact["build"]
    if build.get("run_mode") not in {"fresh", "reused"}:
        problems.append(f"build.run_mode is {build.get('run_mode')!r}")
    if build.get("run_mode") == "reused" and not build.get("reused_from"):
        problems.append("build.run_mode is reused but build.reused_from is empty")
    if build.get("run_mode") == "fresh" and build.get("reused_from"):
        problems.append("build.run_mode is fresh but build.reused_from is set")
    for field in ("total_seconds", "peak_rss_mb"):
        _positive(artifact, f"build.{field}", problems)
    _positive(artifact, "build.band_cells", problems)
    requested = build["requested"]
    if requested <= 0 or requested != build["resolved"] + build["absent"]:
        problems.append(
            f"build counts do not balance: requested={requested}, "
            f"resolved={build['resolved']}, absent={build['absent']}"
        )
    inp = artifact["input"]
    if (inp["requested"], inp["resolved"], inp["absent"]) != (
        build["requested"],
        build["resolved"],
        build["absent"],
    ):
        problems.append("input counts are not the writer-returned build counts")
    if inp["variant_list_sha256"] != build["input_sha256"]:
        problems.append("input.variant_list_sha256 does not equal build.input_sha256")
    if inp.get("counts_reconciled") is not True:
        problems.append("input.counts_reconciled is not true")
    equivalence = artifact["equivalence"]
    if list(equivalence.get("required_fields", [])) != list(REQUIRED_RESULT_FIELDS):
        problems.append("equivalence.required_fields is not the six result fields")
    if equivalence.get("exact") is True and not (
        equivalence.get("required_fields_present") is True
        and equivalence.get("dtypes_match") is True
        and equivalence.get("lengths_match") is True
        and equivalence.get("non_empty") is True
        and equivalence.get("expected_count") == equivalence.get("indexed_count")
        and equivalence.get("expected_count")
    ):
        problems.append("equivalence.exact is true but its guards are not")
    for block in ("ordinary_before", "indexed", "ordinary_after"):
        _positive(artifact, f"timings.{block}.median_ms", problems)
        _positive(artifact, f"timings.{block}.result_count", problems)
        _positive(artifact, f"timings.{block}.repetitions", problems)
    run = artifact["run"]
    if run.get("mode") not in {"fresh", "reused"}:
        problems.append(f"run.mode is {run.get('mode')!r}")
    if not isinstance(run.get("ordinary_bracketed"), bool):
        problems.append("run.ordinary_bracketed must be a bool")
    problems.extend(_bracketing_problems(run, build, artifact["targets"]))
    if artifact["publication"]["published"] != artifact["targets"]["exact_equivalence"]:
        problems.append("publication.published disagrees with targets.exact_equivalence")
    return problems


def _bracketing_problems(
    run: dict[str, Any], build: dict[str, Any], targets: dict[str, Any]
) -> list[str]:
    """Reasons the recorded before/after bracket cannot be believed.

    The fields are mutually constrained: a fresh run reused nothing and is the
    only kind that publishes the subset; a reused run had it present throughout;
    and the two `ordinary_bracketed` copies must agree with those facts. Without
    these, a hand-edited or mis-assembled artifact could claim bracketing it did
    not measure.
    """
    problems: list[str] = []
    fresh = run.get("mode") == "fresh"
    present = run.get("subset_present_during_ordinary_before")
    published = run.get("subset_published_by_this_run")
    if run.get("mode") != build.get("run_mode"):
        problems.append("run.mode disagrees with build.run_mode")
    if fresh and present is not False:
        problems.append("a fresh run must have no subset during ordinary_before")
    if not fresh and present is not True:
        problems.append("a reused run must have had the subset during ordinary_before")
    if published is not fresh:
        problems.append("run.subset_published_by_this_run must equal (run.mode == 'fresh')")
    expected_bracketed = bool(fresh and present is False and published is True)
    if run.get("ordinary_bracketed") is not expected_bracketed:
        problems.append("run.ordinary_bracketed disagrees with the run facts")
    if targets.get("ordinary_bracketed") is not run.get("ordinary_bracketed"):
        problems.append("targets.ordinary_bracketed disagrees with run.ordinary_bracketed")
    if targets.get("peak_rss_within_bound") is not build.get("rss_within_bound"):
        problems.append(
            "targets.peak_rss_within_bound disagrees with build.rss_within_bound"
        )
    return problems


def _store_identity(measured: Path, authoritative: Path, copy_method: str) -> dict[str, Any]:
    manifest = json.loads((measured / "manifest.json").read_text(encoding="utf-8"))
    provenance_block = manifest.get("provenance") or {}
    return {
        "path": str(measured),
        "authoritative_path": str(authoritative),
        "copy_kind": copy_method,
        "store_id": str(manifest.get("store_id", "")),
        "release_id": str(manifest.get("release_id", "")),
        "format_version": str(manifest.get("format_version", "")),
        "reference_assembly": str(manifest.get("reference_assembly", "")),
        "completion_state": str(manifest.get("completion_state", "")),
        "layout": str(manifest.get("primary_layout", "")),
        "n_analyses": int(provenance_block.get("n_analyses", 0)),
        "n_variants": int(provenance_block.get("n_variants", 0)),
        "encoding": manifest.get("encoding", {}),
    }


def _ordinary_unchanged(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    denominator = before["median_ms"] or 1.0
    change = abs(after["median_ms"] - before["median_ms"]) / denominator * 100.0
    return {
        "median_change_percent": round(change, 3),
        "noise_envelope_percent": NOISE_ENVELOPE_PERCENT,
        "within_envelope": change <= NOISE_ENVELOPE_PERCENT,
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the whole measurement and return the artifact.

    The before/after bracket is explicit: ``ordinary_before`` is measured only
    after any pre-existing subset has been removed from the scratch copy (or
    when the copy carries none), the build runs next, and ``ordinary_after``
    follows it. The artifact records each of those facts so a reader can prove
    the bracket rather than take it on trust.
    """
    started_at = dt.datetime.now(dt.UTC).isoformat()
    args.work.mkdir(parents=True, exist_ok=True)
    source = args.hm3 if args.hm3.exists() else args.work / args.hm3.name
    source_md5 = md5_of(source)
    if source_md5 != HM3_GZIP_MD5:
        raise SystemExit(
            f"HapMap3 source {source} has MD5 {source_md5}, expected {HM3_GZIP_MD5} "
            f"(the publisher's checksum for {HM3_URL})"
        )
    reference = read_hapmap3(source)

    copy = args.work / "store-copy.opengwasdb"
    method_file = args.work / "scratch-copy-method.txt"
    if not copy.exists():
        copy_method = scratch_copy(args.store, copy)
        method_file.write_text(copy_method + "\n", encoding="utf-8")
    elif not args.reuse_copy:
        raise SystemExit(f"{copy} exists; pass --reuse-copy to reuse it")
    elif method_file.is_file():
        copy_method = method_file.read_text(encoding="utf-8").strip()
    else:
        # Reusing a copy made before the method was recorded: the method is
        # unknown, and the artifact says so rather than assuming a reflink.
        copy_method = "reused_scratch_copy"

    name = args.subset_name
    index_group = copy / "data.zarr" / "indexed_subsets" / name
    if args.reset_subset and index_group.is_dir():
        removed = remove_indexed_subset(copy, name)
        print(f"Removed existing subset {name!r} from scratch copy (removed={removed})", flush=True)
    subset_present_before = index_group.is_dir()
    reuse_requires_build_stats(subset_present_before, args.build_stats)

    before_original = _metadata_fingerprint(args.store)
    resolution = resolve_hapmap3(copy, reference)
    variant_list = args.work / "hm3.grch38.alid.txt"
    variant_sha256 = write_alid_list(variant_list, resolution.alids)
    baseline_physical = physical_bytes(args.store)
    baseline_logical = logical_bytes(args.store)

    query = query_store(copy)
    try:
        # ordinary_before must precede the build: this is the pre-index baseline,
        # and the artifact records that no subset existed when it ran.
        ordinary_before = timed(lambda: query.analysis(args.analysis_id), args.reps)
        if subset_present_before:
            build = load_prior_build(args.build_stats)
            build_mode = "reused"
            print(f"Reusing published subset {name!r} in {copy}", flush=True)
        else:
            build = run_build(copy, name, variant_list, "GRCh38", BAND_CELLS)
            build_mode = "fresh"
            print(f"Built subset {name!r} in {build['total_seconds']} s", flush=True)
        subset_published_by_this_run = build_mode == "fresh" and index_group.is_dir()
        ordinary_bracketed = (
            not subset_present_before and build_mode == "fresh" and index_group.is_dir()
        )
        # The build wrote ~6 GB; until the filesystem finishes flushing it, the
        # ordinary read competes with writeback and reports a slowdown that is
        # not the query path's. Flush before the after-block is timed.
        os.sync()
        subset = open_indexed_subset(copy, name)
        indexed = timed(
            lambda: query.analysis(args.analysis_id, indexed_subset=name), args.reps
        )
        ordinary_after = timed(lambda: query.analysis(args.analysis_id), args.reps)
        ordinary_full = query.analysis(args.analysis_id)
        indexed_full = query.analysis(args.analysis_id, indexed_subset=name)
    finally:
        query.close()
    finished_at = dt.datetime.now(dt.UTC).isoformat()

    equivalence = compare_indexed_to_ordinary(
        ordinary_full, indexed_full, subset.variant_index
    )
    after_original = _metadata_fingerprint(args.store)
    index_physical = physical_bytes(index_group)
    rss_bound_mb = FIXED_OVERHEAD_MB + BAND_CELLS * BYTES_PER_BAND_CELL / (1024 * 1024)
    peak_rss_mb = float(build["peak_rss_mb"])
    rss_within_bound = peak_rss_mb <= rss_bound_mb

    res = resolution.diagnostics
    writer_requested = int(build["requested"])
    writer_resolved = int(build["resolved"])
    writer_absent = int(build["absent"])
    alids_derived = len(resolution.alids)
    counts_reconciliation = {
        "writer_requested_equals_alids": writer_requested == alids_derived,
        "writer_resolved_equals_alids": writer_resolved == alids_derived,
        "writer_absent_zero": writer_absent == 0,
        "resolution_resolved_equals_alids": res["resolved_rsids"] == alids_derived,
        "rsid_budget_balances": (
            res["resolved_rsids"]
            + res["absent_from_store_axis"]
            + res["allele_incompatible"]
            == res["rsids_requested"]
        ),
        "variant_list_sha256_equals_writer": variant_sha256 == str(build["input_sha256"]),
    }
    counts_reconciled = all(counts_reconciliation.values())
    if not counts_reconciled:
        raise SystemExit(
            "input counts do not reconcile between the HapMap3 resolution and the "
            f"writer: {json.dumps(counts_reconciliation, sort_keys=True)}"
        )

    targets = {
        "warm_median_under_1s": indexed["median_ms"] < SUB_SECOND_TARGET_MS,
        "exact_equivalence": bool(equivalence["exact"]),
        "ordinary_unchanged_within_envelope": bool(
            _ordinary_unchanged(ordinary_before, ordinary_after)["within_envelope"]
        ),
        "original_store_unchanged": before_original == after_original,
        "peak_rss_within_bound": rss_within_bound,
        "ordinary_bracketed": bool(ordinary_bracketed),
    }
    artifact: dict[str, Any] = {
        **provenance(),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "analysis_id": str(args.analysis_id),
        "subset_name": name,
        "store": _store_identity(copy, args.store, copy_method),
        "input": {
            "variant_list_path": str(variant_list),
            "variant_list_sha256": variant_sha256,
            "hapmap3_source_path": str(source),
            "hapmap3_source_md5": source_md5,
            "reference_assembly": "GRCh38",
            "requested": writer_requested,
            "resolved": writer_resolved,
            "absent": writer_absent,
            "alids_derived": alids_derived,
            "counts_reconciled": counts_reconciled,
            "counts_reconciliation": counts_reconciliation,
            "hapmap3_resolution": res,
        },
        "storage": {
            "baseline_logical_bytes": baseline_logical,
            "baseline_physical_bytes": baseline_physical,
            "index_physical_bytes": index_physical,
            "index_logical_bytes": logical_bytes(index_group),
            "increase_percent": round(index_physical / baseline_physical * 100.0, 3),
            "planes": plane_bytes(index_group),
        },
        "build": {
            "run_mode": build_mode,
            "reused_from": str(build.get("reused_from", "")),
            "total_seconds": float(build["total_seconds"]),
            "in_process_seconds": round(
                float(
                    build.get(
                        "in_process_seconds",
                        float(build["phases"]["prepare_seconds"])
                        + float(build["phases"]["publish_seconds"]),
                    )
                ),
                3,
            ),
            "peak_rss_mb": peak_rss_mb,
            "band_cells": BAND_CELLS,
            "rss_bound_mb": round(rss_bound_mb, 1),
            "rss_within_bound": rss_within_bound,
            "phases": build["phases"],
            "peak_rss_source": "/usr/bin/time -v (build subprocess)",
            "requested": writer_requested,
            "resolved": writer_resolved,
            "absent": writer_absent,
            "input_sha256": str(build["input_sha256"]),
        },
        "timings": {
            "method": {
                "first_read": FIRST_READ_SEMANTICS,
                "p95": p95_method(args.reps),
            },
            "ordinary_before": ordinary_before,
            "ordinary_after": ordinary_after,
            "indexed": indexed,
        },
        "equivalence": equivalence,
        "run": {
            "mode": build_mode,
            "started_at": started_at,
            "finished_at": finished_at,
            "subset_present_during_ordinary_before": subset_present_before,
            "subset_published_by_this_run": bool(subset_published_by_this_run),
            "ordinary_bracketed": bool(ordinary_bracketed),
            "reset_subset": bool(args.reset_subset),
        },
        "targets": targets,
        "publication": {
            "published": bool(equivalence["exact"]),
            "reason": (
                ""
                if equivalence["exact"]
                else "indexed result is not exactly the ordinary subset"
            ),
        },
        "ordinary_stability": _ordinary_unchanged(ordinary_before, ordinary_after),
    }
    assert_artifact_complete(artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--analysis-id", default=ANALYSIS_ID)
    parser.add_argument("--subset-name", default=SUBSET_NAME)
    parser.add_argument("--hm3", type=Path, default=HM3_GZIP)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reuse-copy", action="store_true")
    parser.add_argument(
        "--reset-subset",
        action="store_true",
        help=(
            "Remove the named subset from the scratch copy before measuring, so "
            "ordinary_before runs with no subset and this run builds it fresh. Never "
            "touches the authoritative store."
        ),
    )
    parser.add_argument(
        "--build-stats",
        type=Path,
        default=None,
        help="An earlier artifact whose measured build block is reused for a query-only re-run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = benchmark(args)
    write_artifact(args.output, artifact)
    if not artifact["publication"]["published"]:
        print("REFUSING to publish: exact equivalence failed", file=sys.stderr)
        return 1
    print(
        f"indexed warm median {artifact['timings']['indexed']['median_ms']} ms, "
        f"ordinary {artifact['timings']['ordinary_before']['median_ms']} ms",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
