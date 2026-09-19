#!/usr/bin/env python3
"""Scaling benchmark for the windowed extract-variant-reference stage (#188, #191).

Generates a manifest of GWAS-SSF sources that each span the whole genome with
heavy cross-file overlap -- the production shape, where every worker contributes
a shard to every genomic window and the tree reduce does real work. Runs
``extract_variant_reference`` across worker counts, window sizes and reduction
batch sizes, asserts every artifact is byte-identical to the first before any
timing is reported, then reports the map, reduce and write phases separately
alongside the end-to-end total and speedup.

    pixi run -e dev python benchmarks/benchmark_extract_variant_reference.py \
        --n-files 128 --variants-per-file 2000 --output /tmp/evr.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY
from opengwasdb.variants.reference import VariantReferenceExtraction, extract_variant_reference

_SSF_HEADER = (
    "chromosome\tbase_pair_location\tother_allele\teffect_allele\tbeta\tstandard_error"
)

#: Approximate GRCh38 chromosome lengths (bp). Used only to lay the synthetic
#: panel out across the genome; the exact lengths do not matter.
_CHROMOSOME_LENGTHS = {
    "1": 248_956_422,
    "2": 242_193_529,
    "3": 198_295_559,
    "4": 190_214_555,
    "5": 181_538_259,
    "6": 170_805_979,
    "7": 159_345_973,
    "8": 145_138_636,
    "9": 138_394_717,
    "10": 133_797_422,
    "11": 135_086_622,
    "12": 133_275_309,
    "13": 114_364_328,
    "14": 107_043_718,
    "15": 101_991_189,
    "16": 90_338_345,
    "17": 83_257_441,
    "18": 80_373_285,
    "19": 58_617_616,
    "20": 64_444_167,
    "21": 46_709_983,
    "22": 50_818_468,
    "X": 156_040_895,
}


@dataclass(frozen=True)
class Run:
    n_workers: int
    window_size_mb: float
    reduction_batch_size: int
    total_seconds: float
    map_seconds: float
    reduce_seconds: float
    write_seconds: float
    other_seconds: float
    speedup: float
    n_variants: int
    n_windows: int
    n_window_shards: int
    n_reduced_windows: int


def _genome_panel(size: int) -> list[tuple[str, int]]:
    """``size`` coordinates spread across every chromosome, proportional to length.

    Because the spacing is near-uniform and far below the smallest window size
    tested, every window at every tested size holds several panel variants --
    which is what makes every worker contribute a shard to every window.
    """
    total = sum(_CHROMOSOME_LENGTHS.values())
    panel: list[tuple[str, int]] = []
    for chromosome, length in _CHROMOSOME_LENGTHS.items():
        count = max(1, round(size * length / total))
        for i in range(count):
            panel.append((chromosome, max(1, (length * (i + 1)) // (count + 1))))
    return panel


def _write_source(path: Path, panel: list[tuple[str, int]]) -> None:
    """One GWAS-SSF source carrying the whole genome-wide panel.

    Every source writes the same panel, so cross-file overlap is total: each
    worker's shard for a window holds every panel variant in that window, and
    the reduce collapses them.
    """
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(_SSF_HEADER + "\n")
        for chromosome, position in panel:
            fh.write(f"{chromosome}\t{position}\tA\tG\t0.1\t0.05\n")


def _make_manifest(root: Path, n_files: int, panel: list[tuple[str, int]]) -> Path:
    manifest = root / "manifest.tsv"
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\tsource_reader_capability\tsource_assembly"
    ]
    sources = root / "sources"
    sources.mkdir(exist_ok=True)
    for i in range(n_files):
        path = sources / f"source_{i:05d}.tsv.gz"
        _write_source(path, panel)
        lines.append(
            f"trait_{i}\t{path}\ttrait {i}\t1000\tsd\tdeclared_standardised"
            f"\t{GWAS_SSF_CAPABILITY}\thg38"
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _run_once(
    manifest: Path,
    artifact: Path,
    baseline: str | None,
    n_workers: int,
    window_size_mb: float,
    batch: int,
) -> tuple[str, VariantReferenceExtraction, float]:
    start = time.monotonic()
    result = extract_variant_reference(
        manifest,
        artifact,
        n_workers=n_workers,
        window_size_mb=window_size_mb,
        reduction_batch_size=batch,
    )
    total = time.monotonic() - start
    text = gzip.decompress(artifact.read_bytes()).decode("utf-8")
    if baseline is not None and text != baseline:
        raise SystemExit(
            f"artifact differs for n_workers={n_workers}, window={window_size_mb}, batch={batch}"
        )
    return text, result, total


def _warm_up(manifest: Path, root: Path, args: argparse.Namespace) -> str:
    """One un-timed extraction so lazy imports and cold caches do not skew the
    first timed run, and to seed the byte-identical baseline."""
    artifact = root / "warmup.variant-ref.tsv.gz"
    text, _result, _total = _run_once(
        manifest,
        artifact,
        None,
        args.worker_counts[0],
        args.window_sizes_mb[0],
        args.reduction_batch_sizes[0],
    )
    return text


def _timed_runs(
    manifest: Path, root: Path, args: argparse.Namespace
) -> tuple[str, list[Run]]:
    baseline: str = _warm_up(manifest, root, args)
    runs: list[Run] = []
    for n_workers in args.worker_counts:
        for window_size_mb in args.window_sizes_mb:
            for batch in args.reduction_batch_sizes:
                samples: list[tuple[float, VariantReferenceExtraction]] = []
                for rep in range(args.repetitions):
                    artifact = (
                        root / f"artifact-{n_workers}-{window_size_mb}-{batch}-{rep}.tsv.gz"
                    )
                    _text, result, total = _run_once(
                        manifest, artifact, baseline, n_workers, window_size_mb, batch
                    )
                    samples.append((total, result))
                first = samples[0][1]
                total_s = statistics.median([sample[0] for sample in samples])
                map_s = statistics.median([sample[1].map_seconds for sample in samples])
                reduce_s = statistics.median([sample[1].reduce_seconds for sample in samples])
                write_s = statistics.median([sample[1].write_seconds for sample in samples])
                runs.append(
                    Run(
                        n_workers=n_workers,
                        window_size_mb=window_size_mb,
                        reduction_batch_size=batch,
                        total_seconds=total_s,
                        map_seconds=map_s,
                        reduce_seconds=reduce_s,
                        write_seconds=write_s,
                        other_seconds=max(0.0, total_s - map_s - reduce_s - write_s),
                        speedup=1.0,
                        n_variants=first.n_variants,
                        n_windows=first.n_windows,
                        n_window_shards=first.n_window_shards,
                        n_reduced_windows=first.n_reduced_windows,
                    )
                )
    assert baseline is not None
    return baseline, runs


def _assert_reduce_did_work(runs: list[Run]) -> None:
    """Every parallel run must show windows with more than one input shard.

    A benchmark where the tree reduce is skipped would silently measure only the
    map and write phases -- exactly the regression this exercise exists to
    expose.
    """
    parallel = [run for run in runs if run.n_workers >= 2]
    if not parallel:
        return
    for run in parallel:
        assert run.n_windows > 0, run
        assert run.n_window_shards > run.n_windows, (
            f"reduce skipped: {run.n_window_shards} shards over {run.n_windows} windows "
            f"for n_workers={run.n_workers}"
        )
        assert run.n_reduced_windows >= 0.9 * run.n_windows, (
            f"reduce did not run on the great majority of windows: "
            f"{run.n_reduced_windows}/{run.n_windows} for n_workers={run.n_workers}"
        )


def _print_table(runs: list[Run], serial_total: float) -> None:
    header = (
        f"{'workers':>7} {'window':>7} {'batch':>5} {'total':>8} {'map':>8} {'reduce':>8} "
        f"{'write':>8} {'other':>8} {'speedup':>7} {'windows':>8} {'shards':>7} {'reduced':>8}"
    )
    print(header)
    for run in runs:
        speedup = serial_total / run.total_seconds if run.total_seconds else float("inf")
        print(
            f"{run.n_workers:>7} {run.window_size_mb:>7g} {run.reduction_batch_size:>5} "
            f"{run.total_seconds:>8.3f} {run.map_seconds:>8.3f} {run.reduce_seconds:>8.3f} "
            f"{run.write_seconds:>8.3f} {run.other_seconds:>8.3f} {speedup:>6.2f}x "
            f"{run.n_windows:>8} {run.n_window_shards:>7} {run.n_reduced_windows:>8}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-files", type=int, default=64)
    parser.add_argument(
        "--variants-per-file",
        type=int,
        default=2000,
        help="genome-wide panel positions written by every source",
    )
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--window-sizes-mb", type=float, nargs="+", default=[5.0, 20.0])
    parser.add_argument("--reduction-batch-sizes", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    panel = _genome_panel(args.variants_per_file)
    root = Path(tempfile.mkdtemp(prefix="evr-benchmark-"))
    try:
        manifest = _make_manifest(root, args.n_files, panel)
        baseline, runs = _timed_runs(manifest, root, args)
        _assert_reduce_did_work(runs)
        n_variants = runs[0].n_variants
        serial_total = statistics.median(
            [run.total_seconds for run in runs if run.n_workers == 1]
        )
        print(
            f"{args.n_files} sources x {len(panel)} genome-wide panel positions, "
            f"{n_variants} unique variants"
        )
        _print_table(runs, serial_total)
        print(f"byte-identical artifact across all {len(runs)} configurations")
        if args.output is not None:
            payload = {
                "n_files": args.n_files,
                "panel_positions_per_file": len(panel),
                "n_variants": n_variants,
                "runs": [asdict(run) for run in runs],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
