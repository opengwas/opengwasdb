#!/usr/bin/env python3
"""Scaling benchmark for the windowed extract-variant-reference stage (#188).

Generates a synthetic manifest of overlapping GWAS-SSF sources, then runs
``extract_variant_reference`` across a range of worker counts, window sizes and
reduction batch sizes. Every run's artifact is asserted byte-identical to the
first before any timing is reported, so a fast wrong answer cannot pass.

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
from opengwasdb.variants.reference import extract_variant_reference

_SSF_HEADER = (
    "chromosome\tbase_pair_location\tother_allele\teffect_allele\tbeta\tstandard_error"
)


@dataclass(frozen=True)
class Run:
    n_workers: int
    window_size_mb: float
    reduction_batch_size: int
    seconds: float
    n_variants: int


def _write_source(path: Path, start: int, count: int, overlap: int) -> None:
    """One GWAS-SSF file: ``count`` private variants plus ``overlap`` shared.

    The shared block gives the reduction genuine cross-file overlap to collapse
    early, which is the architecture's whole point.
    """
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(_SSF_HEADER + "\n")
        for i in range(count):
            fh.write(f"1\t{start + i * 1000}\tA\tG\t0.1\t0.05\n")
        for i in range(overlap):
            fh.write(f"1\t{100_000 + i * 1000}\tC\tT\t0.1\t0.05\n")


def _make_manifest(root: Path, n_files: int, variants_per_file: int) -> Path:
    manifest = root / "manifest.tsv"
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\tsource_reader_capability\tsource_assembly"
    ]
    sources = root / "sources"
    sources.mkdir(exist_ok=True)
    for i in range(n_files):
        path = sources / f"source_{i:05d}.tsv.gz"
        _write_source(
            path,
            1_000_000 + i * 10_000_000,
            variants_per_file,
            overlap=variants_per_file // 2,
        )
        lines.append(
            f"trait_{i}\t{path}\ttrait {i}\t1000\tsd\tdeclared_standardised"
            f"\t{GWAS_SSF_CAPABILITY}\thg38"
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-files", type=int, default=64)
    parser.add_argument("--variants-per-file", type=int, default=2000)
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--window-sizes-mb", type=float, nargs="+", default=[5.0, 20.0])
    parser.add_argument("--reduction-batch-sizes", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = Path(tempfile.mkdtemp(prefix="evr-benchmark-"))
    try:
        manifest = _make_manifest(root, args.n_files, args.variants_per_file)
        baseline: str | None = None
        runs: list[Run] = []
        for n_workers in args.worker_counts:
            for window_size_mb in args.window_sizes_mb:
                for batch in args.reduction_batch_sizes:
                    times: list[float] = []
                    result = None
                    for rep in range(args.repetitions):
                        artifact = (
                            root / f"artifact-{n_workers}-{window_size_mb}-{batch}-{rep}.tsv.gz"
                        )
                        start = time.monotonic()
                        result = extract_variant_reference(
                            manifest,
                            artifact,
                            n_workers=n_workers,
                            window_size_mb=window_size_mb,
                            reduction_batch_size=batch,
                        )
                        times.append(time.monotonic() - start)
                        text = gzip.decompress(artifact.read_bytes()).decode("utf-8")
                        if baseline is None:
                            baseline = text
                        elif text != baseline:
                            raise SystemExit(
                                f"artifact differs for n_workers={n_workers}, "
                                f"window={window_size_mb}, batch={batch}"
                            )
                    assert result is not None
                    runs.append(
                        Run(
                            n_workers,
                            window_size_mb,
                            batch,
                            statistics.median(times),
                            result.n_variants,
                        )
                    )

        serial = next(run for run in runs if run.n_workers == 1)
        print(f"{'workers':>7} {'window':>7} {'batch':>5} {'seconds':>8} {'speedup':>7}")
        for run in runs:
            speedup = serial.seconds / run.seconds if run.seconds else float("inf")
            print(
                f"{run.n_workers:>7} {run.window_size_mb:>7g} {run.reduction_batch_size:>5} "
                f"{run.seconds:>8.3f} {speedup:>6.2f}x"
            )
        print(f"unique variants: {serial.n_variants}")
        if args.output is not None:
            payload = {
                "n_files": args.n_files,
                "variants_per_file": args.variants_per_file,
                "n_variants": serial.n_variants,
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
