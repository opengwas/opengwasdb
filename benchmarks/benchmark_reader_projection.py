"""Benchmark projected and full-row tabular SourceReader paths (issue #179).

Each workload runs in a fresh child process after the source has been read once
to warm the operating-system cache. This keeps peak RSS attributable to one
workload and gives decompression, projected variants, legacy full-row variants,
and associations the same cache conditions.
"""

from __future__ import annotations

import argparse
import gzip
import multiprocessing
import resource
import statistics
import sys
import time
import traceback
from collections.abc import Iterator, Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import IO, Any, cast

from benchmarks._artifact import provenance, write_artifact
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.finngen import FinnGenR13Reader
from opengwasdb.readers.gwas_ssf import GwasSsfReader
from opengwasdb.readers.interface import ReaderAssociation, SourceVariant

_MODES = (
    "decompression_only",
    "projected_variants",
    "legacy_full_row_variants",
    "associations",
)


def _open_binary(path: Path) -> IO[bytes]:
    handle = gzip.open(path, "rb") if str(path).endswith((".gz", ".bgz")) else path.open("rb")
    return cast(IO[bytes], handle)


def _decompress(path: Path) -> tuple[int, int]:
    uncompressed_bytes = 0
    newline_count = 0
    final_byte = b""
    with _open_binary(path) as fh:
        while chunk := fh.read(1024 * 1024):
            uncompressed_bytes += len(chunk)
            newline_count += chunk.count(b"\n")
            final_byte = chunk[-1:]
    line_count = newline_count + (1 if uncompressed_bytes and final_byte != b"\n" else 0)
    return max(0, line_count - 1), uncompressed_bytes


def _reader(kind: str, path: Path) -> FinnGenR13Reader | GwasSsfReader:
    if kind == "finngen_r13":
        return FinnGenR13Reader(path, StoredEffectScale.LOG_OR)
    if kind == "gwas_ssf":
        return GwasSsfReader(path, StoredEffectScale.SD)
    raise ValueError(f"unknown reader kind {kind!r}")


def _legacy_variants(kind: str, path: Path) -> Iterator[SourceVariant]:
    # The association parser remains in production for association streaming,
    # making it a reproducible statement of the pre-#179 variant semantics.
    if kind == "finngen_r13":
        from opengwasdb.readers.finngen import stream_full_row_variants
    elif kind == "gwas_ssf":
        from opengwasdb.readers.gwas_ssf import stream_full_row_variants
    else:
        raise ValueError(f"unknown reader kind {kind!r}")
    yield from stream_full_row_variants(path)


def _consume(records: Iterator[SourceVariant] | Iterator[ReaderAssociation]) -> int:
    return sum(1 for _ in records)


def _measure(kind: str, path: Path, mode: str) -> tuple[int, int | None]:
    if mode == "decompression_only":
        return _decompress(path)
    if mode == "projected_variants":
        return _consume(_reader(kind, path).stream_variants()), None
    if mode == "legacy_full_row_variants":
        return _consume(_legacy_variants(kind, path)), None
    if mode == "associations":
        return _consume(_reader(kind, path).stream_associations()), None
    raise ValueError(f"unknown benchmark mode {mode!r}")


def _worker(send: Connection, kind: str, path: Path, mode: str) -> None:
    try:
        started = time.perf_counter()
        records, uncompressed_bytes = _measure(kind, path, mode)
        elapsed = time.perf_counter() - started
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            peak_rss //= 1024
        send.send(
            {
                "seconds": elapsed,
                "records": records,
                "uncompressed_bytes": uncompressed_bytes,
                "peak_rss_kb": peak_rss,
            }
        )
    except BaseException:
        send.send({"error": traceback.format_exc()})
    finally:
        send.close()


def _isolated_measurement(kind: str, path: Path, mode: str) -> dict[str, Any]:
    methods = multiprocessing.get_all_start_methods()
    context: Any = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(send, kind, path, mode))
    process.start()
    send.close()
    result: dict[str, Any] = receive.recv()
    process.join()
    receive.close()
    if process.exitcode != 0:
        raise RuntimeError(f"{kind} {mode} worker exited {process.exitcode}")
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def _warm_cache(path: Path) -> None:
    _decompress(path)


def _summarise(samples: list[dict[str, Any]], input_bytes: int) -> dict[str, Any]:
    seconds = [float(sample["seconds"]) for sample in samples]
    records = {int(sample["records"]) for sample in samples}
    if len(records) != 1:
        raise RuntimeError(f"record count changed between repetitions: {sorted(records)}")
    median_seconds = statistics.median(seconds)
    record_count = records.pop()
    uncompressed = {sample["uncompressed_bytes"] for sample in samples}
    uncompressed_bytes = uncompressed.pop() if len(uncompressed) == 1 else None
    return {
        "records": record_count,
        "seconds": seconds,
        "median_seconds": median_seconds,
        "records_per_second": record_count / median_seconds,
        "input_mib_per_second": input_bytes / (1024 * 1024) / median_seconds,
        "uncompressed_bytes": uncompressed_bytes,
        "peak_rss_kb": max(int(sample["peak_rss_kb"]) for sample in samples),
        "peak_rss_kb_by_repetition": [int(sample["peak_rss_kb"]) for sample in samples],
    }


def _benchmark_dataset(kind: str, path: Path, repetitions: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    _warm_cache(path)
    samples: dict[str, list[dict[str, Any]]] = {mode: [] for mode in _MODES}
    for _ in range(repetitions):
        for mode in _MODES:
            samples[mode].append(_isolated_measurement(kind, path, mode))
    input_bytes = path.stat().st_size
    measurements = {
        mode: _summarise(mode_samples, input_bytes) for mode, mode_samples in samples.items()
    }
    projected = measurements["projected_variants"]
    legacy = measurements["legacy_full_row_variants"]
    if projected["records"] != legacy["records"]:
        raise RuntimeError(
            f"{kind}: projected path yielded {projected['records']} variants; "
            f"legacy path yielded {legacy['records']}"
        )
    speedup = legacy["median_seconds"] / projected["median_seconds"]
    return {
        "path": str(path.resolve()),
        "input_bytes": input_bytes,
        "cache_condition": "warm OS cache; round-robin workloads in isolated child processes",
        "measurements": measurements,
        "projected_speedup_over_legacy": speedup,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finngen", type=Path, required=True, help="FinnGen R13 TSV[.gz]")
    parser.add_argument("--gwas-ssf", type=Path, required=True, help="GWAS-SSF TSV[.gz]")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--target-speedup", type=float, default=4.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmark-output/opengwasdb_reader_projection_benchmark.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")
    if args.target_speedup <= 0:
        raise SystemExit("--target-speedup must be positive")
    payload = {
        **provenance(),
        "repetitions": args.repetitions,
        "target_finngen_projected_speedup": args.target_speedup,
        "datasets": {
            "finngen_r13": _benchmark_dataset(
                "finngen_r13", args.finngen, args.repetitions
            ),
            "gwas_ssf": _benchmark_dataset("gwas_ssf", args.gwas_ssf, args.repetitions),
        },
    }
    finngen = payload["datasets"]["finngen_r13"]
    finngen["target_speedup"] = args.target_speedup
    finngen["meets_target"] = (
        finngen["projected_speedup_over_legacy"] >= args.target_speedup
    )
    write_artifact(args.output, payload)


if __name__ == "__main__":
    main()
