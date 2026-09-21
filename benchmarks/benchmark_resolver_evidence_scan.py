"""Bounded-prefix and compiled-parser evaluation for resolver evidence scans (issue #209).

Answers two separate questions against the frozen 106-Analysis evaluation
manifest and the full-scan records already produced by opengwasdb-stores#152:

1. **Early termination.** Does a deterministic prefix -- a fixed number of
   source rows, or a fixed number of usable ancestry-reference sites -- preserve
   the full scan's ancestry and phenotype-SD resolution? Every tested rule is
   reported, including the ones that fail.
2. **Parser throughput.** Does a compiled/decompressed prefix parser beat the
   current Python projection, with decompression cost separated from parsing
   cost? Candidate parsers are checked for field-for-field parity with
   `stream_projected_metrics` before any timing is reported.

The ancestry study resolves each Analysis under the full scan and under every
preregistered `ScanLimit`, comparing the two. The parser benchmark measures
whole-file projection and decompression-only throughput in isolated child
processes so peak RSS is attributable to one workload.

    pixi run -e dev python benchmarks/benchmark_resolver_evidence_scan.py \
      --manifest docs/benchmark-output/opengwasdb_resolver_evidence_scan_manifest.tsv \
      --ancestry-reference /data/opengwasdb/reference/ancestry-mixture/ref_freqs.hg38.tsv.gz \
      --ancestry-groups /data/opengwasdb/reference/ancestry-mixture/ancestry_groups.tsv \
      --cores 64 --output docs/benchmark-output/opengwasdb_resolver_evidence_scan.json
"""

from __future__ import annotations

# Issue #209 evaluation harness;
# see docs/spec/bounded-evidence-scan-preregistration.md for the locked rules.
import argparse
import csv
import gzip
import json
import math
import multiprocessing
import statistics
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, astuple, dataclass, field
from pathlib import Path
from typing import Any

import threadpoolctl

# One worker is one CPU-bound Python scan. `_init_worker` caps every forked
# worker's BLAS/OpenMP pools at one thread; without it NumPy's pool gives each
# of 64 workers ~16 spinning threads and pushes a 224-core node past load 1000.
from benchmarks._artifact import provenance, write_artifact
from opengwasdb.ancestry.mixture import AncestryAssignment, Gates
from opengwasdb.ancestry.reference import AncestryReference, load_reference
from opengwasdb.build.resolve import (
    AnalysisRequest,
    ScanLimit,
    resolve_analysis,
)
from opengwasdb.model.enums import OriginalSdMethod, StoredEffectScale
from opengwasdb.readers.gwas_ssf import _METRICS_COLUMNS, GwasSsfReader
from opengwasdb.readers.tabular import (
    TabularMetricsRow,
    _metrics_fields,
    _project_metrics_row,
    _required_metrics_projection,
    stream_projected_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Preregistered stopping rules (docs/spec/bounded-evidence-scan-preregistration.md).
FIXED_ROW_RULES: tuple[int, ...] = (25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
SITE_RULES: tuple[int, ...] = (5_000, 10_000, 20_000, 50_000)
GATES = Gates(tau=0.50, delta=0.20, n_min=5_000, residual_max=0.06, orientation_flip_r=-0.5)
MAF_FLOOR = 0.01
EVIDENCE_SAMPLE = 20_000

# Locked criteria (preregistration section 5).
CONCORDANCE_MIN = 0.98
MAX_RELATIVE_SD_DIFF = 0.02
MAX_RELATIVE_DISPERSION_DIFF = 0.02


@dataclass(frozen=True)
class AnalysisRow:
    analysis_id: str
    stratum: str
    study_design: str
    sample_size: float | None
    data_file: str
    data_bytes: int
    recorded_sha256: str

    @property
    def method(self) -> OriginalSdMethod:
        return (
            OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF
            if self.study_design == "quantitative"
            else OriginalSdMethod.BINARY_TRAIT
        )

    @property
    def scale(self) -> StoredEffectScale:
        if self.study_design == "quantitative":
            return StoredEffectScale.SD
        return StoredEffectScale.LOG_OR


@dataclass
class _Coverage:
    """Genome coverage observed by one scan, for the prefix-locality question."""

    chromosomes: list[str] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)
    _span: dict[str, list[int]] = field(default_factory=dict)

    def observe(self, row: TabularMetricsRow) -> None:
        if row.chromosome not in self._seen:
            self._seen.add(row.chromosome)
            self.chromosomes.append(row.chromosome)
            self._span[row.chromosome] = [row.position, row.position]
            return
        span = self._span[row.chromosome]
        span[0] = min(span[0], row.position)
        span[1] = max(span[1], row.position)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chromosomes": list(self.chromosomes),
            "n_chromosomes": len(self.chromosomes),
            "chromosome_spans": {
                chrom: {"start": span[0], "end": span[1]} for chrom, span in self._span.items()
            },
        }


@dataclass
class _TrackingReader:
    """A reader that passes rows through while recording prefix locality."""

    inner: GwasSsfReader
    coverage: _Coverage

    def stream_metrics(self) -> Iterator[TabularMetricsRow]:
        for row in self.inner.stream_metrics():
            self.coverage.observe(row)
            yield row


def _load_manifest(path: Path) -> list[AnalysisRow]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise SystemExit(f"{path}: empty manifest")
    out: list[AnalysisRow] = []
    for raw in rows:
        sample_size: float | None
        try:
            sample_size = float(raw["sample_size"]) if raw["sample_size"].strip() else None
        except ValueError:
            sample_size = None
        out.append(
            AnalysisRow(
                analysis_id=raw["analysis_id"],
                stratum=raw["stratum"],
                study_design=raw["study_design"],
                sample_size=sample_size,
                data_file=raw["data_file"],
                data_bytes=int(raw["data_bytes"]),
                recorded_sha256=raw.get("sha256", ""),
            )
        )
    return out


def _assign_fields(a: AncestryAssignment | None) -> dict[str, Any]:
    if a is None:
        return {}
    return {
        "assigned_ancestry": a.assigned_ancestry,
        "dominant_superpop": a.dominant_superpop,
        "dominant_proportion": _finite(a.dominant_proportion),
        "runner_up_margin": _finite(a.runner_up_margin),
        "af_overlap": a.af_overlap,
        "residual": _finite(a.residual),
        "gate_reason": a.gate_reason,
        "eaf_orientation": a.eaf_orientation,
        "eaf_orientation_r": _finite(a.eaf_orientation_r),
        "superpop_composition": dict(a.superpop_composition),
    }


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _sd_fields(p: Any) -> dict[str, Any]:
    if p is None:
        return {"status": "unavailable", "reason": "resolution_unavailable"}
    estimate = p.estimate
    return {
        "status": p.status.value,
        "reason": p.reason.value if p.reason is not None else None,
        "sd": _finite(estimate.sd) if estimate is not None else None,
        "dispersion": _finite(estimate.dispersion) if estimate is not None else None,
        "method": estimate.method.value if estimate is not None else None,
        "n_evidence_considered": p.n_evidence_considered,
        "n_estimate_inputs": p.n_estimate_inputs,
        "evidence_sampled": p.evidence_sampled,
    }


def _resolve_with_limit(
    row: AnalysisRow,
    reference: AncestryReference,
    limit: ScanLimit | None,
) -> dict[str, Any]:
    """One bounded (or full) resolution plus wall time and coverage.

    Timing is deliberately not taken under `tracemalloc`: tracing every row's
    allocation inflates this scan by ~2.5x and would make the reported
    throughput a fact about the tracer. Peak memory is measured separately by
    the isolated parser/memory benchmark; here the memory bound is evidenced by
    `ancestry_sites` and the evidence counts, which are what the resolver holds.
    """
    coverage = _Coverage()
    reader = _TrackingReader(GwasSsfReader(row.data_file), coverage)
    request = _request(row)
    started = time.perf_counter()
    resolution = resolve_analysis(
        request,
        reader=reader,
        reference=reference,
        gates=GATES,
        evidence_sample=EVIDENCE_SAMPLE,
        scan_limit=limit,
    )
    elapsed = time.perf_counter() - started
    return {
        "limit": None if limit is None else asdict(limit),
        "stop_reason": resolution.diagnostics.stop_reason.value,
        "rows_read": resolution.diagnostics.rows_read,
        "ancestry_sites": resolution.diagnostics.ancestry_sites,
        "error": resolution.error,
        "seconds": elapsed,
        "ancestry": _assign_fields(resolution.ancestry),
        "phenotype_sd": _sd_fields(resolution.phenotype_sd),
        "coverage": coverage.as_dict(),
    }


def _request(row: AnalysisRow) -> AnalysisRequest:
    return AnalysisRequest(
        analysis_id=row.analysis_id,
        source_file=Path(row.data_file),
        sample_size=row.sample_size,
        original_sd_method=row.method,
        stored_effect_scale=row.scale,
    )


_WORKER_REFERENCE: AncestryReference | None = None
# Held for the worker's lifetime so the thread limit is not garbage-collected
# and lifted before the scan runs.
_WORKER_THREAD_LIMIT: object | None = None


def _init_worker(reference: AncestryReference) -> None:
    global _WORKER_REFERENCE, _WORKER_THREAD_LIMIT
    _WORKER_REFERENCE = reference
    _WORKER_THREAD_LIMIT = threadpoolctl.threadpool_limits(1)


def _analyse_one(args: tuple[AnalysisRow, dict[str, Any] | None]) -> dict[str, Any]:
    row, reused_full = args
    assert _WORKER_REFERENCE is not None
    if not Path(row.data_file).is_file():
        return {"analysis_id": row.analysis_id, "error": "missing_file", "full": None, "rules": []}
    full = _resolve_with_limit(row, _WORKER_REFERENCE, None)
    rules = []
    for max_rows in FIXED_ROW_RULES:
        rules.append(_resolve_with_limit(row, _WORKER_REFERENCE, ScanLimit(max_rows=max_rows)))
    for max_sites in SITE_RULES:
        rules.append(
            _resolve_with_limit(
                row, _WORKER_REFERENCE, ScanLimit(max_ancestry_sites=max_sites)
            )
        )
    return {
        "analysis_id": row.analysis_id,
        "stratum": row.stratum,
        "study_design": row.study_design,
        "data_bytes": row.data_bytes,
        "method": row.method.value,
        "scale": row.scale.value,
        "error": full["error"],
        "full": full,
        "rules": rules,
        "reused_full_record": reused_full,
    }


def _rule_key(limit: dict[str, Any] | None) -> str:
    if limit is None:
        return "full"
    if limit.get("max_rows") is not None:
        return f"rows_{limit['max_rows']}"
    return f"sites_{limit['max_ancestry_sites']}"


def _relative_difference(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0.0:
        return None
    return abs(candidate - baseline) / abs(baseline)


def _evaluate_criteria(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the locked criteria to every preregistered rule, across all strata."""
    usable = [r for r in results if r["full"] is not None and r["error"] in ("", None)]
    per_rule = {
        key: _evaluate_rule(key, usable)
        for key in (*(f"rows_{n}" for n in FIXED_ROW_RULES), *(f"sites_{n}" for n in SITE_RULES))
    }
    passing = [key for key, value in per_rule.items() if value["passes"]]
    recommended = _smallest_rule(passing, per_rule)
    return {"per_rule": per_rule, "passing_rules": passing, "recommended_rule": recommended}


def _rule_bound(key: str) -> tuple[int, int, str]:
    """Sort key for the decision rule: fewest rows, then fewest sites, then name."""
    kind, _, value = key.rpartition("_")
    if kind == "rows":
        return (int(value), 0, key)
    return (0, int(value), key)


def _smallest_rule(passing: list[str], per_rule: dict[str, dict[str, Any]]) -> str | None:
    if not passing:
        return None
    return min(
        passing,
        key=lambda key: (*_rule_bound(key), per_rule[key]["mean_rows_consumed"] or 0.0),
    )


def _evaluate_rule(key: str, usable: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    ancestry_match = 0
    gate_match = 0
    orientation_fail_total = 0
    orientation_fail_kept = 0
    false_positive_eur = 0
    errors = 0
    sd_status_match = 0
    sd_status_total = 0
    controlled_exclusions = 0
    sd_diffs: list[float] = []
    dispersion_diffs: list[float] = []
    for row in usable:
        full = row["full"]
        run = next((x for x in row["rules"] if _rule_key(x["limit"]) == key), None)
        if run is None:
            continue
        total += 1
        if run["error"]:
            errors += 1
        full_anc = full["ancestry"]
        run_anc = run["ancestry"]
        if full_anc.get("assigned_ancestry") == run_anc.get("assigned_ancestry"):
            ancestry_match += 1
        if full_anc.get("gate_reason") == run_anc.get("gate_reason"):
            gate_match += 1
        if full_anc.get("gate_reason") == "eaf_orientation":
            orientation_fail_total += 1
            if run_anc.get("gate_reason") == "eaf_orientation":
                orientation_fail_kept += 1
        if full_anc.get("assigned_ancestry") != "EUR" and run_anc.get("assigned_ancestry") == "EUR":
            false_positive_eur += 1
        full_sd = full["phenotype_sd"]
        run_sd = run["phenotype_sd"]
        if row["study_design"] == "quantitative":
            sd_status_total += 1
            status_matches = run_sd.get("status") == full_sd.get("status")
            if status_matches and run_sd.get("reason") == full_sd.get("reason"):
                sd_status_match += 1
            if full_sd.get("status") == "unavailable":
                controlled_exclusions += 1
        if full_sd.get("status") == "estimated":
            diff = _relative_difference(run_sd.get("sd"), full_sd.get("sd"))
            if diff is not None:
                sd_diffs.append(diff)
            disp = _relative_difference(run_sd.get("dispersion"), full_sd.get("dispersion"))
            if disp is not None:
                dispersion_diffs.append(disp)
    concordance = (ancestry_match / total) if total else 0.0
    gate_concordance = (gate_match / total) if total else 0.0
    orientation_sensitivity = (
        orientation_fail_kept / orientation_fail_total if orientation_fail_total else 1.0
    )
    status_rate = (sd_status_match / sd_status_total) if sd_status_total else 1.0
    criteria = {
        "assignment_concordance": concordance,
        "gate_concordance": gate_concordance,
        "assignment_and_gate_concordance": min(concordance, gate_concordance),
        "orientation_sensitivity": orientation_sensitivity,
        "false_positive_eur": false_positive_eur,
        "errors": errors,
        "sd_status_agreement": status_rate,
        "sd_relative_diff_p95": _percentile(sd_diffs, 95),
        "sd_relative_diff_max": max(sd_diffs) if sd_diffs else None,
        "dispersion_relative_diff_p95": _percentile(dispersion_diffs, 95),
        "dispersion_relative_diff_max": max(dispersion_diffs) if dispersion_diffs else None,
    }
    passes = (
        min(concordance, gate_concordance) >= CONCORDANCE_MIN
        and orientation_sensitivity >= 1.0
        and false_positive_eur == 0
        and errors == 0
        and status_rate >= 1.0
        and (not sd_diffs or max(sd_diffs) <= MAX_RELATIVE_SD_DIFF)
        and (not dispersion_diffs or max(dispersion_diffs) <= MAX_RELATIVE_DISPERSION_DIFF)
    )
    return {
        "n": total,
        "controlled_exclusions": controlled_exclusions,
        "criteria": criteria,
        "passes": passes,
        "mean_stop_reason": _dominant_stop_reason(key, usable),
        "mean_rows_consumed": _mean_rows(key, usable),
    }


def _mean_rows(key: str, usable: list[dict[str, Any]]) -> float | None:
    rows = [
        x["rows_read"]
        for row in usable
        for x in row["rules"]
        if _rule_key(x["limit"]) == key
    ]
    return statistics.fmean(rows) if rows else None


def _dominant_stop_reason(key: str, usable: list[dict[str, Any]]) -> str:
    reasons = [
        x["stop_reason"]
        for row in usable
        for x in row["rules"]
        if _rule_key(x["limit"]) == key
    ]
    if not reasons:
        return ""
    return max(set(reasons), key=reasons.count)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


# --- parser benchmark -------------------------------------------------------


def _python_projected(path: Path, limit_rows: int | None) -> int:
    count = 0
    for _row in stream_projected_metrics(path, _METRICS_COLUMNS):
        count += 1
        if limit_rows is not None and count >= limit_rows:
            break
    return count


def _external_projected(path: Path, binary: str, limit_rows: int | None) -> int:
    """Count projected rows from an argv-safe external decompressor.

    Filenames are passed as argv, never interpolated into a shell. A prefix
    stop closes the pipe and reaps the expected SIGPIPE; a decompressor that
    fails at EOF raises through `_external_rows`.
    """
    rows = _external_rows(path, binary)
    count = 0
    try:
        for _row in rows:
            count += 1
            if limit_rows is not None and count >= limit_rows:
                break
    finally:
        _close(rows)
    return count


def _pandas_parse(path: Path, limit_rows: int | None) -> int:
    """CSV-parse only: how fast pandas' C engine turns bytes into rows.

    This is deliberately *not* a projected read. It measures the upper bound a
    pandas-backed projection could reach, so the report can separate "the CSV
    parser is fast" from "the projected read is fast": the Python projection is
    the bottleneck, and a parser that only removes the CSV split cannot help it.
    """
    import pandas as pd

    wanted = {
        "chromosome",
        "base_pair_location",
        "other_allele",
        "effect_allele",
        "effect_allele_frequency",
        "beta",
        "standard_error",
    }
    count = 0
    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=lambda name: name in wanted,
        chunksize=200_000,
        dtype=str,
        keep_default_na=False,
        na_values=["", ".", "NA", "NaN", "nan", "None"],
    )
    for chunk in reader:
        count += len(chunk)
        if limit_rows is not None and count >= limit_rows:
            break
    return count


def _r_fread_parse(path: Path, limit_rows: int | None) -> int:
    """R `data.table::fread` reference upper bound; never a runtime dependency.

    The filename is passed through `commandArgs()`, not interpolated into a
    shell expression, and `fread` reads the gzip stream itself.
    """
    script = (
        "suppressMessages(library(data.table)); "
        "args <- commandArgs(TRUE); "
        "n <- if (args[2] == 'NA') Inf else as.integer(args[2]); "
        "cols <- c('chromosome','base_pair_location','other_allele','effect_allele',"
        "'effect_allele_frequency','beta','standard_error'); "
        "d <- fread(args[1], select=cols, nrows=n, showProgress=FALSE, verbose=FALSE); "
        "cat(nrow(d))"
    )
    proc = subprocess.run(
        ["Rscript", "-e", script, str(path), "NA" if limit_rows is None else str(limit_rows)],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(proc.stdout.strip().splitlines()[-1])


def _decompress_only(path: Path, method: str) -> int:
    if method == "python_gzip":
        total = 0
        with gzip.open(path, "rb") as fh:
            while chunk := fh.read(1 << 20):
                total += len(chunk)
        return total
    process = subprocess.run(
        [method.replace("_dc", ""), "-dc", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return int(process.returncode == 0)


def _version(command: Sequence[str]) -> str:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        text = (proc.stdout or proc.stderr).strip().splitlines()
        return text[0] if text else ""
    except FileNotFoundError:
        return ""


def _tool_versions() -> dict[str, str]:
    import pandas

    return {
        "python": sys.version.split()[0],
        "pandas": pandas.__version__,
        "gzip": _version(["gzip", "--version"]),
        "pigz": _version(["pigz", "--version"]),
        "R": _version(["R", "--version"]),
        "data.table": _version(
            ["Rscript", "-e", 'cat(as.character(packageVersion("data.table")))']
        ),
    }


def _parity_check(manifest: list[AnalysisRow], sample: int) -> list[dict[str, Any]]:
    """Candidate projected parsers must agree with `stream_projected_metrics`."""
    chosen = _parser_sample(manifest, sample)
    checks = []
    for row in chosen:
        path = Path(row.data_file)
        production = [
            _row_signature(r)
            for r in _take(stream_projected_metrics(path, _METRICS_COLUMNS), 5_000)
        ]
        compared = 0
        for binary in ("gzip", "pigz"):
            external = [_row_signature(r) for r in _take(_external_rows(path, binary), 5_000)]
            if production != external:
                raise SystemExit(
                    f"parser parity failed: external {binary} differs from production "
                    f"on {row.analysis_id}"
                )
            compared += len(external)
        checks.append(
            {
                "analysis_id": row.analysis_id,
                "projected_rows_compared": len(production),
                "external_rows_compared": compared,
                "production_equals_external": True,
            }
        )
    return checks


def _external_rows(path: Path, binary: str) -> Iterator[TabularMetricsRow]:
    process = subprocess.Popen(
        [binary, "-dc", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.stdout is not None
    reached_eof = False
    try:
        header_line = process.stdout.readline()
        projection = _required_metrics_projection(path, header_line, _METRICS_COLUMNS)
        for line in process.stdout:
            cells = _metrics_fields(line, projection.split_limit)
            if len(cells) <= projection.last_identity:
                continue
            projected = _project_metrics_row(cells, projection)
            if projected is not None:
                yield projected
        reached_eof = True
    finally:
        process.stdout.close()
        process.wait()
        if reached_eof and process.returncode != 0:
            stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
            raise RuntimeError(
                f"{binary} failed on {path}: rc={process.returncode}: {stderr[:400]}"
            )
        if process.stderr is not None:
            process.stderr.close()


def _close(iterator: object) -> None:
    """Close a generator if it has `close`, without a `type: ignore`."""
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _take(iterator: Iterator[Any], count: int | None) -> list[Any]:
    out: list[Any] = []
    try:
        for item in iterator:
            out.append(item)
            if count is not None and len(out) >= count:
                break
    finally:
        _close(iterator)
    return out


def _row_signature(row: TabularMetricsRow) -> tuple[Any, ...]:
    return astuple(row)


def _parser_sample(manifest: list[AnalysisRow], sample: int) -> list[AnalysisRow]:
    """One source per size decile, plus the legacy-layout edge case, cheapest first."""
    ordered = sorted(manifest, key=lambda r: r.data_bytes)
    if sample >= len(ordered):
        return ordered
    step = max(1, len(ordered) // sample)
    chosen = ordered[::step][:sample]
    legacy = next((r for r in manifest if r.analysis_id == "GCST000553"), None)
    if legacy is not None and legacy not in chosen:
        chosen[-1] = legacy
    return chosen


@dataclass
class _ParserMeasurement:
    seconds: float
    rows: int
    peak_rss_kb: int


_PROJECTED_METHODS = ("current", "gzip_dc", "pigz_dc")
_PARSE_ONLY_METHODS = ("pandas_parse", "r_fread_parse")


def _parser_worker(send: Any, path: str, method: str, limit_rows: int | None) -> None:
    try:
        started = time.perf_counter()
        if method == "current":
            rows = _python_projected(Path(path), limit_rows)
        elif method == "gzip_dc":
            rows = _external_projected(Path(path), "gzip", limit_rows)
        elif method == "pigz_dc":
            rows = _external_projected(Path(path), "pigz", limit_rows)
        elif method == "pandas_parse":
            rows = _pandas_parse(Path(path), limit_rows)
        elif method == "r_fread_parse":
            rows = _r_fread_parse(Path(path), limit_rows)
        else:
            raise ValueError(f"unknown parser method {method!r}")
            raise ValueError(f"unknown parser method {method!r}")
        elapsed = time.perf_counter() - started
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            peak //= 1024
        send.send({"seconds": elapsed, "rows": rows, "peak_rss_kb": peak})
    except BaseException as exc:
        import traceback

        send.send({"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
    finally:
        send.close()


def _isolated_parser_measurement(
    path: Path, method: str, limit_rows: int | None
) -> _ParserMeasurement:
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_parser_worker, args=(send, str(path), method, limit_rows))
    process.start()
    send.close()
    result = receive.recv()
    process.join()
    receive.close()
    if "error" in result:
        raise RuntimeError(result["error"])
    return _ParserMeasurement(
        seconds=float(result["seconds"]),
        rows=int(result["rows"]),
        peak_rss_kb=int(result["peak_rss_kb"]),
    )


def _benchmark_parsers(
    manifest: list[AnalysisRow], sample: int, repetitions: int
) -> dict[str, Any]:
    chosen = _parser_sample(manifest, sample)
    datasets = []
    for row in chosen:
        path = Path(row.data_file)
        entry: dict[str, Any] = {
            "analysis_id": row.analysis_id,
            "path": str(path),
            "input_bytes": path.stat().st_size,
            "projected": {},
            "parse_only": {},
        }
        for method in _PROJECTED_METHODS:
            entry["projected"][method] = _measure_method(path, method, repetitions)
        for method in _PARSE_ONLY_METHODS:
            if method == "r_fread_parse" and not _has_r_data_table():
                continue
            entry["parse_only"][method] = _measure_method(path, method, repetitions)
        current_seconds = entry["projected"]["current"]["median_seconds"]
        for method in _PROJECTED_METHODS:
            if method == "current":
                continue
            method_seconds = entry["projected"][method]["median_seconds"]
            entry["projected"][method]["speedup_over_current"] = current_seconds / method_seconds
        datasets.append(entry)

    decompression = _benchmark_decompression(chosen, repetitions)
    return {
        "sample": "one source per size decile plus the legacy-layout edge case",
        "repetitions": repetitions,
        "projected_methods": list(_PROJECTED_METHODS),
        "parse_only_methods": list(_PARSE_ONLY_METHODS),
        "interpretation": (
            "projected methods produce resolver rows and are parity-checked against "
            "stream_projected_metrics; parse-only methods are CSV-parse upper bounds "
            "and are not equivalent projected reads"
        ),
        "datasets": datasets,
        "decompression_only": decompression,
    }


def _measure_method(path: Path, method: str, repetitions: int) -> dict[str, Any]:
    samples = [_isolated_parser_measurement(path, method, None) for _ in range(repetitions)]
    counts = {s.rows for s in samples}
    if len(counts) != 1:
        raise RuntimeError(f"{method} row count changed across repetitions: {counts}")
    count = counts.pop()
    median_seconds = statistics.median(s.seconds for s in samples)
    return {
        "rows": count,
        "median_seconds": median_seconds,
        "rows_per_second": count / median_seconds,
        "seconds": [s.seconds for s in samples],
        "peak_rss_kb": max(s.peak_rss_kb for s in samples),
        "input_mib_per_second": path.stat().st_size / (1024 * 1024) / median_seconds,
    }


def _has_r_data_table() -> bool:
    proc = subprocess.run(
        ["Rscript", "-e", 'cat(requireNamespace("data.table", quietly=TRUE))'],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip().endswith("TRUE")


def _benchmark_decompression(
    chosen: list[AnalysisRow], repetitions: int
) -> list[dict[str, Any]]:
    out = []
    for row in chosen:
        path = Path(row.data_file)
        entry: dict[str, Any] = {"analysis_id": row.analysis_id, "input_bytes": path.stat().st_size}
        for method in ("python_gzip", "gzip_dc", "pigz_dc"):
            samples = []
            for _ in range(repetitions):
                started = time.perf_counter()
                value = _decompress_only(path, method)
                samples.append(time.perf_counter() - started)
            median_seconds = statistics.median(samples)
            entry[method] = {
                "median_seconds": median_seconds,
                "input_mib_per_second": entry["input_bytes"] / (1024 * 1024) / median_seconds,
                "value": value,
            }
        out.append(entry)
    return out


# --- main -------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ancestry-reference", type=Path, required=True)
    parser.add_argument("--ancestry-groups", type=Path, required=True)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--max-analyses", type=int, default=None)
    parser.add_argument("--parsers-only", action="store_true")
    parser.add_argument("--parser-analyses", type=int, default=10)
    parser.add_argument("--parser-repetitions", type=int, default=1)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Also render the Markdown results to this path",
    )
    parser.add_argument(
        "--report-only",
        type=Path,
        default=None,
        help="Render a report from an existing JSON artifact without re-running",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        default=None,
        help="Merge this artifact's sections into the --report-only payload",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs/benchmark-output/opengwasdb_resolver_evidence_scan.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = _load_manifest(args.manifest)
    if args.max_analyses is not None:
        manifest = manifest[: args.max_analyses]

    if args.report_only is not None:
        payload = json.loads(args.report_only.read_text(encoding="utf-8"))
        if args.merge is not None:
            payload.update(json.loads(args.merge.read_text(encoding="utf-8")))
        report = _render_markdown(payload, manifest)
        destination = args.report or args.report_only.with_suffix(".md")
        destination.write_text(report, encoding="utf-8")
        print(f"Wrote {destination}")
        return

    payload: dict[str, Any] = {
        **provenance(),
        "issue": 209,
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "tool_versions": _tool_versions(),
    }

    if not args.parsers_only:
        reference = load_reference(
            args.ancestry_reference, args.ancestry_groups, maf_floor=MAF_FLOOR
        )
        print(f"Loaded ancestry reference: {reference.n_variants} variants", flush=True)
        results = _run_study(manifest, reference, args.cores)
        payload["study"] = {
            "cache_condition": "warm OS cache after the per-Analysis full scan",
            "gates": asdict(GATES),
            "extraction_panel": None,
            "maf_floor": MAF_FLOOR,
            "evidence_sample": EVIDENCE_SAMPLE,
            "fixed_row_rules": list(FIXED_ROW_RULES),
            "site_rules": list(SITE_RULES),
            "results": results,
            "decision": _evaluate_criteria(results),
        }
        _write_report(payload["study"])

    if args.parsers_only or args.parser_analyses > 0:
        if not args.skip_parity:
            payload["parity"] = _parity_check(manifest, args.parser_analyses)
        payload["parsers"] = _benchmark_parsers(
            manifest, args.parser_analyses, args.parser_repetitions
        )

    write_artifact(args.output, payload)
    if args.report is not None:
        args.report.write_text(_render_markdown(payload, manifest), encoding="utf-8")
        print(f"Wrote {args.report}")


def _run_study(
    manifest: list[AnalysisRow], reference: AncestryReference, cores: int
) -> list[dict[str, Any]]:
    tasks = [(row, None) for row in manifest]
    if cores <= 1:
        _init_worker(reference)
        return [_analyse_one(task) for task in tasks]
    context = multiprocessing.get_context("fork")
    with context.Pool(cores, initializer=_init_worker, initargs=(reference,)) as pool:
        return pool.map(_analyse_one, tasks)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(study: dict[str, Any]) -> None:
    """Print the decision table so a run leaves its evidence in the log too."""
    decision = study["decision"]
    print("\nrule                         concordance  orient  fpEUR  errors  sd-status  passes")
    for key, value in sorted(decision["per_rule"].items()):
        c = value["criteria"]
        print(
            f"{key:28s} {c['assignment_and_gate_concordance']:10.4f} "
            f"{c['orientation_sensitivity']:7.3f} {c['false_positive_eur']:6d} "
            f"{c['errors']:7d} {c['sd_status_agreement']:10.3f} {str(value['passes']):>7s}"
        )
    print(f"\npassing rules: {decision['passing_rules'] or 'none'}")


# --- report ----------------------------------------------------------------


def _chromosome_rank(chromosome: str) -> int:
    stripped = chromosome[3:] if chromosome.lower().startswith("chr") else chromosome
    if stripped.isdigit():
        return int(stripped)
    return {"X": 23, "Y": 24, "MT": 25, "M": 25}.get(stripped.upper(), 99)


def _chromosome_ordering(chromosomes: list[str]) -> str:
    if len(chromosomes) <= 1:
        return "single_chromosome"
    ranked = sorted(chromosomes, key=_chromosome_rank)
    return "sorted" if chromosomes == ranked else "interleaved"


def _source_layout(path: Path) -> str:
    """`legacy_hm` when the header carries the old `hm_*` alias columns."""
    try:
        with gzip.open(path, "rb") as fh:
            header = fh.readline()
    except OSError:
        return "unreadable"
    return "legacy_hm" if b"hm_chrom" in header or b"hm_variant_id" in header else "current"


def _augment_results(results: list[dict[str, Any]], manifest: list[AnalysisRow]) -> None:
    """Derive the stratification facts that need the source header or coverage."""
    by_id = {row.analysis_id: row for row in manifest}
    for result in results:
        if result.get("full") is None:
            result["layout"] = "unreadable"
            result["chromosome_ordering"] = "unreadable"
            continue
        row = by_id.get(result["analysis_id"])
        result["layout"] = _source_layout(Path(row.data_file)) if row else "unknown"
        result["chromosome_ordering"] = _chromosome_ordering(
            result["full"]["coverage"]["chromosomes"]
        )


def _size_band(data_bytes: int) -> str:
    for ceiling, label in (
        (10_000_000, "<10MB"),
        (100_000_000, "10-100MB"),
        (500_000_000, "100-500MB"),
        (1_500_000_000, "500MB-1.5GB"),
    ):
        if data_bytes < ceiling:
            return label
    return ">=1.5GB"


def _group_value(result: dict[str, Any], dimension: str) -> str:
    if dimension == "study_design":
        return result["study_design"]
    if dimension == "size":
        return _size_band(result["data_bytes"])
    if dimension == "layout":
        return result["layout"]
    if dimension == "chromosome_ordering":
        return result["chromosome_ordering"]
    if dimension == "sd_availability":
        full = result.get("full") or {}
        return (full.get("phenotype_sd") or {}).get("status", "unknown")
    if dimension == "failure_mode":
        full = result.get("full") or {}
        return (full.get("ancestry") or {}).get("gate_reason", "unknown")
    return "unknown"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _render_markdown(payload: dict[str, Any], manifest: list[AnalysisRow]) -> str:
    study = payload.get("study")
    if study is None:
        raise SystemExit("no study section in the artifact")
    _augment_results(study["results"], manifest)
    usable = [r for r in study["results"] if r.get("full") and not r.get("error")]
    decision = study["decision"]

    out: list[str] = [
        "# Bounded-Prefix and Compiled-Parser Resolver Evidence Scan: Results (#209)",
        "",
        "Generated by `benchmarks/benchmark_resolver_evidence_scan.py --report-only`; ",
        "numbers come from the JSON artifact, never by hand.",
        "",
        "- Issue: opengwasdb#209",
        f"- Measured commit: `{payload.get('commit', '')}`",
        f"- Measured at: {payload.get('measured_at', '')}",
        f"- Manifest: `{payload.get('manifest', '')}` "
        f"(sha256 `{payload.get('manifest_sha256', '')}`)",
        f"- Analyses evaluated: {len(usable)} of {len(study['results'])}",
        f"- Cache condition: {study.get('cache_condition', '')}",
        f"- Gates: `{study.get('gates')}`",
        f"- Tool versions: `{payload.get('tool_versions')}`",
        "",
        "Preregistered rules and locked criteria: "
        "[`docs/spec/bounded-evidence-scan-preregistration.md`](../spec/bounded-evidence-scan-preregistration.md).",
        "",
        "## How to read this",
        "",
        "Each Analysis is resolved twice: once with a full scan and once under each",
        "preregistered bound. Every rule is compared with the full scan for the same",
        "Analysis, so a rule's verdict is a statement about that rule, not about a",
        "different sample. Concordance is the fraction of Analyses whose Assigned",
        "Ancestry *and* gate reason both match the full scan. `FP EUR` counts Analyses",
        "the full scan leaves Unassigned (or gates out) that the rule assigns EUR -- a",
        "locked criterion of zero. SD criteria apply to quantitative Analyses with a",
        "full-scan estimate; quantitative Analyses with no usable source AF are",
        "reported as controlled exclusions in the denominator rather than dropped.",
        "",
        "The parser section separates three things: the projected read (resolver rows,",
        "parity-checked against `stream_projected_metrics`), CSV-parse-only throughput",
        "(an upper bound, not a projected read) and decompression-only throughput.",
        "",
        "## Decision",
        "",
    ]
    recommended = decision.get("recommended_rule")
    if recommended:
        out.append(
            f"**Adopt `{recommended}`** -- the smallest rule satisfying every locked criterion; "
            "full scanning remains the default until this mode is independently validated."
        )
    else:
        out.append(
            "**No early-stop rule passes the locked criteria.** Retain the full scan. "
            "The parser benchmark below decides whether a semantics-preserving parser "
            "improvement is warranted instead."
        )
    out += ["", "### Every tested rule", "", _rule_table(study["decision"]["per_rule"])]

    out += ["", "## Stratified concordance", ""]
    for dimension in (
        "study_design",
        "size",
        "layout",
        "chromosome_ordering",
        "sd_availability",
        "failure_mode",
    ):
        out.append(_stratified_table(usable, dimension))
        out.append("")

    out += ["## Prefix locality and genome coverage", "", _locality_table(usable)]
    out += ["", "## Phenotype-SD stability", "", _sd_table(decision["per_rule"])]

    if "parsers" in payload:
        out += ["", "## Parser benchmark", "", _parser_report(payload)]
    if "parity" in payload:
        out += ["", "### Parser parity", "", _parity_report(payload["parity"])]
    return "\n".join(out) + "\n"


def _rule_table(per_rule: dict[str, dict[str, Any]]) -> str:
    rows = []
    for key, value in sorted(per_rule.items(), key=lambda item: _rule_bound(item[0])):
        c = value["criteria"]
        rows.append(
            [
                key,
                f"{c['assignment_and_gate_concordance']:.4f}",
                f"{c['orientation_sensitivity']:.3f}",
                str(c["false_positive_eur"]),
                str(c["errors"]),
                f"{c['sd_status_agreement']:.3f}",
                _fmt(c["sd_relative_diff_max"]),
                _fmt(c["dispersion_relative_diff_max"]),
                "pass" if value["passes"] else "fail",
            ]
        )
    return _md_table(
        [
            "rule",
            "assignment+gate",
            "orientation",
            "FP EUR",
            "errors",
            "SD status",
            "SD max",
            "disp max",
            "verdict",
        ],
        rows,
    )


def _stratified_table(usable: list[dict[str, Any]], dimension: str) -> str:
    rules = [
        *(f"rows_{n}" for n in FIXED_ROW_RULES),
        *(f"sites_{n}" for n in SITE_RULES),
    ]
    groups = sorted({_group_value(r, dimension) for r in usable})
    rows = []
    for group in groups:
        subset = [r for r in usable if _group_value(r, dimension) == group]
        cells = []
        for rule in rules:
            evaluated = _evaluate_rule(rule, subset)
            c = evaluated["criteria"]
            fp = c["false_positive_eur"]
            suffix = f" (FP {fp})" if fp else ""
            cells.append(f"{c['assignment_and_gate_concordance']:.2f}{suffix}")
        rows.append([f"{group} (n={len(subset)})", *cells])
    headers = [dimension, *(rule.replace("rows_", "r").replace("sites_", "s") for rule in rules)]
    return f"#### By {dimension}\n\n" + _md_table(headers, rows)


def _locality_table(usable: list[dict[str, Any]]) -> str:
    rows = []
    for key in (
        *(f"rows_{n}" for n in FIXED_ROW_RULES),
        *(f"sites_{n}" for n in SITE_RULES),
    ):
        runs = [
            run
            for result in usable
            for run in result["rules"]
            if _rule_key(run["limit"]) == key
        ]
        if not runs:
            continue
        full_chrom = [
            len(result["full"]["coverage"]["chromosomes"]) for result in usable if result["full"]
        ]
        coverage = [
            len(run["coverage"]["chromosomes"]) / max(1, full)
            for run, full in zip(runs, full_chrom, strict=False)
        ]
        rows.append(
            [
                key,
                _fmt(statistics.fmean(run["rows_read"] for run in runs), 1),
                _fmt(statistics.fmean(run["ancestry_sites"] for run in runs), 1),
                _fmt(statistics.fmean(coverage) * 100.0, 1),
                _dominant([run["stop_reason"] for run in runs]),
            ]
        )
    return _md_table(
        ["rule", "mean rows", "mean sites", "mean chromosome coverage %", "modal stop reason"],
        rows,
    )


def _sd_table(per_rule: dict[str, dict[str, Any]]) -> str:
    rows = []
    for key, value in sorted(per_rule.items(), key=lambda item: _rule_bound(item[0])):
        c = value["criteria"]
        rows.append(
            [
                key,
                str(value["n"]),
                str(value["controlled_exclusions"]),
                f"{c['sd_status_agreement']:.3f}",
                _fmt(c["sd_relative_diff_p95"]),
                _fmt(c["sd_relative_diff_max"]),
                _fmt(c["dispersion_relative_diff_p95"]),
                _fmt(c["dispersion_relative_diff_max"]),
            ]
        )
    return _md_table(
        ["rule", "n", "controlled excl.", "SD status", "SD p95", "SD max", "disp p95", "disp max"],
        rows,
    )


def _parser_report(payload: dict[str, Any]) -> str:
    parsers = payload["parsers"]
    datasets = parsers["datasets"]
    projected_rows = []
    for entry in datasets:
        current = entry["projected"]["current"]
        projected_rows.append(
            [
                entry["analysis_id"],
                f"{entry['input_bytes'] / 1e6:.1f}",
                str(current["rows"]),
                f"{current['median_seconds']:.2f}",
                *[
                    _speedup(entry["projected"].get(m), current)
                    for m in ("gzip_dc", "pigz_dc")
                ],
            ]
        )
    parse_rows = [
        [
            entry["analysis_id"],
            *[
                _speedup(entry["parse_only"].get(m), entry["projected"]["current"])
                for m in parsers["parse_only_methods"]
            ],
        ]
        for entry in datasets
    ]
    decomp_rows = []
    for entry in datasets:
        measurement = _decompression_entry(parsers, entry["analysis_id"])
        decomp_rows.append(
            [
                entry["analysis_id"],
                *[
                    f"{measurement[m]['input_mib_per_second']:.0f}"
                    for m in ("python_gzip", "gzip_dc", "pigz_dc")
                ],
            ]
        )
    return "\n\n".join(
        [
            f"Sample: {parsers['sample']}; {parsers['repetitions']} repetition(s). "
            f"{parsers['interpretation']}.",
            "**Projected read (MiB/s speedup over current):**\n\n"
            + _md_table(
                ["analysis", "MiB", "rows", "current s", "gzip -dc", "pigz -dc"], projected_rows
            ),
            "**CSV-parse only (upper bound, speedup over the projected read's current time):**\n\n"
            + _md_table(["analysis", *parsers["parse_only_methods"]], parse_rows),
            "**Decompression only (MiB/s):**\n\n"
            + _md_table(["analysis", "python gzip", "gzip -dc", "pigz -dc"], decomp_rows),
        ]
    )


def _decompression_entry(parsers: dict[str, Any], analysis_id: str) -> dict[str, Any]:
    for entry in parsers.get("decompression_only", []):
        if entry["analysis_id"] == analysis_id:
            return entry
    raise KeyError(f"no decompression measurement for {analysis_id}")


def _speedup(method: dict[str, Any] | None, baseline: dict[str, Any]) -> str:
    if method is None:
        return "-"
    ratio = baseline["median_seconds"] / method["median_seconds"]
    return f"{ratio:.2f}x"


def _parity_report(parity: list[dict[str, Any]]) -> str:
    rows = [
        [
            entry["analysis_id"],
            str(entry["projected_rows_compared"]),
            str(entry["external_rows_compared"]),
        ]
        for entry in parity
    ]
    return _md_table(["analysis", "projected rows compared", "external rows compared"], rows)


def _dominant(values: Sequence[str]) -> str:
    if not values:
        return ""
    return max(set(values), key=list(values).count)


if __name__ == "__main__":
    main()
