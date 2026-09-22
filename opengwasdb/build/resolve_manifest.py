"""Manifest-level Analysis resolution with atomic per-Analysis records and resume (issue #208).

Exposes `opengwasdb.build.resolve.resolve_analysis` across a canonical
`analyses.tsv` manifest, checkpointing each Analysis's resolution into an atomic,
versioned JSON record.

Contracts enforced here:
- **One reference load per invocation:** The Ancestry Reference Panel (~1 GB)
  and any reference-AF resources are loaded once in the parent process and
  fork-shared across workers. No genome-scale reference is ever re-read per Analysis.
- **Content-aware resume:** `--resume` skips an existing record if and only if
  its status is `success` and every fingerprint input (source modification/checksum,
  tool version, references, extraction panel, gates, method tiers, and settings)
  matches the current run. Missing, failed, or stale records are rerun.
- **Atomic writes:** Every per-Analysis record and aggregate index is written to
  a unique sibling temporary file and committed via `os.replace` with `fsync`,
  ensuring an interrupted run never leaves a partial or corrupt record that a
  later resume could treat as valid.
- **Deterministic ordering:** Per-Analysis records are named by `analysis_id`.
  The aggregate `index.json` preserves the original manifest order regardless of
  worker count or task completion order.
- **Straggler mitigation:** When `largest_first` is enabled (the default), tasks
  are ordered by on-disk or recorded source size descending, so large files
  start immediately and do not tail-gate the pool.
- **Error isolation:** Ordinary source, parser, or statistical errors are
  isolated to the affected Analysis and recorded as `controlled_failure`.
  Systemic configuration or setup errors fail the command immediately.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import multiprocessing
import os
import subprocess
import time
import tracemalloc
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from opengwasdb.ancestry.mixture import AncestryAssignment, Gates
from opengwasdb.ancestry.reference import AncestryReference, load_reference
from opengwasdb.build.phenotype_sd_pipeline import load_af_reference
from opengwasdb.build.resolve import (
    DEFAULT_EVIDENCE_SAMPLE,
    AfReference,
    AnalysisRequest,
    AnalysisResolution,
    PhenotypeSdResolution,
    ScanDiagnostics,
    ScanLimit,
    resolve_analysis,
)
from opengwasdb.model.enums import OriginalSdMethod, StoredEffectScale
from opengwasdb.model.manifest_columns import resolve_manifest_columns
from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY
from opengwasdb.readers.registry import known_capabilities, resolve_reader

logger = logging.getLogger(__name__)

RECORD_SCHEMA_VERSION = 1
OPENGWASDB_PACKAGE_VERSION = "0.3.0"

__all__ = [
    "ManifestResolutionSummary",
    "RecordStatus",
    "ResolveManifestRow",
    "compute_file_sha256",
    "compute_fingerprint_digest",
    "load_extraction_panel",
    "parse_af_references",
    "read_resolve_manifest",
    "resolve_analyses_manifest",
]


class RecordStatus(StrEnum):
    """Execution status of one Analysis resolution record (issue #208)."""

    SUCCESS = "success"
    CONTROLLED_FAILURE = "controlled_failure"
    SKIP = "skip"


@dataclass(frozen=True)
class ResolveManifestRow:
    """One Analysis row parsed from the input manifest."""

    manifest_index: int
    analysis_id: str
    source_file: str
    source_reader_capability: str
    stored_effect_scale: StoredEffectScale
    original_sd_method: OriginalSdMethod
    sample_size: float | None
    size_bytes: int | None = None
    checksum: str | None = None
    checksum_algorithm: str | None = None


@dataclass(frozen=True)
class ManifestResolutionSummary:
    """Summary of a completed manifest resolution run."""

    records_dir: Path
    n_total: int
    n_success: int
    n_resumed: int
    n_failed: int
    failed_analyses: list[str]
    index_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "records_dir": str(self.records_dir),
            "n_total": self.n_total,
            "n_success": self.n_success,
            "n_resumed": self.n_resumed,
            "n_failed": self.n_failed,
            "failed_analyses": list(self.failed_analyses),
            "index_path": str(self.index_path),
        }


def compute_file_sha256(path: Path | str, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def compute_fingerprint_digest(fp: dict[str, Any]) -> str:
    """Compute a canonical SHA-256 digest over a fingerprint dictionary."""
    clean = {k: v for k, v in fp.items() if k != "fingerprint_digest"}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_panel_header(first_line: str) -> tuple[int, str | None]:
    tokens = first_line.split()
    lower_tokens = [tok.strip().lower() for tok in tokens]
    for target in ("alid", "variant_id", "id"):
        if target in lower_tokens:
            return lower_tokens.index(target), None
    if first_line.strip() and not first_line.startswith("#"):
        return 0, tokens[0].strip()
    return 0, None


def _read_panel_lines(fh: Any, col_idx: int, initial_var: str | None) -> set[str]:
    variants: set[str] = set()
    if initial_var:
        variants.add(initial_var)
    for line in fh:
        line_str = line.strip()
        if not line_str or line_str.startswith("#"):
            continue
        tokens = line_str.split()
        if len(tokens) > col_idx:
            variants.add(tokens[col_idx])
    return variants


def load_extraction_panel(path: Path | str) -> set[str]:
    """Load variant IDs / ALIDs from a panel file for bounded extraction."""
    panel_path = Path(path)
    if not panel_path.is_file():
        raise FileNotFoundError(f"extraction panel file not found: {panel_path}")
    with open(panel_path, encoding="utf-8") as fh:
        first_line = fh.readline()
        if not first_line:
            raise ValueError(f"empty extraction panel file: {panel_path}")
        col_idx, initial_var = _parse_panel_header(first_line)
        variants = _read_panel_lines(fh, col_idx, initial_var)
    if not variants:
        raise ValueError(f"no valid variants found in extraction panel: {panel_path}")
    return variants


def _parse_single_af_spec(spec: str, default_pop: str) -> tuple[str, Path]:
    if "=" in spec:
        pop, path_str = spec.split("=", 1)
    elif ":" in spec and not Path(spec).exists():
        pop, path_str = spec.split(":", 1)
    else:
        pop, path_str = default_pop, spec
    pop_clean = pop.strip().upper()
    ref_path = Path(path_str.strip())
    if not ref_path.exists():
        raise FileNotFoundError(f"AF reference path not found for {pop_clean}: {ref_path}")
    return pop_clean, ref_path


def parse_af_references(
    af_ref_specs: Sequence[str] | None, default_ancestry: str | None = None
) -> dict[str, AfReference]:
    """Parse `--af-reference` specifications and load their frequency mappings."""
    if not af_ref_specs:
        return {}
    res: dict[str, AfReference] = {}
    default_pop = (default_ancestry or "EUR").strip().upper()
    for spec in af_ref_specs:
        spec_clean = spec.strip()
        if not spec_clean:
            continue
        pop, ref_path = _parse_single_af_spec(spec_clean, default_pop)
        freqs = load_af_reference(ref_path, ancestry=pop)
        res[pop] = AfReference(reference_id=ref_path.name, frequencies=freqs)
    return res


def _parse_effect_scale(raw_scale: str | None) -> StoredEffectScale:
    if not raw_scale:
        return StoredEffectScale.SD
    try:
        return StoredEffectScale(raw_scale.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(s.value for s in StoredEffectScale)
        raise ValueError(
            f"invalid stored_effect_scale {raw_scale!r}; expected one of {allowed}"
        ) from exc


def _parse_sd_method(
    raw_method: str | None, scale: StoredEffectScale
) -> OriginalSdMethod:
    if raw_method and raw_method.strip():
        try:
            return OriginalSdMethod(raw_method.strip().lower())
        except ValueError as exc:
            allowed = ", ".join(m.value for m in OriginalSdMethod)
            raise ValueError(
                f"invalid original_sd_method {raw_method!r}; expected one of {allowed}"
            ) from exc
    if scale in (StoredEffectScale.LOG_OR, StoredEffectScale.LOG_HAZARD):
        return OriginalSdMethod.BINARY_TRAIT
    return OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF


def _parse_sample_size_val(raw_n: str | None, analysis_id: str, path: str | Path) -> float | None:
    if raw_n is None or not raw_n.strip():
        return None
    try:
        val = float(raw_n.strip())
        return val if math.isfinite(val) and val > 0 else None
    except ValueError as exc:
        raise ValueError(
            f"analyses manifest {path}: analysis {analysis_id!r} has invalid "
            f"sample_size {raw_n!r}"
        ) from exc


def _parse_capability(
    raw_cap: str | None,
    default_cap: str | None,
    source_file: str,
    known: tuple[str, ...],
    analysis_id: str,
    path: Path,
) -> str:
    cap = (raw_cap or default_cap or "").strip()
    if not cap:
        cap = (
            GWAS_VCF_CAPABILITY
            if source_file.endswith((".vcf", ".vcf.gz", ".bcf"))
            else GWAS_SSF_CAPABILITY
        )
    if cap not in known:
        known_str = ", ".join(known)
        raise ValueError(
            f"analyses manifest {path}: analysis {analysis_id!r} has unknown "
            f"source_reader_capability {cap!r}; known: {known_str}"
        )
    return cap


def _extract_manifest_checksum(raw: dict[str, str]) -> tuple[str | None, str | None]:
    checksum = (raw.get("checksum") or raw.get("sha256") or "").strip() or None
    algo = (raw.get("checksum_algorithm") or "").strip() or None
    return checksum, algo


def _extract_manifest_size(raw: dict[str, str]) -> int | None:
    raw_size = raw.get("size_bytes")
    if raw_size and raw_size.strip():
        try:
            return int(raw_size.strip())
        except ValueError:
            return None
    return None


def _validate_manifest_ids(
    raw: dict[str, str], cols: Any, idx: int, path: Path, seen_ids: set[str]
) -> tuple[str, str]:
    analysis_id = raw[cols.analysis_id].strip()
    if not analysis_id:
        raise ValueError(f"analyses manifest {path}: row {idx + 1} has blank analysis_id")
    if analysis_id in seen_ids:
        raise ValueError(
            f"analyses manifest {path} contains duplicate analysis_id: {analysis_id!r}"
        )
    seen_ids.add(analysis_id)

    source_file = raw[cols.source_file].strip()
    if not source_file:
        raise ValueError(
            f"analyses manifest {path}: analysis {analysis_id!r} has blank source_file"
        )
    return analysis_id, source_file


def _parse_manifest_row(
    raw: dict[str, str],
    idx: int,
    cols: Any,
    default_cap: str | None,
    known: tuple[str, ...],
    path: Path,
    seen_ids: set[str],
) -> ResolveManifestRow:
    analysis_id, source_file = _validate_manifest_ids(raw, cols, idx, path, seen_ids)
    cap = _parse_capability(
        raw.get("source_reader_capability"), default_cap, source_file, known, analysis_id, path
    )
    scale = _parse_effect_scale(raw.get("stored_effect_scale"))
    method = _parse_sd_method(raw.get("original_sd_method"), scale)
    sample_size = _parse_sample_size_val(
        raw.get(cols.sample_size) if cols.sample_size else None, analysis_id, path
    )
    size_bytes = _extract_manifest_size(raw)
    checksum, algo = _extract_manifest_checksum(raw)

    return ResolveManifestRow(
        manifest_index=idx,
        analysis_id=analysis_id,
        source_file=source_file,
        source_reader_capability=cap,
        stored_effect_scale=scale,
        original_sd_method=method,
        sample_size=sample_size,
        size_bytes=size_bytes,
        checksum=checksum,
        checksum_algorithm=algo,
    )


def read_resolve_manifest(
    path: Path | str, default_capability: str | None = None
) -> list[ResolveManifestRow]:
    """Read and validate an `analyses.tsv` manifest into resolution rows."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"analyses manifest not found: {manifest_path}")
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"empty analyses manifest: {manifest_path}")

    cols = resolve_manifest_columns(fieldnames, manifest_path)
    known = known_capabilities()
    seen_ids: set[str] = set()
    return [
        _parse_manifest_row(raw, idx, cols, default_capability, known, manifest_path, seen_ids)
        for idx, raw in enumerate(raw_rows)
    ]


def _get_opengwasdb_version() -> str:
    try:
        from importlib.metadata import version

        return version("opengwasdb")
    except Exception:
        return OPENGWASDB_PACKAGE_VERSION


def _get_git_hash() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).parent,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return ""


def _optional_float(val: float | None) -> float | None:
    if val is None or not math.isfinite(val):
        return None
    return float(val)


def _ancestry_to_dict(a: AncestryAssignment | None) -> dict[str, Any] | None:
    if a is None:
        return None
    return {
        "assigned_ancestry": a.assigned_ancestry,
        "dominant_superpop": a.dominant_superpop,
        "dominant_proportion": _optional_float(a.dominant_proportion),
        "runner_up_margin": _optional_float(a.runner_up_margin),
        "af_overlap": a.af_overlap,
        "residual": _optional_float(a.residual),
        "gate_reason": a.gate_reason,
        "eaf_orientation": a.eaf_orientation,
        "eaf_orientation_r": _optional_float(a.eaf_orientation_r),
        "superpop_composition": dict(a.superpop_composition),
        "fine_composition": dict(a.fine_composition),
    }


def _phenotype_sd_to_dict(p: PhenotypeSdResolution | None) -> dict[str, Any] | None:
    if p is None:
        return None
    est_dict = None
    if p.estimate is not None:
        est_dict = {
            "sd": _optional_float(p.estimate.sd),
            "dispersion": _optional_float(p.estimate.dispersion),
            "method": p.estimate.method.value,
            "notes": p.estimate.notes or "",
        }
    return {
        "status": p.status.value,
        "reason": p.reason.value if p.reason is not None else None,
        "estimate": est_dict,
        "reference_id": p.reference_id,
        "n_evidence_considered": p.n_evidence_considered,
        "n_estimate_inputs": p.n_estimate_inputs,
        "evidence_sampled": p.evidence_sampled,
    }


def _diagnostics_to_dict(d: ScanDiagnostics) -> dict[str, Any]:
    return {
        "source_file": d.source_file,
        "rows_read": d.rows_read,
        "ancestry_sites": d.ancestry_sites,
        # Issue #209: whether the scan ended at EOF or at a bound. A record
        # written under a future scan limit must not be readable as a full scan.
        "stop_reason": d.stop_reason.value,
    }


def _stat_source_file(source_file: str) -> tuple[int | None, int | None]:
    source_path = Path(source_file)
    if source_path.is_file():
        try:
            st = source_path.stat()
            return st.st_size, st.st_mtime_ns
        except OSError:
            pass
    return None, None


def _build_analysis_fingerprints(
    row: ResolveManifestRow,
    *,
    opengwasdb_version: str,
    opengwasdb_git_hash: str,
    ancestry_reference_id: str,
    ancestry_reference_sha256: str,
    ancestry_groups_sha256: str,
    extraction_panel_sha256: str | None,
    extraction_panel_variants: int | None,
    af_references_fp: list[dict[str, Any]],
    gates: Gates,
    maf_floor: float,
    evidence_sample: int,
    scan_limit: ScanLimit | None,
) -> dict[str, Any]:
    file_bytes, file_mtime_ns = _stat_source_file(row.source_file)
    fp = {
        "source_file": str(row.source_file),
        "source_recorded_sha256": row.checksum,
        "source_recorded_bytes": row.size_bytes,
        "source_file_bytes": file_bytes,
        "source_file_mtime_ns": file_mtime_ns,
        "opengwasdb_version": opengwasdb_version,
        "opengwasdb_git_hash": opengwasdb_git_hash,
        "ancestry_reference_id": ancestry_reference_id,
        "ancestry_reference_sha256": ancestry_reference_sha256,
        "ancestry_groups_sha256": ancestry_groups_sha256,
        "extraction_panel_sha256": extraction_panel_sha256,
        "extraction_panel_variants": extraction_panel_variants,
        "af_references": af_references_fp,
        "resolution_config": {
            "original_sd_method": row.original_sd_method.value,
            "stored_effect_scale": row.stored_effect_scale.value,
            "sample_size": row.sample_size,
            "source_reader_capability": row.source_reader_capability,
            "maf_floor": maf_floor,
            "evidence_sample": evidence_sample,
            # Issue #209: the scan bound travels in the fingerprint, so a record
            # resolved under one bound can never be resumed as another. The
            # version is always present so a change to what a bound *means*
            # invalidates too.
            "scan_limit": None if scan_limit is None else scan_limit.as_fingerprint(),
            "gates": {
                "tau": gates.tau,
                "delta": gates.delta,
                "n_min": gates.n_min,
                "residual_max": gates.residual_max,
                "orientation_flip_r": gates.orientation_flip_r,
                "sum_to_one_penalty": gates.sum_to_one_penalty,
            },
        },
    }
    fp["fingerprint_digest"] = compute_fingerprint_digest(fp)
    return fp


def _is_record_resumable(record_path: Path, expected_digest: str) -> bool:
    """Check if an existing record file matches the expected fingerprint digest."""
    if not record_path.is_file():
        return False
    try:
        with open(record_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("status") != RecordStatus.SUCCESS.value:
            return False
        fps = data.get("fingerprints", {})
        return bool(fps.get("fingerprint_digest") == expected_digest)
    except Exception:
        return False


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write data to JSON via a sibling temp file with fsync and rename."""
    tmp_path = path.parent / f".tmp_{path.name}_{uuid4().hex}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


# Fork-shared module globals inherited by worker processes.
_WORKER_ANCESTRY_REFERENCE: AncestryReference | None = None
_WORKER_GATES: Gates | None = None
_WORKER_EXTRACTION_PANEL: Collection[str] | None = None
_WORKER_AF_REFERENCES: Mapping[str, AfReference] | None = None
_WORKER_EVIDENCE_SAMPLE: int = DEFAULT_EVIDENCE_SAMPLE
_WORKER_SCAN_LIMIT: ScanLimit | None = None
_WORKER_RECORDS_DIR: Path | None = None
_REFERENCE_LOAD_CALLS: int = 0


def _execute_analysis(
    analysis_id: str,
    source_file: str,
    cap: str,
    scale: StoredEffectScale,
    method: OriginalSdMethod,
    sample_size: float | None,
) -> tuple[AnalysisResolution, RecordStatus, str | None]:
    assert _WORKER_ANCESTRY_REFERENCE is not None
    assert _WORKER_GATES is not None
    req = AnalysisRequest(
        analysis_id=analysis_id,
        source_file=source_file,
        sample_size=sample_size,
        original_sd_method=method,
        stored_effect_scale=scale,
    )
    try:
        reader = resolve_reader(cap, source_file, scale)
        res = resolve_analysis(
            req,
            reader=reader,
            reference=_WORKER_ANCESTRY_REFERENCE,
            extraction_panel=_WORKER_EXTRACTION_PANEL,
            gates=_WORKER_GATES,
            af_references=_WORKER_AF_REFERENCES,
            evidence_sample=_WORKER_EVIDENCE_SAMPLE,
            scan_limit=_WORKER_SCAN_LIMIT,
        )
        if res.error:
            return res, RecordStatus.CONTROLLED_FAILURE, res.error
        return res, RecordStatus.SUCCESS, None
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        res = AnalysisResolution(
            analysis_id=analysis_id,
            diagnostics=ScanDiagnostics(source_file=source_file, rows_read=0, ancestry_sites=0),
            error=err_msg,
        )
        return res, RecordStatus.CONTROLLED_FAILURE, err_msg


def _extract_worker_summary(
    res: AnalysisResolution,
    status: RecordStatus,
    method: OriginalSdMethod,
    err_msg: str | None,
    task: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    assigned_pop = res.ancestry.assigned_ancestry if res.ancestry else None
    gate_reason = res.ancestry.gate_reason if res.ancestry else None
    sd_val = (
        res.phenotype_sd.estimate.sd
        if (res.phenotype_sd and res.phenotype_sd.estimate)
        else None
    )
    return {
        "manifest_index": task["manifest_index"],
        "analysis_id": task["analysis_id"],
        "status": status.value,
        "assigned_ancestry": assigned_pop,
        "gate_reason": gate_reason,
        "original_sd": _optional_float(sd_val),
        "original_sd_method": method.value,
        "error": err_msg or (res.error if res.error else None),
        "elapsed_seconds": elapsed,
    }


def _worker_resolve_one(task: dict[str, Any]) -> dict[str, Any]:
    """Worker task: resolve one Analysis and write its atomic JSON record."""
    analysis_id = task["analysis_id"]
    source_file = task["source_file"]
    cap = task["source_reader_capability"]
    scale = StoredEffectScale(task["stored_effect_scale"])
    method = OriginalSdMethod(task["original_sd_method"])
    sample_size = task["sample_size"]
    fingerprints = task["fingerprints"]
    record_path = Path(task["record_path"])

    tracemalloc.start()
    t0 = time.monotonic()
    res, status, err_msg = _execute_analysis(
        analysis_id, source_file, cap, scale, method, sample_size
    )
    elapsed = time.monotonic() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    record_data = {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "status": status.value,
        "fingerprints": fingerprints,
        "diagnostics": _diagnostics_to_dict(res.diagnostics),
        "ancestry": _ancestry_to_dict(res.ancestry),
        "phenotype_sd": _phenotype_sd_to_dict(res.phenotype_sd),
        "error": err_msg or (res.error if res.error else None),
        "warnings": [],
        "metrics": {"elapsed_seconds": round(elapsed, 6), "peak_memory_bytes": peak},
    }
    _atomic_write_json(record_path, record_data)
    return _extract_worker_summary(res, status, method, err_msg, task, elapsed)


def _load_af_refs_and_fingerprints(
    af_references: Sequence[str] | None, af_reference_ancestry: str | None
) -> tuple[dict[str, AfReference], list[dict[str, Any]]]:
    af_refs = parse_af_references(af_references, default_ancestry=af_reference_ancestry)
    af_refs_fp: list[dict[str, Any]] = []
    if af_references:
        for spec in af_references:
            if not spec.strip():
                continue
            pop = (
                spec.split("=", 1)[0].strip().upper()
                if "=" in spec
                else (af_reference_ancestry or "EUR").strip().upper()
            )
            p_str = spec.split("=", 1)[1].strip() if "=" in spec else spec.strip()
            p_path = Path(p_str)
            if p_path.is_file():
                af_refs_fp.append(
                    {
                        "ancestry": pop,
                        "reference_id": p_path.name,
                        "sha256": compute_file_sha256(p_path),
                    }
                )
    return af_refs, af_refs_fp


def _load_references_for_resolution(
    ancestry_reference: Path | str,
    ancestry_groups: Path | str,
    extraction_panel: Path | str | None,
    af_references: Sequence[str] | None,
    af_reference_ancestry: str | None,
    maf_floor: float,
) -> tuple[
    AncestryReference,
    str,
    str,
    set[str] | None,
    str | None,
    int | None,
    dict[str, AfReference],
    list[dict[str, Any]],
]:
    global _REFERENCE_LOAD_CALLS
    anc_ref_path = Path(ancestry_reference)
    anc_grp_path = Path(ancestry_groups)
    if not anc_ref_path.is_file():
        raise FileNotFoundError(f"ancestry reference not found: {anc_ref_path}")
    if not anc_grp_path.is_file():
        raise FileNotFoundError(f"ancestry groups file not found: {anc_grp_path}")

    _REFERENCE_LOAD_CALLS += 1
    reference = load_reference(anc_ref_path, anc_grp_path, maf_floor=maf_floor)
    anc_ref_sha = compute_file_sha256(anc_ref_path)
    anc_grp_sha = compute_file_sha256(anc_grp_path)

    panel_set: set[str] | None = None
    panel_sha: str | None = None
    panel_vars: int | None = None
    if extraction_panel is not None:
        panel_set = load_extraction_panel(extraction_panel)
        panel_sha = compute_file_sha256(extraction_panel)
        panel_vars = len(panel_set)

    af_refs, af_refs_fp = _load_af_refs_and_fingerprints(af_references, af_reference_ancestry)
    return (
        reference,
        anc_ref_sha,
        anc_grp_sha,
        panel_set,
        panel_sha,
        panel_vars,
        af_refs,
        af_refs_fp,
    )


def _load_resumed_summary(rec_path: Path, row: ResolveManifestRow) -> dict[str, Any] | None:
    try:
        with open(rec_path, encoding="utf-8") as fh:
            rec_json = json.load(fh)
        anc_obj = rec_json.get("ancestry") or {}
        sd_obj = rec_json.get("phenotype_sd") or {}
        est_obj = sd_obj.get("estimate") or {}
        return {
            "manifest_index": row.manifest_index,
            "analysis_id": row.analysis_id,
            "status": RecordStatus.SUCCESS.value,
            "assigned_ancestry": anc_obj.get("assigned_ancestry"),
            "gate_reason": anc_obj.get("gate_reason"),
            "original_sd": _optional_float(est_obj.get("sd")),
            "original_sd_method": row.original_sd_method.value,
            "error": rec_json.get("error"),
            "elapsed_seconds": rec_json.get("metrics", {}).get("elapsed_seconds", 0.0),
        }
    except Exception:
        return None


def _build_single_task(
    row: ResolveManifestRow, fp: dict[str, Any], rec_path: Path
) -> dict[str, Any]:
    size_weight = row.size_bytes or 0
    if size_weight <= 0:
        try:
            size_weight = Path(row.source_file).stat().st_size
        except OSError:
            size_weight = 0
    return {
        "manifest_index": row.manifest_index,
        "analysis_id": row.analysis_id,
        "source_file": row.source_file,
        "source_reader_capability": row.source_reader_capability,
        "stored_effect_scale": row.stored_effect_scale.value,
        "original_sd_method": row.original_sd_method.value,
        "sample_size": row.sample_size,
        "fingerprints": fp,
        "record_path": str(rec_path),
        "size_weight": size_weight,
    }


def _prepare_tasks(
    rows: list[ResolveManifestRow],
    out_dir: Path,
    resume: bool,
    fp_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    resumed: dict[str, dict[str, Any]] = {}
    for row in rows:
        fp = _build_analysis_fingerprints(row, **fp_kwargs)
        rec_path = out_dir / f"{row.analysis_id}.json"
        if resume and _is_record_resumable(rec_path, fp["fingerprint_digest"]):
            summary = _load_resumed_summary(rec_path, row)
            if summary is not None:
                resumed[row.analysis_id] = summary
                continue
        tasks.append(_build_single_task(row, fp, rec_path))
    return tasks, resumed


def _run_tasks(tasks: list[dict[str, Any]], n_workers: int) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not tasks:
        return results
    if n_workers <= 1:
        for task in tasks:
            res_dict = _worker_resolve_one(task)
            results[res_dict["analysis_id"]] = res_dict
        return results

    fork_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=fork_ctx) as pool:
        futures = [pool.submit(_worker_resolve_one, task) for task in tasks]
        for future in as_completed(futures):
            res_dict = future.result()
            results[res_dict["analysis_id"]] = res_dict
    return results


def _build_final_entry(row: ResolveManifestRow, entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_index": row.manifest_index,
        "analysis_id": row.analysis_id,
        "record_file": f"{row.analysis_id}.json",
        "status": entry["status"],
        "assigned_ancestry": entry.get("assigned_ancestry"),
        "gate_reason": entry.get("gate_reason"),
        "original_sd": entry.get("original_sd"),
        "original_sd_method": entry.get("original_sd_method"),
        "error": entry.get("error"),
    }


def _assemble_index(
    rows: list[ResolveManifestRow],
    resumed: dict[str, dict[str, Any]],
    executed: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int, list[str]]:
    all_summaries: list[dict[str, Any]] = []
    n_success = 0
    n_failed = 0
    failed_ids: list[str] = []
    for row in rows:
        aid = row.analysis_id
        entry = resumed.get(aid) or executed.get(aid)
        if entry is None:
            entry = {
                "manifest_index": row.manifest_index,
                "analysis_id": aid,
                "status": RecordStatus.CONTROLLED_FAILURE.value,
                "assigned_ancestry": None,
                "gate_reason": None,
                "original_sd": None,
                "original_sd_method": row.original_sd_method.value,
                "error": "Task not executed",
            }
        if entry["status"] == RecordStatus.SUCCESS.value:
            n_success += 1
        else:
            n_failed += 1
            failed_ids.append(aid)
        all_summaries.append(_build_final_entry(row, entry))
    return all_summaries, n_success, n_failed, failed_ids


def _validate_resolution_params(
    evidence_sample: int, n_workers: int, records_dir: Path | str
) -> Path:
    if evidence_sample <= 0:
        raise ValueError(f"evidence_sample must be positive, got {evidence_sample!r}")
    if n_workers <= 0:
        raise ValueError(f"n_workers must be positive, got {n_workers!r}")
    out_dir = Path(records_dir)
    if out_dir.is_file():
        raise ValueError(f"records_dir {out_dir} is an existing file, not a directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _setup_worker_globals(
    ref: AncestryReference,
    gates: Gates,
    panel_set: set[str] | None,
    af_refs: dict[str, AfReference],
    evidence_sample: int,
    scan_limit: ScanLimit | None,
    out_dir: Path,
) -> None:
    global _WORKER_ANCESTRY_REFERENCE, _WORKER_GATES, _WORKER_EXTRACTION_PANEL
    global _WORKER_AF_REFERENCES, _WORKER_EVIDENCE_SAMPLE, _WORKER_RECORDS_DIR
    global _WORKER_SCAN_LIMIT
    _WORKER_ANCESTRY_REFERENCE = ref
    _WORKER_GATES = gates
    _WORKER_EXTRACTION_PANEL = panel_set
    _WORKER_AF_REFERENCES = af_refs
    _WORKER_EVIDENCE_SAMPLE = evidence_sample
    _WORKER_SCAN_LIMIT = scan_limit
    _WORKER_RECORDS_DIR = out_dir


def _prepare_pipeline_context(
    ancestry_reference: Path | str,
    ancestry_groups: Path | str,
    extraction_panel: Path | str | None,
    af_references: Sequence[str] | None,
    af_reference_ancestry: str | None,
    maf_floor: float,
    tau: float,
    delta: float,
    n_min: int,
    residual_max: float,
    orientation_flip_r: float,
    evidence_sample: int,
    scan_limit: ScanLimit | None,
    reference_version: str,
    out_dir: Path,
) -> dict[str, Any]:
    (
        ref,
        ref_sha,
        grp_sha,
        panel_set,
        panel_sha,
        panel_vars,
        af_refs,
        af_refs_fp,
    ) = _load_references_for_resolution(
        ancestry_reference,
        ancestry_groups,
        extraction_panel,
        af_references,
        af_reference_ancestry,
        maf_floor,
    )
    gates = Gates(
        tau=tau,
        delta=delta,
        n_min=n_min,
        residual_max=residual_max,
        orientation_flip_r=orientation_flip_r,
    )
    _setup_worker_globals(ref, gates, panel_set, af_refs, evidence_sample, scan_limit, out_dir)
    return {
        "opengwasdb_version": _get_opengwasdb_version(),
        "opengwasdb_git_hash": _get_git_hash(),
        "ancestry_reference_id": reference_version or Path(ancestry_reference).name,
        "ancestry_reference_sha256": ref_sha,
        "ancestry_groups_sha256": grp_sha,
        "extraction_panel_sha256": panel_sha,
        "extraction_panel_variants": panel_vars,
        "af_references_fp": af_refs_fp,
        "gates": gates,
        "maf_floor": maf_floor,
        "evidence_sample": evidence_sample,
        "scan_limit": scan_limit,
    }


def _execute_resolution_pipeline(
    rows: list[ResolveManifestRow],
    out_dir: Path,
    manifest_path: Path | str,
    fp_kwargs: dict[str, Any],
    resume: bool,
    largest_first: bool,
    n_workers: int,
) -> ManifestResolutionSummary:
    tasks, resumed = _prepare_tasks(rows, out_dir, resume, fp_kwargs)
    if largest_first:
        tasks.sort(key=lambda t: (-t["size_weight"], t["manifest_index"]))

    executed = _run_tasks(tasks, n_workers)
    all_summaries, n_success, n_failed, failed_ids = _assemble_index(rows, resumed, executed)

    index_data = {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "records_dir": str(out_dir),
        "opengwasdb_version": fp_kwargs["opengwasdb_version"],
        "opengwasdb_git_hash": fp_kwargs["opengwasdb_git_hash"],
        "n_total": len(rows),
        "n_success": n_success,
        "n_resumed": len(resumed),
        "n_failed": n_failed,
        "failed_analyses": failed_ids,
        "analyses": all_summaries,
    }
    index_path = out_dir / "index.json"
    _atomic_write_json(index_path, index_data)

    return ManifestResolutionSummary(
        records_dir=out_dir,
        n_total=len(rows),
        n_success=n_success,
        n_resumed=len(resumed),
        n_failed=n_failed,
        failed_analyses=failed_ids,
        index_path=index_path,
    )


def resolve_analyses_manifest(
    manifest_path: Path | str,
    records_dir: Path | str,
    *,
    ancestry_reference: Path | str,
    ancestry_groups: Path | str,
    extraction_panel: Path | str | None = None,
    af_references: Sequence[str] | None = None,
    af_reference_ancestry: str | None = None,
    default_source_reader_capability: str | None = None,
    maf_floor: float = 0.01,
    tau: float = 0.50,
    delta: float = 0.20,
    n_min: int = 5_000,
    residual_max: float = 0.06,
    orientation_flip_r: float = -0.5,
    evidence_sample: int = DEFAULT_EVIDENCE_SAMPLE,
    max_ancestry_sites: int | None = None,
    max_rows: int | None = None,
    n_workers: int = 1,
    resume: bool = False,
    largest_first: bool = True,
    reference_version: str = "",
) -> ManifestResolutionSummary:
    """Resolve an entire manifest with atomic checkpointed records and resume.

    `max_ancestry_sites` and `max_rows` bound each Analysis's source scan (issue
    #209): the scan stops once the ancestry fit holds that many distinct usable
    reference sites, or once that many source rows have been read. Both are
    `None` by default, which reads the whole source. A bounded resolution is
    recorded with its `stop_reason` and bound in the fingerprint, so a record
    produced under one bound can never be resumed as another.
    """
    scan_limit = None
    if max_ancestry_sites is not None or max_rows is not None:
        scan_limit = ScanLimit(max_rows=max_rows, max_ancestry_sites=max_ancestry_sites)
        scan_limit.validate()
    out_dir = _validate_resolution_params(evidence_sample, n_workers, records_dir)
    rows = read_resolve_manifest(manifest_path, default_capability=default_source_reader_capability)
    fp_kwargs = _prepare_pipeline_context(
        ancestry_reference,
        ancestry_groups,
        extraction_panel,
        af_references,
        af_reference_ancestry,
        maf_floor,
        tau,
        delta,
        n_min,
        residual_max,
        orientation_flip_r,
        evidence_sample,
        scan_limit,
        reference_version,
        out_dir,
    )
    return _execute_resolution_pipeline(
        rows, out_dir, manifest_path, fp_kwargs, resume, largest_first, n_workers
    )
