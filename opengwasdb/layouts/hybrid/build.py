"""Integrated Hybrid build — one read per study, routing on-panel → dense fill,
off-panel → ragged overflow (ADR 0026, PRD "Generation").

This is a **thin integration layer** (PRD "Component reuse"): it composes the
dense two-pass builder's liftover, key-lookup, disk-spill, band-write and
fork-pool machinery (``opengwasdb.layouts.dense.build_vcf``) and the ragged
``RaggedCSRWriter`` — the only new logic is per-variant on-panel/off-panel
routing in the single Pass 2 read that both components share.

Like the dense builder, association streaming and the union-variant pass go
through a ``SourceReader`` resolved from each row's ``source_reader_capability``
(issue #20) rather than importing ``opengwasdb.build.vcf_source`` directly.

``build_hybrid_from_vcf_manifest`` is a thin orchestrator over the deep phase
helpers below (issue #130) - lifting (the shared Pass 1), partition/routing,
EAF verification, joint encoding, component writes and shared metadata -
so no single function carries the whole build's branching. Each phase
preserves the contracts its own code enforces (atomicity of the staging
context, collision/provenance rules, the disjoint-partition layout).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.build.eaf_orientation import (
    EafOrientationReport,
    apply_orientation_evidence,
    site_hashes,
    verify_eaf_orientation,
)
from opengwasdb.build.liftover import LiftoverFailureError
from opengwasdb.encoding import (
    EncodingMeasurements,
    StoreEncoding,
    combine_eaf_measurements,
    optimise_dense_se_joint,
)
from opengwasdb.layouts.dense.build import add_hit_counts, write_analyses_tsv
from opengwasdb.layouts.dense.build_vcf import (
    _RESOLVE_BATCH,
    EafSpillSurvey,
    _alid_sort_key,
    _apply_eaf_scope,
    _apply_se_divisor,
    _create_dense_zarr,
    _encode_variant_keys,
    _fork_pool,
    _lift_manifest_variants,
    _log_progress,
    _manifest_row_to_analysis,
    _ManifestRow,
    _read_manifest,
    _write_dense_bands,
    _write_index,
    survey_eaf_spills,
)
from opengwasdb.layouts.dense.constants import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_COMPRESSOR,
    DEFAULT_DTYPE,
)
from opengwasdb.layouts.dense.top_hits import write_top_hit_indexes_for_store
from opengwasdb.layouts.hybrid.layout import (
    DENSE_SUBDIR,
    dense_component_path,
    dense_to_shared_path,
)
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter
from opengwasdb.model.analyses import Analysis
from opengwasdb.model.enums import (
    AssociationCoverage,
    CompletionState,
    PrimaryStorageLayout,
    StoredEffectScale,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY
from opengwasdb.readers.registry import resolve_reader
from opengwasdb.store.open import CURRENT_FORMAT_VERSION, OpenGWASDBStore, StagedRelease
from opengwasdb.variants import CanonicalVariant, write_variant_axis

log = logging.getLogger(__name__)

__all__ = [
    "build_hybrid_from_vcf_manifest",
    "HybridBuildResult",
    "read_reference_panel_alids",
    "LiftoverFailureError",
]


@dataclass(frozen=True)
class HybridBuildResult:
    output_path: Path
    n_variants: int  # shared union table
    n_analyses: int
    n_panel: int  # Dense Component rows
    n_off_panel: int  # off-panel observed variants (overflow variant axis)
    n_overflow: int  # overflow associations


# ── Reference panel input ────────────────────────────────────────────────────


def read_reference_panel_alids(panel_path: str | Path) -> set[str]:
    """Read the reference-panel variant set as canonical hg38 ALIDs.

    Accepts either a plain-text file of one ``chr:pos:a1:a2`` ALID per line, or a
    Store Variant Table (``*.tsv.gz`` with an ``alid`` column). The panel defines
    the Dense Component axis (the imputable set).
    """
    path = Path(panel_path)
    name = str(path).lower()
    alids: set[str] = set()
    if name.endswith((".tsv.gz", ".tsv.bgz", ".bgz")):
        import pysam

        with pysam.BGZFile(str(path), "r") as handle:  # type: ignore[call-arg]
            header: list[str] | None = None
            for raw in handle:
                line = raw.decode("utf-8").rstrip("\n")
                if line.startswith("#"):
                    header = line.lstrip("#").split("\t")
                    continue
                fields = line.split("\t")
                if header is not None and "alid" in header:
                    alids.add(fields[header.index("alid")])
                elif len(fields) >= 6:
                    alids.add(fields[5])
        return alids
    opener = open
    if name.endswith(".gz"):
        import gzip

        opener = gzip.open  # type: ignore[assignment]
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            token = line.strip().split()[0] if line.strip() else ""
            if token and not token.startswith("#") and token.count(":") == 3:
                alids.add(token)
    return alids


# ── Pass 2 fork-inherited routing lookup (see dense.build_vcf for the rationale) ─
_pass2_keys_sorted: np.ndarray | None = None
_pass2_targets_sorted: np.ndarray | None = None  # int64: dense row (panel) or shared idx (off)
_pass2_ispanel_sorted: np.ndarray | None = None  # bool: True = on-panel target
_pass2_spill_dir: Path | None = None


def _build_routing_index(
    source_lookup: dict[tuple[str, int, str, str], str],
    dense_row: dict[str, int],
    shared_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose the liftover + partition into a single fork-safe sorted lookup.

    Returns ``(keys_sorted, targets_sorted, ispanel_sorted)``: for every source
    variant that resolves to a stored variant, its byte key, the target index
    (dense row when on-panel, shared variant_index when off-panel), and whether
    it is on-panel. Workers binary-search this once per association.
    """
    chroms: list[str] = []
    poss: list[int] = []
    refs: list[str] = []
    alts: list[str] = []
    targets: list[int] = []
    ispanel: list[bool] = []
    for (chrom, pos, ref, alt), alid in source_lookup.items():
        row = dense_row.get(alid)
        if row is not None:
            targets.append(row)
            ispanel.append(True)
        else:
            sidx = shared_index.get(alid)
            if sidx is None:
                continue
            targets.append(sidx)
            ispanel.append(False)
        chroms.append(chrom)
        poss.append(pos)
        refs.append(ref)
        alts.append(alt)
    keys = _encode_variant_keys(chroms, poss, refs, alts)
    targets_arr = np.array(targets, dtype=np.int64)
    ispanel_arr = np.array(ispanel, dtype=bool)
    order = np.argsort(keys, kind="stable")
    return keys[order], targets_arr[order], ispanel_arr[order]


def _dedup_last_wins(
    target: np.ndarray, z: np.ndarray, se: np.ndarray, eaf: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep the last stream occurrence per target index (matches dense semantics)."""
    if len(target) == 0:
        return target, z, se, eaf
    _, first_in_rev = np.unique(target[::-1], return_index=True)
    keep = np.sort(len(target) - 1 - first_in_rev)
    return target[keep], z[keep], se[keep], eaf[keep]


def _resolve_column_hybrid(
    file_path: str,
    keys_sorted: np.ndarray,
    targets_sorted: np.ndarray,
    ispanel_sorted: np.ndarray,
    se_divisor: float = 1.0,
    *,
    capability: str = GWAS_VCF_CAPABILITY,
    stored_effect_scale: str = StoredEffectScale.SD.value,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Stream one study once, routing each association to the dense fill (on-panel)
    or the ragged overflow (off-panel). Returns ``(dense, overflow)`` where each is
    ``(index int64, z f32, se f32, eaf f32)`` deduped last-wins by index; `eaf` is
    NaN where the source reports no frequency (ADR 0036).

    ``capability`` resolves a ``SourceReader`` (issue #20) rather than this
    module streaming a VCF itself; ``stored_effect_scale`` is required to
    construct one (see ``dense.build_vcf._resolve_column``).

    ``se_divisor`` divides every returned ``se`` value, dense and overflow alike
    (continuous-trait phenotype-SD standardisation, issue #18): a study's SD
    rescaling applies uniformly regardless of which component an association
    routes to. Defaults to 1.0 (no-op).
    """
    reader = resolve_reader(capability, file_path, StoredEffectScale(stored_effect_scale))
    d_idx: list[np.ndarray] = []
    d_z: list[np.ndarray] = []
    d_se: list[np.ndarray] = []
    d_eaf: list[np.ndarray] = []
    o_idx: list[np.ndarray] = []
    o_z: list[np.ndarray] = []
    o_se: list[np.ndarray] = []
    o_eaf: list[np.ndarray] = []

    chroms: list[str] = []
    poss: list[int] = []
    refs: list[str] = []
    alts: list[str] = []
    zs: list[float] = []
    ses: list[float] = []
    eafs: list[float] = []

    def _flush() -> None:
        if not zs:
            return
        if len(keys_sorted) == 0:
            for lst in (chroms, poss, refs, alts, zs, ses, eafs):
                lst.clear()
            return
        query = _encode_variant_keys(chroms, poss, refs, alts)
        idx = np.searchsorted(keys_sorted, query)
        idx_clip = np.minimum(idx, len(keys_sorted) - 1)
        matched = keys_sorted[idx_clip] == query
        tgt = targets_sorted[idx_clip[matched]]
        panel = ispanel_sorted[idx_clip[matched]]
        z_arr = np.array(zs, dtype=np.float32)[matched]
        se_arr = np.array(ses, dtype=np.float32)[matched]
        eaf_arr = np.array(eafs, dtype=np.float32)[matched]
        if panel.any():
            d_idx.append(tgt[panel])
            d_z.append(z_arr[panel])
            d_se.append(se_arr[panel])
            d_eaf.append(eaf_arr[panel])
        off = ~panel
        if off.any():
            o_idx.append(tgt[off])
            o_z.append(z_arr[off])
            o_se.append(se_arr[off])
            o_eaf.append(eaf_arr[off])
        for lst in (chroms, poss, refs, alts, zs, ses, eafs):
            lst.clear()

    for assoc in reader.stream_associations():
        chroms.append(assoc.chromosome)
        poss.append(assoc.position)
        refs.append(assoc.ref)
        alts.append(assoc.alt)
        zs.append(assoc.z)
        ses.append(assoc.se)
        eafs.append(float("nan") if assoc.eaf is None else assoc.eaf)
        if len(zs) >= _RESOLVE_BATCH:
            _flush()
    _flush()

    def _assemble(
        parts_i: list[np.ndarray],
        parts_z: list[np.ndarray],
        parts_se: list[np.ndarray],
        parts_eaf: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not parts_i:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
        idx, z, se, eaf = _dedup_last_wins(
            np.concatenate(parts_i),
            np.concatenate(parts_z),
            np.concatenate(parts_se),
            np.concatenate(parts_eaf),
        )
        return idx, z, _apply_se_divisor(se, se_divisor), eaf

    return (
        _assemble(d_idx, d_z, d_se, d_eaf),
        _assemble(o_idx, o_z, o_se, o_eaf),
    )


def _spill_hybrid_column(
    spill_dir: Path,
    col_idx: int,
    dense: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    overflow: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Spill one resolved study column: dense rows to ``{col}.npz`` (the layout the
    dense band-writer consumes) and overflow to ``{col}.ovf.npz``."""
    d_rows, d_z, d_se, d_eaf = dense
    o_idx, o_z, o_se, o_eaf = overflow
    for suffix, arrs in (
        ("", {"rows": d_rows, "z": d_z, "se": d_se, "eaf": d_eaf}),
        (".ovf", {"variant_index": o_idx, "z": o_z, "se": o_se, "eaf": o_eaf}),
    ):
        final = spill_dir / f"{col_idx}{suffix}.npz"
        tmp = spill_dir / f"{col_idx}{suffix}.tmp.npz"
        np.savez(tmp, **arrs)
        tmp.replace(final)


def _pass2_worker(task: tuple[int, str, float, str, str]) -> int:
    assert _pass2_keys_sorted is not None
    assert _pass2_targets_sorted is not None
    assert _pass2_ispanel_sorted is not None
    assert _pass2_spill_dir is not None
    col_idx, file_path, se_divisor, capability, stored_effect_scale = task
    dense, overflow = _resolve_column_hybrid(
        file_path,
        _pass2_keys_sorted,
        _pass2_targets_sorted,
        _pass2_ispanel_sorted,
        se_divisor,
        capability=capability,
        stored_effect_scale=stored_effect_scale,
    )
    _spill_hybrid_column(_pass2_spill_dir, col_idx, dense, overflow)
    return col_idx


def _assemble_overflow_csr(
    spill_dir: Path, n_analyses: int, n_variants: int
) -> tuple[RaggedCSRWriter, np.ndarray]:
    """Assemble the overflow CSR from per-column ``.ovf.npz`` spills, in analysis
    order (so CSR offsets align with analysis_index).

    Also returns a per-column bool array saying which Analyses carried an EAF
    into the *overflow* component. An Analysis can have EAF off-panel and none
    on it, so `eaf_scope` is the union of this and the Dense Component's own
    answer, never either alone (ADR 0036).
    """
    csr = RaggedCSRWriter(n_variants)
    column_has_eaf = np.zeros(n_analyses, dtype=bool)
    for col in range(n_analyses):
        path = spill_dir / f"{col}.ovf.npz"
        if not path.exists():
            csr.add_analysis(
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float16),
            )
            continue
        with np.load(path) as data:
            vi = data["variant_index"].astype(np.int32)
            z = data["z"].astype(np.float32)
            se = data["se"].astype(np.float32)
            eaf = data["eaf"].astype(np.float32)
        # Sort by variant_index for consistent within-analysis ordering (matches
        # the ragged BESD builder and lets top-hit CSR cross-validation searchsort).
        order = np.argsort(vi, kind="stable")
        has_eaf = bool(np.isfinite(eaf).any())
        column_has_eaf[col] = has_eaf
        csr.add_analysis(vi[order], z[order], se[order], eaf=eaf[order] if has_eaf else None)
        path.unlink()
    return csr, column_has_eaf


# ── Deep-phase helpers for the build entry point (issue #130) ────────────────


@dataclass(frozen=True)
class _BuildOptions:
    """The public build's scalar configuration, bundled so the seams below
    take one argument rather than a dozen."""

    out: Path
    reference_panel: str | Path
    store_id: str
    release_id: str
    chain_file: str | Path | None
    liftover_failure_threshold: float
    chunk_shape: tuple[int, int]
    dtype: str
    n_workers: int
    eaf_reference: str | Path | None
    eaf_reference_ancestry: str | None
    allow_unverified_eaf: bool


@dataclass(frozen=True)
class _VariantPartition:
    """One Hybrid build's partition of its union variant set.

    The Dense Component axis is exactly the reference panel's ALIDs in genomic
    order; the Ragged Overflow holds the observed ALIDs outside it.
    ``shared_sorted`` is the union of the two -- the store's root variant axis
    -- and ``dense_row``/``shared_index`` map an ALID to the index each
    component stores it under. The layout contract both components share is
    that dense row ``i`` is the ``i``-th panel ALID of ``shared_sorted``, so
    ``dense_to_shared.npy`` is a strictly ascending map.
    """

    panel_sorted: list[str]
    off_panel_alids: list[str]
    shared_sorted: list[str]
    dense_row: dict[str, int]
    shared_index: dict[str, int]
    n_panel: int
    n_off_panel: int
    n_shared: int


@dataclass(frozen=True)
class _SourceAxis:
    """The lifted union axis: every phase between Pass 1 and the Dense
    skeleton consumes this and nothing else."""

    dense_dir: Path
    dense_staged: StagedRelease
    partition: _VariantPartition
    analyses: list[Analysis]
    hg38_to_source: dict[str, str | None]
    rsid_by_alid: dict[str, str]
    keys_sorted: np.ndarray
    targets_sorted: np.ndarray
    ispanel_sorted: np.ndarray


@dataclass(frozen=True)
class _PreparedBuild:
    """Everything the build knows before Pass 2 spills exist: the staged
    paths, the partition/provenance maps, the Dense skeleton's
    ``dense_to_shared`` sidecar, the routing arrays and the spill directory
    Pass 2 writes through."""

    staged: StagedRelease
    dense_dir: Path
    dense_staged: StagedRelease
    partition: _VariantPartition
    manifest_rows: list[_ManifestRow]
    analyses: list[Analysis]
    hg38_to_source: dict[str, str | None]
    rsid_by_alid: dict[str, str]
    dense_to_shared: np.ndarray
    spill_dir: Path
    keys_sorted: np.ndarray
    targets_sorted: np.ndarray
    ispanel_sorted: np.ndarray
    n_analyses: int


@dataclass(frozen=True)
class _RoutedSpills:
    """What Pass 2 leaves for the EAF survey."""

    id_by_col: dict[int, str]
    pass2_start: float


@dataclass(frozen=True)
class _EafEvidence:
    """The per-component frequency surveys and the orientation report."""

    dense_survey: EafSpillSurvey
    overflow_survey: EafSpillSurvey
    report: EafOrientationReport


@dataclass(frozen=True)
class _EncodingPlan:
    """The one encoding both components share (ADR 0037) plus the effective
    chunk shape it created the Dense zarr under."""

    encoding: StoreEncoding
    effective_chunks: tuple[int, int]


@dataclass(frozen=True)
class _DenseWritten:
    """The concatenated top-hit candidates the Dense band writer harvested."""

    all_rows: np.ndarray
    all_cols: np.ndarray
    all_z: np.ndarray
    all_se: np.ndarray
    column_has_eaf: np.ndarray


@dataclass(frozen=True)
class _OverflowAssembled:
    """The assembled overflow CSR and its per-Analysis EAF presence."""

    csr: RaggedCSRWriter
    overflow_has_eaf: np.ndarray


@dataclass(frozen=True)
class _ComponentResult:
    """What the spill-lifetime seam hands to finalisation."""

    csr: RaggedCSRWriter
    encoding: StoreEncoding
    se_coefficients: np.ndarray | None
    eaf_provenance: dict[str, Any]
    analyses: list[Analysis]


def _stage_dense_component(
    staged: StagedRelease,
    reference_panel: str | Path,
) -> tuple[Path, StagedRelease, set[str]]:
    """Open the nested Dense Component's staging directory inside the outer
    store's, and read the reference panel that defines its axis. A panel with
    no ALIDs fails the build loudly rather than building a Dense axis that
    stores nothing."""
    dense_dir = dense_component_path(staged.path)
    dense_dir.mkdir()
    dense_staged = StagedRelease(dense_dir)
    panel_alids = read_reference_panel_alids(reference_panel)
    if not panel_alids:
        raise ValueError(f"reference panel {reference_panel} contained no ALIDs")
    log.info("Reference panel: %d variants", len(panel_alids))
    return dense_dir, dense_staged, panel_alids


def _partition_variants(
    source_lookup: dict[tuple[str, int, str, str], str],
    panel_alids: set[str],
    n_analyses: int,
) -> _VariantPartition:
    """Partition the observed hg38 ALIDs (every source row's lifted or
    passthrough position, Pass 1's union) into the on-panel set the Dense
    Component stores and the off-panel set the Ragged Overflow stores (ADR
    0026). Nothing observed is dropped, so an off-panel variant is always on
    the shared root axis."""
    observed_alids = set(source_lookup.values())
    off_panel_alids = sorted(observed_alids - panel_alids, key=_alid_sort_key)
    panel_sorted = sorted(panel_alids, key=_alid_sort_key)
    shared_sorted = sorted(panel_alids | set(off_panel_alids), key=_alid_sort_key)
    dense_row = {alid: i for i, alid in enumerate(panel_sorted)}
    shared_index = {alid: i for i, alid in enumerate(shared_sorted)}
    log.info(
        "Partition: %d panel (dense), %d off-panel (overflow), %d shared variants, %d analyses",
        len(panel_sorted),
        len(off_panel_alids),
        len(shared_sorted),
        n_analyses,
    )
    return _VariantPartition(
        panel_sorted=panel_sorted,
        off_panel_alids=off_panel_alids,
        shared_sorted=shared_sorted,
        dense_row=dense_row,
        shared_index=shared_index,
        n_panel=len(panel_sorted),
        n_off_panel=len(off_panel_alids),
        n_shared=len(shared_sorted),
    )


def _source_origin_map(
    source_lookup: dict[tuple[str, int, str, str], str],
) -> dict[str, str | None]:
    """Map each hg38 ALID to the source-build ALID its row was resolved from
    -- the Store Variant Table's ``source_alid`` provenance column. Several
    source sites can resolve onto one hg38 ALID (two manifest rows landing on
    one variant from different assemblies, say); when their origins differ
    the provenance is genuinely ambiguous and the map records ``None`` rather
    than guessing which source owns the stored row (issue #85) -- a store
    must not misattribute one row's association to another's variant.
    """
    hg38_to_source: dict[str, str | None] = {}
    for (chrom, pos, ref, alt), hg38_alid in source_lookup.items():
        a1, a2 = sorted((ref, alt))
        origin = f"{chrom}:{pos}:{a1}:{a2}"
        if hg38_alid not in hg38_to_source:
            hg38_to_source[hg38_alid] = origin
        elif hg38_to_source[hg38_alid] != origin:
            hg38_to_source[hg38_alid] = None
    return hg38_to_source


def _load_manifest(manifest_path: str | Path) -> list[_ManifestRow]:
    """Read the build manifest, failing loudly on an empty one rather than
    building a store with no Analyses (a plausible empty answer)."""
    manifest_rows = _read_manifest(manifest_path)
    if not manifest_rows:
        raise ValueError(f"manifest {manifest_path} contains no rows")
    return manifest_rows


def _lift_and_partition(
    staged: StagedRelease,
    manifest_rows: list[_ManifestRow],
    options: _BuildOptions,
) -> _SourceAxis:
    """Phase - lifting and partition/routing: open the Dense staging dir,
    read the panel, run Pass 1 (the union of every source row's variants and
    the hg19 -> hg38 lift for the rows that need one), partition the union
    into on-panel/off-panel, derive the provenance map (collision handling)
    and the Analyses, and compose the fork-safe routing index."""
    dense_dir, dense_staged, panel_alids = _stage_dense_component(
        staged,
        options.reference_panel,
    )
    source_lookup, rsid_by_alid = _lift_manifest_variants(
        manifest_rows,
        chain_file=options.chain_file,
        liftover_failure_threshold=options.liftover_failure_threshold,
    )
    partition = _partition_variants(
        source_lookup,
        panel_alids,
        len(manifest_rows),
    )
    analyses: list[Analysis] = [_manifest_row_to_analysis(row) for row in manifest_rows]
    hg38_to_source = _source_origin_map(source_lookup)
    keys_sorted, targets_sorted, ispanel_sorted = _build_routing_index(
        source_lookup,
        partition.dense_row,
        partition.shared_index,
    )
    # The routing index is built: the source union is freed before Pass 2.
    del source_lookup
    return _SourceAxis(
        dense_dir=dense_dir,
        dense_staged=dense_staged,
        partition=partition,
        analyses=analyses,
        hg38_to_source=hg38_to_source,
        rsid_by_alid=rsid_by_alid,
        keys_sorted=keys_sorted,
        targets_sorted=targets_sorted,
        ispanel_sorted=ispanel_sorted,
    )


def _write_dense_component_skeleton(
    staged: StagedRelease,
    axis: _SourceAxis,
    chunk_shape: tuple[int, int],
) -> np.ndarray:
    """Write the Dense Component's valid-store skeleton: index, Store Variant
    Table and the dense row -> shared variant_index sidecar. ``dense_to_shared``
    is returned because the EAF survey samples both components on the shared
    axis through it and the query facade reads it back."""
    _write_index(axis.dense_staged, axis.partition.panel_sorted, axis.analyses, chunk_shape)
    _write_variant_table(
        axis.dense_dir,
        axis.partition.panel_sorted,
        axis.hg38_to_source,
        axis.rsid_by_alid,
    )
    # Ascending: the panel keeps genomic order, so dense row i is shared row
    # dense_to_shared[i] -- the mapping validation checks against.
    dense_to_shared = np.array(
        [axis.partition.shared_index[alid] for alid in axis.partition.panel_sorted],
        dtype=np.int32,
    )
    np.save(dense_to_shared_path(staged.path), dense_to_shared)
    return dense_to_shared


def _route_serial(
    prepared: _PreparedBuild,
    analysis_index: dict[str, int],
    n_analyses: int,
    pass2_start: float,
) -> None:
    """Route each study once, in this process, spilling the dense rows and the
    overflow associations it resolves (last-wins dedup per target index)."""
    for i, row in enumerate(prepared.manifest_rows):
        dense, overflow = _resolve_column_hybrid(
            row.file_path,
            prepared.keys_sorted,
            prepared.targets_sorted,
            prepared.ispanel_sorted,
            row.se_divisor,
            capability=row.source_reader_capability,
            stored_effect_scale=row.stored_effect_scale,
        )
        _spill_hybrid_column(prepared.spill_dir, analysis_index[row.trait_id], dense, overflow)
        _log_progress("Pass 2", i + 1, n_analyses, pass2_start, f"last: {row.trait_id}", every=25)


def _route_parallel(
    prepared: _PreparedBuild,
    analysis_index: dict[str, int],
    id_by_col: dict[int, str],
    n_analyses: int,
    options: _BuildOptions,
    pass2_start: float,
) -> None:
    """Route each study through the fork pool. Workers read the routing arrays
    through the module-level globals below rather than as arguments: they are
    inherited by fork, which is what keeps a genome-scale lookup out of the
    per-column pickling the pool would otherwise do (dense.build_vcf's
    rationale)."""
    global _pass2_keys_sorted, _pass2_targets_sorted, _pass2_ispanel_sorted
    global _pass2_spill_dir
    _pass2_keys_sorted = prepared.keys_sorted
    _pass2_targets_sorted = prepared.targets_sorted
    _pass2_ispanel_sorted = prepared.ispanel_sorted
    _pass2_spill_dir = prepared.spill_dir
    try:
        with _fork_pool(options.n_workers) as pool:
            tasks = [
                (
                    analysis_index[row.trait_id],
                    row.file_path,
                    row.se_divisor,
                    row.source_reader_capability,
                    row.stored_effect_scale,
                )
                for row in prepared.manifest_rows
            ]
            futures = [pool.submit(_pass2_worker, task) for task in tasks]
            for i, future in enumerate(as_completed(futures)):
                col = future.result()
                _log_progress(
                    "Pass 2", i + 1, n_analyses, pass2_start, f"last: {id_by_col[col]}", every=25
                )
    finally:
        _pass2_keys_sorted = None
        _pass2_targets_sorted = None
        _pass2_ispanel_sorted = None
        _pass2_spill_dir = None


def _route_studies(
    prepared: _PreparedBuild,
    options: _BuildOptions,
) -> _RoutedSpills:
    """Phase - Pass 2: read each study once and route every association into
    the dense spill or the overflow spill (fork pool when n_workers > 1).
    Returns the {column: analysis_id} map the EAF survey keys and the pass
    start time the band writer's progress reports from."""
    rows = prepared.manifest_rows
    analysis_index = {row.trait_id: i for i, row in enumerate(rows)}
    id_by_col = {i: row.trait_id for i, row in enumerate(rows)}
    n_analyses = len(rows)
    log.info("Pass 2: routing %d analyses (n_workers=%d)", n_analyses, options.n_workers)
    pass2_start = time.monotonic()
    if options.n_workers <= 1:
        _route_serial(
            prepared,
            analysis_index,
            n_analyses,
            pass2_start,
        )
    else:
        _route_parallel(
            prepared,
            analysis_index,
            id_by_col,
            n_analyses,
            options,
            pass2_start,
        )
    return _RoutedSpills(id_by_col=id_by_col, pass2_start=pass2_start)


def _verify_eaf_orientation(
    prepared: _PreparedBuild,
    routed: _RoutedSpills,
    options: _BuildOptions,
) -> _EafEvidence:
    """Phase - EAF orientation (issue #115): check both components at once,
    sampled on the shared axis so an Analysis whose frequencies live mostly in
    the overflow is checked on the same footing as one sitting on the panel.
    The components are sampled separately and merged, so an Analysis present
    in both contributes up to twice the per-Analysis budget -- more evidence
    than asked for, never less."""
    shared_hashes = site_hashes(prepared.partition.shared_sorted)
    dense_survey = survey_eaf_spills(
        prepared.spill_dir,
        routed.id_by_col,
        prepared.partition.shared_sorted,
        shared_hashes,
        row_map=prepared.dense_to_shared,
    )
    overflow_survey = survey_eaf_spills(
        prepared.spill_dir,
        routed.id_by_col,
        prepared.partition.shared_sorted,
        shared_hashes,
        suffix=".ovf",
        index_key="variant_index",
    )
    observations = dense_survey.observations
    for analysis_id, off_panel in overflow_survey.observations.items():
        observations[analysis_id].update(off_panel)
    report = verify_eaf_orientation(
        observations,
        eaf_reference=options.eaf_reference,
        eaf_reference_ancestry=options.eaf_reference_ancestry,
        allow_unverified=options.allow_unverified_eaf,
    )
    return _EafEvidence(
        dense_survey=dense_survey,
        overflow_survey=overflow_survey,
        report=report,
    )


def _plan_joint_encoding(
    prepared: _PreparedBuild,
    evidence: _EafEvidence,
    options: _BuildOptions,
) -> _EncodingPlan:
    """Phase - one encoding plan for both components (issue #119, ADR 0037).
    The Dense Component and the Ragged Overflow partition one Analysis's
    associations, so a shared result contract needs a shared encoding: their
    measurements are combined rather than either one taken alone. Materialises
    the Dense zarr skeleton under the plan and returns the effective chunks."""
    encoding = StoreEncoding.decide(
        EncodingMeasurements(
            n_analyses=prepared.n_analyses,
            eaf=combine_eaf_measurements(
                [
                    evidence.dense_survey.measurements(
                        n_cells=prepared.partition.n_panel * prepared.n_analyses,
                        n_variants=prepared.partition.n_panel,
                    ),
                    evidence.overflow_survey.measurements(
                        n_cells=evidence.overflow_survey.n_spill_cells,
                        n_variants=prepared.partition.n_shared,
                    ),
                ]
            ),
        )
    )
    log.info("Encoding plan: %s", encoding.to_manifest())
    effective_chunks = _create_dense_zarr(
        prepared.dense_staged,
        prepared.partition.n_panel,
        prepared.n_analyses,
        options.chunk_shape,
        options.dtype,
        encoding,
    )
    return _EncodingPlan(encoding=encoding, effective_chunks=effective_chunks)


def _write_dense_component_bands(
    prepared: _PreparedBuild,
    plan: _EncodingPlan,
    pass2_start: float,
    options: _BuildOptions,
) -> _DenseWritten:
    """Phase - the Dense band write, reusing the dense builder's band-streamer
    and top-hit harvest."""
    all_rows, all_cols, all_z, all_se, column_has_eaf = _write_dense_bands(
        prepared.dense_staged,
        prepared.spill_dir,
        prepared.partition.n_panel,
        prepared.n_analyses,
        plan.effective_chunks,
        options.dtype,
        pass2_start,
        plan.encoding,
    )
    return _DenseWritten(
        all_rows=all_rows,
        all_cols=all_cols,
        all_z=all_z,
        all_se=all_se,
        column_has_eaf=column_has_eaf,
    )


def _assemble_overflow(
    prepared: _PreparedBuild,
) -> _OverflowAssembled:
    """Phase - assemble the Ragged Overflow CSR from the per-column overflow
    spills, in analysis order so CSR offsets align with analysis_index."""
    log.info("Assembling Ragged Overflow CSR from %d columns", prepared.n_analyses)
    csr, overflow_has_eaf = _assemble_overflow_csr(
        prepared.spill_dir,
        prepared.n_analyses,
        prepared.partition.n_shared,
    )
    return _OverflowAssembled(csr=csr, overflow_has_eaf=overflow_has_eaf)


def _stamp_analyses(
    prepared: _PreparedBuild,
    dense: _DenseWritten,
    overflow: _OverflowAssembled,
    evidence: _EafEvidence,
) -> list[Analysis]:
    """Stamp each Analysis's eaf_scope (ADR 0036) and orientation columns.
    eaf_scope is the union of what the two components stored, so neither
    analyses.tsv may be written until both have been read."""
    return apply_orientation_evidence(
        _apply_eaf_scope(prepared.analyses, dense.column_has_eaf | overflow.overflow_has_eaf),
        evidence.report,
    )


def _fit_joint_se(
    prepared: _PreparedBuild,
    plan: _EncodingPlan,
    overflow: _OverflowAssembled,
) -> tuple[StoreEncoding, np.ndarray | None]:
    """Phase - one SE model and one decision across both components. They
    partition the same Analyses, so fitting or gating either in isolation
    could leave the shared manifest describing only half of the data it
    governs."""
    dense_group = prepared.dense_staged.arrays(mode="a")
    return optimise_dense_se_joint(
        dense_group,
        plan.encoding,
        overflow=overflow.csr.se_fit_inputs(plan.encoding),
    )


def _finish_dense_component(
    prepared: _PreparedBuild,
    dense: _DenseWritten,
    analyses: list[Analysis],
    encoding: StoreEncoding,
    evidence: _EafEvidence,
    options: _BuildOptions,
) -> dict[str, Any]:
    """Phase - finish the Dense Component as a valid dense store: top-hit
    index (after the SE decision -- the index carries the values a query reads
    back), manifest.json, and its own analyses.tsv counting only on-panel hits
    (the shared root counts both; issue #107)."""
    write_top_hit_indexes_for_store(
        prepared.dense_dir,
        dense.all_rows,
        dense.all_cols,
        dense.all_z,
        dense.all_se,
        encoding,
    )
    eaf_provenance = evidence.report.provenance(allow_unverified=options.allow_unverified_eaf)
    _write_dense_manifest(
        prepared.dense_staged,
        options.store_id,
        options.release_id,
        prepared.partition.n_panel,
        prepared.n_analyses,
        options.chain_file,
        options.dtype,
        encoding=encoding,
        eaf_orientation=eaf_provenance,
    )
    write_analyses_tsv(prepared.dense_dir, add_hit_counts(prepared.dense_dir, analyses))
    return eaf_provenance


def _flush_overflow_component(
    staged: StagedRelease,
    csr: RaggedCSRWriter,
    encoding: StoreEncoding,
    se_coefficients: np.ndarray | None,
) -> int:
    """Flush the assembled overflow CSR into the store's root zarr and build
    its top-hit index. Returns the overflow association count the shared
    manifest's provenance records."""
    csr.flush(staged.path, encoding, se_coefficients=se_coefficients)
    n_overflow = csr.n_associations
    log.info("Building Ragged Overflow top-hit index")
    build_ragged_top_hit_indexes(staged.path, encoding=encoding)
    return n_overflow


def _write_shared_metadata(
    prepared: _PreparedBuild,
    components: _ComponentResult,
    options: _BuildOptions,
    n_overflow: int,
) -> None:
    """Phase - the shared union table and shared metadata: the Hybrid
    manifest (again before analyses.tsv), then the root variant axis, index
    and analyses.tsv whose Top-Hit Counts are the Dense Component's and Ragged
    Overflow's counts summed (ADR 0032) -- the two partition an Analysis's
    associations disjointly, so neither alone is the whole picture."""
    _write_hybrid_manifest(
        prepared.staged,
        options.store_id,
        options.release_id,
        n_variants=prepared.partition.n_shared,
        n_analyses=prepared.n_analyses,
        n_panel=prepared.partition.n_panel,
        n_off_panel=prepared.partition.n_off_panel,
        n_overflow=n_overflow,
        chain_file=options.chain_file,
        chunk_shape=options.chunk_shape,
        dtype=options.dtype,
        encoding=components.encoding,
        eaf_orientation=components.eaf_provenance,
    )
    dense_counted = add_hit_counts(prepared.dense_dir, components.analyses)
    shared_analyses = add_hit_counts(prepared.staged.path, dense_counted)
    _write_index(
        prepared.staged,
        prepared.partition.shared_sorted,
        components.analyses,
        options.chunk_shape,
    )
    write_analyses_tsv(prepared.staged.path, shared_analyses)
    _write_variant_table(
        prepared.staged.path,
        prepared.partition.shared_sorted,
        prepared.hg38_to_source,
        prepared.rsid_by_alid,
    )


def _prepare_build(
    staged: StagedRelease,
    manifest_rows: list[_ManifestRow],
    options: _BuildOptions,
) -> _PreparedBuild:
    """Seam - preparation: lifting, partition/routing and the Dense skeleton,
    then the routing index and spill directory. Nothing here reads a spill;
    the returned record is the whole handoff to the spill-lifetime seam."""
    axis = _lift_and_partition(staged, manifest_rows, options)
    dense_to_shared = _write_dense_component_skeleton(
        staged,
        axis,
        options.chunk_shape,
    )
    spill_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{options.out.name}.hybridspill.",
            dir=staged.path.parent,
        )
    )
    return _PreparedBuild(
        staged=staged,
        dense_dir=axis.dense_dir,
        dense_staged=axis.dense_staged,
        partition=axis.partition,
        manifest_rows=manifest_rows,
        analyses=axis.analyses,
        hg38_to_source=axis.hg38_to_source,
        rsid_by_alid=axis.rsid_by_alid,
        dense_to_shared=dense_to_shared,
        spill_dir=spill_dir,
        keys_sorted=axis.keys_sorted,
        targets_sorted=axis.targets_sorted,
        ispanel_sorted=axis.ispanel_sorted,
        n_analyses=len(manifest_rows),
    )


def _build_components(
    prepared: _PreparedBuild,
    options: _BuildOptions,
) -> _ComponentResult:
    """Seam - the spill-lifetime build: Pass 2 routing, EAF verification,
    joint encoding, the component writes (Dense bands, Overflow CSR, shared SE
    fit, Dense top hits/manifest/analyses.tsv). The spill directory is removed
    in a finally whichever phase fails, and the store's files are only touched
    while the spills exist."""
    spill_dir = prepared.spill_dir
    try:
        routed = _route_studies(prepared, options)
        evidence = _verify_eaf_orientation(prepared, routed, options)
        plan = _plan_joint_encoding(prepared, evidence, options)
        dense = _write_dense_component_bands(prepared, plan, routed.pass2_start, options)
        overflow = _assemble_overflow(prepared)
        analyses = _stamp_analyses(prepared, dense, overflow, evidence)
        encoding, se_coefficients = _fit_joint_se(prepared, plan, overflow)
        eaf_provenance = _finish_dense_component(
            prepared,
            dense,
            analyses,
            encoding,
            evidence,
            options,
        )
    finally:
        shutil.rmtree(spill_dir, ignore_errors=True)
    return _ComponentResult(
        csr=overflow.csr,
        encoding=encoding,
        se_coefficients=se_coefficients,
        eaf_provenance=eaf_provenance,
        analyses=analyses,
    )


def _finalise_store(
    prepared: _PreparedBuild,
    components: _ComponentResult,
    options: _BuildOptions,
) -> HybridBuildResult:
    """Seam - finalisation: flush the Overflow CSR and build its top-hit
    index, then write the Hybrid manifest and the shared union table/metadata
    that make the store complete, and construct the result."""
    n_overflow = _flush_overflow_component(
        prepared.staged,
        components.csr,
        components.encoding,
        components.se_coefficients,
    )
    _write_shared_metadata(prepared, components, options, n_overflow)
    log.info(
        "Hybrid build complete: %d shared variants (%d panel + %d off-panel), "
        "%d analyses, %d overflow associations",
        prepared.partition.n_shared,
        prepared.partition.n_panel,
        prepared.partition.n_off_panel,
        prepared.n_analyses,
        n_overflow,
    )
    return HybridBuildResult(
        output_path=options.out,
        n_variants=prepared.partition.n_shared,
        n_analyses=prepared.n_analyses,
        n_panel=prepared.partition.n_panel,
        n_off_panel=prepared.partition.n_off_panel,
        n_overflow=n_overflow,
    )


# ── Public build entry point ─────────────────────────────────────────────────


def build_hybrid_from_vcf_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    reference_panel: str | Path,
    chain_file: str | Path | None = None,
    store_id: str,
    release_id: str,
    liftover_failure_threshold: float = 0.01,
    chunk_shape: tuple[int, int] = DEFAULT_CHUNK_SHAPE,
    dtype: str = DEFAULT_DTYPE,
    overwrite: bool = False,
    n_workers: int = 1,
    eaf_reference: str | Path | None = None,
    eaf_reference_ancestry: str | None = None,
    allow_unverified_eaf: bool = False,
) -> HybridBuildResult:
    """Build a Hybrid store from a manifest of GWAS-VCF files and a reference
    panel. A thin orchestrator over three deep seams (issue #130):
    ``_prepare_build`` (lifting, partition/routing, Dense skeleton), the
    spill-lifetime ``_build_components`` (Pass 2 routing, EAF verification,
    joint encoding, component writes) and ``_finalise_store`` (overflow flush,
    shared metadata, result). Each seam and phase helper preserves the
    contracts its docstring names: the staging context's atomicity, the
    collision/provenance rules, the disjoint-partition layout and the one
    encoding both components share (ADR 0037).

    The Dense Component axis is exactly ``reference_panel`` (hg38 ALIDs). Each
    study is read **once**: on-panel associations fill the nested Dense
    Component, off-panel associations go to the Ragged Overflow. Rows are
    assumed hg19 and lifted inline unless the manifest declares
    ``source_assembly=hg38`` (issue #85). ``eaf_reference`` drives the
    orientation check (issue #115) over both components; ``reference_panel``
    is a variant set and carries no frequencies, so it cannot serve as that
    reference - the two inputs are separate.
    """
    manifest_rows = _load_manifest(manifest_path)
    with OpenGWASDBStore.staging(Path(output_path), overwrite=overwrite) as staged:
        options = _BuildOptions(
            out=Path(output_path),
            reference_panel=reference_panel,
            store_id=store_id,
            release_id=release_id,
            chain_file=chain_file,
            liftover_failure_threshold=liftover_failure_threshold,
            chunk_shape=chunk_shape,
            dtype=dtype,
            n_workers=n_workers,
            eaf_reference=eaf_reference,
            eaf_reference_ancestry=eaf_reference_ancestry,
            allow_unverified_eaf=allow_unverified_eaf,
        )
        prepared = _prepare_build(staged, manifest_rows, options)
        components = _build_components(prepared, options)
        result = _finalise_store(prepared, components, options)
    return result


def _write_variant_table(
    store_path: Path,
    alids: list[str],
    hg38_to_source: dict[str, str | None],
    rsid_by_alid: dict[str, str],
) -> None:
    """Write one component's Store Variant Table.

    `rsid_by_alid` spans the whole build; each component writes the subset its
    own `alids` cover, so the Dense Component and the shared table agree on
    every row's identifier without either recomputing it (issue #109).
    """
    canonical = [
        CanonicalVariant(chromosome=chrom, position=int(pos), effect_allele=a1, other_allele=a2)
        for alid in alids
        for chrom, pos, a1, a2 in [alid.split(":")]
    ]
    source_alids = [hg38_to_source.get(alid) for alid in alids]
    write_variant_axis(store_path, canonical, rsid_by_alid, source_alids)


def _write_dense_manifest(
    dense_staged: StagedRelease,
    store_id: str,
    release_id: str,
    n_variants: int,
    n_analyses: int,
    chain_file: str | Path | None,
    dtype: str,
    encoding: StoreEncoding,
    eaf_orientation: dict[str, Any] | None = None,
) -> None:
    manifest = StoreManifest(
        encoding=encoding,
        store_id=f"{store_id}-dense",
        release_id=release_id,
        format_version=CURRENT_FORMAT_VERSION,
        primary_layout=PrimaryStorageLayout.DENSE,
        association_coverage=AssociationCoverage.FULL,
        completion_state=CompletionState.OBSERVED_ONLY,
        reference_assembly="GRCh38",
        created_at=datetime.now(UTC).isoformat(),
        provenance={
            "builder": "opengwasdb.v0.1_hybrid_dense_component",
            "chain_file": str(chain_file) if chain_file else "pyliftover_builtin_hg19_hg38",
            "n_variants": n_variants,
            "n_analyses": n_analyses,
            "dense": {"statistic_arrays": ["z", "se"], "se_dtype": encoding.se.dtype},
            **({"eaf_orientation": eaf_orientation} if eaf_orientation is not None else {}),
        },
    )
    dense_staged.write_manifest(manifest)


def _write_hybrid_manifest(
    staged: StagedRelease,
    store_id: str,
    release_id: str,
    n_variants: int,
    n_analyses: int,
    n_panel: int,
    n_off_panel: int,
    n_overflow: int,
    chain_file: str | Path | None,
    chunk_shape: tuple[int, int],
    dtype: str,
    encoding: StoreEncoding,
    eaf_orientation: dict[str, Any] | None = None,
) -> None:
    manifest = StoreManifest(
        encoding=encoding,
        store_id=store_id,
        release_id=release_id,
        format_version=CURRENT_FORMAT_VERSION,
        primary_layout=PrimaryStorageLayout.HYBRID,
        association_coverage=AssociationCoverage.FULL,
        completion_state=CompletionState.OBSERVED_ONLY,
        reference_assembly="GRCh38",
        created_at=datetime.now(UTC).isoformat(),
        provenance={
            "builder": "opengwasdb.v0.1_hybrid_vcf",
            "chain_file": str(chain_file) if chain_file else "pyliftover_builtin_hg19_hg38",
            "n_variants": n_variants,
            "n_analyses": n_analyses,
            "hybrid": {
                "dense_component": DENSE_SUBDIR,
                "n_panel": n_panel,
                "n_off_panel": n_off_panel,
                "n_overflow_associations": n_overflow,
                "se_dtype": encoding.se.dtype,
                "chunk_shape": list(chunk_shape),
                "compressor": DEFAULT_COMPRESSOR,
            },
            **({"eaf_orientation": eaf_orientation} if eaf_orientation is not None else {}),
        },
    )
    staged.write_manifest(manifest)
