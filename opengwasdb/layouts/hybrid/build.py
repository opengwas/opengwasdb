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
from dataclasses import dataclass, replace
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
from opengwasdb.build.liftover import LiftoverFailureError, build_liftover_lookup
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
    _apply_eaf_scope,
    _apply_se_divisor,
    _create_dense_zarr,
    _encode_variant_keys,
    _fork_pool,
    _lift_manifest_variants,
    _log_progress,
    _manifest_row_to_analysis,
    _ManifestRow,
    _pass2_worker_tasks,
    _read_manifest,
    _sorted_alids,
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
from opengwasdb.layouts.hybrid.unknown_keys import (
    decode_keys,
    encode_keys,
    is_hashed,
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
from opengwasdb.variants.reference import VariantReference, read_variant_reference

log = logging.getLogger(__name__)

# One column's off-reference spill: encoded uint64 keys, z, se, eaf, and the
# side-file half of the hashed keys (row positions plus their raw strings).
_OffReferenceSpill = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]

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
    keys_list = [
        f"{chrom}:{pos}:{ref}:{alt}".encode()
        for chrom, pos, ref, alt in zip(chroms, poss, refs, alts, strict=True)
    ]
    keys = np.array(keys_list, dtype=object)
    del keys_list, chroms, poss, refs, alts
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


def _match_hybrid_batch(
    chroms: list[str],
    keys_sorted: np.ndarray,
    poss: list[int],
    targets_sorted: np.ndarray,
    refs: list[str],
    ispanel_sorted: np.ndarray,
    alts: list[str],
    zs: list[float],
    ses: list[float],
    eafs: list[float],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Resolve one batch to ``(dense, overflow, off_reference)`` arrays.

    Dense and overflow entries carry their pre-assigned target index. An
    off-reference entry -- a source coordinate the routing index does not hold
    -- carries its raw ``chrom:pos:ref:alt`` key instead: the reference never
    named it, so its shared index is only assigned after Pass 2. Order
    preserving within each result.
    """
    z_arr = np.array(zs, dtype=np.float32)
    se_arr = np.array(ses, dtype=np.float32)
    eaf_arr = np.array(eafs, dtype=np.float32)
    empty = (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
    )
    if len(keys_sorted) == 0:
        keys = np.array(
            [f"{c}:{p}:{r}:{a}" for c, p, r, a in zip(chroms, poss, refs, alts, strict=True)],
            dtype=object,
        )
        return empty, empty, (keys, z_arr, se_arr, eaf_arr)
    query = _encode_variant_keys(chroms, poss, refs, alts)
    idx = np.searchsorted(keys_sorted, query)
    idx_clip = np.minimum(idx, len(keys_sorted) - 1)
    matched = keys_sorted[idx_clip] == query
    panel = ispanel_sorted[idx_clip[matched]]
    tgt = targets_sorted[idx_clip[matched]]
    z_m, se_m, eaf_m = z_arr[matched], se_arr[matched], eaf_arr[matched]
    unmatched = ~matched
    keys = np.array(
        [f"{chroms[j]}:{poss[j]}:{refs[j]}:{alts[j]}" for j in np.flatnonzero(unmatched)],
        dtype=object,
    )
    return (
        (tgt[panel], z_m[panel], se_m[panel], eaf_m[panel]),
        (tgt[~panel], z_m[~panel], se_m[~panel], eaf_m[~panel]),
        (keys, z_arr[unmatched], se_arr[unmatched], eaf_arr[unmatched]),
    )


def _extend(accumulators: tuple[list[np.ndarray], ...], parts: tuple[np.ndarray, ...]) -> None:
    """Append each non-empty part of ``parts`` to its accumulator."""
    for accumulator, part in zip(accumulators, parts, strict=True):
        if len(part):
            accumulator.append(part)


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
    _OffReferenceSpill,
]:
    """Stream one study once, routing each association to the dense fill (on-panel),
    the ragged overflow (off-panel/reference), or -- when the routing index does
    not hold its source coordinate at all -- an off-reference bucket keyed by the
    raw coordinate. Returns ``(dense, overflow, off_reference)`` where each is
    ``(index/key, z f32, se f32, eaf f32)`` deduped last-wins; `eaf` is NaN where
    the source reports no frequency (ADR 0036).

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
    u_keys: list[str] = []
    u_z: list[np.ndarray] = []
    u_se: list[np.ndarray] = []
    u_eaf: list[np.ndarray] = []

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
        dense, overflow, unknown = _match_hybrid_batch(
            chroms,
            keys_sorted,
            poss,
            targets_sorted,
            refs,
            ispanel_sorted,
            alts,
            zs,
            ses,
            eafs,
        )
        _extend((d_idx, d_z, d_se, d_eaf), dense)
        _extend((o_idx, o_z, o_se, o_eaf), overflow)
        if len(unknown[0]):
            u_keys.extend(str(key) for key in unknown[0])
            _extend((u_z, u_se, u_eaf), unknown[1:])
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

    def _assemble_unknown() -> _OffReferenceSpill:
        if not u_keys:
            return (
                np.empty(0, dtype=np.uint64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
                [],
            )
        encoded = encode_keys(u_keys)
        z, se, eaf = np.concatenate(u_z), np.concatenate(u_se), np.concatenate(u_eaf)
        keys, z, se, eaf = _dedup_last_wins(encoded.values, z, se, eaf)
        # Only the hashed survivors need a raw string in the side file; packed
        # keys decode from their own bits.
        lookup = encoded.hashed_lookup()
        hashed_index = np.flatnonzero(is_hashed(keys)).astype(np.int64)
        hashed_raw = [lookup[int(keys[position])] for position in hashed_index.tolist()]
        return keys, z, _apply_se_divisor(se, se_divisor), eaf, hashed_index, hashed_raw

    return (
        _assemble(d_idx, d_z, d_se, d_eaf),
        _assemble(o_idx, o_z, o_se, o_eaf),
        _assemble_unknown(),
    )


def _spill_hybrid_column(
    spill_dir: Path,
    col_idx: int,
    dense: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    overflow: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    off_reference: _OffReferenceSpill,
) -> None:
    """Spill one resolved study column: dense rows to ``{col}.npz`` (the layout the
    dense band-writer consumes), overflow to ``{col}.ovf.npz``, and off-reference
    associations -- keyed by one uint64 per raw source coordinate -- to
    ``{col}.unk.npz`` with a ``{col}.unk.raw`` side file for any key the packed
    encoding could not represent."""
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
    u_keys, u_z, u_se, u_eaf, u_hashed_index, u_hashed_raw = off_reference
    if len(u_keys):
        final = spill_dir / f"{col_idx}.unk.npz"
        tmp = spill_dir / f"{col_idx}.unk.tmp.npz"
        np.savez(
            tmp, keys=u_keys, z=u_z, se=u_se, eaf=u_eaf, hashed_index=u_hashed_index
        )
        tmp.replace(final)
        if len(u_hashed_index):
            _write_unknown_side_file(spill_dir, col_idx, u_hashed_raw)


def _pass2_worker(task: tuple[int, str, float, str, str]) -> int:
    assert _pass2_keys_sorted is not None
    assert _pass2_targets_sorted is not None
    assert _pass2_ispanel_sorted is not None
    assert _pass2_spill_dir is not None
    col_idx, file_path, se_divisor, capability, stored_effect_scale = task
    dense, overflow, off_reference = _resolve_column_hybrid(
        file_path,
        _pass2_keys_sorted,
        _pass2_targets_sorted,
        _pass2_ispanel_sorted,
        se_divisor,
        capability=capability,
        stored_effect_scale=stored_effect_scale,
    )
    _spill_hybrid_column(_pass2_spill_dir, col_idx, dense, overflow, off_reference)
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
    reference_panel: str | Path | None
    variant_reference: str | Path | None
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


def _open_dense_component(staged: StagedRelease) -> tuple[Path, StagedRelease]:
    """Open the nested Dense Component's staging directory inside the outer store's."""
    dense_dir = dense_component_path(staged.path)
    dense_dir.mkdir()
    return dense_dir, StagedRelease(dense_dir)


def _panel_alids(options: _BuildOptions) -> set[str]:
    """Read the legacy ``--reference-panel`` ALIDs, failing loudly when empty."""
    if options.reference_panel is None:
        raise ValueError("a variant reference or reference panel is required")
    panel_alids = read_reference_panel_alids(options.reference_panel)
    if not panel_alids:
        raise ValueError(f"reference panel {options.reference_panel} contained no ALIDs")
    log.info("Reference panel: %d variants", len(panel_alids))
    return panel_alids


def _resolve_reference_panel(options: _BuildOptions, reference: VariantReference) -> set[str]:
    """The Dense axis a ``--variant-reference`` build stores.

    The reference defines it. When ``--reference-panel`` is also supplied the two
    must be consistent -- every panel ALID must be one the reference carries --
    and the panel then narrows the Dense axis (off-panel variants go to the
    Overflow). An inconsistent panel is ignored and the reference axis used
    instead, with a warning (issue #186).
    """
    reference_alids = set(reference.alids)
    if options.reference_panel is None:
        return reference_alids
    panel_alids = _panel_alids(options)
    unknown = _sorted_alids(panel_alids - reference_alids)
    if unknown:
        log.warning(
            "--reference-panel lists %d ALID(s) absent from variant reference %s "
            "(e.g. %r); --variant-reference takes precedence",
            len(unknown),
            options.variant_reference,
            unknown[0],
        )
        return reference_alids
    return panel_alids


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
    off_panel_set = observed_alids - panel_alids
    off_panel_alids = _sorted_alids(off_panel_set)
    panel_sorted = _sorted_alids(panel_alids)
    shared_sorted = _sorted_alids(panel_alids | off_panel_set)
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


def _load_manifest(
    manifest_path: str | Path,
    *,
    default_source_reader_capability: str | None = None,
    default_source_assembly: str | None = None,
) -> list[_ManifestRow]:
    """Read the build manifest, failing loudly on an empty one rather than
    building a store with no Analyses (a plausible empty answer)."""
    manifest_rows = _read_manifest(
        manifest_path,
        default_source_reader_capability=default_source_reader_capability,
        default_source_assembly=default_source_assembly,
    )
    if not manifest_rows:
        raise ValueError(f"manifest {manifest_path} contains no rows")
    return manifest_rows


def _axis_source(
    staged: StagedRelease, manifest_rows: list[_ManifestRow], options: _BuildOptions
) -> tuple[
    Path,
    StagedRelease,
    dict[tuple[str, int, str, str], str],
    dict[str, str],
    set[str],
]:
    """Open the Dense staging dir and resolve the source-coordinate routing.

    With ``--variant-reference`` the reference replaces Pass 1 entirely: its
    ``source_lookup`` is the routing and its ALIDs define the Dense axis. Without
    one, the legacy Pass 1 reads every source once and lifts hg19 rows.
    """
    dense_dir, dense_staged = _open_dense_component(staged)
    if options.variant_reference is not None:
        reference = read_variant_reference(options.variant_reference)
        panel_alids = _resolve_reference_panel(options, reference)
        log.info(
            "Single-pass build: variant axis loaded from %s (%d panel variants); "
            "Pass 1 variant discovery bypassed",
            options.variant_reference,
            len(panel_alids),
        )
        return dense_dir, dense_staged, reference.source_lookup, reference.rsid_by_alid, panel_alids
    panel_alids = _panel_alids(options)
    source_lookup, rsid_by_alid = _lift_manifest_variants(
        manifest_rows,
        chain_file=options.chain_file,
        liftover_failure_threshold=options.liftover_failure_threshold,
        n_workers=options.n_workers,
    )
    return dense_dir, dense_staged, source_lookup, rsid_by_alid, panel_alids


def _lift_and_partition(
    staged: StagedRelease,
    manifest_rows: list[_ManifestRow],
    options: _BuildOptions,
) -> _SourceAxis:
    """Phase - axis source and partition/routing: open the Dense staging dir,
    resolve the Dense panel and the source-coordinate routing, partition the
    known variants into on-panel/off-panel, derive the provenance map and the
    Analyses, and compose the fork-safe routing index.

    Source coordinates the reference does not hold are not known until Pass 2;
    ``_finalise_reference_partition`` adds them to the shared axis there.
    """
    dense_dir, dense_staged, source_lookup, rsid_by_alid, panel_alids = _axis_source(
        staged, manifest_rows, options
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


def _parse_source_key(key: str) -> tuple[str, int, str, str]:
    chrom, position, ref, alt = key.split(":")
    return chrom, int(position), ref, alt


def _canonical_key(key: str) -> str:
    chrom, position, ref, alt = _parse_source_key(key)
    a1, a2 = sorted((ref, alt))
    return f"{chrom}:{position}:{a1}:{a2}"


def _unknown_side_path(spill_dir: Path, col: int) -> Path:
    """The per-column side file holding raw keys for the hashed entries."""
    return spill_dir / f"{col}.unk.raw"


def _write_unknown_side_file(spill_dir: Path, col: int, raw_keys: list[str]) -> None:
    """Write a column's hashed raw keys, one per line, atomically.

    ``chrom:pos:ref:alt`` cannot contain a newline, so the line order is the
    side file's only structure and it parallels ``hashed_index`` exactly.
    """
    side = _unknown_side_path(spill_dir, col)
    tmp = side.with_name(side.name + ".tmp")
    tmp.write_text("\n".join(raw_keys) + "\n", encoding="utf-8")
    tmp.replace(side)


def _read_unknown_side_file(spill_dir: Path, col: int) -> list[str]:
    side = _unknown_side_path(spill_dir, col)
    if not side.exists():
        return []
    return side.read_text(encoding="utf-8").splitlines()


@dataclass(frozen=True)
class _UnknownSpill:
    """One column's off-reference spill with its keys decoded to raw strings."""

    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray
    raw_keys: list[str]


def _load_unknown_spill(spill_dir: Path, col: int) -> _UnknownSpill | None:
    """Read one column's ``.unk`` spill; ``None`` when the column spilled none.

    The keys are uint64 by contract (issue #218) -- no pickle -- and a hashed
    key whose side-file entry is missing raises in ``decode_keys`` rather than
    silently shrinking the column.
    """
    path = spill_dir / f"{col}.unk.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        keys = data["keys"]
        z, se, eaf = data["z"], data["se"], data["eaf"]
        hashed_index = data["hashed_index"]
    raw_keys = decode_keys(keys, hashed_index, _read_unknown_side_file(spill_dir, col))
    return _UnknownSpill(z=z, se=se, eaf=eaf, raw_keys=raw_keys)


def _unknown_key_assembly(prepared: _PreparedBuild) -> dict[str, str | None]:
    """``{raw source key: declared assembly}`` for every off-reference spill entry.

    A raw coordinate string declared hg19 in one row and hg38 in another names
    two different physical loci; it cannot be resolved to one hg38 ALID and is
    left out (``None``) rather than guessed -- the same rule the inline Pass 1
    applies to its cross-assembly collisions.
    """
    key_assembly: dict[str, str | None] = {}
    for col, row in enumerate(prepared.manifest_rows):
        spill = _load_unknown_spill(prepared.spill_dir, col)
        if spill is None:
            continue
        for name in spill.raw_keys:
            if name in key_assembly and key_assembly[name] != row.source_assembly:
                key_assembly[name] = None
            else:
                key_assembly[name] = row.source_assembly
    return key_assembly


def _resolve_unknown_keys(
    prepared: _PreparedBuild, options: _BuildOptions
) -> dict[str, str]:
    """Map every resolvable off-reference source key to its hg38 ALID.

    Each spill belongs to one manifest row, so its keys are on that row's own
    assembly: hg38 rows canonicalise directly, hg19 rows go through one shared
    lift. A key that fails liftover, or that two rows declared on different
    assemblies, is omitted.
    """
    key_to_alid: dict[str, str] = {}
    hg19_keys: list[str] = []
    for key, assembly in _unknown_key_assembly(prepared).items():
        if assembly == "hg38":
            key_to_alid[key] = _canonical_key(key)
        elif assembly is not None:
            hg19_keys.append(key)
    if hg19_keys:
        tuples_by_key = {key: _parse_source_key(key) for key in hg19_keys}
        lifted = build_liftover_lookup(
            tuples_by_key.values(),
            from_build="hg19",
            to_build="hg38",
            failure_threshold=options.liftover_failure_threshold,
            chain_file=options.chain_file,
        )
        for key, parsed in tuples_by_key.items():
            alid = lifted.get(parsed)
            if alid is not None:
                key_to_alid[key] = alid
    return key_to_alid


def _unknown_origins(key_to_alid: dict[str, str]) -> dict[str, str | None]:
    """The source-build ALID each off-reference hg38 ALID came from (collisions blank)."""
    origins: dict[str, str | None] = {}
    for key, alid in key_to_alid.items():
        origin = _canonical_key(key)
        if alid not in origins:
            origins[alid] = origin
        elif origins[alid] != origin:
            origins[alid] = None
    return origins


def _append_overflow_spill(
    spill_dir: Path, col: int, idx: np.ndarray, z: np.ndarray, se: np.ndarray, eaf: np.ndarray
) -> None:
    """Append resolved off-reference entries to a column's existing overflow spill."""
    overflow_path = spill_dir / f"{col}.ovf.npz"
    if overflow_path.exists():
        with np.load(overflow_path) as data:
            idx, z, se, eaf = (
                np.concatenate([data["variant_index"], idx]),
                np.concatenate([data["z"], z]),
                np.concatenate([data["se"], se]),
                np.concatenate([data["eaf"], eaf]),
            )
    idx, z, se, eaf = _dedup_last_wins(idx, z, se, eaf)
    np.savez(overflow_path, variant_index=idx, z=z, se=se, eaf=eaf)


def _merge_unknown_column(
    spill_dir: Path, col: int, key_to_alid: dict[str, str], shared_index: dict[str, int]
) -> None:
    """Fold one column's ``.unk.npz`` into its ``.ovf.npz`` with shared indices.

    Keys that failed liftover (or were declared on two assemblies) are absent
    from ``key_to_alid`` and their associations are dropped with them.
    """
    spill = _load_unknown_spill(spill_dir, col)
    if spill is None:
        return
    keys = spill.raw_keys
    keep = np.array([i for i, key in enumerate(keys) if key in key_to_alid], dtype=np.int64)
    if len(keep):
        idx = np.array(
            [shared_index[key_to_alid[keys[int(i)]]] for i in keep.tolist()], dtype=np.int64
        )
        _append_overflow_spill(
            spill_dir, col, idx, spill.z[keep], spill.se[keep], spill.eaf[keep]
        )
    (spill_dir / f"{col}.unk.npz").unlink()
    _unknown_side_path(spill_dir, col).unlink(missing_ok=True)


def _remap_overflow_spills(
    prepared: _PreparedBuild, shared_index: dict[str, int]
) -> None:
    """Re-key existing overflow spills from the initial shared axis to the final one.

    Pass 2 wrote each on-reference off-panel association under the shared index
    the *initial* partition assigned it. Adding off-reference variants shifts
    every shared index at or after the first insertion point, so those entries
    have to be translated through their ALID before the two sets are combined --
    otherwise they silently point at whichever variant now occupies their old
    index (issue #186 review).
    """
    old_alids = prepared.partition.shared_sorted
    for col in range(prepared.n_analyses):
        path = prepared.spill_dir / f"{col}.ovf.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            vi = data["variant_index"].astype(np.int64)
            z, se, eaf = data["z"], data["se"], data["eaf"]
        remapped = np.array([shared_index[old_alids[int(i)]] for i in vi], dtype=np.int64)
        remapped, z, se, eaf = _dedup_last_wins(remapped, z, se, eaf)
        np.savez(path, variant_index=remapped, z=z, se=se, eaf=eaf)


def _merge_unknown_spills(
    prepared: _PreparedBuild, key_to_alid: dict[str, str], shared_index: dict[str, int]
) -> None:
    """Fold every column's ``.unk.npz`` into its ``.ovf.npz`` with shared indices."""
    for col in range(prepared.n_analyses):
        _merge_unknown_column(prepared.spill_dir, col, key_to_alid, shared_index)


def _finalise_reference_partition(
    prepared: _PreparedBuild, options: _BuildOptions
) -> _PreparedBuild:
    """Add Pass 2-discovered off-reference variants to the shared axis.

    With ``--variant-reference`` the reference fixes the Dense axis and the
    routing for the variants it knows, but a study may observe variants the
    reference never listed. Those are off-reference and belong in the Ragged
    Overflow; they are unknown until Pass 2 streams them, so their shared
    indices -- and therefore the panel's mapping onto them -- are only
    computable now. A legacy ``--reference-panel`` build already knows its
    whole union before Pass 2 and is returned unchanged.
    """
    if options.variant_reference is None:
        return prepared
    key_to_alid = _resolve_unknown_keys(prepared, options)
    if not key_to_alid:
        return prepared
    unknown_alids = set(key_to_alid.values())
    off_panel = _sorted_alids(set(prepared.partition.off_panel_alids) | unknown_alids)
    panel = prepared.partition.panel_sorted
    shared_sorted = _sorted_alids(set(panel) | set(off_panel))
    shared_index = {alid: i for i, alid in enumerate(shared_sorted)}
    # Existing overflow entries carry the *initial* axis's indices; translate
    # them before mixing in the off-reference entries keyed to the new one.
    _remap_overflow_spills(prepared, shared_index)
    _merge_unknown_spills(prepared, key_to_alid, shared_index)
    dense_to_shared = np.array([shared_index[alid] for alid in panel], dtype=np.int32)
    np.save(dense_to_shared_path(prepared.staged.path), dense_to_shared)
    hg38_to_source = dict(prepared.hg38_to_source)
    for alid, origin in _unknown_origins(key_to_alid).items():
        if alid in hg38_to_source and hg38_to_source[alid] != origin:
            hg38_to_source[alid] = None
        else:
            hg38_to_source[alid] = origin
    log.info(
        "Single-pass build: %d off-reference variant(s) routed to the Ragged Overflow",
        len(unknown_alids),
    )
    partition = replace(
        prepared.partition,
        off_panel_alids=off_panel,
        shared_sorted=shared_sorted,
        shared_index=shared_index,
        n_off_panel=len(off_panel),
        n_shared=len(shared_sorted),
    )
    return replace(
        prepared,
        partition=partition,
        dense_to_shared=dense_to_shared,
        hg38_to_source=hg38_to_source,
    )


def _route_serial(
    prepared: _PreparedBuild,
    analysis_index: dict[str, int],
    n_analyses: int,
    pass2_start: float,
) -> None:
    """Route each study once, in this process, spilling the dense rows and the
    overflow associations it resolves (last-wins dedup per target index)."""
    for i, row in enumerate(prepared.manifest_rows):
        dense, overflow, off_reference = _resolve_column_hybrid(
            row.file_path,
            prepared.keys_sorted,
            prepared.targets_sorted,
            prepared.ispanel_sorted,
            row.se_divisor,
            capability=row.source_reader_capability,
            stored_effect_scale=row.stored_effect_scale,
        )
        _spill_hybrid_column(
            prepared.spill_dir, analysis_index[row.trait_id], dense, overflow, off_reference
        )
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
            tasks = _pass2_worker_tasks(prepared.manifest_rows, analysis_index)
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
        eaf_provenance=components.eaf_provenance,
        variant_reference=(
            str(options.variant_reference) if options.variant_reference is not None else None
        ),
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


def _off_reference_spill_bytes(spill_dir: Path) -> tuple[int, int]:
    """``(encoded key bytes, raw side-file bytes)`` for a Pass 2 spill directory.

    Read after Pass 2 and before the ``.unk`` spills are folded away, so the peak
    scratch the off-reference keys cost is visible in the build log (issue #218).
    """
    encoded = side = 0
    for path in spill_dir.iterdir():
        if path.name.endswith(".unk.npz"):
            encoded += path.stat().st_size
        elif path.name.endswith(".unk.raw"):
            side += path.stat().st_size
    return encoded, side


def _build_components(
    prepared: _PreparedBuild,
    options: _BuildOptions,
) -> tuple[_PreparedBuild, _ComponentResult]:
    """Seam - the spill-lifetime build: Pass 2 routing, EAF verification,
    joint encoding, the component writes (Dense bands, Overflow CSR, shared SE
    fit, Dense top hits/manifest/analyses.tsv). The spill directory is removed
    in a finally whichever phase fails, and the store's files are only touched
    while the spills exist. Returns the (possibly repartitioned) prepared build
    alongside the components, because a variant-reference build learns its
    off-reference variants only here."""
    spill_dir = prepared.spill_dir
    try:
        routed = _route_studies(prepared, options)
        encoded_bytes, side_bytes = _off_reference_spill_bytes(spill_dir)
        log.info(
            "Pass 2 off-reference spill: %.2f GiB encoded keys + %.2f GiB raw side files",
            encoded_bytes / 2**30,
            side_bytes / 2**30,
        )
        # Off-reference variants are only known once Pass 2 has streamed the
        # sources; fold them into the shared axis before anything reads it.
        prepared = _finalise_reference_partition(prepared, options)
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
    return prepared, _ComponentResult(
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
    manifest_path: str | Path, output_path: str | Path, *,
    reference_panel: str | Path | None = None, variant_reference: str | Path | None = None,
    chain_file: str | Path | None = None, store_id: str, release_id: str,
    liftover_failure_threshold: float = 0.01, chunk_shape: tuple[int, int] = DEFAULT_CHUNK_SHAPE,
    dtype: str = DEFAULT_DTYPE, overwrite: bool = False, n_workers: int = 1,
    eaf_reference: str | Path | None = None, eaf_reference_ancestry: str | None = None,
    allow_unverified_eaf: bool = False, source_reader_capability: str | None = None,
    source_assembly: str | None = None,
) -> HybridBuildResult:
    """Build a Hybrid store from a manifest of GWAS-VCF files and a reference
    panel or precomputed variant reference. A thin orchestrator over three deep
    seams (issue #130): ``_prepare_build`` (axis source, partition/routing,
    Dense skeleton), the spill-lifetime ``_build_components`` (Pass 2 routing,
    EAF verification, joint encoding, component writes) and ``_finalise_store``
    (overflow flush, shared metadata, result). Each seam and phase helper
    preserves the contracts its docstring names: the staging context's
    atomicity, the collision/provenance rules, the disjoint-partition layout
    and the one encoding both components share (ADR 0037).

    The Dense Component axis is exactly ``variant_reference``'s ALIDs (or, when
    both are given, the ``--reference-panel`` subset, which the reference must
    carry; an inconsistent panel is ignored in favour of the reference). With a
    reference, Pass 1 variant discovery is bypassed
    (single-pass build, issue #186) and the reference's source-coordinate map
    routes on-reference associations to the Dense Component and off-reference
    ones to the Ragged Overflow during Pass 2. With only ``reference_panel``,
    the legacy two-pass build reads every source once and lifts hg19 rows.
    Rows are assumed hg19 and lifted inline unless the manifest declares
    ``source_assembly=hg38`` (issue #85); ``source_assembly`` and
    ``source_reader_capability`` options supply per-release defaults (#174).
    ``eaf_reference`` drives the orientation check (issue #115).
    """
    if reference_panel is None and variant_reference is None:
        raise ValueError("build-hybrid needs --reference-panel or --variant-reference")
    manifest_rows = _load_manifest(
        manifest_path,
        default_source_reader_capability=source_reader_capability,
        default_source_assembly=source_assembly,
    )
    with OpenGWASDBStore.staging(Path(output_path), overwrite=overwrite) as staged:
        options = _BuildOptions(
            out=Path(output_path),
            reference_panel=reference_panel,
            variant_reference=variant_reference,
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
        prepared, components = _build_components(prepared, options)
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
    eaf_provenance: dict[str, Any] | None = None,
    variant_reference: str | None = None,
) -> None:
    provenance: dict[str, Any] = {
        "builder": (
            "opengwasdb.v0.1_hybrid_single_pass"
            if variant_reference is not None
            else "opengwasdb.v0.1_hybrid_vcf"
        ),
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
    }
    if eaf_provenance is not None:
        provenance["eaf_orientation"] = eaf_provenance
    if variant_reference is not None:
        provenance["variant_reference"] = variant_reference
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
        provenance=provenance,
    )
    staged.write_manifest(manifest)
