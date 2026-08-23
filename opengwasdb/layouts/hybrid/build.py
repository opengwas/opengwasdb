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
    apply_orientation_evidence,
    site_hashes,
    verify_eaf_orientation,
)
from opengwasdb.build.liftover import LiftoverFailureError
from opengwasdb.encoding import EncodingMeasurements, StoreEncoding
from opengwasdb.layouts.dense.build import add_hit_counts, write_analyses_tsv
from opengwasdb.layouts.dense.build_vcf import (
    _RESOLVE_BATCH,
    _alid_sort_key,
    _apply_eaf_scope,
    _apply_se_divisor,
    _create_dense_zarr,
    _eaf_observations_from_spills,
    _encode_variant_keys,
    _fork_pool,
    _lift_manifest_variants,
    _log_progress,
    _manifest_row_to_analysis,
    _read_manifest,
    _write_dense_bands,
    _write_index,
)
from opengwasdb.layouts.dense.constants import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_COMPRESSOR,
    DEFAULT_DTYPE,
)
from opengwasdb.layouts.dense.top_hits import write_top_hit_indexes
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
    n_variants: int          # shared union table
    n_analyses: int
    n_panel: int             # Dense Component rows
    n_off_panel: int         # off-panel observed variants (overflow variant axis)
    n_overflow: int          # overflow associations


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
_pass2_targets_sorted: np.ndarray | None = None   # int64: dense row (panel) or shared idx (off)
_pass2_ispanel_sorted: np.ndarray | None = None   # bool: True = on-panel target
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
    spill_dir: Path, n_analyses: int, encoding: StoreEncoding
) -> tuple[RaggedCSRWriter, np.ndarray]:
    """Assemble the overflow CSR from per-column ``.ovf.npz`` spills, in analysis
    order (so CSR offsets align with analysis_index).

    Also returns a per-column bool array saying which Analyses carried an EAF
    into the *overflow* component. An Analysis can have EAF off-panel and none
    on it, so `eaf_scope` is the union of this and the Dense Component's own
    answer, never either alone (ADR 0036).
    """
    csr = RaggedCSRWriter(encoding)
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
            se = data["se"].astype(np.float16)
            eaf = data["eaf"].astype(np.float32)
        # Sort by variant_index for consistent within-analysis ordering (matches
        # the ragged BESD builder and lets top-hit CSR cross-validation searchsort).
        order = np.argsort(vi, kind="stable")
        has_eaf = bool(np.isfinite(eaf).any())
        column_has_eaf[col] = has_eaf
        csr.add_analysis(
            vi[order], z[order], se[order], eaf=eaf[order] if has_eaf else None
        )
        path.unlink()
    return csr, column_has_eaf


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
    """Build a Hybrid store from a manifest of GWAS-VCF files and a reference panel.

    The Dense Component axis is exactly ``reference_panel`` (hg38 ALIDs). Each
    study is read **once**: on-panel associations fill the nested Dense Component,
    off-panel associations go to the Ragged Overflow. Each row's source file is
    assumed hg19 and lifted to hg38 inline, unless its manifest row declares
    ``source_assembly=hg38`` (issue #85, e.g. a harmonised GWAS-SSF source),
    in which case it passes through unchanged -- see `_read_manifest`.

    ``eaf_reference``/``eaf_reference_ancestry``/``allow_unverified_eaf`` drive
    the EAF orientation check (issue #115), exactly as for the dense builder --
    over both components, since an Analysis's off-panel frequencies are as
    capable of being reported against the wrong allele as its on-panel ones.
    Note that ``reference_panel`` is a *variant set* (it defines the Dense
    Component axis) and carries no frequencies, so it cannot serve as the
    reference here; the two are separate inputs.
    """
    manifest_rows = _read_manifest(manifest_path)
    if not manifest_rows:
        raise ValueError(f"manifest {manifest_path} contains no rows")

    out = Path(output_path)
    with OpenGWASDBStore.staging(out, overwrite=overwrite) as staged:
        dense_dir = dense_component_path(staged.path)
        dense_dir.mkdir()
        dense_staged = StagedRelease(dense_dir)

        panel_alids = read_reference_panel_alids(reference_panel)
        if not panel_alids:
            raise ValueError(f"reference panel {reference_panel} contained no ALIDs")
        log.info("Reference panel: %d variants", len(panel_alids))

        # ── Pass 1: union of source variants + liftover ──────────────────────────
        source_lookup, rsid_by_alid = _lift_manifest_variants(
            manifest_rows,
            chain_file=chain_file,
            liftover_failure_threshold=liftover_failure_threshold,
        )

        # Partition observed hg38 ALIDs into on-panel and off-panel.
        observed_alids = set(source_lookup.values())
        off_panel_alids = sorted(observed_alids - panel_alids, key=_alid_sort_key)
        panel_sorted = sorted(panel_alids, key=_alid_sort_key)
        shared_sorted = sorted(panel_alids | set(off_panel_alids), key=_alid_sort_key)

        dense_row = {alid: i for i, alid in enumerate(panel_sorted)}
        shared_index = {alid: i for i, alid in enumerate(shared_sorted)}
        n_panel = len(panel_sorted)
        n_off_panel = len(off_panel_alids)
        n_shared = len(shared_sorted)
        n_analyses = len(manifest_rows)
        analysis_index = {row.trait_id: i for i, row in enumerate(manifest_rows)}
        log.info(
            "Partition: %d panel (dense), %d off-panel (overflow), %d shared variants, %d analyses",
            n_panel, n_off_panel, n_shared, n_analyses,
        )

        # Provenance: source-build ALID each hg38 row resolved from -- lifted from
        # hg19, or passed through unchanged from hg38 (None on collision).
        hg38_to_source: dict[str, str | None] = {}
        for (chrom, pos, ref, alt), hg38_alid in source_lookup.items():
            a1, a2 = sorted((ref, alt))
            origin = f"{chrom}:{pos}:{a1}:{a2}"
            if hg38_alid in hg38_to_source:
                if hg38_to_source[hg38_alid] != origin:
                    hg38_to_source[hg38_alid] = None
            else:
                hg38_to_source[hg38_alid] = origin

        # stored_effect_scale comes from the manifest, not the VCF header (issue
        # #17 -- the ieu-a-7 fix: the source header is not authoritative for
        # effect scale).
        analyses: list[Analysis] = [_manifest_row_to_analysis(row) for row in manifest_rows]

        # ── Write the Dense Component skeleton (a valid dense store) ──────────────
        _write_index(dense_staged, panel_sorted, analyses, chunk_shape, dtype)
        _write_variant_table(dense_dir, panel_sorted, hg38_to_source, rsid_by_alid)
        # One plan for both components (issue #119): the Dense Component and
        # the Ragged Overflow partition one Analysis's associations, so a
        # shared result contract needs a shared encoding.
        encoding = StoreEncoding.decide(EncodingMeasurements(n_analyses=n_analyses))
        effective_chunks = _create_dense_zarr(
            dense_staged, n_panel, n_analyses, chunk_shape, dtype, encoding
        )

        # dense row -> shared variant_index (ascending — panel keeps genomic order).
        dense_to_shared = np.array(
            [shared_index[alid] for alid in panel_sorted], dtype=np.int32
        )
        np.save(dense_to_shared_path(staged.path), dense_to_shared)

        # ── Pass 2: route each study once (dense spill + overflow spill) ──────────
        log.info("Pass 2: routing %d analyses (n_workers=%d)", n_analyses, n_workers)
        keys_sorted, targets_sorted, ispanel_sorted = _build_routing_index(
            source_lookup, dense_row, shared_index
        )
        del source_lookup
        t2 = time.monotonic()
        spill_dir = Path(
            tempfile.mkdtemp(prefix=f".{out.name}.hybridspill.", dir=staged.path.parent)
        )
        try:
            id_by_col = {analysis_index[row.trait_id]: row.trait_id for row in manifest_rows}
            if n_workers <= 1:
                for i, row in enumerate(manifest_rows):
                    col = analysis_index[row.trait_id]
                    dense, overflow = _resolve_column_hybrid(
                        row.file_path,
                        keys_sorted,
                        targets_sorted,
                        ispanel_sorted,
                        row.se_divisor,
                        capability=row.source_reader_capability,
                        stored_effect_scale=row.stored_effect_scale,
                    )
                    _spill_hybrid_column(spill_dir, col, dense, overflow)
                    _log_progress("Pass 2", i + 1, n_analyses, t2, f"last: {row.trait_id}", every=25)
            else:
                global _pass2_keys_sorted, _pass2_targets_sorted, _pass2_ispanel_sorted
                global _pass2_spill_dir
                _pass2_keys_sorted = keys_sorted
                _pass2_targets_sorted = targets_sorted
                _pass2_ispanel_sorted = ispanel_sorted
                _pass2_spill_dir = spill_dir
                try:
                    with _fork_pool(n_workers) as pool:
                        tasks = [
                            (
                                analysis_index[row.trait_id],
                                row.file_path,
                                row.se_divisor,
                                row.source_reader_capability,
                                row.stored_effect_scale,
                            )
                            for row in manifest_rows
                        ]
                        futures = [pool.submit(_pass2_worker, t) for t in tasks]
                        for i, fut in enumerate(as_completed(futures)):
                            col = fut.result()
                            _log_progress("Pass 2", i + 1, n_analyses, t2,
                                          f"last: {id_by_col[col]}", every=25)
                finally:
                    _pass2_keys_sorted = None
                    _pass2_targets_sorted = None
                    _pass2_ispanel_sorted = None
                    _pass2_spill_dir = None

            # EAF orientation (issue #115): both components at once, sampled
            # on the shared axis so an Analysis whose frequencies live mostly
            # in the overflow is checked on the same footing as one sitting on
            # the panel. The two components are sampled separately and merged,
            # so an Analysis present in both contributes up to twice the
            # per-Analysis budget -- more evidence than asked for, never less.
            # As in the dense builder, this runs before anything is written.
            shared_hashes = site_hashes(shared_sorted)
            eaf_observations = _eaf_observations_from_spills(
                spill_dir, id_by_col, shared_sorted, shared_hashes,
                row_map=dense_to_shared,
            )
            for analysis_id, off_panel in _eaf_observations_from_spills(
                spill_dir, id_by_col, shared_sorted, shared_hashes,
                suffix=".ovf", index_key="variant_index",
            ).items():
                eaf_observations[analysis_id].update(off_panel)
            eaf_report = verify_eaf_orientation(
                eaf_observations,
                eaf_reference=eaf_reference,
                eaf_reference_ancestry=eaf_reference_ancestry,
                allow_unverified=allow_unverified_eaf,
            )

            # Dense band-write (reuses the dense builder's band-streamer + top-hit harvest).
            all_rows, all_cols, all_z, all_se, column_has_eaf = _write_dense_bands(
                dense_staged, spill_dir, n_panel, n_analyses, effective_chunks, dtype, t2,
                encoding,
            )
            log.info("Writing Dense Component top-hit index (%d candidates)", len(all_rows))
            write_top_hit_indexes(dense_dir, all_rows, all_cols, all_z, all_se)

            # Overflow CSR assembly (reuses RaggedCSRWriter). Assembled before
            # either analyses.tsv is written: `eaf_scope` is the union of what
            # the two components stored, so neither can be stamped until both
            # have been read (ADR 0036).
            log.info("Assembling Ragged Overflow CSR from %d columns", n_analyses)
            csr, overflow_has_eaf = _assemble_overflow_csr(spill_dir, n_analyses, encoding)
            analyses = apply_orientation_evidence(
                _apply_eaf_scope(analyses, column_has_eaf | overflow_has_eaf), eaf_report
            )

            # manifest.json before analyses.tsv/overview.html: overview.html
            # reads manifest.json fresh from output_path for its header (ADR 0032).
            eaf_provenance = eaf_report.provenance(allow_unverified=allow_unverified_eaf)
            _write_dense_manifest(
                dense_staged, store_id, release_id, n_panel, n_analyses, chain_file, dtype,
                encoding=encoding,
                eaf_orientation=eaf_provenance,
            )
            write_analyses_tsv(dense_dir, add_hit_counts(dense_dir, analyses))
        finally:
            shutil.rmtree(spill_dir, ignore_errors=True)

        csr.flush(staged.path)
        n_overflow = csr.n_associations
        log.info("Building Ragged Overflow top-hit index")
        build_ragged_top_hit_indexes(staged.path, encoding=encoding)

        # manifest.json before analyses.tsv/overview.html, same reasoning as the
        # Dense Component write above.
        _write_hybrid_manifest(
            staged, store_id, release_id, n_shared, n_analyses, n_panel, n_off_panel,
            n_overflow, chain_file, chunk_shape, dtype, encoding=encoding,
            eaf_orientation=eaf_provenance,
        )

        # ── Shared union table + shared index ─────────────────────────────────────
        # Top-Hit Counts here are the Dense Component's and Ragged Overflow
        # Component's counts summed (ADR 0032): the two partition an Analysis's
        # associations disjointly, so neither alone is the whole picture.
        dense_counted = add_hit_counts(dense_dir, analyses)
        shared_analyses = add_hit_counts(staged.path, dense_counted)
        _write_index(staged, shared_sorted, analyses, chunk_shape, dtype)
        write_analyses_tsv(staged.path, shared_analyses)
        _write_variant_table(staged.path, shared_sorted, hg38_to_source, rsid_by_alid)

        log.info(
            "Hybrid build complete: %d shared variants (%d panel + %d off-panel), "
            "%d analyses, %d overflow associations",
            n_shared, n_panel, n_off_panel, n_analyses, n_overflow,
        )

    return HybridBuildResult(
        output_path=out, n_variants=n_shared, n_analyses=n_analyses,
        n_panel=n_panel, n_off_panel=n_off_panel, n_overflow=n_overflow,
    )


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
        CanonicalVariant(
            chromosome=chrom, position=int(pos), effect_allele=a1, other_allele=a2
        )
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
            "dense": {"statistic_arrays": ["z", "se"], "se_dtype": dtype},
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
                "se_dtype": dtype,
                "chunk_shape": list(chunk_shape),
                "compressor": DEFAULT_COMPRESSOR,
            },
            **({"eaf_orientation": eaf_orientation} if eaf_orientation is not None else {}),
        },
    )
    staged.write_manifest(manifest)
