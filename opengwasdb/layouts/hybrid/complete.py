"""Hybrid Reference Completion — impute only the Dense Component (ADR 0026, issue 058).

The Dense Component's axis is the LD reference panel, so completing it is exactly
the dense completion problem: this reuses ``complete_dense_store`` **unchanged** on
``<store>/dense``. The Ragged Overflow Component is off-panel (no LD structure) and
is left observed-only — its associations are copied through untouched.

Because dense completion may extend the panel axis (if the LD panel is a superset
of the build panel), the shared union table and the ``dense_to_shared`` map are
rebuilt from the *completed* Dense Component's axis, and the overflow
``variant_index`` values are remapped onto the rebuilt shared index space. When the
build panel already equals the LD panel (the intended case) nothing is added, so
the overflow indices are unchanged.

When the LD panel *does* extend the axis, a variant that was off-panel (with a
real observed overflow association) can cross onto the newly-extended panel.
``complete_dense_store`` never sees that observation -- it only completes from
the *source* Dense Component's own previously-on-panel cells -- so left alone it
would either impute that cell from LD-correlated neighbours or leave it missing,
while the overflow still carried the real value: the same variant would then
appear in both components, violating the disjoint-partition invariant
(``opengwasdb.validation.validate``), and the real observation would be at risk
of being silently shadowed by a lower-fidelity one. ``_fold_panel_crossovers``
detects every such crossover after dense completion, writes the real observed
value into the completed Dense Component (marked ``imputed=0``, a genuine
observation, not a guess) in place of whatever dense completion computed there,
and excludes it from the rebuilt overflow -- so no association ever needs to
choose between the two components, and no real observation is ever discarded in
favour of an estimate (issue #99).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from opengwasdb.encoding import (
    EAF_BASELINE,
    DenseEafPlane,
    DenseZPlane,
    StoreEncoding,
)
from opengwasdb.layouts.dense.build import add_hit_counts, write_analyses_tsv
from opengwasdb.layouts.dense.build_vcf import _alid_sort_key, _write_index
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.dense.constants import DEFAULT_COMPRESSOR, DEFAULT_DTYPE
from opengwasdb.layouts.dense.top_hits import build_top_hit_indexes as build_dense_top_hit_indexes
from opengwasdb.layouts.hybrid.build import _write_variant_table
from opengwasdb.layouts.hybrid.layout import (
    DENSE_SUBDIR,
    dense_component_path,
    dense_to_shared_path,
)
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.layouts.ragged.zarr_csr import (
    RAGGED_ZARR_PATH,
    RaggedCSRReader,
    RaggedCSRWriter,
)
from opengwasdb.model.analyses import Analysis, read_analysis_records
from opengwasdb.model.enums import (
    AssociationCoverage,
    CompletionState,
    PrimaryStorageLayout,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store.open import (
    OpenGWASDBStore,
    StagedRelease,
    check_writable_format_version,
    open_store,
)
from opengwasdb.variants import VariantAxis

log = logging.getLogger(__name__)

__all__ = ["complete_hybrid_store", "HybridCompletionResult"]


@dataclass(frozen=True)
class HybridCompletionResult:
    output_path: Path
    n_variants: int
    n_analyses: int
    n_panel: int
    n_off_panel: int
    n_overflow: int
    n_imputed: int


def complete_hybrid_store(
    source_path: str | Path,
    dest_path: str | Path,
    ld_dir: str | Path,
    *,
    ancestry: str = "EUR",
    min_cor: float = 0.7,
    thresh: float = 0.9,
    release_id: str | None = None,
    n_workers: int = 1,
    overwrite: bool = False,
) -> HybridCompletionResult:
    """Produce a Reference-Completed Hybrid store from an Observed-Only Hybrid source."""
    src = Path(source_path)
    dst = Path(dest_path)

    source = open_store(src)
    src_manifest = source.manifest
    # See the dense path: refused before the work, not after it (ADR 0038 §4).
    check_writable_format_version(
        src_manifest.format_version, source=f"source release {src}"
    )
    if src_manifest.primary_layout is not PrimaryStorageLayout.HYBRID:
        raise ValueError(
            f"source store is not Hybrid (primary_layout={src_manifest.primary_layout})"
        )
    if src_manifest.completion_state is not CompletionState.OBSERVED_ONLY:
        raise ValueError(
            "source hybrid store is not Observed-Only "
            f"(completion_state={src_manifest.completion_state})"
        )

    with OpenGWASDBStore.staging(dst, overwrite=overwrite) as staged:
        # ── 1. Complete the Dense Component (dense pipeline, unchanged) ────────
        # complete_dense_store derives the per-Analysis ancestry-match filter
        # (ADR 0028) itself from the Dense Component's own analyses.tsv when
        # impute_analysis_ids is not given -- the same filter hybrid used to
        # compute here inline, now applied uniformly regardless of entry point.
        log.info("Completing Dense Component via the dense reference-completion pipeline")
        dense_result = complete_dense_store(
            dense_component_path(src),
            dense_component_path(staged.path),
            ld_dir,
            ancestry=ancestry,
            min_cor=min_cor,
            thresh=thresh,
            release_id=release_id,
            n_workers=n_workers,
            overwrite=True,
        )

        # ── 2. Rebuild the shared union table from the completed dense axis ────
        dense_axis = VariantAxis(dense_component_path(staged.path))
        dense_records = dense_axis.all()
        dense_axis.close()
        dense_alids = [r.alid for r in dense_records]  # dense row order
        dense_alid_to_row = {alid: i for i, alid in enumerate(dense_alids)}

        src_csr = RaggedCSRReader(src)
        offsets = src_csr._offsets[:]
        n_analyses = len(offsets) - 1
        src_vi = src_csr._variant_index[:]
        src_z = src_csr.z_all()
        src_se = src_csr._se[:]
        src_eaf = src_csr.eaf_slice(0, len(src_vi))

        src_shared_axis = VariantAxis(src)
        vi_to_record = src_shared_axis.by_indices(np.unique(src_vi).tolist())
        src_shared_all = src_shared_axis.all()
        src_shared_axis.close()
        source_alid_by_alid = {r.alid: r.source_alid for r in src_shared_all}
        # Carry the observed store's rsids into the completed one (issue #109):
        # Reference Completion adds panel rows, it does not rename the rows the
        # source already named. Completed-only rows get no rsid.
        rsid_by_alid = {r.alid: r.rsid for r in src_shared_all if r.rsid}

        overflow_alids = np.array(
            [vi_to_record[int(v)].alid for v in src_vi], dtype=object
        ) if len(src_vi) else np.empty(0, dtype=object)

        # A variant off-panel in the source but newly on-panel after dense
        # completion (LD panel extension) "crosses over": fold its real
        # observation into the completed Dense Component and drop it from the
        # rebuilt overflow below, rather than let the same variant appear in
        # both components -- see the module docstring (issue #99).
        is_crossover = np.array(
            [alid in dense_alid_to_row for alid in overflow_alids], dtype=bool
        ) if len(overflow_alids) else np.empty(0, dtype=bool)
        # The Dense Component's own plan, read from the manifest
        # `complete_dense_store` just wrote: identical to the source's for
        # `z`/`se`, plus the `eaf_reference` that completion gave it.
        dense_encoding = open_store(dense_component_path(staged.path)).manifest.encoding
        n_reclaimed_imputed = _fold_panel_crossovers(
            dense_component_path(staged.path), dense_alid_to_row,
            offsets, src_z, src_se, src_eaf, overflow_alids, is_crossover,
            encoding=src_manifest.encoding,
            dense_encoding=dense_encoding,
        )
        n_imputed = dense_result.n_imputed - n_reclaimed_imputed
        if is_crossover.any():
            log.info(
                "%d off-panel association(s) crossed onto the newly-extended dense "
                "panel; folded in as real observations (%d had been imputed there, "
                "now corrected to the real value)",
                int(is_crossover.sum()), n_reclaimed_imputed,
            )
            # Rebuild the Dense Component's top-hit index from the patched z
            # values -- complete_dense_store already built one from its own
            # pre-patch data, which the fold above can invalidate for the
            # (small) fraction of cells it just overwrote. The completed
            # analyses.tsv's completion-quality rollup columns (median
            # pearson_r etc.), written by complete_dense_store before the
            # fold, are not similarly recomputed -- an accepted, narrow
            # imprecision limited to summary statistics for the crossed-over
            # cells, not a correctness invariant like top-hit presence.
            build_dense_top_hit_indexes(
                dense_component_path(staged.path), encoding=src_manifest.encoding
            )

        union = sorted(set(dense_alids) | set(overflow_alids.tolist()), key=_alid_sort_key)
        new_shared_index = {alid: i for i, alid in enumerate(union)}
        n_shared = len(union)
        n_panel = len(dense_alids)
        n_off_panel = n_shared - n_panel

        # ── 3. Write shared table + index at the destination ──────────────────
        # The completed Dense Component's analyses.tsv already carries
        # completed_against (written by complete_dense_store, issue #22) --
        # re-reading it here and writing it back at the shared root is the one
        # place Analysis metadata is written, not a second provenance carry.
        analyses = _read_analyses(dense_component_path(staged.path))
        _write_index(staged, union, analyses, _chunk_shape(src_manifest), DEFAULT_DTYPE)
        source_by_alid = {a: source_alid_by_alid.get(a) for a in union}
        _write_variant_table(staged.path, union, source_by_alid, rsid_by_alid)

        # ── 4. dense_to_shared map for the completed axis ─────────────────────
        dense_to_shared = np.array([new_shared_index[a] for a in dense_alids], dtype=np.int32)
        np.save(dense_to_shared_path(staged.path), dense_to_shared)

        # ── 5. Rebuild the overflow CSR with remapped shared indices, excluding
        #        any association folded into the Dense Component in step 2 ──────
        # Completion writes into the source's arrays, so it writes in the
        # source's encoding -- never re-encoding data an operator asked only to
        # complete (ADR 0038 §4). `check_writable_format_version` above is what
        # makes that honest.
        csr = RaggedCSRWriter(n_shared)
        # The Ragged Overflow's per-variant baselines travel with its values
        # across the shared-axis remap. Recomputing them from the decoded
        # frequencies would move every baseline by up to half a step and
        # re-quantise every cell against it, so a completed release would be
        # less accurate than the source it was completed from (ADR 0037 §2).
        overflow_baseline = _remapped_overflow_baseline(
            src, vi_to_record, new_shared_index, n_shared
        )
        for ai in range(n_analyses):
            s, e = int(offsets[ai]), int(offsets[ai + 1])
            keep = ~is_crossover[s:e] if e > s else np.empty(0, dtype=bool)
            if s == e or not keep.any():
                csr.add_analysis(
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.float16),
                )
                continue
            new_vi = np.array(
                [new_shared_index[vi_to_record[int(v)].alid] for v in src_vi[s:e][keep]],
                dtype=np.int32,
            )
            z = src_z[s:e][keep].astype(np.float32)
            se = src_se[s:e][keep].astype(np.float16)
            # Observed EAF survives the remap (ADR 0036), like rsids above:
            # Reference Completion adds rows, it does not change what the
            # source reported for the ones it already had.
            eaf = src_eaf[s:e][keep].astype(np.float32)
            order = np.argsort(new_vi, kind="stable")
            csr.add_analysis(
                new_vi[order],
                z[order],
                se[order],
                eaf=eaf[order] if np.isfinite(eaf).any() else None,
            )
        # The Ragged Overflow has no imputed cells -- completion imputes into
        # the Dense Component alone -- so it declares no `eaf_reference`, where
        # the nested Dense Component's own manifest does. Each component's plan
        # describes that component (ADR 0037 §4).
        csr.flush(staged.path, src_manifest.encoding, eaf_baseline=overflow_baseline)

        # ── 6. Hybrid manifest (reference-completed) ──────────────────────────
        # Written before analyses.tsv/overview.html below: overview.html
        # reads manifest.json fresh from output_path for its header (ADR 0032).
        new_release = release_id or f"{src_manifest.release_id}-completed"
        _write_completed_manifest(
            staged, src_manifest, new_release, n_shared, n_analyses, n_panel, n_off_panel,
            csr.n_associations, n_imputed,
        )

        build_ragged_top_hit_indexes(staged.path, encoding=src_manifest.encoding)
        # Dense Component counts already live on `analyses` (read back from
        # complete_dense_store's already-completed output); add the Ragged
        # Overflow Component's counts on top -- the two partition an
        # Analysis's associations disjointly (ADR 0032).
        write_analyses_tsv(staged.path, add_hit_counts(staged.path, analyses))

        log.info(
            "Hybrid completion complete: %d shared variants (%d panel + %d off-panel), "
            "%d imputed dense cells, %d overflow associations (observed-only)",
            n_shared, n_panel, n_off_panel, n_imputed, csr.n_associations,
        )

    return HybridCompletionResult(
        output_path=dst, n_variants=n_shared, n_analyses=n_analyses,
        n_panel=n_panel, n_off_panel=n_off_panel, n_overflow=csr.n_associations,
        n_imputed=n_imputed,
    )


def _fold_panel_crossovers(
    dense_dir: Path,
    dense_alid_to_row: dict[str, int],
    offsets: np.ndarray,
    src_z: np.ndarray,
    src_se: np.ndarray,
    src_eaf: np.ndarray,
    overflow_alids: np.ndarray,
    is_crossover: np.ndarray,
    encoding: StoreEncoding,
    dense_encoding: StoreEncoding,
) -> int:
    """Write every crossover association's real observed value into the
    completed Dense Component at ``(dense_row, analysis)``, marked
    ``imputed=0``, in place of whatever dense completion computed there.

    ``is_crossover`` is a boolean mask over the flat source-overflow
    association array (same order/length as ``overflow_alids``/``src_z``/
    ``src_se``/``src_eaf``); ``offsets`` partitions that flat array by analysis, exactly
    as ``RaggedCSRReader`` exposes it. Returns how many of the overwritten
    cells were previously marked imputed, so the caller can correct the
    reported imputed count -- see the module docstring (issue #99).
    """
    if not is_crossover.any():
        return 0

    n_analyses = len(offsets) - 1
    assoc_ai = np.repeat(np.arange(n_analyses), np.diff(offsets))
    crossover_idx = np.flatnonzero(is_crossover)

    row_idx = np.fromiter(
        (dense_alid_to_row[overflow_alids[k]] for k in crossover_idx),
        dtype=np.int64, count=len(crossover_idx),
    )
    col_idx = assoc_ai[crossover_idx].astype(np.int64)
    z_vals = src_z[crossover_idx].astype(np.float32)
    se_vals = src_se[crossover_idx].astype(np.float16)
    eaf_vals = src_eaf[crossover_idx].astype(np.float32)

    root = zarr.open_group(str(dense_dir / "data.zarr"), mode="a")
    was_imputed = np.asarray(root["imputed"].vindex[row_idx, col_idx])
    n_reclaimed = int(was_imputed.sum())
    # Through the plane, so the Dense Component's overflow table moves with the
    # cells being overwritten rather than being left describing their old values.
    DenseZPlane.open(root, encoding).patch(row_idx, col_idx, z_vals)
    root["se"].vindex[row_idx, col_idx] = se_vals
    root["imputed"].vindex[row_idx, col_idx] = 0
    # The crossed-over cell's EAF moves with its z/se (ADR 0036) -- the whole
    # point of the fold is that this is one real observation, not two. Through
    # the plane, so the cell is coded against the Dense Component's own
    # baseline and its exception table moves with it. The Dense Component has
    # already been completed at this point, so its plan carries reference EAF;
    # the fold clears `imputed` above, which is what stops the panel value from
    # standing in for the observation now being written.
    if "eaf" in root:
        DenseEafPlane.open(root, dense_encoding).patch(row_idx, col_idx, eaf_vals)
    return n_reclaimed


def _chunk_shape(manifest: StoreManifest) -> tuple[int, int]:
    cs = manifest.provenance.get("hybrid", {}).get("chunk_shape")
    if cs and len(cs) == 2:
        return (int(cs[0]), int(cs[1]))
    from opengwasdb.layouts.dense.constants import DEFAULT_CHUNK_SHAPE

    return DEFAULT_CHUNK_SHAPE


def _read_analyses(dense_dir: Path) -> list[Analysis]:
    return sorted(
        read_analysis_records(dense_dir / "analyses.tsv"), key=lambda a: int(a.analysis_index)
    )


def _write_completed_manifest(
    staged: StagedRelease,
    src_manifest: StoreManifest,
    release_id: str,
    n_variants: int,
    n_analyses: int,
    n_panel: int,
    n_off_panel: int,
    n_overflow: int,
    n_imputed: int,
) -> None:
    manifest = StoreManifest(
        # Preserved with the format version below: a completed release is
        # written into its source's arrays, so it is in its source's encoding.
        encoding=src_manifest.encoding,
        store_id=src_manifest.store_id,
        release_id=release_id,
        # Preserved, not re-stamped -- see ADR 0038 §4 and the dense path,
        # which checked it writable before any work began.
        format_version=src_manifest.format_version,
        primary_layout=PrimaryStorageLayout.HYBRID,
        association_coverage=AssociationCoverage.FULL,
        completion_state=CompletionState.REFERENCE_COMPLETED,
        reference_assembly=src_manifest.reference_assembly,
        created_at=datetime.now(UTC).isoformat(),
        provenance={
            **src_manifest.provenance,
            "source_release_id": src_manifest.release_id,
            "hybrid": {
                **src_manifest.provenance.get("hybrid", {}),
                "dense_component": DENSE_SUBDIR,
                "n_panel": n_panel,
                "n_off_panel": n_off_panel,
                "n_overflow_associations": n_overflow,
                "compressor": DEFAULT_COMPRESSOR,
            },
            "n_variants": n_variants,
            "n_analyses": n_analyses,
            "completion": {
                "component_completed": "dense",
                "overflow_completion_state": "observed_only",
                "n_imputed_dense": n_imputed,
            },
        },
    )
    staged.write_manifest(manifest)


def _remapped_overflow_baseline(
    source_path: Path,
    vi_to_record: Mapping[int, Any],
    new_shared_index: Mapping[str, int],
    n_shared: int,
) -> np.ndarray | None:
    """The source overflow's `eaf_baseline`, moved onto the completed axis.

    A variant with no overflow association after the remap keeps a NaN
    baseline: it has no cell to code against one.
    """
    group = zarr.open_group(str(Path(source_path) / RAGGED_ZARR_PATH), mode="r")
    if EAF_BASELINE not in group:
        return None
    src_baseline = np.asarray(group[EAF_BASELINE][:], dtype=np.float32)
    out = np.full(int(n_shared), np.nan, dtype=np.float32)
    for src_index, record in vi_to_record.items():
        new_index = new_shared_index.get(record.alid)
        if new_index is not None and 0 <= int(src_index) < len(src_baseline):
            out[new_index] = src_baseline[int(src_index)]
    return out
