"""Two-pass Dense Observed-Only writer from GWAS-VCF manifests with inline liftover.

Association streaming and the union-variant pass go through a ``SourceReader``
resolved from each row's ``source_reader_capability`` (issue #20) rather than
importing ``opengwasdb.build.vcf_source`` directly -- GWAS-VCF remains the only
implementation, so nothing about the build changes, but no bcftools-specific
assumption is left in this module.
"""

from __future__ import annotations

import csv
import logging
import multiprocessing
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs import Blosc

from opengwasdb.build.eaf_orientation import (
    DEFAULT_SAMPLE_SITES,
    EafOrientationReport,
    apply_orientation_evidence,
    sample_column_rows,
    site_hashes,
    verify_eaf_orientation,
)
from opengwasdb.build.liftover import LiftoverFailureError, build_liftover_lookup, normalise_build
from opengwasdb.encoding import (
    EafExceptionBuilder,
    EafMeasurements,
    EncodingMeasurements,
    StoreCodec,
    StoreEncoding,
    ZOverflowBuilder,
    eaf_baseline_from_grid,
    measure_eaf_sample,
    optimise_dense_se,
    positions_row_band,
    write_eaf_baseline,
)
from opengwasdb.index import initialise_schema, set_metadata
from opengwasdb.layouts.dense.build import (
    DenseBuildResult,
    add_hit_counts,
    write_analyses_tsv,
)
from opengwasdb.layouts.dense.constants import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_COMPRESSOR,
    DEFAULT_DTYPE,
    TOP_HIT_THRESHOLDS,
    dense_index_metadata,
)
from opengwasdb.layouts.dense.top_hits import (
    write_top_hit_indexes_for_store,
    z_critical,
)
from opengwasdb.model.analyses import Analysis, PassthroughMetadata
from opengwasdb.model.enums import (
    AssociationCoverage,
    CompletionState,
    EafScope,
    OriginalSdMethod,
    PrimaryStorageLayout,
    StoredEffectScale,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY
from opengwasdb.readers.registry import resolve_reader
from opengwasdb.store.open import CURRENT_FORMAT_VERSION, OpenGWASDBStore, StagedRelease
from opengwasdb.variants import CanonicalVariant, write_variant_axis
from opengwasdb.variants.normalise import chromosome_sort_key

log = logging.getLogger(__name__)

# One compressor for every dense statistic array (z/se/eaf), so a new array
# cannot quietly ship with different settings from the ones beside it.
_DENSE_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

__all__ = ["build_dense_from_vcf_manifest", "LiftoverFailureError"]


@dataclass(frozen=True)
class _ManifestRow:
    trait_id: str
    file_path: str
    trait_name: str
    n: int
    stored_effect_scale: str
    se_divisor: float  # divides original_se to standardise to SD units (issue #18); 1.0 = no-op
    source_reader_capability: str  # resolves to a SourceReader (issue #20); GWAS_VCF_CAPABILITY
    # when the manifest omits the column -- the only format this builder has ever supported.
    source_assembly: str  # normalised "hg19"/"hg38" (issue #85); "hg19" when the manifest omits
    # the column -- every source this builder read before GWAS-SSF (#84) was hg19 GWAS-VCF.
    original_sd: str  # raw manifest value ("" when the sd_method tier carries no magnitude);
    # not part of PassthroughMetadata because this builder *uses* it (se_divisor), rather
    # than only copying it.
    assigned_ancestry: str  # optional manifest column (issue #22); "" when the manifest omits it
    # or carries no assignment for this row -- e.g. a Catalogue subset's kept rows (already
    # filtered to one target ancestry) stamp this in verbatim; a bare manifest has no column.
    # Shared-core analyses.tsv columns the manifest supplies and this builder only copies
    # (issues #86, #83) -- sample-size interpretation and counts, Original Effect Scale,
    # ancestry-assignment method and proportions, Attribution. Blank when absent; the shared
    # builder must not infer their meaning. One type, shared with Ragged, so a column added
    # to the contract reaches both layouts.
    metadata: PassthroughMetadata = field(default_factory=PassthroughMetadata)
    # Trait-ontology columns (ADR 0034, issue #68): optional manifest columns, resolved
    # directly rather than via a phenotype_id/phenotype_label intermediary. "" when the
    # manifest omits them -- never fabricated. Not passthrough-shared: Ragged resolves its
    # own (a gene ID, ADR 0035) rather than copying a manifest column.
    trait_ontology_id: str = ""
    trait_ontology_label: str = ""


def _manifest_row_to_analysis(row: _ManifestRow) -> Analysis:
    """A manifest row's Analytical + Attribution Metadata (issue #22, ADR
    0034), shared by the dense and hybrid manifest builders.

    Optional fields are passed through verbatim and remain blank when absent.
    In particular, the shared builder cannot infer how ancestry was assigned:
    the manifest producer owns that fact (issue #86).
    """
    return row.metadata.applied_to(
        Analysis(
            analysis_id=row.trait_id,
            analysis_label=row.trait_name,
            trait_ontology_id=row.trait_ontology_id,
            trait_ontology_label=row.trait_ontology_label,
            stored_effect_scale=row.stored_effect_scale,
            assigned_ancestry=row.assigned_ancestry,
            sample_size=str(row.n) if row.n else "",
            original_sd=row.original_sd,
        )
    )


# original_sd_method tiers that carry an actual phenotype-SD magnitude to divide
# by (ADR-0029 methods 2-5). declared_standardised implies sd=1 (no magnitude
# recorded); binary_trait is not SD-scale at all -- neither rescales.
_SD_RESCALE_METHODS = frozenset(
    {
        OriginalSdMethod.SOURCE_PROVIDED,
        OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF,
        OriginalSdMethod.ESTIMATED_FROM_REFERENCE_MAF,
        OriginalSdMethod.ESTIMATED_FROM_BETA_DISTRIBUTION,
    }
)


# Pass 2 fork-safe lookup + spill dir, set in the parent immediately before the
# process pool is created. Forked workers (fork start method — see _fork_pool())
# inherit these; because the lookup is a pair of numpy arrays (one contiguous C
# buffer with a single object refcount) rather than Python dicts, a worker reading
# it via searchsorted never touches per-element refcounts, so it does not
# refcount-COW ~n_variants dict pages per worker (issue 043 item 2).
#
# Why disk-spill and not return arrays: an earlier design returned each file's
# result over IPC. At genome-wide scale (~9.85M rows/file × thousands of files)
# that pipe traffic deadlocked the pool. Workers write a compact per-file .npz to
# _pass2_spill_dir and return only col_idx, so no large object crosses the pipe.
_pass2_keys_sorted: np.ndarray | None = None  # sorted 'S' byte keys
_pass2_rows_sorted: np.ndarray | None = None  # int32 row per key, same order
_pass2_spill_dir: Path | None = None

# Top hits are harvested inline during Pass 2 rather than by a post-hoc scan of
# the full matrix (which had to reload ~200 GB of float32 and compute a p-value
# for every finite cell). Each column emits only the cells clearing the loosest
# threshold's |z| cutoff; the band-write phase accumulates them for the index.
_TOP_HIT_Z_CRIT = z_critical(max(TOP_HIT_THRESHOLDS))


def _fork_pool(n_workers: int) -> ProcessPoolExecutor:
    """A ProcessPoolExecutor pinned to fork start — required for _pass2_worker to
    inherit the numpy lookup arrays without re-pickling them per task. Only
    correct on platforms with fork (Linux)."""
    fork_ctx = multiprocessing.get_context("fork")
    return ProcessPoolExecutor(max_workers=n_workers, mp_context=fork_ctx)


def _encode_variant_keys(chrom: object, pos: object, ref: object, alt: object) -> np.ndarray:
    """Encode variant fields as ``chrom:pos:ref:alt`` byte-string keys, vectorised.

    Used identically for the parent's sorted key table and each worker's query
    keys, so the two encodings match exactly. ``pos`` int->bytes via numpy astype
    (``np.int64(10).astype('S') == b'10'``) avoids per-element Python string work.
    """
    chrom_b = np.asarray(chrom, dtype="S")
    pos_b = np.asarray(pos, dtype=np.int64).astype("S")
    ref_b = np.asarray(ref, dtype="S")
    alt_b = np.asarray(alt, dtype="S")
    key = np.char.add(chrom_b, b":")
    key = np.char.add(key, pos_b)
    key = np.char.add(key, b":")
    key = np.char.add(key, ref_b)
    key = np.char.add(key, b":")
    key = np.char.add(key, alt_b)
    return key


def _build_variant_key_index(
    source_lookup: dict[tuple[str, int, str, str], str],
    variant_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the two Pass 2 dicts into a fork-safe sorted numpy lookup.

    Returns ``(keys_sorted, rows_sorted)``: a sorted array of byte keys for every
    source variant that maps to a stored row, and the matching int32 row indices.
    Workers binary-search this instead of chaining two Python dicts.
    """
    chroms: list[str] = []
    poss: list[int] = []
    refs: list[str] = []
    alts: list[str] = []
    rows: list[int] = []
    for (chrom, pos, ref, alt), alid in source_lookup.items():
        row = variant_index.get(alid)
        if row is None:
            continue
        chroms.append(chrom)
        poss.append(pos)
        refs.append(ref)
        alts.append(alt)
        rows.append(row)
    keys = _encode_variant_keys(chroms, poss, refs, alts)
    rows_arr = np.array(rows, dtype=np.int32)
    order = np.argsort(keys, kind="stable")
    return keys[order], rows_arr[order]


# Streamed in fixed-size batches so a worker never materialises a whole
# (genome-wide) VCF as Python lists at once — peak per-worker memory is one batch
# of columns plus the accumulated matched rows, not the entire file (issue 043).
_RESOLVE_BATCH = 250_000


def _match_batch(
    chroms: list[str],
    poss: list[int],
    refs: list[str],
    alts: list[str],
    zs: list[float],
    ses: list[float],
    eafs: list[float],
    keys_sorted: np.ndarray,
    rows_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve one batch of associations to (rows, z, se, eaf) for cells whose
    variant is present in the panel (exact key match). Order-preserving.
    `eaf` is NaN for associations whose source reported none (ADR 0036)."""
    if len(keys_sorted) == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_f, empty_f, empty_f
    query = _encode_variant_keys(chroms, poss, refs, alts)
    idx = np.searchsorted(keys_sorted, query)
    idx_clip = np.minimum(idx, len(keys_sorted) - 1)
    matched = keys_sorted[idx_clip] == query
    rows = rows_sorted[idx_clip[matched]].astype(np.int64)
    z_arr = np.array(zs, dtype=np.float32)[matched]
    se_arr = np.array(ses, dtype=np.float32)[matched]
    eaf_arr = np.array(eafs, dtype=np.float32)[matched]
    return rows, z_arr, se_arr, eaf_arr


def _apply_se_divisor(se: np.ndarray, se_divisor: float) -> np.ndarray:
    """Continuous-trait phenotype-SD standardisation (issue #18):
    ``stored_se = original_se / sd``. Shared by the dense and hybrid builders,
    both of which resolve an association stream to ``(index, z, se)`` and need
    the same no-op-when-1.0 divide applied to ``se`` before spilling."""
    if se_divisor == 1.0:
        return se
    return se / np.float32(se_divisor)


def _resolve_column(
    file_path: str,
    keys_sorted: np.ndarray,
    rows_sorted: np.ndarray,
    se_divisor: float = 1.0,
    *,
    capability: str = GWAS_VCF_CAPABILITY,
    stored_effect_scale: str = StoredEffectScale.SD.value,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream one source file, resolve each association's variant to a row via
    the sorted key lookup, and return deduped
    ``(rows int64, z f32, se f32, eaf f32)`` for the column. ``eaf`` is NaN
    where the source reports no frequency (ADR 0036).

    The stream is processed in ``_RESOLVE_BATCH``-sized batches (vectorised
    searchsorted per batch), so worker peak memory is bounded by one batch rather
    than the whole file. Batches are matched in stream order and concatenated, so
    the final last-wins dedup by row is identical to processing the whole file at
    once: when two source variants lift to the same row, the later stream
    occurrence wins, making the scattered matrix cell deterministic.

    ``capability`` resolves a ``SourceReader`` (issue #20) rather than this
    module streaming a VCF itself -- ``stored_effect_scale`` is required to
    construct one but unused past construction here (it is a Pass-2 concern
    only for readers that attach it to each yielded association).

    ``se_divisor`` divides the returned ``se`` (continuous-trait phenotype-SD
    standardisation, issue #18: ``stored_se = original_se / sd``). ``z`` is left
    untouched -- ``z = beta/se`` is invariant to dividing both by the same
    constant, so only ``se`` needs rescaling. Defaults to 1.0 (no-op) for
    binary-trait analyses and callers that pre-date issue #18.
    """
    reader = resolve_reader(capability, file_path, StoredEffectScale(stored_effect_scale))
    rows_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    se_parts: list[np.ndarray] = []
    eaf_parts: list[np.ndarray] = []
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
        r, z, se, eaf = _match_batch(
            chroms, poss, refs, alts, zs, ses, eafs, keys_sorted, rows_sorted
        )
        rows_parts.append(r)
        z_parts.append(z)
        se_parts.append(se)
        eaf_parts.append(eaf)
        chroms.clear()
        poss.clear()
        refs.clear()
        alts.clear()
        zs.clear()
        ses.clear()
        eafs.clear()

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

    if not rows_parts:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_f, empty_f, empty_f

    rows = np.concatenate(rows_parts)
    z_arr = np.concatenate(z_parts)
    se_arr = np.concatenate(se_parts)
    eaf_arr = np.concatenate(eaf_parts)

    if len(rows):
        # last-wins dedup by row: unique on the reversed rows returns the first
        # index in reversed order == the last occurrence in original order.
        _, first_in_rev = np.unique(rows[::-1], return_index=True)
        keep = np.sort(len(rows) - 1 - first_in_rev)
        rows, z_arr, se_arr, eaf_arr = rows[keep], z_arr[keep], se_arr[keep], eaf_arr[keep]
    se_arr = _apply_se_divisor(se_arr, se_divisor)
    return rows, z_arr, se_arr, eaf_arr


def _spill_column(
    spill_dir: Path,
    col_idx: int,
    rows: np.ndarray,
    z: np.ndarray,
    se: np.ndarray,
    eaf: np.ndarray,
) -> None:
    """Atomically spill one resolved column to ``{spill_dir}/{col_idx}.npz``
    (temp-then-rename; both names end in .npz because np.savez appends that suffix
    unless already present).

    Top hits are NOT harvested here — they are harvested during the band-write
    phase from the *stored* float16 values, so the index matches exactly what a
    query reads back from the ``z`` array (issue 046)."""
    final = spill_dir / f"{col_idx}.npz"
    tmp = spill_dir / f"{col_idx}.tmp.npz"
    np.savez(tmp, rows=rows, z=z, se=se, eaf=eaf)
    tmp.replace(final)


def _pass2_worker(task: tuple[int, str, float, str, str]) -> int:
    """Resolve one column against the fork-inherited numpy lookup and spill it.
    Returns col_idx only — the compact result stays on disk, never in a pipe."""
    assert _pass2_keys_sorted is not None
    assert _pass2_rows_sorted is not None
    assert _pass2_spill_dir is not None
    col_idx, file_path, se_divisor, capability, stored_effect_scale = task
    rows, z, se, eaf = _resolve_column(
        file_path,
        _pass2_keys_sorted,
        _pass2_rows_sorted,
        se_divisor,
        capability=capability,
        stored_effect_scale=stored_effect_scale,
    )
    _spill_column(_pass2_spill_dir, col_idx, rows, z, se, eaf)
    return col_idx


def _pass2_worker_tasks(
    manifest_rows: Sequence[_ManifestRow],
    analysis_index: Mapping[str, int],
) -> list[tuple[int, str, float, str, str]]:
    """One fork-pool task per manifest row, in manifest order: the column
    index from ``analysis_index`` followed by the four source fields
    ``_pass2_worker`` needs to resolve the row's file.

    The Dense and Hybrid builders submit identical task tuples to their own
    ``_pass2_worker``s, so the list is built here once rather than at each
    call site.
    """
    return [
        (
            analysis_index[row.trait_id],
            row.file_path,
            row.se_divisor,
            row.source_reader_capability,
            row.stored_effect_scale,
        )
        for row in manifest_rows
    ]


def _log_progress(
    label: str, completed: int, total: int, start_time: float, extra: str, every: int
) -> None:
    if completed % every != 0 and completed != total:
        return
    elapsed = time.monotonic() - start_time
    eta = (elapsed / completed) * (total - completed) if completed else 0.0
    log.info(
        "%s: %d/%d done (%s) — elapsed %s, ETA %s",
        label,
        completed,
        total,
        extra,
        _fmt_duration(elapsed),
        _fmt_duration(eta),
    )


def _collect_manifest_variant_sites(
    manifest_rows: list[_ManifestRow],
) -> tuple[dict[str, set[tuple[str, int, str, str]]], dict[tuple[str, int, str, str], str]]:
    """Pass 1: one serial read of every manifest source for its variant sites.

    Returns the ``tuples_by_assembly`` union and ``rsid_by_site`` (first named
    rsid wins, issue #109) so the serial source scan happens exactly once and
    the rsids it saw ride along on that same read rather than a second,
    genome-scale re-read. Streaming order is preserved within each file;
    progress is reported every 250 files.
    """
    tuples_by_assembly: dict[str, set[tuple[str, int, str, str]]] = {}
    rsid_by_site: dict[tuple[str, int, str, str], str] = {}
    log.info("Pass 1: collecting source variants from %d files (serial)", len(manifest_rows))
    t0 = time.monotonic()
    for i, row in enumerate(manifest_rows):
        reader = resolve_reader(
            row.source_reader_capability, row.file_path, StoredEffectScale(row.stored_effect_scale)
        )
        sites = tuples_by_assembly.setdefault(row.source_assembly, set())
        for variant in reader.stream_variants():
            sites.add(variant.site)
            if variant.rsid:
                rsid_by_site.setdefault(variant.site, variant.rsid)
        n_total = sum(len(t) for t in tuples_by_assembly.values())
        _log_progress(
            "Pass 1", i + 1, len(manifest_rows), t0, f"{n_total} unique variants so far", every=250
        )
    return tuples_by_assembly, rsid_by_site


def _lift_manifest_variants(
    manifest_rows: list[_ManifestRow],
    *,
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
) -> tuple[dict[tuple[str, int, str, str], str], dict[str, str]]:
    """Resolve every manifest row's union of source variants to hg38 ALIDs
    (issue #85; the dense and hybrid builders' shared Pass 1).

    Each row declares its own ``source_assembly`` (`_read_manifest`). Rows
    already on hg38 -- a harmonised GWAS-SSF source, say -- map straight to
    their ALID with no liftover: running an already-hg38 coordinate through
    the hg19->hg38 chain a second time silently shifts it to the wrong
    position (the issue #85 bug), and pyliftover has no way to detect that
    from the coordinate alone. Rows declaring hg19 (the only other build this
    package knows) go through one shared ``LiftOver`` object for that group,
    same as when GWAS-VCF was the only source this builder ever saw.
    ``liftover_failure_threshold`` therefore applies only to the hg19 group's
    own failure rate, not diluted by (or inflated against) hg38 rows that
    were never at risk of failing.

    Returns one merged ``{(chrom, pos, ref, alt): hg38_alid}`` lookup -- the
    shape the fork-safe Pass 2 key index (`_build_variant_key_index` /
    `_build_routing_index`) is already built from, so nothing downstream of
    this function changes -- plus ``{hg38_alid: rsid}`` for the rows whose
    source named one (issue #109). The rsid map rides along on this pass
    rather than a second one: Pass 1 is already the serial read of every
    source file, and re-reading them just to recover identifiers the first
    read saw would double the most expensive part of a genome-scale build.
    Where two sources name one variant differently, the first wins; where a
    source names none, the ALID is simply absent from the map and its row's
    rsid column is written blank. A raw tuple present in *both* groups is not a
    same-locus dedup the way a tuple shared by two same-assembly files is:
    the hg38 group's tuple is a literal coordinate, the hg19 group's
    identical-looking tuple is a *pre-lift* coordinate bound for a different
    hg38 position, so the two groups agreeing on a raw tuple means two
    physically different loci coincidentally share one string, not one real
    variant reported twice. Binding both to a single stored row would
    misattribute one row's association to the other's variant -- exactly the
    kind of silent corruption this fix exists to remove -- so any such tuple
    is dropped from both groups (never guessed) before returning.
    """
    tuples_by_assembly, rsid_by_site = _collect_manifest_variant_sites(manifest_rows)

    passthrough_lookup: dict[tuple[str, int, str, str], str] = {}
    passthrough = tuples_by_assembly.pop("hg38", set())
    if passthrough:
        log.info("%d variants already GRCh38 -- no liftover needed", len(passthrough))
        for chrom, pos, ref, alt in passthrough:
            a1, a2 = sorted((ref, alt))
            passthrough_lookup[(chrom, pos, ref, alt)] = f"{chrom}:{pos}:{a1}:{a2}"

    lifted_lookup: dict[tuple[str, int, str, str], str] = {}
    hg19_tuples = tuples_by_assembly.pop("hg19", set())
    if hg19_tuples:
        log.info("Running liftover hg19 → hg38 (%d variants)", len(hg19_tuples))
        lifted_lookup = build_liftover_lookup(
            hg19_tuples,
            from_build="hg19",
            to_build="hg38",
            failure_threshold=liftover_failure_threshold,
            chain_file=chain_file,
        )
        log.info("Liftover complete: %d variants mapped", len(lifted_lookup))

    assert not tuples_by_assembly, f"unhandled source_assembly values: {sorted(tuples_by_assembly)}"

    ambiguous = passthrough_lookup.keys() & lifted_lookup.keys()
    if ambiguous:
        log.warning(
            "%d raw variant tuple(s) declared both hg38 and hg19 in this manifest "
            "(same chrom/pos/ref/alt string, two different builds -> two different "
            "physical loci) -- dropped from both rather than guessed which one owns "
            "the stored row",
            len(ambiguous),
        )
        for key in ambiguous:
            del passthrough_lookup[key]
            del lifted_lookup[key]

    source_lookup = {**passthrough_lookup, **lifted_lookup}
    # Re-key the identifiers onto the hg38 ALIDs the store's rows are keyed by.
    # Several source sites can lift onto one ALID; the first named wins, the
    # same first-wins rule the Ragged builders already apply.
    rsid_by_alid: dict[str, str] = {}
    for site, alid in source_lookup.items():
        rsid = rsid_by_site.get(site)
        if rsid:
            rsid_by_alid.setdefault(alid, rsid)
    return source_lookup, rsid_by_alid


def build_dense_from_vcf_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
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
) -> DenseBuildResult:
    """Build a Dense Observed-Only Store from a manifest of GWAS-VCF files.

    A thin orchestrator over the private phases below (issue #130): the build
    is staged atomically at ``output_path``, its two streaming passes never
    materialise the full association matrix (issue 043), and every phase runs
    before the store is committed. Keyword semantics live with the phase that
    consumes each -- the manifest columns with `_read_manifest`; ``chain_file``
    and ``liftover_failure_threshold`` with the hg19→hg38 lift (issue #85);
    ``eaf_reference``/``eaf_reference_ancestry``/``allow_unverified_eaf``
    with EAF orientation verification (issue #115, ADR 0037 §6).
    """
    manifest_rows = _read_manifest(manifest_path)
    if not manifest_rows:
        raise ValueError(f"manifest {manifest_path} contains no rows")

    out = Path(output_path)
    with OpenGWASDBStore.staging(out, overwrite=overwrite) as staged:
        prepared = _prepare_axis(
            staged, out, manifest_rows, chain_file,
            liftover_failure_threshold, chunk_shape,
        )
        # Phases 5-7 run inside one spill-dir lifetime: every spill is removed
        # even when a phase fails, keeping the staged release atomic.
        eaf_report, encoded = _spill_verify_and_encode(
            staged, out, manifest_rows, prepared,
            n_workers, chunk_shape, dtype,
            eaf_reference, eaf_reference_ancestry, allow_unverified_eaf,
        )
        # Phase 8: top-hit indexes, manifest and analyses.tsv metadata.
        _finalize_store(
            staged, prepared, encoded, eaf_report,
            store_id, release_id, chain_file, chunk_shape, dtype,
            allow_unverified_eaf,
        )
        log.info(
            "Build complete: %d variants × %d analyses",
            len(prepared.axis.alids), len(prepared.axis.analyses),
        )

    return DenseBuildResult(
        output_path=out,
        n_variants=len(prepared.axis.alids),
        n_analyses=len(prepared.axis.analyses),
    )


@dataclass(frozen=True)
class _AxisMetadata:
    """The store's axis: sorted hg38 ALIDs, the row maps, and Analysis records.

    ``analyses`` are the manifest's Analytical Metadata (issue #17, #86) --
    derived from each row, never from a VCF header.
    """

    alids: list[str]
    analysis_index: dict[str, int]
    analyses: list[Analysis]


@dataclass(frozen=True)
class _HitCandidates:
    """The cells the band-write's z pass harvested above the loosest threshold.

    Harvested from the *stored* values so the top-hit index matches exactly
    what a query reads back from the ``z`` array (issue 046)."""

    rows: np.ndarray
    cols: np.ndarray
    z: np.ndarray
    se: np.ndarray


@dataclass(frozen=True)
class _EncodedBands:
    """Phase-7 outcome the index and metadata phases consume."""

    encoding: StoreEncoding
    hits: _HitCandidates
    column_has_eaf: np.ndarray


@dataclass(frozen=True)
class _PreparedBuild:
    """What the preparation phase hands the spill phases: the store's axis and
    the fork-safe Pass 2 lookup arrays resolved against it.

    The parent-side ``source_lookup``/``variant_index`` dicts are dropped when
    the preparation phase returns, before any worker forks (issue 043).
    """

    axis: _AxisMetadata
    keys_sorted: np.ndarray
    rows_sorted: np.ndarray


def _axis_metadata(
    source_lookup: Mapping[tuple[str, int, str, str], str],
    manifest_rows: Sequence[_ManifestRow],
) -> tuple[_AxisMetadata, dict[str, int]]:
    """Resolve the store's axis metadata.

    Sorting hg38 ALIDs by (chromosome, position, a1, a2) gives the variant
    axis a stable, position-ordered row index; the manifest's own row order
    fixes the analysis column order. The variant index is returned alongside
    the axis so the caller can free it (it is the one ~n_variants-sized dict)
    once the Pass 2 lookup arrays are built.
    """
    hg38_alids = sorted(set(source_lookup.values()), key=_alid_sort_key)
    variant_index: dict[str, int] = {alid: i for i, alid in enumerate(hg38_alids)}
    analysis_index: dict[str, int] = {row.trait_id: i for i, row in enumerate(manifest_rows)}
    analyses: list[Analysis] = [_manifest_row_to_analysis(row) for row in manifest_rows]
    return (
        _AxisMetadata(
            alids=hg38_alids,
            analysis_index=analysis_index,
            analyses=analyses,
        ),
        variant_index,
    )


def _source_alids_by_alid(
    source_lookup: Mapping[tuple[str, int, str, str], str],
    hg38_alids: Sequence[str],
) -> list[str | None]:
    """The source-build canonical ALID each stored row was resolved from.

    Provenance (the variant axis's ``source_alid`` column): the hg19 tuple's
    own canonical ALID, or the passed-through hg38 tuple's, so a reader sees
    which build coordinate a row's associations came from. A single hg38 ALID
    can be the target of several source variants (a liftover collapse); when
    they disagree on the origin the row is ambiguous, so it is left blank --
    never guessed (a guess would silently misattribute provenance).
    """
    hg38_to_source: dict[str, str | None] = {}
    for (chrom, pos, ref, alt), hg38_alid in source_lookup.items():
        a1, a2 = sorted((ref, alt))
        origin = f"{chrom}:{pos}:{a1}:{a2}"
        if hg38_alid in hg38_to_source:
            if hg38_to_source[hg38_alid] != origin:
                hg38_to_source[hg38_alid] = None  # collision → ambiguous
        else:
            hg38_to_source[hg38_alid] = origin
    return [hg38_to_source.get(alid) for alid in hg38_alids]


def _write_axis_and_index(
    staged: StagedRelease,
    source_lookup: Mapping[tuple[str, int, str, str], str],
    axis: _AxisMetadata,
    rsid_by_alid: Mapping[str, str],
    chunk_shape: tuple[int, int],
) -> None:
    """Write the SQLite index and the tabix variant axis.

    The variant axis carries one canonical variant per stored row with its
    rsid (first named wins) and the source-build provenance each row resolved
    from (``_source_alids_by_alid`` -- collisions stay blank, never guessed).
    """
    _write_index(staged, axis.alids, axis.analyses, chunk_shape)
    canonical_variants = [
        CanonicalVariant(
            chromosome=chrom,
            position=int(pos_str),
            effect_allele=a1,
            other_allele=a2,
        )
        for alid in axis.alids
        for chrom, pos_str, a1, a2 in [alid.split(":")]
    ]
    source_alids = _source_alids_by_alid(source_lookup, axis.alids)
    write_variant_axis(staged.path, canonical_variants, rsid_by_alid, source_alids)


def _build_pass2_lookup(
    source_lookup: dict[tuple[str, int, str, str], str],
    variant_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the Pass 2 dicts into the sorted numpy lookup workers search.

    Fork-safe by construction: the returned arrays are a contiguous C buffer
    and an int32 row array, so a worker reading them via searchsorted never
    touches per-element refcounts (issue 043 item 2). The caller frees the
    dicts before the pool is created.
    """
    keys_sorted, rows_sorted = _build_variant_key_index(source_lookup, variant_index)
    max_key_len = keys_sorted.dtype.itemsize if len(keys_sorted) else 0
    log.info(
        "Pass 2 lookup: %d variant keys, max key length %d bytes",
        len(keys_sorted),
        max_key_len,
    )
    return keys_sorted, rows_sorted


def _prepare_axis(
    staged: StagedRelease,
    out: Path,
    manifest_rows: list[_ManifestRow],
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
    chunk_shape: tuple[int, int],
) -> _PreparedBuild:
    """Phases 1-4: union/liftover, axis metadata, index + axis, Pass 2 lookup.

    hg19-declared rows are lifted to GRCh38 once (issue #85), the axis
    metadata and provenance-carrying variant axis are written, and the Pass 2
    lookup arrays are composed -- the parent ``source_lookup``/``variant_index``
    dicts then drop out of scope before any worker forks (issue 043).
    """
    source_lookup, rsid_by_alid = _lift_manifest_variants(
        manifest_rows,
        chain_file=chain_file,
        liftover_failure_threshold=liftover_failure_threshold,
    )
    axis, variant_index = _axis_metadata(source_lookup, manifest_rows)
    _write_axis_and_index(staged, source_lookup, axis, rsid_by_alid, chunk_shape)
    keys_sorted, rows_sorted = _build_pass2_lookup(source_lookup, variant_index)
    return _PreparedBuild(axis=axis, keys_sorted=keys_sorted, rows_sorted=rows_sorted)


def _spill_columns_serial(
    manifest_rows: Sequence[_ManifestRow],
    analysis_index: Mapping[str, int],
    spill_dir: Path,
    keys_sorted: np.ndarray,
    rows_sorted: np.ndarray,
    axis: _AxisMetadata,
    pass2_start: float,
) -> None:
    """Pass 2 for ``n_workers <= 1``: resolve and spill each column in this
    process, one Analysis at a time, so peak memory is one resolved column."""
    log.info(
        "Pass 2: resolving %d analyses × %d variants (n_workers=1)",
        len(axis.analyses),
        len(axis.alids),
    )
    for i, row in enumerate(manifest_rows):
        col_idx = analysis_index[row.trait_id]
        rows, z, se, eaf = _resolve_column(
            row.file_path,
            keys_sorted,
            rows_sorted,
            row.se_divisor,
            capability=row.source_reader_capability,
            stored_effect_scale=row.stored_effect_scale,
        )
        _spill_column(spill_dir, col_idx, rows, z, se, eaf)
        _log_progress(
            "Pass 2", i + 1, len(axis.analyses), pass2_start, f"last: {row.trait_id}", every=25
        )


def _spill_columns_parallel(
    manifest_rows: Sequence[_ManifestRow],
    analysis_index: Mapping[str, int],
    spill_dir: Path,
    keys_sorted: np.ndarray,
    rows_sorted: np.ndarray,
    axis: _AxisMetadata,
    n_workers: int,
    pass2_start: float,
) -> None:
    """Pass 2 for ``n_workers > 1``: resolve each column in a fork pool.

    Workers inherit the numpy lookup arrays at fork (see the module note on
    ``_pass2_keys_sorted``) and write only their own ``.npz``, so nothing large
    crosses the pipe; the parent waits on completion. The globals are reset
    whether or not every task succeeded.
    """
    log.info(
        "Pass 2: resolving %d analyses × %d variants (n_workers=%d)",
        len(axis.analyses),
        len(axis.alids),
        n_workers,
    )
    global _pass2_keys_sorted, _pass2_rows_sorted, _pass2_spill_dir
    _pass2_keys_sorted = keys_sorted
    _pass2_rows_sorted = rows_sorted
    _pass2_spill_dir = spill_dir
    try:
        with _fork_pool(n_workers) as pool:
            id_by_col = {analysis_index[row.trait_id]: row.trait_id for row in manifest_rows}
            tasks = _pass2_worker_tasks(manifest_rows, analysis_index)
            futures = [pool.submit(_pass2_worker, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures)):
                col_idx = fut.result()
                _log_progress(
                    "Pass 2",
                    i + 1,
                    len(axis.analyses),
                    pass2_start,
                    f"last: {id_by_col[col_idx]}",
                    every=25,
                )
    finally:
        _pass2_keys_sorted = None
        _pass2_rows_sorted = None
        _pass2_spill_dir = None


def _survey_and_verify_eaf(
    spill_dir: Path,
    manifest_rows: Sequence[_ManifestRow],
    axis: _AxisMetadata,
    *,
    eaf_reference: str | Path | None,
    eaf_reference_ancestry: str | None,
    allow_unverified_eaf: bool,
) -> tuple[EafSpillSurvey, EafOrientationReport]:
    """Sample each Analysis's stored frequencies and verify their orientation.

    The check (issue #115, ADR 0037 §6) runs off the Pass 2 spills -- the exact
    values about to be stored -- and before any statistic array is written, so
    a build that would store a frequency column against the wrong allele fails
    here rather than after an hour of band-writing. The survey also measures
    the frequency spread the encoding plan reads, so both answer in one pass.
    ``allow_unverified_eaf`` records Analyses the supplied reference could not
    verify in the store's provenance instead of rejecting them.
    """
    id_by_col = {axis.analysis_index[row.trait_id]: row.trait_id for row in manifest_rows}
    eaf_survey = survey_eaf_spills(spill_dir, id_by_col, axis.alids, site_hashes(axis.alids))
    eaf_report = verify_eaf_orientation(
        eaf_survey.observations,
        eaf_reference=eaf_reference,
        eaf_reference_ancestry=eaf_reference_ancestry,
        allow_unverified=allow_unverified_eaf,
    )
    return eaf_survey, eaf_report


def _write_encoded_bands(
    staged: StagedRelease,
    spill_dir: Path,
    eaf_survey: EafSpillSurvey,
    axis: _AxisMetadata,
    chunk_shape: tuple[int, int],
    dtype: str,
    pass2_start: float,
) -> _EncodedBands:
    """Create the statistic arrays and fill them from the spills.

    One encoding plan per build is decided here -- after Pass 2 and the EAF
    survey, because the ``eaf`` rules read the frequencies the sources
    actually carried -- and recorded in manifest.json (ADR 0037, issue #119).
    The z/se planes are then filled in chunk-column bands so peak memory is
    one band (issue 043), the eaf plane is written under the plan, and
    ``optimise_dense_se`` rewrites ``se`` under the plan's codec. The
    top-hit candidates are harvested in the same pass (issue 046).
    """
    n_variants, n_analyses = len(axis.alids), len(axis.analyses)
    encoding = StoreEncoding.decide(
        EncodingMeasurements(
            n_analyses=n_analyses,
            eaf=eaf_survey.measurements(
                n_cells=n_variants * n_analyses, n_variants=n_variants
            ),
        )
    )
    log.info("Encoding plan: %s", encoding.to_manifest())
    effective_chunks = _create_dense_zarr(
        staged, n_variants, n_analyses, chunk_shape, dtype, encoding
    )
    rows, cols, z, se, column_has_eaf = _write_dense_bands(
        staged,
        spill_dir,
        n_variants,
        n_analyses,
        effective_chunks,
        dtype,
        pass2_start,
        encoding,
    )
    encoding = optimise_dense_se(staged.arrays(mode="a"), encoding)
    return _EncodedBands(
        encoding=encoding,
        hits=_HitCandidates(rows=rows, cols=cols, z=z, se=se),
        column_has_eaf=column_has_eaf,
    )


def _spill_verify_and_encode(
    staged: StagedRelease,
    out: Path,
    manifest_rows: Sequence[_ManifestRow],
    prepared: _PreparedBuild,
    n_workers: int,
    chunk_shape: tuple[int, int],
    dtype: str,
    eaf_reference: str | Path | None,
    eaf_reference_ancestry: str | None,
    allow_unverified_eaf: bool,
) -> tuple[EafOrientationReport, _EncodedBands]:
    """Phases 5-7: spill each column, verify EAF, and write the encoded planes.

    Runs inside the spill dir's lifecycle: every spill is removed when this
    returns or raises, so a phase failure leaves nothing behind and the staged
    release stays atomic. ``n_workers`` > 1 resolves through the fork pool;
    EAF orientation is verified against ``eaf_reference`` before any array is
    written (issue #115, ADR 0037 §6).
    """
    spill_dir = Path(
        tempfile.mkdtemp(prefix=f".{out.name}.pass2spill.", dir=staged.path.parent)
    )
    try:
        pass2_start = time.monotonic()
        if n_workers <= 1:
            _spill_columns_serial(
                manifest_rows, prepared.axis.analysis_index, spill_dir,
                prepared.keys_sorted, prepared.rows_sorted, prepared.axis, pass2_start,
            )
        else:
            _spill_columns_parallel(
                manifest_rows, prepared.axis.analysis_index, spill_dir,
                prepared.keys_sorted, prepared.rows_sorted,
                prepared.axis, n_workers, pass2_start,
            )
        eaf_survey, eaf_report = _survey_and_verify_eaf(
            spill_dir,
            manifest_rows,
            prepared.axis,
            eaf_reference=eaf_reference,
            eaf_reference_ancestry=eaf_reference_ancestry,
            allow_unverified_eaf=allow_unverified_eaf,
        )
        encoded = _write_encoded_bands(
            staged,
            spill_dir,
            eaf_survey,
            prepared.axis,
            chunk_shape,
            dtype,
            pass2_start,
        )
    finally:
        shutil.rmtree(spill_dir, ignore_errors=True)
    return eaf_report, encoded


def _finalize_store(
    staged: StagedRelease,
    prepared: _PreparedBuild,
    encoded: _EncodedBands,
    eaf_report: EafOrientationReport,
    store_id: str,
    release_id: str,
    chain_file: str | Path | None,
    chunk_shape: tuple[int, int],
    dtype: str,
    allow_unverified_eaf: bool,
) -> None:
    """Phase 8: write the store's final metadata.

    The manifest names the encoding and the EAF-orientation evidence; the
    top-hit indexes are written from the band-write's stored-value harvest
    (issue 046); analyses.tsv is written last, with each Analysis's
    ``eaf_scope`` stamped from what was actually stored (ADR 0036) and the
    orientation evidence applied (issue #115).
    """
    axis = prepared.axis
    _write_manifest(
        staged,
        store_id,
        release_id,
        len(axis.alids),
        len(axis.analyses),
        chain_file,
        chunk_shape,
        dtype,
        encoding=encoded.encoding,
        eaf_orientation=eaf_report.provenance(allow_unverified=allow_unverified_eaf),
    )
    write_top_hit_indexes_for_store(
        staged.path, encoded.hits.rows, encoded.hits.cols, encoded.hits.z, encoded.hits.se,
        encoded.encoding,
    )
    analyses = apply_orientation_evidence(
        _apply_eaf_scope(axis.analyses, encoded.column_has_eaf), eaf_report
    )
    write_analyses_tsv(staged.path, add_hit_counts(staged.path, analyses))


def _fmt_duration(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _read_manifest(manifest_path: str | Path) -> list[_ManifestRow]:
    """Read the build manifest: the existing ``trait_id``/``file_path``/
    ``trait_name``/``n`` columns -- also the Analysis Catalogue's
    ``BUILD_COLUMNS`` (`opengwasdb.ancestry.catalogue`), so a
    Catalogue-annotated file remains readable here for those four columns --
    plus required ``stored_effect_scale`` (issue #17) and ``original_sd_method``
    (issue #18) columns, and an ``original_sd`` column required only for the
    ``original_sd_method`` tiers that carry an actual SD magnitude.

    Neither column is part of the Catalogue/ancestry manifest shape: ancestry
    assignment runs on allele frequencies alone and may run before a study's
    effect scale or phenotype SD is even resolved, so these stay independent
    build inputs rather than part of one combined schema. A manifest missing a
    required column, carrying an out-of-vocabulary value, declaring
    ``original_sd_method=unavailable``, omitting ``original_sd`` for a tier
    that needs it, or supplying a stray ``original_sd`` for a tier that carries
    no SD magnitude (``declared_standardised``, ``binary_trait``), fails the
    build loudly rather than falling back to the old VCF-header inference or a
    silently assumed ``sd=1`` (issue #18 AC3).

    An optional ``source_reader_capability`` column (issue #20) selects the
    ``SourceReader`` each row is built through; a manifest that omits it (every
    manifest before this change) defaults every row to ``GWAS_VCF_CAPABILITY``,
    the only format this builder has ever supported.

    An optional ``source_assembly`` column (issue #85) declares the genome
    build each row's *source file* is already in -- ``hg19``/``GRCh37`` or
    ``hg38``/``GRCh38`` (aliases per `opengwasdb.build.liftover`). A row
    omitting it defaults to ``hg19``, matching every source this builder read
    before GWAS-SSF (#84): GWAS-VCF is conventionally hg19 here. This is not
    inferred from the source file itself -- a harmonised GWAS-Catalog-SSF
    file is already hg38 and must declare so, or `_lift_manifest_variants`
    would liftover its already-correct coordinates a second time (issue #85).
    An invalid value fails the build loudly, same as ``stored_effect_scale``.

    An optional ``assigned_ancestry`` column (issue #22) carries a row's
    Assigned Ancestry straight into the built store's ``analyses.tsv`` --
    a Catalogue subset's kept rows already have this column (the Catalogue
    is a superset of the build manifest), so ``opengwasdb.ancestry.subset``
    no longer needs a separate post-build sidecar write to record it. Blank
    or absent for a manifest with no ancestry annotation.

    Optional sample-size interpretation/counts, Original Effect Scale, and
    ancestry-assignment method columns pass straight through as Analytical
    Metadata (issue #86). They remain blank when omitted; the shared builder
    never fabricates values that only the manifest producer can know.

    Optional ``trait_ontology_id``/``trait_ontology_label`` and Attribution
    (``license``/``publication_doi``/``publication_pmid``/``consortium``/
    ``first_author``) columns (ADR 0034, issue #68) flow straight into
    ``analyses.tsv`` when the manifest supplies them; blank or absent
    otherwise -- never fabricated.
    """
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for column in ("stored_effect_scale", "original_sd_method"):
        if column not in fieldnames:
            raise ValueError(f"manifest {manifest_path} is missing required column: {column!r}")
    result = []
    for row in rows:
        trait_id = row["trait_id"]
        scale = row["stored_effect_scale"]
        try:
            StoredEffectScale(scale)
        except ValueError as exc:
            raise ValueError(
                f"manifest {manifest_path}: analysis {trait_id!r} has invalid "
                f"stored_effect_scale {scale!r}"
            ) from exc
        source_assembly_raw = row.get("source_assembly") or "hg19"
        try:
            source_assembly = normalise_build(source_assembly_raw)
        except ValueError as exc:
            raise ValueError(
                f"manifest {manifest_path}: analysis {trait_id!r} has invalid "
                f"source_assembly {source_assembly_raw!r}"
            ) from exc
        sd_method_raw = row["original_sd_method"]
        try:
            sd_method = OriginalSdMethod(sd_method_raw)
        except ValueError as exc:
            raise ValueError(
                f"manifest {manifest_path}: analysis {trait_id!r} has invalid "
                f"original_sd_method {sd_method_raw!r}"
            ) from exc
        if sd_method is OriginalSdMethod.UNAVAILABLE:
            raise ValueError(
                f"manifest {manifest_path}: analysis {trait_id!r} has "
                "original_sd_method='unavailable' -- its phenotype SD could not be "
                "established upstream, so the build cannot standardise its effects (issue #18)"
            )
        se_divisor = 1.0
        original_sd_raw = row.get("original_sd", "")
        if sd_method in _SD_RESCALE_METHODS:
            try:
                se_divisor = float(original_sd_raw)
            except ValueError as exc:
                raise ValueError(
                    f"manifest {manifest_path}: analysis {trait_id!r} has "
                    f"original_sd_method={sd_method.value!r} but original_sd "
                    f"{original_sd_raw!r} is not a valid number"
                ) from exc
            if not se_divisor > 0:
                raise ValueError(
                    f"manifest {manifest_path}: analysis {trait_id!r} has "
                    f"non-positive original_sd {original_sd_raw!r}"
                )
        elif original_sd_raw:
            raise ValueError(
                f"manifest {manifest_path}: analysis {trait_id!r} has "
                f"original_sd_method={sd_method.value!r}, which carries no SD "
                f"magnitude, but original_sd={original_sd_raw!r} was supplied"
            )
        result.append(
            _ManifestRow(
                trait_id=trait_id,
                file_path=row["file_path"],
                trait_name=row.get("trait_name", trait_id),
                n=int(row.get("n", 0) or 0),
                stored_effect_scale=scale,
                se_divisor=se_divisor,
                source_reader_capability=row.get("source_reader_capability") or GWAS_VCF_CAPABILITY,
                source_assembly=source_assembly,
                original_sd=original_sd_raw,
                assigned_ancestry=row.get("assigned_ancestry") or "",
                metadata=PassthroughMetadata.from_manifest_row(row),
                trait_ontology_id=row.get("trait_ontology_id") or "",
                trait_ontology_label=(
                    row.get("trait_ontology_label") or row.get("trait_ontology_name") or ""
                ),
            )
        )
    return result


def _alid_sort_key(alid: str) -> tuple[tuple[int, str], int, str, str]:
    chrom, pos_str, a1, a2 = alid.split(":")
    return (chromosome_sort_key(chrom), int(pos_str), a1, a2)


def _write_index(
    staged: StagedRelease,
    hg38_alids: list[str],
    analyses: list[Analysis],
    chunk_shape: tuple[int, int],
) -> None:
    with staged.index_connection() as connection:
        initialise_schema(connection)
        set_metadata(connection, "schema_version", 1)
        set_metadata(connection, "n_variants", len(hg38_alids))
        set_metadata(connection, "n_analyses", len(analyses))
        set_metadata(connection, "dense", dense_index_metadata(chunk_shape))
        connection.commit()


def _apply_eaf_scope(analyses: list[Analysis], column_has_eaf: np.ndarray) -> list[Analysis]:
    """Stamp each Analysis's `eaf_scope` from what the build actually stored.

    Derived, never copied from the manifest (ADR 0036): only the build knows
    whether the source file turned out to carry a usable frequency, so a
    manifest claiming EAF for an Analysis whose file has none must not be able
    to make the store declare it. `column_has_eaf` is positional, matching
    `analysis_index`.
    """
    return [
        replace(
            analysis,
            eaf_scope=(
                EafScope.ASSOCIATION.value if bool(column_has_eaf[index]) else EafScope.ABSENT.value
            ),
        )
        for index, analysis in enumerate(analyses)
    ]


def _eaf_row_band(effective_chunks: tuple[int, int], n_analyses: int) -> int:
    """Rows per band for the re-encode pass: whole chunk rows, ~50M cells."""
    chunk_rows = max(int(effective_chunks[0]), 1)
    target = max(50_000_000 // max(n_analyses, 1), 1)
    return max(chunk_rows, (target // chunk_rows) * chunk_rows)


def _write_dense_eaf(
    staged: StagedRelease,
    spill_dir: Path,
    n_variants: int,
    n_analyses: int,
    effective_chunks: tuple[int, int],
    band_cols: int,
    codec: StoreCodec,
    pass2_start: float,
) -> None:
    """Write the `eaf` plane the store's plan declares.

    A `float32` plane is written straight from the spills, one chunk-column
    band at a time, exactly as ADR 0036 shipped it.

    An `int8_residual` plane cannot be, and this is the one place in the build
    where that costs a pass. The baseline is per *variant*, so encoding a cell
    needs every Analysis's frequency at that variant; the spills are per
    *Analysis*. So the frequencies are staged as `float32` in column bands, and
    then read back in row bands -- where a whole variant is resident -- to
    compute each baseline and encode against it. The staging array is deleted
    afterwards, and peak disk is 5 bytes per cell against the 4 the `float32`
    plane occupied on its own.
    """
    residual = codec.encoding.eaf.is_residual
    staging = _EAF_STAGING if residual else "eaf"
    _create_eaf_array(staged, n_variants, n_analyses, effective_chunks, name=staging)
    root = staged.arrays(mode="a")
    eaf_zarr = root[staging]
    eaf_band = np.empty((n_variants, band_cols), dtype="float32")
    for c0 in range(0, n_analyses, band_cols):
        c1 = min(c0 + band_cols, n_analyses)
        w = c1 - c0
        eaf_band[:, :w] = np.nan
        for c in range(c0, c1):
            local = c - c0
            with np.load(spill_dir / f"{c}.npz") as data:
                eaf_band[data["rows"], local] = data["eaf"]
        eaf_zarr[:, c0:c1] = eaf_band[:, :w]
        _log_progress(
            "Band-write eaf", c1, n_analyses, pass2_start, f"cols {c0}:{c1}", every=band_cols
        )
    del eaf_band
    if not residual:
        return
    _encode_residual_dense_eaf(
        staged, n_variants, n_analyses, effective_chunks, codec, pass2_start
    )


def _encode_residual_dense_eaf(
    staged: StagedRelease,
    n_variants: int,
    n_analyses: int,
    effective_chunks: tuple[int, int],
    codec: StoreCodec,
    pass2_start: float,
) -> None:
    """Residual re-encode: transpose the staged float32 grid, then delete it.

    The staging array is only deleted once the baseline and exception tables it
    produced are both written -- a failed encode leaves the staging plane in
    place so a resume can tell the pass never finished. Full-float staging and
    exact exception values are preserved; nothing is encoded by column or
    clipped.
    """
    root = staged.arrays(mode="a")
    encoded = _create_eaf_array(
        staged,
        n_variants,
        n_analyses,
        effective_chunks,
        dtype=codec.eaf_dtype,
        fill_value=codec.eaf_fill_value,
    )
    baseline = np.full(n_variants, np.nan, dtype=np.float32)
    exceptions = EafExceptionBuilder()
    band_rows = _eaf_row_band(effective_chunks, n_analyses)
    for r0 in range(0, n_variants, band_rows):
        r1 = min(r0 + band_rows, n_variants)
        block = np.asarray(root[_EAF_STAGING][r0:r1], dtype=np.float32)
        rows_baseline = eaf_baseline_from_grid(block)
        baseline[r0:r1] = rows_baseline
        encoded[r0:r1] = codec.encode_eaf(
            block,
            baseline=np.repeat(rows_baseline[:, None], n_analyses, axis=1),
            positions=positions_row_band(r0, n_analyses),
            exceptions=exceptions,
        )
        _log_progress("Encode eaf", r1, n_variants, pass2_start, f"rows {r0}:{r1}", every=band_rows)
    write_eaf_baseline(root, baseline, compressor=_DENSE_COMPRESSOR)
    exceptions.table().write(root)
    log.info(
        "eaf: int8 residual at +/-%.1f, %d exception cell(s)",
        codec.encoding.eaf.residual_range,
        len(exceptions),
    )
    del root[_EAF_STAGING]


#: Name of the transient `float32` plane a residual-coded build stages into.
#: The spills are per Analysis and the baseline is per variant, so the grid has
#: to be transposed through something; staging it is the honest way to say so.
_EAF_STAGING = "eaf_source"


def _create_eaf_array(
    staged: StagedRelease,
    n_variants: int,
    n_analyses: int,
    effective_chunks: tuple[int, int],
    *,
    name: str = "eaf",
    dtype: str = "float32",
    fill_value: Any = float("nan"),
) -> Any:
    """Create the missing-filled `eaf` array (ADR 0036, ADR 0037 §2).

    Created after Pass 2, not alongside `z`/`se`, because how -- and whether --
    a build stores frequencies is only known once the sources have been read.
    A build whose plan says `absent` writes no array at all, so its store is
    byte-identical in shape to one built before EAF existed.

    Never `float16`: its spacing near 1.0 is 0.00049, and the canonical A1 is
    the lexicographically smaller allele rather than the minor one, so EAF near
    1 is ordinary here -- `float16` would silently round "MAF 1e-4" to "MAF 0"
    for half the ALID space.
    """
    root = staged.arrays(mode="a")
    if name in root:
        del root[name]
    return root.create_dataset(
        name,
        shape=(n_variants, n_analyses),
        chunks=effective_chunks,
        compressor=_DENSE_COMPRESSOR,
        dtype=dtype,
        fill_value=fill_value,
    )


def _create_dense_zarr(
    staged: StagedRelease,
    n_variants: int,
    n_analyses: int,
    chunk_shape: tuple[int, int],
    dtype: str,
    encoding: StoreEncoding,
) -> tuple[int, int]:
    """Create the empty missing-filled z/se datasets and return the effective chunks.

    Chunk shape is clipped to the array dimensions (ADR-0021) so zarr's declared
    chunks match what is physically stored. The arrays are created without data;
    ``_write_dense_bands`` fills them by chunk-column band afterwards. Each plane
    is filled with **its own** missing marker (spec §15): NaN for `se`, the
    reserved sentinel for a fixed-point `z`, so an untouched cell reads as
    missing under either encoding.
    """
    compressor = _DENSE_COMPRESSOR
    codec = StoreCodec(encoding)
    effective_chunks = (min(chunk_shape[0], n_variants), min(chunk_shape[1], n_analyses))
    root = staged.arrays(mode="w")
    for name, plane_dtype, fill in (
        ("z", codec.z_dtype, codec.z_fill_value),
        # Scratch in float32, as dense.complete does: the band-writer fills
        # this before the SE encoding is decided, and an exact residual
        # exception must be the source's own value, not one already rounded
        # to the dtype the plane happened to start in (spec §6a).
        ("se", "float32", float("nan")),
    ):
        root.create_dataset(
            name,
            shape=(n_variants, n_analyses),
            chunks=effective_chunks,
            compressor=compressor,
            dtype=plane_dtype,
            fill_value=fill,
        )
    root.attrs["layout"] = "dense"
    root.attrs["completion_state"] = "observed_only"
    root.attrs["compressor"] = DEFAULT_COMPRESSOR
    root.attrs["chunk_shape"] = list(effective_chunks)
    return effective_chunks


# ── EAF orientation (issue #115, ADR 0037 §6) ────────────────────────────────
#
# Shared with the Hybrid builder, which spills the same per-column arrays. The
# check runs off the spills rather than re-reading the sources: the spilled
# `eaf` is exactly what the band-write is about to store, so what is verified
# is what a query will read back, and it costs no extra pass over the source
# files.


@dataclass(frozen=True)
class EafSpillSurvey:
    """One pass over the column spills, answering both EAF questions at once.

    The orientation check (§9.1) needs a deterministic per-Analysis sample of
    frequencies; the encoding tree (ADR 0037 §2) needs the residual spread over
    the same sample plus the build's exact cell counts. They are read together
    because the spills are large and the sample is the same one.
    """

    observations: dict[str, dict[str, float]]
    #: Cells the spills hold, whether or not they carry a frequency. For a
    #: Dense component this is not the plane's size -- the plane is the whole
    #: grid -- so the caller says which count the bytes should be reckoned on.
    n_spill_cells: int
    n_eaf_cells: int
    sample_rows: np.ndarray
    sample_values: np.ndarray

    def measurements(self, *, n_cells: int, n_variants: int) -> EafMeasurements:
        return measure_eaf_sample(
            self.sample_rows,
            self.sample_values,
            n_variants=n_variants,
            n_cells=n_cells,
            n_eaf_cells=self.n_eaf_cells,
        )


def survey_eaf_spills(
    spill_dir: Path,
    id_by_col: Mapping[int, str],
    alids: Sequence[str],
    hashes: np.ndarray,
    *,
    k: int = DEFAULT_SAMPLE_SITES,
    suffix: str = "",
    index_key: str = "rows",
    row_map: np.ndarray | None = None,
) -> EafSpillSurvey:
    """Sample each Analysis's frequencies and count them, from the spills.

    Every Analysis appears in `observations`, including ones whose source
    carried no frequency: an absent key would read as "not checked yet"
    downstream, where the honest answer is "checked, and there was nothing to
    check."

    `row_map` translates a spill's own row indices onto the axis `alids` and
    `hashes` describe — the Hybrid builder's Dense Component spills are indexed
    by dense row, while both of its components are sampled on the shared axis
    so that an Analysis living mostly off-panel is checked on the same footing
    as one sitting on it.
    """
    observations: dict[str, dict[str, float]] = {aid: {} for aid in id_by_col.values()}
    n_spill_cells = 0
    n_eaf_cells = 0
    sample_rows: list[np.ndarray] = []
    sample_values: list[np.ndarray] = []
    for col, analysis_id in id_by_col.items():
        path = spill_dir / f"{col}{suffix}.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            rows = data[index_key].astype(np.int64)
            if row_map is not None:
                rows = row_map[rows].astype(np.int64)
            eaf = np.asarray(data["eaf"], dtype=np.float64)
            n_spill_cells += int(eaf.size)
            n_eaf_cells += int(np.count_nonzero(np.isfinite(eaf)))
            selected, values = sample_column_rows(rows, eaf, hashes, k=k)
            sample_rows.append(selected)
            sample_values.append(values)
            observations[analysis_id].update(
                {
                    alids[row]: float(value)
                    for row, value in zip(selected.tolist(), values.tolist(), strict=True)
                }
            )
    return EafSpillSurvey(
        observations=observations,
        n_spill_cells=n_spill_cells,
        n_eaf_cells=n_eaf_cells,
        sample_rows=(np.concatenate(sample_rows) if sample_rows else np.empty(0, dtype=np.int64)),
        sample_values=(
            np.concatenate(sample_values) if sample_values else np.empty(0, dtype=np.float64)
        ),
    )


def _write_dense_z_bands(
    root: Any,
    spill_dir: Path,
    n_variants: int,
    n_analyses: int,
    band_cols: int,
    codec: StoreCodec,
    dtype: str,
    pass2_start: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """z pass: encode, write, and harvest stored-value top hits.

    One ``(n_variants × band_cols)`` buffer is resident at a time. The top-hit
    harvest thresholds on the **quantised/decoded stored** z -- what a query
    reads back -- not the source z, and reads each hit's se straight from the
    spill rounded to the stored dtype. The overflow table is part of the z
    plane, not an addendum: it is written in the same pass that finished
    writing z. Returns the per-band hit parts and the per-\Analysis
    ``column_has_eaf`` survey for the coordinator.
    """
    z_arr = root["z"]
    band = np.empty((n_variants, band_cols), dtype=codec.z_dtype)
    overflow = ZOverflowBuilder()
    hit_rows_parts: list[np.ndarray] = []
    hit_cols_parts: list[np.ndarray] = []
    hit_z_parts: list[np.ndarray] = []
    hit_se_parts: list[np.ndarray] = []
    column_has_eaf = np.zeros(n_analyses, dtype=bool)

    log.info("Band-write z: %d analyses in bands of %d", n_analyses, band_cols)
    for c0 in range(0, n_analyses, band_cols):
        c1 = min(c0 + band_cols, n_analyses)
        w = c1 - c0
        band[:, :w] = codec.z_fill_value
        for c in range(c0, c1):
            local = c - c0
            with np.load(spill_dir / f"{c}.npz") as data:
                rows = data["rows"]
                column_has_eaf[c] = bool(np.isfinite(data["eaf"]).any())
                band[rows, local] = codec.encode_z(
                    data["z"],
                    positions=rows.astype(np.int64) * n_analyses + c,
                    overflow=overflow,
                )
                zc = codec.quantise_z(data["z"])  # what a query will read back
                hit = np.abs(zc) >= _TOP_HIT_Z_CRIT
                if np.any(hit):
                    hit_rows_parts.append(rows[hit])
                    hit_cols_parts.append(np.full(int(np.count_nonzero(hit)), c, dtype=np.int64))
                    hit_z_parts.append(zc[hit])
                    hit_se_parts.append(data["se"][hit].astype(dtype).astype(np.float32))
        z_arr[:, c0:c1] = band[:, :w]
        _log_progress(
            "Band-write z", c1, n_analyses, pass2_start, f"cols {c0}:{c1}", every=band_cols
        )
    overflow.table().write(root)
    return hit_rows_parts, hit_cols_parts, hit_z_parts, hit_se_parts, column_has_eaf


def _write_dense_se_bands(
    root: Any,
    spill_dir: Path,
    n_variants: int,
    n_analyses: int,
    band_cols: int,
    dtype: str,
    pass2_start: float,
) -> None:
    """se pass: the independent float-scratch write over one band at a time.

    ``z`` and ``se`` no longer share a dtype (ADR 0037), so this pass owns a
    fresh buffer of its own rather than reusing the z pass's.
    """
    se_arr = root["se"]
    band = np.empty((n_variants, band_cols), dtype=dtype)
    for c0 in range(0, n_analyses, band_cols):
        c1 = min(c0 + band_cols, n_analyses)
        w = c1 - c0
        band[:, :w] = np.nan
        for c in range(c0, c1):
            local = c - c0
            with np.load(spill_dir / f"{c}.npz") as data:
                band[data["rows"], local] = data["se"]
        se_arr[:, c0:c1] = band[:, :w]
        _log_progress(
            "Band-write se", c1, n_analyses, pass2_start, f"cols {c0}:{c1}", every=band_cols
        )


def _write_dense_bands(
    staged: StagedRelease,
    spill_dir: Path,
    n_variants: int,
    n_analyses: int,
    effective_chunks: tuple[int, int],
    dtype: str,
    pass2_start: float,
    encoding: StoreEncoding,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream the retained per-column spills into the zarr in chunk-column bands.

    ``z`` and ``se`` are written in two separate passes so only one band is
    ever resident, and the top-hit harvest runs in the z-pass on the stored z.
    Spills are retained through the se pass and the EAF write, and are only
    unlinked once both consumed them. Returns the concatenated top-hit
    candidate arrays ``(rows, cols, z, se)`` plus the per-\Analysis
    ``column_has_eaf`` survey for the index build and EAF decision.
    """
    root = staged.arrays(mode="a")
    band_cols = effective_chunks[1]
    codec = StoreCodec(encoding)
    hit_rows_parts, hit_cols_parts, hit_z_parts, hit_se_parts, column_has_eaf = (
        _write_dense_z_bands(
            root, spill_dir, n_variants, n_analyses, band_cols, codec, dtype, pass2_start
        )
    )
    _write_dense_se_bands(
        root, spill_dir, n_variants, n_analyses, band_cols, dtype, pass2_start
    )

    # Pass 3 -- eaf. Its own float32 buffer, since eaf cannot share z/se's
    # float16 (see `_create_eaf_array`). What is written is decided by the
    # plan, not by whether the array happens to be wanted here.
    if column_has_eaf.any() and encoding.eaf.is_absent:
        raise ValueError(
            "the encoding plan declares no eaf plane, but "
            f"{int(column_has_eaf.sum())} of {n_analyses} Analyses carried a frequency; "
            "the plan and the data disagree (ADR 0037 §2)"
        )
    if not encoding.eaf.is_absent:
        # Written whenever the plan says so, even if *this* component carries
        # no frequency: a Hybrid release's two components share one plan, and
        # a Dense Component with no EAF where the Ragged Overflow has some
        # would otherwise declare a plane it does not have. An all-absent
        # `int8` plane costs essentially nothing compressed.
        _write_dense_eaf(
            staged,
            spill_dir,
            n_variants,
            n_analyses,
            effective_chunks,
            band_cols,
            codec,
            pass2_start,
        )
    for c in range(n_analyses):
        (spill_dir / f"{c}.npz").unlink(missing_ok=True)

    if hit_rows_parts:
        return (
            np.concatenate(hit_rows_parts),
            np.concatenate(hit_cols_parts),
            np.concatenate(hit_z_parts),
            np.concatenate(hit_se_parts),
            column_has_eaf,
        )
    return (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        column_has_eaf,
    )


def _write_manifest(
    staged: StagedRelease,
    store_id: str,
    release_id: str,
    n_variants: int,
    n_analyses: int,
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
        primary_layout=PrimaryStorageLayout.DENSE,
        association_coverage=AssociationCoverage.FULL,
        completion_state=CompletionState.OBSERVED_ONLY,
        reference_assembly="GRCh38",
        created_at=datetime.now(UTC).isoformat(),
        provenance={
            "builder": "opengwasdb.v0.1_dense_vcf_two_pass",
            "chain_file": str(chain_file) if chain_file else "pyliftover_builtin_hg19_hg38",
            "n_variants": n_variants,
            "n_analyses": n_analyses,
            "dense": {
                "statistic_arrays": ["z", "se"],
                "se_dtype": encoding.se.dtype,
                "chunk_shape": list(chunk_shape),
                "compressor": DEFAULT_COMPRESSOR,
                "top_hit_thresholds": [5e-8, 5e-6, 5e-4],
            },
            **({"eaf_orientation": eaf_orientation} if eaf_orientation is not None else {}),
        },
    )
    staged.write_manifest(manifest)
