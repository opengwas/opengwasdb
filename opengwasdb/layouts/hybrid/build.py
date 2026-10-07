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

import csv
import logging
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import as_completed
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.build.eaf_orientation import (
    EafOrientationMethod,
    EafOrientationOutcome,
    EafOrientationReport,
    OrientationEvidence,
    apply_orientation_evidence,
    site_hashes,
    verify_eaf_orientation,
)
from opengwasdb.build.info_score_filter import declared_score_state
from opengwasdb.build.liftover import LiftoverFailureError
from opengwasdb.build.ordered_pool import ordered_map
from opengwasdb.build.row_admission import AdmissionCounts, admit_rows
from opengwasdb.encoding import (
    EafMeasurements,
    EncodingMeasurements,
    StoreEncoding,
    combine_eaf_measurements,
    optimise_dense_se_joint,
)
from opengwasdb.encoding.timing import format_duration
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
    _reference_axis_rsids,
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
from opengwasdb.layouts.hybrid.checkpoint import (
    AXIS,
    AXIS_CANONICAL_RAW,
    AXIS_OFF_PANEL,
    AXIS_PANEL,
    AXIS_SHARED,
    FOLD_DIR,
    HITS,
    INFO_COUNTS,
    ORIENTATION,
    PLAN,
    PROVENANCE_RSID,
    PROVENANCE_SOURCE,
    RESUME_FUNCTION,
    CheckpointState,
    checkpoint_dir_for,
    completed_phases,
    input_identities,
    input_identity,
    load_npz,
    mark_phase,
    read_build_params,
    read_json,
    read_lines,
    read_str_map,
    record_plates,
    require_fresh_destination,
    require_intact_plates,
    require_matching_params,
    save_npz,
    write_build_params,
    write_json,
    write_lines,
    write_str_map,
)
from opengwasdb.layouts.hybrid.key_runs import (
    HG19,
    HG38,
    HashedKeyCollision,
    KeyRun,
    column_run,
    merge_key_stream,
    read_run,
    write_run,
)
from opengwasdb.layouts.hybrid.key_table import (
    RAW_KEY_DTYPE,
    AlidIndex,
    CanonicalRawKeys,
    DistinctKeys,
    KeyTable,
    ResolvedKeys,
    collision_message,
    lookup_matched,
    resolve_keys,
)
from opengwasdb.layouts.hybrid.layout import (
    DENSE_SUBDIR,
    dense_component_path,
    dense_to_shared_path,
)
from opengwasdb.layouts.hybrid.unknown_keys import (
    UnknownKeyEncodingError,
    check_hash,
    encode_keys,
    is_hashed,
    placed_hashed_values,
    validated_hashed_values,
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
from opengwasdb.model.info_score_policy import (
    InfoScorePolicy,
    InfoScoreState,
    parse_info_score_policy,
)
from opengwasdb.model.maf_policy import MafPolicy, MafState, parse_maf_policy
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY
from opengwasdb.readers.interface import ImputationScoreStatus
from opengwasdb.readers.registry import resolve_reader
from opengwasdb.store.open import CURRENT_FORMAT_VERSION, OpenGWASDBStore, StagedRelease
from opengwasdb.variants import CanonicalVariant, write_variant_axis
from opengwasdb.variants.reference import (
    VariantReference,
    read_variant_reference,
    require_written_rsids_match,
)

log = logging.getLogger(__name__)

# One column's off-reference spill: encoded uint64 keys, z, se, eaf, and the
# side-file half of the hashed keys (row positions plus their raw strings).
_OffReferenceSpill = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]

__all__ = [
    "build_hybrid_from_vcf_manifest",
    "resume_hybrid_build",
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
# Column index -> that Analysis's declared INFO policy (stores #175), so a
# worker applies the policy of the row its task came from without the task
# tuple growing a sixth element the Dense builder does not share.
_pass2_info_policies: Mapping[int, InfoScorePolicy] | None = None
# Column index -> that Analysis's declared MAF policy (stores #176), carried the
# same way as the INFO policy above.
_pass2_maf_policies: Mapping[int, MafPolicy] | None = None


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
        # SourceReaders retain the source's allele spelling, while resolver
        # ALIDs and identity reference keys use upper-case alleles. Case is not
        # an effect-orientation change: keep ref/alt order intact.
        refs.append(ref.upper())
        alts.append(alt.upper())
    keys_list = [
        f"{chrom}:{pos}:{ref}:{alt}".encode()
        for chrom, pos, ref, alt in zip(chroms, poss, refs, alts, strict=True)
    ]
    keys = np.array(keys_list, dtype=object)
    del keys_list, chroms, poss, refs, alts
    targets_arr = np.array(targets, dtype=np.int64)
    ispanel_arr = np.array(ispanel, dtype=bool)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_targets = targets_arr[order]
    sorted_panel = ispanel_arr[order]
    # Two artifact keys differing only in case must not route to different
    # stored variants after folding: that would make searchsorted's choice
    # dependent on insertion order and silently misplace an association.
    duplicate = sorted_keys[1:] == sorted_keys[:-1]
    conflicting = duplicate & (
        (sorted_targets[1:] != sorted_targets[:-1])
        | (sorted_panel[1:] != sorted_panel[:-1])
    )
    if np.any(conflicting):
        key = sorted_keys[1:][conflicting][0]
        raise ValueError(f"source key {key!r} maps to conflicting variants after case folding")
    return sorted_keys, sorted_targets, sorted_panel


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
            [
                f"{c}:{p}:{r.upper()}:{a.upper()}"
                for c, p, r, a in zip(chroms, poss, refs, alts, strict=True)
            ],
            dtype=object,
        )
        return empty, empty, (keys, z_arr, se_arr, eaf_arr)
    query = _encode_variant_keys(
        chroms, poss, [ref.upper() for ref in refs], [alt.upper() for alt in alts]
    )
    idx = np.searchsorted(keys_sorted, query)
    idx_clip = np.minimum(idx, len(keys_sorted) - 1)
    matched = keys_sorted[idx_clip] == query
    panel = ispanel_sorted[idx_clip[matched]]
    tgt = targets_sorted[idx_clip[matched]]
    z_m, se_m, eaf_m = z_arr[matched], se_arr[matched], eaf_arr[matched]
    unmatched = ~matched
    keys = np.array(
        [
            f"{chroms[j]}:{poss[j]}:{refs[j].upper()}:{alts[j].upper()}"
            for j in np.flatnonzero(unmatched)
        ],
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


def _require_one_batch(scores: Sequence[float | None], zs: Sequence[float]) -> None:
    """Fail loudly unless the INFO score buffers are the association batch's rows.

    Nine buffers hold one batch -- the INFO filter reads the two score ones and
    routes the seven statistic ones. One left holding an earlier batch would
    filter one batch's rows against another's mask, which either raises or, worse,
    silently keeps the wrong rows, so it is refused here rather than routed
    (stores #175).
    """
    if len(scores) != len(zs):
        raise ValueError(
            f"INFO score buffers hold {len(scores)} row(s) for {len(zs)} association(s)"
        )


def _retain_rows(lists: tuple[list[Any], ...], keep: np.ndarray) -> None:
    """Keep `keep`'s rows in every positionally parallel batch list, in place.

    The batch lists are one buffer shared with the caller's own loop, so a
    dropped row must leave all of them together -- a filter that kept a
    coordinate and dropped its z would route an association the policy
    excluded.
    """
    if bool(keep.all()):
        return
    indices = np.flatnonzero(keep).tolist()
    for values in lists:
        values[:] = [values[index] for index in indices]


def _resolve_column_hybrid(
    file_path: str,
    keys_sorted: np.ndarray,
    targets_sorted: np.ndarray,
    ispanel_sorted: np.ndarray,
    se_divisor: float = 1.0,
    *,
    capability: str = GWAS_VCF_CAPABILITY,
    stored_effect_scale: str = StoredEffectScale.SD.value,
    info_score_policy: InfoScorePolicy | None = None,
    maf_policy: MafPolicy | None = None,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    _OffReferenceSpill,
    AdmissionCounts,
]:
    """Stream one study once, routing each association to the dense fill (on-panel),
    the ragged overflow (off-panel/reference), or -- when the routing index does
    not hold its source coordinate at all -- an off-reference bucket keyed by the
    raw coordinate. Returns ``(dense, overflow, off_reference, info_counts)``
    where each is ``(index/key, z f32, se f32, eaf f32)`` deduped last-wins; `eaf`
    is NaN where the source reports no frequency (ADR 0036).

    ``capability`` resolves a ``SourceReader`` (issue #20) rather than this
    module streaming a VCF itself; ``stored_effect_scale`` is required to
    construct one (see ``dense.build_vcf._resolve_column``).

    ``se_divisor`` divides every returned ``se`` value, dense and overflow alike
    (continuous-trait phenotype-SD standardisation, issue #18): a study's SD
    rescaling applies uniformly regardless of which component an association
    routes to. Defaults to 1.0 (no-op).

    ``info_score_policy``/``maf_policy`` are this Analysis's declared
    imputation-score and MAF policies (stores #175, #176). Its declaration
    reaches the reader, and the one shared ``admit_rows`` rule is applied to
    each batch *before* the batch is matched, so a row either filter excludes
    reaches neither component, the EAF survey nor the top-hit counts.
    ``counts`` reports the dispositions behind that filter: the associations
    this reader yielded, not canonical source rows.
    """
    info_score_policy = info_score_policy or InfoScorePolicy()
    maf_policy = maf_policy or MafPolicy()
    reader = resolve_reader(
        capability,
        file_path,
        StoredEffectScale(stored_effect_scale),
        imputation_score_declaration=info_score_policy.imputation_score_declaration,
    )
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
    scores: list[float | None] = []
    statuses: list[ImputationScoreStatus] = []
    counts = AdmissionCounts()

    def _flush() -> None:
        nonlocal counts
        _require_one_batch(scores, zs)
        if not zs:
            return
        block_scores = np.asarray(scores, dtype=np.float64)
        block_statuses = np.asarray(statuses, dtype=object)
        block_af = np.asarray(eafs, dtype=np.float64)
        # The nine buffers are consumed into arrays above and routed below: all
        # of them are emptied on every exit path, so no batch is filtered or
        # counted against another batch's rows.
        scores.clear()
        statuses.clear()
        admission = admit_rows(
            block_scores, block_statuses, block_af, info_score_policy, maf_policy
        )
        counts += admission.counts
        _retain_rows(
            (chroms, poss, refs, alts, zs, ses, eafs),
            admission.keep,
        )
        try:
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
        finally:
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
        scores.append(assoc.imputation_score.value)
        statuses.append(assoc.imputation_score.status)
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
        counts,
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


def _pass2_worker(task: tuple[int, str, float, str, str]) -> tuple[int, AdmissionCounts]:
    assert _pass2_keys_sorted is not None
    assert _pass2_targets_sorted is not None
    assert _pass2_ispanel_sorted is not None
    assert _pass2_spill_dir is not None
    assert _pass2_info_policies is not None
    assert _pass2_maf_policies is not None
    col_idx, file_path, se_divisor, capability, stored_effect_scale = task
    dense, overflow, off_reference, counts = _resolve_column_hybrid(
        file_path,
        _pass2_keys_sorted,
        _pass2_targets_sorted,
        _pass2_ispanel_sorted,
        se_divisor,
        capability=capability,
        stored_effect_scale=stored_effect_scale,
        info_score_policy=_pass2_info_policies[col_idx],
        maf_policy=_pass2_maf_policies[col_idx],
    )
    _spill_hybrid_column(_pass2_spill_dir, col_idx, dense, overflow, off_reference)
    return col_idx, counts


@dataclass(frozen=True)
class _OverflowColumn:
    """One Analysis's overflow spill, read, sorted and ready for the CSR.

    `eaf` is ``None`` when the column carries no finite frequency, which is
    what the writer turns into an all-NaN row (ADR 0036).
    """

    variant_index: np.ndarray
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray | None


def _assemble_overflow_column(task: tuple[int, str]) -> _OverflowColumn | None:
    """Load and sort one ``.ovf.npz`` column. Runs in a forked worker.

    Returns `None` for a column with no spill file. The spill is *not* deleted
    here: the parent deletes it once the column has been added to the CSR, so a
    failure partway through combination cannot lose a spill that was never
    used.
    """
    col, spill_dir = task
    path = Path(spill_dir) / f"{col}.ovf.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        vi = data["variant_index"].astype(np.int32)
        z = data["z"].astype(np.float32)
        se = data["se"].astype(np.float32)
        eaf = data["eaf"].astype(np.float32)
    # Sort by variant_index for consistent within-analysis ordering (matches
    # the ragged BESD builder and lets top-hit CSR cross-validation searchsort).
    order = np.argsort(vi, kind="stable")
    has_eaf = bool(np.isfinite(eaf).any())
    return _OverflowColumn(
        variant_index=vi[order],
        z=z[order],
        se=se[order],
        eaf=eaf[order] if has_eaf else None,
    )


def _assemble_overflow_csr(
    spill_dir: Path,
    n_analyses: int,
    n_variants: int,
    n_workers: int = 1,
    *,
    consume_spills: bool = True,
) -> tuple[RaggedCSRWriter, np.ndarray]:
    """Assemble the overflow CSR from per-column ``.ovf.npz`` spills, in analysis
    order (so CSR offsets align with analysis_index).

    Also returns a per-column bool array saying which Analyses carried an EAF
    into the *overflow* component. An Analysis can have EAF off-panel and none
    on it, so `eaf_scope` is the union of this and the Dense Component's own
    answer, never either alone (ADR 0036).

    The columns are independent, so `n_workers` > 1 loads and sorts them through
    `ordered_map`, which keeps only a bounded number of results in flight, and
    the parent adds them to the CSR in analysis order. `n_workers <= 1` is the
    serial path.

    ``consume_spills=False`` keeps each plate after it has been added (issue
    #227): the build tail that a checkpoint re-runs wholesale re-assembles the
    CSR, and the assembled writer lives only in memory, so the plates are its
    only durable input. The plates are read and combined identically either
    way -- the two differ in what is left on disk, not in what is written.
    """
    csr = RaggedCSRWriter(n_variants)
    column_has_eaf = np.zeros(n_analyses, dtype=bool)
    tasks = ((col, str(spill_dir)) for col in range(n_analyses))
    for col, result in enumerate(ordered_map(_assemble_overflow_column, tasks, n_workers)):
        if result is None:
            csr.add_analysis(
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
            continue
        column_has_eaf[col] = result.eaf is not None
        csr.add_analysis(result.variant_index, result.z, result.se, eaf=result.eaf)
        if consume_spills:
            (spill_dir / f"{col}.ovf.npz").unlink()
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
    -- the store's root variant axis. The layout contract both components
    share is that dense row ``i`` is the ``i``-th panel ALID of
    ``shared_sorted``, so ``dense_to_shared.npy`` is a strictly ascending map.
    No ALID -> index dict is kept here: at tens of millions of variants one
    would stay resident through Pass 2 and consolidation (ticket #222 review);
    the routing index builds its own and drops it, and later lookups go
    through ``AlidIndex``.
    """

    panel_sorted: list[str]
    off_panel_alids: list[str]
    shared_sorted: list[str]
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
    info_score_policies: Mapping[str, InfoScorePolicy]
    maf_policies: Mapping[str, MafPolicy]
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
    """What Pass 2 leaves for the EAF survey: the {column: analysis_id} map, the
    pass start time, and each Analysis's declared-score dispositions (#175)."""

    id_by_col: dict[int, str]
    pass2_start: float
    info_counts: Mapping[str, AdmissionCounts] = field(default_factory=dict)


@dataclass(frozen=True)
class _EafEvidence:
    """The EAF orientation phase's product: each component's frequency survey,
    which the encoding plan is measured from, and the orientation report the
    Analyses are stamped and the manifest's provenance written from.

    A resumed run that reloads the *report* records no survey -- the plan the
    surveys fed is recorded too, so nothing measures again (issue #227)."""

    dense_survey: EafSpillSurvey | None
    overflow_survey: EafSpillSurvey | None
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
    info_counts: Mapping[str, AdmissionCounts] = field(default_factory=dict)


def _open_dense_component(staged: StagedRelease) -> tuple[Path, StagedRelease]:
    """Open the nested Dense Component's staging directory inside the outer store's.

    Tolerates an existing one: a resumed checkpointed build adopts the release
    the interrupted run was writing, Dense Component included (issue #227).
    """
    dense_dir = dense_component_path(staged.path)
    dense_dir.mkdir(exist_ok=True)
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
        a1, a2 = sorted((ref.upper(), alt.upper()))
        origin = f"{chrom}:{pos}:{a1}:{a2}"
        if hg38_alid not in hg38_to_source:
            hg38_to_source[hg38_alid] = origin
        elif hg38_to_source[hg38_alid] != origin:
            hg38_to_source[hg38_alid] = None
    return hg38_to_source


@dataclass(frozen=True)
class DeclaredPolicies:
    """The per-Analysis declared INFO and MAF policies one manifest carries.

    Read together from a single re-read of the manifest (`_read_declared_policies`)
    because they travel together everywhere: Pass 2 applies both to one batch, the
    manifest provenance records both, and a resumed build re-derives both.
    """

    info: Mapping[str, InfoScorePolicy]
    maf: Mapping[str, MafPolicy]


def _load_manifest(
    manifest_path: str | Path,
    *,
    default_source_reader_capability: str | None = None,
    default_source_assembly: str | None = None,
) -> tuple[list[_ManifestRow], DeclaredPolicies]:
    """Read the build manifest, failing loudly on an empty one rather than
    building a store with no Analyses (a plausible empty answer), alongside each
    row's declared INFO and MAF policies (`_read_declared_policies`).
    """
    manifest_rows = _read_manifest(
        manifest_path,
        default_source_reader_capability=default_source_reader_capability,
        default_source_assembly=default_source_assembly,
    )
    if not manifest_rows:
        raise ValueError(f"manifest {manifest_path} contains no rows")
    return manifest_rows, _read_declared_policies(manifest_path, manifest_rows)


def _read_declared_policies(
    manifest_path: str | Path, rows: list[_ManifestRow]
) -> DeclaredPolicies:
    """Parse each Analysis's declared INFO and MAF policies (stores #175, #176).

    `_ManifestRow` carries the columns this builder routes with, not the policy
    columns, so the manifest is read once more here -- it lists Analyses, not
    associations -- and matched row for row against the rows `_read_manifest`
    produced. The reader capability is the row's own resolved value, exactly as
    the manifest resolver parses the same row, so a CLI default applies to both.
    A partial or contradictory declaration is a manifest error naming its
    Analysis, never a filter silently not applied.
    """
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        raw_rows = list(csv.DictReader(fh, delimiter="\t"))
    if len(raw_rows) != len(rows):
        raise ValueError(
            f"manifest {manifest_path} changed while it was read: {len(rows)} rows parsed, "
            f"{len(raw_rows)} re-read"
        )
    info: dict[str, InfoScorePolicy] = {}
    maf: dict[str, MafPolicy] = {}
    for row, raw in zip(rows, raw_rows, strict=True):
        try:
            info[row.trait_id] = parse_info_score_policy(
                raw, reader_capability=row.source_reader_capability
            )
        except ValueError as exc:
            raise ValueError(
                f"manifest {manifest_path}: analysis {row.trait_id!r} has invalid INFO "
                f"policy: {exc}"
            ) from exc
        try:
            maf[row.trait_id] = parse_maf_policy(raw)
        except ValueError as exc:
            raise ValueError(
                f"manifest {manifest_path}: analysis {row.trait_id!r} has invalid MAF "
                f"policy: {exc}"
            ) from exc
    return DeclaredPolicies(info=info, maf=maf)


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
        rsid_by_alid = _reference_axis_rsids(
            reference,
            manifest_rows,
            options.chain_file,
            options.liftover_failure_threshold,
            options.n_workers,
        )
        log.info(
            "Single-pass build: variant axis loaded from %s (%d panel variants); "
            "Pass 1 variant discovery bypassed",
            options.variant_reference,
            len(panel_alids),
        )
        return dense_dir, dense_staged, reference.source_lookup, rsid_by_alid, panel_alids
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
        {alid: i for i, alid in enumerate(partition.panel_sorted)},
        {alid: i for i, alid in enumerate(partition.shared_sorted)},
    )
    # The routing index is built: the source union (and the ALID -> index
    # dicts, which lived only for that call) are freed before Pass 2.
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
    dense_to_shared = (
        AlidIndex(axis.partition.shared_sorted)
        .lookup(axis.partition.panel_sorted)
        .astype(np.int32)
    )
    np.save(dense_to_shared_path(staged.path), dense_to_shared)
    return dense_to_shared


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
    # Line by line: the whole file as one string next to its split lines would
    # double the peak in every key-table worker (ticket #222).
    with side.open(encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


@dataclass(frozen=True)
class _UnknownKeySpill:
    """One column's ``.unk.npz`` keys and the raw strings of its hashed half.

    Only the keys and the side-file strings are needed to resolve the
    build-wide table; the association arrays are left on disk until the fold
    (ticket #222), so this deliberately does not decode every row.
    """

    keys: np.ndarray
    hashed_values: np.ndarray
    hashed_raw: list[str]


def _load_unknown_key_spill(spill_dir: Path, col: int) -> _UnknownKeySpill | None:
    """Read one column's encoded keys and its validated hashed raw keys.

    The keys are uint64 by contract (issue #218) -- no pickle -- and
    ``validated_hashed_values`` refuses a side file that does not name every
    tagged row exactly once, or names a key that does not encode to its row, so
    a corrupt spill fails here rather than resolving a shorter, plausible key
    set. ``hashed_raw`` stays in side-file order, paired with ``hashed_values``.
    """
    path = spill_dir / f"{col}.unk.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        keys = data["keys"]
        hashed_index = data["hashed_index"]
    raw = _read_unknown_side_file(spill_dir, col)
    named = validated_hashed_values(keys, hashed_index, raw)
    return _UnknownKeySpill(keys=keys, hashed_values=named, hashed_raw=raw)


def _assembly_bit(assembly: str) -> int:
    """The key-table bit for a manifest row's normalised source assembly."""
    return HG38 if assembly == "hg38" else HG19


_KeyChunkTask = tuple[Path, tuple[int, ...], tuple[str, ...], Path]


def _summarise_key_chunk(task: _KeyChunkTask) -> Path:
    """The distinct off-reference keys of one chunk of columns (ticket #222).

    Each column becomes a ``KeyRun`` -- its distinct values, one assembly bit,
    and its hashed keys' check hashes, but no raw strings -- and
    ``merge_key_stream`` folds them in one at a time, so the worker holds its
    running distinct set and never every column's keys at once. The result is
    spilled to ``out`` and only the path crosses the pool boundary (see
    ``write_run``).
    """
    spill_dir, cols, assemblies, out = task
    write_run(merge_key_stream(_column_run_stream(spill_dir, cols, assemblies)), out)
    return out


def _column_run_stream(
    spill_dir: Path, cols: Sequence[int], assemblies: Sequence[str]
) -> Iterator[KeyRun]:
    """Each spilled column's key run, tagged with its assembly.

    A generator, so each column's run is released once ``merge_key_stream``
    has folded it in; the full per-association key array and the side file's
    strings never outlive the column's reduction to a run.
    """
    for col, assembly in zip(cols, assemblies, strict=True):
        run = _column_key_run(spill_dir, col, assembly)
        if run is None:
            continue
        yield run
        del run


def _column_key_run(spill_dir: Path, col: int, assembly: str) -> KeyRun | None:
    """One column's key run; ``None`` if it spilled no off-reference keys."""
    loaded = _load_unknown_key_spill(spill_dir, col)
    if loaded is None:
        return None
    checks = np.fromiter(
        (check_hash(raw) for raw in loaded.hashed_raw),
        dtype=np.uint64,
        count=len(loaded.hashed_raw),
    )
    return column_run(
        loaded.keys,
        loaded.hashed_values,
        checks,
        column=col,
        assembly_bit=_assembly_bit(assembly),
    )


def _key_chunk_tasks(
    spill_dir: Path, prepared: _PreparedBuild, n_workers: int
) -> list[_KeyChunkTask]:
    """Split the manifest's columns into a bounded number of worker chunks."""
    rows = prepared.manifest_rows
    n = len(rows)
    if n == 0:
        return []
    n_chunks = max(1, min(n, 4 * max(1, n_workers)))
    size = -(-n // n_chunks)
    tasks: list[_KeyChunkTask] = []
    for start in range(0, n, size):
        cols = tuple(range(start, min(start + size, n)))
        assemblies = tuple(rows[col].source_assembly for col in cols)
        tasks.append((spill_dir, cols, assemblies, spill_dir / f"keyrun.{start}.npz"))
    return tasks


def _merge_key_runs(prepared: _PreparedBuild, n_workers: int) -> KeyRun | None:
    """The build-wide distinct off-reference keys, merged in parallel.

    Workers return spill paths, so a pending result costs the parent nothing;
    each chunk is loaded only when its turn comes and ``merge_key_stream``
    releases it once merged. The parent never holds every worker result
    (ticket #222 review round 1) -- ``list(...)`` here would. A hash collision
    found by any merge is re-raised naming both raw keys.
    """
    tasks = _key_chunk_tasks(prepared.spill_dir, prepared, n_workers)
    if not tasks:
        return None
    paths = ordered_map(_summarise_key_chunk, tasks, n_workers)
    try:
        return merge_key_stream(_load_key_run(path) for path in paths)
    except HashedKeyCollision as collision:
        raise UnknownKeyEncodingError(
            _collision_message(prepared.spill_dir, collision)
        ) from collision


def _load_key_run(path: Path) -> KeyRun:
    """Read one worker's spilled key run and delete the spill."""
    run = read_run(path)
    path.unlink()
    return run


def _collision_message(spill_dir: Path, collision: HashedKeyCollision) -> str:
    """Name every raw key the two colliding columns hold for the shared value."""
    raws: list[str] = []
    for col in dict.fromkeys(collision.columns):
        loaded = _load_unknown_key_spill(spill_dir, col)
        if loaded is None:
            continue
        for value, raw in zip(loaded.hashed_values.tolist(), loaded.hashed_raw, strict=True):
            if value == collision.value and raw not in raws:
                raws.append(raw)
    if len(raws) == 2:
        return collision_message(raws[0], raws[1], collision.value)
    named = " and ".join(repr(raw) for raw in raws)
    return (
        f"hash collision between off-reference keys {named} (both encode to "
        f"{collision.value}); refusing to merge them"
    )


_HashedRawTask = tuple[Path, int, np.ndarray, np.ndarray]


def _column_hashed_raw(task: _HashedRawTask) -> list[str]:
    """The raw keys one column's side file holds for ``values``, verified.

    Each raw key must re-encode to its value (``validated_hashed_values``) and
    match the check hash the merge carried for it, so a side file that changed
    since the merge -- or a value the column never held -- fails loudly.
    """
    spill_dir, col, values, checks = task
    loaded = _load_unknown_key_spill(spill_dir, col)
    if loaded is None:
        raise UnknownKeyEncodingError(
            f"column {col}'s off-reference spill is gone; cannot read its hashed raw keys"
        )
    order = np.argsort(loaded.hashed_values, kind="stable")
    held = loaded.hashed_values[order]
    position = np.minimum(np.searchsorted(held, values), max(len(held) - 1, 0))
    if not len(held) or not np.array_equal(held[position], values):
        raise UnknownKeyEncodingError(
            f"column {col}'s side file lacks a hashed key the merge assigned to it"
        )
    raws = [loaded.hashed_raw[index] for index in order[position].tolist()]
    recomputed = np.fromiter(
        (check_hash(raw) for raw in raws), dtype=np.uint64, count=len(raws)
    )
    if not np.array_equal(recomputed, checks):
        raise UnknownKeyEncodingError(
            f"column {col}'s side file no longer matches the merged key checks"
        )
    return raws


def _fetch_hashed_raw(spill_dir: Path, merged: KeyRun, n_workers: int) -> np.ndarray:
    """The raw key of every distinct hashed key, from its lowest declaring column.

    One task per origin column, in parallel; each hashed key's string is read
    once, after the merge, instead of being carried through it. Returned as a
    ``RAW_KEY_DTYPE`` array parallel to ``merged.hashed_values``: it is kept
    through the fold as the canonical key of each value.
    """
    origin = merged.hashed_origin
    order = np.argsort(origin, kind="stable")
    cols, starts = np.unique(origin[order], return_index=True)
    bounds = [*starts.tolist(), len(order)]
    groups = [order[bounds[i] : bounds[i + 1]] for i in range(len(cols))]
    tasks = (
        (spill_dir, int(col), merged.hashed_values[group], merged.hashed_check[group])
        for col, group in zip(cols.tolist(), groups, strict=True)
    )
    raw = np.empty(len(origin), dtype=RAW_KEY_DTYPE)
    filled = np.zeros(len(origin), dtype=bool)
    for group, raws in zip(groups, ordered_map(_column_hashed_raw, tasks, n_workers), strict=True):
        raw[group] = np.array(raws, dtype=RAW_KEY_DTYPE)
        filled[group] = True
    if not bool(filled.all()):
        raise UnknownKeyEncodingError("a merged hashed key has no raw side-file key")
    return raw


@dataclass(frozen=True)
class _OffReferenceKeys:
    """The resolved off-reference keys, and the canonical raw key of every
    distinct hashed one -- which the fold checks each column's rows against."""

    resolved: ResolvedKeys
    canonical: CanonicalRawKeys

    @property
    def keys(self) -> np.ndarray:
        return self.resolved.keys

    @property
    def alids(self) -> list[str]:
        return self.resolved.alids

    @property
    def origins(self) -> list[str | None]:
        return self.resolved.origins


def _resolve_off_reference_keys(
    prepared: _PreparedBuild, options: _BuildOptions
) -> _OffReferenceKeys | None:
    """Resolve build-wide distinct off-reference keys to hg38 ALIDs, in parallel.

    Workers reduce column chunks to key runs; the parent merges them, reads
    the distinct hashed keys' raw strings, and resolves once per distinct key.
    ``None`` means no column spilled an off-reference key. The resolved keys
    can be empty (every key dropped); the canonical raw keys are returned
    either way, because every column's hashed rows must still be checked.
    """
    merged = _merge_key_runs(prepared, options.n_workers)
    if merged is None:
        return None
    distinct = DistinctKeys(
        values=merged.values,
        assembly_bits=merged.assembly_bits,
        hashed_values=merged.hashed_values,
        hashed_raw=_fetch_hashed_raw(prepared.spill_dir, merged, options.n_workers),
    )
    del merged
    resolved = resolve_keys(
        distinct,
        liftover_failure_threshold=options.liftover_failure_threshold,
        chain_file=options.chain_file,
    )
    canonical = CanonicalRawKeys(values=distinct.hashed_values, raw=distinct.hashed_raw)
    return _OffReferenceKeys(resolved=resolved, canonical=canonical)


def _unknown_origins(resolved: ResolvedKeys) -> dict[str, str | None]:
    """The source-build ALID each off-reference hg38 ALID came from (collisions blank)."""
    origins: dict[str, str | None] = {}
    for alid, origin in zip(resolved.alids, resolved.origins, strict=True):
        if alid not in origins:
            origins[alid] = origin
        elif origins[alid] != origin:
            origins[alid] = None
    return origins


def _merge_unknown_provenance(
    known: dict[str, str | None], resolved: ResolvedKeys
) -> dict[str, str | None]:
    """Fold the off-reference origins into the Pass 1 provenance map.

    A stored variant reached from two different source coordinates keeps no
    origin (issue #85); the off-reference side blanks exactly as Pass 1's does.
    """
    hg38_to_source = dict(known)
    for alid, origin in _unknown_origins(resolved).items():
        if alid in hg38_to_source and hg38_to_source[alid] != origin:
            hg38_to_source[alid] = None
        else:
            hg38_to_source[alid] = origin
    return hg38_to_source


@dataclass(frozen=True)
class _FoldContext:
    spill_dir: Path
    table_keys: np.ndarray
    table_shared_index: np.ndarray
    canonical_values: np.ndarray
    canonical_raw: np.ndarray
    old_to_new: np.ndarray | None
    marker_dir: Path | None = None


_fold_context: _FoldContext | None = None


#: The axes that fold under nothing: no off-reference key resolved, so no
#: column has an entry to rewrite -- only its unresolved rows to drop.
_EMPTY_KEY_TABLE = KeyTable(
    keys=np.empty(0, dtype=np.uint64), shared_index=np.empty(0, dtype=np.int64)
)
_EMPTY_CANONICAL = CanonicalRawKeys(
    values=np.empty(0, dtype=np.uint64), raw=np.empty(0, dtype=RAW_KEY_DTYPE)
)

#: Empty routing arrays. They are a resume's only: Pass 2 and the fold are the
#: two readers of the routing index, and a resumed run that reaches either of
#: them re-derives it from the recorded inputs in `_prepare_build` instead.
_NO_ROUTING = (
    np.empty(0, dtype=object),
    np.empty(0, dtype=np.int64),
    np.empty(0, dtype=bool),
)


@dataclass(frozen=True)
class _FoldInputs:
    """What the off-reference fold runs under: the one build-wide key table
    every column's keys are looked up in, the old->new shared-index remap, the
    canonical raw keys each hashed row is checked against, and the axis and
    provenance the fold leaves behind.

    Recorded whole, before the fold's first column (issue #227): the key table
    is resolved from *every* column's spill, so a fold resumed without it could
    not re-derive it from the columns that are left, and an association whose
    key only an already-folded column carried would be dropped in silence.
    """

    table: KeyTable
    old_to_new: np.ndarray | None
    canonical: CanonicalRawKeys
    off_panel: list[str]
    shared_sorted: list[str]
    dense_to_shared: np.ndarray
    hg38_to_source: dict[str, str | None]


def _folded_columns(state: CheckpointState | None) -> frozenset[int]:
    """The columns a checkpoint records as already folded.

    One file per column, written by the worker that folded it: the fold's
    results are yielded to the parent in order but computed ahead of that, so
    the parent's own progress is not the record of which columns are done.
    """
    if state is None:
        return frozenset()
    marker_dir = state.path / FOLD_DIR
    if not marker_dir.exists():
        return frozenset()
    return frozenset(int(path.stem) for path in marker_dir.glob("*.done"))


def _combine_overflow(
    ex: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    off: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    needs_remap: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Combine existing overflow and off-reference entries with last-wins precedence.

    Off-reference entries are concatenated after existing overflow entries, so
    they always win on duplicate shared indices (issue #223).
    """
    if ex is not None and off is not None:
        comb = tuple(np.concatenate([e, o]) for e, o in zip(ex, off, strict=True))
        return _dedup_last_wins(*comb)
    if off is not None:
        return _dedup_last_wins(*off)
    if ex is not None and needs_remap:
        return _dedup_last_wins(*ex)
    return None


def _load_existing_overflow(
    spill_dir: Path, col: int, old_to_new: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load and re-index existing overflow spill if present."""
    ovf_path = spill_dir / f"{col}.ovf.npz"
    if not ovf_path.exists():
        return None
    with np.load(ovf_path) as data:
        raw_vi = data["variant_index"].astype(np.int64)
        z, se, eaf = data["z"], data["se"], data["eaf"]
    vi = old_to_new[raw_vi] if old_to_new is not None else raw_vi
    return vi, z, se, eaf


def _load_fold_existing(
    ctx: _FoldContext, spill_dir: Path, col: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load existing overflow unless an empty fold leaves it unchanged."""
    if ctx.old_to_new is None and not len(ctx.table_keys):
        return None
    return _load_existing_overflow(spill_dir, col, ctx.old_to_new)


def _load_off_reference(
    ctx: _FoldContext, spill_dir: Path, col: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load, verify, and resolve off-reference associations with searchsorted."""
    unk_path = spill_dir / f"{col}.unk.npz"
    if not unk_path.exists():
        return None
    with np.load(unk_path) as data:
        keys = data["keys"]
        u_z, u_se, u_eaf = data["z"], data["se"], data["eaf"]
        hashed_index = data["hashed_index"]
    canonical = CanonicalRawKeys(ctx.canonical_values, ctx.canonical_raw)
    _verify_hashed_rows(spill_dir, col, keys, hashed_index, canonical)
    keep, shared = lookup_matched(ctx.table_keys, ctx.table_shared_index, keys)
    if not len(keep):
        return None
    return shared, u_z[keep], u_se[keep], u_eaf[keep]


def _fold_column(col: int) -> int:
    """Fold one column's ``.unk.npz`` into its ``.ovf.npz`` with shared indices.

    Remaps existing overflow entries if an axis shift occurred (old_to_new),
    verifies hashed keys against canonical raw keys (failing loudly on hash collision),
    looks up off-reference keys in the key table using sorted searchsorted, drops
    unresolved keys, and deduplicates duplicate Variant Indices last-wins with
    off-reference entries winning over existing overflow entries (ticket #223).
    Writes the overflow spill once atomically (temp file then rename), and unlinks
    the off-reference spill only after the overflow spill is safely written.
    """
    assert _fold_context is not None
    ctx = _fold_context
    spill_dir = ctx.spill_dir
    ovf_path = spill_dir / f"{col}.ovf.npz"
    unk_path = spill_dir / f"{col}.unk.npz"

    # With no remap (the all-keys-drop path), the existing overflow spill is
    # already on the final axis and must remain untouched.  Avoid reading it.
    ex = _load_fold_existing(ctx, spill_dir, col)
    off = _load_off_reference(ctx, spill_dir, col)
    if ex is None and off is None and not unk_path.exists():
        _mark_folded(ctx, col)
        return col

    final = _combine_overflow(ex, off, ctx.old_to_new is not None)
    if final is not None:
        tmp_ovf = spill_dir / f"{col}.ovf.tmp.npz"
        try:
            np.savez(tmp_ovf, variant_index=final[0], z=final[1], se=final[2], eaf=final[3])
            tmp_ovf.replace(ovf_path)
        except BaseException:
            tmp_ovf.unlink(missing_ok=True)
            raise

    if unk_path.exists():
        unk_path.unlink()
        _unknown_side_path(spill_dir, col).unlink(missing_ok=True)

    _mark_folded(ctx, col)
    return col


def _mark_folded(ctx: _FoldContext, col: int) -> None:
    """Record that one column's fold is complete, after its write is durable.

    Written last, and only when a checkpoint asked for it: the mark is what a
    resumed fold skips the column on, so a mark before the atomic overflow
    write has landed would drop the column's off-reference associations.
    """
    if ctx.marker_dir is not None:
        (ctx.marker_dir / f"{col}.done").touch()


def _verify_hashed_rows(
    spill_dir: Path,
    col: int,
    keys: np.ndarray,
    hashed_index: np.ndarray,
    canonical: CanonicalRawKeys,
) -> None:
    """Check each of one column's hashed rows carries its value's canonical raw key.

    The exact half of the build-wide collision guarantee (#218, ticket #222
    review round 2): the merge's check hash only filters, and two keys can
    collide on both hashes. The side file is already this column's to read,
    so the check costs no extra pass over the spills.
    """
    raws = _read_unknown_side_file(spill_dir, col)
    values = placed_hashed_values(keys, hashed_index, raws)
    canonical.verify(col, values, np.array(raws, dtype=RAW_KEY_DTYPE))


def _fold_unknown_spills(
    spill_dir: Path,
    n_analyses: int,
    table: KeyTable,
    canonical: CanonicalRawKeys,
    old_to_new: np.ndarray | None,
    n_workers: int,
    *,
    marker_dir: Path | None = None,
    already_folded: frozenset[int] = frozenset(),
) -> None:
    """Fold every column's ``.unk.npz`` into its ``.ovf.npz`` with shared indices.

    Runs across ``--n-workers`` workers via ``ordered_map``. With ``n_workers <= 1``
    it runs in-process on the serial path (ticket #223).
    Workers receive the key table, canonical raw keys, and remapping index as
    read-only arrays across the fork, with zero per-key Python objects or dicts.
    This function takes ownership of those arrays' mutability: it marks the
    caller-owned arrays read-only and deliberately leaves them read-only after
    returning.  Callers must not write them after handing them to the fold.
    Each column's overflow spill is written at most once, atomically.

    ``marker_dir`` (issue #227) makes each column's completion a file the
    worker writes, and ``already_folded`` the columns a previous run wrote one
    for: the two together are what lets an interrupted fold resume on the
    columns it has left, rather than re-remapping entries it already rewrote.
    """
    table_keys = table.keys
    table_shared = table.shared_index
    table_keys.flags.writeable = False
    table_shared.flags.writeable = False

    canonical_values = canonical.values
    canonical_raw = canonical.raw
    canonical_values.flags.writeable = False
    canonical_raw.flags.writeable = False

    if old_to_new is not None:
        old_to_new.flags.writeable = False

    global _fold_context
    _fold_context = _FoldContext(
        spill_dir=spill_dir,
        table_keys=table_keys,
        table_shared_index=table_shared,
        canonical_values=canonical_values,
        canonical_raw=canonical_raw,
        old_to_new=old_to_new,
        marker_dir=marker_dir,
    )
    pending = [col for col in range(n_analyses) if col not in already_folded]
    try:
        for _ in ordered_map(_fold_column, pending, n_workers):
            pass
    finally:
        _fold_context = None


def _build_shared_key_table(
    prepared: _PreparedBuild, resolved: ResolvedKeys
) -> tuple[KeyTable, np.ndarray, AlidIndex, list[str], list[str]]:
    """Build key table and axis remapping index for resolved off-reference variants."""
    unknown_alids = set(resolved.alids)
    off_panel = _sorted_alids(set(prepared.partition.off_panel_alids) | unknown_alids)
    panel = prepared.partition.panel_sorted
    shared_sorted = _sorted_alids(set(panel) | set(off_panel))
    index = AlidIndex(shared_sorted)
    table = KeyTable(keys=resolved.keys, shared_index=index.lookup(resolved.alids))
    old_to_new = index.lookup(prepared.partition.shared_sorted)
    return table, old_to_new, index, off_panel, shared_sorted


def _finalise_reference_partition(
    prepared: _PreparedBuild,
    options: _BuildOptions,
    state: CheckpointState | None = None,
) -> _PreparedBuild:
    """Add Pass 2-discovered off-reference variants to the shared axis.

    With ``--variant-reference`` the reference fixes the Dense axis and the
    routing for the variants it knows, but a study may observe variants the
    reference never listed. Those are off-reference and belong in the Ragged
    Overflow; they are unknown until Pass 2 streams them, so their shared
    indices -- and therefore the panel's mapping onto them -- are only
    computable now. A legacy ``--reference-panel`` build already knows its
    whole union before Pass 2 and is returned unchanged.

    With a checkpoint (issue #227) the axis this computes is recorded before
    the fold's first column, and a resumed fold runs under exactly that record
    instead of resolving the keys again: the resolved key set comes from every
    column's spill, and the columns already folded have none left.
    """
    if options.variant_reference is None:
        _record_axis(state, prepared, None)
        return prepared
    recorded = _recorded_axis(state)
    if recorded is not None:
        return _fold_recorded_axis(prepared, options, state, recorded)
    fold = _off_reference_fold_inputs(prepared, options)
    _record_axis(state, prepared, fold)
    if fold is None:
        return prepared
    return _fold_under(prepared, options, state, fold)


def _fold_recorded_axis(
    prepared: _PreparedBuild,
    options: _BuildOptions,
    state: CheckpointState | None,
    fold: _FoldInputs,
) -> _PreparedBuild:
    """Fold this run's remaining columns under the axis the first run recorded.

    Nothing is resolved and nothing is measured here: the key table, the
    remap and the provenance are the first run's, which is the only way the
    columns folded after the interruption can land on the axis the ones before
    it did.
    """
    return _fold_under(prepared, options, state, fold)


def _fold_under(
    prepared: _PreparedBuild,
    options: _BuildOptions,
    state: CheckpointState | None,
    fold: _FoldInputs,
) -> _PreparedBuild:
    """Fold this run's columns under `fold`, then carry the axis it produced.

    The recorded axis is written back to the release as the fold's own output:
    the EAF survey and the query facade both read that sidecar, and a resumed
    run must leave the map its indices were built against rather than the
    pre-fold one `_prepare_build` may have re-written.
    """
    _fold_unknown_spills(
        prepared.spill_dir,
        prepared.n_analyses,
        fold.table,
        fold.canonical,
        old_to_new=fold.old_to_new,
        n_workers=options.n_workers,
        marker_dir=_fold_marker_dir(state),
        already_folded=_folded_columns(state),
    )
    np.save(dense_to_shared_path(prepared.staged.path), fold.dense_to_shared)
    return _update_partition(
        prepared, fold.off_panel, fold.shared_sorted, fold.dense_to_shared, fold.hg38_to_source
    )


def _off_reference_fold_inputs(
    prepared: _PreparedBuild, options: _BuildOptions
) -> _FoldInputs | None:
    """Resolve the build's off-reference keys into the axis the fold runs under.

    ``None`` means no column spilled an off-reference key at all, so there is
    nothing to fold: the axis is Pass 1's partition and its provenance map.
    An empty but present resolution still folds -- every hashed row is checked
    exactly, and the fold then drops the unresolved rows as it always did.
    """
    off_reference = _resolve_off_reference_keys(prepared, options)
    if off_reference is None:
        return None
    resolved = off_reference.resolved
    if not len(resolved.keys):
        return _FoldInputs(
            table=_EMPTY_KEY_TABLE,
            old_to_new=None,
            canonical=off_reference.canonical,
            off_panel=prepared.partition.off_panel_alids,
            shared_sorted=prepared.partition.shared_sorted,
            dense_to_shared=prepared.dense_to_shared,
            hg38_to_source=prepared.hg38_to_source,
        )
    table, old_to_new, index, off_panel, shared_sorted = _build_shared_key_table(
        prepared, resolved
    )
    dense_to_shared = index.lookup(prepared.partition.panel_sorted).astype(np.int32)
    log.info(
        "Single-pass build: %d off-reference variant(s) routed to the Ragged Overflow",
        len(off_panel) - prepared.partition.n_off_panel,
    )
    return _FoldInputs(
        table=table,
        old_to_new=old_to_new,
        canonical=off_reference.canonical,
        off_panel=off_panel,
        shared_sorted=shared_sorted,
        dense_to_shared=dense_to_shared,
        hg38_to_source=_merge_unknown_provenance(prepared.hg38_to_source, resolved),
    )


def _fold_marker_dir(state: CheckpointState | None) -> Path | None:
    """Where a fold records its per-column completions, or ``None`` without one.

    Created here rather than by the workers that write into it: those run in a
    forked pool, and a missing directory would be one ENOENT per column.
    """
    if state is None:
        return None
    marker_dir = state.path / FOLD_DIR
    marker_dir.mkdir(exist_ok=True)
    return marker_dir


def _record_axis(
    state: CheckpointState | None, prepared: _PreparedBuild, fold: _FoldInputs | None
) -> None:
    """Record the axis the fold is about to run under, before its first column.

    `None` for ``fold`` is the no-off-reference case: the axis is Pass 1's
    partition, and it is recorded all the same, because every phase after the
    fold reads it and none of them may re-derive it.
    """
    if state is None:
        return
    inputs = fold or _FoldInputs(
        table=_EMPTY_KEY_TABLE,
        old_to_new=None,
        canonical=_EMPTY_CANONICAL,
        off_panel=prepared.partition.off_panel_alids,
        shared_sorted=prepared.partition.shared_sorted,
        dense_to_shared=prepared.dense_to_shared,
        hg38_to_source=prepared.hg38_to_source,
    )
    save_npz(
        state.path / AXIS,
        keys=inputs.table.keys,
        shared_index=inputs.table.shared_index,
        old_to_new=(
            inputs.old_to_new if inputs.old_to_new is not None else np.empty(0, dtype=np.int64)
        ),
        canonical_values=inputs.canonical.values,
        dense_to_shared=inputs.dense_to_shared,
    )
    write_lines(state.path / AXIS_CANONICAL_RAW, inputs.canonical.raw.tolist())
    write_lines(state.path / AXIS_PANEL, prepared.partition.panel_sorted)
    write_lines(state.path / AXIS_OFF_PANEL, inputs.off_panel)
    write_lines(state.path / AXIS_SHARED, inputs.shared_sorted)
    write_str_map(state.path / PROVENANCE_SOURCE, inputs.hg38_to_source)
    write_str_map(state.path / PROVENANCE_RSID, prepared.rsid_by_alid)


def _recorded_axis(state: CheckpointState | None) -> _FoldInputs | None:
    """The axis an interrupted fold left behind, or ``None`` when there is none."""
    if state is None or not (state.path / AXIS).exists():
        return None
    arrays = load_npz(state.path / AXIS)
    old_to_new = arrays["old_to_new"]
    return _FoldInputs(
        table=KeyTable(keys=arrays["keys"], shared_index=arrays["shared_index"]),
        old_to_new=old_to_new if len(old_to_new) else None,
        canonical=CanonicalRawKeys(
            values=arrays["canonical_values"],
            raw=np.array(read_lines(state.path / AXIS_CANONICAL_RAW), dtype=RAW_KEY_DTYPE),
        ),
        off_panel=read_lines(state.path / AXIS_OFF_PANEL),
        shared_sorted=read_lines(state.path / AXIS_SHARED),
        dense_to_shared=arrays["dense_to_shared"].astype(np.int32),
        hg38_to_source=read_str_map(state.path / PROVENANCE_SOURCE),
    )


def _update_partition(
    prepared: _PreparedBuild,
    off_panel: list[str],
    shared_sorted: list[str],
    dense_to_shared: np.ndarray,
    hg38_to_source: dict[str, str | None],
) -> _PreparedBuild:
    """Return prepared build updated with final shared partition and provenance."""
    partition = replace(
        prepared.partition,
        off_panel_alids=off_panel,
        shared_sorted=shared_sorted,
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
) -> dict[str, AdmissionCounts]:
    """Route each study once, in this process, spilling the dense rows and the
    overflow associations it resolves (last-wins dedup per target index).
    Returns each Analysis's admitted-row dispositions (stores #175, #176)."""
    info_counts: dict[str, AdmissionCounts] = {}
    for i, row in enumerate(prepared.manifest_rows):
        dense, overflow, off_reference, counts = _resolve_column_hybrid(
            row.file_path,
            prepared.keys_sorted,
            prepared.targets_sorted,
            prepared.ispanel_sorted,
            row.se_divisor,
            capability=row.source_reader_capability,
            stored_effect_scale=row.stored_effect_scale,
            info_score_policy=prepared.info_score_policies[row.trait_id],
            maf_policy=prepared.maf_policies[row.trait_id],
        )
        info_counts[row.trait_id] = counts
        _spill_hybrid_column(
            prepared.spill_dir, analysis_index[row.trait_id], dense, overflow, off_reference
        )
        _log_progress("Pass 2", i + 1, n_analyses, pass2_start, f"last: {row.trait_id}", every=25)
    return info_counts


def _route_parallel(
    prepared: _PreparedBuild,
    analysis_index: dict[str, int],
    id_by_col: dict[int, str],
    n_analyses: int,
    options: _BuildOptions,
    pass2_start: float,
) -> dict[str, AdmissionCounts]:
    """Route each study through the fork pool. Workers read the routing arrays
    through the module-level globals below rather than as arguments: they are
    inherited by fork, which is what keeps a genome-scale lookup out of the
    per-column pickling the pool would otherwise do (dense.build_vcf's
    rationale). Each worker also applies its own Analysis's declared INFO and MAF
    policies (stores #175, #176) and returns its dispositions for the shared
    manifest."""
    global _pass2_keys_sorted, _pass2_targets_sorted, _pass2_ispanel_sorted
    global _pass2_spill_dir, _pass2_info_policies, _pass2_maf_policies
    _pass2_keys_sorted = prepared.keys_sorted
    _pass2_targets_sorted = prepared.targets_sorted
    _pass2_ispanel_sorted = prepared.ispanel_sorted
    _pass2_spill_dir = prepared.spill_dir
    _pass2_info_policies = {
        column: prepared.info_score_policies[row.trait_id]
        for column, row in enumerate(prepared.manifest_rows)
    }
    _pass2_maf_policies = {
        column: prepared.maf_policies[row.trait_id]
        for column, row in enumerate(prepared.manifest_rows)
    }
    info_counts: dict[str, AdmissionCounts] = {}
    try:
        with _fork_pool(options.n_workers) as pool:
            tasks = _pass2_worker_tasks(prepared.manifest_rows, analysis_index)
            futures = [pool.submit(_pass2_worker, task) for task in tasks]
            for i, future in enumerate(as_completed(futures)):
                col, counts = future.result()
                info_counts[id_by_col[col]] = counts
                _log_progress(
                    "Pass 2", i + 1, n_analyses, pass2_start, f"last: {id_by_col[col]}", every=25
                )
    finally:
        _pass2_keys_sorted = None
        _pass2_targets_sorted = None
        _pass2_ispanel_sorted = None
        _pass2_spill_dir = None
        _pass2_info_policies = None
        _pass2_maf_policies = None
    return info_counts


def _route_studies(
    prepared: _PreparedBuild,
    options: _BuildOptions,
) -> _RoutedSpills:
    """Phase - Pass 2: read each study once and route every association into
    the dense spill or the overflow spill (fork pool when n_workers > 1).
    Returns the {column: analysis_id} map the EAF survey keys, the pass start
    time the band writer's progress reports from, and each Analysis's declared-
    score dispositions for the shared manifest (stores #175)."""
    rows = prepared.manifest_rows
    analysis_index = {row.trait_id: i for i, row in enumerate(rows)}
    id_by_col = {i: row.trait_id for i, row in enumerate(rows)}
    n_analyses = len(rows)
    log.info("Pass 2: routing %d analyses (n_workers=%d)", n_analyses, options.n_workers)
    pass2_start = time.monotonic()
    if options.n_workers <= 1:
        info_counts = _route_serial(
            prepared,
            analysis_index,
            n_analyses,
            pass2_start,
        )
    else:
        info_counts = _route_parallel(
            prepared,
            analysis_index,
            id_by_col,
            n_analyses,
            options,
            pass2_start,
        )
    return _RoutedSpills(
        id_by_col=id_by_col, pass2_start=pass2_start, info_counts=info_counts
    )


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
        n_workers=options.n_workers,
    )
    overflow_survey = survey_eaf_spills(
        prepared.spill_dir,
        routed.id_by_col,
        prepared.partition.shared_sorted,
        shared_hashes,
        suffix=".ovf",
        index_key="variant_index",
        n_workers=options.n_workers,
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
    the Dense zarr skeleton under the plan and returns the effective chunks.

    This is the only site that measures the plan, and `_plan_phase` calls it
    only when no plan is recorded: a resumed run reloads the frozen one, because
    the Dense bands already on disk were written under exactly those codes.
    """
    encoding = StoreEncoding.decide(
        EncodingMeasurements(
            n_analyses=prepared.n_analyses, eaf=_combined_eaf(prepared, evidence)
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


def _combined_eaf(prepared: _PreparedBuild, evidence: _EafEvidence) -> EafMeasurements:
    """Both components' EAF measurements, merged into the plan's one input.

    Only a run that is *measuring* the plan has surveys; a resumed run reloads
    the plan instead of re-measuring it (issue #227).
    """
    dense, overflow = evidence.dense_survey, evidence.overflow_survey
    if dense is None or overflow is None:
        raise RuntimeError(
            "the encoding plan cannot be measured without both components' EAF surveys"
        )
    return combine_eaf_measurements(
        [
            dense.measurements(
                n_cells=prepared.partition.n_panel * prepared.n_analyses,
                n_variants=prepared.partition.n_panel,
            ),
            overflow.measurements(
                n_cells=overflow.n_spill_cells, n_variants=prepared.partition.n_shared
            ),
        ]
    )


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
        options.n_workers,
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
    options: _BuildOptions,
    *,
    consume_spills: bool = True,
) -> _OverflowAssembled:
    """Phase - assemble the Ragged Overflow CSR from the per-column overflow
    spills, in analysis order so CSR offsets align with analysis_index."""
    log.info("Assembling Ragged Overflow CSR from %d columns", prepared.n_analyses)
    csr, overflow_has_eaf = _assemble_overflow_csr(
        prepared.spill_dir,
        prepared.n_analyses,
        prepared.partition.n_shared,
        n_workers=options.n_workers,
        consume_spills=consume_spills,
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


def _write_overflow_eaf_plane(
    prepared: _PreparedBuild,
    plan: _EncodingPlan,
    overflow: _OverflowAssembled,
) -> None:
    """Phase - write the Ragged Overflow's frequency half before the SE fit.

    The `eaf` plane's encoding is decided before the joint fit runs, so writing
    the plane here lets both the fit and its byte measurement read the
    frequencies a reader will get back, instead of each re-encoding every cell
    (issue #232). The write also creates the component's zarr group exactly
    once; the later SE flush adds to that group rather than replacing it. A
    build failing in between is discarded whole by the staged release, and the
    group it leaves carries no `completion_state` for a later phase to mistake
    for a finished component."""
    log.info(
        "Ragged Overflow eaf plane: writing %d associations", overflow.csr.n_associations
    )
    overflow.csr.write_eaf_plane(prepared.staged.path, plan.encoding)


def _fit_joint_se(
    prepared: _PreparedBuild,
    plan: _EncodingPlan,
    overflow: _OverflowAssembled,
    options: _BuildOptions,
) -> tuple[StoreEncoding, np.ndarray | None]:
    """Phase - one SE model and one decision across both components. They
    partition the same Analyses, so fitting or gating either in isolation
    could leave the shared manifest describing only half of the data it
    governs. The Dense row chunks fit, measure and rewrite across
    ``--n-workers`` (issue #221); the Overflow cells arrive as a streamed
    source that reads its frequencies back from the plane
    ``_write_overflow_eaf_plane`` already wrote (issue #232)."""
    dense_group = prepared.dense_staged.arrays(mode="a")
    return optimise_dense_se_joint(
        dense_group,
        plan.encoding,
        # A streamed source, not the materialised bundle: on OGS-00011 the flat
        # planes are 15,078,327,210 cells, and building them cost 85.8 bytes a
        # cell -- 1.29 TB on a 1,006 GB host (issue #228).
        overflow=overflow.csr.se_fit_source(),
        n_workers=options.n_workers,
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
        n_workers=options.n_workers,
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
    """Flush the assembled overflow CSR's SE half into the zarr group the eaf
    write created, and build its top-hit index. The group already holds the
    frequency half; this adds to it rather than replacing it (issue #232).
    Returns the overflow association count the shared manifest's provenance
    records. Each step logs its start and elapsed time (issue #221)."""
    log.info("Ragged Overflow CSR flush: start (%d associations)", csr.n_associations)
    started = time.monotonic()
    csr.flush_se(staged.path, encoding, se_coefficients=se_coefficients)
    log.info("Ragged Overflow CSR flush: done in %s", format_duration(time.monotonic() - started))
    n_overflow = csr.n_associations
    log.info("Building Ragged Overflow top-hit index")
    build_ragged_top_hit_indexes(staged.path, encoding=encoding)
    return n_overflow


def _info_score_counts_fields(counts: AdmissionCounts) -> dict[str, int]:
    """One Analysis's admitted-row dispositions, as the store records them.

    The `associations_` prefix is load-bearing: the denominator is the
    associations the Analysis's Source Reader yielded after its own effect/SE
    admission, not the canonical rows the resolver's `canonical_rows_*`
    diagnostics count (stores #175). `associations_retained` is the rows the
    combined INFO-then-MAF rule admits, so it matches the resolver's
    `canonical_rows_retained` on the same population (stores #176).
    """
    info = counts.info
    return {
        "associations_observed": info.observed,
        "associations_retained": counts.admitted,
        "associations_below_threshold": info.below_threshold,
        "associations_missing": info.missing,
        "associations_malformed": info.malformed,
        "associations_nonfinite": info.nonfinite,
        "associations_out_of_range": info.out_of_range,
        "associations_usable": info.usable,
    }


def _info_score_provenance(
    prepared: _PreparedBuild, counts_by_id: Mapping[str, AdmissionCounts]
) -> dict[str, Any] | None:
    """The `provenance["info_score"]` block a store records (stores #175, #176).

    `None` when no Analysis's row carried an INFO policy -- an absent block
    means the manifest declared no policy at all, so a legacy table builds a
    manifest byte for byte as it did before this filter existed. When the block
    is written, every Analysis appears in table order, so the list is the whole
    table rather than only the rows a threshold dropped. A declared score with no
    usable value reports `no_usable_scores`; its rows are all retained.
    """
    policies = prepared.info_score_policies
    if all(policy.state is InfoScoreState.LEGACY_ABSENT for policy in policies.values()):
        return None
    return {
        "analyses": [
            {
                "analysis_id": row.trait_id,
                "info_score_state": declared_score_state(
                    policies[row.trait_id], counts_by_id[row.trait_id].info.usable
                ).value,
                "info_score_threshold": policies[row.trait_id].info_score_threshold,
                **_info_score_counts_fields(counts_by_id[row.trait_id]),
            }
            for row in prepared.manifest_rows
        ]
    }


def _maf_provenance(
    prepared: _PreparedBuild, counts_by_id: Mapping[str, AdmissionCounts]
) -> dict[str, Any] | None:
    """The `provenance["maf"]` block a store records (stores #176).

    `None` when no Analysis's row carried a numeric MAF threshold, so a table
    that never declares one builds exactly as it did before this filter
    existed. The counts are the associations the Source Reader yielded, the same
    population as the INFO block beside it; a row both filters would drop is
    counted under INFO, not here.
    """
    policies = prepared.maf_policies
    if all(policy.state is MafState.UNAVAILABLE for policy in policies.values()):
        return None
    return {
        "analyses": [
            {
                "analysis_id": row.trait_id,
                "maf_state": policies[row.trait_id].state.value,
                "maf_threshold": policies[row.trait_id].maf_threshold,
                "associations_below_threshold": counts_by_id[row.trait_id].maf_below_threshold,
                "associations_missing": counts_by_id[row.trait_id].maf_missing,
            }
            for row in prepared.manifest_rows
        ]
    }


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
        info_score=_info_score_provenance(prepared, components.info_counts),
        maf=_maf_provenance(prepared, components.info_counts),
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
    # The tables just written must be the tables the resolved map describes: a
    # partial harvest loss, an off-reference row dropped from the write, or a
    # table written from a stale map fails here, on the bytes a query reads,
    # rather than being trusted from the map that produced them (issue #255).
    # Root first, then the Dense Component -- its table was written before
    # Pass 2 and must carry the same identifiers as the shared table.
    require_written_rsids_match(
        prepared.staged.path, prepared.partition.shared_sorted, prepared.rsid_by_alid
    )
    require_written_rsids_match(
        prepared.dense_dir, prepared.partition.panel_sorted, prepared.rsid_by_alid
    )


def _prepare_build(
    staged: StagedRelease,
    manifest_rows: list[_ManifestRow],
    policies: DeclaredPolicies,
    options: _BuildOptions,
    spill_dir: Path | None = None,
) -> _PreparedBuild:
    """Seam - preparation: lifting, partition/routing and the Dense skeleton,
    then the routing index and spill directory. Nothing here reads a spill;
    the returned record is the whole handoff to the spill-lifetime seam.

    ``spill_dir`` names where the spills go: a checkpointed build hands in its
    own directory, so a failure leaves them where the resume looks for them
    (issue #227), and everything else gets this invocation's private mkdtemp.
    """
    axis = _lift_and_partition(staged, manifest_rows, options)
    dense_to_shared = _write_dense_component_skeleton(
        staged,
        axis,
        options.chunk_shape,
    )
    if spill_dir is None:
        spill_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{options.out.name}.hybridspill.",
                dir=staged.path.parent,
            )
        )
    else:
        spill_dir.mkdir(parents=True, exist_ok=True)
    return _PreparedBuild(
        staged=staged,
        dense_dir=axis.dense_dir,
        dense_staged=axis.dense_staged,
        partition=axis.partition,
        manifest_rows=manifest_rows,
        info_score_policies=policies.info,
        maf_policies=policies.maf,
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
    state: CheckpointState | None = None,
) -> tuple[_PreparedBuild, _ComponentResult]:
    """Seam - the spill-lifetime build: Pass 2 routing, EAF verification,
    joint encoding, the component writes (Dense bands, Overflow CSR, shared SE
    fit, Dense top hits/manifest/analyses.tsv). Returns the (possibly
    repartitioned) prepared build alongside the components, because a
    variant-reference build learns its off-reference variants only here.

    The spill directory is removed once the usable phases have read it, and the
    store's files are only touched while the spills exist. A `None` state is a
    build that did not ask for a checkpoint: its spills are removed whichever
    phase fails, exactly as they always were. With a checkpoint they are kept
    on failure -- they sit inside it, and the resumed build reads them (issue
    #227).
    """
    spill_dir = prepared.spill_dir
    try:
        prepared, components = _run_build_phases(prepared, options, state)
    except BaseException:
        if state is None:
            shutil.rmtree(spill_dir, ignore_errors=True)
        raise
    shutil.rmtree(spill_dir, ignore_errors=True)
    return prepared, components


def _run_build_phases(
    prepared: _PreparedBuild,
    options: _BuildOptions,
    state: CheckpointState | None,
) -> tuple[_PreparedBuild, _ComponentResult]:
    """Walk the build's phases, re-entering at the one the checkpoint recorded.

    A phase whose completion the checkpoint records is reloaded from it; one it
    does not is computed, its product recorded and its marker set. The tail
    after the recorded phases runs wholesale either way. Without a checkpoint
    every phase is computed, which is the build this seam has always run.
    """
    routed = _pass2_phase(prepared, options, state)
    prepared = _fold_phase(prepared, options, state)
    evidence = _orientation_phase(prepared, routed, options, state)
    plan = _plan_phase(prepared, evidence, options, state)
    dense = _dense_bands_phase(prepared, plan, routed, options, state)
    return _tail_phase(prepared, routed, plan, dense, evidence, options, state)


def _pass2_phase(
    prepared: _PreparedBuild, options: _BuildOptions, state: CheckpointState | None
) -> _RoutedSpills:
    """Phase - Pass 2: route each study once, or reload the recorded routing.

    The recorded form is each Analysis's declared-score dispositions (stores
    #175), which the shared manifest's `provenance.info_score` is written from
    and which nothing else can recover: they are what the sources yielded, not
    a function of the store.
    """
    if state is not None and state.has("pass2"):
        return _recorded_routing(prepared, state)
    routed = _route_studies(prepared, options)
    _log_spill_bytes(prepared.spill_dir)
    if state is not None:
        _record_pass2_product(state, prepared, routed)
    return routed


def _log_spill_bytes(spill_dir: Path) -> None:
    """Log the off-reference scratch Pass 2 has just spilled (issue #218)."""
    encoded_bytes, side_bytes = _off_reference_spill_bytes(spill_dir)
    log.info(
        "Pass 2 off-reference spill: %.2f GiB encoded keys + %.2f GiB raw side files",
        encoded_bytes / 2**30,
        side_bytes / 2**30,
    )


def _record_pass2_product(
    state: CheckpointState, prepared: _PreparedBuild, routed: _RoutedSpills
) -> None:
    """Record Pass 2's product: the dispositions its workers returned.

    The plates are recorded too, with their sizes, so a resume can refuse a
    spill directory that lost one rather than build a store short of the
    associations it claims.
    """
    write_json(
        state.path / INFO_COUNTS,
        {
            row.trait_id: routed.info_counts[row.trait_id].as_dict()
            for row in prepared.manifest_rows
        },
    )
    record_plates(state.path, (path.name for path in state.spill_dir.iterdir()))
    mark_phase(state.path, "pass2")


def _recorded_routing(prepared: _PreparedBuild, state: CheckpointState) -> _RoutedSpills:
    """Rebuild the routing records a resumed run does not route again."""
    recorded = read_json(state.path / INFO_COUNTS)
    return _RoutedSpills(
        id_by_col={col: row.trait_id for col, row in enumerate(prepared.manifest_rows)},
        pass2_start=time.monotonic(),
        info_counts={
            row.trait_id: AdmissionCounts.from_dict(recorded[row.trait_id])
            for row in prepared.manifest_rows
        },
    )


def _fold_phase(
    prepared: _PreparedBuild, options: _BuildOptions, state: CheckpointState | None
) -> _PreparedBuild:
    """Phase - the off-reference fold, or the axis a completed fold left.

    When the fold is recorded the incoming prepared build already carries the
    post-fold axis: `_resume_prepared` rebuilt it from the checkpoint rather
    than re-deriving it, so there is nothing left to fold. When it is not, the
    fold runs -- under the recorded axis if the interrupted run got as far as
    writing one, and under a freshly resolved one otherwise.
    """
    if state is None:
        return _finalise_reference_partition(prepared, options)
    if state.has("fold"):
        return prepared
    prepared = _finalise_reference_partition(prepared, options, state)
    record_plates(
        state.path,
        [
            *(f"{col}.npz" for col in range(prepared.n_analyses)),
            *(f"{col}.ovf.npz" for col in range(prepared.n_analyses)),
        ],
    )
    mark_phase(state.path, "fold")
    return prepared


def _orientation_phase(
    prepared: _PreparedBuild,
    routed: _RoutedSpills,
    options: _BuildOptions,
    state: CheckpointState | None,
) -> _EafEvidence:
    """Phase - EAF orientation, or the report a resumed run reloads.

    The report is reloaded only when the plan it fed is recorded as well: a
    crash inside the plan phase leaves it to be measured again, and the plan is
    measured from these surveys, so they are recomputed on the spills that are
    still there. Recomputing the report is safe -- it is a pure function of the
    same retained frequencies -- and the recorded one is not rewritten.
    """
    if state is not None and state.has("orientation") and state.has("plan"):
        return _recorded_evidence(state)
    evidence = _verify_eaf_orientation(prepared, routed, options)
    if state is not None and not state.has("orientation"):
        write_json(state.path / ORIENTATION, _report_payload(evidence.report))
        mark_phase(state.path, "orientation")
    return evidence


def _recorded_evidence(state: CheckpointState) -> _EafEvidence:
    """The orientation report alone, which is all a resumed run still reads."""
    return _EafEvidence(
        dense_survey=None,
        overflow_survey=None,
        report=_report_from_payload(read_json(state.path / ORIENTATION)),
    )


def _plan_phase(
    prepared: _PreparedBuild,
    evidence: _EafEvidence,
    options: _BuildOptions,
    state: CheckpointState | None,
) -> _EncodingPlan:
    """Phase - the joint encoding plan, frozen before the first band write.

    The plan is measured from the data and the Dense bands are written under it,
    so a resumed run reloads it verbatim: re-measuring could choose different
    codes for cells already written, which no later check would catch. The order
    is what makes that safe -- the marker is set after the plan (and the Dense
    zarr skeleton it creates) is on disk, and the first band write happens after
    that, so a plan that has to be measured again has no bands to contradict.
    """
    if state is not None and state.has("plan"):
        return _recorded_plan(state)
    plan = _plan_joint_encoding(prepared, evidence, options)
    if state is not None:
        write_json(
            state.path / PLAN,
            {
                "encoding": plan.encoding.to_manifest(),
                "effective_chunks": list(plan.effective_chunks),
            },
        )
        mark_phase(state.path, "plan")
    return plan


def _recorded_plan(state: CheckpointState) -> _EncodingPlan:
    """The plan a resumed run must write its remaining bands under."""
    recorded = read_json(state.path / PLAN)
    return _EncodingPlan(
        encoding=StoreEncoding.from_manifest(recorded["encoding"]),
        effective_chunks=tuple(recorded["effective_chunks"]),
    )


def _dense_bands_phase(
    prepared: _PreparedBuild,
    plan: _EncodingPlan,
    routed: _RoutedSpills,
    options: _BuildOptions,
    state: CheckpointState | None,
) -> _DenseWritten:
    """Phase - the Dense band write, or the harvest a completed write left.

    The band write unlinks each column's dense spill once both its passes have
    read it, so its top-hit harvest has to be recorded: a resumed run cannot
    re-write the bands from spills that are gone, and the finish phase needs the
    candidates whether the bands were just written or written hours ago.
    """
    if state is not None and state.has("dense_bands"):
        return _recorded_hits(state)
    dense = _write_dense_component_bands(prepared, plan, routed.pass2_start, options)
    if state is not None:
        save_npz(
            state.path / HITS,
            all_rows=dense.all_rows,
            all_cols=dense.all_cols,
            all_z=dense.all_z,
            all_se=dense.all_se,
            column_has_eaf=dense.column_has_eaf,
        )
        record_plates(
            state.path, (f"{col}.ovf.npz" for col in range(prepared.n_analyses))
        )
        mark_phase(state.path, "dense_bands")
    return dense


def _recorded_hits(state: CheckpointState) -> _DenseWritten:
    """The Dense band write's own products, as the finish phase reads them."""
    arrays = load_npz(state.path / HITS)
    return _DenseWritten(
        all_rows=arrays["all_rows"],
        all_cols=arrays["all_cols"],
        all_z=arrays["all_z"],
        all_se=arrays["all_se"],
        column_has_eaf=arrays["column_has_eaf"],
    )


def _tail_phase(
    prepared: _PreparedBuild,
    routed: _RoutedSpills,
    plan: _EncodingPlan,
    dense: _DenseWritten,
    evidence: _EafEvidence,
    options: _BuildOptions,
    state: CheckpointState | None,
) -> tuple[_PreparedBuild, _ComponentResult]:
    """Phase - everything after the Dense band write, re-run wholesale.

    No marker, deliberately (issue #227): the CSR assembly, the frequency
    plane, the joint SE fit, the Dense finish and the Overflow flush are cheap
    beside the phases above once the spills and the plan are in hand, and each
    is idempotent over what a failed attempt left -- the frequency plane is
    recreated, zarr chunk writes and the top-hit tiers are replaced, the
    manifest and analyses are rewritten. The `.ovf` plates survive a
    checkpointed run precisely so the CSR can be re-assembled here.
    """
    overflow = _assemble_overflow(prepared, options, consume_spills=state is None)
    analyses = _stamp_analyses(prepared, dense, overflow, evidence)
    _write_overflow_eaf_plane(prepared, plan, overflow)
    encoding, se_coefficients = _fit_joint_se(prepared, plan, overflow, options)
    eaf_provenance = _finish_dense_component(
        prepared, dense, analyses, encoding, evidence, options
    )
    return prepared, _ComponentResult(
        csr=overflow.csr,
        encoding=encoding,
        se_coefficients=se_coefficients,
        eaf_provenance=eaf_provenance,
        analyses=analyses,
        info_counts=routed.info_counts,
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
    source_assembly: str | None = None, checkpoint: bool = False,
    resume: bool = False,
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
    carry; an inconsistent panel defers to the reference). With a reference,
    Pass 1 variant discovery is bypassed (single-pass build, issue #186) and
    the reference's source-coordinate map routes on-reference associations to
    the Dense Component and off-reference ones to the Ragged Overflow during
    Pass 2. With only ``reference_panel``, the legacy two-pass build reads every
    source once and lifts hg19 rows; a reference that names no rsids runs Pass 1's
    harvest for them (`_reference_axis_rsids`, #255). Rows are assumed hg19 and
    lifted inline unless the manifest declares ``source_assembly=hg38`` (#85);
    ``source_assembly`` and ``source_reader_capability`` supply per-release
    defaults (#174), and ``eaf_reference`` drives the orientation check (#115).

    ``checkpoint=True`` opts the build into phase-granularity resume; ``resume=True``
    continues one from the checkpoint this call's ``output_path`` implies (issue #227;
    `_run_checkpointed_build`, `_resume_requested`), and a destination whose checkpoint
    is still there is refused unless ``overwrite=True`` discards it.
    """
    if reference_panel is None and variant_reference is None:
        raise ValueError("build-hybrid needs --reference-panel or --variant-reference")
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
    defaults = _ManifestDefaults(source_reader_capability, source_assembly)
    if resume:
        return _resume_requested(manifest_path, options, defaults, overwrite)
    return _run_checkpointed_build(manifest_path, options, defaults, overwrite, checkpoint)


def _run_checkpointed_build(
    manifest_path: str | Path,
    options: _BuildOptions,
    defaults: _ManifestDefaults,
    overwrite: bool,
    checkpoint: bool,
) -> HybridBuildResult:
    """Run one build, keeping its phases in a checkpoint when asked to.

    The spill directory a checkpointed build uses is the checkpoint's own, so a
    failure leaves the spills where the resume looks for them, and the Staged
    Release work directory is retained inside the checkpoint instead of removed
    (issue #227).
    """
    manifest_rows, policies = _load_manifest(
        manifest_path,
        default_source_reader_capability=defaults.source_reader_capability,
        default_source_assembly=defaults.source_assembly,
    )
    state = None
    if checkpoint:
        state = _open_checkpoint(manifest_path, options, defaults, overwrite)
    else:
        # A build that is not continuing a checkpoint must not orphan one in
        # silence: refusing is the default, --overwrite the explicit discard.
        _require_clear_checkpoint(options.out, overwrite)
    try:
        # The whole window is guarded, publication included: a commit that
        # loses a race still leaves a checkpoint, and the operator is told so.
        with OpenGWASDBStore.staging(
            options.out,
            overwrite=overwrite,
            retain_on_failure_to=None if state is None else state.staged_dir,
        ) as staged:
            prepared = _prepare_build(
                staged,
                manifest_rows,
                policies,
                options,
                None if state is None else state.spill_dir,
            )
            result = _run_and_publish(prepared, options, state)
    except BaseException:
        _log_failed_checkpoint(state)
        raise
    if state is not None:
        state.discard()
    return result


def _run_and_publish(
    prepared: _PreparedBuild, options: _BuildOptions, state: CheckpointState | None
) -> HybridBuildResult:
    """Run the phase sequence over the open release, then finalise the store."""
    prepared, components = _build_components(prepared, options, state)
    return _finalise_store(prepared, components, options)


@dataclass(frozen=True)
class _ManifestDefaults:
    """The per-release manifest defaults the caller supplied, recorded with
    every other build parameter because they decide what each Manifest Row
    means -- and so what the store holds (issue #174)."""

    source_reader_capability: str | None
    source_assembly: str | None


# ── Checkpointed build and resume (issue #227) ──────────────────────────────


def _resume_requested(
    manifest_path: str | Path,
    options: _BuildOptions,
    defaults: _ManifestDefaults,
    overwrite: bool,
) -> HybridBuildResult:
    """Resume the checkpoint ``output_path`` implies, refusing a different request.

    The requested parameters are compared against the recorded ones before
    anything is read from the checkpoint: a resumed run writes into a store the
    first run configured, and a parameter that differs would mix records from
    one configuration with the rest of a store built under another.
    """
    checkpoint_dir = checkpoint_dir_for(options.out)
    recorded = read_build_params(checkpoint_dir)
    require_matching_params(recorded, _build_params(manifest_path, options, defaults, overwrite))
    return resume_hybrid_build(checkpoint_dir, n_workers=options.n_workers)


def resume_hybrid_build(
    checkpoint_dir: str | Path, *, n_workers: int | None = None
) -> HybridBuildResult:
    """Resume an interrupted checkpointed Hybrid build from its checkpoint.

    Takes only the checkpoint directory (ADR 0023): every build parameter and
    every external input's identity rides in the ``build_params.json`` the first
    run wrote, so a resumed run cannot silently apply a different configuration
    than the records on disk were computed under. ``n_workers`` is the one
    exception and the only one -- a pure runtime knob no computed value depends
    on.

    Re-enters at the phase the checkpoint records as last complete, reloading
    the frozen encoding plan and the post-Pass-2 axis rather than re-measuring
    either, and re-running the tail wholesale. It publishes through the same
    Staged Release commit an uninterrupted build uses -- adopting the release
    the failed run left, rather than writing beside it -- and removes the
    checkpoint once the release is published.
    """
    path = Path(checkpoint_dir)
    params = read_build_params(path)
    _require_unchanged_inputs(params)
    options = _options_from_params(params, n_workers)
    defaults = _ManifestDefaults(
        params.get("source_reader_capability"), params.get("source_assembly")
    )
    rows, policies = _load_manifest(
        params["manifest_path"],
        default_source_reader_capability=defaults.source_reader_capability,
        default_source_assembly=defaults.source_assembly,
    )
    state = CheckpointState(path=path, params=params, completed=completed_phases(path))
    if not state.staged_dir.exists():
        raise FileNotFoundError(
            f"The checkpoint at {path} holds no staged release to resume: "
            f"{state.staged_dir} is absent, so the Dense Component it had written "
            f"is gone. Start the build again with --checkpoint."
        )
    _require_intact_plates(state, len(rows))
    try:
        with OpenGWASDBStore.staging(
            Path(params["output_path"]),
            overwrite=bool(params["overwrite"]),
            adopt=state.staged_dir,
            retain_on_failure_to=state.staged_dir,
        ) as staged:
            prepared = _resume_prepared(staged, rows, policies, options, state)
            result = _run_and_publish(prepared, options, state)
    except BaseException:
        _log_failed_checkpoint(state)
        raise
    state.discard()
    return result


def _resume_prepared(
    staged: StagedRelease,
    manifest_rows: list[_ManifestRow],
    policies: DeclaredPolicies,
    options: _BuildOptions,
    state: CheckpointState,
) -> _PreparedBuild:
    """The prepared build a resumed phase sequence re-enters with.

    Before the fold it is `_prepare_build`'s own output: the routing index and
    the pre-fold partition are pure functions of the recorded inputs, and the
    fold's own records are the checkpoint's. After the fold it is rebuilt from
    the recorded axis instead -- no Pass 1 and no measurement -- because the
    Dense rows and the Overflow plates are already keyed on that axis.
    """
    if state.has("fold"):
        return _recorded_prepared(staged, manifest_rows, policies, state)
    return _prepare_build(staged, manifest_rows, policies, options, state.spill_dir)


def _recorded_prepared(
    staged: StagedRelease,
    manifest_rows: list[_ManifestRow],
    policies: DeclaredPolicies,
    state: CheckpointState,
) -> _PreparedBuild:
    """Rebuild the prepared build from the checkpoint's recorded axis alone.

    The routing index is deliberately empty: Pass 2 and the fold are its only
    readers, and both are recorded complete when this runs -- a run that
    reaches either re-derives it in `_prepare_build` instead. The Dense row map
    is re-saved from the record, so the release being resumed carries the map
    the record's indices were built against rather than whatever it had left.
    """
    # The Dense Component's staging directory already exists -- it is the one
    # the interrupted run wrote its bands into -- so it is opened, not made.
    dense_dir, dense_staged = _open_dense_component(staged)
    panel = read_lines(state.path / AXIS_PANEL)
    off_panel = read_lines(state.path / AXIS_OFF_PANEL)
    shared_sorted = read_lines(state.path / AXIS_SHARED)
    dense_to_shared = load_npz(state.path / AXIS)["dense_to_shared"].astype(np.int32)
    np.save(dense_to_shared_path(staged.path), dense_to_shared)
    return _PreparedBuild(
        staged=staged,
        dense_dir=dense_dir,
        dense_staged=dense_staged,
        partition=_VariantPartition(
            panel_sorted=panel,
            off_panel_alids=off_panel,
            shared_sorted=shared_sorted,
            n_panel=len(panel),
            n_off_panel=len(off_panel),
            n_shared=len(shared_sorted),
        ),
        manifest_rows=manifest_rows,
        info_score_policies=policies.info,
        maf_policies=policies.maf,
        analyses=[_manifest_row_to_analysis(row) for row in manifest_rows],
        hg38_to_source=read_str_map(state.path / PROVENANCE_SOURCE),
        rsid_by_alid={
            alid: rsid
            for alid, rsid in read_str_map(state.path / PROVENANCE_RSID).items()
            if rsid
        },
        dense_to_shared=dense_to_shared,
        spill_dir=state.spill_dir,
        keys_sorted=_NO_ROUTING[0],
        targets_sorted=_NO_ROUTING[1],
        ispanel_sorted=_NO_ROUTING[2],
        n_analyses=len(manifest_rows),
    )


def _require_intact_plates(state: CheckpointState, n_analyses: int) -> None:
    """Refuse a checkpoint whose retained plates the phases still to run need.

    Which plates those are follows from the recorded phase: before the fold
    the `.unk` plates of the columns it has not folded (and every column's
    dense spill, which the band write reads), through the band write every
    column's dense spill, and after it the `.ovf` plates the CSR is re-assembled
    from. A plate the inventory never recorded is one that was not on disk when
    its phase completed, which is ordinary -- a column can spill no off-reference
    key at all -- so only recorded plates are checked.

    With no phase recorded there is nothing to check: Pass 2 writes every plate
    there is, over whatever the failed attempt left behind.
    """
    if not state.completed:
        return
    require_intact_plates(state.path, _plate_names(state, n_analyses))


def _plate_names(state: CheckpointState, n_analyses: int) -> list[str]:
    """The plates the phases after the recorded one will read."""
    dense = [f"{col}.npz" for col in range(n_analyses)]
    if state.has("fold"):
        overflow = [f"{col}.ovf.npz" for col in range(n_analyses)]
        return overflow if state.has("dense_bands") else [*dense, *overflow]
    unknown: list[str] = []
    for col in _pending_columns(state, n_analyses):
        unknown.extend((f"{col}.unk.npz", f"{col}.unk.raw"))
    return [*dense, *unknown]


def _pending_columns(state: CheckpointState, n_analyses: int) -> list[int]:
    """The columns an interrupted fold has still to fold."""
    folded = _folded_columns(state)
    return [col for col in range(n_analyses) if col not in folded]


def _require_unchanged_inputs(params: Mapping[str, Any]) -> None:
    """Refuse a checkpoint whose recorded inputs are no longer what they were.

    A resume's *parameters* come from the record, so there is nothing to compare
    them against -- but its inputs are still the operator's files, and a manifest
    edited or a reference rewritten between the two runs would have the resumed
    phases finish a store the first run's measured values no longer describe.
    """
    for name, identity in sorted(params.get("inputs", {}).items()):
        if input_identity(identity["path"]) != identity:
            raise ValueError(
                f"The checkpoint at {params['output_path']} was built from a different "
                f"{name} ({identity['path']}), which has changed since. Resume it only "
                f"over the inputs it was built from, or start again with --checkpoint."
            )


def _require_clear_checkpoint(out: Path, overwrite: bool) -> None:
    """Refuse to build for a destination whose checkpoint is still there.

    Called by *every* build, checkpointed or not: a checkpoint is a release's
    only copy of a Dense Component it had already written -- hundreds of
    gigabytes at release scale -- so a build that is not the resume of it is
    either an explicit ``--overwrite`` or a refusal naming the function that
    continues it. Without a checkpoint directory this says nothing, so a plain
    build with nothing to resume behaves exactly as it always did.
    """
    require_fresh_destination(out, checkpoint_dir_for(out), overwrite, RESUME_FUNCTION)


def _open_checkpoint(
    manifest_path: str | Path,
    options: _BuildOptions,
    defaults: _ManifestDefaults,
    overwrite: bool,
) -> CheckpointState:
    """Create the checkpoint a build that opted in writes its phases into,
    refusing a stale one first (`_require_clear_checkpoint`)."""
    _require_clear_checkpoint(options.out, overwrite)
    checkpoint_dir = checkpoint_dir_for(options.out)
    checkpoint_dir.mkdir(parents=True)
    params = _build_params(manifest_path, options, defaults, overwrite)
    write_build_params(checkpoint_dir, params)
    return CheckpointState(path=checkpoint_dir, params=params, completed=())


def _build_params(
    manifest_path: str | Path,
    options: _BuildOptions,
    defaults: _ManifestDefaults,
    overwrite: bool,
) -> dict[str, Any]:
    """The parameters and input identities a checkpoint records.

    The path of every external input, with its size, mtime and SHA-256: a
    checkpoint describes one build of one set of inputs, and a resume against
    an edited manifest or a rewritten reference would produce a store belonging
    to neither.
    """
    def _text(value: str | Path | None) -> str | None:
        return None if value is None else str(value)

    return {
        "manifest_path": str(Path(manifest_path)),
        "output_path": str(options.out.resolve()),
        "store_id": options.store_id,
        "release_id": options.release_id,
        "reference_panel": _text(options.reference_panel),
        "variant_reference": _text(options.variant_reference),
        "chain_file": _text(options.chain_file),
        "liftover_failure_threshold": options.liftover_failure_threshold,
        "chunk_shape": list(options.chunk_shape),
        "dtype": options.dtype,
        "n_workers": options.n_workers,
        "eaf_reference": _text(options.eaf_reference),
        "eaf_reference_ancestry": options.eaf_reference_ancestry,
        "allow_unverified_eaf": options.allow_unverified_eaf,
        "overwrite": bool(overwrite),
        "source_reader_capability": defaults.source_reader_capability,
        "source_assembly": defaults.source_assembly,
        "inputs": input_identities(
            {
                "manifest": manifest_path,
                "variant_reference": options.variant_reference,
                "reference_panel": options.reference_panel,
                "chain_file": options.chain_file,
                "eaf_reference": options.eaf_reference,
            }
        ),
    }


def _options_from_params(params: Mapping[str, Any], n_workers: int | None) -> _BuildOptions:
    """Rebuild the build options a checkpoint recorded.

    ``n_workers`` comes from the caller: it is the one parameter a resume may
    change (ADR 0023).
    """
    return _BuildOptions(
        out=Path(params["output_path"]),
        reference_panel=params["reference_panel"],
        variant_reference=params["variant_reference"],
        store_id=params["store_id"],
        release_id=params["release_id"],
        chain_file=params["chain_file"],
        liftover_failure_threshold=params["liftover_failure_threshold"],
        chunk_shape=tuple(params["chunk_shape"]),
        dtype=params["dtype"],
        n_workers=params["n_workers"] if n_workers is None else n_workers,
        eaf_reference=params["eaf_reference"],
        eaf_reference_ancestry=params["eaf_reference_ancestry"],
        allow_unverified_eaf=params["allow_unverified_eaf"],
    )


def _log_failed_checkpoint(state: CheckpointState | None) -> None:
    """Name the checkpoint a failed build left and how to carry on from it.

    The failure itself propagates; this line is what tells an operator that
    hours of work are still on disk and what to type to use them.
    """
    if state is None:
        return
    log.error(
        "Hybrid build failed; its checkpoint is at %s. Resume it with "
        "%s(%r).",
        state.path,
        RESUME_FUNCTION,
        str(state.path),
    )


def _report_payload(report: EafOrientationReport) -> dict[str, Any]:
    """The orientation report, losslessly, for the checkpoint.

    `provenance()` rounds each correlation for the manifest -- provenance is not
    a computation input. A resumed run stamps every Analysis from this report,
    so it is recorded at full precision: a rounded value would put a different
    number in `analyses.tsv` than an uninterrupted build.
    """
    return {
        "method": report.method.value,
        "reference_id": report.reference_id,
        "reference_checksum": report.reference_checksum,
        "n_reference_variants": int(report.n_reference_variants),
        "n_sites": int(report.n_sites),
        "min_overlap": int(report.min_overlap),
        "min_variance": float(report.min_variance),
        "evidence": [
            {
                "analysis_id": item.analysis_id,
                "outcome": item.outcome.value,
                "n_overlap": int(item.n_overlap),
                "r": repr(float(item.r)),
                "stores_eaf": bool(item.stores_eaf),
                "note": item.note,
            }
            for item in report.evidence
        ],
    }


def _report_from_payload(payload: Mapping[str, Any]) -> EafOrientationReport:
    """The report `_report_payload` recorded, restored exactly."""
    return EafOrientationReport(
        method=EafOrientationMethod(payload["method"]),
        evidence=tuple(
            OrientationEvidence(
                analysis_id=item["analysis_id"],
                outcome=EafOrientationOutcome(item["outcome"]),
                n_overlap=int(item["n_overlap"]),
                r=float(item["r"]),
                note=item["note"],
                stores_eaf=bool(item["stores_eaf"]),
            )
            for item in payload["evidence"]
        ),
        reference_id=payload["reference_id"],
        reference_checksum=payload["reference_checksum"],
        n_reference_variants=int(payload["n_reference_variants"]),
        n_sites=int(payload["n_sites"]),
        min_overlap=int(payload["min_overlap"]),
        min_variance=float(payload["min_variance"]),
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
    info_score: dict[str, Any] | None = None,
    maf: dict[str, Any] | None = None,
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
    if info_score is not None:
        provenance["info_score"] = info_score
    if maf is not None:
        provenance["maf"] = maf
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
