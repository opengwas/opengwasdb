"""Completion's fill shards and its duplicate rule, shared by Dense and Ragged.

A block checkpoint records each fill by ALID, because the union variant axis is
built by the parent. Phase 3 resolves those ALIDs to axis rows once and writes
each fill as a compact ``(row, analysis, z, se)`` record to a shard file on
disk, so it never holds every fill of every block in memory (1954e5d; issue
#424). The layouts differ only in the key a fill is sharded by, the unit each
writes in: Dense writes the completed grid a variant-row band at a time
(`by_row_band`), Ragged writes its association sequence an Analysis at a time
(`by_analysis_batch`).

A cell two blocks fill is resolved by Association Admission's duplicate rule
(D5, ADR 0072) in one place, `apply_fill_batch`: fills that agree on z and SE
are one fill, and fills that disagree are all dropped, leaving the cell missing
(`clear_conflicted`), every dropped fill logged by `log_dropped` (D8). The
block that starts at a shared endpoint alone fills it (`shared_endpoints`), so
two fills reach a cell only where two blocks overlap elsewhere.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.completion.checkpoint import ALID_DTYPE
from opengwasdb.variants.routing import collapse_duplicates

log = logging.getLogger(__name__)

FILL_RECORD_DTYPE = np.dtype(
    [("row", np.int32), ("ai", np.int32), ("z", np.float32), ("se", np.float32)]
)
FILL_RECORD_READ_COUNT = 5_000_000

#: The imputed-band value of a cell two fills disagree on, until the band's
#: fills are all applied and the cell is left missing.
CONFLICTED = 2

_QUALITY_BATCH_SIZE = 100_000

#: The shard each fill record goes to, from its axis rows and Analysis indices.
ShardKey = Callable[[np.ndarray, np.ndarray], np.ndarray]

_QualityRow = tuple[int, str, float | None, int, int]


def by_row_band(band_rows: int) -> ShardKey:
    """Shard by variant-row band: Dense writes its grid `band_rows` rows at a time."""

    def key(rows: np.ndarray, _ai: np.ndarray) -> np.ndarray:
        return rows // band_rows

    return key


def by_analysis_batch(batch_of: np.ndarray) -> ShardKey:
    """Shard by Analysis batch: Ragged writes its sequence an Analysis at a time,
    and `batch_of[analysis]` is the batch whose shard holds that Analysis's fills."""

    def key(_rows: np.ndarray, ai: np.ndarray) -> np.ndarray:
        batches: np.ndarray = batch_of[ai]
        return batches

    return key


def sorted_union_map(new_alid_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """The union ALID → row lookup, sorted by ALID, so checkpoint fills (which
    record ALIDs, not rows) resolve to union rows by binary search."""
    union_alids = np.fromiter(
        (alid.encode("ascii") for alid in new_alid_to_idx),
        dtype=ALID_DTYPE,
        count=len(new_alid_to_idx),
    )
    union_rows = np.fromiter(new_alid_to_idx.values(), dtype=np.int32, count=len(new_alid_to_idx))
    o = np.argsort(union_alids)
    return union_alids[o], union_rows[o]


def fill_shard_path(fill_shard_dir: Path, band_index: int) -> Path:
    return fill_shard_dir / f"band-{band_index:06d}.bin"


def iter_fill_records(path: Path) -> Iterator[np.ndarray]:
    if not path.exists():
        return
    with open(path, "rb") as fh:
        while True:
            records = np.fromfile(fh, dtype=FILL_RECORD_DTYPE, count=FILL_RECORD_READ_COUNT)
            if len(records) == 0:
                break
            yield records


def insert_completion_quality_batch(db: Any, batch: list[_QualityRow]) -> None:
    if not batch:
        return
    db.executemany(
        "INSERT INTO completion_quality "
        "(analysis_index, block_id, pearson_r, n_imputed, n_missing) "
        "VALUES (?, ?, ?, ?, ?)",
        batch,
    )
    db.commit()


def shard_checkpoint_fills(
    blocks_dir: Path,
    staged: Any,
    union_alids_s: np.ndarray,
    union_rows_s: np.ndarray,
    n_variants: int,
    shard_key: ShardKey,
    impute_mask: np.ndarray | None = None,
) -> tuple[Path, int]:
    """Resolve checkpoint fills into raw shard files, one per `shard_key` value.

    Checkpoints store fill rows by ALID because the union variant axis is built
    by the parent. This pass resolves those ALIDs once, writes compact
    ``(row, analysis, z, se)`` records to per-shard files, and streams
    completion_quality directly into SQLite.

    ``impute_mask`` (bool per analysis; ``None`` = impute all) is applied here,
    because this is where a worker's *candidate* fills become the release's.
    The blocks are imputed for every Analysis and the ancestry-match filter
    (ADR 0028) is decided afterwards, so a nonmatching Analysis arrives with
    both fills and completion-quality rows. Dropping only the fills, later, at
    the write, left ``completion_quality`` -- and through it ``analyses.tsv``'s
    ``completion_n_imputed_total`` and ``eaf_scope`` -- counting cells the
    release does not contain. Filtered once, here, the table and the arrays
    cannot disagree.
    """
    fill_shard_dir = staged.path / "fill_shards"
    if fill_shard_dir.exists():
        shutil.rmtree(fill_shard_dir)
    fill_shard_dir.mkdir()

    quality_count = 0
    quality_batch: list[_QualityRow] = []

    with staged.index_connection() as dst_db:
        for ckpt in sorted(blocks_dir.glob("*.npz")):
            with np.load(ckpt, allow_pickle=False) as d:
                rows = _checkpoint_quality_rows(d, impute_mask)
                quality_count += len(rows)
                quality_batch.extend(rows)
                if len(quality_batch) >= _QUALITY_BATCH_SIZE:
                    insert_completion_quality_batch(dst_db, quality_batch)
                    quality_batch.clear()
                fills = _resolved_fills(d, union_alids_s, union_rows_s, n_variants, impute_mask)
                if fills is not None:
                    _append_to_shards(fill_shard_dir, fills, shard_key(fills[0], fills[1]))

        insert_completion_quality_batch(dst_db, quality_batch)

    return fill_shard_dir, quality_count


def _checkpoint_quality_rows(d: Any, impute_mask: np.ndarray | None) -> list[_QualityRow]:
    """One checkpoint's completion_quality rows, the filtered-out Analyses' left out."""
    bid = str(d["block_id"][0])
    return [
        (int(ai), bid, None if not np.isfinite(p) else float(p), int(ni), int(nm))
        for ai, p, ni, nm in zip(
            d["q_ai"], d["q_pearson"], d["q_nimp"], d["q_nmiss"], strict=True
        )
        if impute_mask is None or impute_mask[int(ai)]
    ]


def _resolved_fills(
    d: Any,
    union_alids_s: np.ndarray,
    union_rows_s: np.ndarray,
    n_variants: int,
    impute_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """One checkpoint's fills as ``(rows, ai, z, se)``: each ALID resolved to its
    union row, those no row names and the filtered-out Analyses' dropped."""
    f_alid = d["f_alid"]
    if not len(f_alid):
        return None
    if f_alid.dtype.kind == "U":
        f_alid = f_alid.astype(ALID_DTYPE)

    idx = np.minimum(np.searchsorted(union_alids_s, f_alid), len(union_alids_s) - 1)
    matched = union_alids_s[idx] == f_alid
    if not matched.any():
        return None

    rows = union_rows_s[idx[matched]].astype(np.int32, copy=False)
    in_bounds = (rows >= 0) & (rows < n_variants)
    if not in_bounds.any():
        return None

    rows = rows[in_bounds]
    ai = d["f_ai"][matched].astype(np.int32, copy=False)[in_bounds]
    z = d["f_z"][matched].astype(np.float32, copy=False)[in_bounds]
    se = d["f_se"][matched].astype(np.float32, copy=False)[in_bounds]
    if impute_mask is not None:
        keep = impute_mask[ai]
        if not keep.any():
            return None
        rows, ai, z, se = rows[keep], ai[keep], z[keep], se[keep]
    return rows, ai, z, se


def _append_to_shards(
    fill_shard_dir: Path,
    fills: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    shard_ids: np.ndarray,
) -> None:
    """Append each fill's record to the shard file its key names."""
    rows, ai, z, se = fills
    for shard_index in np.unique(shard_ids):
        selected = shard_ids == shard_index
        records = np.empty(int(selected.sum()), dtype=FILL_RECORD_DTYPE)
        records["row"] = rows[selected]
        records["ai"] = ai[selected]
        records["z"] = z[selected]
        records["se"] = se[selected]
        with open(fill_shard_path(fill_shard_dir, int(shard_index)), "ab") as fh:
            records.tofile(fh)


def apply_fill_batch(
    zb: np.ndarray,
    sb: np.ndarray,
    imputed_band: np.ndarray,
    lr: np.ndarray,
    ai: np.ndarray,
    z: np.ndarray,
    se: np.ndarray,
) -> int:
    """Apply one batch of fill records; returns the net count of cells filled.

    A cell is a target while the source leaves it missing: unfilled, filled
    by an earlier record, or already in conflict. The batch's own repeats are
    resolved first; a kept fill then fills an open cell, or is compared with
    the fill the cell holds.
    """
    state = imputed_band[lr, ai]
    target = (state > 0) | ~np.isfinite(zb[lr, ai])
    lr, ai, z, se, state = lr[target], ai[target], z[target], se[target], state[target]
    if not len(lr):
        return 0
    found = collapse_duplicates(lr.astype(np.int64) * zb.shape[1] + ai, z, se)
    filled = -_mark_conflicted(imputed_band, lr[found.dropped], ai[found.dropped])
    k = found.keep
    lr, ai, z, se, state = lr[k], ai[k], z[k], se[k], state[k]
    open_ = state == 0
    zb[lr[open_], ai[open_]] = z[open_]
    sb[lr[open_], ai[open_]] = se[open_]
    imputed_band[lr[open_], ai[open_]] = 1
    filled += int(open_.sum())
    held = state == 1
    disagree = held & ((zb[lr, ai] != z) | (sb[lr, ai] != se))
    return filled - _mark_conflicted(imputed_band, lr[disagree], ai[disagree])


def _mark_conflicted(imputed_band: np.ndarray, lr: np.ndarray, ai: np.ndarray) -> int:
    """Mark these cells conflicted; returns how many had been counted filled."""
    if not len(lr):
        return 0
    n_analyses = imputed_band.shape[1]
    rows, cols = np.divmod(np.unique(lr.astype(np.int64) * n_analyses + ai), n_analyses)
    was_filled = int((imputed_band[rows, cols] == 1).sum())
    imputed_band[rows, cols] = CONFLICTED
    return was_filled


def clear_conflicted(
    zb: np.ndarray, sb: np.ndarray, imputed_band: np.ndarray, conflicted: np.ndarray
) -> None:
    """Leave each cell whose fills disagree missing: no z, no SE, not imputed (D5)."""
    zb[conflicted] = np.nan
    sb[conflicted] = np.nan
    imputed_band[conflicted] = 0


def log_dropped(what: str, rows: Iterable[tuple[int, str, float, float]]) -> int:
    """Log every row D5 drops, one line each (D8), as ``(analysis, variant, z,
    se)``; returns how many.

    D8 logs each row dropped for a reason other than a declared filter, and a
    completion writes no Admission record (ADR 0072), so its log is where those
    rows are named, all of them. A fill either layout drops and an observed row
    Ragged completion drops (`what`) break the same rule and get the same line.
    """
    n_rows = 0
    for analysis, variant, z, se in rows:
        log.warning(
            "Analysis %d: dropped %s %s (z=%r, se=%r): duplicate_conflicting",
            analysis, what, variant, z, se,
        )
        n_rows += 1
    return n_rows
