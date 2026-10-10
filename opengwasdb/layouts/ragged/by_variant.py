"""The variant-centric (`by_variant/`) index of a Ragged component (ADR 0060).

The Analysis-sorted CSR answers a per-Analysis question in O(1) and a
per-variant one only by reading the whole `variant_index`.  ADR 0060 decides a
`ragged/by_variant/` CSR duplicate ordered `(variant_index, analysis_index)`, so
a variant's rows are contiguous and a variant range's rows are contiguous.

This module is that index's one implementation:

* `build_variant_index` writes it, bounded in memory, by a **counting sort by
  variant** over the finished Analysis-sorted component;
* `ByVariantReader` reads a variant's (or a variant range's) row block and
  decodes it with the same `StoreCodec` a query would use;
* `add_variant_index` is the in-place augment path for an existing 0.2.0
  release, with the install and the manifest/consolidated-metadata refresh
  ordered so an interrupted run leaves either no index or a complete one.

Nothing here decodes or re-encodes a statistic: the index copies the stored
codes (`z`, `se`, `eaf`, `imputed`) and re-keys only what is keyed on the flat
CSR ordinal -- `analysis_index` (derived from the source `offsets`) and the
three exception/overflow tables.  The per-variant `eaf_baseline`, the
`eaf_reference` and the `se_coefficients` are shared, never duplicated
(ADR 0060), so a re-keyed exception table is the only thing a decode needs
beyond the copied codes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.encoding import (
    EafBaselineError,
    EafExceptionTable,
    SeExceptionTable,
    StoreCodec,
    StoreEncoding,
    ZOverflowTable,
    positions_flat,
)
from opengwasdb.encoding.codec import (
    EAF_BASELINE,
    EAF_EXCEPTION_INDEX,
    EAF_EXCEPTION_VALUE,
    EAF_REFERENCE,
    SE_EXCEPTION_INDEX,
    SE_EXCEPTION_VALUE,
    Z_OVERFLOW_INDEX,
    Z_OVERFLOW_VALUE,
)
from opengwasdb.encoding.planes import SE_COEFFICIENTS
from opengwasdb.layouts.ragged.zarr_csr import RAGGED_ZARR_PATH
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store.arrays import ArrayRole, array_length

log = logging.getLogger(__name__)

#: The index group, under the Ragged component's own group (ADR 0060).
BY_VARIANT_GROUP = "by_variant"

#: Destination rows one build band spans: one Ragged sequence **shard**.  A band
#: is written as whole shards, so a band that is not a whole number of them would
#: be a read-modify-write of the shard it ends inside (#249).  One shard is the
#: smallest whole-shard unit, and bounds the working set to about 0.9 GB of
#: codes at OGS-00011's widest dtype.
BAND_ROWS = store_arrays.RAGGED_SEQUENCE_SHARD_ELEMENTS

#: The cell-keyed exception tables, by the name a `by_variant/` leaf drops the
#: component prefix to.  Each is keyed on the flat CSR ordinal, so the index's
#: copy must be re-keyed to the by-variant ordinal.
_EXCEPTION_TABLES: tuple[tuple[str, str, str], ...] = (
    ("z", Z_OVERFLOW_INDEX, Z_OVERFLOW_VALUE),
    ("eaf", EAF_EXCEPTION_INDEX, EAF_EXCEPTION_VALUE),
    ("se", SE_EXCEPTION_INDEX, SE_EXCEPTION_VALUE),
)

#: Destination array name -> spill-record field name.  `imputed` is stored in
#: the record as `imp` because the record packs 1 byte a cell and the longer
#: name would invite a dtype mismatch with the source's `uint8` mask.
_PAYLOAD_FIELDS: dict[str, str] = {
    "z": "z",
    "se": "se",
    "eaf": "eaf",
    "imputed": "imp",
}


@dataclass(frozen=True)
class VariantIndexResult:
    """What one index build produced, for a caller's provenance and the artifact."""

    n_axis: int
    n_rows: int
    elapsed_seconds: float
    peak_rss_bytes: int
    disk_bytes: int

    def provenance(self) -> dict[str, Any]:
        """The `provenance.ragged.by_variant` block a manifest records."""
        return {
            "group": BY_VARIANT_GROUP,
            "n_axis": self.n_axis,
            "n_rows": self.n_rows,
        }


def _peak_rss_bytes() -> int:
    """This process's **peak** RSS, in bytes.

    `VmHWM` is the process's own high-water mark: unlike `ru_maxrss` it does not
    inherit the ~2 GiB a `pixi run` launcher hands a Python process, and unlike
    a `statm` sample after the build it is not the *current* RSS -- the build's
    arrays have been freed by the time the result is assembled, so a current
    sample under-reports a 9.9 GiB build as the process's baseline (review
    round 4, finding 1).
    """
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError(
        "VmHWM is not in /proc/self/status; the build's peak RSS cannot be reported"
    )


def _directory_bytes(path: Path) -> int:
    """Bytes on disk of a directory tree, as `du` would count it."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _read_offsets(source: Any) -> np.ndarray:
    """The component's Analysis offsets, as int64, with their length checked."""
    offsets = np.asarray(source["offsets"][:], dtype=np.int64)
    if offsets.size < 2 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError(
            "ragged/offsets must start at 0 and be non-decreasing; the index's "
            "counting sort reads the source's own offsets to assign analyses"
        )
    return offsets


def _cell_arrays(source: Any) -> list[str]:
    """The cell-keyed arrays to duplicate, in creation order, present ones only."""
    names = ["analysis_index"]
    names.extend(name for name in ("z", "se", "eaf", "imputed") if name in source)
    return names


def _has_imputed(source: Any) -> bool:
    return "imputed" in source


def _record_dtype(source: Any) -> np.dtype:
    """The fixed-width spill record one cell is partitioned as.

    `dest` is the destination ordinal **relative to its band**, which fits in
    int32 because a band is one 50,000,000-row shard.  The statistic fields keep
    the source plane's own dtype, so the spill is a copy of the stored codes.
    """
    fields: list[tuple[str, Any]] = [
        ("dest", "<i4"),
        ("ai", "<i4"),
        ("z", np.dtype(source["z"].dtype)),
        ("se", np.dtype(source["se"].dtype)),
    ]
    if "eaf" in source:
        fields.append(("eaf", np.dtype(source["eaf"].dtype)))
    if _has_imputed(source):
        fields.append(("imp", "u1"))
    return np.dtype(fields, align=False)


def _destination_ordinals(variant_index: np.ndarray, next_destination: np.ndarray) -> np.ndarray:
    """Each cell's destination ordinal, consuming one slot per cell.

    `variant_index` is non-decreasing (the CSR's format invariant), so equal
    variants form runs; a run of `L` cells of variant `v` takes the next `L`
    destination ordinals after `next_destination[v]`.  Within a variant the
    source order is ascending Analysis, which is exactly the `(variant,
    analysis)` order the index requires, so no comparison sort is needed -- the
    counting sort is this running cursor.
    """
    length = len(variant_index)
    if length == 0:
        return np.empty(0, dtype=np.int64)
    change = np.empty(length, dtype=bool)
    change[0] = True
    np.not_equal(variant_index[1:], variant_index[:-1], out=change[1:])
    run_starts = np.flatnonzero(change)
    run_variants = variant_index[run_starts].astype(np.int64)
    run_lengths = np.diff(np.append(run_starts, length))
    bases = next_destination[run_variants]
    within = np.arange(length, dtype=np.int64) - np.repeat(run_starts, run_lengths)
    next_destination[run_variants] = bases + run_lengths
    return np.repeat(bases, run_lengths) + within


def _count_by_variant(source: Any, n_axis: int, n_associations: int) -> np.ndarray:
    """Rows per variant over the whole component, windowed (issue #254 memory).

    Reads the 12.3 GB `variant_index` once in 50,000,000-cell windows and adds
    each window's histogram into an int64 vector over the variant axis.  The
    window is one shard, so no shard is read twice.
    """
    counts = np.zeros(n_axis, dtype=np.int64)
    for start in range(0, n_associations, BAND_ROWS):
        stop = min(start + BAND_ROWS, n_associations)
        window = np.asarray(source["variant_index"][start:stop], dtype=np.int64)
        counts += np.bincount(window, minlength=n_axis)
    return counts


def _band_file(spill_dir: Path, band: int) -> Path:
    return spill_dir / f"band-{band:05d}.bin"


def _partition_cells(
    source: Any,
    offsets: np.ndarray,
    by_offsets: np.ndarray,
    record: np.dtype,
    spill_dir: Path,
    n_bands: int,
) -> tuple[
    list[list[np.ndarray]],
    list[list[np.ndarray]],
    list[tuple[str, str, str, np.ndarray, np.ndarray]],
]:
    """One pass over the source, splitting every cell into its band.

    Returns the per-table re-keyed exception entries as two parallel lists of
    lists (index and value), in `_EXCEPTION_TABLES` order for the tables the
    component carries, and the source tables those entries were drawn from.
    """
    source_tables = _read_source_exception_tables(source)
    rekeyed_index: list[list[np.ndarray]] = [[] for _ in source_tables]
    rekeyed_value: list[list[np.ndarray]] = [[] for _ in source_tables]
    handles = [_band_file(spill_dir, band).open("wb") for band in range(n_bands)]
    try:
        for analysis in range(len(offsets) - 1):
            start, stop = int(offsets[analysis]), int(offsets[analysis + 1])
            if stop <= start:
                continue
            variant_index = np.asarray(source["variant_index"][start:stop], dtype=np.int32)
            destination = _destination_ordinals(variant_index, by_offsets)
            _partition_segment(source, analysis, start, stop, destination, record, handles)
            _rekey_segment(
                source_tables, rekeyed_index, rekeyed_value, start, stop, destination
            )
    finally:
        for handle in handles:
            handle.close()
    return rekeyed_index, rekeyed_value, source_tables


def _read_source_exception_tables(
    source: Any,
) -> list[tuple[str, str, str, np.ndarray, np.ndarray]]:
    """The component's exception tables as `(label, index name, value name, index, value)`.

    Held whole because the index must hold them whole to write the re-keyed
    copy: at OGS-00011 the largest is the 180,396,687-entry EAF exception table,
    about 2.1 GB of int64 index plus float32 value.
    """
    tables: list[tuple[str, str, str, np.ndarray, np.ndarray]] = []
    for label, index_name, value_name in _EXCEPTION_TABLES:
        if index_name not in source:
            continue
        index = np.asarray(source[index_name][:], dtype=np.int64)
        value = np.asarray(source[value_name][:], dtype=np.float32)
        tables.append((label, index_name, value_name, index, value))
    return tables


def _partition_segment(
    source: Any,
    analysis: int,
    start: int,
    stop: int,
    destination: np.ndarray,
    record: np.dtype,
    handles: list[Any],
) -> None:
    """Partition one Analysis's segment into the band spill files."""
    band = (destination // BAND_ROWS).astype(np.int64)
    payload = {
        "ai": np.full(stop - start, analysis, dtype=np.int32),
        "z": np.asarray(source["z"][start:stop]),
        "se": np.asarray(source["se"][start:stop]),
    }
    if "eaf" in source:
        payload["eaf"] = np.asarray(source["eaf"][start:stop])
    if _has_imputed(source):
        payload["imp"] = np.asarray(source["imputed"][start:stop])
    for b in np.unique(band):
        mask = band == b
        chunk = np.empty(int(mask.sum()), dtype=record)
        chunk["dest"] = (destination[mask] - int(b) * BAND_ROWS).astype(np.int32)
        for name, values in payload.items():
            chunk[name] = values[mask]
        chunk.tofile(handles[int(b)])


def _rekey_segment(
    source_tables: list[tuple[str, str, str, np.ndarray, np.ndarray]],
    rekeyed_index: list[list[np.ndarray]],
    rekeyed_value: list[list[np.ndarray]],
    start: int,
    stop: int,
    destination: np.ndarray,
) -> None:
    """Re-key the exception cells in `[start, stop)` to their by-variant ordinals.

    An exception is only a cell that happens to carry an exact value, so it is
    re-keyed from the same destination ordinals the partition computed.  Each
    Analysis's slice is appended to a list and concatenated once at the end, not
    concatenated per Analysis: the latter copies O(Analyses x E) (review round 1,
    finding 14).
    """
    for slot, (_label, _index_name, _value_name, index, value) in enumerate(source_tables):
        lo = int(np.searchsorted(index, start, side="left"))
        hi = int(np.searchsorted(index, stop, side="left"))
        if lo == hi:
            continue
        positions = index[lo:hi]
        rekeyed_index[slot].append(destination[positions - start])
        rekeyed_value[slot].append(value[lo:hi])


def _write_index_arrays(source: Any, index: Any, by_offsets: np.ndarray, n_rows: int) -> None:
    """Create every `by_variant/` array: offsets, the codes and the mask."""
    compressor = store_arrays.compressor()
    store_arrays.create_array(
        index,
        "offsets",
        ArrayRole.RAGGED_PER_VARIANT,
        data=by_offsets,
        dtype=np.int64,
        compressor=compressor,
        hint=store_arrays.BY_VARIANT_OFFSETS_CHUNK,
    )
    store_arrays.create_array(
        index,
        "analysis_index",
        ArrayRole.ASSOCIATION_SEQUENCE,
        shape=(n_rows,),
        dtype=np.int32,
        fill_value=0,
        compressor=compressor,
    )
    for name in ("z", "se", "eaf", "imputed"):
        if name in source:
            store_arrays.create_array(
                index,
                name,
                ArrayRole.ASSOCIATION_SEQUENCE,
                shape=(n_rows,),
                dtype=source[name].dtype,
                fill_value=source[name].fill_value,
                compressor=compressor,
            )


def _write_bands(
    source: Any,
    index: Any,
    record: np.dtype,
    spill_dir: Path,
    n_bands: int,
    n_rows: int,
) -> None:
    """Assemble each band's spill into the destination arrays, whole shards.

    A band is a whole number of destination shards, so every write covers whole
    shards.  `dest` is a permutation of the band's row range (every row has
    exactly one cell), which is asserted per band: a lost or duplicated row
    would otherwise leave a fill value reading as a plausible statistic.
    """
    for band in range(n_bands):
        start = band * BAND_ROWS
        stop = min(start + BAND_ROWS, n_rows)
        path = _band_file(spill_dir, band)
        cells = np.fromfile(path, dtype=record)
        if len(cells) != stop - start:
            raise ValueError(
                f"by_variant band {band} holds {len(cells)} cells but its destination "
                f"range [{start}, {stop}) has {stop - start} rows; the index would read "
                "a fill value where an association belongs"
            )
        order = np.argsort(cells["dest"].astype(np.int64), kind="stable")
        filled = cells["dest"][order].astype(np.int64)
        if not np.array_equal(filled, np.arange(stop - start, dtype=np.int64)):
            raise ValueError(
                f"by_variant band {band} does not place exactly one cell on every row; "
                "the destination ordinals are not a permutation of the band, so the "
                "index would hold a duplicated or missing association"
            )
        index["analysis_index"][start:stop] = cells["ai"][order]
        for array_name, field in _PAYLOAD_FIELDS.items():
            if array_name in index:
                index[array_name][start:stop] = cells[field][order]
        path.unlink()


def _write_rekeyed_tables(
    index: Any,
    source_tables: list[tuple[str, str, str, np.ndarray, np.ndarray]],
    rekeyed_index: list[list[np.ndarray]],
    rekeyed_value: list[list[np.ndarray]],
) -> None:
    """Write the re-keyed exception tables, uncompressed, beside the copied codes."""
    for (_label, index_name, value_name, _index, _value), index_parts, value_parts in zip(
        source_tables, rekeyed_index, rekeyed_value, strict=True
    ):
        positions = (
            np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int64)
        )
        values = (
            np.concatenate(value_parts) if value_parts else np.empty(0, dtype=np.float32)
        )
        order = np.argsort(positions, kind="stable")
        positions, values = positions[order], values[order]
        if len(positions) > 1 and np.any(positions[1:] == positions[:-1]):
            raise ValueError(
                f"the re-keyed exception table at {index_name} holds a cell twice; the "
                "source table is keyed by flat ordinal and cannot address one cell twice"
            )
        store_arrays.create_array(
            index,
            index_name,
            ArrayRole.RAGGED_EXCEPTION_TABLE,
            data=positions,
            dtype=np.int64,
            compressor=None,
            overwrite=True,
        )
        store_arrays.create_array(
            index,
            value_name,
            ArrayRole.RAGGED_EXCEPTION_TABLE,
            data=values,
            dtype=np.float32,
            compressor=None,
            overwrite=True,
        )


def build_variant_index(
    store_path: str | Path,
    *,
    n_axis: int,
    group_name: str = BY_VARIANT_GROUP,
    spill_dir: str | Path | None = None,
) -> VariantIndexResult:
    """Write the by-variant index of the Ragged component at `store_path`.

    Bounded in memory: neither the component's codes nor its `variant_index`
    is ever held whole.  The build is four passes -- count rows per variant,
    partition cells into band-sized spill files, assemble each band, write the
    re-keyed tables -- and holds only the `n_axis + 1` offsets, one band's cells
    and the (small) exception tables.

    `group_name` lets the augment path build into `by_variant.building` before an
    atomic rename; a builder passes the default and writes in place inside the
    staged release, which a failure discards whole.
    """
    ragged_path = Path(store_path) / RAGGED_ZARR_PATH
    started = time.monotonic()
    temporary_spill = spill_dir is None
    spill = Path(spill_dir) if spill_dir is not None else _default_spill_dir(store_path)
    spill.mkdir(parents=True, exist_ok=True)
    try:
        n_axis_result, n_rows = _build(
            ragged_path, n_axis=int(n_axis), group_name=group_name, spill=spill
        )
    finally:
        if temporary_spill:
            shutil.rmtree(spill, ignore_errors=True)
    return VariantIndexResult(
        n_axis=n_axis_result,
        n_rows=n_rows,
        elapsed_seconds=time.monotonic() - started,
        peak_rss_bytes=_peak_rss_bytes(),
        disk_bytes=_directory_bytes(ragged_path / group_name),
    )


def _default_spill_dir(store_path: str | Path) -> Path:
    """A spill directory beside the release, never inside its documented envelope."""
    store = Path(store_path)
    return store.parent / f".{store.name}.byvariant-spill"


def _build(
    ragged_path: Path, *, n_axis: int, group_name: str, spill: Path
) -> tuple[int, int]:
    """The four-pass build, opening the component and writing the group."""
    source = store_arrays.open_group_for_write(ragged_path, "a")
    offsets = _read_offsets(source)
    n_associations = int(offsets[-1])
    if array_length(source["variant_index"]) != n_associations:
        raise ValueError(
            f"ragged/variant_index has {array_length(source['variant_index'])} entries but "
            f"offsets imply {n_associations}; the index cannot be built over a "
            "component whose parallel arrays disagree"
        )
    if n_axis <= 0:
        raise ValueError(f"the variant axis length must be positive, got {n_axis}")
    _refuse_a_stale_group(source, group_name)
    log.info("by_variant: counting %d associations over %d variants", n_associations, n_axis)
    counts = _count_by_variant(source, n_axis, n_associations)
    by_offsets = np.empty(n_axis + 1, dtype=np.int64)
    by_offsets[0] = 0
    np.cumsum(counts, out=by_offsets[1:])
    del counts
    index = store_arrays.create_group(source, group_name, replace=True)
    _write_index_arrays(source, index, by_offsets, n_associations)
    record = _record_dtype(source)
    n_bands = max(1, -(-n_associations // BAND_ROWS))
    log.info("by_variant: partitioning into %d bands", n_bands)
    rekeyed_index, rekeyed_value, source_tables = _partition_cells(
        source, offsets, by_offsets, record, spill, n_bands
    )
    log.info("by_variant: assembling bands")
    _write_bands(source, index, record, spill, n_bands, n_associations)
    _write_rekeyed_tables(index, source_tables, rekeyed_index, rekeyed_value)
    return n_axis, n_associations


def _refuse_a_stale_group(source: Any, group_name: str) -> None:
    """Refuse to build into a group that already exists; the augment does otherwise."""
    if group_name in source:
        raise ValueError(
            f"ragged/{group_name} already exists; a variant index is written once, and "
            "an existing one must be removed deliberately (the augment command refuses "
            "to overwrite without --force)"
        )


# ── reading ──────────────────────────────────────────────────────────────────


def has_variant_index(store_path: str | Path, *, group_name: str = BY_VARIANT_GROUP) -> bool:
    """Whether the component at `store_path` carries a by-variant index."""
    path = Path(store_path) / RAGGED_ZARR_PATH / group_name
    return (path / "zarr.json").is_file() or (path / ".zgroup").is_file()


def recorded_provenance(store_path: str | Path) -> dict[str, Any] | None:
    """The `provenance.ragged.by_variant` block a release should record, or `None`.

    Read from the group's own `offsets` rather than threaded through every
    builder, so a manifest written after the index cannot disagree with it and
    a builder that skipped the index records no block.  Two entries are read:
    the axis length and the row count, both cheap.
    """
    if not has_variant_index(store_path):
        return None
    root = store_arrays.open_group(Path(store_path) / RAGGED_ZARR_PATH)
    offsets = root[BY_VARIANT_GROUP]["offsets"]
    return {
        "group": BY_VARIANT_GROUP,
        "n_axis": array_length(offsets) - 1,
        "n_rows": int(offsets[-1]),
    }


def with_variant_index(store_path: str | Path, ragged: dict[str, Any]) -> dict[str, Any]:
    """`ragged` provenance with the by-variant block added when the index exists."""
    block = recorded_provenance(store_path)
    return ragged if block is None else {**ragged, "by_variant": block}


class ByVariantReader:
    """Read a Ragged component's by-variant index (ADR 0060).

    A block is contiguous in the index's own order, so `rows_for_variant` and
    `rows_for_variant_range` turn a variant question into a row range, and
    `decode` reads and decodes exactly that range.  The per-row variant is
    recovered from the offsets rather than stored (the duplicate deliberately
    does not carry `variant_index`, ADR 0060), and the exception tables, the
    `eaf_baseline`/`eaf_reference` and the `se_coefficients` come from the
    index and the shared component respectively.
    """

    def __init__(
        self,
        store_path: str | Path,
        encoding: StoreEncoding | None = None,
        *,
        n_axis: int | None = None,
    ):
        self._ragged = store_arrays.open_group(Path(store_path) / RAGGED_ZARR_PATH)
        if BY_VARIANT_GROUP not in self._ragged:
            raise ValueError(
                f"{store_path}: {RAGGED_ZARR_PATH}/{BY_VARIANT_GROUP} is absent; a variant "
                "index must be built before it is read (ADR 0060)"
            )
        self._index = self._ragged[BY_VARIANT_GROUP]
        if encoding is None:
            encoding = StoreManifest.load(Path(store_path)).encoding
        self._encoding = encoding
        self._require_arrays()
        self._require_members()
        self._offsets = self._index["offsets"]
        self._require_span(store_path, n_axis)
        self._analysis_index = self._index["analysis_index"]
        self._z = self._index["z"]
        self._se = self._index["se"]
        self._eaf = self._index["eaf"] if "eaf" in self._index else None
        self._imputed = self._index["imputed"] if "imputed" in self._index else None
        self._baseline = self._ragged[EAF_BASELINE] if EAF_BASELINE in self._ragged else None
        self._reference = self._ragged[EAF_REFERENCE] if EAF_REFERENCE in self._ragged else None
        self._coefficients = (
            self._ragged[SE_COEFFICIENTS] if SE_COEFFICIENTS in self._ragged else None
        )
        self._codec_cache: StoreCodec | None = None

    def _require_arrays(self) -> None:
        """A missing required array is named, not a bare `KeyError` from a later read."""
        missing = [
            name
            for name in ("offsets", "analysis_index", "z", "se")
            if name not in self._index
        ]
        if missing:
            raise ValueError(
                f"{self._index.name}: the variant index is missing {missing}; a "
                "half-written group is invalid (ADR 0060)"
            )

    def _require_members(self) -> None:
        """The index must carry exactly the component's optional members.

        A missing `imputed` on a release that declares reference EAF is refused
        here for the same reason the scan plane refuses it: without the mask the
        panel substitution cannot be applied, and substituting zeros would
        silently read every imputed cell as observed.
        """
        if self._encoding.eaf.reference and "imputed" not in self._index:
            raise EafBaselineError(
                "this release declares reference EAF for its imputed cells but its "
                "variant index carries no imputed mask; without it an indexed read "
                "would read every imputed cell as observed (spec §9, §15)"
            )
        for name in ("eaf", "imputed"):
            if (name in self._index) != (name in self._ragged):
                raise ValueError(
                    f"{self._index.name}/{name} is present on one side only; the "
                    "component and its variant index must carry the same members"
                )

    def _require_span(self, store_path: str | Path, n_axis: int | None) -> None:
        """The index must span the component it duplicates, and the declared axis.

        `offsets[-1]` is the row count and must equal the Analysis-sorted
        component's own; a mismatch is a stale index answering with another
        store's rows.  When the caller knows the variant axis length, the
        offsets array must be `n_axis + 1` long.  Only scalars are read -- the
        length, the first and the last offset -- so opening the reader does not
        materialise a 164 M-entry (1.28 GB) array (review round 2, finding 2).
        """
        entries = array_length(self._offsets)
        component_rows = int(self._ragged["offsets"][-1])
        if entries == 0 or int(self._offsets[-1]) != component_rows:
            last = int(self._offsets[-1]) if entries else None
            raise ValueError(
                f"{store_path}: {BY_VARIANT_GROUP}/offsets ends at {last} but the "
                f"component holds {component_rows} associations; the index is stale"
            )
        if int(self._offsets[0]) != 0:
            raise ValueError(
                f"{store_path}: {BY_VARIANT_GROUP}/offsets starts at "
                f"{int(self._offsets[0])}, not 0"
            )
        if n_axis is not None and entries != int(n_axis) + 1:
            raise ValueError(
                f"{store_path}: {BY_VARIANT_GROUP}/offsets has {entries} entries "
                f"but the variant axis is {n_axis} (expected {int(n_axis) + 1})"
            )

    @property
    def _codec(self) -> StoreCodec:
        """The codec, built on first decode -- and not before.

        The index's re-keyed exception tables are read whole by `StoreCodec`, and
        at OGS-00011 the EAF exception table is 180,396,687 entries (~2.1 GiB).
        Reading them at open made every query that merely *opened* the store pay
        that 2.1 GiB, including the analysis-side shapes the index does not
        serve; a first decode is the first time the tables are needed.
        """
        if self._codec_cache is None:
            self._codec_cache = StoreCodec(
                self._encoding,
                z_overflow=ZOverflowTable.open(self._index),
                eaf_exceptions=EafExceptionTable.open(self._index),
                se_exceptions=SeExceptionTable.open(self._index),
            )
        return self._codec_cache

    def warm(self) -> None:
        """Read the codec's exception tables now, so a caller can time past them.

        The tables are a one-off per store open -- ~2.1 GiB and ~2.4 s at
        OGS-00011 -- and a benchmark that wants the per-query cost excludes them.
        """
        _ = self._codec

    @property
    def n_axis(self) -> int:
        return array_length(self._offsets) - 1

    @property
    def n_rows(self) -> int:
        return array_length(self._analysis_index)

    def rows_for_variant(self, variant_index: int) -> tuple[int, int]:
        """The half-open row range one variant's associations occupy.

        A variant the index does not cover (a Hybrid's on-panel variants) has an
        empty range, so the caller returns no rows rather than falling back to a
        scan -- the index *is* the answer for every variant it indexes.
        """
        variant = int(variant_index)
        if variant < 0 or variant >= self.n_axis:
            raise ValueError(
                f"variant index {variant} is outside the index's axis [0, {self.n_axis})"
            )
        pair = self._offsets[variant : variant + 2]
        return int(pair[0]), int(pair[1])

    def rows_for_variant_range(self, low: int, high: int) -> tuple[int, int]:
        """The half-open row range a contiguous variant range covers.

        `low` is the first and `high` the last variant in the range (inclusive);
        every row between their offsets belongs to a variant in `[low, high]`,
        because the index is variant-ordered.
        """
        first, _ = self.rows_for_variant(low)
        _, last = self.rows_for_variant(high)
        return first, last

    def decode(
        self, first_variant: int, last_variant: int, start: int, end: int
    ) -> dict[str, np.ndarray]:
        """Decode the index's rows `[start, end)`, whose variants span `[first, last]`.

        Returns the six result arrays' raw parts: `variant_index`,
        `analysis_index`, decoded `z`/`se`/`eaf` and the `imputed` mask (all
        zeros on an observed-only component).
        """
        start, end = int(start), int(end)
        empty = {
            "variant_index": np.empty(0, dtype="int32"),
            "analysis_index": np.empty(0, dtype="int32"),
            "z": np.empty(0, dtype="float32"),
            "se": np.empty(0, dtype="float32"),
            "eaf": np.empty(0, dtype="float32"),
            "imputed": np.empty(0, dtype="uint8"),
        }
        if end <= start:
            return empty
        variants = self._row_variants(first_variant, last_variant, start, end)
        analysis_index = np.asarray(self._analysis_index[start:end], dtype=np.int32)
        imputed = self._imputed_slice(start, end)
        eaf = self._decode_eaf(start, end, variants, imputed)
        se = self._decode_se(start, end, analysis_index, eaf)
        z = self._codec.decode_z(
            np.asarray(self._z[start:end]), positions=positions_flat(start)
        )
        return {
            "variant_index": variants.astype("int32"),
            "analysis_index": analysis_index,
            "z": z,
            "se": se,
            "eaf": eaf,
            "imputed": imputed,
        }

    def _row_variants(self, first: int, last: int, start: int, end: int) -> np.ndarray:
        """The variant of every row in `[start, end)`, from the offsets.

        The duplicate stores no per-row `variant_index` (ADR 0060), so the
        variant is recovered from the offsets: the local offsets of `[first,
        last]` are enough, which is why a query never holds the whole array.
        """
        local = np.asarray(self._offsets[int(first) : int(last) + 2], dtype=np.int64)
        rows = np.arange(start, end, dtype=np.int64)
        return (np.searchsorted(local, rows, side="right") - 1 + int(first)).astype(np.int64)

    def _imputed_slice(self, start: int, end: int) -> np.ndarray:
        if self._imputed is None:
            return np.zeros(end - start, dtype=np.uint8)
        return np.asarray(self._imputed[start:end], dtype=np.uint8)

    def _decode_eaf(
        self, start: int, end: int, variants: np.ndarray, imputed: np.ndarray
    ) -> np.ndarray:
        """Decoded frequencies, with the panel's value on imputed cells (ADR 0037 §4)."""
        baseline = self._gather(self._baseline, variants)
        reference = self._gather(self._reference, variants)
        want_imputed = self._encoding.eaf.reference
        imputed_arg = imputed if want_imputed else None
        if self._eaf is None:
            if not self._encoding.eaf.reference:
                return np.full(end - start, np.nan, dtype=np.float32)
            blank = np.full(end - start, np.nan, dtype=np.float32)
            return self._codec.decode_eaf(blank, imputed=imputed_arg, reference=reference)
        return self._codec.decode_eaf(
            np.asarray(self._eaf[start:end]),
            baseline=baseline,
            positions=positions_flat(start),
            imputed=imputed_arg,
            reference=reference,
        )

    @staticmethod
    def _gather(array: Any, variants: np.ndarray) -> np.ndarray | None:
        if array is None:
            return None
        return np.asarray(array.oindex[variants], dtype=np.float32)

    def _decode_se(
        self, start: int, end: int, analysis_index: np.ndarray, eaf: np.ndarray
    ) -> np.ndarray:
        raw = np.asarray(self._se[start:end])
        if not self._encoding.se.is_residual:
            return self._codec.decode_se(
                raw,
                eaf=eaf,
                analysis_index=analysis_index.astype(np.int64),
                coefficients=np.empty((0, 2), dtype=np.float32),
                positions=positions_flat(start),
            )
        if self._coefficients is None:
            raise ValueError(
                "the release declares int8_residual se but the component carries no "
                "se_coefficients; the index cannot decode it"
            )
        return self._codec.decode_se(
            raw,
            eaf=eaf,
            analysis_index=analysis_index.astype(np.int64),
            coefficients=np.asarray(self._coefficients[:], dtype=np.float32),
            positions=positions_flat(start),
        )


# ── augmenting an existing release ───────────────────────────────────────────


class VariantIndexError(RuntimeError):
    """The augment path cannot install an index into this release."""


def component_n_axis(store_path: str | Path) -> int:
    """The variant axis length the index at `store_path` is keyed on.

    The store's own union table for a standalone Ragged or a Hybrid release --
    a Hybrid Overflow indexes the **shared** axis, not an off-axis rank
    (ADR 0060).
    """
    from opengwasdb.variants import VariantAxis

    axis = VariantAxis(store_path)
    try:
        return int(axis.n_variants)
    finally:
        axis.close()


def add_variant_index(
    store_path: str | Path,
    *,
    force: bool = False,
    spill_dir: str | Path | None = None,
) -> VariantIndexResult:
    """Add the by-variant index to an existing 0.2.0 Ragged or Hybrid release.

    The release is written in place.  The group is built under
    `by_variant.building`, renamed into place, and only then are the manifest and
    any consolidated-metadata record refreshed.  The install is **ordered, not
    atomic**: each rename is atomic but the sequence is not, so a process killed
    between them leaves a state `_recover_interrupted_install` settles on the
    next run -- a leftover build is dropped, a half-finished swap is completed
    or undone.  A failed run restores the previous group, the previous manifest
    and any consolidated record; a `--force` rebuild builds the new index before
    the old one is removed, so a failed rebuild leaves the old index in place.
    """
    store, ragged, target = _augment_target(store_path, force)
    n_axis = component_n_axis(store.path)
    manifest = Path(store_path) / "manifest.json"
    manifest_backup = manifest.with_name("manifest.json.variant-index.bak")
    shutil.copy2(manifest, manifest_backup)
    suspended = _suspend_consolidated_metadata(store.data_path)
    installed = False
    try:
        result = build_variant_index(
            store.path,
            n_axis=n_axis,
            group_name=f"{BY_VARIANT_GROUP}.building",
            spill_dir=spill_dir,
        )
        _install_group(ragged)
        installed = True
        _record_index_in_manifest(store_path, result)
    except BaseException:
        _roll_back_build(ragged, installed=installed)
        if manifest_backup.exists():
            os.replace(manifest_backup, manifest)
        _restore_consolidated_metadata(suspended)
        raise
    manifest_backup.unlink(missing_ok=True)
    shutil.rmtree(ragged / f"{BY_VARIANT_GROUP}{_OLD_SUFFIX}", ignore_errors=True)
    _refresh_consolidated_metadata(store.data_path, suspended)
    return result


#: The suffix an index being replaced is renamed to while the new one is swapped in.
_OLD_SUFFIX = ".variant-index.old"


def _recover_interrupted_install(ragged: Path) -> None:
    """Settle the states a process killed mid-augment can leave.

    A leftover `.building` group is incomplete by definition and dropped.  A
    leftover `.old` means a swap was in progress: if the target is present the
    swap finished and the backup is stale; otherwise the process died between
    the two renames and the backup is the whole previous index.
    """
    building = ragged / f"{BY_VARIANT_GROUP}.building"
    old = ragged / f"{BY_VARIANT_GROUP}{_OLD_SUFFIX}"
    target = ragged / BY_VARIANT_GROUP
    if building.exists():
        shutil.rmtree(building, ignore_errors=True)
    if not old.exists():
        return
    if target.exists():
        shutil.rmtree(old, ignore_errors=True)
    else:
        os.replace(old, target)


def _augment_target(store_path: str | Path, force: bool) -> tuple[Any, Path, Path]:
    """Open the release and settle the preconditions for an in-place augment."""
    from opengwasdb.store import open_store
    from opengwasdb.store.open import CURRENT_FORMAT_VERSION

    store = open_store(store_path)
    if store.manifest.format_version != CURRENT_FORMAT_VERSION:
        raise VariantIndexError(
            f"{store_path}: format_version is {store.manifest.format_version!r}; the index "
            f"is added to a {CURRENT_FORMAT_VERSION} release. Convert a 0.1.0 release "
            "first with scripts/convert_store_to_0_2_0.py, as completion does."
        )
    ragged = store.data_path / "ragged"
    if not ragged.exists():
        raise VariantIndexError(f"{store_path}: the release has no Ragged component")
    _recover_interrupted_install(ragged)
    target = ragged / BY_VARIANT_GROUP
    if target.exists() and not force:
        raise VariantIndexError(
            f"{store_path}: ragged/{BY_VARIANT_GROUP} already exists; pass --force to "
            "rebuild it"
        )
    return store, ragged, target


def _install_group(ragged: Path) -> None:
    """Rename the built group into place, keeping the replaced one as a backup.

    The renamed-aside previous index is left for the caller to delete only after
    the manifest rewrite succeeds, so a failure can restore it.
    """
    building = ragged / f"{BY_VARIANT_GROUP}.building"
    target = ragged / BY_VARIANT_GROUP
    old = ragged / f"{BY_VARIANT_GROUP}{_OLD_SUFFIX}"
    if target.exists():
        os.replace(target, old)
    os.replace(building, target)


def _roll_back_build(ragged: Path, *, installed: bool) -> None:
    """Undo an interrupted augment: drop the new build, restore the replaced index."""
    building = ragged / f"{BY_VARIANT_GROUP}.building"
    target = ragged / BY_VARIANT_GROUP
    old = ragged / f"{BY_VARIANT_GROUP}{_OLD_SUFFIX}"
    shutil.rmtree(building, ignore_errors=True)
    if old.exists():
        shutil.rmtree(target, ignore_errors=True)
        os.replace(old, target)
    elif installed:
        # A fresh index was installed but the restored manifest does not claim it.
        shutil.rmtree(target, ignore_errors=True)


def _record_index_in_manifest(store_path: str | Path, result: VariantIndexResult) -> None:
    """Rewrite `manifest.json`'s provenance atomically (temp file, then rename)."""
    path = Path(store_path) / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    provenance = dict(data.get("provenance", {}))
    ragged = dict(provenance.get("ragged", {}))
    ragged["by_variant"] = result.provenance()
    provenance["ragged"] = ragged
    data["provenance"] = provenance
    temporary = path.with_name(f"{path.name}.variant-index.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _suspend_consolidated_metadata(data_path: Path) -> list[tuple[str, str]]:
    """Take every consolidated record out of the way so the write seam allows the build.

    The seam refuses *any* write beneath a record (it cannot keep one up to
    date).  The augment path is the one writer that promises to refresh it, so
    it removes the record first and puts a regenerated one back once the group
    and manifest are complete.  A Zarr v2 record is its own ``.zmetadata`` file
    and is renamed aside; a Zarr v3 record lives inside the group's own
    ``zarr.json``, so that file is copied and rewritten **without** the
    ``consolidated_metadata`` key rather than removed (removing it would delete
    the group's metadata too).
    """
    suspended: list[tuple[str, str]] = []
    # Backups live in the release directory, not inside `data.zarr`: a stray
    # file under the store's root pollutes zarr's own tree walk.
    backup_dir = data_path.parent / ".variant-index-backup"
    for record in store_arrays.consolidated_metadata_records(data_path):
        if record.name == "zarr.json":
            backup_dir.mkdir(exist_ok=True)
            backup = backup_dir / f"{len(suspended)}-{record.name}"
            shutil.copy2(record, backup)
            data = json.loads(record.read_text(encoding="utf-8"))
            data.pop("consolidated_metadata", None)
            temporary = record.with_name(f"{record.name}.variant-index.tmp")
            temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, record)
        else:
            backup = record.with_name(f"{record.name}.variant-index.bak")
            os.replace(record, backup)
        suspended.append((str(record), str(backup)))
    return suspended


def _restore_consolidated_metadata(suspended: list[tuple[str, str]]) -> None:
    for record, backup in suspended:
        os.replace(backup, record)


def _refresh_consolidated_metadata(data_path: Path, suspended: list[tuple[str, str]]) -> None:
    """Re-consolidate every suspended group, then drop the backups.

    Re-consolidating regenerates the record from the live metadata, so the
    refreshed record describes the new `by_variant` group the install added --
    atomically with the manifest rewrite that ran just before it.
    """
    if not suspended:
        return
    for record, backup in suspended:
        group_dir = Path(record).parent
        zarr_format = 3 if Path(record).name == "zarr.json" else 2
        store_arrays.consolidate_group_metadata(group_dir, zarr_format=zarr_format)
        Path(backup).unlink(missing_ok=True)
    backup_dir = Path(suspended[0][1]).parent
    if backup_dir.name == ".variant-index-backup" and not any(backup_dir.iterdir()):
        backup_dir.rmdir()
