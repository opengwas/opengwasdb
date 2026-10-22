"""The Observed-View Check on a Hybrid Store (issue #295, stores ADR 0037).

A Hybrid release holds its observations in two components over one shared
variant axis: the Dense Component, whose axis is the panel, and the Ragged
Overflow, which holds every observation off it. `complete-hybrid` completes the
Dense Component and rebuilds the Overflow on the completed shared axis. When the
LD panel extends the Dense axis, an Overflow association at a variant the panel
added moves into the Dense Component as an observed cell (`imputed == 0`) and
leaves the Overflow: a **crossover** (issue #99). Neither component survives
completion unchanged, so the two cannot be checked apart. This compares the
Store's observed view across both, by ALID:

* **Axes, exhaustively.** The Dense Components' axes and the shared axes are
  mapped as the Dense check maps one (`_map_axis`), and every completed Dense
  row must be on the completed shared axis.
* **Every crossover.** `complete-hybrid` records none, so they are derived from
  the axes: a source Overflow association whose ALID is a Dense row of the
  completed Store. Each must be an observed, unflagged Dense cell holding what
  the Dense Component's own encoding gives for the source value -- the value
  encoded and decoded with that codec, which is exact and still differs from a
  moved or altered value. A crossover is coded against the Dense Component's
  baseline, not the Overflow's, so bit-identity with the source is not the rule.
* **The Dense Component, sampled.** The Dense check, unchanged: seeded
  chunk-rows and strata over the components' own geometry. Crossover cells are
  set aside for the check above; they are the only observed, unflagged cells an
  inserted row may hold.
* **The Overflow, sampled and counted.** Every association of a seeded sample
  of Analyses is matched in both directions with bit-identical Z, SE and EAF.
  Over every Analysis, the completed Overflow holds the source's count less its
  crossovers, none of its associations is at a Dense variant, and none is
  marked imputed.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from opengwasdb.encoding import (
    EAF_BASELINE,
    EafExceptionBuilder,
    SeExceptionBuilder,
    StoreCodec,
    ZOverflowBuilder,
    positions_pairs,
)
from opengwasdb.encoding.planes import SE_COEFFICIENTS
from opengwasdb.layouts.dense.complete import completion_band_rows
from opengwasdb.layouts.hybrid.layout import dense_component_path
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.model.enums import CompletionState, PrimaryStorageLayout
from opengwasdb.store.arrays import inner_chunk_of
from opengwasdb.store.open import OpenGWASDBStore
from opengwasdb.validation.observed_view import (
    _DIFF_PLANES,
    FAILING_COUNTERS,
    ObservedViewRefused,
    ObservedViewVerdict,
    _analyses_record,
    _AxisMap,
    _Cells,
    _header_record,
    _map_axis,
    _observed,
    _open_checked,
    _Pair,
    _partial_cells,
    _plane_difference,
    _Planes,
    _refuse_if_incoherent,
    _sample_and_compare,
    _shard_rows,
    _unsampled_record,
)
from opengwasdb.variants import VariantAxis

__all__ = ["CROSSOVER_FAILING", "OVERFLOW_FAILING", "check_hybrid"]

# The Overflow's counters over the sampled Analyses, and over every Analysis.
OVERFLOW_FAILING = (
    "z_mismatch",
    "se_mismatch",
    "eaf_mismatch",
    "observed_missing_in_child",
    "missing_became_observed",
    "z_se_missingness_mismatch",
)
_OVERFLOW_INFORMATIONAL = ("parent_associations", "crossovers", "child_associations")
_EVERY_ANALYSIS_FAILING = ("observed_in_both", "imputed_cells", "analyses_count_mismatch")
CROSSOVER_FAILING = (
    "z_mismatch",
    "se_mismatch",
    "eaf_mismatch",
    "observed_marked_imputed",
    "observed_missing_in_child",
)


def check_hybrid(
    source_path: Path, target_path: Path, *, seed: int, n_random: int
) -> ObservedViewVerdict:
    """Compare an Observed-Only Hybrid Store with the `complete-hybrid` output made from it.

    `n_random` is both the Dense Component's random chunk-rows and the Overflow's
    sampled Analyses; each is drawn with its own `default_rng(seed)`.
    """
    started = time.time()
    pair = _open_hybrid_pair(source_path, target_path)
    analyses = _analyses_record(pair.dense)
    analyses["identical"] = analyses["identical"] and pair.overflow_widths_agree(
        analyses["n_parent"]
    )
    axes = _HybridAxes.map(pair)
    chunk_rows = int(inner_chunk_of(pair.dense.parent.z.array)[0])
    band_rows = completion_band_rows(_shard_rows(pair.dense.child.z.array))
    geometry = (chunk_rows, band_rows)
    record = _header_record(pair.dense, seed, n_random, analyses, axes.dense, geometry)
    dense = {"axis": record.pop("axis"), "geometry": record.pop("geometry")}
    overflow = {"axis": axes.overflow_record(), **pair.association_counts()}
    axis_seconds = time.time() - started
    if axes.ok and analyses["identical"]:
        sides = _compare_both(pair, axes, geometry, seed, n_random)
    else:
        # Rows that do not line up would only be compared against the wrong rows.
        sides = _unchecked()
    dense.update(sides.dense)
    overflow.update(sides.overflow)
    record = {"layout": "hybrid", **record}
    record["components"] = {"dense": dense, "overflow": overflow}
    record["crossovers"] = sides.crossovers
    record["failures"] = _failures(record, axes)
    record["passed"] = not record["failures"]
    record["seconds"] = {"axis": round(axis_seconds, 1), "total": round(time.time() - started, 1)}
    return ObservedViewVerdict(record)


# --- opening the pair -------------------------------------------------------


@dataclass(frozen=True)
class _Overflow:
    """One Store's Ragged Overflow and the shared axis its variant indices point into."""

    reader: RaggedCSRReader
    offsets: np.ndarray
    imputed: Any  # the zarr array, or None: an Overflow is observed-only (ADR 0026)
    n_shared: int

    def segment(self, analysis: int) -> tuple[int, int]:
        return int(self.offsets[analysis]), int(self.offsets[analysis + 1])

    def variant_indices(self, analysis: int, role: str) -> np.ndarray:
        """One Analysis's shared-axis indices, refused if any falls off the axis."""
        indices = np.asarray(self.reader.variant_indices(analysis), dtype=np.int64)
        if indices.size and (indices.min() < 0 or indices.max() >= self.n_shared):
            raise ObservedViewRefused(
                f"the {role} store's Overflow holds a variant index outside its "
                f"{self.n_shared}-variant shared axis (Analysis {analysis})"
            )
        return indices


@dataclass(frozen=True)
class _HybridPair:
    """Both Hybrid Stores, opened and found checkable.

    `dense` is the roots paired with their Dense Components' planes, which is the
    shape the Dense check's helpers take.
    """

    dense: _Pair
    child_component: OpenGWASDBStore
    parent_overflow: _Overflow
    child_overflow: _Overflow

    def overflow_widths_agree(self, n_analyses: int) -> bool:
        return all(
            len(side.offsets) == n_analyses + 1
            for side in (self.parent_overflow, self.child_overflow)
        )

    def association_counts(self) -> dict[str, int]:
        return {
            "n_parent_associations": int(self.parent_overflow.offsets[-1]),
            "n_child_associations": int(self.child_overflow.offsets[-1]),
        }


def _open_hybrid_pair(source_path: Path, target_path: Path) -> _HybridPair:
    source = _open_checked(source_path, "source", CompletionState.OBSERVED_ONLY)
    target = _open_checked(target_path, "target", CompletionState.REFERENCE_COMPLETED)
    parent = _Planes(_open_component(source, "source", CompletionState.OBSERVED_ONLY))
    child_component = _open_component(target, "target", CompletionState.REFERENCE_COMPLETED)
    child = _Planes(child_component)
    _refuse_if_incoherent(dense_component_path(source_path), parent, "source Dense Component")
    _refuse_if_incoherent(dense_component_path(target_path), child, "target Dense Component")
    if child.imputed is None:
        raise ObservedViewRefused(
            f"the Dense Component of target store {target_path} has no `imputed` mask to check"
        )
    return _HybridPair(
        _Pair(source_path, target_path, source, target, parent, child),
        child_component,
        _open_overflow(source, "source"),
        _open_overflow(target, "target"),
    )


def _open_component(store: OpenGWASDBStore, role: str, wanted: CompletionState) -> OpenGWASDBStore:
    path = dense_component_path(store.path)
    component = _open_checked(path, f"{role} Dense Component", wanted)
    layout = component.manifest.primary_layout
    if layout is not PrimaryStorageLayout.DENSE:
        raise ObservedViewRefused(
            f"the {role} Dense Component {path} is not Dense (primary_layout={layout.value})"
        )
    return component


def _open_overflow(store: OpenGWASDBStore, role: str) -> _Overflow:
    try:
        reader = RaggedCSRReader(store.path, store.manifest.encoding)
        group = store.arrays(mode="r")["ragged"]
    except (OSError, ValueError, KeyError) as exc:
        raise ObservedViewRefused(
            f"cannot open the Overflow of {role} store {store.path}: {exc}"
        ) from exc
    with closing(VariantAxis(store.path)) as axis:
        n_shared = axis.n_variants
    offsets = np.asarray(group["offsets"][:], dtype=np.int64)
    imputed = group["imputed"] if "imputed" in group else None
    return _Overflow(reader, offsets, imputed, n_shared)


# --- the exhaustive axis maps -----------------------------------------------


@dataclass(frozen=True)
class _HybridAxes:
    """The three exact ALID maps the comparison stands on.

    `dense`: source Dense rows into completed Dense rows. `shared`: source shared
    rows into completed shared rows. `dense_in_shared`: completed Dense rows into
    the completed shared axis, from which `dense_row_of_shared` (each completed
    shared row's Dense row, or -1) and `inserted` (completed Dense rows no source
    Dense row maps to) follow.
    """

    dense: _AxisMap
    shared: _AxisMap
    dense_in_shared: _AxisMap
    dense_row_of_shared: np.ndarray
    inserted: np.ndarray

    @classmethod
    def map(cls, pair: _HybridPair) -> _HybridAxes:
        source, target = pair.dense.source_path, pair.dense.target_path
        n_parent, n_child = pair.dense.parent.n_rows, pair.dense.child.n_rows
        n_shared = pair.child_overflow.n_shared
        dense = _map_axis(dense_component_path(source), dense_component_path(target), n_parent,
                          n_child)
        shared = _map_axis(source, target, pair.parent_overflow.n_shared, n_shared)
        dense_in_shared = _map_axis(dense_component_path(target), target, n_child, n_shared)
        dense_row_of_shared = np.full(n_shared, -1, dtype=np.int64)
        on_shared = dense_in_shared.source_to_target >= 0
        dense_row_of_shared[dense_in_shared.source_to_target[on_shared]] = np.flatnonzero(on_shared)
        inserted = np.ones(n_child, dtype=bool)
        inserted[dense.source_to_target[dense.source_to_target >= 0]] = False
        return cls(dense, shared, dense_in_shared, dense_row_of_shared, inserted)

    @property
    def ok(self) -> bool:
        return self.dense.ok and self.shared.ok and self.dense_in_shared.ok

    def overflow_record(self) -> dict[str, Any]:
        record = self.shared.record(
            len(self.shared.source_to_target), len(self.dense_row_of_shared)
        )
        record["child_dense_rows_unshared"] = self.dense_in_shared.unmapped
        record["child_dense_order_preserved"] = self.dense_in_shared.order_preserved
        return record


# --- the comparison ---------------------------------------------------------


class _Sides(NamedTuple):
    dense: dict[str, Any]
    overflow: dict[str, Any]
    crossovers: dict[str, Any]


def _compare_both(
    pair: _HybridPair,
    axes: _HybridAxes,
    geometry: tuple[int, int],
    seed: int,
    n_random: int,
) -> _Sides:
    crossovers, every_analysis = _scan_overflows(pair, axes)
    crossover_record = _check_crossovers(pair, crossovers)
    on_inserted = axes.inserted[crossovers.rows]
    child = _CrossoversSetAside(
        pair.child_component, crossovers.rows[on_inserted], crossovers.cols[on_inserted]
    )
    dense = _sample_and_compare(axes.dense, pair.dense.parent, child, geometry, seed, n_random)
    dense["totals"]["crossover_cells"] = child.set_aside
    overflow = _sample_overflow(pair, axes, seed, n_random)
    overflow["all_analyses"] = every_analysis
    return _Sides(dense, overflow, crossover_record)


def _unchecked() -> _Sides:
    overflow = {
        "analyses_checked": 0,
        "analyses": [],
        "analyses_with_mismatch": [],
        "totals": dict.fromkeys((*_OVERFLOW_INFORMATIONAL, *OVERFLOW_FAILING), 0),
        "max_abs_diff": dict.fromkeys(_DIFF_PLANES, 0.0),
        "all_analyses": dict.fromkeys(_EVERY_ANALYSIS_FAILING, 0),
    }
    dense = _unsampled_record()
    dense["totals"]["crossover_cells"] = 0
    # Not derived, so not reported as none.
    unchecked = _crossover_tally()
    crossovers = {
        "n": None,
        "rows": None,
        **unchecked.record(),
        "reencoding_max_abs_diff": unchecked.max_abs_diff,
    }
    return _Sides(dense, overflow, crossovers)


def _failures(record: Mapping[str, Any], axes: _HybridAxes) -> list[str]:
    failures = _axis_failures(axes)
    if not record["analyses"]["identical"]:
        failures.append("analyses_mismatch")
    dense, overflow = record["components"]["dense"], record["components"]["overflow"]
    for prefix, counts, names in (
        ("dense", dense["totals"], FAILING_COUNTERS),
        ("overflow", overflow["totals"], OVERFLOW_FAILING),
        ("overflow", overflow["all_analyses"], _EVERY_ANALYSIS_FAILING),
        ("crossovers", record["crossovers"]["totals"], CROSSOVER_FAILING),
    ):
        failures.extend(f"{prefix}.{name}" for name in names if counts[name])
    return failures


def _axis_failures(axes: _HybridAxes) -> list[str]:
    failures = []
    for name, axis in (("dense", axes.dense), ("overflow", axes.shared)):
        if axis.unmapped:
            failures.append(f"{name}.parent_rows_unmapped")
        if not axis.order_preserved:
            failures.append(f"{name}.order_not_preserved")
    if not axes.dense_in_shared.ok:
        failures.append("overflow.child_dense_rows_not_on_shared_axis")
    return failures


# --- every Analysis: crossovers, counts, the partition ----------------------


class _Crossovers(NamedTuple):
    """Every crossover: its completed Dense cell, and the source Overflow's values."""

    rows: np.ndarray
    cols: np.ndarray
    source: _Cells


def _scan_overflows(pair: _HybridPair, axes: _HybridAxes) -> tuple[_Crossovers, dict[str, int]]:
    """Read both Overflows' variant indices once, Analysis by Analysis.

    Each source association is a crossover when its completed shared row is a
    completed Dense row. The completed Overflow must then hold the source's count
    less those, none at a Dense variant, and none marked imputed.
    """
    parent, child = pair.parent_overflow, pair.child_overflow
    counts = dict.fromkeys(_EVERY_ANALYSIS_FAILING, 0)
    found: list[tuple[np.ndarray, np.ndarray, _Cells]] = []
    for analysis in range(len(parent.offsets) - 1):
        dense_rows = axes.dense_row_of_shared[
            axes.shared.source_to_target[parent.variant_indices(analysis, "source")]
        ]
        crossing = np.flatnonzero(dense_rows >= 0)
        if crossing.size:
            found.append((dense_rows[crossing], np.full(crossing.size, analysis),
                          _values_at(parent.reader, parent.segment(analysis)[0] + crossing)))
        child_indices = child.variant_indices(analysis, "target")
        counts["observed_in_both"] += int((axes.dense_row_of_shared[child_indices] >= 0).sum())
        counts["analyses_count_mismatch"] += int(
            child_indices.size != len(dense_rows) - crossing.size
        )
        counts["imputed_cells"] += _imputed_in(child, analysis)
    return _gather_crossovers(found), counts


def _values_at(reader: RaggedCSRReader, positions: np.ndarray) -> _Cells:
    eaf = reader.eaf_at(positions)
    return _Cells(reader.z_at(positions), reader.se_at(positions, eaf=eaf), eaf)


def _imputed_in(overflow: _Overflow, analysis: int) -> int:
    if overflow.imputed is None:
        return 0
    lo, hi = overflow.segment(analysis)
    return int(np.count_nonzero(overflow.imputed[lo:hi]))


def _gather_crossovers(found: list[tuple[np.ndarray, np.ndarray, _Cells]]) -> _Crossovers:
    if not found:
        none = np.empty(0, dtype=np.int64)
        empty = np.empty(0, dtype=np.float32)
        return _Crossovers(none, none, _Cells(empty, empty, empty))
    rows = np.concatenate([rows for rows, _, _ in found]).astype(np.int64)
    cols = np.concatenate([cols for _, cols, _ in found]).astype(np.int64)
    order = np.lexsort((cols, rows))
    source = _Cells(
        *(np.concatenate([getattr(cells, p) for _, _, cells in found])[order] for p in _DIFF_PLANES)
    )
    return _Crossovers(rows[order], cols[order], source)


# --- every crossover against the completed Dense Component ------------------


class _Tally:
    """Counters, and the largest gaps, summed over pieces of one comparison."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        self.totals = dict.fromkeys(keys, 0)
        self.max_abs_diff = dict.fromkeys(_DIFF_PLANES, 0.0)

    def add_planes(self, parent: _Cells, child: _Cells) -> bool:
        """Count each plane's differing cells; whether any differed."""
        differed = False
        for plane in _DIFF_PLANES:
            n, gap = _plane_difference(getattr(parent, plane), getattr(child, plane))
            self.totals[f"{plane}_mismatch"] += n
            self.max_abs_diff[plane] = max(self.max_abs_diff[plane], gap)
            differed = differed or n > 0
        return differed

    def record(self) -> dict[str, Any]:
        return {"totals": self.totals, "max_abs_diff": self.max_abs_diff}


def _crossover_tally() -> _Tally:
    return _Tally(CROSSOVER_FAILING)


def _check_crossovers(pair: _HybridPair, crossovers: _Crossovers) -> dict[str, Any]:
    """Compare every crossover with its completed Dense cell, one completed chunk-row at a time.

    A chunk-row is read whole across its Analyses, as a Dense read costs whatever
    it touches. `reencoding_max_abs_diff` is how far the Dense codec moved the
    source values; it is informational, as moving them is what the codec does.
    """
    child = pair.dense.child
    codec = _DenseCodec(pair.child_component)
    tally, shift = _crossover_tally(), _crossover_tally()
    for lo, hi in _runs_by_chunk_row(crossovers.rows, int(inner_chunk_of(child.z.array)[0])):
        rows, cols = crossovers.rows[lo:hi], crossovers.cols[lo:hi]
        source = _Cells(*(getattr(crossovers.source, p)[lo:hi] for p in _DIFF_PLANES))
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        band = child.read(r0, r1)
        assert child.imputed is not None  # refused when the pair was opened
        imputed = np.asarray(child.imputed[r0:r1])[rows - r0, cols]
        found = _Cells(*(getattr(band, p)[rows - r0, cols] for p in _DIFF_PLANES))
        expected = codec.round_trip(rows, cols, source)
        tally.totals["observed_missing_in_child"] += int((~_observed(found)).sum())
        tally.totals["observed_marked_imputed"] += int((imputed != 0).sum())
        tally.add_planes(expected, found)
        shift.add_planes(source, expected)
    return {
        "n": len(crossovers.rows),
        "rows": len(np.unique(crossovers.rows)),
        **tally.record(),
        "reencoding_max_abs_diff": shift.max_abs_diff,
    }


def _runs_by_chunk_row(rows: np.ndarray, chunk_rows: int) -> Iterator[tuple[int, int]]:
    """`[lo, hi)` runs of the row-sorted `rows` that fall in one chunk-row."""
    if not rows.size:
        return
    chunk = rows // chunk_rows
    starts = np.flatnonzero(np.r_[True, chunk[1:] != chunk[:-1]])
    ends = np.r_[starts[1:], rows.size]
    yield from zip(starts.tolist(), ends.tolist(), strict=True)


class _DenseCodec:
    """What a completed Dense Component stores for a value: encoded, then decoded.

    The crossover fold writes each value through the Dense planes (`patch`):
    Z, then EAF against that row's `eaf_baseline`, then SE against the EAF the
    cell now decodes to. This does the same arithmetic without writing.
    """

    def __init__(self, component: OpenGWASDBStore) -> None:
        root = component.arrays(mode="r")
        self._encoding = component.manifest.encoding
        self._n_analyses = int(root["z"].shape[1])
        self._baseline = (
            np.asarray(root[EAF_BASELINE][:], dtype=np.float32) if EAF_BASELINE in root else None
        )
        self._coefficients = (
            np.asarray(root[SE_COEFFICIENTS][:], dtype=np.float32)
            if self._encoding.se.is_residual
            else np.empty((0, 2), dtype=np.float32)
        )

    def round_trip(self, rows: np.ndarray, cols: np.ndarray, values: _Cells) -> _Cells:
        positions = positions_pairs(rows, cols, self._n_analyses)
        encoder = StoreCodec(self._encoding)
        z_overflow, eaf_exceptions, se_exceptions = (
            ZOverflowBuilder(), EafExceptionBuilder(), SeExceptionBuilder()
        )
        baseline = None if self._baseline is None else self._baseline[rows]
        z_codes = encoder.encode_z(values.z, positions=positions, overflow=z_overflow)
        eaf_codes = encoder.encode_eaf(
            values.eaf, baseline=baseline, positions=positions, exceptions=eaf_exceptions
        )
        decoder = StoreCodec(
            self._encoding, z_overflow=z_overflow.table(), eaf_exceptions=eaf_exceptions.table()
        )
        # An observed cell: the panel's frequency never stands in for it.
        eaf = decoder.decode_eaf(
            eaf_codes, baseline=baseline, positions=positions,
            imputed=np.zeros(len(rows), dtype=np.uint8),
            reference=np.full(len(rows), np.nan, dtype=np.float32),
        )
        se_codes = encoder.encode_se(
            values.se, eaf=eaf, analysis_index=cols, coefficients=self._coefficients,
            positions=positions, exceptions=se_exceptions,
        )
        se = StoreCodec(self._encoding, se_exceptions=se_exceptions.table()).decode_se(
            se_codes, eaf=eaf, analysis_index=cols, coefficients=self._coefficients,
            positions=positions,
        )
        return _Cells(decoder.decode_z(z_codes, positions=positions), se, eaf)


class _CrossoversSetAside(_Planes):
    """The completed Dense planes, with the crossover cells on inserted rows read as missing.

    The Dense check counts an observed, unflagged cell on an inserted row as
    `inserted_observed_cells`. A crossover is exactly that, and is compared on
    its own (`_check_crossovers`), so the Dense comparison is handed the
    completed Store without it. `set_aside` counts the cells withheld.
    """

    def __init__(self, store: OpenGWASDBStore, rows: np.ndarray, cols: np.ndarray) -> None:
        super().__init__(store)
        self._rows, self._cols = rows, cols  # sorted by row
        self.set_aside = 0

    def read(self, r0: int, r1: int) -> _Cells:
        cells = super().read(r0, r1)
        lo, hi = np.searchsorted(self._rows, [r0, r1])
        local = (self._rows[lo:hi] - r0, self._cols[lo:hi])
        for plane in cells:
            plane[local] = np.nan
        self.set_aside += int(hi - lo)
        return cells


# --- the sampled Overflow ---------------------------------------------------


def _sample_overflow(
    pair: _HybridPair, axes: _HybridAxes, seed: int, n_random: int
) -> dict[str, Any]:
    """Every association of `n_random` seeded Analyses, matched in both directions."""
    n_analyses = len(pair.parent_overflow.offsets) - 1
    rng = np.random.default_rng(seed)
    chosen = sorted(
        int(a) for a in rng.choice(n_analyses, size=min(n_random, n_analyses), replace=False)
    )
    tally = _Tally((*_OVERFLOW_INFORMATIONAL, *OVERFLOW_FAILING))
    dirty = []
    for analysis in chosen:
        if _compare_analysis(pair, axes, analysis, tally):
            dirty.append(analysis)
    return {
        "analyses_checked": len(chosen),
        "analyses": chosen,
        "analyses_with_mismatch": dirty,
        **tally.record(),
    }


def _compare_analysis(pair: _HybridPair, axes: _HybridAxes, analysis: int, tally: _Tally) -> bool:
    """One Analysis's Overflow, source against completed; whether anything failed.

    Source crossovers are left to `_check_crossovers`, and completed associations
    at a Dense variant to `observed_in_both`, so neither is counted twice.
    """
    parent = _segment(pair.parent_overflow, analysis)
    child = _segment(pair.child_overflow, analysis)
    keys = axes.shared.source_to_target[parent.variant_index]
    crossing = axes.dense_row_of_shared[keys] >= 0
    kept = np.flatnonzero(~crossing)
    off_dense = np.flatnonzero(axes.dense_row_of_shared[child.variant_index] < 0)
    p, c = _match(keys[kept], child.variant_index[off_dense])
    p, c = kept[p], off_dense[c]
    counts = {
        "parent_associations": len(keys),
        "crossovers": int(crossing.sum()),
        "child_associations": len(child.variant_index),
        "observed_missing_in_child": len(kept) - len(p),
        "missing_became_observed": len(off_dense) - len(c),
        "z_se_missingness_mismatch": _partial_cells(parent.cells) + _partial_cells(child.cells),
    }
    for key, value in counts.items():
        tally.totals[key] += value
    differed = tally.add_planes(_cells_at(parent.cells, p), _cells_at(child.cells, c))
    return differed or any(counts[name] for name in OVERFLOW_FAILING if name in counts)


class _Segment(NamedTuple):
    variant_index: np.ndarray
    cells: _Cells


def _segment(overflow: _Overflow, analysis: int) -> _Segment:
    found = overflow.reader.get_analysis(analysis)
    return _Segment(
        np.asarray(found.variant_index, dtype=np.int64), _Cells(found.z, found.se, found.eaf)
    )


def _cells_at(cells: _Cells, index: np.ndarray) -> _Cells:
    return _Cells(cells.z[index], cells.se[index], cells.eaf[index])


def _match(parent: np.ndarray, child: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices pairing equal keys, the k-th occurrence in one with the k-th in the other.

    A writer accepts an Analysis holding one variant twice, so a key alone is not
    an identity; its occurrence rank makes it one.
    """
    _, p, c = np.intersect1d(
        _ranked(parent), _ranked(child), assume_unique=True, return_indices=True
    )
    return p, c


def _ranked(keys: np.ndarray) -> np.ndarray:
    """Each key with its occurrence rank among equal keys, packed into one int64."""
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    rank = np.arange(len(keys)) - np.searchsorted(ordered, ordered, side="left")
    packed = np.empty(len(keys), dtype=np.int64)
    packed[order] = (ordered.astype(np.int64) << 32) | rank
    return packed
