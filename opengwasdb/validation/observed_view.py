"""The Observed-View Check: does a Reference-Completed Store reproduce its source? (issue #289)

Reference completion replaces an Observed-Only Store with the completed Store
built from it, and the Observed-Only one is then deleted (stores ADR 0037).
That swap is only safe if every observed association survived completion
unchanged, and nothing about a completed Store says so on its face: a shifted
row map, a re-encoded cell or an imputed value written over an observation all
leave a Store that validates and answers queries. This re-derives the claim
from the two Stores' arrays.

Two parts, as the ADR specifies:

* **Exhaustive axis check.** Every source variant is found in the completed
  axis by its ALID, in the same order, and both Stores list the same Analyses in
  the same order. Without that the cell comparison would line up the wrong rows.
* **Sampled cell check.** Whole source chunk-rows across all Analyses, because a
  Dense read costs a whole chunk however few cells it uses. The sample is a
  seeded uniform draw plus deliberate strata (`STRATA`). In each chunk-row, each
  source-observed cell has identical decoded Z, SE and EAF and is not marked
  imputed; no source-missing cell is observed; no inserted panel row carries an
  observed cell.

The verdict records the seed, so a check is reproducible from its record, and
the counts, so a pass is a measured zero and not an absence of output. A
refusal (`ObservedViewRefused`) is a different outcome from a failure: nothing
was checked, so there is no verdict to keep.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, assert_never

import numpy as np

from opengwasdb.encoding import DenseEafPlane, DenseSePlane, DenseZPlane
from opengwasdb.layouts.dense.complete import completion_band_rows
from opengwasdb.model.analyses import read_analysis_records
from opengwasdb.model.enums import CompletionState, PrimaryStorageLayout
from opengwasdb.store.arrays import inner_chunk_of
from opengwasdb.store.open import OpenGWASDBStore, open_store
from opengwasdb.variants import (
    VariantAxis,
    is_indexable_alid,
    variant_alid_bytes_path,
    variant_alid_rows_path,
)

__all__ = [
    "FAILING_COUNTERS",
    "STRATA",
    "ObservedViewRefused",
    "ObservedViewVerdict",
    "check_observed_view",
]

STRATA = ("random", "chrom_edge", "final_partial", "writer_band_boundary")

# These fail the check when nonzero. `missing_now_imputed` and `inserted_rows`
# are what completion is *for*, so they are reported but never failures.
FAILING_COUNTERS = (
    "z_mismatch",
    "se_mismatch",
    "eaf_mismatch",
    "observed_marked_imputed",
    "observed_missing_in_child",
    "missing_became_observed",
    "inserted_observed_cells",
    "imputed_mask_invalid",
    "z_se_missingness_mismatch",
)
_INFORMATIONAL_COUNTERS = (
    "missing_now_imputed",
    "inserted_rows",
    "parent_rows_without_finite",
)
_TOTAL_KEYS = ("cells_compared", "parent_observed", *FAILING_COUNTERS, *_INFORMATIONAL_COUNTERS)
_DIFF_PLANES = ("z", "se", "eaf")
_CHROMOSOME_WIDTH = 32  # the ALID search index's byte width, which bounds a prefix of it


class ObservedViewRefused(ValueError):
    """The pair cannot be checked: nothing was compared, so no verdict exists."""


@dataclass(frozen=True)
class ObservedViewVerdict:
    """What the check found. `record` is the JSON verdict, verbatim."""

    record: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.record["passed"])

    @property
    def failures(self) -> list[str]:
        return list(self.record["failures"])


class _Cells(NamedTuple):
    """Decoded statistics of a row band; NaN is missing."""

    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray


class _Planes:
    """One Store's Dense planes, opened once for reading."""

    def __init__(self, store: OpenGWASDBStore) -> None:
        root = store.arrays(mode="r")
        encoding = store.manifest.encoding
        self.z = DenseZPlane.open(root, encoding)
        self.se = DenseSePlane.open(root, encoding)
        self.eaf = DenseEafPlane.open(root, encoding)
        self.imputed = root["imputed"] if "imputed" in root else None

    def read(self, r0: int, r1: int) -> _Cells:
        # SE is residual-coded against EAF, so it is decoded with the very EAF
        # block read for these rows rather than letting it read its own.
        eaf = self.eaf.read_band(r0, r1).values
        return _Cells(self.z.band(r0, r1), self.se.band(r0, r1, eaf=eaf), eaf)

    @property
    def n_rows(self) -> int:
        return int(self.z.array.shape[0])

    @property
    def n_analyses(self) -> int:
        return int(self.z.array.shape[1])


@dataclass(frozen=True)
class _AxisMap:
    """Where each source row sits in the completed axis."""

    source_to_target: np.ndarray  # int64, -1 where the ALID was not found
    chromosome: np.ndarray  # one |S32 name per source row
    unmapped: int
    order_preserved: bool
    source_unindexed: int
    target_unindexed: int

    @property
    def ok(self) -> bool:
        return self.unmapped == 0 and self.order_preserved

    def record(self, n_source: int, n_target: int) -> dict[str, Any]:
        return {
            "parent_rows_unmapped": self.unmapped,
            "order_preserved": self.order_preserved,
            "n_parent": n_source,
            "n_child": n_target,
            "parent_rows_unindexed": self.source_unindexed,
            "child_rows_unindexed": self.target_unindexed,
        }


def check_observed_view(
    source_path: str | Path,
    target_path: str | Path,
    *,
    seed: int,
    n_random: int = 1000,
    ld_panel: str | Path | None = None,
    ancestry: str | None = None,
) -> ObservedViewVerdict:
    """Compare an Observed-Only Store with the Reference-Completed one made from it.

    Both Stores must have one layout, which picks the check: Dense here, Hybrid
    in `observed_view_hybrid` (issue #295). Raises `ObservedViewRefused` when
    the pair cannot be checked (a mixed-layout pair, a layout the check does not
    support, a source that is not Observed-Only, a target that is not
    Reference-Completed, or a Store whose arrays do not agree with its own
    axis). A mismatch is not an error: it is a verdict with `passed` false.

    `seed` is required so that every verdict can be reproduced from its record.
    `n_random` chunk-rows are drawn uniformly; a grid with fewer is sampled in
    full (for a Ragged pair, Analyses: `observed_view_ragged`, issue #296, which
    alone takes `ld_panel` and `ancestry`).
    """
    if n_random < 0:
        raise ValueError(f"n_random must be >= 0, got {n_random}")
    source, target = Path(source_path), Path(target_path)
    layout = _pair_layout(source, target)
    # The layout dispatch: one entry per layout. A layout's own module is
    # imported here, not at the top, because it imports this one's helpers.
    if layout is PrimaryStorageLayout.RAGGED:
        from opengwasdb.validation.observed_view_ragged import check_ragged

        return check_ragged(
            source, target, seed=seed, n_random=n_random, ld_panel=ld_panel, ancestry=ancestry
        )
    if ld_panel is not None or ancestry is not None:
        raise ObservedViewRefused(
            "--ld-panel and --ancestry are only for a Ragged pair, "
            f"not a {_LAYOUT_NAMES[layout]} one"
        )
    if layout is PrimaryStorageLayout.DENSE:
        return _check_dense(source, target, seed=seed, n_random=n_random)
    if layout is PrimaryStorageLayout.HYBRID:
        from opengwasdb.validation.observed_view_hybrid import check_hybrid

        return check_hybrid(source, target, seed=seed, n_random=n_random)
    assert_never(layout)


def _pair_layout(source_path: Path, target_path: Path) -> PrimaryStorageLayout:
    """The layout the two Stores share: a pair of two layouts has nothing to line up."""
    source = _open_checked(source_path, "source", CompletionState.OBSERVED_ONLY)
    target = _open_checked(target_path, "target", CompletionState.REFERENCE_COMPLETED)
    layout = target.manifest.primary_layout
    if source.manifest.primary_layout is not layout:
        raise ObservedViewRefused(
            f"source store {source_path} is not {_LAYOUT_NAMES[layout]} "
            f"(primary_layout={source.manifest.primary_layout.value}) as target store "
            f"{target_path} is; a mixed-layout pair cannot be checked"
        )
    return layout


_LAYOUT_NAMES = {
    PrimaryStorageLayout.DENSE: "Dense",
    PrimaryStorageLayout.RAGGED: "Ragged",
    PrimaryStorageLayout.HYBRID: "Hybrid",
}


def _check_dense(
    source_path: Path, target_path: Path, *, seed: int, n_random: int
) -> ObservedViewVerdict:
    started = time.time()
    pair = _open_pair(source_path, target_path)
    analyses = _analyses_record(pair)
    axis = _map_axis(pair.source_path, pair.target_path, pair.parent.n_rows, pair.child.n_rows)
    chunk_rows = int(inner_chunk_of(pair.parent.z.array)[0])
    band_rows = completion_band_rows(_shard_rows(pair.child.z.array))
    record = _header_record(pair, seed, n_random, analyses, axis, (chunk_rows, band_rows))
    axis_seconds = time.time() - started
    if axis.ok and analyses["identical"]:
        record.update(
            _sample_and_compare(
                axis, pair.parent, pair.child, (chunk_rows, band_rows), seed, n_random
            )
        )
    else:
        # Rows that do not line up would only be compared against the wrong
        # rows, so sampling is skipped and the verdict says why it failed.
        record.update(_unsampled_record())
    record["failures"] = _failures(record, axis)
    record["passed"] = not record["failures"]
    record["seconds"] = {
        "axis": round(axis_seconds, 1),
        "total": round(time.time() - started, 1),
    }
    return ObservedViewVerdict(record)


@dataclass(frozen=True)
class _Pair:
    """Both Stores, opened and found checkable."""

    source_path: Path
    target_path: Path
    source: OpenGWASDBStore
    target: OpenGWASDBStore
    parent: _Planes
    child: _Planes


def _open_pair(source_path: Path, target_path: Path) -> _Pair:
    source = _open_checked(source_path, "source", CompletionState.OBSERVED_ONLY)
    target = _open_checked(target_path, "target", CompletionState.REFERENCE_COMPLETED)
    parent, child = _Planes(source), _Planes(target)
    _refuse_if_incoherent(source_path, parent, "source")
    _refuse_if_incoherent(target_path, child, "target")
    if child.imputed is None:
        raise ObservedViewRefused(f"target store {target_path} has no `imputed` mask to check")
    return _Pair(source_path, target_path, source, target, parent, child)


def _header_record(
    pair: _Pair,
    seed: int,
    n_random: int,
    analyses: dict[str, Any],
    axis: _AxisMap,
    geometry: tuple[int, int],
) -> dict[str, Any]:
    chunk_rows, band_rows = geometry
    return {
        "seed": seed,
        "n_random": n_random,
        "source": _store_record(pair.source, pair.source_path),
        "target": _store_record(pair.target, pair.target_path),
        "analyses": analyses,
        "axis": axis.record(pair.parent.n_rows, pair.child.n_rows),
        "geometry": {
            "parent_chunk_rows": chunk_rows,
            "n_parent_chunk_rows": -(-pair.parent.n_rows // chunk_rows),
            "completion_band_rows": band_rows,
        },
    }


def _open_checked(path: Path, role: str, wanted: CompletionState) -> OpenGWASDBStore:
    try:
        store = open_store(path)
    except (OSError, ValueError, KeyError) as exc:
        raise ObservedViewRefused(f"cannot open {role} store {path}: {exc}") from exc
    state = store.manifest.completion_state
    if state is not wanted:
        raise ObservedViewRefused(
            f"{role} store {path} is not {_STATE_NAMES[wanted]} (completion_state={state.value})"
        )
    return store


_STATE_NAMES = {
    CompletionState.OBSERVED_ONLY: "Observed-Only",
    CompletionState.REFERENCE_COMPLETED: "Reference-Completed",
}


def _refuse_if_incoherent(path: Path, planes: _Planes, role: str) -> None:
    """A Store whose planes and variant axis disagree cannot be mapped row for row."""
    with closing(VariantAxis(path)) as axis:
        n_axis = axis.n_variants
    if n_axis != planes.n_rows:
        raise ObservedViewRefused(
            f"{role} store {path} lists {n_axis} variants but its z plane has "
            f"{planes.n_rows} rows"
        )
    if planes.n_rows == 0:
        raise ObservedViewRefused(f"{role} store {path} has no variants to check")


def _store_record(store: OpenGWASDBStore, path: Path) -> dict[str, str]:
    manifest = store.manifest
    return {
        "store_id": manifest.store_id,
        "release_id": manifest.release_id,
        "path": str(path),
    }


def _analyses_record(pair: _Pair) -> dict[str, Any]:
    source_ids = [
        a.analysis_id for a in read_analysis_records(pair.source_path / "analyses.tsv")
    ]
    target_ids = [
        a.analysis_id for a in read_analysis_records(pair.target_path / "analyses.tsv")
    ]
    columns_agree = (
        pair.parent.n_analyses == len(source_ids) and pair.child.n_analyses == len(target_ids)
    )
    return {
        "n_parent": len(source_ids),
        "n_child": len(target_ids),
        "identical": source_ids == target_ids and columns_agree,
    }


def _shard_rows(z_array: Any) -> int:
    """The variant-axis height of a completed plane's shard, which the writer's bands use."""
    shards = getattr(z_array, "shards", None)
    return int(shards[0]) if shards is not None else int(inner_chunk_of(z_array)[0])


# --- the exhaustive axis map ------------------------------------------------


class _AlidIndex(NamedTuple):
    """A Store's sorted ALID search index, which leaves out ALIDs wider than its key."""

    alids: np.ndarray
    rows: np.ndarray

    @classmethod
    def load(cls, store_path: Path) -> _AlidIndex:
        bytes_path = variant_alid_bytes_path(store_path)
        rows_path = variant_alid_rows_path(store_path)
        if not (bytes_path.exists() and rows_path.exists()):
            # A Store built before the index existed: every row is resolved by scan.
            return cls(np.empty(0, dtype=f"S{_CHROMOSOME_WIDTH}"), np.empty(0, dtype="int64"))
        return cls(np.load(bytes_path, mmap_mode="r"), np.load(rows_path, mmap_mode="r"))

    def unindexed_rows(self, n_rows: int) -> np.ndarray:
        indexed = np.zeros(n_rows, dtype=bool)
        indexed[np.asarray(self.rows, dtype="int64")] = True
        return np.flatnonzero(~indexed)


def _map_axis(source_path: Path, target_path: Path, n_source: int, n_target: int) -> _AxisMap:
    """Map every source row to its row in the completed axis, exactly.

    The ALID index is a fast path, not the definition: it holds only ALIDs that
    fit its fixed key, so a long indel is absent from it (#127). Rows the index
    does not cover are read from the variant table, so a long-allele row is
    mapped rather than reported unmapped, and a Store with no index at all is
    still checkable. Nothing is skipped, truncated or hashed.
    """
    source_index, target_index = _AlidIndex.load(source_path), _AlidIndex.load(target_path)
    source_loose = source_index.unindexed_rows(n_source)
    target_loose = target_index.unindexed_rows(n_target)
    mapped = np.full(n_source, -1, dtype="int64")
    _map_through_index(mapped, source_index, target_index)
    chromosome = _source_chromosomes(source_index, n_source)
    source_alids: dict[int, str] = {}
    if source_loose.size:
        with closing(VariantAxis(source_path)) as axis:
            for row, record in axis.by_indices(source_loose.tolist()).items():
                source_alids[row] = record.alid
                chromosome[row] = _chromosome_bytes(record.chromosome)
    _map_residual(mapped, source_index, source_alids, target_index, target_path, target_loose)
    found = mapped >= 0
    return _AxisMap(
        source_to_target=mapped,
        chromosome=chromosome,
        unmapped=int((~found).sum()),
        order_preserved=bool(np.all(np.diff(mapped[found]) > 0)),
        source_unindexed=int(source_loose.size),
        target_unindexed=int(target_loose.size),
    )


def _map_through_index(mapped: np.ndarray, source: _AlidIndex, target: _AlidIndex) -> None:
    if not (source.alids.size and target.alids.size):
        return
    slot = np.minimum(np.searchsorted(target.alids, source.alids), target.alids.size - 1)
    hit = target.alids[slot] == source.alids
    mapped[np.asarray(source.rows, dtype="int64")[hit]] = np.asarray(target.rows)[slot[hit]]


def _source_chromosomes(index: _AlidIndex, n_rows: int) -> np.ndarray:
    chromosome = np.zeros(n_rows, dtype=f"S{_CHROMOSOME_WIDTH}")
    if index.alids.size:
        names = np.char.partition(index.alids, b":")[:, 0]
        chromosome[np.asarray(index.rows, dtype="int64")] = names
    return chromosome


def _chromosome_bytes(name: str) -> bytes:
    encoded = name.encode("utf-8")
    if len(encoded) > _CHROMOSOME_WIDTH:
        raise ValueError(f"chromosome name {name!r} is wider than {_CHROMOSOME_WIDTH} bytes")
    return encoded


def _map_residual(
    mapped: np.ndarray,
    source_index: _AlidIndex,
    source_alids: dict[int, str],
    target_index: _AlidIndex,
    target_path: Path,
    target_loose: np.ndarray,
) -> None:
    """Resolve, by ALID string, the rows the index fast path did not map.

    A source ALID recovered by scanning can name a target row the target's index
    covers *or* one it leaves out, so both are searched. Rows of an indexed source
    that an indexed target does not contain are genuinely unmapped, so the
    target's table is read only when a scanned source row or an unindexed target
    row makes there something to find.
    """
    _map_scanned_through_index(mapped, source_alids, target_index)
    residual = np.flatnonzero(mapped < 0)
    if not (residual.size and target_loose.size):
        return
    alids = _unmapped_source_alids(mapped, source_index, source_alids)
    with closing(VariantAxis(target_path)) as axis:
        target_row_of = {r.alid: row for row, r in axis.by_indices(target_loose.tolist()).items()}
    for row in residual.tolist():
        if alids[row] in target_row_of:
            mapped[row] = target_row_of[alids[row]]


def _map_scanned_through_index(
    mapped: np.ndarray, source_alids: dict[int, str], target: _AlidIndex
) -> None:
    rows = [row for row, alid in source_alids.items() if is_indexable_alid(alid)]
    if not (rows and target.alids.size):
        return
    keys = np.array([source_alids[row].encode("utf-8") for row in rows], dtype=target.alids.dtype)
    slot = np.minimum(np.searchsorted(target.alids, keys), target.alids.size - 1)
    hit = target.alids[slot] == keys
    mapped[np.asarray(rows, dtype="int64")[hit]] = np.asarray(target.rows)[slot[hit]]


def _unmapped_source_alids(
    mapped: np.ndarray, source_index: _AlidIndex, source_alids: dict[int, str]
) -> dict[int, str]:
    """The ALID of every source row still unmapped: scanned ones, then indexed ones."""
    rows = np.asarray(source_index.rows, dtype="int64")
    unfound = mapped[rows] < 0
    alids = dict(source_alids)
    for row, alid in zip(rows[unfound], source_index.alids[unfound], strict=True):
        alids[int(row)] = bytes(alid).decode("utf-8")
    return alids


# --- the sample -------------------------------------------------------------


def _choose_bands(
    axis: _AxisMap, *, chunk_rows: int, band_rows: int, n_target: int, seed: int, n_random: int
) -> dict[int, set[str]]:
    """Source chunk-rows to compare, each with the strata that chose it.

    The draw order is part of the record: `default_rng(seed)` is used for the
    random stratum only, so a given seed picks the same chunk-rows however the
    deliberate strata are defined.
    """
    n_source = len(axis.source_to_target)
    n_bands = -(-n_source // chunk_rows)
    bands: dict[int, set[str]] = {}

    def add(band: int, stratum: str) -> None:
        bands.setdefault(int(band), set()).add(stratum)

    rng = np.random.default_rng(seed)
    for band in rng.choice(n_bands, size=min(n_random, n_bands), replace=False):
        add(band, "random")
    chrom = axis.chromosome
    starts = np.flatnonzero(np.r_[True, chrom[1:] != chrom[:-1]])
    ends = np.r_[starts[1:] - 1, n_source - 1]
    for first, last in zip(starts.tolist(), ends.tolist(), strict=True):
        add(first // chunk_rows, "chrom_edge")
        add(last // chunk_rows, "chrom_edge")
    add(n_bands - 1, "final_partial")
    for boundary in range(band_rows, n_target, band_rows):
        # The chunk-rows whose spans hold the child rows either side of the boundary.
        for child_row in (boundary - 1, boundary):
            add(_span_owner(axis.source_to_target, child_row, chunk_rows, n_bands),
                "writer_band_boundary")
    return bands


def _span_owner(source_to_target: np.ndarray, child_row: int, chunk_rows: int, n_bands: int) -> int:
    """The chunk-row whose span holds `child_row`: the first whose last row maps at or after it.

    Rows past the last source row belong to the final chunk-row.
    """
    first_source = int(np.searchsorted(source_to_target, child_row))
    return min(first_source // chunk_rows, n_bands - 1)


def _child_span(source_to_target: np.ndarray, r0: int, r1: int, n_child: int) -> tuple[int, int]:
    """Child rows `[c0, c1)` of the chunk-row of source rows `[r0, r1)`.

    The spans partition the child axis: a span starts after the previous chunk-row's
    last mapped row and ends at its own, the first starts at row 0 and the last runs
    to the end. Every child row, inserted ones included, is then in exactly one
    chunk-row's span, so sampling the whole grid examines the whole child.
    """
    c0 = 0 if r0 == 0 else int(source_to_target[r0 - 1]) + 1
    c1 = n_child if r1 == len(source_to_target) else int(source_to_target[r1 - 1]) + 1
    return c0, c1


def _sample_and_compare(
    axis: _AxisMap,
    parent: _Planes,
    child: _Planes,
    geometry: tuple[int, int],
    seed: int,
    n_random: int,
) -> dict[str, Any]:
    chunk_rows, band_rows = geometry
    bands = _choose_bands(
        axis, chunk_rows=chunk_rows, band_rows=band_rows, n_target=child.n_rows,
        seed=seed, n_random=n_random,
    )
    totals = dict.fromkeys(_TOTAL_KEYS, 0)
    max_diff = dict.fromkeys(_DIFF_PLANES, 0.0)
    listing: list[dict[str, Any]] = []
    dirty: list[int] = []
    for band in sorted(bands):
        r0, r1 = band * chunk_rows, min((band + 1) * chunk_rows, parent.n_rows)
        span = _child_span(axis.source_to_target, r0, r1, child.n_rows)
        counts, diffs = _check_band(axis.source_to_target[r0:r1], (r0, r1), span, parent, child)
        for key in totals:
            totals[key] += counts[key]
        for plane in _DIFF_PLANES:
            max_diff[plane] = max(max_diff[plane], diffs[plane])
        if any(counts[key] for key in FAILING_COUNTERS):
            dirty.append(band)
        listing.append(
            {"band": band, "parent_rows": [r0, r1], "strata": sorted(bands[band])}
        )
    return {
        "bands_checked": len(bands),
        "bands_by_stratum": {s: sum(s in strata for strata in bands.values()) for s in STRATA},
        "bands": listing,
        "bands_with_mismatch": dirty,
        "totals": totals,
        "max_abs_diff": max_diff,
    }


def _unsampled_record() -> dict[str, Any]:
    return {
        "bands_checked": 0,
        "bands_by_stratum": dict.fromkeys(STRATA, 0),
        "bands": [],
        "bands_with_mismatch": [],
        "totals": dict.fromkeys(_TOTAL_KEYS, 0),
        "max_abs_diff": dict.fromkeys(_DIFF_PLANES, 0.0),
    }


def _failures(record: Mapping[str, Any], axis: _AxisMap) -> list[str]:
    failures = []
    if axis.unmapped:
        failures.append("parent_rows_unmapped")
    if not axis.order_preserved:
        failures.append("order_not_preserved")
    if not record["analyses"]["identical"]:
        failures.append("analyses_mismatch")
    failures.extend(name for name in FAILING_COUNTERS if record["totals"][name])
    return failures


# --- one chunk-row ----------------------------------------------------------


def _check_band(
    rows: np.ndarray,
    source_rows: tuple[int, int],
    span: tuple[int, int],
    parent: _Planes,
    child: _Planes,
) -> tuple[dict[str, int], dict[str, float]]:
    """Compare one source chunk-row with the completed span it owns (see `_child_span`)."""
    (r0, r1), (c0, c1) = source_rows, span
    assert child.imputed is not None  # checked before sampling began
    raw_imputed = np.asarray(child.imputed[c0:c1])
    return _compare(parent.read(r0, r1), child.read(c0, c1), raw_imputed, rows - c0)


def _compare(
    parent: _Cells, child: _Cells, raw_imputed: np.ndarray, local: np.ndarray
) -> tuple[dict[str, int], dict[str, float]]:
    """The cell checks. `local` maps each source row to its row within the span.

    Missingness and the imputed mask's domain are checked *before* cells are
    classified, so a cell that is neither observed nor missing, or neither imputed
    nor not, is a counted failure rather than being read as the nearest category.
    """
    imputed = raw_imputed.astype(np.uint8)
    mapped = _Cells(child.z[local], child.se[local], child.eaf[local])
    mapped_imputed = imputed[local]
    observed = _observed(parent)
    child_observed = _observed(mapped)
    inserted = np.ones(len(imputed), dtype=bool)
    inserted[local] = False
    on_inserted = _observed(_Cells(child.z[inserted], child.se[inserted], child.eaf[inserted]))
    counts = {
        "cells_compared": int(parent.z.size),
        "parent_observed": int(observed.sum()),
        "imputed_mask_invalid": int(((raw_imputed != 0) & (raw_imputed != 1)).sum()),
        "z_se_missingness_mismatch": _partial_cells(parent) + _partial_cells(child),
        "observed_marked_imputed": int((mapped_imputed[observed] != 0).sum()),
        "observed_missing_in_child": int((~child_observed[observed]).sum()),
        "missing_became_observed": int((child_observed & ~observed & (mapped_imputed == 0)).sum()),
        "missing_now_imputed": int((child_observed & ~observed & (mapped_imputed == 1)).sum()),
        "inserted_rows": int(inserted.sum()),
        "inserted_observed_cells": int((on_inserted & (imputed[inserted] == 0)).sum()),
        "parent_rows_without_finite": int((~observed.any(axis=1)).sum()),
    }
    diffs: dict[str, float] = {}
    for plane in _DIFF_PLANES:
        counts[f"{plane}_mismatch"], diffs[plane] = _plane_difference(
            getattr(parent, plane)[observed], getattr(mapped, plane)[observed]
        )
    return counts, diffs


def _observed(cells: _Cells) -> np.ndarray:
    """An observed cell has a finite Z and SE; EAF is carried, not what makes it observed."""
    observed: np.ndarray = np.isfinite(cells.z) & np.isfinite(cells.se)
    return observed


def _partial_cells(cells: _Cells) -> int:
    """Cells with a Z but no SE, or the reverse: their missingness must agree (spec section 20)."""
    return int((np.isfinite(cells.z) != np.isfinite(cells.se)).sum())


def _plane_difference(parent: np.ndarray, child: np.ndarray) -> tuple[int, float]:
    """Cells that differ, NaN-equal, and the largest gap where both sides are finite.

    A NaN against a number is a mismatch the count carries and the gap cannot,
    so `max_abs_diff` of 0.0 never stands in for agreement: `passed` and the
    counts decide that.
    """
    equal = (parent == child) | (np.isnan(parent) & np.isnan(child))
    gap = np.abs(parent.astype(np.float64) - child.astype(np.float64))
    gap = gap[np.isfinite(gap)]
    return int((~equal).sum()), float(gap.max()) if gap.size else 0.0
