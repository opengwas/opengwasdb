"""On-disk layout of a Hybrid store.

A single integrated Store directory holds two components over one shared Store
Variant Table::

    <store>/
      manifest.json                 # primary_layout = hybrid
      variants.tsv.gz (+ sidecars)  # SHARED union table (panel ∪ off-panel-observed)
      analyses.tsv                  # the Analyses, one analysis_index space
      dense/                        # nested Dense Component
        manifest.json,
        data.zarr/ (z, se, top_hits[, imputed, on_panel]),
        dense_to_shared.npy         # int32[n_panel]: dense row -> shared variant_index
        index.sqlite                # Reference-Completed only: completion_quality
      data.zarr/
        ragged/                     # Overflow CSR (offsets, variant_index=SHARED, z, se)
        top_hits/                   # (optional) merged hybrid top-hit cache

The Dense Component's rows are panel-local (0..n_panel-1, == its zarr rows).
``dense_to_shared.npy`` maps those rows onto the shared variant index space, which
the Ragged Overflow references directly. The map is strictly ascending (panel
variants keep their genomic order within the union), so ``searchsorted`` gives
shared → dense translation (`panel_rows`).

**From format 0.3.0 the Dense Component carries no variant identity of its own**
(#472, ADR 0102 §4): no Variant Table, `.tbi`, ALID, position or rsID sidecar, no
`analyses.tsv` and no `overview.html`. Its Variant Axis is the shared one seen through
``dense_to_shared`` (`opengwasdb.store_reader.panel_axis`), and its Analyses are the
root's. Before 0.3.0 it carried a copy of each, which a completed release let fall out
of step with the shared table (#481).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from opengwasdb.store_reader.format import dense_component_shares_axis

DENSE_SUBDIR = "dense"
DENSE_TO_SHARED_FILENAME = "dense_to_shared.npy"


def dense_component_path(store_path: str | Path) -> Path:
    """Path of the nested Dense Component store."""
    return Path(store_path) / DENSE_SUBDIR


def dense_to_shared_path(store_path: str | Path) -> Path:
    """Path of the dense-row → shared-variant-index map (int32, length n_panel)."""
    return dense_component_path(store_path) / DENSE_TO_SHARED_FILENAME


class DenseComponentSharesAxis(ValueError):
    """A 0.3.0 Hybrid's Dense Component was asked for a Variant Axis, an index or a table
    of its own. It carries none (#472): `StoreRelease.variant_axis()` gives the shared
    axis in its panel rows, and that axis refuses an index it does not have."""


class UnreadableHybridManifest(ValueError):
    """A `dense/` directory sits under a `manifest.json` that cannot be read as JSON, so
    whether it is a Hybrid's Dense Component cannot be told."""


def hybrid_root_of(component_path: str | Path) -> Path | None:
    """The Hybrid release that nests the release at `component_path` as its Dense
    Component, or None when it is not one.

    A Dense Component is the `dense/` directory of a release whose manifest's
    `primary_layout` is `hybrid` (spec §16). Its own manifest says `dense`, so only its
    place says what it is. A parent `manifest.json` that exists but cannot be read is
    refused (`UnreadableHybridManifest`), never read as "not a Hybrid": that would open
    a 0.3.0 Dense Component as a standalone Dense release, missing every index.
    """
    path = Path(component_path)
    if path.name != DENSE_SUBDIR:
        return None
    manifest = path.parent / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UnreadableHybridManifest(
            f"{manifest}: cannot read the manifest of the release {path} would be the Dense "
            f"Component of: {exc}"
        ) from exc
    if not isinstance(recorded, dict):
        raise UnreadableHybridManifest(
            f"{manifest}: the manifest of the release {path} would be the Dense Component "
            f"of is not a JSON object"
        )
    return path.parent if recorded.get("primary_layout") == "hybrid" else None


def shared_axis_root(component_path: str | Path, format_version: str) -> Path | None:
    """The Hybrid whose Variant Axis and `analyses.tsv` the release at `component_path`,
    of `format_version`, reads; None when it carries its own.

    The one decision of whether a release is a Dense Component that shares the axis
    (#472): a Hybrid's Dense Component (`hybrid_root_of`) of a version that carries no
    copy (`dense_component_shares_axis`). `StoreRelease`, the Variant Axis's guard, the
    rewrite and the validator all ask it.

    It is decided from the filesystem, so it changes when the root's `manifest.json`
    appears. Hybrid completion writes the root manifest (phase 4) before it drops the
    completed component's own copies (`_drop_dense_copies`); between the two, the
    component opens as one that shares the axis, beside copies it is about to lose.
    Nothing opens it there today; anything added there would read the shared axis.
    """
    if not dense_component_shares_axis(format_version):
        return None
    return hybrid_root_of(component_path)


def panel_rows(
    dense_to_shared: np.ndarray, shared_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`(on_panel, rows)` for shared variant indices: which are rows of the Dense
    Component's panel, and the panel row of each that is (-1 for each that is not).

    The one place the panel map is searched: the Hybrid query facade and the Dense
    Component's Variant Axis (`panel_axis`) both ask it. It is one vectorised
    `searchsorted` with the needles cast to the map's own dtype. The per-variant
    Python form cast the whole 13.4 M-entry `int32` map on every call (22.6 ms against
    0.004 ms for an `int32` scalar on OGS-00011), and `range_phewas` made one call per
    shared variant -- 58,006 of them for a 1 Mb TCF7L2 window (#252). Searching that
    map with `int64` needles still promoted and copied it on every call (review round
    1, and again in #465 for one needle: 38 ms on OGS-00004), so the needles are
    narrowed instead; a needle outside the map's range cannot be a panel row and is
    masked out before the cast, so narrowing cannot wrap it into range, and a needle
    between two panel rows is off panel, never its neighbour's row.
    """
    shared_indices = np.asarray(shared_indices)
    on_panel = np.zeros(shared_indices.shape, dtype=bool)
    rows = np.full(shared_indices.shape, -1, dtype=np.int64)
    panel = dense_to_shared
    if shared_indices.size == 0 or panel.size == 0:
        return on_panel, rows
    low, high = int(panel[0]), int(panel[-1])
    in_range = (shared_indices >= low) & (shared_indices <= high)
    if in_range.any():
        needles = shared_indices[in_range].astype(panel.dtype, copy=False)
        pos = np.minimum(np.searchsorted(panel, needles), panel.size - 1)
        hit = panel[pos] == needles
        on_panel[in_range] = hit
        rows[np.flatnonzero(in_range)[hit]] = pos[hit]
    return on_panel, rows


def check_dense_to_shared(
    mapped: np.ndarray, n_shared: int, n_dense: int | None = None
) -> list[str]:
    """What is structurally wrong with a `dense_to_shared` map; nothing if it is sound.

    The one statement of the map's invariants, which the validator, the rewrite and the
    build each judge a map by: every entry names a row of the shared table of `n_shared`
    rows; the entries strictly ascend, so no shared row is named twice and `panel_rows`
    can search it; and, when `n_dense` is given, there is one entry per Dense row. Each
    fault is a finding of its own, naming the first row it is at.
    """
    rows = np.asarray(mapped, dtype=np.int64).reshape(-1)
    if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= n_shared):
        outside = int(np.flatnonzero((rows < 0) | (rows >= n_shared))[0])
        return [
            "dense_to_shared references rows outside the shared variant table: Dense "
            f"Component row {outside} names shared row {int(rows[outside])} of {n_shared}"
        ]
    found = _order_faults(rows)
    length = None if n_dense is None else dense_to_shared_length_fault(rows, n_dense)
    return found if length is None else [*found, length]


def dense_to_shared_length_fault(mapped: np.ndarray, n_dense: int) -> str | None:
    """The finding for a map that does not name one shared row per Dense row, if any:
    the one part of `check_dense_to_shared` that needs the Dense side."""
    if len(mapped) == n_dense:
        return None
    return (
        f"dense_to_shared maps {len(mapped)} Dense Component rows, but the Dense "
        f"Component holds {n_dense}: each of its rows must name its shared variant"
    )


def _order_faults(rows: np.ndarray) -> list[str]:
    """A shared row named twice, and a descending pair: each a finding of its own."""
    found: list[str] = []
    steps = np.diff(rows)
    if (steps == 0).any():
        twice = int(np.flatnonzero(steps == 0)[0])
        found.append(
            "dense_to_shared maps two Dense Component rows to one shared row: rows "
            f"{twice} and {twice + 1} both name {int(rows[twice])}"
        )
    elif len(rows) != len(np.unique(rows)):
        found.append("dense_to_shared maps two Dense Component rows to one shared row")
    if (steps < 0).any():
        down = int(np.flatnonzero(steps < 0)[0])
        found.append(
            "dense_to_shared is not ascending: Dense Component row "
            f"{down + 1} names shared row {int(rows[down + 1])}, before row {down}'s "
            f"{int(rows[down])}; the map is searched, and must keep the panel's order"
        )
    return found


def misnamed_panel_rows(
    mapped: np.ndarray,
    dense_alids: Iterable[str | None],
    shared_alids: Iterable[tuple[int, str | None]],
) -> Iterator[tuple[int, str, str | None]]:
    """Each Dense row whose mapped shared row holds another variant, or none: `(row,
    its ALID, the shared row's ALID or None)`.

    The one walk comparing a Dense Component's variants with the shared rows the map
    names. `dense_alids` are the Dense rows' ALIDs in row order; `shared_alids` are
    `(shared row, ALID)` pairs ascending, every shared row or only the mapped ones; a
    Dense or shared ALID that is None (a row an index leaves out) is a mismatch. Both
    are read once, side by side, so a streamed table costs no more than its read. Call it
    after `check_dense_to_shared`, which it relies on for the map's order and length.
    """
    shared = iter(shared_alids)
    for row, (target, alid) in enumerate(zip(mapped.tolist(), dense_alids, strict=True)):
        found = next(((r, a) for r, a in shared if r >= target), None)
        held = found[1] if found is not None and found[0] == target else None
        if alid is None or held != alid:
            yield row, str(alid), held
