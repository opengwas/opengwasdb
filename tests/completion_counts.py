"""What a completion's `completion_quality` counts, held against the rows it wrote.

The owner's ruling of 11 Nov 2026 (ADR 0039, ADR 0079's note): a block counts, per
Analysis, only rows the Store holds. Each panel row it names that the Analysis does
not observe, inside the Analysis's regions and not left to the block that starts at
its end (ADR 0072), is counted imputed when the block filled it and missing when it
did not; a panel row no ALID names is never a Store row and is counted nowhere. So
a quality row's `n_imputed + n_missing` is the number of rows completion added that
Analysis that the block owns. Counted per block, the rule holds where blocks
overlap too: a row two blocks own is counted by each.

Ragged writes a missing row as a row; Dense writes every panel row on its axis, so
there a quality row's counts are its owned cells: imputed, or left without a value.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import numpy as np
from stored_rows import stored_rows

from opengwasdb.completion.ld_panel import LDBlock, list_all_blocks, shared_endpoints
from opengwasdb.encoding import DenseZPlane
from opengwasdb.store.open import open_store
from opengwasdb.variants.axis import VariantAxis


def _alids(store: Path, analysis: int) -> set[str]:
    rows = stored_rows(store, analysis)
    with closing(VariantAxis(store)) as axis:
        records = axis.by_indices(rows.variant_index.tolist())
    return {records[int(v)].alid for v in rows.variant_index}


def quality_counts(store: Path) -> dict[tuple[int, str], tuple[int, int]]:
    """`(n_imputed, n_missing)` of every `completion_quality` row, by (Analysis, block)."""
    with closing(open_store(store).index_connection()) as db:
        return {
            (int(a), str(b)): (int(i), int(m))
            for a, b, i, m in db.execute(
                "SELECT analysis_index, block_id, n_imputed, n_missing FROM completion_quality"
            )
        }


def quality_rows(store: Path) -> dict[tuple[int, str], int]:
    """`n_imputed + n_missing` of every `completion_quality` row, by (Analysis, block)."""
    return {key: i + m for key, (i, m) in quality_counts(store).items()}


def _blocks(panel: Path, ancestry: str, block_ids: set[str]) -> dict[str, LDBlock]:
    by_id: dict[str, LDBlock] = {}
    for chromosome in {block_id.split("/")[0] for block_id in block_ids}:
        for block in list_all_blocks(panel, ancestry, chromosome):
            by_id[block.block_id] = block
    missing = block_ids - by_id.keys()
    assert not missing, f"completion_quality names blocks the panel lacks: {sorted(missing)}"
    return by_id


def owned_alids(
    keys: set[tuple[int, str]], panel: Path, ancestry: str = "EUR"
) -> dict[tuple[int, str], set[str]]:
    """Per (Analysis, block) of `keys`, the ALIDs the block names and does not cede to
    the next block the Analysis is completed against (ADR 0072)."""
    blocks = _blocks(panel, ancestry, {block_id for _a, block_id in keys})
    endpoints = shared_endpoints(blocks.values())
    blocks_of: dict[int, set[str]] = {}
    for analysis, block_id in keys:
        blocks_of.setdefault(analysis, set()).add(block_id)
    owned: dict[tuple[int, str], set[str]] = {}
    for analysis, block_id in keys:
        named = {alid for alid in blocks[block_id].alids if alid is not None}
        endpoint = endpoints.get(block_id)
        if endpoint is not None and endpoint.following in blocks_of[analysis]:
            named -= endpoint.alids
        owned[(analysis, block_id)] = named
    return owned


def owned_added_rows(
    completed: Path, source: Path, panel: Path, ancestry: str = "EUR"
) -> dict[tuple[int, str], int]:
    """Per (Analysis, block) of `completion_quality`, the rows completion added the
    Analysis -- in `completed`, not in `source` -- that the block owns."""
    owned = owned_alids(set(quality_rows(completed)), panel, ancestry)
    added = {a: _alids(completed, a) - _alids(source, a) for a in {a for a, _b in owned}}
    return {key: len(added[key[0]] & alids) for key, alids in owned.items()}


def _differing(found: dict, counted: dict, what: str) -> dict:
    return {
        key: {what: found.get(key), "counted": counted.get(key)}
        for key in found.keys() | counted.keys()
        if found.get(key) != counted.get(key)
    }


def assert_counts_are_rows(
    completed: Path, source: Path, panel: Path, ancestry: str = "EUR"
) -> None:
    """Every `completion_quality` row counts exactly the rows its block owns."""
    counted = quality_rows(completed)
    assert counted, "the completion records no quality row, so nothing is under test"
    differing = _differing(
        owned_added_rows(completed, source, panel, ancestry), counted, "rows owned"
    )
    assert not differing, differing


def dense_owned_cells(
    completed: Path, panel: Path, ancestry: str = "EUR"
) -> dict[tuple[int, str], tuple[int, int]]:
    """Per (Analysis, block) of a Dense completion's `completion_quality`, the cells
    of the rows the block owns that completion imputed, and those left without a
    value. Every owned row is on the axis: Dense writes the panel's rows as its own."""
    owned = owned_alids(set(quality_counts(completed)), panel, ancestry)
    store = open_store(completed)
    root = store.arrays(mode="r")
    n_variants = int(root["z"].shape[0])
    z = DenseZPlane.open(root, store.manifest.encoding).band(0, n_variants)
    imputed = root["imputed"][:].astype(bool)
    with closing(VariantAxis(completed)) as axis:
        row_of = {record.alid: record.variant_index for record in axis.all()}
    cells: dict[tuple[int, str], tuple[int, int]] = {}
    for (analysis, block_id), alids in owned.items():
        off_axis = alids - row_of.keys()
        assert not off_axis, f"{block_id} owns rows the axis lacks: {sorted(off_axis)[:5]}"
        rows = np.array(sorted(row_of[alid] for alid in alids), dtype=np.int64)
        filled = imputed[rows, analysis]
        empty = np.isnan(z[rows, analysis]) & ~filled
        cells[(analysis, block_id)] = (int(filled.sum()), int(empty.sum()))
    return cells


def assert_dense_counts_are_cells(completed: Path, panel: Path, ancestry: str = "EUR") -> None:
    """Every Dense quality row's `n_imputed` is its owned cells completion imputed,
    and its `n_missing` the owned cells it left without a value."""
    counted = quality_counts(completed)
    assert counted, "the completion records no quality row, so nothing is under test"
    differing = _differing(dense_owned_cells(completed, panel, ancestry), counted, "cells")
    assert not differing, differing
