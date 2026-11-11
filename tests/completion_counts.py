"""What a completion's `completion_quality` counts, held against the rows it wrote.

The owner's ruling of 11 Nov 2026 (ADR 0039, ADR 0079's note): a block counts, per
Analysis, only rows the Store holds. Each panel row it names that the Analysis does
not observe, inside the Analysis's regions and not left to the block that starts at
its end (ADR 0072), is counted imputed when the block filled it and missing when it
did not; a panel row no ALID names is never a Store row and is counted nowhere. So
a quality row's `n_imputed + n_missing` is the number of rows completion added that
Analysis that the block owns. Counted per block, the rule holds where blocks
overlap too: a row two blocks own is counted by each.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from stored_rows import stored_rows

from opengwasdb.completion.ld_panel import LDBlock, list_all_blocks, shared_endpoints
from opengwasdb.variants.axis import VariantAxis


def _alids(store: Path, analysis: int) -> set[str]:
    rows = stored_rows(store, analysis)
    with closing(VariantAxis(store)) as axis:
        records = axis.by_indices(rows.variant_index.tolist())
    return {records[int(v)].alid for v in rows.variant_index}


def quality_rows(store: Path) -> dict[tuple[int, str], int]:
    """`n_imputed + n_missing` of every `completion_quality` row, by (Analysis, block)."""
    with closing(sqlite3.connect(store / "index.sqlite")) as db:
        return {
            (int(a), str(b)): int(i) + int(m)
            for a, b, i, m in db.execute(
                "SELECT analysis_index, block_id, n_imputed, n_missing FROM completion_quality"
            )
        }


def _blocks(panel: Path, ancestry: str, block_ids: set[str]) -> dict[str, LDBlock]:
    by_id: dict[str, LDBlock] = {}
    for chromosome in {block_id.split("/")[0] for block_id in block_ids}:
        for block in list_all_blocks(panel, ancestry, chromosome):
            by_id[block.block_id] = block
    missing = block_ids - by_id.keys()
    assert not missing, f"completion_quality names blocks the panel lacks: {sorted(missing)}"
    return by_id


def owned_added_rows(
    completed: Path, source: Path, panel: Path, ancestry: str = "EUR"
) -> dict[tuple[int, str], int]:
    """Per (Analysis, block) of `completion_quality`, the rows completion added the
    Analysis -- in `completed`, not in `source` -- that the block's panel names and
    the block does not cede to the next one the Analysis is completed against."""
    counted = quality_rows(completed)
    blocks = _blocks(panel, ancestry, {block_id for _a, block_id in counted})
    endpoints = shared_endpoints(blocks.values())
    blocks_of: dict[int, set[str]] = {}
    for analysis, block_id in counted:
        blocks_of.setdefault(analysis, set()).add(block_id)
    added = {a: _alids(completed, a) - _alids(source, a) for a in blocks_of}
    owned: dict[tuple[int, str], int] = {}
    for (analysis, block_id) in counted:
        named = {alid for alid in blocks[block_id].alids if alid is not None}
        endpoint = endpoints.get(block_id)
        if endpoint is not None and endpoint.following in blocks_of[analysis]:
            named -= endpoint.alids
        owned[(analysis, block_id)] = len(added[analysis] & named)
    return owned


def assert_counts_are_rows(
    completed: Path, source: Path, panel: Path, ancestry: str = "EUR"
) -> None:
    """Every `completion_quality` row counts exactly the rows its block owns."""
    counted = quality_rows(completed)
    assert counted, "the completion records no quality row, so nothing is under test"
    owned = owned_added_rows(completed, source, panel, ancestry)
    differing = {
        key: {"rows owned": owned.get(key), "counted": counted.get(key)}
        for key in owned.keys() | counted.keys()
        if owned.get(key) != counted.get(key)
    }
    assert not differing, differing
