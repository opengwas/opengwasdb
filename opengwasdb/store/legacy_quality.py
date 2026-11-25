"""Count a masked cell of a completed release in its block's `completion_quality` (#492).

A Reference-Completed release's `completion_quality` counts, per Analysis and LD block,
the panel cells completion left imputed or missing (`completion.quality_counts`). A
cell the legacy conversion masks on a panel row becomes one of those missing cells, as
a completion that left it out (ADR 0068) counts it. The release records no
block-to-cell membership, so the block is found only where it is unambiguous:

- the Analysis has one `completion_quality` row: that row;
- otherwise its rows' `block_id`s spell ``<chromosome>/<start>-<end>``, the form every
  completion writes, and exactly one holds the row's position -- where two share a
  boundary variant, the one that starts there (the owner's ruling, 5 Nov 2026).

Any other cell is refused, never guessed: its count would be silently wrong.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

import numpy as np

from opengwasdb.index.release_index import ReleaseIndex
from opengwasdb.store.open import OpenGWASDBStore, connect_index

#: A block_id as completion writes it.
_BLOCK = re.compile(r"^(?P<chrom>[^/]+)/(?P<start>\d+)-(?P<end>\d+)$")


class UnattributableCell(ValueError):
    """A masked cell whose `completion_quality` block cannot be told."""


def attribute(
    store: OpenGWASDBStore, rows: np.ndarray, analyses: np.ndarray
) -> dict[tuple[int, str], int]:
    """How many of the cells `(rows, analyses)` each `(analysis, block_id)` row gains in
    `n_missing`: the cells on panel rows, each in its one block (module docstring)."""
    rows, analyses = _on_panel(store, rows, analyses)
    if not rows.size:
        return {}
    blocks = _blocks(store)
    axis = store.variant_axis(aliases=False)
    try:
        records = axis.by_indices(rows.tolist())
    finally:
        axis.close()
    counted: Counter[tuple[int, str]] = Counter()
    for row, analysis in zip(rows.tolist(), analyses.tolist(), strict=True):
        record = records[row]
        block = _block_of(blocks.get(analysis, []), record.chromosome, record.position)
        if block is None:
            raise UnattributableCell(
                f"{store.path}: the cell at row {row} ({record.alid}), Analysis {analysis}, "
                "is masked on a panel row, and no one completion_quality block of that "
                f"Analysis holds it ({len(blocks.get(analysis, []))} block(s)); its n_missing "
                "would be wrong, so the release is rebuilt from source"
            )
        counted[(analysis, block)] += 1
    return dict(counted)


def _on_panel(
    store: OpenGWASDBStore, rows: np.ndarray, analyses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The cells on panel rows; a release with cells and no `on_panel` is refused, never
    read as having none on the panel."""
    if not rows.size:
        return rows, analyses
    root = store.arrays()
    if "on_panel" not in root:
        raise UnattributableCell(
            f"{store.path}: a Reference-Completed release with no on_panel array cannot say "
            "which masked cells are panel cells its completion_quality counts"
        )
    held = np.asarray(root["on_panel"][:])[rows] == 1
    return rows[held], analyses[held]


def _blocks(store: OpenGWASDBStore) -> dict[int, list[str]]:
    """Each Analysis's `completion_quality` block ids."""
    blocks: dict[int, list[str]] = {}
    with ReleaseIndex(store.path) as index:
        for analysis, block_id, *_counts in index.completion_quality():
            blocks.setdefault(int(analysis), []).append(str(block_id))
    return blocks


def _block_of(blocks: list[str], chromosome: str, position: int) -> str | None:
    """The one block of `blocks` holding the row, or None where none or several do."""
    if len(blocks) == 1:
        return blocks[0]
    holding = _holding(blocks, chromosome, position)
    if len(holding) > 1:
        holding = [(start, block) for start, block in holding if start == position]
    return holding[0][1] if len(holding) == 1 else None


def _holding(blocks: list[str], chromosome: str, position: int) -> list[tuple[int, str]]:
    """Each block spelt ``<chromosome>/<start>-<end>`` that holds the position, with its
    start; none when any block is spelt otherwise, as nothing then says where it is."""
    spelt = [_BLOCK.match(block) for block in blocks]
    if not all(spelt):
        return []
    return [
        (int(match["start"]), block)
        for block, match in zip(blocks, spelt, strict=True)
        if match and match["chrom"] == chromosome
        and int(match["start"]) <= position <= int(match["end"])
    ]


def count_missing(staged: Path, counted: Iterable[tuple[tuple[int, str], int]]) -> None:
    """Add each count to its row's `n_missing` in the staged `index.sqlite`."""
    with closing(connect_index(staged / "index.sqlite")) as index:
        for (analysis, block), n in counted:
            changed = index.execute(
                "UPDATE completion_quality SET n_missing = n_missing + ? "
                "WHERE analysis_index = ? AND block_id = ?",
                (n, analysis, block),
            ).rowcount
            if changed != 1:
                raise UnattributableCell(
                    f"{staged}: completion_quality has no row for Analysis {analysis}, "
                    f"block {block!r}"
                )
        index.commit()


__all__ = ["UnattributableCell", "attribute", "count_missing"]
