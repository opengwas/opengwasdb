"""How many rows of a Dense grid one streamed pass decodes at a time.

Every pass that reads a release's planes a band at a time -- validation's value
rules (issue #254), the Top-Hit Index scan (issue #313) -- is bounded by the same
decoded band, so the band is sized from one byte budget instead of a row count
that a wider release multiplies. The passes decode several raw, float32/float64
and boolean views of a cell at once, and the Top-Hit one holds a cell's worth per
plane rather than per rule: OGS-00009/10 profiling measured at most 59 bytes per
cell in the validation pass and 47 in the Top-Hit pass. One GiB at a 96-byte
allowance leaves 63% headroom inside this decoded-band budget, and the row cap
keeps the old band for a narrow release, where the budget never binds.

The budget bounds the cells one band decodes, nothing else: a worker's process
also holds its interpreter, its fork's copy of the parent and zarr's own buffers,
and the parent holds the harvest.
"""

from __future__ import annotations

from collections.abc import Iterator

#: The row cap: the band a narrow release had before the byte budget existed.
BAND_ROWS = 250_000

#: The decoded bytes one band may hold, however wide the release is.
DECODED_BAND_BYTES = 1024**3

#: The per-cell allowance the budget is divided by (see the module docstring).
BYTES_PER_CELL = 96


def decoded_band_bytes(working_bytes: int | None = None) -> int:
    """The decoded-band budget a pass uses: `working_bytes`, or this module's.

    One resolution point, read at call time, so a caller that replaces
    `DECODED_BAND_BYTES` for a whole pass -- a test shrinking it until a fixture
    spans several bands -- moves every reader of it, including the defaults of
    the functions below.
    """
    return DECODED_BAND_BYTES if working_bytes is None else working_bytes


def band_rows(n_analyses: int, row_chunk: int, working_bytes: int | None = None) -> int:
    """Rows whose decoded band fits the budget (`DECODED_BAND_BYTES` by default).

    Whole inner chunks avoid decoding a physical chunk into two consecutive
    bands. A single row is the irreducible minimum when an exceptionally wide
    store exceeds the budget by itself.
    """
    budget = decoded_band_bytes(working_bytes)
    if n_analyses <= 0:
        return BAND_ROWS
    rows = min(BAND_ROWS, max(1, budget // (BYTES_PER_CELL * n_analyses)))
    if row_chunk > 0 and rows >= row_chunk:
        rows = max(row_chunk, rows // row_chunk * row_chunk)
    return rows


def band_limits(
    n_variants: int, n_analyses: int, row_chunk: int, working_bytes: int | None = None
) -> Iterator[tuple[int, int]]:
    """Every Dense row band, exhaustively, under the decoded-band budget."""
    rows = band_rows(n_analyses, row_chunk, working_bytes)
    for r0 in range(0, n_variants, rows):
        yield r0, min(r0 + rows, n_variants)


def window_cells(cell_chunk: int, working_bytes: int | None = None) -> int:
    """Cells one sequence window holds: the same budget, counted in cells."""
    cells = max(1, decoded_band_bytes(working_bytes) // BYTES_PER_CELL)
    if cell_chunk > 0 and cells >= cell_chunk:
        cells = cells // cell_chunk * cell_chunk
    return cells
