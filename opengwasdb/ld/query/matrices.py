"""LD matrices: for a list of ALIDs, and for a region (spec §8; ogld#4).

Both return one signed-r matrix (owner, 7 November 2026). Every pair the store does not
hold is NaN, the Bound and Not Measured alike, and the matrix's positions and t_far say
which (`LDMatrix.kinds`). A caller's `fill` replaces exactly those cells, in this module
and nowhere else, and is recorded on the result, so a filled cell is never mistaken for a
stored one. The store itself is never written.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ogld.format import judge_region
from ogld.identity import normalise_chromosome
from ogld.query.reader import Reader as StoreReader
from ogld.query.results import LDMatrix


def matrix(reader: StoreReader, alids: Sequence[str], *, fill: float | None) -> LDMatrix:
    """The signed-r matrix of the present `alids`, in request order (spec §8; ogld#4).

    Unstored pairs are NaN, replaced by `fill` only when the caller passed one. Never a
    refusal: a pair more than far_bp apart is a NaN whose kind is Not Measured.
    """
    found = reader.rows_of(alids)
    absent = tuple(alid for alid, row in zip(alids, found, strict=True) if row < 0)
    rows = found[found >= 0]
    return _ld_matrix(reader, rows, reader.pair_r(rows), absent, fill)


def region(
    reader: StoreReader, chrom: str, centre: int, radius: int, *, fill: float | None
) -> LDMatrix:
    """The signed-r matrix of every variant with |position − centre| ≤ radius on `chrom`,
    in genomic order (spec §8; ogld#4). Refused unless 2 × radius ≤ far_bp."""
    judge_region(radius=radius, far_bp=reader.provenance["far_bp"]).raise_if_refused()
    rows = reader.rows_in(normalise_chromosome(chrom), centre - radius, centre + radius)
    # The region's rows are contiguous, so the reader reads each one as one slice (ogld#14).
    start, stop = (int(rows[0]), int(rows[-1]) + 1) if len(rows) else (0, 0)
    return _ld_matrix(reader, rows, reader.region_r(start, stop), (), fill)


def _ld_matrix(
    reader: StoreReader,
    rows: npt.NDArray[np.int64],
    r: npt.NDArray[np.float64],
    absent: tuple[str, ...],
    fill: float | None,
) -> LDMatrix:
    """The matrix `r` of `rows`, with the caller's `fill`, if any, in place of every NaN:
    written in place, so a query holds one float64 copy of its result (ogld#14)."""
    if fill is not None:
        r[np.isnan(r)] = float(fill)
    p = reader.provenance
    return LDMatrix(
        r=r,
        variants=reader.variants(rows),
        absent=absent,
        t_far=p["t_far"],
        near_bp=p["near_bp"],
        far_bp=p["far_bp"],
        fill=fill,
    )
