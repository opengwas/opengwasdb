"""LD matrices: for a list of ALIDs, and for a region (spec §8; ogld#4)."""

from __future__ import annotations

from collections.abc import Sequence

from ogld.query.results import LDMatrix
from ogld.store import StoreReader


def matrix(reader: StoreReader, alids: Sequence[str], *, fill: float | None) -> LDMatrix:
    """The signed-r matrix of the present `alids`, in request order (spec §8; ogld#4).

    Unstored pairs are NaN, replaced by `fill` only when the caller passed one. Never a
    refusal: a pair more than far_bp apart is a NaN whose kind is Not Measured.
    """
    raise NotImplementedError("ogld#4")


def region(
    reader: StoreReader, chrom: str, centre: int, radius: int, *, fill: float | None
) -> LDMatrix:
    """The signed-r matrix of every variant with |position − centre| ≤ radius on `chrom`,
    in genomic order (spec §8; ogld#4). Refused unless 2 × radius ≤ far_bp."""
    raise NotImplementedError("ogld#4")
