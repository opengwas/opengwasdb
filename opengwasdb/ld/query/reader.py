"""The reader interface the four queries name (#91).

`ogld.store.StoreReader` and `ogld.genotypes.GenotypeReader` both satisfy it structurally: one
reads materialised arrays, the other computes the same answers from pinned pgens. The queries
depend on this seam only, so neither reader has to be the other's base class (fix round 1,
Standards: composition, not inheritance).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ogld.format import Codes, Provenance, REncoding
from ogld.store.tables import Variants


@runtime_checkable
class Reader(Protocol):
    """What a query reads: the store's facts, its variants, its pair answers and its rows.

    `_query_workers` and `_set_query_workers` are the component pool's size; `clump` reads it
    from here so no query signature carries a thread count (reader of either kind holds it).
    """

    _query_workers: int

    @property
    def location(self) -> str: ...

    @property
    def provenance(self) -> Provenance: ...

    @property
    def encoding(self) -> REncoding: ...

    def close(self) -> None: ...

    def request(self) -> AbstractContextManager[object]:
        """What one public query holds for its duration: the genotype reader's BLAS bound, and
        nothing for an LD Store, whose reads decode stored codes."""
        ...

    def _set_query_workers(self, requested: int) -> None: ...

    def rows_of(self, alids: Sequence[str]) -> npt.NDArray[np.int64]: ...

    def rows_in(self, chrom: str, start: int, stop: int) -> npt.NDArray[np.int64]: ...

    def chromosome_range(self, chrom: str) -> tuple[int, int]: ...

    def alids_of(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]: ...

    def loci_of(
        self, rows: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.int64]]: ...

    def variants(self, rows: npt.NDArray[np.int64]) -> Variants: ...

    def pair_r(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]: ...

    def region_r(self, start: int, stop: int) -> npt.NDArray[np.float64]: ...

    def require_partner_rows(self) -> float: ...

    def partner_rows(
        self, chrom: str, local_rows: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], Codes]: ...

    def clump_rows(
        self, chrom: str, rows: npt.NDArray[np.int64], window: float
    ) -> AbstractContextManager[Callable[[int], tuple[npt.NDArray[np.int32], Codes]]]: ...
