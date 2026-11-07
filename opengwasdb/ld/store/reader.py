"""Reading an LD Store: the seam every query is built on (spec §1–§6, §10; ogld#4).

`StoreReader` is the store layer's whole read interface. It hides Zarr, the chunk and
shard shapes, chromosome-local rows and the int16 codes: callers pass and receive global
Variant Set rows and Stored r as float64. Every unstored pair it reports is NaN, never a
number, and the meaning of a NaN is the format's to say (`ogld.format.answer_kinds`).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ogld.format import Provenance
from ogld.store.tables import Partners, Variants


class StoreReader:
    """An opened LD Store, local or over HTTP (spec §1).

    Opening reads the provenance and the chromosome ranges and refuses a store whose
    `format_version` or `encoding` this package does not implement (spec §13), naming
    it, before anything is decoded.
    """

    @classmethod
    def open(cls, location: str | os.PathLike[str]) -> StoreReader:
        """Open the LD Store at a local path or an HTTP(S) URL (ogld#4)."""
        raise NotImplementedError("ogld#4")

    @property
    def provenance(self) -> Provenance:
        """The root group's `ld_store` attribute, as recorded (spec §7)."""
        raise NotImplementedError("ogld#4")

    def rows_of(self, alids: Sequence[str]) -> npt.NDArray[np.int64]:
        """The Variant Set row of each ALID, in order; −1 where the ALID is Absent (spec §2).

        A malformed ALID is Absent too: it names no variant in the store.
        """
        raise NotImplementedError("ogld#4")

    def rows_in(self, chrom: str, start: int, stop: int) -> npt.NDArray[np.int64]:
        """The rows on `chrom` with start ≤ position ≤ stop, ascending (spec §2)."""
        raise NotImplementedError("ogld#4")

    def variants(self, rows: npt.NDArray[np.int64]) -> Variants:
        """The Variant Set's columns at `rows`, in the order given (spec §2)."""
        raise NotImplementedError("ogld#4")

    def pair_r(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """The square matrix of Stored r among `rows`, in the order given (spec §8, §10).

        Pairs within near_bp come from the Near Band, farther ones from the Proxy Index.
        The diagonal is 1. Every pair the store does not hold, the Bound and Not Measured
        alike, is NaN.
        """
        raise NotImplementedError("ogld#4")

    def index(self, rows: npt.NDArray[np.int64]) -> Partners:
        """Each row's Proxy Index entries: every partner within far_bp, either side, with
        r̂² ≥ t_far, and its Stored r (spec §5). One sparse row per variant (§10)."""
        raise NotImplementedError("ogld#4")

    def band(self, rows: npt.NDArray[np.int64]) -> Partners:
        """Each row's Near Band partners on both sides, with their Stored r (spec §4, §10).

        The downstream half is the row; the upstream half is a column, one element in
        each row of the preceding near_bp. It is the slow read the Proxy Index exists to
        avoid, used only by `proxies` at an r2 below t_far.
        """
        raise NotImplementedError("ogld#4")
