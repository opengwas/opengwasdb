"""Writing an LD Store: the seam the builder writes through (spec §1–§7, §12; ogld#4).

`StoreWriter` takes Stored r as int16 codes the builder quantised with
`ogld.format.quantise`, chromosome by chromosome and the Near Band in row blocks, because
one chromosome's band can exceed 70 GB raw (spec §4). A store is not one until `finish`
writes its provenance and every validation rule passes; a reader refuses a directory
without provenance, so a half-written store cannot be read as a short one.
"""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt

from ogld.format import Provenance
from ogld.store.tables import Variants


class StoreWriter:
    """Writes one LD Store (spec §1)."""

    @classmethod
    def create(
        cls, location: str | os.PathLike[str], *, variants: Variants, near_bp: int, far_bp: int
    ) -> StoreWriter:
        """Start a store at a new `location`: write the Variant Set, in engine order, and
        its chromosome ranges (spec §2). Refuses a location that exists (ogld#4)."""
        raise NotImplementedError("ogld#4")

    def write_near(
        self, first_row: int, lengths: npt.NDArray[np.int64], codes: npt.NDArray[np.int16]
    ) -> None:
        """Append the Near Band rows `first_row`, `first_row + 1`, … (global rows, one
        chromosome, in order): `lengths[i]` codes for each, back to back (spec §4)."""
        raise NotImplementedError("ogld#4")

    def write_index(
        self,
        chrom: str,
        indptr: npt.NDArray[np.int64],
        partner: npt.NDArray[np.int64],
        codes: npt.NDArray[np.int16],
    ) -> None:
        """Write one chromosome's Proxy Index in CSR form over its rows, partners as
        global rows ascending within each row, both directions of every pair (spec §5)."""
        raise NotImplementedError("ogld#4")

    def finish(self, provenance: Provenance) -> Provenance:
        """Write the provenance, run every validation rule (spec §12), and return the
        provenance as written. Raises `ValidationFailed` naming each failing rule."""
        raise NotImplementedError("ogld#4")
