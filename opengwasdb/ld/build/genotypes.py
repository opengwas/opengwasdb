"""Reading genotypes from a pgen: ALT dosages of the kept samples, in the sample order.

A build reads dosages, not hard calls: for a pgen of hard calls they are the ALT counts 0, 1
and 2, and for one that carries dosages they are those, as plink2 uses them ("Dosages are used
when present", build-inputs §7). Phase is ignored. A missing genotype reads as −9 and is the
caller's to refuse: a build never imputes one (ADR 0002, decision 2).

pgenlib is imported here, inside the read path only (ADR 0002, decision 10).
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt

from ogld.build.inputs import Fileset
from ogld.format import InputError


def import_pgenlib() -> Any:
    """pgenlib, or a refusal naming the extra that installs it (ADR 0002, decision 10)."""
    try:
        import pgenlib
    except ImportError as missing:
        raise InputError(
            "reading a pgen needs pgenlib, which ogld installs with the build extra: "
            "pip install 'ogld[build]'"
        ) from missing
    return pgenlib


class Genotypes:
    """One fileset's genotypes, open for reading: [variants, samples] float64 ALT dosages."""

    def __init__(self, fileset: Fileset) -> None:
        self._fileset = fileset
        self._reader = import_pgenlib().PgenReader(
            str(fileset.pgen).encode(), sample_subset=fileset.subset
        )
        self.variant_count = int(self._reader.get_variant_ct())

    def __enter__(self) -> Genotypes:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._reader.close()

    def _ordered(self, out: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        order = self._fileset.order
        return out if order is None else np.ascontiguousarray(out[:, order])

    def range(self, first: int, stop: int) -> npt.NDArray[np.float64]:
        """The pgen rows `first` … `stop − 1`."""
        out = np.empty((stop - first, len(self._fileset.subset)), dtype=np.float64)
        if stop > first:
            self._reader.read_dosages_range(first, stop, out)
        return self._ordered(out)

    def rows(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """The pgen rows `rows`, in the order given."""
        wanted, back = np.unique(rows, return_inverse=True)
        out = np.empty((len(wanted), len(self._fileset.subset)), dtype=np.float64)
        if len(wanted):
            self._reader.read_dosages_list(wanted.astype(np.uint32), out)
        return self._ordered(out)[back]
