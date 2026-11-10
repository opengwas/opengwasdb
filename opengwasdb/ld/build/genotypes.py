"""Reading genotypes from a pgen: ALT dosages of the kept samples, in the sample order.

A build reads dosages, not hard calls: for a pgen of hard calls they are the ALT counts 0, 1
and 2, and for one that carries dosages they are those, as plink2 uses them ("Dosages are used
when present", build-inputs §7). Phase is ignored. A missing genotype reads as −9 and is the
caller's to refuse: a build never imputes one (ADR 0002, decision 2).

pgenlib is imported here, inside the read path only (ADR 0002, decision 10).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from ogld.build.inputs import Fileset, fileset, read_keep
from ogld.format import InputError

T = TypeVar("T")


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
        pgenlib = import_pgenlib()
        self._reader = self._read(
            lambda: pgenlib.PgenReader(str(fileset.pgen).encode(), sample_subset=fileset.subset)
        )
        self.variant_count = int(self._reader.get_variant_ct())
        held = int(self._reader.get_raw_sample_ct())
        if held != fileset.samples:
            self._reader.close()
            raise InputError(
                f"panel {fileset.panel}: {fileset.psam.name} lists {fileset.samples} samples, "
                f"but {fileset.pgen.name} holds {held}: a psam must name every sample of its "
                "pgen, in order (build-inputs §2)"
            )

    def _read(self, call: Callable[[], T]) -> T:
        """`call`, with pgenlib's own error (a truncated or corrupt pgen) a refusal naming it."""
        try:
            return call()
        except RuntimeError as broken:
            raise InputError(
                f"panel {self._fileset.panel}: {self._fileset.pgen.name} cannot be read as a "
                f"pgen: {str(broken).strip()}"
            ) from broken

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
            self._read(lambda: self._reader.read_dosages_range(first, stop, out))
        return self._ordered(out)

    def rows(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """The pgen rows `rows`, in the order given."""
        wanted, back = np.unique(rows, return_inverse=True)
        out = np.empty((len(wanted), len(self._fileset.subset)), dtype=np.float64)
        if len(wanted):
            self._read(lambda: self._reader.read_dosages_list(wanted.astype(np.uint32), out))
        return self._ordered(out)[back]


def read_alt_counts(
    pgen: Path, psam: Path, keep: Path, rows: npt.NDArray[np.int64], *, panel: str = ""
) -> npt.NDArray[np.float64]:
    """The ALT dosages of pgen `rows` (0-based) for the keep file's samples, in the psam's order:
    [rows, samples]. The build's own readers, keep file and fileset, so there is one of each; a
    missing call is refused naming the variant and the panel (`panel`, else the pgen's name).
    """
    import_pgenlib()
    name = panel or pgen.name
    kept = read_keep(keep, name)
    found = fileset(pgen, kept, None, name)
    if found.psam != psam:
        raise InputError(f"panel {name}: {psam} is not the psam beside {pgen}")
    with Genotypes(found) as genotypes:
        dosages = genotypes.rows(np.asarray(rows, dtype=np.int64))
    holed = np.flatnonzero((dosages < 0).any(axis=1))
    if len(holed):
        raise InputError(
            f"panel {name}: missing call in pgen row {int(rows[holed[0]])} of {pgen.name}: "
            "a build refuses incomplete genotypes rather than impute them"
        )
    return dosages
