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
from typing import Any, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from opengwasdb.ld.format import InputError

T = TypeVar("T")
D = TypeVar("D", np.float64, np.int8)


class PgenFileset(Protocol):
    """The read-only input description; builders supply their Fileset without an upward import."""

    @property
    def pgen(self) -> Path: ...

    @property
    def psam(self) -> Path: ...

    @property
    def subset(self) -> npt.NDArray[np.uint32]: ...

    @property
    def order(self) -> npt.NDArray[np.int64] | None: ...

    @property
    def samples(self) -> int: ...

    @property
    def panel(self) -> str: ...


def import_pgenlib() -> Any:
    """pgenlib, or a refusal naming the extra that installs it (ADR 0002, decision 10)."""
    try:
        import pgenlib
    except ImportError as missing:
        raise InputError(
            "reading a pgen needs pgenlib, which the ld-build extra installs: "
            "pip install 'opengwasdb[ld-build]'"
        ) from missing
    return pgenlib


def _requested(
    ordered: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
    back: npt.NDArray[np.int64] | None,
) -> None:
    if back is None:
        if ordered is not out:
            out[:] = ordered
    elif ordered is out:
        changed = np.flatnonzero(back != np.arange(len(back)))
        out[changed] = out[back[changed]]
    else:
        out[:] = ordered[back]


class Genotypes:
    """One fileset's genotypes, open for reading: [variants, samples] float64 ALT dosages."""

    def __init__(self, fileset: PgenFileset) -> None:
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
        self.close()

    def close(self) -> None:
        """Release the native reader; a caller that keeps a handle for reuse closes it here."""
        self._reader.close()

    def _ordered(self, out: npt.NDArray[D]) -> npt.NDArray[D]:
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
        out = np.empty((len(rows), len(self._fileset.subset)), dtype=np.float64)
        self.rows_into(rows, out)
        return out

    def rows_into(self, rows: npt.NDArray[np.int64], out: npt.NDArray[np.float64]) -> None:
        """Decode into a caller-owned contiguous buffer, including a disjoint thread's slice.

        Genomic ALID order differs from physical pgen order at some tied positions. Reorder
        only those rows in place, not a full dosage matrix (#91: 214 ms proxy read/copy).
        Each caller/thread owns its Genotypes handle; no native reader is shared.
        """
        if out.shape != (len(rows), len(self._fileset.subset)) or not out.flags.c_contiguous:
            raise ValueError(
                "pgen output must be contiguous, with one row per variant and kept sample"
            )
        ascending = len(rows) < 2 or bool(np.all(np.diff(rows) > 0))
        wanted, back = (rows, None) if ascending else np.unique(rows, return_inverse=True)
        direct = len(wanted) == len(rows) and self._fileset.order is None
        buffer = out if direct else np.empty((len(wanted), out.shape[1]), dtype=np.float64)
        if len(wanted):
            self._read(lambda: self._reader.read_dosages_list(wanted.astype(np.uint32), buffer))
        _requested(self._ordered(buffer), out, back)

    def range_int8(self, first: int, stop: int) -> npt.NDArray[np.int8]:
        """Hard calls only, with missing calls still −9 for the caller's named refusal.

        This path deliberately does not substitute for float64 dosage reads (ADR 0013).
        """
        out = np.empty((stop - first, len(self._fileset.subset)), dtype=np.int8)
        if stop > first:
            self._read(lambda: self._reader.read_range(first, stop, out))
        return self._ordered(out)

    def rows_int8(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.int8]:
        """Hard calls at the requested rows, retaining duplicates and sample order."""
        wanted, back = np.unique(rows, return_inverse=True)
        out = np.empty((len(wanted), len(self._fileset.subset)), dtype=np.int8)
        if len(wanted):
            self._read(lambda: self._reader.read_list(wanted.astype(np.uint32), out))
        return self._ordered(out)[back]
