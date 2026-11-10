"""The unadjusted (k = 0) Near Band, from a pgen: the walking skeleton's thin path (ogld#12).

pgen → ALT counts of the kept samples → standardised A1 dosages → r → int16 codes, one
chromosome at a time. It is not `ogld.build`: no filters, no PC Check, no regression, no Proxy
Index. ogld#5 builds those around it. pgenlib is imported inside `read_alt_counts` only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ogld.format import InputError, near_row_lengths, quantise

#: Rows of the Near Band computed per matrix product: bounds the work array.
BLOCK_ROWS = 512


@dataclass(frozen=True)
class NearBand:
    """One chromosome's Near Band as `StoreWriter.write_near` takes it (spec §4).

    `indptr` has one more entry than there are rows; row l's codes are
    `codes[indptr[l]:indptr[l + 1]]`, its partners l + 1, l + 2, …
    """

    indptr: npt.NDArray[np.int64]
    codes: npt.NDArray[np.int16]

    @property
    def lengths(self) -> npt.NDArray[np.int64]:
        return np.diff(self.indptr)


def _import_pgenlib() -> Any:
    """pgenlib, or a refusal naming the extra that installs it (ADR 0002, decision 10)."""
    try:
        import pgenlib
    except ImportError as missing:
        raise InputError(
            "reading a pgen needs pgenlib, which ogld installs with the build extra: "
            "pip install 'ogld[build]'"
        ) from missing
    return pgenlib


def _refuse_repeats(listed: list[str]) -> None:
    """A duplicate IID in the keep file is refused, naming it: n must not shrink silently."""
    repeated = sorted({iid for iid in listed if listed.count(iid) > 1})
    if repeated:
        raise InputError(
            f"the keep file has a duplicate IID: {', '.join(repeated)} (build-inputs §4)"
        )


def _kept_columns(psam: Path, keep: Path) -> npt.NDArray[np.int64]:
    """The psam columns of the kept samples, in psam order."""
    iids = [line.split("\t")[0] for line in psam.read_text().splitlines()[1:]]
    listed = [line.strip() for line in keep.read_text().splitlines()[1:] if line.strip()]
    _refuse_repeats(listed)
    kept = set(listed)
    columns = np.array([i for i, iid in enumerate(iids) if iid in kept], dtype=np.int64)
    if len(columns) != len(kept):
        raise InputError(f"the keep file names samples that {psam} does not have")
    return columns


def read_alt_counts(
    pgen: Path, psam: Path, keep: Path, rows: npt.NDArray[np.int64]
) -> npt.NDArray[np.int8]:
    """The ALT allele count (0, 1, 2) of each of `rows`, for the kept samples: [rows, samples].

    `rows` are 0-based pgen variant indexes. The genotypes must be complete: a missing call
    is refused naming the variant and the panel, never imputed (ADR 0002).
    """
    pgenlib = _import_pgenlib()
    columns = _kept_columns(psam, keep)
    reader = pgenlib.PgenReader(str(pgen).encode())
    out = np.empty((len(rows), reader.get_raw_sample_ct()), dtype=np.int8)
    reader.read_list(rows.astype(np.uint32), out)
    reader.close()
    out = out[:, columns]
    missing_calls = np.flatnonzero((out < 0).any(axis=1))
    if len(missing_calls):
        row = int(rows[missing_calls[0]])
        raise InputError(
            f"missing call in variant {_variant_name(pgen, row)} of panel {pgen}: "
            "a build refuses incomplete genotypes rather than impute them"
        )
    return out


def _variant_name(pgen: Path, row: int) -> str:
    """The pvar ID of variant `row` when the pvar sits beside the pgen, else its index."""
    pvar = pgen.with_suffix(".pvar")
    if not pvar.exists():
        return f"#{row}"
    lines = [line for line in pvar.read_text().splitlines() if not line.startswith("##")]
    return lines[1 + row].split("\t")[2]


def unadjusted_near_band(
    alt_counts: npt.NDArray[np.int8],
    a1_is_alt: npt.NDArray[np.bool_],
    positions: npt.NDArray[np.int64],
    near_bp: int,
    *,
    names: Sequence[str] | None = None,
    panel: str = "the panel",
) -> NearBand:
    """The Near Band of one chromosome's variants at k = 0 (spec §3, §4).

    `alt_counts` is [variants, samples] in genomic order and `a1_is_alt` says which allele is
    each variant's A1, so r is the correlation of the A1 dosages. `names` and `panel` say which
    variant and panel a refusal is about.
    """
    z = alt_counts.astype(np.float64)
    z -= z.mean(axis=1, keepdims=True)
    norm = np.sqrt((z * z).sum(axis=1, keepdims=True))
    flat = np.flatnonzero(norm[:, 0] == 0)
    if len(flat):
        which = names[flat[0]] if names is not None else f"#{flat[0]}"
        raise InputError(
            f"variant {which} of panel {panel} has no genotype variance, so no r (spec §3): "
            "the builder drops such a variant before it computes r"
        )
    z /= norm
    # r is the correlation of ALT dosages; A1 dosages are 2 − ALT where A1 is REF, so the
    # standardised column changes sign and so does every r the variant is in (spec §3).
    z[~a1_is_alt] *= -1
    n = len(positions)
    lengths = near_row_lengths(positions, near_bp)
    stop = np.arange(n) + 1 + lengths
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    codes = np.empty(int(indptr[-1]), dtype=np.int16)
    for first in range(0, n, BLOCK_ROWS):
        last = min(first + BLOCK_ROWS, n)
        reach = int(stop[first:last].max())
        r = z[first:last] @ z[first:reach].T
        for row in range(first, last):
            lo = row + 1
            hi = int(stop[row])
            codes[indptr[row] : indptr[row + 1]] = quantise(r[row - first, lo - first : hi - first])
    return NearBand(indptr, codes)
