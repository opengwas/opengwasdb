"""PC Adjustment and one A1-signed r for tiles and on-demand rows (ADR 0013).

The covariates are an intercept and the kept PCs in sample order. At k = 0 the
intercept alone centres the dosages, giving Pearson r on complete genotypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

#: A residual whose squared length is this small a share of the centred dosages' has no
#: variance after PC Adjustment, to float error: r with it is undefined (build-inputs §6, 4).
FLAT: Final = 1e-12


@dataclass(frozen=True)
class Covariates:
    """The covariates regressed out, by name, and an orthonormal basis of them with the
    intercept, (samples, k + 1), in the sample order."""

    names: list[str]
    basis: npt.NDArray[np.float64]

    @property
    def k(self) -> int:
        return len(self.names)


def covariates(
    names: list[str], values: npt.NDArray[np.float64] | None, samples: int
) -> Covariates:
    """The intercept and the named PCs' `values` (samples, k), or the intercept alone."""
    columns = [np.ones(samples)]
    if values is not None:
        columns += [values[:, j] for j in range(values.shape[1])]
    basis, _ = np.linalg.qr(np.column_stack(columns))
    return Covariates(list(names), basis)


def residuals(
    dosages: npt.NDArray[np.float64], covariates: Covariates
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Each row of `dosages` (variants × samples) with the covariates regressed out, and which
    rows have no variance left: their residual is at most `FLAT` of their centred dosages'."""
    q = covariates.basis
    resid = dosages - (dosages @ q) @ q.T
    centred = dosages - dosages.mean(axis=1, keepdims=True)
    left = np.einsum("ij,ij->i", resid, resid)
    flat = left <= FLAT * np.einsum("ij,ij->i", centred, centred)
    return resid, flat


@dataclass(frozen=True)
class Block:
    """ALT dosages and their PC projections, residual norms and A1 signs (ADR 0013).

    Rows retain the sample order of the covariate basis. Two blocks passed to `r` must use
    the same samples and basis; a row-against-window call is a one-row slice of a block.
    """

    calls: npt.NDArray[np.float64]
    projection: npt.NDArray[np.float64]
    norm: npt.NDArray[np.float64]
    sign: npt.NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.calls)

    def __getitem__(self, rows: slice) -> Block:
        return Block(self.calls[rows], self.projection[rows], self.norm[rows], self.sign[rows])


def prepare(
    dosages: npt.NDArray[np.float64],
    covariates: Covariates,
    a1_is_alt: npt.NDArray[np.bool_],
) -> Block:
    """Project ALT dosages once; refuse the same FLAT residuals as the builder (spec §6).

    Keeping the residual calculation (rather than subtracting squared norms) avoids
    cancellation in the FLAT check. Only projections and norms survive it (ADR 0013).
    """
    projection = dosages @ covariates.basis
    resid, flat = residuals(dosages, covariates)
    if flat.any():
        raise ValueError(f"{int(flat.sum())} variants have no residual variance: not dropped")
    norm = np.sqrt(np.einsum("ij,ij->i", resid, resid))
    return Block(dosages, projection, norm, np.where(a1_is_alt, 1.0, -1.0))


def r(rows: Block, span: Block) -> npt.NDArray[np.float64]:
    """A1-signed r for a block against a span, including a single row against a window.

    For complete hard calls the float64 BLAS raw dot is exact: every partial sum is a
    whole number at most 4n. Only the (k + 1)-term correction and normalisation round.
    Dosage inputs use the same formula, without the integer exactness claim (ADR 0013).
    Quantise the result only through `opengwasdb.ld.format.REncoding`.
    """
    out = rows.calls @ span.calls.T
    out -= rows.projection @ span.projection.T
    out /= rows.norm[:, None]
    out /= span.norm[None, :]
    out *= rows.sign[:, None]
    out *= span.sign[None, :]
    return out
