"""PC Adjustment: each variant's A1 dosages, the covariates regressed out, as a unit vector.

r(i, j) is then the dot product of two variants' vectors (ld-store §3). The covariates are an
intercept and the PCs the PC Check chose, in the sample order; at k = 0 the intercept alone,
which centres the dosages, so r is Pearson's r of the dosages, as `plink2 --r-unphased`
computes it on complete genotypes (build-inputs §7).

The residual is the dosages minus their projection on an orthonormal basis of the covariates
(a QR factorisation), as `numpy.linalg.lstsq` residuals are, to float error.
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


def unit_vectors(
    dosages: npt.NDArray[np.float64],
    covariates: Covariates,
    a1_is_alt: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Each variant's residual A1 dosages as a unit vector: a row product is r signed to A1.

    The dosages count ALT; A1's are 2 − ALT where A1 is REF, so its vector changes sign, and so
    does every r it is in (ld-store §3). A variant with no residual variance is a builder fault
    here: the Variant Set dropped them all before r is computed.
    """
    resid, flat = residuals(dosages, covariates)
    if flat.any():
        raise ValueError(f"{int(flat.sum())} variants have no residual variance: not dropped")
    resid /= np.sqrt(np.einsum("ij,ij->i", resid, resid))[:, None]
    resid[~a1_is_alt] *= -1
    return resid
