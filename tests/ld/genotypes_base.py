"""PC Adjustment: each variant's A1 dosages, the covariates regressed out, as a unit vector.

r(i, j) is then the dot product of two variants' vectors (spec §3). The covariates are an
intercept and the PCs the PC Check chose, in the sample order; at k = 0 the intercept alone,
which centres the dosages, so r is Pearson's r of the dosages, as `plink2 --r-unphased`
computes it on complete genotypes (build-inputs §7).

The residual is the dosages minus their projection on an orthonormal basis of the covariates
(a QR factorisation), as `numpy.linalg.lstsq` residuals are, to float error.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# The covariates and the residual are the production functions (one QR, one FLAT test), not a copy
# of them; what this file keeps of the frozen builder is `unit_vectors` below, the one-vector-
# per-variant arithmetic `prepare` and `r` replaced.
from opengwasdb.ld.genotypes.adjust import Covariates, covariates, residuals

__all__ = ["covariates", "unit_vectors"]


def unit_vectors(
    dosages: npt.NDArray[np.float64],
    covariates: Covariates,
    a1_is_alt: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Each variant's residual A1 dosages as a unit vector: a row product is r signed to A1.

    The dosages count ALT; A1's are 2 − ALT where A1 is REF, so its vector changes sign, and so
    does every r it is in (spec §3). A variant with no residual variance is a builder fault
    here: the Variant Set dropped them all before r is computed.
    """
    resid, flat = residuals(dosages, covariates)
    if flat.any():
        raise ValueError(f"{int(flat.sum())} variants have no residual variance: not dropped")
    resid /= np.sqrt(np.einsum("ij,ij->i", resid, resid))[:, None]
    resid[~a1_is_alt] *= -1
    return resid
