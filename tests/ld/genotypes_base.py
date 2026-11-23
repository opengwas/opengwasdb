"""An independent oracle for PC Adjustment: least squares on the covariate design, not QR.

Production (`opengwasdb.ld.genotypes.adjust`) regresses each variant's dosages on an orthonormal
basis of the covariates, from a QR factorisation, and projects. This oracle shares none of that: it
builds the design itself (an intercept column and the PCs, in the sample order), solves the least
squares problem with `numpy.linalg.lstsq` for every variant at once, takes the residuals, and scales
each variant's residual to a unit vector signed to A1. r(i, j) is then the dot product of two such
vectors (ld-store §3); at k = 0 the intercept alone centres the dosages and r is Pearson's.

The two agree to float error: measured on the Test Panel's 771 samples, the largest difference in r
is 3.4e-13 at k = 0 and 4.5e-13 at k = 10. The tests compare them to 1e-11, some twenty times that
and some three thousand times below the finest quantisation step (1/32767).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Agreement between this oracle's r and production's, absolute; see the module docstring.
R_TOLERANCE = 1e-11


def covariates(
    names: list[str], values: npt.NDArray[np.float64] | None, samples: int
) -> npt.NDArray[np.float64]:
    """The design matrix, (samples, k + 1): a column of ones, then each PC in `values`."""
    design = np.ones((samples, 1))
    if values is not None and values.shape[1]:
        design = np.hstack([design, np.asarray(values, dtype=np.float64)])
    assert design.shape[1] == len(names) + 1, "one column per named PC, and the intercept"
    return design


def unit_vectors(
    dosages: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    a1_is_alt: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Each variant's A1 dosages, the design regressed out, as a unit row vector.

    The dosages count ALT; A1's are 2 − ALT where A1 is REF, so that variant's vector changes
    sign, and so does every r it is in. A variant with no variance left is a builder fault here.
    """
    fitted, *_ = np.linalg.lstsq(design, dosages.T, rcond=None)
    residual = (dosages.T - design @ fitted).T
    centred = dosages - dosages.mean(axis=1, keepdims=True)
    left = (residual**2).sum(axis=1)
    flat = left <= 1e-12 * (centred**2).sum(axis=1)
    if flat.any():
        raise ValueError(f"{int(flat.sum())} variants have no residual variance: not dropped")
    unit = residual / np.sqrt(left)[:, None]
    unit[~a1_is_alt] *= -1
    return unit
