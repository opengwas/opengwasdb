"""How r is stored, and which pairs beyond the Near Band an LD Store keeps (spec §3).

The int16 code, the Far Threshold and the far code floor are the three numbers every
component agrees on: the builder (ogld#5) quantises with them and keeps a 3–10 Mb pair by
them, the reader (ogld#4) decodes with them, and validation recomputes them. They live
here, once, so no component can round or threshold differently from another.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import numpy.typing as npt

#: r̂ = code / R_SCALE. magenpy's int16 scale: symmetric, so negating a code never
#: overflows and r = ±1 is exact (spec §3).
R_SCALE: Final = 32767

#: Never a correlation. Every r array's fill value, so a chunk that was never written
#: cannot read as r = 0 (spec §3, §6).
RESERVED_CODE: Final = -32768

#: Half a quantisation step: |r̂ − r| never exceeds it (spec §3).
QUANTISATION_ERROR: Final = 1 / (2 * R_SCALE)

#: The χ²₁ upper quantile at p = 5×10⁻⁶ in t_far = max(0.001, q / (n − k − 1)) (ADR 0001).
T_FAR_Q: Final = 20.84

#: t_far's floor: MR's default r² threshold (ADR 0001).
T_FAR_MIN: Final = 0.001

#: The builder's distances (ADR 0001): the Near Band is complete up to NEAR_BP, and the
#: Proxy Index reaches FAR_BP. A store records the distances it measured (spec §7) and
#: readers use those, never these.
NEAR_BP: Final = 3_000_000
FAR_BP: Final = 10_000_000

# A dot product of standardised vectors can land a few ulps past ±1. Further than this is
# not rounding error but a builder fault, and clamping it would store a plausible r = ±1.
_R_SLACK: Final = 1e-6


def quantise(r: npt.ArrayLike) -> npt.NDArray[np.int16]:
    """The int16 code of each r: clamp(rint(r × 32767), −32767, +32767) (spec §3).

    `numpy.rint` rounds half to even, as the spec requires. A value that is not finite,
    or whose magnitude exceeds 1 by more than float error, raises: r is undefined only
    for a zero-variance variant, which the builder drops (spec §3), so such a value means
    the computation went wrong, and a clamped code would hide it.
    """
    values = np.asarray(r, dtype=np.float64)
    bad = ~np.isfinite(values) | (np.abs(values) > 1 + _R_SLACK)
    if bad.any():
        raise ValueError(
            f"not a correlation, so it has no code: {values[bad].ravel()[:5].tolist()} (spec §3)"
        )
    codes = np.clip(np.rint(values * R_SCALE), -R_SCALE, R_SCALE)
    return codes.astype(np.int16)


def dequantise(codes: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The Stored r of each code: code / 32767 (spec §3).

    The reserved code −32768 raises. It is what an unwritten chunk reads as, so decoding
    it to a number is exactly the silent zero the reserved code exists to prevent.
    """
    values = np.asarray(codes)
    if (values == RESERVED_CODE).any():
        raise ValueError(
            f"code {RESERVED_CODE} is reserved: an unwritten chunk, never a correlation (spec §3)"
        )
    return values.astype(np.float64) / R_SCALE


def t_far(n: int, k: int) -> float:
    """The Far Threshold: max(0.001, q / (n − k − 1)), in float64 (ADR 0001, spec §3).

    `n` is the samples the LD was computed from and `k` the covariates adjusted for,
    counted by name in provenance (spec §7). A store records this value and every reader
    uses the record; the builder and validation are the only callers. A panel with
    n − k − 1 ≤ q would have t_far ≥ 1, which no pair can reach, so it raises (§12).
    """
    if k < 0:
        raise ValueError(f"k counts covariates, so it cannot be negative: n={n}, k={k}")
    dof = n - k - 1
    if dof <= T_FAR_Q:
        raise ValueError(
            f"n − k − 1 = {dof} must exceed q = {T_FAR_Q} for t_far < 1: n={n}, k={k} (spec §12)"
        )
    return max(T_FAR_MIN, T_FAR_Q / dof)


def far_code_floor(t_far: float) -> int:
    """The smallest code c ≥ 0 with (c / 32767)² ≥ t_far, in float64 (spec §3).

    A pair more than `near_bp` apart is in the Proxy Index exactly when |code| reaches
    it, which is the same as r̂² ≥ t_far. Computed from an estimate and then stepped to the
    exact boundary, because the float64 comparison, not the real-number one, is the rule.
    """
    if not 0.0 <= t_far <= 1.0:
        raise ValueError(f"t_far must be an r² in [0, 1]: {t_far}")
    code = max(0, math.floor(math.sqrt(t_far) * R_SCALE) - 2)
    while (code / R_SCALE) ** 2 < t_far:
        code += 1
    return code


def row_reach(position: npt.ArrayLike, distance: int) -> npt.NDArray[np.int64]:
    """For each row l of one chromosome, the first row more than `distance` past it: rows l + 1
    … reach − 1 are its partners within `distance` (spec §4, §5). The one rule for where a
    band, near_bp or far_bp, ends; `position` is ascending, in row order."""
    at = np.asarray(position, dtype=np.int64)
    reach: npt.NDArray[np.int64] = np.searchsorted(at, at + distance, side="right").astype(np.int64)
    return reach


def near_row_lengths(position: npt.ArrayLike, near_bp: int) -> npt.NDArray[np.int64]:
    """The length of each Near Band row of one chromosome: len_l = |{m > l : p_m − p_l ≤ near_bp}|
    (spec §4). `position` is the chromosome's positions in row order, ascending."""
    reach = row_reach(position, near_bp)
    return (reach - np.arange(len(reach)) - 1).astype(np.int64)
