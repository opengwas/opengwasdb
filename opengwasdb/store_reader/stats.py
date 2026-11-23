"""The p-value a stored z implies, and the |z| a p-value threshold cuts at.

A Store Release stores z; its two-sided p is ``erfc(|z| / sqrt(2))``. Three rules
follow from that one, and all three live here, without scipy (#464):

- `p_value_from_z`, the scalar p;
- `log10_p_two_sided`, its log10 over an array, which result resolution reports
  as `log10_p`. It stays finite where p itself underflows to 0.0 (|z| beyond
  about 37.5), as the FADS1/FADS2 window's |z| = 47.8 does (issue #104);
- `z_critical`, the |z| a threshold cuts at. The Top-Hit Index build, the
  Ragged scan fallback and the validator all cut there and nowhere else.

They replace scipy's ``norm.logsf`` and ``erfcinv``. `log10_p_two_sided` is
within 1e-15 relative error of the exact value everywhere, and agrees with the
scipy form to 1e-12 wherever that form is itself that accurate: for |z| below
about 1.4e-4 scipy's p is ``0.5 - tiny`` rounded, and its log10 is off by up to
1.3e-4 relative, where this one is not. `z_critical` agrees with scipy's to a
few ulps, with no float32 |z| between the two at any Top-Hit Index threshold
(`tests/test_store_reader_stats.py`).
"""

from __future__ import annotations

import math
import statistics

import numpy as np

_LN10 = math.log(10.0)
_SQRT_HALF = math.sqrt(0.5)
_LOG_SQRT_PI = 0.5 * math.log(math.pi)

#: Below this argument `math.erf` is the accurate route: p is close to 1, and
#: ``log1p(-erf(y))`` keeps the digits ``log(erfc(y))`` would round away.
_NEAR_ONE = 0.5

#: From here on ``erfc(y)`` nears float64's subnormal range (it underflows past
#: about 26.5), so the asymptotic series takes over; erfc(26) is 5.7e-296, still
#: a normal float with its full precision.
_TAIL = 26.0

#: Terms of the asymptotic series kept. At y = 26 the eighth is below 1e-20 of
#: the first, well past float64's precision.
_TAIL_TERMS = 8


def p_value_from_z(z: float) -> float:
    """Return the two-sided normal p-value implied by a Z score."""

    return math.erfc(abs(z) / math.sqrt(2.0))


def _log_erfc_tail(y: float) -> float:
    """``log(erfc(y))`` for large y, by the asymptotic expansion

    ``erfc(y) = exp(-y²) / (y sqrt(pi)) * (1 - 1/(2y²) + 3/(2y²)² - 15/(2y²)³ + ...)``.
    """
    step = 1.0 / (2.0 * y * y)
    term = series = 1.0
    for k in range(1, _TAIL_TERMS):
        term *= -(2 * k - 1) * step
        series += term
    return -y * y - math.log(y) - _LOG_SQRT_PI + math.log(series)


def _log_erfc(y: float) -> float:
    """``log(erfc(y))`` for y >= 0 (or NaN), accurate across the whole range."""
    if y == 0.0:
        return 0.0  # p = 1, and +0.0 rather than log1p(-0.0)'s -0.0
    if y < _NEAR_ONE:
        return math.log1p(-math.erf(y))
    if y < _TAIL:
        return math.log(math.erfc(y))
    return _log_erfc_tail(y)


def log10_p_two_sided(z: np.ndarray) -> np.ndarray:
    """Vectorised log10 of `p_value_from_z`, stable past the point the plain
    erfc-based p itself underflows to 0.0 in float64 (|z| >~ 38) -- e.g. the
    FADS1/FADS2 window's |z| = 47.8 (issue #104). Computed in log space rather
    than as ``log(p_value_from_z(z))``, which would already have lost the value
    to underflow before the log is taken. NaN stays NaN; an infinite |z| is
    ``-inf``.
    """
    scaled = np.abs(np.asarray(z, dtype="float64")) * _SQRT_HALF
    logs = [_log_erfc(float(y)) for y in scaled.ravel()]
    result: np.ndarray = np.asarray(logs, dtype="float64").reshape(scaled.shape) / _LN10
    return result


def z_critical(threshold: float) -> float:
    """|z| cutoff equivalent to two-sided p <= threshold.

    The stored p-value is ``p = erfc(|z| / sqrt(2))`` (`p_value_from_z`), so
    ``p <= threshold`` is exactly ``|z| >= -Φ⁻¹(threshold / 2)``. Thresholding on
    ``|z|`` lets the hot path avoid computing a p-value for every cell. The index
    build, the Ragged scan fallback and the validator all cut here and nowhere
    else.
    """

    return -statistics.NormalDist().inv_cdf(threshold / 2.0)
