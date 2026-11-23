"""The statistics a stored z and SE imply, and the |z| a p-value threshold cuts at.

A Store Release stores z and SE; its beta is ``z * se`` and its two-sided p is
``erfc(|z| / sqrt(2))``. The rules that follow live here, without scipy (#464):

- `association_statistics`, the one derivation of `beta`, `pval` and `log10_p`
  from a result's z and SE arrays (ADR 0104). Every query record and every
  resolved row carries its output; nothing else in the package derives them;
- `p_value_from_z`, the scalar p;
- `log10_p_two_sided`, its log10 over an array, which result resolution reports
  as `log10_p`. It stays finite and exact where p itself is lost in float64 --
  subnormal from |z| = 37.5, 0.0 from |z| = 38.5 -- as at the FADS1/FADS2
  window's |z| = 47.8 (issue #104);
- `z_critical`, the |z| a threshold cuts at. The Top-Hit Index build, the
  Ragged scan fallback and the validator all cut there and nowhere else.

They replace scipy's ``norm.logsf`` and ``erfcinv``. Against a 50-digit
reference on 809 points across -40 <= z <= 40, `log10_p_two_sided`'s largest
relative error is 3.9e-16 (|z| = 36.35). It agrees with the scipy form to 1e-12
wherever that form is itself that accurate: for |z| below about 1e-3 scipy's p
is ``0.5 - tiny`` rounded, and its log10 is off by up to 1.3e-4 relative at
|z| = 1e-12, where this one is not. `z_critical` agrees with scipy's to a
few ulps, with no float32 |z| between the two at any Top-Hit Index threshold
(`tests/test_store_reader_stats.py`).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

_LN10 = math.log(10.0)
_SQRT_HALF = math.sqrt(0.5)
_LOG_SQRT_PI = 0.5 * math.log(math.pi)

#: Below this argument `math.erf` is the accurate route: p is close to 1, and
#: ``log1p(-erf(y))`` keeps the digits ``log(erfc(y))`` would round away.
_NEAR_ONE = 0.5

#: From here on ``erfc(y)`` nears float64's subnormal range (it goes subnormal at
#: y = 26.54, |z| = 37.5), so the asymptotic series takes over; erfc(26) is
#: 5.7e-296, still a normal float with its full precision.
_TAIL = 26.0

#: Terms of the asymptotic series kept. At y = 26 the eighth is 1.6e-17 of the
#: first and the first one left out 1.8e-19, below float64's precision.
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
    """Vectorised log10 of `p_value_from_z`, stable where the plain erfc-based p
    itself is lost: it goes subnormal at |z| = 37.5 and is 0.0 from |z| = 38.5 in
    float64, and the FADS1/FADS2 window's |z| = 47.8 (issue #104) is past both.
    Computed in log space rather than as ``log(p_value_from_z(z))``, which would
    have lost the value to underflow before the log is taken. NaN stays NaN; an
    infinite |z| is ``-inf``.

    Evaluated once per distinct |z|: a stored z is quantised -- a fixed-point
    plane holds at most 65,536 codes, a `float16` one 65,536 values -- so a
    result of millions of associations has a few thousand distinct |z|, and this
    is as fast as scipy's vectorised form on them (2M quantised z: 0.11 s
    against 0.17 s). A result of millions of distinct unquantised z, which no
    Store holds, takes about four times scipy's.
    """
    return _each_distinct_magnitude(z, lambda y: _log_erfc(y * _SQRT_HALF) / _LN10)


def _each_distinct_magnitude(z: np.ndarray, rule: Callable[[float], float]) -> np.ndarray:
    """``rule(|z|)`` for every element of `z`, as float64, evaluated once per
    distinct |z| (see `log10_p_two_sided` for why that is fast on a stored z)."""
    magnitude = np.abs(np.asarray(z))
    distinct, where = np.unique(magnitude.ravel(), return_inverse=True)
    values = np.fromiter(
        map(rule, distinct.astype("float64").tolist()), dtype="float64", count=len(distinct)
    )
    result: np.ndarray = values[where].reshape(magnitude.shape)
    return result


class AssociationStatistics(NamedTuple):
    """What a stored z and SE imply, one float64 value per association (ADR 0104)."""

    #: ``z * se``, on the Stored Effect Scale.
    beta: np.ndarray
    #: The two-sided p, ``erfc(|z| / sqrt(2))``: 0.0 where it is below float64's
    #: range (|z| above about 38.5) and subnormal just before; `log10_p` is exact there.
    pval: np.ndarray
    #: log10 of `pval`, finite and exact where `pval` underflows.
    log10_p: np.ndarray


def association_statistics(z: np.ndarray, se: np.ndarray) -> AssociationStatistics:
    """`beta`, `pval` and `log10_p` for each association, from its z and SE (ADR 0104).

    float64 whatever the planes' dtype. A missing association -- z or SE NaN --
    is NaN in all three. So is one whose SE is infinite: an SE above a `float16`
    plane's range is an error a release before format 0.2.3 may still hold
    (ADR 0068), and a beta of ``z * inf`` or a p from its z alone would read as
    a value. An infinite z with a finite SE keeps its infinities: beta ±inf, p
    0.0, log10_p -inf.

    `log10_p` is `log10_p_two_sided`'s, and `pval` is `p_value_from_z`'s, bit for
    bit; each is evaluated once per distinct |z|.
    """
    z64 = np.asarray(z, dtype="float64")
    se64 = np.asarray(se, dtype="float64")
    if z64.shape != se64.shape:
        raise ValueError(f"z and se differ in shape: {z64.shape} and {se64.shape}")
    with np.errstate(invalid="ignore"):  # 0 * inf; set to NaN below with the rest
        beta = z64 * se64
    pval = _each_distinct_magnitude(z64, p_value_from_z)
    log10_p = log10_p_two_sided(z64)
    unusable = ~np.isfinite(se64)
    for values in (beta, pval, log10_p):
        values[unusable] = np.nan
    return AssociationStatistics(beta, pval, log10_p)


def z_critical(threshold: float) -> float:
    """|z| cutoff equivalent to two-sided p <= threshold.

    The stored p-value is ``p = erfc(|z| / sqrt(2))`` (`p_value_from_z`), so
    ``p <= threshold`` is exactly ``|z| >= -Φ⁻¹(threshold / 2)``. Thresholding on
    ``|z|`` lets the hot path avoid computing a p-value for every cell. The index
    build, the Ragged scan fallback and the validator all cut here and nowhere
    else.
    """

    return -statistics.NormalDist().inv_cdf(threshold / 2.0)
