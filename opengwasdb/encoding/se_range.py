"""The one rule for a standard error float16 cannot hold (issue #338, ADR 0068).

An association whose SE is strictly above `SE_MAX_STORED` -- float16's largest
finite value -- is almost certainly an error in its source, and a build stores
it as **missing**: both its `z` and its `se` carry the plane's Missing Marker
(spec §15), whatever the plane's encoding.  The rule is applied once, to an
Analysis's `z` and `se` together, so the two markers cannot disagree.

Two functions, with different jobs:

* `mask_se_out_of_range` is the rule.  Every point that narrows an SE calls it,
  at the place where both `z` and `se` are in hand.  No caller restates the
  comparison.
* `require_se_in_range` is the guard.  The encoders call it so a path that
  forgot the rule fails loudly instead of writing `+inf` (numpy's
  `RuntimeWarning: overflow encountered in cast` was the only signal before).
"""

from __future__ import annotations

import numpy as np

#: float16's largest finite value, 65,504.  An SE strictly above it is missing.
SE_MAX_STORED: float = float(np.finfo(np.float16).max)


class SeOutOfRangeError(ValueError):
    """An SE above `SE_MAX_STORED` reached an encoder that must not narrow it."""


def se_out_of_range(se: np.ndarray) -> np.ndarray:
    """Boolean mask of the cells whose SE exceeds `SE_MAX_STORED` (``NaN`` is not)."""
    with np.errstate(invalid="ignore"):
        return np.asarray(se, dtype=np.float64) > SE_MAX_STORED


def mask_se_out_of_range(
    z: np.ndarray, se: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """`z` and `se` with both set to ``NaN`` where `se` exceeds the bound, and the count.

    Returns copies only when a cell is masked; otherwise the inputs come back
    unchanged.  ``NaN`` is the in-memory form of the Missing Marker in every
    builder's scratch; the encoders turn it into the plane's marker.
    """
    z = np.asarray(z)
    se = np.asarray(se)
    if z.shape != se.shape:
        raise ValueError(f"z shape {z.shape} does not match se shape {se.shape}")
    bad = se_out_of_range(se)
    count = int(np.count_nonzero(bad))
    if not count:
        return z, se, 0
    z, se = z.copy(), se.copy()
    z[bad] = np.nan
    se[bad] = np.nan
    return z, se, count


def require_se_in_range(values: np.ndarray, *, writer: str) -> None:
    """Raise `SeOutOfRangeError` naming the count and first value if any SE exceeds the bound."""
    arr = np.asarray(values)
    bad = se_out_of_range(arr)
    if not np.any(bad):
        return
    n = int(np.count_nonzero(bad))
    first = float(arr[bad].flat[0])
    raise SeOutOfRangeError(
        f"{writer}: {n} standard error(s) above float16's maximum {SE_MAX_STORED:g} "
        f"(first: {first!r}); such an association is stored as missing, so the build "
        "must mask it (z and se together) with `mask_se_out_of_range` before encoding "
        "(issue #338)"
    )


def mask_se_grid(
    z: np.ndarray, se: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`mask_se_out_of_range` over a (variants x Analyses) grid, counted per Analysis.

    Returns `z`, `se` and an int64 vector of one count per column.
    """
    counts = se_out_of_range_counts(se)
    z_out, se_out, _ = mask_se_out_of_range(z, se)
    return z_out, se_out, counts


def se_out_of_range_counts(se: np.ndarray) -> np.ndarray:
    """Per-column count of cells the rule would mask in a 2-D `se` (no copy of `z`)."""
    return np.count_nonzero(se_out_of_range(se), axis=0).astype(np.int64)


def mask_se(se: np.ndarray) -> np.ndarray:
    """`se` with every cell above the bound set to ``NaN``: the rule where `z` is not in hand.

    A completion pass that carries only the `se` plane applies the rule here;
    the pass that carries `z` applies `mask_se_out_of_range` to the same cells,
    so the two planes mask exactly the same ones.
    """
    return mask_se_out_of_range(se, se)[1]
