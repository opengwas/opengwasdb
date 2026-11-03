"""How a component stores associations: what it may hold, and how it encodes them.

`storable` is the one judgement a writer makes before any plane is written:
every cell must be one Association Admission would admit (ADR 0072), and an SE
above what a `float16` plane can represent is counted per Analysis. Admission
guarantees the first for every source row, so a cell that breaks it here was
produced after admission -- a completion fill, a merged pass, a re-encode --
and the write is refused, naming Admission's rule. Nothing is filtered here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from opengwasdb.admission import require_admitted

log = logging.getLogger(__name__)

__all__ = [
    "SE_FLOAT16_MAX",
    "CellValues",
    "Storable",
    "log_se_out_of_range",
    "storable",
]

#: The largest SE a `float16` plane holds as itself (65,504). A larger one is
#: stored as `+inf`, which is still an association (spec §15, issue #337).
SE_FLOAT16_MAX: Final = float(np.finfo(np.float16).max)

#: Cells `storable` checks at a time, so a Dense band costs window-sized masks.
_WINDOW = 1 << 22


@dataclass(frozen=True)
class CellValues:
    """A batch of association cells' statistics, before they are encoded.

    Parallel arrays of one shape, a column, a cell list or a row band. NaN is
    missing in `z` and `se`, and absent in `eaf`; an SE of `+inf` is a value.
    `eaf` is None when the batch carries no frequency plane, and `imputed`
    (the 0/1 mask) when every present cell is observed.
    """

    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray | None = None
    imputed: np.ndarray | None = None


@dataclass(frozen=True)
class Storable:
    """What a Store holds of a batch, and what the batch counted."""

    #: The cells to encode.
    cells: CellValues
    #: Which cells hold an association once stored.
    stored: np.ndarray
    #: Per Analysis, how many stored SEs exceed `SE_FLOAT16_MAX`, `+inf` included.
    se_out_of_range: np.ndarray


def storable(
    cells: CellValues, analyses: np.ndarray | int, n_analyses: int, *, where: str
) -> Storable:
    """What a Store holds of `cells`, whose Analysis indices are `analyses`.

    `analyses` broadcasts against the cells' shape: one Analysis's index for a
    column, one per cell for a cell list, one per column for a row band.
    `where` names the writer, for the refusal. Pure and idempotent: the plan
    plays no part, so a writer calls it before the SE plan exists.

    Raises `opengwasdb.admission.NotAdmitted`, through Admission's own check,
    when a cell is one Admission would not admit (one of z and SE missing, an
    infinite z, an SE not positive, an imputed flag that is not 0/1 or sits on
    a missing cell). An SE of `+inf` is not refused: it is a stored value.
    """
    z, se, imputed = _parallel(cells, where)
    owner = np.broadcast_to(np.asarray(analyses, dtype=np.int64), z.shape)
    over = np.zeros(n_analyses, dtype=np.int64)
    for window in _windows(z.shape):
        z_w, se_w = z[window], se[window]
        imputed_w = None if imputed is None else imputed[window]
        if _breaks_admission(z_w, se_w, imputed_w):
            _refuse(z_w, se_w, imputed_w, where=_where(where, window, z.shape))
        over += _out_of_range(se_w, owner[window], n_analyses)
    return Storable(cells=cells, stored=~np.isnan(z), se_out_of_range=over)


def log_se_out_of_range(counts: np.ndarray, analysis_ids: Sequence[str], *, where: str) -> None:
    """Log, as one warning, each Analysis's SEs above `SE_FLOAT16_MAX`.

    Such an SE is stored as `+inf` and stays an association; the count is the
    one issue #338 makes each Store record. Nothing is logged for none.
    """
    counts = np.asarray(counts, dtype=np.int64)
    found = np.flatnonzero(counts)
    if not len(found):
        return
    named = ", ".join(f"{analysis_ids[i]}: {int(counts[i]):,}" for i in found.tolist())
    log.warning(
        "%s: %s SE value(s) above %s, stored as +inf (spec §15): %s",
        where,
        f"{int(counts.sum()):,}",
        f"{SE_FLOAT16_MAX:,.0f}",
        named,
    )


def _parallel(cells: CellValues, where: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    z, se = np.asarray(cells.z), np.asarray(cells.se)
    imputed = None if cells.imputed is None else np.asarray(cells.imputed)
    if se.shape != z.shape or (imputed is not None and imputed.shape != z.shape):
        raise ValueError(
            f"{where}: z {z.shape}, se {se.shape} and imputed "
            f"{None if imputed is None else imputed.shape} must be parallel"
        )
    return z, se, imputed


def _out_of_range(se: np.ndarray, owner: np.ndarray, n_analyses: int) -> np.ndarray:
    """Per Analysis, the SEs above `SE_FLOAT16_MAX` among these cells."""
    with np.errstate(invalid="ignore"):
        over = se > SE_FLOAT16_MAX
    return np.bincount(owner[over], minlength=n_analyses).astype(np.int64)


def _windows(shape: tuple[int, ...]) -> list[tuple[slice, ...]]:
    """Leading-axis slices of at most `_WINDOW` cells (one row at least)."""
    if not shape:
        return [()]
    width = int(np.prod(shape[1:], dtype=np.int64)) or 1
    step = max(1, _WINDOW // width)
    return [(slice(i, min(i + step, shape[0])),) for i in range(0, max(shape[0], 1), step)]


def _breaks_admission(z: np.ndarray, se: np.ndarray, imputed: np.ndarray | None) -> bool:
    """Whether any cell breaks Admission's guarantee, `+inf` SE allowed; in the
    cells' own dtype, so a band is not copied to `float64` to find nothing."""
    z_missing, se_missing = np.isnan(z), np.isnan(se)
    if np.any(z_missing != se_missing):
        return True
    present = ~z_missing
    with np.errstate(invalid="ignore"):
        if np.any(present & (~np.isfinite(z) | ~(se > 0))):
            return True
    if imputed is not None:
        flagged = imputed != 0
        if np.any(flagged & ((imputed != 1) | z_missing)):
            return True
    return False


def _refuse(z: np.ndarray, se: np.ndarray, imputed: np.ndarray | None, *, where: str) -> None:
    """Raise Admission's own refusal for these cells.

    An SE narrowed to `+inf` is handed on as `SE_FLOAT16_MAX`, so the check
    judges every other rule and names the cell that broke one.
    """
    finite_se = np.where(np.isposinf(se), np.float32(SE_FLOAT16_MAX), se)
    require_admitted(z, finite_se, imputed=imputed, where=where)
    raise AssertionError(f"{where}: a cell broke Admission's guarantee, yet its check passed")


def _where(where: str, window: tuple[slice, ...], shape: tuple[int, ...]) -> str:
    """`where`, naming the window when the batch spans more than one."""
    if not window or (window[0].start == 0 and window[0].stop == shape[0]):
        return where
    return f"{where} (from index {window[0].start:,} of the batch's first axis)"
