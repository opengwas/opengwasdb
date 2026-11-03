"""What Admission guarantees of every cell a Store holds, and its checks.

A present cell has a finite z and a finite, positive SE; z is missing exactly
where SE is (spec §15, `opengwasdb.associations.record.states`); a frequency is
NaN or inside (0, 1); and a cell is flagged imputed only when it is present.
A builder's rows reach a Store through a `Gate`, which admits only such
cells; completion's candidates are checked by `admissible`, and a writer
handed cells from anywhere else checks them with `require_admitted`.
"""

from __future__ import annotations

import numpy as np

from opengwasdb.associations.record import State, states

__all__ = ["NotAdmitted", "admissible", "require_admitted", "require_paired"]


class NotAdmitted(ValueError):
    """A cell Admission would not admit reached a Store's writer."""


def admissible(z: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Which candidate cells may be stored as present: z finite, SE finite and positive."""
    with np.errstate(invalid="ignore"):
        keep: np.ndarray = np.isfinite(z) & np.isfinite(se) & (se > 0)
    return keep


def require_admitted(
    z: np.ndarray,
    se: np.ndarray,
    eaf: np.ndarray | None = None,
    imputed: np.ndarray | None = None,
    *,
    where: str,
) -> None:
    """Refuse cells Admission would not have admitted, naming ``where`` and the first.

    ``z`` and ``se`` are parallel arrays of any shape, NaN where missing;
    ``eaf`` (NaN where absent) and ``imputed`` (the 0/1 mask) are optional and
    parallel too. Raises `NotAdmitted` listing each broken rule's cell count.
    """
    z, se = np.asarray(z, dtype=np.float64), np.asarray(se, dtype=np.float64)
    z_missing, se_missing = np.isnan(z), np.isnan(se)
    state = states(z_missing, se_missing, None if imputed is None else np.asarray(imputed))
    present = ~z_missing & ~se_missing
    broken = {
        "one of z and SE missing": state == State.PARTIAL,
        "imputed flag on a missing cell": state == State.IMPUTED_MISSING,
        "imputed flag neither 0 nor 1": state == State.FLAG_INVALID,
        "present with an infinite z or an SE not finite and positive": present
        & ~admissible(z, se),
    }
    if eaf is not None:
        eaf = np.asarray(eaf, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            broken["frequency outside (0, 1)"] = ~np.isnan(eaf) & ~((eaf > 0) & (eaf < 1))
    _raise_if_broken(broken, where)


def require_paired(z: np.ndarray, se: np.ndarray, *, where: str) -> None:
    """Refuse cells with one of z and SE missing (spec §15, D1), naming ``where``.

    What a completion checks of the cells it carries from a source Store, as
    the Dense completion does of its bands: a half-missing pair would be stored
    beside an imputed partner, or carried as a cell no reader can use.
    """
    unpaired = np.isnan(np.asarray(z, dtype=np.float64)) != np.isnan(
        np.asarray(se, dtype=np.float64)
    )
    _raise_if_broken({"one of z and SE missing": unpaired}, where)


def _raise_if_broken(broken: dict[str, np.ndarray], where: str) -> None:
    found = {rule: mask for rule, mask in broken.items() if mask.any()}
    if not found:
        return
    first_rule, first_mask = next(iter(found.items()))
    first = tuple(int(axis[0]) for axis in np.nonzero(first_mask))
    counts = "; ".join(f"{rule}: {int(mask.sum()):,}" for rule, mask in found.items())
    raise NotAdmitted(
        f"{where}: cells Association Admission does not admit ({counts}); first "
        f"{first_rule} at {first}"
    )
