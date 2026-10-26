"""Validation of the optional per-association sample-size plane (ADR 0067).

Every rule here is one way the `n` plane could read as a plausible sample size
while being wrong: a plane the plan does not declare, so a reader following the
plan never sees it; one the plan declares and the component lacks; a length out
of step with the CSR, so every later cell is another association's N; a code
that is neither a count nor the missing code; a plane that holds no N at all,
which reads exactly like a source without one; and an Analysis whose
`sample_size_scope` claims one N describes all its associations while the
plane holds a different one per association.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from opengwasdb.encoding import N_MAX, N_MISSING, StoreEncoding
from opengwasdb.model.analyses import read_analyses
from opengwasdb.model.enums import SampleSizeScope
from opengwasdb.store.arrays import array_length

#: Cells one validation window reads (4 MB of `uint32`), so the plane is never
#: held whole (issue #254).
_WINDOW = 1_000_000


def validate_n_plan(group: Any, encoding: StoreEncoding, errors: list[str], *, label: str) -> None:
    """The `n` plane is present exactly when the plan declares it, in its dtype,
    and only beside a `float16` `se` plane (ADR 0067 §4)."""
    present = "n" in group
    if encoding.n is not None and encoding.se.is_residual:
        errors.append(
            f"{label} declares an n plane beside se {encoding.se.kind}, but an n plane "
            "requires float16 se (ADR 0067 §4)"
        )
    if encoding.n is None:
        if present:
            errors.append(
                f"{label} carries an n plane but the manifest declares none; a reader "
                "following the plan would never read those sample sizes (ADR 0067)"
            )
        return
    if not present:
        errors.append(f"{label} declares an n plane but carries none (ADR 0067)")
        return
    actual = str(group["n"].dtype)
    if actual != encoding.n.dtype:
        errors.append(
            f"{label}/n has dtype {actual} but the manifest declares "
            f"{encoding.n.kind} ({encoding.n.dtype})"
        )


@dataclass(frozen=True)
class _PlaneScan:
    """One windowed pass over the plane: what it holds, per Analysis."""

    held: np.ndarray
    n_invalid: int
    first_invalid: int | None


def _scan(plane: Any, offsets: np.ndarray) -> _PlaneScan:
    """Count each Analysis's cells holding an N, and the cells holding nonsense."""
    n_analyses = len(offsets) - 1
    held = np.zeros(n_analyses, dtype=np.int64)
    n_invalid = 0
    first_invalid: int | None = None
    total = int(offsets[-1])
    for lo in range(0, total, _WINDOW):
        codes = np.asarray(plane[lo : min(lo + _WINDOW, total)])
        invalid = (codes < 1) | ((codes > N_MAX) & (codes != N_MISSING))
        if invalid.any():
            n_invalid += int(invalid.sum())
            if first_invalid is None:
                first_invalid = lo + int(np.argmax(invalid))
        positions = np.flatnonzero(~invalid & (codes != N_MISSING)) + lo
        if len(positions):
            analyses = np.searchsorted(offsets[1:], positions, side="right")
            held += np.bincount(analyses, minlength=n_analyses)
    return _PlaneScan(held, n_invalid, first_invalid)


def validate_ragged_n(
    store: Any, root: Any, encoding: StoreEncoding, errors: list[str]
) -> None:
    """The CSR's `n` plane: parallel, valid or missing, and agreeing with the scope.

    Runs after the analyses seam, because the scope rule reads `analyses.tsv`.
    A release without the plane has nothing here to check.
    """
    if encoding.n is None or "n" not in root:
        return
    offsets = np.asarray(root["offsets"][:], dtype=np.int64)
    n_assoc = int(offsets[-1])
    length = array_length(root["n"])
    if length != n_assoc:
        errors.append(f"data.zarr/ragged/n has {length} entries but offsets imply {n_assoc}")
        return
    scan = _scan(root["n"], offsets)
    if scan.n_invalid:
        errors.append(
            f"data.zarr/ragged/n holds {scan.n_invalid} cell(s) that are neither a sample "
            f"size from 1 to {N_MAX} nor the missing code {N_MISSING} (first at flat "
            f"position {scan.first_invalid})"
        )
    if n_assoc and not scan.held.any():
        errors.append(
            "data.zarr/ragged/n holds no sample size at all; a plane of missing cells "
            "reads exactly like a release without one, and no build writes it (ADR 0067)"
        )
    _scope_errors(store, scan.held, errors)


def _scope_errors(store: Any, held: np.ndarray, errors: list[str]) -> None:
    """An Analysis holding per-association N must declare `variant_level` scope.

    The converse is not required: `variant_level` with no stored N says the N
    varies by variant and the release does not hold it, which is true of every
    Analysis whose pairs the table omitted.
    """
    for row in read_analyses(store.analyses_path).rows:
        index = int(row["analysis_index"])
        if index >= len(held) or held[index] == 0:
            continue
        scope = row.get("sample_size_scope", "")
        if scope != SampleSizeScope.VARIANT_LEVEL.value:
            errors.append(
                f"analysis {row.get('analysis_id', '')!r} holds a per-association N for "
                f"{int(held[index])} association(s) but declares sample_size_scope="
                f"{scope!r}; per-association sample sizes are "
                f"{SampleSizeScope.VARIANT_LEVEL.value} (ADR 0067)"
            )
