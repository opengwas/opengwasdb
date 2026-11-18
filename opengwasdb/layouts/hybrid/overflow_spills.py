"""The Hybrid Ragged Overflow, assembled from its Pass 2 spills.

A spill holds what Pass 2's `storable` returned for each column
(`build._stored_parts`): rows already divided by the Analysis's phenotype SD
(ADR 0084). This is therefore a writer of cells already standardised, and says
so with `PhenotypeSd.APPLIED`; dividing again would standardise them twice. It
lives apart from `build.py`, which reads source rows, because only a module on
the stored-cell allowlist may name `APPLIED`
(`tests/test_phenotype_sd_standardisation.py`), and it reads no source row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opengwasdb.build.ordered_pool import ordered_map
from opengwasdb.encoding.phenotype_sd import PhenotypeSd
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter

__all__ = ["assemble_overflow_csr"]


@dataclass(frozen=True)
class _OverflowColumn:
    """One Analysis's overflow spill, read, sorted and ready for the CSR.

    `eaf` is ``None`` when the column carries no finite frequency, which is
    what the writer turns into an all-NaN row (ADR 0036).
    """

    variant_index: np.ndarray
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray | None


def _assemble_overflow_column(task: tuple[int, str]) -> _OverflowColumn | None:
    """Load and sort one ``.ovf.npz`` column. Runs in a forked worker.

    Returns `None` for a column with no spill file. The spill is *not* deleted
    here: the parent deletes it once the column has been added to the CSR, so a
    failure partway through combination cannot lose a spill that was never
    used.
    """
    col, spill_dir = task
    path = Path(spill_dir) / f"{col}.ovf.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        vi = data["variant_index"].astype(np.int32)
        z = data["z"].astype(np.float32)
        se = data["se"].astype(np.float32)
        eaf = data["eaf"].astype(np.float32)
    # Sort by variant_index for consistent within-analysis ordering (matches
    # the ragged BESD builder and lets top-hit CSR cross-validation searchsort).
    order = np.argsort(vi, kind="stable")
    has_eaf = bool(np.isfinite(eaf).any())
    return _OverflowColumn(
        variant_index=vi[order],
        z=z[order],
        se=se[order],
        eaf=eaf[order] if has_eaf else None,
    )


def assemble_overflow_csr(
    spill_dir: Path,
    n_analyses: int,
    n_variants: int,
    n_workers: int = 1,
    *,
    consume_spills: bool = True,
) -> tuple[RaggedCSRWriter, np.ndarray]:
    """Assemble the overflow CSR from per-column ``.ovf.npz`` spills, in analysis
    order (so CSR offsets align with analysis_index).

    Also returns a per-column bool array saying which Analyses carried an EAF
    into the *overflow* component. An Analysis can have EAF off-panel and none
    on it, so `eaf_scope` is the union of this and the Dense Component's own
    answer, never either alone (ADR 0036).

    The columns are independent, so `n_workers` > 1 loads and sorts them through
    `ordered_map`, which keeps only a bounded number of results in flight, and
    the parent adds them to the CSR in analysis order. `n_workers <= 1` is the
    serial path.

    ``consume_spills=False`` keeps each plate after it has been added (issue
    #227): the build tail that a checkpoint re-runs wholesale re-assembles the
    CSR, and the assembled writer lives only in memory, so the plates are its
    only durable input. The plates are read and combined identically either
    way -- the two differ in what is left on disk, not in what is written.
    """
    csr = RaggedCSRWriter(n_variants)
    column_has_eaf = np.zeros(n_analyses, dtype=bool)
    tasks = ((col, str(spill_dir)) for col in range(n_analyses))
    for col, result in enumerate(ordered_map(_assemble_overflow_column, tasks, n_workers)):
        if result is None:
            csr.add_analysis(
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                phenotype_sd=PhenotypeSd.APPLIED,
            )
            continue
        column_has_eaf[col] = result.eaf is not None
        csr.add_analysis(
            result.variant_index, result.z, result.se, eaf=result.eaf,
            phenotype_sd=PhenotypeSd.APPLIED,
        )
        if consume_spills:
            (spill_dir / f"{col}.ovf.npz").unlink()
    return csr, column_has_eaf
