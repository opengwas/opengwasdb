"""Reference-panel EAF for a completed release's imputed cells (ADR 0037 §4).

An imputed cell's EAF *is* the panel's, and it is identical for every Analysis
imputed at that variant, so it is one `float32` per variant rather than
per-cell data. Both the Dense and the Ragged completion pipelines need the same
two things from it, and they need them to agree: which frequency each variant
of the completed axis gets, and what an Analysis that gained imputed cells now
declares as its `eaf_scope`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.build.eaf_orientation import panel_a1_eaf
from opengwasdb.model.enums import EafScope
from opengwasdb.variants import CanonicalVariant


def panel_reference_eaf(
    ld_dir: str | Path, ancestry: str, variants: Sequence[CanonicalVariant]
) -> np.ndarray | None:
    """One A1-oriented panel frequency per variant of the completed axis.

    `None` when the panel declares none, so a release only claims to carry
    reference EAF when it has some to carry. Variants the panel does not hold
    -- every off-panel row the source contributed -- are NaN, which is correct:
    they have no imputed cells to describe.
    """
    panel = panel_a1_eaf(ld_dir, ancestry)
    if not panel:
        return None
    reference = np.fromiter(
        (panel.get(v.alid, np.nan) for v in variants), dtype=np.float32, count=len(variants)
    )
    if not np.any(np.isfinite(reference)):
        return None
    print(
        f"Reference EAF: {int(np.count_nonzero(np.isfinite(reference))):,} of "
        f"{len(reference):,} variants carry a panel frequency"
    )
    return reference


def completed_eaf_scope(analysis: Any, rollup: Any, carries_reference: bool) -> str:
    """`eaf_scope` for a completed Analysis, from what the release now holds.

    An Analysis that gained imputed cells in a release carrying reference EAF
    now stores a frequency for them, whatever its source reported (ADR 0037
    §4). The declaration is derived from the release rather than copied
    forward: `eaf_scope` disagreeing with the arrays is the defect that got
    through review on #106.
    """
    if carries_reference and int(rollup.n_imputed_total or 0) > 0:
        return str(EafScope.ASSOCIATION.value)
    return str(analysis.eaf_scope)
