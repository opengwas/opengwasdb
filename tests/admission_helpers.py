"""What a build stores of a Source Reader's stream: the rows Association Admission keeps.

A reader yields a row with no effect or no usable SE (a NaN z, an unusable SE)
for Admission to drop and log, so a test that asks what a build would store
judges the stream through a real counting gate rather than restating the rules.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from opengwasdb.admission import AnalysisRules, SourceRows, counting_gate
from opengwasdb.readers.interface import ReaderAssociation


def admitted(associations: Iterable[ReaderAssociation]) -> list[ReaderAssociation]:
    """The associations a build with no declared filter admits, in stream order."""
    rows = list(associations)
    keep = counting_gate(AnalysisRules("test")).source_rows(
        SourceRows(
            eaf=np.array([np.nan if a.eaf is None else a.eaf for a in rows], dtype=np.float64),
            se=np.array([a.se for a in rows], dtype=np.float64),
            z=np.array([a.z for a in rows], dtype=np.float64),
            keys_at=lambda positions: [str(rows[i].position) for i in positions.tolist()],
        )
    )
    return [a for a, kept in zip(rows, keep.tolist(), strict=True) if kept]
