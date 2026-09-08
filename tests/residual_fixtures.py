"""Source records shared by the suites that need a residual-codable `se` plane.

Two suites need the same eligible source and had grown their own copies of it,
which the duplication gate caught: one checks the format-3 coding round-trips
end to end, the other builds a release for the migration to derive from. It
lives here rather than in `conftest.py` because it is a builder, not a fixture
-- callers want the records at their own moment, with their own store path.
"""

from __future__ import annotations

import numpy as np

from opengwasdb.build.source import NormalisedAssociation
from opengwasdb.variants import CanonicalVariant


def residual_eligible_records(
    n_variants: int = 600,
) -> tuple[list[NormalisedAssociation], dict[str, np.ndarray]]:
    """Two Analyses whose SE tracks its MAF-predicted value closely.

    `log(se)` is linear in `log(2f(1-f))` with a small periodic wobble, so
    nearly every cell lands inside ±0.5 and a format-3 build residual-codes the
    plane -- which is also what makes it data a format-2.0 build would have had
    to store as `float16` without being able to express as a residual.

    Returns the records and, per `analysis_id`, the SE values they carry, so a
    caller can assert what a query gives back against what it put in.
    """
    frequencies = np.linspace(0.05, 0.95, n_variants, dtype=np.float32)
    records: list[NormalisedAssociation] = []
    expected: dict[str, np.ndarray] = {}
    for col, analysis_id in enumerate(("a", "b")):
        values = np.exp(
            (-3.0 + col * 0.2)
            - 0.5 * np.log(2 * frequencies * (1 - frequencies))
            + 0.12 * np.sin(np.arange(len(frequencies)) * (0.07 + col * 0.01))
        ).astype(np.float32)
        expected[analysis_id] = values
        records.extend(
            NormalisedAssociation(
                analysis_id=analysis_id,
                variant=CanonicalVariant("1", row + 1, "A", "G"),
                z=8.0 if row % 100 == 0 else 1.0,
                se=float(values[row]),
                eaf=float(frequencies[row]),
            )
            for row in range(len(frequencies))
        )
    return records, expected
