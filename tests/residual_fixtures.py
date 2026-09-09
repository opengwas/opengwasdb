"""Source records and writers shared by the suites that need residual-codable
`se` planes (issues #118/#141 acceptance).

Two suites need the same eligible source and had grown their own copies of it,
which the duplication gate caught: one checks the format-3 coding round-trips
end to end, the other builds a release for the migration to derive from. It
lives here rather than in `conftest.py` because it is a builder, not a fixture
-- callers want the records at their own moment, with their own store path.
The same is true of the GWAS-VCF-with-EAF writer below: a format-3 build and
its Hybrid end-to-end coverage must feed the builder bytes whose FORMAT
contract cannot drift apart between them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from opengwasdb.build.source import NormalisedAssociation
from opengwasdb.variants import CanonicalVariant

#: A format-4.2 GWAS-VCF header declaring ES, SE, EZ and AF, one sample
#: ``S``. Shared because several suites build residual-SE releases from VCF
#: rows whose SE column only means anything when the FORMAT block names it
#: identically.
GWAS_VCF_WITH_EAF_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##FORMAT=<ID=ES,Number=A,Type=Float,Description=\"Effect size\">\n"
    "##FORMAT=<ID=SE,Number=A,Type=Float,Description=\"Standard error\">\n"
    "##FORMAT=<ID=EZ,Number=A,Type=Float,Description=\"Z-score\">\n"
    "##FORMAT=<ID=AF,Number=A,Type=Float,Description=\"Alternate allele frequency\">\n"
    "##SAMPLE=<ID=S,StudyType=Continuous>\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
)


def write_gwas_vcf_with_eaf(path: Path, rows: list[str]) -> Path:
    """Write ``rows`` under ``GWAS_VCF_WITH_EAF_HEADER`` and return the path."""
    path.write_text(GWAS_VCF_WITH_EAF_HEADER + "".join(rows), encoding="utf-8")
    return path


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
