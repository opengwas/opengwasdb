"""Source records and writers shared by the suites that need residual-codable
`se` planes (issues #118/#141 acceptance).

Two suites need the same eligible source and had grown their own copies of it,
which the duplication gate caught: one checks the format-3 coding round-trips
end to end, the other builds a release for the migration to derive from. It
lives here rather than in `conftest.py` because it is a builder, not a fixture
-- callers want the records at their own moment, with their own store path.
The same is true of the GWAS-VCF-with-EAF writer below: a format-3 build and
its Hybrid end-to-end coverage must feed the builder bytes whose FORMAT
contract cannot drift apart between them. The frequency-tracking rows
(`maf_vcf_rows`) and the manifest writer join it for the same reason: several
suites build residual-coded releases from them, and the shape that makes a
plane residual-code -- 600 variants whose SE tracks their frequency, in an
AF-bearing FORMAT -- is a fact they must share.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from opengwasdb.layouts.dense.build import NormalisedAssociation
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


#: The variant count the GWAS-VCF frequency fixtures share. The planes only
#: residual-code when there are enough cells to measure a fit, and every caller
#: of `maf_vcf_rows` wants that.
MAF_VCF_N_VARIANTS = 600

#: The effect-allele frequencies those rows carry, one per variant row.
MAF_VCF_FREQUENCIES = np.linspace(0.05, 0.95, MAF_VCF_N_VARIANTS, dtype=np.float64)


def model_se(column: int, frequency: float, row: int) -> float:
    """A standard error tracking the row's frequency, so `se` residual-codes.

    `log(se)` is linear in `log(2f(1-f))` with a small periodic wobble, so nearly
    every cell lands inside the residual coding's range. The per-Analysis column
    offset and phase make the Analyses different from each other, as a fit over
    one shared model would otherwise not require.
    """
    return float(
        np.exp(
            (-3.0 + column * 0.2)
            - 0.5 * np.log(2 * frequency * (1 - frequency))
            + 0.12 * np.sin(row * (0.07 + column * 0.01))
        )
    )


def maf_vcf_rows(
    column: int, z_for: Callable[[int, int], float], *, with_af: bool = True
) -> list[str]:
    """GWAS-VCF body rows for one Analysis, from the shared frequency fixture.

    ``z_for(row, column)`` gives each cell's z-score, and the ES column carries
    ``z * se`` so the build stores exactly that z. ``with_af=False`` drops the
    frequency, which is the same data under a FORMAT block that cannot yield an
    `eaf` plane.
    """
    rows: list[str] = []
    for row in range(MAF_VCF_N_VARIANTS):
        frequency = float(MAF_VCF_FREQUENCIES[row])
        se = model_se(column, frequency, row)
        effect = f"{z_for(row, column) * se:.6f}:{se:.6f}"
        format_and_sample = f"ES:SE:AF\t{effect}:{frequency:.6f}" if with_af else f"ES:SE\t{effect}"
        rows.append(f"1\t{(row + 1) * 1000}\t.\tA\tG\t.\tPASS\t.\t{format_and_sample}\n")
    return rows


def write_hg38_manifest(path: Path, vcfs: dict[str, Path]) -> Path:
    """A `build-dense-vcf` manifest of already-GRCh38 GWAS-VCFs, one row each.

    Every Analysis is labelled after its id and declares the SD effect scale with
    no rescaling, so a caller's VCF rows are what the build stores.
    """
    path.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\tsource_assembly\n"
        + "\n".join(
            f"{analysis_id}\t{vcf}\tTrait {analysis_id}\t1000\tsd"
            "\tdeclared_standardised\thg38"
            for analysis_id, vcf in vcfs.items()
        )
        + "\n",
        encoding="utf-8",
    )
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
