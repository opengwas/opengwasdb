"""The PC Adjustment a build makes, and the pgen reading and PC-adjusted r it shares with a
Genotype Store's reads (ADR 0099).

`Block`, `Genotypes`, `import_pgenlib` and `r` are ogquery's (`ogquery.ld.genotypes`;
opengwas/ogquery#11, ogquery ADR 0002, amended): a Genotype Store is read there, and a build reads
its pgens and computes its tiles with the same objects. What stays here is what builds, writes
and validates one: `Covariates`, `covariates`, `residuals` and `prepare` (the PC Adjustment), the
sidecar's writer and its validation. Reading a pgen needs ogquery's `genotype` extra, which this
package's `ld-build` extra installs.
"""

from ogquery.ld.genotypes import Block, Genotypes, import_pgenlib, r

from opengwasdb.ld.genotypes.adjust import Covariates, covariates, prepare, residuals

__all__ = [
    "Block",
    "Covariates",
    "Genotypes",
    "covariates",
    "import_pgenlib",
    "prepare",
    "r",
    "residuals",
]
