"""Pgen reading and PC-adjusted r shared by builds and on-demand reads (ADR 0013)."""

from ogld.genotypes.adjust import Block, Covariates, covariates, prepare, r, residuals
from ogld.genotypes.pgen import Genotypes, import_pgenlib

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
