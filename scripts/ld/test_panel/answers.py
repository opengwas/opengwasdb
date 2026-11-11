"""Compute every answer the Test Panel commits, from the converted panel's own genotypes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ogld.build import pc_check
from test_panel import reference
from test_panel.assemble import Panel
from test_panel.params import Params


@dataclass(frozen=True)
class Answers:
    """`i`, `j` are Test Panel rows; `r[k]` and `r_plink2` are aligned with them. `pc_check`
    is the PC Check on the Test Panel's own common variants; its `covariates` are what
    `r[10]` and the GWAS regress out, as a build of the Test Panel would."""

    i: np.ndarray
    j: np.ndarray
    r: dict[int, np.ndarray]
    r_plink2: np.ndarray
    z: np.ndarray
    causal: list[int]
    betas: list[float]
    pc_check: dict[str, object]
    leads: dict[float, list[int]] = field(default_factory=dict)
    clumps: dict[float, Path] = field(default_factory=dict)


def compute(
    params: Params, panel: Panel, files: dict[str, Path], g: np.ndarray, pcs: np.ndarray
) -> Answers:
    ok = panel.passing
    extract = params.work / "passing.extract"
    extract.write_text("".join(f"r{r}\n" for r in ok))
    check, covariates = _covariates(params, panel, g, pcs)
    pi, pj = reference.pairs(np.array(panel.chrom)[ok], panel.pos[ok], params.far_bp)
    r = _reference(params, g[ok], covariates, pi, pj)
    r_plink2 = reference.plink2_r(params, files, extract, len(ok), pi, pj)
    causal = [int(row) for row in ok if "causal" in panel.components[row]]
    where = {int(row): n for n, row in enumerate(ok)}
    z_ok, betas = reference.simulate_gwas(params, g[ok], [where[c] for c in causal], covariates)
    z = np.full(len(panel.pos), np.nan)
    z[ok] = z_ok
    a = Answers(ok[pi], ok[pj], r, r_plink2, z, causal, betas, check)
    for r2 in params.clump_r2:
        a.leads[r2], a.clumps[r2] = reference.clump(params, files, extract, ok, z_ok, r2)
    return a


def _reference(
    params: Params, g: np.ndarray, covariates: np.ndarray, i: np.ndarray, j: np.ndarray
) -> dict[int, np.ndarray]:
    return {
        k: reference.reference_r(reference.residual_unit(g, covariates, k), i, j) for k in params.ks
    }


def _covariates(
    params: Params, panel: Panel, g: np.ndarray, pcs: np.ndarray
) -> tuple[dict[str, object], np.ndarray]:
    """The PC Check over the passing variants, as a build of the Test Panel runs it (the module
    selects the common ones, MAF >= 0.05), and the PCs it keeps, in order."""
    ok = panel.passing
    w = pc_check.windows(np.array(panel.chrom)[ok], panel.pos[ok], g[ok])
    checked = pc_check.check(w, pcs, k_target=max(params.ks))
    return checked.report(), pcs[:, [int(n[2:]) - 1 for n in checked.covariates]]
