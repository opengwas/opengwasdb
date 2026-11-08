"""The independent answers: reference r, plink2's r and clumps, and the simulated GWAS.

**Reference r** is the short numpy implementation of PC Adjustment that a builder is tested
against: each passing variant's ALT count, residualised on an intercept and PC1...PCk by least
squares, then correlated. With k = 0 that is Pearson r. Genotypes are complete (ADR 0002), so no
missing-call rule is involved. Signs are ALT-allele: the engine A1 is ogld#3's to form, and
re-signing is the builder's job.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from test_panel.params import Params
from test_panel.tools import plink2


def residual_unit(g: np.ndarray, pcs: np.ndarray, k: int) -> np.ndarray:
    """Rows of `g` (ALT counts): regress out [1, PC1..PCk], scale each row to length 1."""
    if (g < 0).any():
        raise SystemExit("a passing variant has a missing call; genotypes must be complete")
    x = g.astype(np.float64)
    q, _ = np.linalg.qr(np.column_stack([np.ones(g.shape[1]), pcs[:, :k]]))
    e = x - (x @ q) @ q.T
    norm = np.linalg.norm(e, axis=1)
    if not (norm > 1e-9).all():
        raise SystemExit(f"{int((norm <= 1e-9).sum())} passing variants have no variance at k={k}")
    return e / norm[:, None]


def pairs(chrom: np.ndarray, pos: np.ndarray, far_bp: int) -> tuple[np.ndarray, np.ndarray]:
    """Every (i, j), i < j, on one chromosome and at most `far_bp` apart, in row order."""
    ii, jj = [], []
    for i in range(len(pos)):
        j = np.arange(i + 1, len(pos))
        j = j[(chrom[j] == chrom[i]) & (pos[j] - pos[i] <= far_bp)]
        ii.append(np.full(len(j), i))
        jj.append(j)
    return np.concatenate(ii), np.concatenate(jj)


def reference_r(z: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", z[i], z[j])


def _rowid_inputs(panel: dict[str, Path], extract: Path) -> list[str]:
    """plink2's inputs for the passing variants, named by row (`rowid_pvar`)."""
    return [
        "--pgen",
        str(panel["pgen"]),
        "--pvar",
        str(panel["rowid_pvar"]),
        "--psam",
        str(panel["psam"]),
        "--extract",
        str(extract),
    ]


def plink2_r(
    params: Params, panel: dict[str, Path], extract: Path, n: int, i: np.ndarray, j: np.ndarray
) -> np.ndarray:
    """plink2 --r-unphased on the passing variants, float64 square matrix, REF-based sign
    (the same as ALT-based: flipping both variants leaves r unchanged)."""
    out = params.work / "plink2_r"
    plink2(
        params,
        [
            *_rowid_inputs(panel, extract),
            "--r-unphased",
            "square",
            "bin",
            "ref-based",
            "--out",
            str(out),
        ],
        "plink2_r",
    )
    m = np.fromfile(f"{out}.unphased.vcor1.bin", dtype=np.float64).reshape(n, n)
    return m[i, j]


def simulate_gwas(
    params: Params, g: np.ndarray, causal: list[int], pcs: np.ndarray
) -> tuple[np.ndarray, list[float]]:
    """z of every passing variant for one phenotype made from `causal` (rows of `g`).

    Each causal variant explains `h2_each` of the phenotype's variance and the rest is Gaussian
    noise from `seed`. z is the t statistic of the variant in y ~ 1 + PC1..PC10 + variant, as
    plink2 --glm would fit it.
    """
    rng = np.random.default_rng(params.seed)
    n = g.shape[1]
    y = rng.standard_normal(n) * math.sqrt(1 - params.h2_each * len(causal))
    betas = []
    for c, sign in zip(causal, rng.choice([-1.0, 1.0], len(causal)), strict=True):
        x = g[c].astype(np.float64)
        y += sign * math.sqrt(params.h2_each) * (x - x.mean()) / x.std()
        betas.append(float(sign * math.sqrt(params.h2_each) / x.std()))
    covar = np.column_stack([np.ones(n), pcs[:, : params.gwas_pcs]])
    return np.array([_t_stat(g[v], y, covar) for v in range(len(g))]), betas


def _t_stat(gv: np.ndarray, y: np.ndarray, covar: np.ndarray) -> float:
    x = np.column_stack([covar, gv.astype(np.float64)])
    beta, rss, *_ = np.linalg.lstsq(x, y, rcond=None)
    sigma2 = float(rss[0]) / (len(y) - x.shape[1])
    se = math.sqrt(sigma2 * np.linalg.inv(x.T @ x)[-1, -1])
    return float(beta[-1] / se)


def clump(
    params: Params,
    panel: dict[str, Path],
    extract: Path,
    rows: np.ndarray,
    z: np.ndarray,
    r2: float,
) -> tuple[list[int], Path]:
    """plink2 --clump lead rows, k = 0, on the passing variants; p = erfc(|z| / sqrt 2).

    `--clump-unphased`: plink2 clumps on phased r^2 when the pgen holds phase, as v2's does, but
    an LD Store's r is unphased (ADR 0002), and #6 is held to equal these leads."""
    sumstats = params.work / "gwas.sumstats"
    sumstats.write_text(
        "ID\tP\n"
        + "".join(
            f"r{r}\t{math.erfc(abs(v) / math.sqrt(2))!r}\n" for r, v in zip(rows, z, strict=True)
        )
    )
    out = params.work / f"clump_r2_{r2}"
    plink2(
        params,
        [
            *_rowid_inputs(panel, extract),
            "--clump",
            str(sumstats),
            "--clump-p1",
            str(params.clump_p1),
            "--clump-p2",
            str(params.clump_p1),
            "--clump-r2",
            str(r2),
            "--clump-kb",
            str(params.clump_kb),
            "--clump-unphased",
            "--clump-id-field",
            "ID",
            "--clump-p-field",
            "P",
            "--out",
            str(out),
        ],
        f"clump_r2_{r2}",
    )
    clumps = Path(f"{out}.clumps")
    lines = clumps.read_text().splitlines()
    col = lines[0].lstrip("#").split("\t").index("ID")
    return [int(x.split("\t")[col][1:]) for x in lines[1:]], clumps
