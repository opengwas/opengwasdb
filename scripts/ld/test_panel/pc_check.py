"""The PC Check: the owner's local-PC rule (7 November 2026; ADR 0002), as a reference.

A PC is **local**, and dropped from the covariates, only if one 4 Mb window explains at least 10%
of it **and** the rest of the genome explains it no better than chance. The covariates are the
first k PCs that are kept. `docs/spec/build-inputs.md` ("The PC Check") is the authority; this
is the numpy implementation the scripts use, and ogld#5's builder is held to the same numbers.

- **Windows:** 4 Mb, stepped by 2 Mb from each chromosome's first variant, so every stretch of
  up to 2 Mb lies wholly in one. A window's variants are the common ones (MAF >= 0.05) given,
  evenly thinned to at most 5,000; a window with fewer than 5 is skipped.
- **Basis:** the window's top min(5, m) sample-space PCs: eigenvectors of X'X, with X the
  window's genotypes standardised per variant.
- **Share:** the share of a PC's centred variance in the span of a window's basis.
- **Scan, per PC** (names as `ogld.format.PcScan`): `window`, the window with the largest share;
  `window_share`, that share; `rest_share`, the mean share over every window not overlapping it;
  `local`, the verdict.
- **Null:** the same `rest_share` for 200 permutations of the PC's values across samples (seed
  405). The rest of the genome explains a PC "no better than chance" when its `rest_share` is at
  most the permutations' 99th percentile, recorded per PC as `rest_null_q99`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WINDOW_BP = 4_000_000
STEP_BP = 2_000_000
MAX_VARIANTS = 5_000
N_LOCAL = 5
MIN_SHARE = 0.10
PERMUTATIONS = 200
SEED = 405
QUANTILE = 0.99
NULL = {"kind": "rest_share of each PC's values permuted across samples",
        "permutations": float(PERMUTATIONS), "seed": float(SEED), "quantile": QUANTILE}


@dataclass(frozen=True)
class Windows:
    """Every scanned window's (chromosome, start) and its basis, stacked (samples, 5 x windows)."""

    where: list[tuple[str, int]]
    basis: np.ndarray


def windows(chrom: np.ndarray, pos: np.ndarray, g: np.ndarray) -> Windows:
    """Scan windows over common variants `g` (variants x samples, complete), in position order."""
    where: list[tuple[str, int]] = []
    blocks: list[np.ndarray] = []
    for c in dict.fromkeys(chrom.tolist()):
        on = np.flatnonzero(chrom == c)
        for start in range(int(pos[on[0]]), int(pos[on[-1]]) + 1, STEP_BP):
            rows = on[(pos[on] >= start) & (pos[on] < start + WINDOW_BP)]
            if len(rows) >= N_LOCAL:
                where.append((str(c), start))
                blocks.append(_basis(g[rows]))
    return Windows(where, np.concatenate(blocks, axis=1))


def _basis(g: np.ndarray) -> np.ndarray:
    keep = np.unique(np.linspace(0, len(g) - 1, min(MAX_VARIANTS, len(g))).astype(int))
    x = g[keep].astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    sd = np.sqrt((x * x).sum(axis=1))
    x = x[sd > 0] / sd[sd > 0, None]
    _, v = np.linalg.eigh(x.T @ x)
    top = v[:, ::-1][:, : min(N_LOCAL, len(x))]
    return np.pad(top, ((0, 0), (0, N_LOCAL - top.shape[1])))


def shares(w: Windows, vectors: np.ndarray) -> np.ndarray:
    """(windows, vectors): each centred vector's share of variance in each window's basis."""
    v = vectors - vectors.mean(axis=0)
    v = v / np.linalg.norm(v, axis=0)
    p = (w.basis.T @ v).reshape(len(w.where), N_LOCAL, v.shape[1])
    return (p * p).sum(axis=1)


def scan(w: Windows, pcs: np.ndarray) -> list[dict[str, object]]:
    """One PcScan per PC, in PC File order (PC1 first)."""
    rng = np.random.default_rng(SEED)
    observed = shares(w, pcs)
    out = []
    for k in range(pcs.shape[1]):
        best = int(np.argmax(observed[:, k]))
        bc, bs = w.where[best]
        rest = np.array([c != bc or abs(s - bs) >= WINDOW_BP for c, s in w.where])
        perm = np.stack([pcs[rng.permutation(len(pcs)), k] for _ in range(PERMUTATIONS)], axis=1)
        q99 = float(np.quantile(shares(w, perm)[rest].mean(axis=0), QUANTILE))
        window_share = float(observed[best, k])
        rest_share = float(observed[rest, k].mean())
        out.append({"pc": f"PC{k + 1}", "window": f"{bc}:{bs}-{bs + WINDOW_BP - 1}",
                    "window_share": window_share, "rest_share": rest_share,
                    "local": window_share >= MIN_SHARE and rest_share <= q99,
                    "rest_null_q99": q99})
    return out


def check(w: Windows, pcs: np.ndarray, k: int) -> dict[str, object]:
    """The PC Check's provenance record (`ogld.format.PcCheck`) and the covariates it keeps."""
    s = scan(w, pcs)
    kept = [str(x["pc"]) for x in s if not x["local"]]
    if len(kept) < k:
        raise SystemExit(f"the PC Check keeps {len(kept)} PCs, fewer than k = {k}")
    return {"window_bp": WINDOW_BP, "step_bp": STEP_BP, "min_share": MIN_SHARE,
            "windows": len(w.where), "scan": s, "dropped": [str(x["pc"]) for x in s if x["local"]],
            "null": NULL, "covariates": kept[:k]}
