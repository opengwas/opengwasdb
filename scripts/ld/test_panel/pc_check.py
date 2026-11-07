"""The PC Check: the owner's local-PC rule (7 November 2026; ADR 0002), as a reference.

A PC is **local**, and dropped from the covariates, when one 4 Mb window explains at least 10% of
it, more than a random vector's best window does, **and** the rest of the genome explains at most
0.05 of what that window does (the owner's rule, 7 November 2026). The covariates are the first up
to 10 PCs that are kept; fewer are used if fewer survive, as long as at least 5 do, and
t_far's k is the number used (owner, 7 November 2026). `docs/spec/build-inputs.md` ("The PC
Check") is the authority; this is the numpy implementation the scripts use, and ogld#5's builder
is held to the same numbers.

- **Windows:** 4 Mb, stepped by 2 Mb from each chromosome's first variant, so every stretch of
  up to 2 Mb lies wholly in one. A window's variants are the common ones (MAF >= 0.05) given,
  evenly thinned to at most 5,000; a window with fewer than 5 is skipped.
- **Basis:** the window's top min(5, m) sample-space PCs: eigenvectors of X'X, with X the
  window's genotypes standardised per variant.
- **Share:** the share of a PC's centred variance in the span of a window's basis.
- **Scan, per PC** (names as `ogld.format.PcScan`): `window`, the window with the largest share;
  `window_share`, that share; `rest_share`, the mean share over every window not overlapping it;
  `ratio`, rest_share / window_share; `local`, the verdict.
- **The window-side null:** 200 permutations of the PC's values across samples (seed 405), which
  carry no genotype structure; `window_null_q99` is their 99th percentile of the largest window
  share. At n = 345 a random vector's best window reaches about 0.08-0.10 by chance, so without it
  a small panel's noise PC could be dropped (proto/1-build-inputs@c1c1639,
  results/rule_power.tsv).
- **The ratio** replaced "the rest of the genome explains it no better than chance": a karyotype
  axis that carries a trace of ancestry beats that null (EUR's 17q21.31 score, rest share 0.0073
  against 0.0071) but sits at a ratio of 0.008, while every real PC sits at >= 0.104. The rest
  null's 99th percentile, `rest_null_q99`, is still reported, and decides nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WINDOW_BP = 4_000_000
STEP_BP = 2_000_000
MAX_VARIANTS = 5_000
N_LOCAL = 5
MIN_SHARE = 0.10
MAX_RATIO = 0.05
PERMUTATIONS = 200
SEED = 405
QUANTILE = 0.99
K_TARGET = 10
K_MIN = 5
NULL = {
    "kind": "each PC's values permuted across samples: q99 of the largest window share "
    "(decides) and of the rest share (diagnostic)",
    "permutations": PERMUTATIONS,
    "seed": SEED,
    "quantile": QUANTILE,
}


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


def _shares(w: Windows, vectors: np.ndarray) -> np.ndarray:
    """(windows, vectors): each centred vector's share of variance in each window's basis."""
    v = vectors - vectors.mean(axis=0)
    v = v / np.linalg.norm(v, axis=0)
    p = (w.basis.T @ v).reshape(len(w.where), N_LOCAL, v.shape[1])
    return (p * p).sum(axis=1)


def _scan(w: Windows, pcs: np.ndarray) -> list[dict[str, object]]:
    """One PcScan per PC, in PC File order (PC1 first)."""
    rng = np.random.default_rng(SEED)
    observed = _shares(w, pcs)
    return [_scan_one(w, observed[:, k], pcs[:, k], k, rng) for k in range(pcs.shape[1])]


def is_local(window_share: float, window_null_q99: float, rest_share: float) -> bool:
    """The owner's rule: one window explains >= 10% of the PC, more than chance does, and the
    rest of the genome at most 0.05 of what that window does."""
    return (
        window_share >= MIN_SHARE
        and window_share > window_null_q99
        and rest_share / window_share <= MAX_RATIO
    )


def _scan_one(
    w: Windows, observed: np.ndarray, pc: np.ndarray, k: int, rng: np.random.Generator
) -> dict[str, object]:
    best = int(np.argmax(observed))
    bc, bs = w.where[best]
    rest = np.array([c != bc or abs(s - bs) >= WINDOW_BP for c, s in w.where])
    null = _shares(w, np.stack([pc[rng.permutation(len(pc))] for _ in range(PERMUTATIONS)], axis=1))
    window_q99 = float(np.quantile(null.max(axis=0), QUANTILE))
    rest_q99 = float(np.quantile(null[rest].mean(axis=0), QUANTILE))
    window_share, rest_share = float(observed[best]), float(observed[rest].mean())
    return {
        "pc": f"PC{k + 1}",
        "window": f"{bc}:{bs}-{bs + WINDOW_BP - 1}",
        "window_share": window_share,
        "rest_share": rest_share,
        "ratio": rest_share / window_share,
        "local": is_local(window_share, window_q99, rest_share),
        "window_null_q99": window_q99,
        "rest_null_q99": rest_q99,
    }


def _refusal(s: list[dict[str, object]], kept: int, k_min: int) -> SystemExit:
    """Fewer than `k_min` kept PCs, refused naming the dropped ones (build-inputs.md §8)."""
    local = [str(x["pc"]) for x in s if x["local"]]
    dropped = f"dropped as local: {', '.join(local)}" if local else "none dropped as local"
    return SystemExit(f"the PC Check keeps {kept} PCs, fewer than the minimum {k_min}; {dropped}")


def check(
    w: Windows, pcs: np.ndarray, k_target: int = K_TARGET, k_min: int = K_MIN
) -> dict[str, object]:
    """The PC Check's provenance record (`ogld.format.PcCheck`) and the covariates it keeps:
    the first `k_target` kept PCs, or all of them if fewer survive; fewer than `k_min` is
    refused. `k` is the number used, t_far's k."""
    s = _scan(w, pcs)
    kept = [str(x["pc"]) for x in s if not x["local"]]
    if len(kept) < k_min:
        raise _refusal(s, len(kept), k_min)
    covariates = kept[:k_target]
    return {
        "window_bp": WINDOW_BP,
        "step_bp": STEP_BP,
        "min_share": MIN_SHARE,
        "max_ratio": MAX_RATIO,
        "windows": len(w.where),
        "scan": s,
        "dropped": [str(x["pc"]) for x in s if x["local"]],
        "null": NULL,
        "kept": len(kept),
        "covariates": covariates,
        "k_target": k_target,
        "k_min": k_min,
        "k": len(covariates),
    }
