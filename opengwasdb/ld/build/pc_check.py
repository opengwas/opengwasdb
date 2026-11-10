"""The PC Check: the owner's local-PC rule (7 November 2026; ADR 0002, decision 5).

`docs/spec/build-inputs.md` §8 is the authority, and this is its one implementation: the
builder runs it on every PC File, and `scripts/make_pcs.py` and the Test Panel's script import
it (ogld#5 adopted lane 2d's reference, `scripts/test_panel/pc_check.py`, rather than keep two).

A PC is **local**, and dropped from the covariates, when all three hold: one 4 Mb window
explains at least 10% of it; that share beats the window-side null, the 99th percentile of the
largest window share of the PC permuted across samples (200 permutations, seed 405); and the
rest of the genome explains at most 0.05 of what that window does (`ratio`). The ratio replaced
"the rest of the genome explains it no better than chance", which kept EUR's 17q21.31 karyotype
axis: its rest share (0.0073) beat its rest null (0.0071), yet its ratio is 0.008, while every
real PC sits at ≥ 0.104 (`proto/1-build-inputs@c1c1639`, `results/rule_power.tsv`). The rest
null's 99th percentile is still reported, and decides nothing.

- **Variants:** the Variant Set's variants with MAF ≥ 0.05 on the kept samples; `windows`
  selects them, so no caller can scan another set.
- **Windows:** 4 Mb, stepped by 2 Mb from each chromosome's first such variant, so every stretch
  of up to 2 Mb lies wholly in one; thinned evenly to at most 5,000 variants; a window with
  fewer than 5 is skipped.
- **Basis:** a window's top min(5, m) sample-space principal components of its genotypes,
  standardised per variant.
- **Share:** the fraction of a PC's centred variance in the span of a window's basis.

The covariates are the first up to `k_target` (10) PCs kept, in order; fewer are used if fewer
survive, as long as at least `k_min` (5) do, and the build is refused otherwise, naming the
dropped PCs (owner, 7 November 2026). The rule needs a few hundred samples: by chance a window
holds about 5 / n of any vector's variance, so below about 100 samples no ratio reaches 0.05.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from ogld.format import InputError, PcCheck, PcNull, PcScan

WINDOW_BP: Final = 4_000_000
STEP_BP: Final = 2_000_000
MAX_VARIANTS: Final = 5_000
N_LOCAL: Final = 5
MIN_SHARE: Final = 0.10
MAX_RATIO: Final = 0.05
PERMUTATIONS: Final = 200
SEED: Final = 405
QUANTILE: Final = 0.99
COMMON_MAF: Final = 0.05
K_TARGET: Final = 10
K_MIN: Final = 5
NULL: Final[PcNull] = {
    "kind": "each PC's values permuted across samples: q99 of the largest window share "
    "(decides) and of the rest share (diagnostic)",
    "permutations": PERMUTATIONS,
    "seed": SEED,
    "quantile": QUANTILE,
}
#: An eigenvalue this far below a window's largest spans no genotype direction: a window of
#: fewer than 5 independent variants has fewer than 5 components, not arbitrary extra ones.
_RANK_TOLERANCE: Final = 1e-9


@dataclass(frozen=True)
class Windows:
    """Every scanned window's (chromosome, start), and their bases stacked side by side:
    `basis` is (samples, N_LOCAL × windows), window i's in columns 5i … 5i + 4."""

    where: list[tuple[str, int]]
    basis: npt.NDArray[np.float64]


@dataclass(frozen=True)
class Checked:
    """The PC Check's provenance record and the covariates it chose (build-inputs §8).

    `covariates` are the first up to `k_target` kept PCs, in PC File order; `kept` counts every
    PC the rule kept, of which `k`, the number used, is t_far's k.
    """

    record: PcCheck
    covariates: list[str]
    kept: int
    k_target: int
    k_min: int

    @property
    def k(self) -> int:
        return len(self.covariates)

    def report(self) -> dict[str, object]:
        """The record and the choice as one JSON object, as `scripts/make_pcs.py` reports it
        and the Test Panel's manifest holds it (`pc_check`)."""
        r = self.record
        return {
            "window_bp": r["window_bp"],
            "step_bp": r["step_bp"],
            "min_share": r["min_share"],
            "max_ratio": r["max_ratio"],
            "windows": r["windows"],
            "scan": [dict(s) for s in r["scan"]],
            "dropped": list(r["dropped"]),
            "null": dict(r["null"]),
            "kept": self.kept,
            "covariates": list(self.covariates),
            "k_target": self.k_target,
            "k_min": self.k_min,
            "k": self.k,
        }


def common(dosages: npt.NDArray[np.floating] | npt.NDArray[np.integer]) -> npt.NDArray[np.bool_]:
    """Which rows of ALT dosages (variants × samples) have MAF ≥ 0.05 on those samples."""
    frequency = dosages.mean(axis=1) / 2
    selected: npt.NDArray[np.bool_] = np.minimum(frequency, 1 - frequency) >= COMMON_MAF
    return selected


def windows(
    chrom: npt.NDArray[np.str_],
    pos: npt.NDArray[np.int64],
    g: npt.NDArray[np.floating] | npt.NDArray[np.integer],
    *,
    threads: int = 1,
) -> Windows:
    """The scan's windows over variants `g` (variants × samples, complete), in position order
    within each chromosome. Only the common ones are used (MAF ≥ 0.05, `common`)."""
    keep = common(g)
    chrom, pos, g = chrom[keep], pos[keep], g[keep]
    where: list[tuple[str, int]] = []
    rows: list[npt.NDArray[np.intp]] = []
    for c in dict.fromkeys(chrom.tolist()):
        on = np.flatnonzero(chrom == c)
        for start in range(int(pos[on[0]]), int(pos[on[-1]]) + 1, STEP_BP):
            inside = on[(pos[on] >= start) & (pos[on] < start + WINDOW_BP)]
            if len(inside) >= N_LOCAL:
                where.append((str(c), start))
                rows.append(inside)
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        blocks = list(pool.map(lambda inside: _basis(g[inside]), rows))
    samples = g.shape[1]
    basis = np.concatenate(blocks, axis=1) if blocks else np.empty((samples, 0))
    return Windows(where, basis)


def merge(parts: Sequence[Windows]) -> Windows:
    """One scan's windows from per-chromosome parts, in the order given."""
    where = [x for part in parts for x in part.where]
    if not parts:
        return Windows(where, np.empty((0, 0)))
    return Windows(where, np.concatenate([part.basis for part in parts], axis=1))


def _basis(g: npt.NDArray[np.floating] | npt.NDArray[np.integer]) -> npt.NDArray[np.float64]:
    """A window's top min(5, rank) sample-space components, padded with zero columns to 5."""
    picked = np.unique(np.linspace(0, len(g) - 1, min(MAX_VARIANTS, len(g))).astype(int))
    x = g[picked].astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    sd = np.sqrt((x * x).sum(axis=1))
    x = x[sd > 0] / sd[sd > 0, None]
    if len(x) >= x.shape[1]:
        values, vectors = np.linalg.eigh(x.T @ x)
        values, vectors = values[::-1][:N_LOCAL], vectors[:, ::-1][:, :N_LOCAL]
    else:
        # Fewer variants than samples: the same components from the small Gram matrix, so a
        # window of 40 variants costs a 40 × 40 eigenproblem rather than a 771 × 771 one.
        values, small = np.linalg.eigh(x @ x.T)
        values, small = values[::-1][:N_LOCAL], small[:, ::-1][:, :N_LOCAL]
        vectors = x.T @ small / np.sqrt(np.maximum(values, np.finfo(np.float64).tiny))
    top = vectors[:, values > _RANK_TOLERANCE * max(float(values[0]), 0.0)]
    return np.pad(top, ((0, 0), (0, N_LOCAL - top.shape[1])))


def _shares(w: Windows, vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(windows, vectors): each centred vector's share of its variance in each window's basis."""
    v = vectors - vectors.mean(axis=0)
    v = v / np.linalg.norm(v, axis=0)
    p = (w.basis.T @ v).reshape(len(w.where), N_LOCAL, v.shape[1])
    shares: npt.NDArray[np.float64] = (p * p).sum(axis=1)
    return shares


def is_local(window_share: float, window_null_q99: float, rest_share: float) -> bool:
    """The owner's rule: one window explains ≥ 10% of the PC, more than chance does, and the
    rest of the genome at most 0.05 of what that window does."""
    return (
        window_share >= MIN_SHARE
        and window_share > window_null_q99
        and rest_share / window_share <= MAX_RATIO
    )


def _scan_one(
    w: Windows,
    observed: npt.NDArray[np.float64],
    pc: npt.NDArray[np.float64],
    name: str,
    rng: np.random.Generator,
) -> PcScan:
    best = int(np.argmax(observed))
    bc, bs = w.where[best]
    rest = _apart(w, best, name)
    permuted = np.stack([pc[rng.permutation(len(pc))] for _ in range(PERMUTATIONS)], axis=1)
    null = _shares(w, permuted)
    window_q99 = float(np.quantile(null.max(axis=0), QUANTILE))
    rest_q99 = float(np.quantile(null[rest].mean(axis=0), QUANTILE))
    window_share, rest_share = float(observed[best]), float(observed[rest].mean())
    return {
        "pc": name,
        "window": f"{bc}:{bs}-{bs + WINDOW_BP - 1}",
        "window_share": window_share,
        "rest_share": rest_share,
        "ratio": rest_share / window_share,
        "local": is_local(window_share, window_q99, rest_share),
        "window_null_q99": window_q99,
        "rest_null_q99": rest_q99,
    }


def _refusal(scan: list[PcScan], kept: int, k_min: int) -> InputError:
    """Fewer than `k_min` kept PCs, refused naming the dropped ones (build-inputs §8)."""
    local = [s["pc"] for s in scan if s["local"]]
    dropped = f"dropped as local: {', '.join(local)}" if local else "none dropped as local"
    return InputError(f"the PC Check keeps {kept} PCs, fewer than the minimum {k_min}; {dropped}")


def _require_windows(w: Windows) -> None:
    """A scan needs at least one window, or no PC has a share to judge."""
    if not w.where:
        raise InputError(
            "the PC Check found no 4 Mb window with 5 common variants (MAF ≥ 0.05), "
            "so it cannot judge a PC (build-inputs §8)"
        )


def _apart(w: Windows, best: int, name: str) -> npt.NDArray[np.bool_]:
    """The windows that do not overlap window `best`: the rest share's. None is refused, since
    a mean over no windows is undefined, not zero."""
    bc, bs = w.where[best]
    rest = np.array([c != bc or abs(s - bs) >= WINDOW_BP for c, s in w.where])
    if not rest.any():
        raise InputError(
            f"every window of the PC Check overlaps {name}'s best one, {bc}:{bs}, so its rest "
            "share is undefined: the input spans too little genome (build-inputs §8)"
        )
    return rest


def check(
    w: Windows,
    pcs: npt.NDArray[np.float64],
    *,
    names: Sequence[str] | None = None,
    k_target: int = K_TARGET,
    k_min: int = K_MIN,
) -> Checked:
    """Scan every PC (the columns of `pcs`, samples in the order of `w.basis`'s rows) and choose
    the covariates: the first `k_target` kept PCs, or all of them if fewer survive. Fewer than
    `k_min` is refused, naming the dropped PCs. `names` are the PCs' names, `PC1` … by default.
    """
    _require_windows(w)
    labels = list(names) if names is not None else [f"PC{i + 1}" for i in range(pcs.shape[1])]
    rng = np.random.default_rng(SEED)
    observed = _shares(w, pcs)
    scan = [_scan_one(w, observed[:, i], pcs[:, i], labels[i], rng) for i in range(len(labels))]
    kept = [s["pc"] for s in scan if not s["local"]]
    if len(kept) < k_min:
        raise _refusal(scan, len(kept), k_min)
    record: PcCheck = {
        "window_bp": WINDOW_BP,
        "step_bp": STEP_BP,
        "windows": len(w.where),
        "min_share": MIN_SHARE,
        "max_ratio": MAX_RATIO,
        "scan": scan,
        "dropped": [s["pc"] for s in scan if s["local"]],
        "null": NULL,
    }
    return Checked(record, kept[:k_target], len(kept), k_target, k_min)


__all__ = [
    "COMMON_MAF",
    "K_MIN",
    "K_TARGET",
    "NULL",
    "Checked",
    "Windows",
    "check",
    "common",
    "is_local",
    "merge",
    "windows",
]
