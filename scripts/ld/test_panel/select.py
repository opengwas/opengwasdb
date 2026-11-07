"""Which records of the converted chromosomes the Test Panel holds, by component.

Every rule is deterministic: windows and anchors are parameters, and every choice among
candidates takes the first by position (or the strongest, then the first). A component maps
to the converted chromosome's record rows, so the README can say where each edge case is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from test_panel import pcs as pc_file
from test_panel import reference
from test_panel.params import Params
from test_panel.sites import Sites
from test_panel.tools import export_rows

Components = dict[str, list[int]]


def select(
    params: Params, s: Sites, reason: np.ndarray, rowid_pvar: Path, covariates: list[int]
) -> Components:
    """The chosen rows by component. `covariates` are the 0-based PC File columns the reference
    regresses out at k = 10, so a Far Pair chosen here is one in the reference."""
    ok = reason == ""
    common_full = ok & (s.maf >= params.common_maf) & (s.pos >= params.q_arm_start[s.chrom])
    c: Components = {
        "proxy_cluster": _window(s, ok, params.proxy_window[s.chrom], params.proxy_max),
        "gwas_region": _window(s, ok, params.gwas_window[s.chrom], params.gwas_max[s.chrom]),
        "backbone": _backbone(s, common_full, params),
        "causal": _causal(s, common_full, params),
    }
    c["far_pair_k10"] = _far_pairs(params, s, common_full, rowid_pvar, covariates)
    if s.chrom == params.edge_chrom:
        c.update(_edge_cases(params, s, ok, common_full, reason))
    return c


def _window(s: Sites, ok: np.ndarray, window: tuple[int, int], cap: int) -> list[int]:
    rows = np.flatnonzero(ok & (s.pos >= window[0]) & (s.pos <= window[1]))
    if len(rows) > cap:
        rows = rows[np.unique(np.linspace(0, len(rows) - 1, cap).round().astype(int))]
    return rows.tolist()


def _nearest(s: Sites, eligible: np.ndarray, target: int) -> int:
    rows = np.flatnonzero(eligible)
    return int(rows[np.argmin(np.abs(s.pos[rows] - target))])


def _backbone(s: Sites, eligible: np.ndarray, params: Params) -> list[int]:
    lo, hi = params.q_arm_start[s.chrom], int(s.pos[eligible].max())
    grid = range(lo, hi, params.backbone_bp)
    return sorted({_nearest(s, eligible, g) for g in grid})


def _causal(s: Sites, common_full: np.ndarray, params: Params) -> list[int]:
    lo, hi = params.gwas_window[s.chrom]
    eligible = common_full & (s.maf >= params.causal_min_maf) & (s.pos >= lo) & (s.pos <= hi)
    return [_nearest(s, eligible, a) for a in params.causal_anchor[s.chrom]]


def _far_pairs(
    params: Params, s: Sites, common: np.ndarray, rowid_pvar: Path, covariates: list[int]
) -> list[int]:
    """Real Far Pairs at k = 10: the strongest 3-10 Mb pairs with Stored r^2 >= t_far among
    common variants thinned to one per `far_search_thin_bp`."""
    got, fam, g = export_rows(
        params,
        s.prefix,
        rowid_pvar,
        _thin(s.pos, common, params.far_search_thin_bp),
        params.work / f"far_search.chr{s.chrom}",
    )
    cov = pc_file.load(params, fam)[:, covariates]
    z = reference.residual_unit(g, cov, len(covariates))
    hits = _strong_far(z, s.pos[got], params.t_far(len(fam), 10), params.far_bp)
    chosen: list[int] = []
    for _, a, b in hits:
        pair = [got[a], got[b]]
        if len(chosen) < 2 * params.far_pairs_per_chrom and not set(pair) & set(chosen):
            chosen += pair
    if not chosen:
        raise SystemExit(f"chr{s.chrom}: no 3-10 Mb pair reaches t_far at k = 10")
    return chosen


def _thin(pos: np.ndarray, eligible: np.ndarray, bp: int) -> list[int]:
    rows, last = [], -(10**12)
    for r in np.flatnonzero(eligible):
        if pos[r] - last >= bp:
            rows.append(int(r))
            last = int(pos[r])
    return rows


def _strong_far(
    z: np.ndarray, pos: np.ndarray, t_far: float, far_bp: int
) -> list[tuple[float, int, int]]:
    """(-Stored r^2, a, b) for every 3 Mb < d <= far_bp pair reaching t_far, strongest first."""
    hits: list[tuple[float, int, int]] = []
    for a in range(len(pos)):
        lo = int(np.searchsorted(pos, pos[a] + 3_000_001))
        hi = int(np.searchsorted(pos, pos[a] + far_bp, side="right"))
        q2 = (np.rint((z[lo:hi] @ z[a]) * 32767) / 32767) ** 2
        hits += [(-float(q2[b]), a, lo + int(b)) for b in np.flatnonzero(q2 >= t_far)]
    return sorted(hits)


def _edge_cases(
    params: Params, s: Sites, ok: np.ndarray, common_full: np.ndarray, reason: np.ndarray
) -> Components:
    start = params.q_arm_start[s.chrom]
    c: Components = {f"exact_{d}": _exact(s, common_full, d) for d in params.exact_distances}
    c["multiallelic"] = _multiallelic(s, ok & (s.pos >= start) & _is_snv(s))
    lengths = np.char.str_len(s.ref.astype(str)) + np.char.str_len(s.alt.astype(str))
    c["long_indel"] = [
        int(np.flatnonzero(ok & (s.pos >= start))[np.argmax(lengths[ok & (s.pos >= start)])])
    ]
    c.update(_failing(params, s, reason, start))
    return c


def _exact(s: Sites, eligible: np.ndarray, d: int) -> list[int]:
    rows = np.flatnonzero(eligible)
    at = {int(s.pos[r]): int(r) for r in rows[::-1]}
    for r in rows:
        partner = at.get(int(s.pos[r]) + d)
        if partner is not None:
            return [int(r), partner]
    raise SystemExit(f"chr{s.chrom}: no passing pair exactly {d} bp apart")


def _is_snv(s: Sites) -> np.ndarray:
    return (np.char.str_len(s.ref.astype(str)) == 1) & (np.char.str_len(s.alt.astype(str)) == 1)


def _multiallelic(s: Sites, eligible: np.ndarray) -> list[int]:
    """The first position with exactly two passing SNV records: one REF, two ALTs, as splitting
    an A>C,G record gives."""
    rows = np.flatnonzero(eligible)
    positions, counts = np.unique(s.pos[rows], return_counts=True)
    first = positions[np.flatnonzero(counts == 2)[0]]
    return rows[s.pos[rows] == first].tolist()


def _failing(params: Params, s: Sites, reason: np.ndarray, start: int) -> Components:
    """Records each filter removes: the first of each MAC value, the first of each non-PASS
    FILTER value and the first site with no genotype variance. v2 has no FILTER values and, at
    MAC >= 20, no site without variance, so those two come back empty (README)."""
    after = s.pos >= start
    c: Components = {
        "fail_mac": [
            int(np.flatnonzero(after & (reason == "mac") & (s.mac == m))[0])
            for m in params.fail_mac_values
        ]
    }
    f_rows = np.flatnonzero(after & (reason == "filter") & (s.mac >= params.mac_min))
    _, first = np.unique(s.filt[f_rows], return_index=True)
    c["fail_filter"] = sorted(f_rows[first].tolist())[: params.fail_filters_max]
    zv = np.flatnonzero(after & s.zero_variance & (s.mac >= params.mac_min))
    c["zero_variance"] = zv[:1].tolist()
    return c
