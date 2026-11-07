"""Where each edge case is: the rows and values the README lists and the tests check."""

from __future__ import annotations

import numpy as np

from test_panel.answers import Answers
from test_panel.assemble import Panel
from test_panel.params import Params

SCALE = 32767


def edge_cases(params: Params, panel: Panel, a: Answers, n: int) -> dict[str, object]:
    d = panel.pos[a.j] - panel.pos[a.i]
    out: dict[str, object] = {"clusters": _clusters(panel, a), "exact": _exact(panel, a, d)}
    out["far_pairs_k10"] = _far(params, panel, a, d, n)
    out["beyond_far_bp"] = _beyond(params, panel)
    out["multiallelic"] = _rows_with(panel, "multiallelic", a)
    out["long_indel"] = _long_indel(panel)
    out["failing"] = {reason: [_row(panel, r) for r in np.flatnonzero(panel.reason == reason)]
                      for reason in ("filter", "mac", "zero_variance")}
    out["zero_variance_site"] = [_row(panel, r) for r, c in enumerate(panel.components)
                                 if "zero_variance" in c]
    out["gwas"] = {"causal_rows": a.causal, "betas": a.betas,
                   "max_abs_z": float(np.nanmax(np.abs(a.z))),
                   "leads": {str(k): v for k, v in a.leads.items()}}
    return out


def _row(panel: Panel, r: int) -> dict[str, object]:
    return {"row": int(r), "chrom": panel.chrom[r], "pos": int(panel.pos[r]),
            "ref": panel.ref[r], "alt": panel.alt[r], "filter": panel.filt[r],
            "mac": int(panel.mac[r]), "drop_reason": str(panel.reason[r])}


def _pair_value(a: Answers, i: int, j: int) -> dict[str, object] | None:
    hit = np.flatnonzero((a.i == i) & (a.j == j))
    if not len(hit):
        return None
    return {f"r_k{k}": float(v[hit[0]]) for k, v in a.r.items()}


def _clusters(panel: Panel, a: Answers) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in ("proxy_cluster", "gwas_region", "backbone"):
        for chrom in sorted(set(panel.chrom)):
            rows = [r for r, c in enumerate(panel.components)
                    if name in c and panel.chrom[r] == chrom]
            inside = np.isin(a.i, rows) & np.isin(a.j, rows)
            out[f"{name}_chr{chrom}"] = {
                "first_row": rows[0], "last_row": rows[-1], "rows": len(rows),
                "pairs": int(inside.sum()),
                "pairs_r2_ge_0.8_k0": int((inside & (a.r[0] ** 2 >= 0.8)).sum()),
                "pairs_r2_ge_0.8_k10": int((inside & (a.r[10] ** 2 >= 0.8)).sum())}
    return out


def _exact(panel: Panel, a: Answers, d: np.ndarray) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in [c for c in sorted({x for cs in panel.components for x in cs})
                 if c.startswith("exact_")]:
        i, j = (r for r, c in enumerate(panel.components) if name in c)
        out[name] = {"rows": [i, j], "positions": [int(panel.pos[i]), int(panel.pos[j])],
                     "distance": int(panel.pos[j] - panel.pos[i]),
                     "in_reference": _pair_value(a, i, j)}
    return out


def _far(params: Params, panel: Panel, a: Answers, d: np.ndarray, n: int) -> dict[str, object]:
    t_far = params.t_far(n, 10)
    code = np.rint(a.r[10] * SCALE)
    band = (d > 3_000_000) & (d <= params.far_bp)
    far = band & ((code / SCALE) ** 2 >= t_far)
    named = [(int(x), int(y)) for x, y in zip(a.i[far], a.j[far], strict=True)
             if "far_pair_k10" in panel.components[x] and "far_pair_k10" in panel.components[y]]
    bound = np.flatnonzero(band & ~far)
    return {"t_far_k10": t_far, "far_pairs_k10": int(far.sum()),
            "far_pairs_named": [{"rows": [x, y], "distance": int(panel.pos[y] - panel.pos[x]),
                                 **(_pair_value(a, x, y) or {})} for x, y in named],
            "bound_pairs_k10": len(bound),
            "first_bound_pair": {"rows": [int(a.i[bound[0]]), int(a.j[bound[0]])],
                                 "distance": int(d[bound[0]]),
                                 "r_k10": float(a.r[10][bound[0]])}}


def _beyond(params: Params, panel: Panel) -> dict[str, object]:
    ok = panel.passing
    chrom = np.array(panel.chrom)[ok]
    pos = panel.pos[ok]
    i, j = np.triu_indices(len(ok), k=1)
    same = (chrom[i] == chrom[j]) & (pos[j] - pos[i] > params.far_bp)
    cross = chrom[i] != chrom[j]
    first_same, first_cross = np.flatnonzero(same)[0], np.flatnonzero(cross)[0]
    return {"same_chromosome_pairs_over_far_bp": int(same.sum()),
            "first_same_chromosome": [int(ok[i[first_same]]), int(ok[j[first_same]])],
            "cross_chromosome_pairs": int(cross.sum()),
            "first_cross_chromosome": [int(ok[i[first_cross]]), int(ok[j[first_cross]])]}


def _rows_with(panel: Panel, name: str, a: Answers) -> dict[str, object]:
    rows = [r for r, c in enumerate(panel.components) if name in c]
    return {"rows": rows, "positions": [int(panel.pos[r]) for r in rows],
            "alleles": [[panel.ref[r], panel.alt[r]] for r in rows],
            "pair_at_distance_0": _pair_value(a, rows[0], rows[1])}


def _long_indel(panel: Panel) -> dict[str, object]:
    r = next(r for r, c in enumerate(panel.components) if "long_indel" in c)
    text = f"{panel.chrom[r]}:{panel.pos[r]}:{panel.ref[r]}:{panel.alt[r]}"
    return {**_row(panel, r), "ref_length": len(panel.ref[r]), "alt_length": len(panel.alt[r]),
            "chrom_pos_ref_alt_bytes": len(text.encode())}
