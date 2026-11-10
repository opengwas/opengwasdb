"""Greedy clumping on an LD Store, equal to `plink2 --clump` (spec §9; ogld#6)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ogld.format import Refusal, RefusalReason, judge_request
from ogld.query.results import Clumps, genomic_key
from ogld.store import StoreReader


def clump(
    reader: StoreReader, alids: Sequence[str], score: npt.ArrayLike, *, r2: float, kb: float
) -> Clumps:
    """Clump the present `alids` by `score`, strongest first (ogld#6).

    `score` is |z| or −log10 p, one per ALID, finite and ≥ 0; never raw p, which
    underflows to 0 past |z| ≈ 38. Judged by `ogld.format.judge_request` first. Pairs
    come from `StoreReader.pair_r`; an unstored pair within an answered window is the
    Bound, which cannot reach an answered r2.

    Greedy: the strongest unclumped variant leads, and every unclumped variant within
    the window with r̂² ≥ r2 of it joins it. Because leads are taken strongest first, a
    member joins the strongest lead it reaches. Ties go to genomic order, then ALID.
    """
    p = reader.provenance
    decision = judge_request(
        r2=r2, kb=kb, t_far=p["t_far"], near_bp=p["near_bp"], far_bp=p["far_bp"]
    )
    decision.raise_if_refused()
    window = decision.window_bp
    scores = _checked_score(score, len(alids))

    found = reader.rows_of(alids)
    present = found >= 0
    absent = tuple(alid for alid, ok in zip(alids, present, strict=True) if not ok)
    rows, scores = found[present], scores[present]
    variants = reader.variants(rows)
    n = len(rows)

    # Strongest first; ties by genomic order, then ALID.
    keys = [
        (-scores[i], *genomic_key(variants.chrom[i], variants.position[i], variants.alid[i]))
        for i in range(n)
    ]
    order = sorted(range(n), key=keys.__getitem__)

    lead = np.full(n, -1, dtype=np.int64)
    r_lead = np.full(n, np.nan)
    by_chrom = {c: np.flatnonzero(variants.chrom == c) for c in dict.fromkeys(variants.chrom)}
    pair = {c: reader.pair_r(rows[members]) for c, members in by_chrom.items()}
    where = {c: {int(m): k for k, m in enumerate(members)} for c, members in by_chrom.items()}
    for i in order:
        if lead[i] >= 0:
            continue
        chrom = variants.chrom[i]
        members = by_chrom[chrom]
        r = pair[chrom][where[chrom][i]]
        near = np.abs(variants.position[members] - variants.position[i]) <= window
        joins = near & (r * r >= r2) & (lead[members] < 0)
        lead[members[joins]] = i
        r_lead[members[joins]] = r[joins]
        lead[i], r_lead[i] = i, 1.0
    return Clumps(
        alid=variants.alid,
        chrom=variants.chrom,
        position=variants.position,
        score=scores,
        lead=variants.alid[lead],
        r=r_lead,
        absent=absent,
        r2=r2,
        window_bp=window,
    )


def _checked_score(score: npt.ArrayLike, expected: int) -> npt.NDArray[np.float64]:
    """`score` as floats, if there is one finite, non-negative number per ALID; else refused."""
    try:
        values = np.asarray(score, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise Refusal(RefusalReason.INVALID, f"a score must be numbers: {error}") from error
    if values.ndim != 1 or len(values) != expected:
        raise Refusal(
            RefusalReason.INVALID,
            f"a score is one number per ALID: {expected} ALIDs, score of shape {values.shape}",
        )
    if not (np.isfinite(values).all() and (values >= 0).all()):
        raise Refusal(RefusalReason.INVALID, "a score must be finite and ≥ 0 (|z| or −log10 p)")
    return values
