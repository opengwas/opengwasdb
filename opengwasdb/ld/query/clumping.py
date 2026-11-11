"""Greedy clumping on an LD Store, equal to `plink2 --clump` (spec §9; ogld#6)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ogld.format import Refusal, RefusalReason
from ogld.query.requests import window_of
from ogld.query.results import Clumps, strongest_first
from ogld.store import StoreReader, Variants


def clump(
    reader: StoreReader, alids: Sequence[str], score: npt.ArrayLike, *, r2: float, kb: float
) -> Clumps:
    """Clump the present `alids` by `score`, strongest first (ogld#6).

    `score` is |z| or −log10 p, one per ALID, finite and ≥ 0; never raw p, which
    underflows to 0 past |z| ≈ 38. Judged by `ogld.format.judge_request` first. Pairs
    come from `StoreReader.pair_r`; an unstored pair within an answered window is the
    Bound, which cannot reach an answered r2.

    Greedy: the strongest unclumped variant leads, and every unclumped variant within
    the window (distance < W) with r̂² > r2 of it joins it, as `plink2 --clump` does (spec
    §9). Because leads are taken strongest first, a member joins the strongest lead it
    reaches. Ties go to genomic order, then ALID.
    """
    window = window_of(reader, r2=r2, kb=kb)
    scores = _checked_score(score, len(alids))

    found = reader.rows_of(alids)
    present = found >= 0
    absent = tuple(alid for alid, ok in zip(alids, present, strict=True) if not ok)
    rows, scores = found[present], scores[present]
    variants = reader.variants(rows)

    lead, r_lead = _greedy(reader, rows, variants, scores, r2=r2, window=window)
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


def _greedy(
    reader: StoreReader,
    rows: npt.NDArray[np.int64],
    variants: Variants,
    scores: npt.NDArray[np.float64],
    *,
    r2: float,
    window: float,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Each variant's lead, as an index into `rows`, and its signed r with it.

    Strongest first; ties by genomic order, then ALID. plink2's edges, which the owner
    ruled `clump` follows (10 November 2026): a variant joins at r̂² **>** r2 and distance
    **<** W. A pair the store does not hold (NaN) never reaches r2. Each lead reads only the
    still-unclumped variants within its window, so memory is hits × hits-within-W, never n².
    """
    n = len(rows)
    keys = [
        strongest_first(scores[i], variants.chrom[i], variants.position[i], variants.alid[i])
        for i in range(n)
    ]
    lead = np.full(n, -1, dtype=np.int64)
    r_lead = np.full(n, np.nan)
    for i in sorted(range(n), key=keys.__getitem__):
        if lead[i] >= 0:
            continue
        lead[i], r_lead[i] = i, 1.0
        same = np.flatnonzero((variants.chrom == variants.chrom[i]) & (lead < 0))
        near = same[np.abs(variants.position[same] - variants.position[i]) < window]
        if len(near) == 0:
            continue
        r = reader.pair_r(rows[np.concatenate([[i], near])])[0, 1:]
        joins = r * r > r2
        lead[near[joins]] = i
        r_lead[near[joins]] = r[joins]
    return lead, r_lead


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
