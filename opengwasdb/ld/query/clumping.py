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
    the window with r̂² ≥ r2 of it joins it. Because leads are taken strongest first, a
    member joins the strongest lead it reaches. Ties go to genomic order, then ALID.
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
    **<** W. A pair the store does not hold (NaN) never reaches r2. Each lead reads its own
    row of the store once (`_row_r`), so the cost per lead is that row's length, never the
    square of the candidates within its window.
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
        r = _lead_r(reader, rows[i], rows[near], r2=r2)
        joins = r * r > r2
        lead[near[joins]] = i
        r_lead[near[joins]] = r[joins]
    return lead, r_lead


#: Up to this many candidates, a lead's r with them is one `pair_r` block of (1 + m)² pairs, at most
#: about 10⁶ and under 0.2 s, which beats reading a whole row (a Near Band row and column is tens of
#: thousands of elements). Beyond it the block grows with m², and the lead reads its own row.
BLOCK_CANDIDATES = 1000


def _lead_r(
    reader: StoreReader, lead: np.int64, candidates: npt.NDArray[np.int64], *, r2: float
) -> npt.NDArray[np.float64]:
    """The lead's signed r with each candidate row, NaN where the store holds none."""
    if len(candidates) <= BLOCK_CANDIDATES:
        return reader.pair_r(np.concatenate([[lead], candidates]))[0, 1:]
    return _row_r(reader, lead, candidates, r2=r2)


def _row_r(
    reader: StoreReader, lead: np.int64, candidates: npt.NDArray[np.int64], *, r2: float
) -> npt.NDArray[np.float64]:
    """The lead's signed r with each candidate row, NaN where the store holds none.

    At r2 ≥ t_far the Proxy Index row has every pair that can reach r2 (spec §5), out to far_bp;
    below t_far, which `judge_request` answers only within near_bp, the Near Band row and column
    hold every pair (spec §4). A pair in neither is the Bound or Not Measured, and stays NaN.
    """
    row = np.array([lead], dtype=np.int64)
    held = reader.index(row) if r2 >= reader.provenance["t_far"] else reader.band(row)
    partner, r = held.partner, held.r  # one row: ascending global rows
    if len(partner) == 0:
        return np.full(len(candidates), np.nan)
    at = np.minimum(np.searchsorted(partner, candidates), len(partner) - 1)
    return np.where(partner[at] == candidates, r[at], np.nan)


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
