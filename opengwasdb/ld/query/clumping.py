"""Greedy clumping on an LD Store, equal to `plink2 --clump` (spec §9; ogld#6)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ogld.format import Refusal, RefusalReason
from ogld.format.geometry import indptr_of, spans
from ogld.query.requests import window_of
from ogld.query.results import Clumps
from ogld.store import StoreReader


def clump(
    reader: StoreReader, alids: Sequence[str], score: npt.ArrayLike, *, r2: float, kb: float
) -> Clumps:
    """Clump the present `alids` by `score`, strongest first (ogld#6).

    `score` is |z| or −log10 p, one per ALID, finite and ≥ 0; never raw p, which
    underflows to 0 past |z| ≈ 38. Judged by `ogld.format.judge_request` first. Each
    lead's joiners come from `StoreReader.partners`, among the hits still unclumped, so
    the reader chooses the Proxy Index or the Near Band (ogld#14); a pair the store does
    not hold within an answered window is the Bound, which cannot reach an answered r2.

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
    chrom, position = reader.loci_of(rows)
    names = reader.alids_of(rows)

    lead, r_lead = _chromosome_greedy(reader, rows, scores, chrom, r2=r2, window=window)
    return Clumps(
        alid=names,
        chrom=chrom,
        position=position,
        score=scores,
        lead=names[lead],
        r=r_lead,
        absent=absent,
        r2=r2,
        window_bp=window,
    )


def _chromosome_greedy(
    reader: StoreReader,
    rows: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
    chrom: npt.NDArray[np.str_],
    *,
    r2: float,
    window: float,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """A clump cannot cross chromosomes: keep each one's chunks together (ADR 0009).

    Scatter back into request order, including duplicates. Only the order of independent
    chromosome passes changes, never strongest-first order inside one chromosome.
    """
    lead = np.empty(len(rows), dtype=np.int64)
    r_lead = np.empty(len(rows), dtype=np.float64)
    for name in np.unique(chrom):
        members = np.flatnonzero(chrom == name)
        local, values = _greedy(reader, rows[members], scores[members], r2=r2, window=window)
        lead[members], r_lead[members] = members[local], values
    return lead, r_lead


def _greedy(
    reader: StoreReader,
    rows: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
    *,
    r2: float,
    window: float,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Each variant's lead, as an index into `rows`, and its signed r with it.

    Strongest first; ties by the global row, which is the engine's order (spec §2), then
    by request order for an ALID asked twice. plink2's edges, which the owner ruled `clump`
    follows (10 November 2026): a variant joins at r̂² **>** r2 and distance **<** W. Each
    lead reads its own partners among the hits still open, one call each, so a lead resolves
    only its pairs with its candidates: never a square among them (ogld#14).
    """
    warm = reader.clump_warmer(rows, r2=r2, window_bp=window)
    hits = _Hits(rows)
    lead = np.full(len(rows), -1, dtype=np.int64)
    r_lead = np.full(len(rows), np.nan)
    for i in np.lexsort((rows, -scores)).tolist():
        if lead[i] >= 0:
            continue
        lead[i], r_lead[i] = i, 1.0
        # The same variant asked again is r = 1 at d = 0, and joins on the same edges.
        if _joins(*_ITSELF, r2=r2, window=window).all():
            hits.join(lead, r_lead, i, np.array([rows[i]]), np.ones(1))
        hits.close(rows[i], lead)
        warm(rows[i])
        found = reader.partners(rows[i : i + 1], r2=r2, window_bp=window, among=hits.open())
        joins = _joins(found.r, found.distance, r2=r2, window=window)
        hits.join(lead, r_lead, i, found.partner[joins], found.r[joins])
    return lead, r_lead


#: A variant with itself, as a pair: r = 1 at distance 0 (spec §8).
_ITSELF = (np.ones(1), np.array([0], dtype=np.int64))


def _joins(
    r: npt.NDArray[np.float64], distance: npt.NDArray[np.int64], *, r2: float, window: float
) -> npt.NDArray[np.bool_]:
    """plink2's edges, which `clump` keeps (spec §9): r̂² **>** r2 and d **<** W. The reader
    hands back r̂² ≥ r2 and d ≤ W, and this is where `clump` narrows them."""
    return np.asarray((r * r > r2) & (distance < window))


class _Hits:
    """The requested rows, each distinct row with every request of it, and which are open."""

    def __init__(self, rows: npt.NDArray[np.int64]) -> None:
        self.rows, self.which = np.unique(rows, return_inverse=True)
        counts = np.bincount(self.which, minlength=len(self.rows))
        self.starts = indptr_of(counts)
        self.order = np.argsort(self.which, kind="stable")
        self.open_rows = np.ones(len(self.rows), dtype=bool)

    def open(self) -> npt.NDArray[np.int64]:
        """The distinct rows some request of which no lead has taken, ascending."""
        return self.rows[self.open_rows]

    def close(self, row: np.int64, lead: npt.NDArray[np.int64]) -> None:
        """Mark `row` taken once no request of it is still open."""
        k = int(np.searchsorted(self.rows, row))
        self.open_rows[k] = bool((lead[self._requests(np.array([k]))] < 0).any())

    def join(
        self,
        lead: npt.NDArray[np.int64],
        r_lead: npt.NDArray[np.float64],
        i: int,
        rows: npt.NDArray[np.int64],
        r: npt.NDArray[np.float64],
    ) -> None:
        """Give every open request of `rows` the lead i, with its r."""
        k = np.searchsorted(self.rows, rows)
        requests = self._requests(k)
        each = np.repeat(r, self.starts[k + 1] - self.starts[k])
        still = lead[requests] < 0
        lead[requests[still]] = i
        r_lead[requests[still]] = each[still]
        self.open_rows[k] = False

    def _requests(self, k: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """The request indices of the distinct rows k, back to back."""
        return self.order[spans(self.starts[k], self.starts[k + 1] - self.starts[k])]


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
