"""Greedy clumping on an LD Store, equal to `plink2 --clump` (spec §9; ogld#6).

From format 0.2.0 a lead's partners are its Partner Rows row (partner-rows §7, §8.2): one frame
read and one decode, filtered to the hits still open in its component. The hits are split into
components first — per chromosome, cut wherever two consecutive positions are W apart or more —
because no lead can reach a hit in another one, and so each component's greedy is independent
(partner-rows §8.2). Components are therefore the unit of parallel work: one task each, on the
reader's bounded pool, and the answer is the same at any number of them (§9.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ogld.format import Refusal, RefusalReason
from ogld.query.requests import window_of
from ogld.query.results import Clumps
from ogld.store import StoreReader


def clump(
    reader: StoreReader, alids: Sequence[str], score: npt.ArrayLike, *, r2: float, kb: float
) -> Clumps:
    """Clump the present `alids` by `score`, strongest first (ogld#6).

    `score` is |z| or −log10 p, one per ALID, finite and ≥ 0; never raw p, which
    underflows to 0 past |z| ≈ 38. Judged by `ogld.format.judge_request` first, which refuses a
    threshold below t_partner within `near_bp` and a store that holds no Partner Rows, before any
    frame is read. Each lead's joiners are its Partner Rows row among the hits still open in its
    component (partner-rows §8.2); a pair the row does not hold is below the Partner Floor or is
    no pair, and cannot reach an answered r2.

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
    # rows_of has already compared the complete stored ALID, not only its locus.
    names = np.array([a for a, ok in zip(alids, present, strict=True) if ok], dtype=np.str_)

    # Each distinct row is one hit, however many requests name it; duplicates scatter afterwards,
    # and a row is ordered by its strongest request's score, as the greedy takes it.
    hits, first, back = np.unique(rows, return_index=True, return_inverse=True)
    strength = np.zeros(len(hits), dtype=np.float64)
    np.maximum.at(strength, back, scores)
    chrom, position = reader.loci_of(hits)
    lead, r_lead = _components(reader, hits, strength, chrom, position, r2=r2, window=window)
    return Clumps(
        alid=names,
        chrom=chrom[back],
        position=position[back],
        score=scores,
        lead=names[first[lead[back]]],
        r=r_lead[back],
        absent=absent,
        r2=r2,
        window_bp=window,
    )


@dataclass(frozen=True)
class _Component:
    """Hits with no gap of W or more between neighbours: one unit of independent work.

    `part` is the component's indices in the clump's hit array, so its results scatter back
    without a search; `rows` are those hits' Variant Set rows, ascending, and `position` their
    positions. `start` is the first Variant Set row of the hits' chromosome, which is what turns a
    Partner Rows row's chromosome-local partner into a Variant Set row (partner-rows §7).
    """

    chrom: str
    start: int
    part: npt.NDArray[np.int64]
    rows: npt.NDArray[np.int64]
    position: npt.NDArray[np.int64]
    score: npt.NDArray[np.float64]


def _components(
    reader: StoreReader,
    rows: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
    chrom: npt.NDArray[np.str_],
    position: npt.NDArray[np.int64],
    *,
    r2: float,
    window: float,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Each hit's lead, as an index into `rows`, and its signed r with it.

    A clump cannot cross chromosomes (ADR 0009), and inside one it cannot reach across a gap of W
    or more (partner-rows §8.2), so the hits are split there and each part clumped on its own. The
    parts are independent, so they run serially or on the reader's bounded pool alike, and the
    results are scattered by hit index: the answer is the same at any concurrency (§9.2). A refusal
    was decided before the first task; a frame that fails its checks fails the whole request, with
    no partial result.
    """
    lead = np.full(len(rows), -1, dtype=np.int64)
    r_lead = np.full(len(rows), np.nan)
    tasks: list[_Component] = []
    for name in np.unique(chrom):
        members = np.flatnonzero(chrom == name)
        start = reader.chromosome_range(str(name))[0]
        # Hits on one chromosome ascend with their rows, and so with their positions (spec §2).
        cuts = np.flatnonzero(np.diff(position[members]) >= window) + 1
        tasks.extend(
            _Component(str(name), start, part, rows[part], position[part], scores[part])
            for part in np.split(members, cuts)
        )
    workers = reader.query_workers
    if workers == 1 or len(tasks) <= 1:
        results = [_component(reader, task, r2=r2, window=window) for task in tasks]
    else:
        results = _pooled(reader, tasks, workers, r2=r2, window=window)
    for part, local, values in results:
        lead[part], r_lead[part] = part[local], values
    return lead, r_lead


def _pooled(
    reader: StoreReader,
    tasks: list[_Component],
    workers: int,
    *,
    r2: float,
    window: float,
) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]]:
    """The tasks on `workers` threads, in task order (partner-rows §9.2).

    A failed task fails the whole request: the others are cancelled, the pool drains what it is
    already running, and the failure is raised with no result assembled from the tasks that
    finished (§8.2).
    """
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ogld_clump") as pool:
        futures = [pool.submit(_component, reader, task, r2=r2, window=window) for task in tasks]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _component(
    reader: StoreReader, task: _Component, *, r2: float, window: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """One component's greedy: each hit's lead, as an index into `task.rows`, and its signed r.

    Strongest first; ties by the global row, which is the engine's order (spec §2). plink2's edges,
    which the owner ruled `clump` follows (10 November 2026): a variant joins at r̂² **>** r2 and
    distance **<** W. Each lead reads its own Partner Rows row once, one frame at most, and keeps
    the entries that are hits of its component and still open: never a pair with a hit another lead
    took, and never a square among the hits (ogld#14). A hit that another lead takes is never read.
    """
    lo, hi = int(task.rows[0]), int(task.rows[-1])
    # A hit's slot, -1 for a row of the span that is no hit: filtering a row is array indexing.
    slots = np.full(hi - lo + 1, -1, dtype=np.int64)
    slots[task.rows - lo] = np.arange(len(task.rows), dtype=np.int64)
    lead = np.full(len(task.rows), -1, dtype=np.int64)
    r_lead = np.full(len(task.rows), np.nan)
    for i in np.lexsort((task.rows, -task.score)).tolist():
        if lead[i] >= 0:
            continue
        lead[i], r_lead[i] = i, 1.0
        partners, codes = reader.partner_row(task.chrom, int(task.rows[i]) - task.start)
        at_rows = partners.astype(np.int64) + task.start
        inside = (at_rows >= lo) & (at_rows <= hi)
        at = slots[at_rows[inside] - lo]
        r = reader.encoding.dequantise(codes[inside])
        exists = at >= 0
        at, r = at[exists], r[exists]
        joins = (
            (lead[at] < 0) & (r * r > r2) & (np.abs(task.position[at] - task.position[i]) < window)
        )
        at, r = at[joins], r[joins]
        lead[at], r_lead[at] = i, r
    return task.part, lead, r_lead


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
