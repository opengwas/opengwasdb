"""Proxies: each query variant's partners above an r², within a window (spec §9; ogld#6)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ogld.format import Codes
from ogld.query.requests import window_of
from ogld.query.results import Proxies
from ogld.store import StoreReader


def proxies(reader: StoreReader, alids: Sequence[str], *, r2: float, kb: float) -> Proxies:
    """Every partner of each present ALID within kb × 1,000 bp with r̂² ≥ r2 (ogld#6).

    Judged by `ogld.format.judge_request` first, which refuses a threshold below t_partner within
    `near_bp` and a store that holds no Partner Rows before any frame is read. Each present ALID's
    proxies are its Partner Rows row — one frame read, both sides, every distance — filtered to
    r̂² ≥ r2 and d ≤ W here, because the edges are the query's, not the reader's (partner-rows
    §8.3). A row asked for twice is answered twice, and the reader filters on the positions it
    keeps, so the only string read is the ALIDs of the proxies kept.
    """
    p = reader.provenance
    window = window_of(reader, r2=r2, kb=kb)

    found = reader.rows_of(alids)
    absent = tuple(alid for alid, row in zip(alids, found, strict=True) if row < 0)
    present = [alid for alid, row in zip(alids, found, strict=True) if row >= 0]
    queries = found[found >= 0]
    chrom, query_position = reader.loci_of(queries)

    owner, partner, r = _partner_rows(reader, queries, chrom)
    keep = (r * r >= r2) & (partner != queries[owner])
    owner, partner, r = owner[keep], partner[keep], r[keep]
    distance = np.abs(reader.loci_of(partner)[1] - query_position[owner])
    keep = distance <= window
    owner, partner, r, distance = owner[keep], partner[keep], r[keep], distance[keep]
    proxy = reader.alids_of(partner)

    # Rows go by query (request order), r² descending, distance ascending, then proxy ALID.
    order = np.lexsort((proxy, distance, -(r * r), owner))
    owner, proxy, r, distance = owner[order], proxy[order], r[order], distance[order]
    # A present ALID is the stored one, compared in full by `rows_of`, so it is the query.
    query = np.array([present[i] for i in owner.tolist()], dtype=np.str_)
    return Proxies(
        query=query,
        proxy=proxy,
        r=r,
        r2=r * r,
        distance=distance,
        absent=absent,
        r2_min=r2,
        window_bp=window,
        t_far=p["t_far"],
    )


def _partner_rows(
    reader: StoreReader, queries: npt.NDArray[np.int64], chrom: npt.NDArray[np.str_]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Every query's whole Partner Rows row, as (its query's index, its Variant Set row, its r).

    One bulk read a chromosome, which coalesces only the frames that are physically adjacent
    (partner-rows §9.2), and each entry's code dequantised at the store's own scale (§4). A query
    asked twice has its row read once and answered twice: the entries repeat with their own
    query's index, as request order requires.
    """
    owner: list[npt.NDArray[np.int64]] = []
    partner: list[npt.NDArray[np.int64]] = []
    codes: list[Codes] = []
    for name in np.unique(chrom):
        members = np.flatnonzero(chrom == name)
        start = reader.chromosome_range(str(name))[0]
        indptr, at, row_codes = reader.partner_rows(str(name), queries[members] - start)
        owner.append(np.repeat(members, np.diff(indptr)))
        partner.append(at.astype(np.int64) + start)
        codes.append(row_codes)
    if not codes:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.concatenate(owner),
        np.concatenate(partner),
        reader.encoding.dequantise(np.concatenate(codes)),
    )
