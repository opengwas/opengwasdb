"""Proxies: each query variant's partners above an r², within a window (spec §9; ogld#6)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ogld.format import judge_request
from ogld.query.results import Proxies
from ogld.store import StoreReader


def proxies(reader: StoreReader, alids: Sequence[str], *, r2: float, kb: float) -> Proxies:
    """Every partner of each present ALID within kb × 1,000 bp with r̂² ≥ r2 (ogld#6).

    Judged by `ogld.format.judge_request` first. At r2 ≥ t_far it reads one Proxy Index
    row per variant (`StoreReader.index`); below t_far, only within near_bp, it reads the
    Near Band on both sides (`StoreReader.band`).
    """
    p = reader.provenance
    decision = judge_request(
        r2=r2, kb=kb, t_far=p["t_far"], near_bp=p["near_bp"], far_bp=p["far_bp"]
    )
    decision.raise_if_refused()
    window = decision.window_bp

    found = reader.rows_of(alids)
    absent = tuple(alid for alid, row in zip(alids, found, strict=True) if row < 0)
    queries = found[found >= 0]
    partners = reader.index(queries) if r2 >= p["t_far"] else reader.band(queries)

    owner = np.repeat(np.arange(len(queries)), np.diff(partners.indptr))
    query_rows = queries[owner]
    position = reader.variants(np.concatenate([query_rows, partners.partner])).position
    distance = np.abs(position[len(query_rows) :] - position[: len(query_rows)])
    r = partners.r
    keep = (distance <= window) & (r * r >= r2) & (partners.partner != query_rows)
    owner, rows, r, distance = owner[keep], partners.partner[keep], r[keep], distance[keep]
    proxy = reader.variants(rows).alid

    # Rows go by query (request order), r² descending, distance ascending, then proxy ALID.
    order = np.lexsort((proxy, distance, -(r * r), owner))
    owner, proxy, r, distance = owner[order], proxy[order], r[order], distance[order]
    query = reader.variants(queries[owner]).alid
    return Proxies(
        query=query,
        proxy=proxy,
        r=r,
        r2=r * r,
        distance=distance.astype(np.int64),
        absent=absent,
        r2_min=r2,
        window_bp=window,
        t_far=p["t_far"],
    )
