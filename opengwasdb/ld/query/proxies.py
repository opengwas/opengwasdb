"""Proxies: each query variant's partners above an r², within a window (spec §9; ogld#6)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ogld.format.geometry import owners
from ogld.query.requests import window_of
from ogld.query.results import Proxies
from ogld.store import StoreReader


def proxies(reader: StoreReader, alids: Sequence[str], *, r2: float, kb: float) -> Proxies:
    """Every partner of each present ALID within kb × 1,000 bp with r̂² ≥ r2 (ogld#6).

    Judged by `ogld.format.judge_request` first. Its rows are `StoreReader.partners`, which
    chooses the stored part: one Proxy Index row per variant at r2 ≥ t_far, the Near Band
    on both sides below it (ogld#14). The reader filters on the positions it keeps, so the
    only string read is the ALIDs of the proxies kept.
    """
    p = reader.provenance
    window = window_of(reader, r2=r2, kb=kb)

    found = reader.rows_of(alids)
    absent = tuple(alid for alid, row in zip(alids, found, strict=True) if row < 0)
    present = [alid for alid, row in zip(alids, found, strict=True) if row >= 0]
    partners = reader.partners(found[found >= 0], r2=r2, window_bp=window)

    owner = owners(np.diff(partners.indptr))
    r, distance = partners.r, partners.distance
    proxy = reader.alids_of(partners.partner)

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
