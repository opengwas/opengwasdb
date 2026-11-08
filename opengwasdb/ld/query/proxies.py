"""Proxies: each query variant's partners above an r², within a window (spec §9; ogld#6)."""

from __future__ import annotations

from collections.abc import Sequence

from ogld.query.results import Proxies
from ogld.store import StoreReader


def proxies(reader: StoreReader, alids: Sequence[str], *, r2: float, kb: float) -> Proxies:
    """Every partner of each present ALID within kb × 1,000 bp with r̂² ≥ r2 (ogld#6).

    Judged by `ogld.format.judge_request` first. At r2 ≥ t_far it reads one Proxy Index
    row per variant (`StoreReader.index`); below t_far, only within near_bp, it reads the
    Near Band on both sides (`StoreReader.band`).
    """
    raise NotImplementedError("ogld#6")
