"""Greedy clumping on an LD Store, equal to `plink2 --clump` (spec §9; ogld#6)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy.typing as npt

from ogld.query.results import Clumps
from ogld.store import StoreReader


def clump(
    reader: StoreReader, alids: Sequence[str], score: npt.ArrayLike, *, r2: float, kb: float
) -> Clumps:
    """Clump the present `alids` by `score`, strongest first (ogld#6).

    `score` is |z| or −log10 p, one per ALID, finite and ≥ 0; never raw p, which
    underflows to 0 past |z| ≈ 38. Judged by `ogld.format.judge_request` first. Pairs
    come from `StoreReader.pair_r`; an unstored pair within an answered window is the
    Bound, which cannot reach an answered r2.
    """
    raise NotImplementedError("ogld#6")
