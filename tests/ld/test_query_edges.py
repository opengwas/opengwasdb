"""Edges of ogld#6's queries beyond the contract: Absent ALIDs, and ties on the window."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import ogld
from ogld.store import StoreReader

TEST_STORE = Path(__file__).parent / "data" / "test-panel" / "store"


def first_alids(count: int) -> list[str]:
    return [str(a) for a in StoreReader.open(TEST_STORE).variants(np.arange(count)).alid]


def test_clump_lists_absent_alids_and_clumps_only_the_present() -> None:
    store = ogld.open(TEST_STORE)
    names = first_alids(3)
    clumps = store.clump(["nowhere", *names], np.array([9.0, 3.0, 2.0, 1.0]), r2=0.5, kb=3000)
    assert clumps.absent == ("nowhere",)
    assert list(clumps.alid) == names
    assert clumps.lead[0] == names[0], "the strongest present variant leads"


def test_a_window_of_exactly_the_distance_reaches_the_partner() -> None:
    # Spec §9: d ≤ W. plink2 uses d < W, so the two differ here, and the spec records it.
    store = ogld.open(TEST_STORE)
    query = first_alids(1)
    wide = store.proxies(query, r2=0.0, kb=3000)
    assert len(wide) > 0
    edge = int(wide.distance.max())
    at_edge = store.proxies(query, r2=0.0, kb=edge / 1000)
    assert edge in set(at_edge.distance.tolist())
    assert len(store.proxies(query, r2=0.0, kb=(edge - 1) / 1000)) < len(at_edge)
