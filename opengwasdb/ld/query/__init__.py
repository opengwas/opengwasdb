"""Queries on an LD Store: LD matrices, proxies and clumping (layer 2; ogld#4, ogld#6).

It reads through `ogld.store.StoreReader` and judges every request with
`ogld.format`; it imports nothing from `ogld.build`. An unstored pair is NaN, and only a
caller's `fill` turns it into a number.
"""

from ogld.query.ldstore import LDStore, open
from ogld.query.results import Clumps, LDMatrix, Proxies

__all__ = ["Clumps", "LDMatrix", "LDStore", "Proxies", "open"]
