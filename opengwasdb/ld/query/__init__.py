"""Queries on an LD Store: LD matrices, regions, proxies and clumping (layer 2; ogld#4, ogld#6).

The queries and their results are ogquery's (`ogquery.ld.query`; opengwas/ogquery#2): they read
through the `ogquery.ld.Reader` protocol, which ogquery's `StoreReader` and `GenotypeReader`
satisfy (opengwas/ogld#91; opengwas/ogquery#11). `open` is ogquery's and opens either kind. An
unstored pair is NaN, and only a caller's `fill` turns it into a number.
"""

from ogquery.ld.query import Clumps, LDMatrix, LDStore, Proxies, Reader, open

__all__ = ["Clumps", "LDMatrix", "LDStore", "Proxies", "Reader", "open"]
