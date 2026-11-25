"""Queries on an LD Store: LD matrices, proxies and clumping (layer 2; ogld#4, ogld#6).

It reads through the `opengwasdb.ld.query.reader.Reader` interface (`StoreReader` or the
Genotype Store reader, opengwas/ogld#91) and judges every request with
`opengwasdb.ld.format`; it imports nothing from `opengwasdb.ld.build`. An unstored pair is NaN, and
only a
caller's `fill` turns it into a number.
"""

from ogquery.ld.query.ldstore import LDStore
from opengwasdb.ld.query.ldstore import open
from ogquery.ld.query.results import Clumps, LDMatrix, Proxies

__all__ = ["Clumps", "LDMatrix", "LDStore", "Proxies", "open"]
