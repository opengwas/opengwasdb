"""`opengwasdb.ld.open`: opening an LD Store or a Genotype Store as an `LDStore` (ld-store §8, §9).

`LDStore` and everything it reads through are ogquery's (opengwas/ogquery#2, ogquery ADR 0002):
`ogquery.ld.open` opens LD Stores, and refuses a Genotype Store, naming this function. A Genotype
Store computes its answers from pinned pgens with pgenlib and threadpoolctl, which ogquery does not
depend on, so its reader stays here (ADR 0100). This `open` reads the root once, through ogquery's
kind dispatch (`ogquery.ld.query.kind.open_kind`), and plugs the reader of that kind into the same
`LDStore`, so both kinds answer through one set of queries (opengwas/ogld#91).
"""

from __future__ import annotations

import os
from typing import cast

from ogquery.ld.format import Provenance
from ogquery.ld.query.kind import GENOTYPE, open_kind
from ogquery.ld.query.ldstore import LDStore as LDStore
from ogquery.ld.query.reader import Reader
from ogquery.ld.store.location import OpenedStore
from ogquery.ld.store.reader import StoreReader, opened_ld


def open(location: str | os.PathLike[str], *, cache_bytes: int | None = None) -> LDStore:
    """Open the LD Store or Genotype Store at a local path or an HTTP(S) URL (ld-store §1; ogld#4).

    The store keeps at most `cache_bytes` of the chunks and shard indexes its reads fetch, 4 GiB
    if None, and none if 0; anything but an int ≥ 0 is an `InputError` (ld-store §10; ogld#32).
    An LD Store is read by ogquery's `StoreReader`, exactly as `ogquery.ld.open` reads it; a
    Genotype Store by `GenotypeReader`, imported only when one is opened."""
    kinded = open_kind(location, cache_bytes=cache_bytes)
    reader: Reader
    if kinded.kind == GENOTYPE:
        from opengwasdb.ld.genotypes.reader import GenotypeReader

        record = kinded.record()
        provenance = cast(Provenance, record if isinstance(record, dict) else {})
        reader = GenotypeReader.from_opened(
            OpenedStore(kinded.root, kinded.location, provenance, kinded.cache)
        )
    else:
        reader = StoreReader.from_opened(opened_ld(kinded.root, kinded.location, kinded.cache))
    return LDStore(reader)
