"""One read of a store's root metadata, and which kind of store it says it is (opengwas/ogld#91).

`opengwasdb.ld.open` and `opengwasdb.ld.validate` both come through here: the root is opened once,
its kind is read
from the attribute it records, and the opened group is handed to that kind's reader or validator.
One read is enough: opening the root again in the reader would add a metadata read — an HTTP
round trip for a remote store — to every open.

This reads the kind and nothing more: an envelope a reader refuses (a Zarr v2 hierarchy, a root
with no provenance) is still validation's to report as findings, not this seam's to raise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Literal

import zarr

from opengwasdb.ld.genotypes.layout import KEY
from opengwasdb.ld.store.location import ChunkCache, open_group

LD: Final = "ld_store"
GENOTYPE: Final = "genotype_store"


@dataclass(frozen=True)
class Kinded:
    """The root's kind, the one opening of its metadata, and what it was opened through."""

    kind: Literal["ld_store", "genotype_store"]
    root: zarr.Group
    location: str
    cache: ChunkCache

    def record(self) -> object:
        """The kind's recorded provenance, or None where the root records none."""
        return self.root.attrs.get(KEY if self.kind == GENOTYPE else LD)


def open_kind(location: str | os.PathLike[str], *, cache_bytes: int | None = None) -> Kinded:
    """Open `location` once and say which kind it is: a root with the Genotype Store key is one,
    and every other root is read as an LD Store, whose envelope its reader or validation judges."""
    cache = ChunkCache(cache_bytes)
    root = open_group(location, cache=cache)
    kind: Literal["ld_store", "genotype_store"] = GENOTYPE if KEY in root.attrs else LD
    return Kinded(kind, root, str(location), cache)
