"""One read of a store's root metadata, and which kind of store it says it is (#91).

`ogld.open` and `ogld.validate` both come through here: the root is opened once, its kind is read
from the attribute it records, and the opened group is handed to that kind's reader or validator.
Before this, each dispatch opened the root and then the reader opened it again, which added a
metadata read — an HTTP round trip for a remote store — to every LD Store open (fix round 1,
Standards H4, Repeated Switches).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Literal, cast

from ogld.format import Provenance
from ogld.genotypes.layout import KEY
from ogld.store.location import ChunkCache, OpenedStore, open_group
from ogld.store.reader import opened_ld

LD: Final = "ld_store"
GENOTYPE: Final = "genotype_store"


@dataclass(frozen=True)
class Kinded:
    """The root's kind and the one `OpenedStore` both halves read it through."""

    kind: Literal["ld_store", "genotype_store"]
    opened: OpenedStore


def open_kind(location: str | os.PathLike[str], *, cache_bytes: int | None = None) -> Kinded:
    """Open `location` once and say which kind it is, refusing neither kind's envelope here.

    The reader or validator that follows judges the rest (format version, provenance and
    encoding), so a wrong kind is still refused by name by its own owner. A Genotype Store root
    whose record is not an object still dispatches to the genotype side, whose validation reports
    it rather than claiming the root is an LD Store.
    """
    cache = ChunkCache(cache_bytes)
    root = open_group(location, cache=cache)
    if KEY in root.attrs:
        record = root.attrs.get(KEY)
        provenance = cast(Provenance, record if isinstance(record, dict) else {})
        return Kinded(GENOTYPE, OpenedStore(root, str(location), provenance, cache))
    return Kinded(LD, opened_ld(root, str(location), cache))
