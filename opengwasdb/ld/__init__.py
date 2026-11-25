"""Ogld: per-ancestry LD Stores for clumping, proxies and LD matrices.

The public surface (`docs/spec/ld-store.md` §8, §9):

    store = opengwasdb.ld.open(path_or_url, cache_bytes=None)  # LDStore; keeps 4 GiB of chunks
    store.matrix(alids, fill=None)                    # LDMatrix: signed r, NaN where unstored
    store.region(chrom, centre, radius, fill=None)    # LDMatrix; radius ≤ 5 Mb
    store.proxies(alids, r2=0.8, kb=10_000)           # Proxies
    store.clump(alids, score, r2=..., kb=...)         # Clumps
    opengwasdb.ld.build(BuildInputs(...), out)                 # the new store's Provenance
    opengwasdb.ld.validate(location)                           # findings, by rule name

A request the store cannot answer exactly raises `Refusal`, naming its reason.
"""

from opengwasdb.ld.build import BuildInputs, build
from ogquery.ld.format import AnswerKind, InputError, Provenance, Refusal, RefusalReason
from ogquery.ld.query import Clumps, LDMatrix, LDStore, Proxies
from opengwasdb.ld.query import open
from opengwasdb.ld.query.validation import validate
from opengwasdb.ld.store import Finding
from ogquery.ld.store import Variants
from opengwasdb.version import __version__

__all__ = [
    "AnswerKind",
    "BuildInputs",
    "Clumps",
    "Finding",
    "InputError",
    "LDMatrix",
    "LDStore",
    "Provenance",
    "Proxies",
    "Refusal",
    "RefusalReason",
    "Variants",
    "__version__",
    "build",
    "open",
    "validate",
]
