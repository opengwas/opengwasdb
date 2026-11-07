"""Ogld: per-ancestry LD Stores for clumping, proxies and LD matrices.

The public surface (`docs/spec/ld-store.md` §8, §9):

    store = ogld.open(path_or_url)                    # LDStore
    store.matrix(alids, fill=None)                    # LDMatrix: signed r, NaN where unstored
    store.region(chrom, centre, radius, fill=None)    # LDMatrix; radius ≤ 5 Mb
    store.proxies(alids, r2=0.8, kb=10_000)           # Proxies
    store.clump(alids, score, r2=..., kb=...)         # Clumps
    ogld.build(BuildInputs(...), out)                 # the new store's Provenance
    ogld.validate(location)                           # findings, by rule name

A request the store cannot answer exactly raises `Refusal`, naming its reason.
"""

from ogld.build import BuildInputs, build
from ogld.format import AnswerKind, InputError, Provenance, Refusal, RefusalReason
from ogld.query import Clumps, LDMatrix, LDStore, Proxies, open
from ogld.store import Finding, Variants, validate
from ogld.version import __version__

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
