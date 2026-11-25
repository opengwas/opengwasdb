"""LD Stores: per-ancestry LD for clumping, proxies and LD matrices (formerly ogld).

The public surface (`docs/spec/ld-store.md` §8, §9):

    store = opengwasdb.ld.open(path_or_url, cache_bytes=None)  # LDStore; keeps 4 GiB of chunks
    store.matrix(alids, fill=None)                    # LDMatrix: signed r, NaN where unstored
    store.region(chrom, centre, radius, fill=None)    # LDMatrix; radius ≤ far_bp / 2
    store.proxies(alids, r2=0.8, kb=10_000)           # Proxies
    store.clump(alids, score, r2=..., kb=...)         # Clumps
    opengwasdb.ld.build(BuildInputs(...), out)        # the new store's Provenance
    opengwasdb.ld.validate(location)                  # findings, by rule name

A request the store cannot answer exactly raises `Refusal`, naming its reason.

The read side is ogquery's (`ogquery.ld`; opengwas/ogquery#2, ogquery ADR 0002): `LDStore`, the
results and the refusals here are ogquery's objects. opengwasdb keeps what builds, writes and
validates a store, and the Genotype Store reader, which `open` dispatches a Genotype Store to.
Every name is imported on first use, so the read names load none of the build code.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

from opengwasdb.lazy_exports import exported_names, lazy_hooks

if TYPE_CHECKING:
    from ogquery.ld.format import AnswerKind as AnswerKind
    from ogquery.ld.format import InputError as InputError
    from ogquery.ld.format import Provenance as Provenance
    from ogquery.ld.format import Refusal as Refusal
    from ogquery.ld.format import RefusalReason as RefusalReason
    from ogquery.ld.query import Clumps as Clumps
    from ogquery.ld.query import LDMatrix as LDMatrix
    from ogquery.ld.query import LDStore as LDStore
    from ogquery.ld.query import Proxies as Proxies
    from ogquery.ld.store import Variants as Variants

    from opengwasdb.ld.build import BuildInputs as BuildInputs
    from opengwasdb.ld.build import build as build
    from opengwasdb.ld.query.ldstore import open as open
    from opengwasdb.ld.query.validation import validate as validate
    from opengwasdb.ld.store.validation import Finding as Finding
    from opengwasdb.version import __version__ as __version__

#: Each name, by the module it comes from; imported on first use (`opengwasdb.lazy_exports`),
#: so reading one LD read name never imports the build, the writer or validation (ogquery#2).
_EXPORTS = {
    "ogquery.ld.format": (
        "AnswerKind",
        "InputError",
        "Provenance",
        "Refusal",
        "RefusalReason",
    ),
    "ogquery.ld.query": (
        "Clumps",
        "LDMatrix",
        "LDStore",
        "Proxies",
    ),
    "ogquery.ld.store": ("Variants",),
    "opengwasdb.ld.query.ldstore": ("open",),
    "opengwasdb.ld.build": (
        "BuildInputs",
        "build",
    ),
    "opengwasdb.ld.query.validation": ("validate",),
    "opengwasdb.ld.store.validation": ("Finding",),
    "opengwasdb.version": ("__version__",),
}
__all__ = exported_names(_EXPORTS)
__getattr__, __dir__ = lazy_hooks(__name__, _EXPORTS)


class _Package(types.ModuleType):
    """This package, keeping `build` the function when its submodule `opengwasdb.ld.build` loads.

    Importing a submodule binds it on its package under its own name, and the function
    `opengwasdb.ld.build` shares that name. The eager initialiser rebound the function after the
    import; a lazy one imports nothing, so the import system's binding would otherwise leave
    `opengwasdb.ld.build(...)` calling a module. The submodule stays in `sys.modules`, where
    `from opengwasdb.ld.build import ...` finds it.
    """

    def __setattr__(self, name: str, value: object) -> None:
        if name == "build" and isinstance(value, types.ModuleType):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _Package
