"""A package's re-exports, imported on first use (PEP 562).

`import opengwasdb.store_reader` runs `opengwasdb/__init__.py` first, and every
package `__init__` on the way. One that re-exports build code eagerly makes every
import of anything beneath it pay for the builders, and pandas, scipy and pysam
with them (#464). A package that re-exports through `lazy_exports` keeps its
public names and imports each one's module only when the name is first read.

    _EXPORTS = {"opengwasdb.encoding.codec": ("StoreCodec", "expit")}
    __all__ = exported_names(_EXPORTS)
    __getattr__, __dir__ = lazy_hooks(__name__, _EXPORTS)

A package also imports the same names under `if TYPE_CHECKING:`, so a type checker
sees each one with its type; at run time that block never executes. A name
re-exported under another one is written as the import writes it,
`"PER_VARIANT_CHUNK as DEFAULT_PER_VARIANT_CHUNK"`.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

#: Each module a package re-exports from, and the names it takes from it.
Exports = Mapping[str, Sequence[str]]


def _public(entry: str) -> tuple[str, str]:
    """An entry's (public name, the module's own name for it)."""
    own, _, public = entry.partition(" as ")
    return (public or own), own


def exported_names(exports: Exports) -> list[str]:
    """The package's public names, for its `__all__`, in the order `exports` lists them."""
    return [_public(entry)[0] for entries in exports.values() for entry in entries]


def lazy_attribute(package: str, exports: Exports, name: str) -> Any:
    """`name` from the module `exports` assigns it, cached in `package` once read.

    A name `exports` does not list raises `AttributeError`, as an ordinary
    missing attribute does, so `from package import submodule` still falls
    through to importing the submodule.
    """
    for module, entries in exports.items():
        for public, own in map(_public, entries):
            if public == name:
                value = getattr(importlib.import_module(module), own)
                setattr(sys.modules[package], name, value)
                return value
    raise AttributeError(f"module {package!r} has no attribute {name!r}")


def exported_dir(package: str, exports: Exports) -> list[str]:
    """The package's names, the lazy ones included before their first use."""
    return sorted(set(vars(sys.modules[package])) | set(exported_names(exports)))


def lazy_hooks(
    package: str, exports: Exports
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """The package's module `__getattr__` and `__dir__`, bound to its `exports`."""

    def attribute(name: str) -> Any:
        return lazy_attribute(package, exports, name)

    def listing() -> list[str]:
        return exported_dir(package, exports)

    return attribute, listing
