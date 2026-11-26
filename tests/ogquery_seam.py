"""What the two seam tests share: a re-export of an `ogquery` module is the module's objects.

`tests/test_store_reader_is_ogquery.py` (pure shims: every module under
`opengwasdb/store_reader/`) and `tests/ld/test_read_side_is_ogquery.py` (mixed modules, which
keep their own build names beside the re-exported read ones) hold different completeness rules
but one notion of identity: every name an opengwasdb module takes from ogquery is the very
object ogquery has, so one process holds one class of each and an `isinstance` across the two
import paths cannot say no. They also read ogquery's installed source, found here once.
"""

from __future__ import annotations
import __future__

import importlib
import types
from collections.abc import Collection
from pathlib import Path
from typing import Any

import ogquery


def ogquery_root() -> Path:
    """The directory of the installed `ogquery` package, which the source-scanning guards read."""
    return Path(ogquery.__file__).resolve().parent


def ogquery_source(relative: str) -> Path:
    """The installed source file or directory `relative` (``"stats.py"``, ``"ld"``)."""
    path = ogquery_root() / relative
    assert path.exists(), f"ogquery has no {relative}: {path}"
    return path


def exported(module: types.ModuleType) -> dict[str, object]:
    """The names `module` exports: `__all__`, or every public name when it has none, less
    submodules and the module's own `annotations` and `Any` imports."""
    names = getattr(module, "__all__", None)
    if names is None:
        names = [name for name in vars(module) if not name.startswith("_")]
    return {
        name: value
        for name, value in ((name, getattr(module, name)) for name in names)
        if not isinstance(value, types.ModuleType)
        and value is not Any
        and not isinstance(value, __future__._Feature)
    }


def module_pair(ours: str, theirs: str) -> tuple[types.ModuleType, types.ModuleType]:
    """The opengwasdb module `ours` and the ogquery module `theirs` it re-exports."""
    return importlib.import_module(ours), importlib.import_module(theirs)


def not_the_same_object(
    ours: types.ModuleType,
    theirs: types.ModuleType,
    names: Collection[str],
) -> list[str]:
    """The `names` that `ours` does not have, or has as a copy rather than `theirs`' object."""
    return sorted(
        name
        for name in names
        if not hasattr(ours, name) or getattr(ours, name) is not getattr(theirs, name)
    )
