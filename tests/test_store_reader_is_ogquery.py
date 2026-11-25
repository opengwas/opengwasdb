"""`opengwasdb.store_reader` is the `ogquery` package, not a copy of it (#475).

The Store Reader moved to opengwas/ogquery; opengwasdb keeps the import path its
callers (ogdb-api, opengwasdb-stores, its own builders) use as thin re-exports. Two
properties make that true, and a third makes the copy's deletion hold:

* every name a `opengwasdb.store_reader.<module>` exports is the very object
  `ogquery.<module>` has, so one process holds one `StoreRelease` and one
  `StoreManifest` and an `isinstance` across the two import paths cannot say no;
* no module under `opengwasdb/store_reader/` defines a function or a class, so
  nothing in opengwasdb can re-implement a Store Reader operation and drift from
  ogquery's answers, which is how a wrong answer that looks right gets made.

`opengwasdb/store_reader/facade.py` and `rho.py` are the owner's exception for now
(25 Nov): the Hybrid Rho work (#495) edits them, and a later phase shims them once
that is ported to ogquery and released. They are named in `STAYS_IN_TREE` so the
exception is one line to remove.

"""

from __future__ import annotations
import __future__

import ast
import importlib
import types
from pathlib import Path
from typing import Any

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "opengwasdb" / "store_reader"
#: Modules that stay in opengwasdb until the Hybrid Rho change (#495) is ported to
#: ogquery and released; the owner's ruling of 25 Nov 2026.
STAYS_IN_TREE = frozenset({"facade", "rho"})



def _modules() -> list[str]:
    """The dotted name, below `store_reader`, of every module the package holds."""
    names = []
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        names.append(".".join(parts))
    return names


def _exported(module: object) -> dict[str, object]:
    names = getattr(module, "__all__", None)
    if names is None:
        names = [name for name in vars(module) if not name.startswith("_")]
    values = {name: getattr(module, name) for name in names}
    # A submodule is its own shim here, and `annotations`/`Any` are the module's own imports.
    return {
        name: value
        for name, value in values.items()
        if not isinstance(value, types.ModuleType) and value is not Any
        and not isinstance(value, __future__._Feature)
    }


def test_the_package_holds_modules() -> None:
    # The parametrised tests below are worthless on an empty list.
    assert len(_modules()) > 30


@pytest.mark.parametrize("name", [m for m in _modules() if m not in STAYS_IN_TREE])
def test_a_store_reader_module_re_exports_the_ogquery_module(name: str) -> None:
    shim = importlib.import_module(f"opengwasdb.store_reader.{name}".rstrip("."))
    reader = importlib.import_module(f"ogquery.{name}".rstrip("."))
    theirs = _exported(reader)
    ours = _exported(shim)
    # A package that only holds submodules has nothing to compare; a module must.
    assert theirs or hasattr(reader, "__path__"), f"ogquery.{name} exports nothing"
    assert sorted(set(theirs) - set(ours)) == [], "names the shim does not re-export"
    assert [n for n in theirs if ours[n] is not theirs[n]] == [], "names that are copies"


def test_no_store_reader_module_defines_a_function_or_class() -> None:
    defining = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.stem in STAYS_IN_TREE:
            continue
        tree = ast.parse(path.read_text())
        found = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        ]
        if found:
            defining[str(path.relative_to(PACKAGE))] = found[:3]
    assert defining == {}


def test_the_in_tree_remainder_reads_through_one_class_of_each() -> None:
    """`facade.py` and `rho.py` stay in opengwasdb for now and import the Store Reader
    through the shims: they must see ogquery's classes, not a second copy."""
    import ogquery.manifest
    import ogquery.release

    from opengwasdb.store_reader import facade, rho

    assert facade.StoreRelease is ogquery.release.StoreRelease
    assert rho.DenseRhoReader.__module__ == "opengwasdb.store_reader.rho"
    import opengwasdb.store_reader.manifest as shim_manifest

    assert shim_manifest.StoreManifest is ogquery.manifest.StoreManifest


def test_the_digests_default_to_query_store_and_nothing_else_knows_it() -> None:
    from ogquery import digests as theirs

    from opengwasdb.store import query_digests as ours

    assert ours.release_digests is not theirs.release_digests
    assert ours.differences is theirs.differences  # everything else is ogquery's own
    assert ours.RANGE_ROWS_ASKED == theirs.RANGE_ROWS_ASKED
    assert ours.adapter_name is theirs.adapter_name
