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

Phase 1 of #475 pins ogquery 0.1.0, which lacks the Clumps reads and the narrowed
proof range `main` gained since, so the shims cannot yet be written. Both tests are
`xfail(strict=True)`: they must fail now, and the commit that writes the shims
turns them on by deleting the marker.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "opengwasdb" / "store_reader"
#: Modules that stay in opengwasdb until the Hybrid Rho change (#495) is ported to
#: ogquery and released; the owner's ruling of 25 Nov 2026.
STAYS_IN_TREE = frozenset({"facade", "rho"})

phase_2 = pytest.mark.xfail(strict=True, reason="phase 2: ogquery 0.2.0")


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
    return {name: getattr(module, name) for name in names}


def test_the_package_holds_modules() -> None:
    # The parametrised tests below are worthless on an empty list.
    assert len(_modules()) > 30


@phase_2
@pytest.mark.parametrize("name", [m for m in _modules() if m not in STAYS_IN_TREE])
def test_a_store_reader_module_re_exports_the_ogquery_module(name: str) -> None:
    shim = importlib.import_module(f"opengwasdb.store_reader.{name}".rstrip("."))
    reader = importlib.import_module(f"ogquery.{name}".rstrip("."))
    theirs = _exported(reader)
    ours = _exported(shim)
    assert theirs, f"ogquery.{name} exports nothing: the comparison would be empty"
    assert sorted(set(theirs) - set(ours)) == [], "names the shim does not re-export"
    assert [n for n in theirs if ours[n] is not theirs[n]] == [], "names that are copies"


@phase_2
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
