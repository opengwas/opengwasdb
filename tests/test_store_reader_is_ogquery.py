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

Every module is a re-export, `facade.py` and `rho.py` included: the Hybrid Rho work
(#495) was ported to ogquery 0.4 and those two were shimmed after it (#475, phase 3).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from ogquery_seam import exported, module_pair, not_the_same_object

PACKAGE = Path(__file__).resolve().parent.parent / "opengwasdb" / "store_reader"


def _modules() -> list[str]:
    """The dotted name, below `store_reader`, of every module the package holds."""
    names = []
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        names.append(".".join(parts))
    return names


def test_the_package_holds_modules() -> None:
    # The parametrised tests below are worthless on an empty list.
    assert len(_modules()) > 30


@pytest.mark.parametrize("name", _modules())
def test_a_store_reader_module_re_exports_the_ogquery_module(name: str) -> None:
    shim, reader = module_pair(
        f"opengwasdb.store_reader.{name}".rstrip("."), f"ogquery.{name}".rstrip(".")
    )
    theirs = exported(reader)
    ours = exported(shim)
    # A package that only holds submodules has nothing to compare; a module must.
    assert theirs or hasattr(reader, "__path__"), f"ogquery.{name} exports nothing"
    # A pure shim is complete: it has every name ogquery exports, as ogquery's own object.
    assert sorted(set(theirs) - set(ours)) == [], "names the shim does not re-export"
    assert not_the_same_object(shim, reader, theirs) == [], "names that are copies"


def test_no_store_reader_module_defines_a_function_or_class() -> None:
    defining = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        found = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        ]
        if found:
            defining[str(path.relative_to(PACKAGE))] = found[:3]
    assert defining == {}


def test_the_shims_hold_one_release_class_of_each() -> None:
    """The per-name identity above covers every export; this is the claim it exists for, in
    one place: a process holds one `StoreRelease` and one `StoreManifest`."""
    import ogquery.manifest
    import ogquery.release

    import opengwasdb.store_reader.manifest as shim_manifest
    from opengwasdb.store_reader import facade

    assert facade.StoreRelease is ogquery.release.StoreRelease
    assert shim_manifest.StoreManifest is ogquery.manifest.StoreManifest


def test_both_ways_to_a_query_facade_return_the_same_class(dense_store_path) -> None:
    """`StoreRelease.query()` and `query_store` both build ogquery's facade, since `facade.py`
    is a re-export (phase 3). Before that `query_store` built an in-tree copy that #495 edited."""
    from opengwasdb.query import query_store
    from opengwasdb.store.open import open_store

    with open_store(dense_store_path).query() as through_release:
        with query_store(dense_store_path) as through_query_store:
            assert type(through_release) is type(through_query_store)


def test_the_ruff_ignore_covers_the_re_exports_and_nothing_else() -> None:
    """`F401`/`F403` are ignored per file for the re-exports, so a file that holds code
    is still linted."""
    import tomllib

    ignores = tomllib.loads((PACKAGE.parent.parent / "pyproject.toml").read_text())
    listed = {
        path
        for path, rules in ignores["tool"]["ruff"]["lint"]["per-file-ignores"].items()
        if path.startswith("opengwasdb/store_reader/") and set(rules) == {"F401", "F403"}
    }
    shims = {
        path.relative_to(PACKAGE.parent.parent).as_posix()
        for path in PACKAGE.rglob("*.py")
    }
    assert listed == shims


#: The two functions the wrapper overrides to default `query=`.
OVERRIDDEN_DIGESTS = {"release_answers", "release_digests"}


def test_the_digests_default_to_query_store_and_nothing_else_knows_it() -> None:
    from ogquery import digests as theirs

    from opengwasdb.store import query_digests as ours

    assert ours.release_digests is not theirs.release_digests
    assert ours.differences is theirs.differences  # everything else is ogquery's own
    assert ours.RANGE_ROWS_ASKED == theirs.RANGE_ROWS_ASKED
    assert ours.adapter_name is theirs.adapter_name
    # Explicit exports: every name ogquery's `__all__` has is listed, so a name it adds is
    # missed here loudly, not silently dropped as a star import would not.
    assert set(theirs.__all__) - set(ours.__all__) == set()
    assert not_the_same_object(ours, theirs, set(ours.__all__) - OVERRIDDEN_DIGESTS) == []
