"""The Store Reader modules still in opengwasdb import only the Store Reader (#475, #464).

`opengwasdb/store_reader/facade.py` and `rho.py` stay in-tree until the Hybrid Rho change
(#495) is ported to ogquery, and a later phase shims them. That is a move, not a rewrite,
only while neither reaches back into the rest of opengwasdb or into a heavy dependency;
the check that held this for the whole subpackage (`test_store_reader_imports`, moved to
ogquery with the code) is kept for the two files that are left:

- the static check refuses any import -- at module level, in a function or under
  `TYPE_CHECKING` -- outside numpy, zarr, numcodecs, the standard library less `sqlite3`,
  `ogquery` and `opengwasdb.store_reader` (the re-exports);
- the run-time check imports both in a fresh interpreter and requires that pandas, scipy,
  pysam, sqlite3 and threadpoolctl were never loaded.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from architecture_model import imports_in_source
from test_store_reader_is_ogquery import STAYS_IN_TREE

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "opengwasdb/store_reader"
OWN = ("opengwasdb.store_reader", "ogquery")
THIRD_PARTY = frozenset({"numpy", "zarr", "numcodecs"})
REFUSED_STDLIB = frozenset({"sqlite3"})
HEAVY = ("pandas", "scipy", "pysam", "sqlite3", "threadpoolctl")


def foreign_imports(path: str, source: str) -> list[str]:
    """Every name `source` imports that an in-tree Store Reader module may not."""
    refused = []
    for name in sorted(imports_in_source(path, source)):
        top = name.split(".")[0]
        if any(name == own or name.startswith(own + ".") for own in OWN):
            continue
        if top in THIRD_PARTY:
            continue
        if top in sys.stdlib_module_names and top not in REFUSED_STDLIB:
            continue
        refused.append(name)
    return refused


def in_tree_sources() -> dict[str, str]:
    return {
        f"{PACKAGE}/{name}.py": (ROOT / PACKAGE / f"{name}.py").read_text(encoding="utf-8")
        for name in sorted(STAYS_IN_TREE)
    }


def test_the_check_refuses_what_it_should():
    path = f"{PACKAGE}/example.py"
    source = "\n".join(
        [
            "import numpy as np",
            "import sqlite3",
            "from typing import TYPE_CHECKING",
            "from ogquery.release import StoreRelease",
            "from opengwasdb.store_reader.alid import parse_canonical_alid",
            "if TYPE_CHECKING:",
            "    from opengwasdb.encoding.association_codec import Encoded",
            "def f():",
            "    import scipy.stats",
            "    from opengwasdb import top_hits",
        ]
    )
    assert foreign_imports(path, source) == [
        "opengwasdb.encoding.association_codec.Encoded",
        "opengwasdb.top_hits",
        "scipy.stats",
        "sqlite3",
    ]


def test_an_injected_import_fails_the_guard():
    """The guard fails on the real files with a foreign import appended to each."""
    for path, source in in_tree_sources().items():
        assert foreign_imports(path, source) == [], path
        injected = source + "\nfrom opengwasdb.layouts.dense import complete\nimport pysam\n"
        assert foreign_imports(path, injected) == [
            "opengwasdb.layouts.dense.complete",
            "pysam",
        ], path


def test_the_in_tree_modules_import_only_the_store_reader():
    sources = in_tree_sources()
    assert sorted(sources) == [f"{PACKAGE}/facade.py", f"{PACKAGE}/rho.py"]
    # The guard is worthless on a file that is a shim: each holds real code.
    assert all(len(text.splitlines()) > 50 for text in sources.values())
    refused = {p: names for p, s in sources.items() if (names := foreign_imports(p, s))}
    assert refused == {}


def test_importing_the_in_tree_modules_loads_no_heavy_dependency():
    modules = [f"opengwasdb.store_reader.{name}" for name in sorted(STAYS_IN_TREE)]
    probe = (
        "import importlib, json, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        f"print(json.dumps([name for name in {HEAVY!r} if name in sys.modules]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout) == []


def test_the_run_time_probe_sees_a_heavy_import():
    """The probe's check, on a module that does load one, finds it: Reference Completion's
    LD panel reader, which imports pysam at load."""
    probe = (
        "import json, sys\n"
        "import opengwasdb.completion.ld_panel\n"
        f"print(json.dumps([name for name in {HEAVY!r} if name in sys.modules]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert "pysam" in json.loads(out.stdout)
