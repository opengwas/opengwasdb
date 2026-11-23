"""The Store Reader imports numpy, zarr, numcodecs, the standard library and itself (#464).

`opengwasdb.store_reader` is the read path every reader of a Store Release
uses, and becomes the `ogquery` package; its extraction is a directory move only
while nothing in it reaches back into the rest of opengwasdb or into a heavy
dependency. Two checks hold that:

- the static one parses every module in the subpackage and refuses any import --
  at module level, inside a function or under `TYPE_CHECKING`, which all couple
  a file to what it names -- outside numpy, zarr, numcodecs, the standard
  library less `sqlite3`, and `opengwasdb.store_reader` itself;
- the run-time one imports the subpackage in a fresh interpreter and requires
  that pandas, scipy, pysam, sqlite3 and threadpoolctl were never loaded, which
  the static check cannot see through a parent package's `__init__`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from architecture_model import imports_in_source

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "opengwasdb/store_reader"
OWN = "opengwasdb.store_reader"
THIRD_PARTY = frozenset({"numpy", "zarr", "numcodecs"})
REFUSED_STDLIB = frozenset({"sqlite3"})
HEAVY = ("pandas", "scipy", "pysam", "sqlite3", "threadpoolctl")


def foreign_imports(path: str, source: str) -> list[str]:
    """Every name `source` imports that the Store Reader may not."""
    refused = []
    for name in sorted(imports_in_source(path, source)):
        top = name.split(".")[0]
        if name == OWN or name.startswith(OWN + "."):
            continue
        if top in THIRD_PARTY:
            continue
        if top in sys.stdlib_module_names and top not in REFUSED_STDLIB:
            continue
        refused.append(name)
    return refused


def store_reader_sources() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((ROOT / PACKAGE).rglob("*.py"))
    }


def test_the_check_refuses_what_it_should():
    path = f"{PACKAGE}/example.py"
    source = "\n".join(
        [
            "import numpy as np",
            "import sqlite3",
            "from typing import TYPE_CHECKING",
            "from . import facade",
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


def test_the_store_reader_imports_only_numpy_zarr_numcodecs_stdlib_and_itself():
    sources = store_reader_sources()
    # The Store Reader is some 9,600 lines in 31 modules; a check over a handful
    # of files would pass a read path that had not reached the subpackage.
    assert len(sources) >= 20, sorted(sources)
    assert sum(len(text.splitlines()) for text in sources.values()) > 7000
    refused = {
        path: names for path, source in sources.items() if (names := foreign_imports(path, source))
    }
    assert refused == {}


def test_importing_the_store_reader_loads_no_heavy_dependency():
    """Every module of the subpackage, the facade included, imported in one fresh
    interpreter: the package `__init__` alone imports nothing, so importing only it
    would prove nothing about the modules a query loads."""
    modules = sorted(
        path[: -len(".py")].replace("/", ".").removesuffix(".__init__")
        for path in store_reader_sources()
    )
    assert "opengwasdb.store_reader.facade" in modules
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
    """The probe's check, on a module that does load one, finds it."""
    probe = (
        "import json, sys\n"
        "import opengwasdb.variants.axis\n"
        f"print(json.dumps([name for name in {HEAVY!r} if name in sys.modules]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert "pysam" in json.loads(out.stdout)
