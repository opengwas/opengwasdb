"""Every engine read of a Store Release goes through the Store Reader (#473).

The validator, Reference Completion, the top-hit harvest, the overview and Rho
read Store Releases only through `opengwasdb.store_reader`: there is one
implementation of the read path before it becomes ogquery. A static check over
every module of `opengwasdb/` holds that. Outside the Store Reader no module

- imports `pysam` or `sqlite3`;
- opens a Zarr group or array for reading (`open_group`, `zarr.open*`, in mode
  ``r``, or in a mode the source does not spell out);
- constructs a `VariantAxis` itself, rather than asking a release for it
  (`StoreRelease.variant_axis`);
- scans the 0.2.x Store Variant Table (`iter_variant_records`) or opens a
  release's `index.sqlite` (`index_connection`) to query it.

The Store Reader's injected 0.2.x Variant Table implementation counts as the
Store Reader (#473 ruling 1). Writers, the rewrite command's 0.2.x source side
and inputs that are not Store Releases are exempt, each named below with its
reason: a whole module in `EXEMPT_MODULES`, or one function of a module that
also reads in `WRITER_FUNCTIONS` -- a writer reading back what it is writing,
before it is a release.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "opengwasdb"

#: The Store Reader, and its injected 0.2.x Store Variant Table (#473 ruling 1,
#: #464 ruling 5): pysam's and SQLite's reads of that table live here until #476.
STORE_READER = {
    "opengwasdb/store_reader/": "the Store Reader",
    "opengwasdb/variants/axis.py": (
        "the Store Reader's VariantAxis constructed from a path, for opengwasdb-stores' audit; "
        "it also writes the Variant Table"
    ),
    "opengwasdb/variants/legacy_table.py": (
        "the Store Reader's LegacyVariantTable: the 0.2.x table's pysam and SQLite reads"
    ),
    "opengwasdb/index/release_index.py": (
        "the Store Reader's index.sqlite reads, which it may not make itself (no sqlite3)"
    ),
}

#: Modules exempt as a whole, with the reason.
EXEMPT_MODULES = {
    "opengwasdb/store/arrays.py": (
        "the array-creation seam: every write-mode open; its mode-r open_group forwards to the "
        "Store Reader's"
    ),
    "opengwasdb/store/open.py": (
        "the writer's release envelope: index_connection and arrays(mode) for writes, and "
        "open_store, which hands the Store Reader its 0.2.x Variant Table"
    ),
    "opengwasdb/store/convert.py": (
        "the rewrite command: its 0.2.x source side is the one read the ticket exempts"
    ),
    "opengwasdb/index/sqlite.py": "writes index.sqlite's schema and metadata",
    "opengwasdb/completion/ld_panel.py": "reads an LD panel, not a Store Release",
    "opengwasdb/ld/": "LD Stores and genotype stores, not Store Releases",
    "opengwasdb/layouts/dense/build.py": "a builder: writes index.sqlite",
    "opengwasdb/layouts/dense/build_vcf.py": "a builder: writes index.sqlite",
    "opengwasdb/layouts/ragged/build_ssf.py": "a builder: writes index.sqlite",
    "opengwasdb/layouts/ragged/build_besd.py": "a builder: writes index.sqlite",
    "opengwasdb/completion/fill_shards.py": "completion's writer: writes completion_quality",
}

#: Functions that read what their own module is writing, before it is a release.
WRITER_FUNCTIONS = {
    ("opengwasdb/layouts/dense/complete.py", "_write_union_axis_tables"): (
        "writes the completed release's index.sqlite"
    ),
    ("opengwasdb/layouts/dense/complete.py", "_record_effective_dense_chunk_shape"): (
        "records the staged arrays' layout in the staged index.sqlite"
    ),
    ("opengwasdb/layouts/dense/complete.py", "_completed_analysis_rows"): (
        "rolls up the completion_quality rows it staged"
    ),
    ("opengwasdb/layouts/ragged/complete.py", "_checkpoint_and_assemble"): (
        "creates the staged release's completion_quality table"
    ),
    ("opengwasdb/layouts/ragged/complete.py", "_write_top_hits_and_analyses"): (
        "rolls up the completion_quality rows it staged"
    ),
    ("opengwasdb/layouts/hybrid/complete.py", "_correct_crossover_quality"): (
        "corrects the completion_quality of the Dense Component it is completing"
    ),
    ("opengwasdb/layouts/dense/indexed_subsets.py", "_validate_staged"): (
        "checks the subset group it staged before publishing it"
    ),
    ("opengwasdb/variants/reference.py", "require_written_rsids_match"): (
        "reads back the Variant Table it just wrote, before the release exists"
    ),
}

REFUSED_IMPORTS = frozenset({"pysam", "sqlite3"})
ZARR_OPENERS = frozenset({"open", "open_group", "open_array", "open_consolidated"})
READ_HELPERS = frozenset({"iter_variant_records", "index_connection"})


@dataclass(frozen=True, order=True)
class Read:
    """One raw read of a Store Release outside the Store Reader."""

    path: str
    line: int
    function: str
    what: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} ({self.function}): {self.what}"


def _mode(call: ast.Call) -> str | None:
    """The mode a call opens in: its literal, ``"r"`` when it names none, None when unknown."""
    for keyword in call.keywords:
        if keyword.arg == "mode":
            return keyword.value.value if isinstance(keyword.value, ast.Constant) else None
    if len(call.args) >= 2:
        value = call.args[1]
        return value.value if isinstance(value, ast.Constant) else None
    return "r"


def _called(call: ast.Call) -> tuple[str | None, str | None]:
    """The called name, and the name it is an attribute of (``zarr`` in ``zarr.open``)."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id, None
    if isinstance(func, ast.Attribute):
        owner = func.value.id if isinstance(func.value, ast.Name) else None
        return func.attr, owner
    return None, None


def _raw_read(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        refused = [a.name for a in node.names if a.name.split(".")[0] in REFUSED_IMPORTS]
        return f"import {', '.join(refused)}" if refused else None
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level == 0 and module.split(".")[0] in REFUSED_IMPORTS:
            return f"from {module} import"
        return None
    if not isinstance(node, ast.Call):
        return None
    name, owner = _called(node)
    if name == "open_group" or (owner == "zarr" and name in ZARR_OPENERS):
        mode = _mode(node)
        return f"{name} in mode {mode!r}" if mode in (None, "r") else None
    if name == "VariantAxis":
        return "VariantAxis(...) constructed directly"
    if name in READ_HELPERS:
        return f"{name}()"
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.scope: list[str] = []
        self.found: list[Read] = []

    def _enter(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _enter

    def generic_visit(self, node: ast.AST) -> None:
        what = _raw_read(node)
        if what is not None:
            function = ".".join(self.scope) or "<module>"
            self.found.append(Read(self.path, getattr(node, "lineno", 0), function, what))
        super().generic_visit(node)


def raw_reads(path: str, source: str) -> list[Read]:
    """Every raw read `source` makes, wherever it sits."""
    visitor = _Visitor(path)
    visitor.visit(ast.parse(source, filename=path))
    return sorted(visitor.found)


def _covered(path: str, table: dict[str, str]) -> bool:
    return any(path == key or (key.endswith("/") and path.startswith(key)) for key in table)


def _exempt(read: Read) -> bool:
    if _covered(read.path, STORE_READER) or _covered(read.path, EXEMPT_MODULES):
        return True
    outermost = read.function.split(".")[0]
    return (read.path, outermost) in WRITER_FUNCTIONS


def engine_sources() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(PACKAGE.rglob("*.py"))
    }


def test_the_check_sees_every_kind_of_raw_read():
    source = "\n".join(
        [
            "import sqlite3",
            "from pysam import BGZFile",
            "import zarr",
            "from opengwasdb.store.arrays import open_group",
            "def f(path, mode):",
            "    zarr.open_group(path)",
            "    zarr.open(path, mode='r')",
            "    open_group(path, mode)",
            "    open_group(path, 'w')",
            "    open_group(path, mode='r+')",
            "class C:",
            "    def g(self, store):",
            "        VariantAxis(store.path)",
            "        store.index_connection()",
            "        iter_variant_records(store.variant_table_path)",
            "        store.variant_axis(aliases=False)",
        ]
    )
    found = [(r.line, r.function, r.what) for r in raw_reads("opengwasdb/example.py", source)]
    assert found == [
        (1, "<module>", "import sqlite3"),
        (2, "<module>", "from pysam import"),
        (6, "f", "open_group in mode 'r'"),
        (7, "f", "open in mode 'r'"),
        (8, "f", "open_group in mode None"),
        (13, "C.g", "VariantAxis(...) constructed directly"),
        (14, "C.g", "index_connection()"),
        (15, "C.g", "iter_variant_records()"),
    ]


def test_every_exemption_names_a_module_or_function_that_exists():
    sources = engine_sources()
    for key in [*STORE_READER, *EXEMPT_MODULES]:
        assert key in sources or any(path.startswith(key) for path in sources), key
    for path, function in WRITER_FUNCTIONS:
        tree = ast.parse(sources[path])
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef)}
        assert function in names, (path, function)


def test_every_exempt_writer_function_still_reads_raw():
    """An exemption nothing needs any more would excuse the next raw read added there."""
    used = {
        (read.path, read.function.split(".")[0])
        for path, source in engine_sources().items()
        for read in raw_reads(path, source)
    }
    assert sorted(set(WRITER_FUNCTIONS) - used) == []


def test_no_engine_module_reads_a_release_around_the_store_reader():
    sources = engine_sources()
    # The package is some 200 modules; a check over a handful would pass a read
    # that had never been looked at.
    assert len(sources) > 150
    reads = [
        str(read)
        for path, source in sources.items()
        for read in raw_reads(path, source)
        if not _exempt(read)
    ]
    assert reads == []
