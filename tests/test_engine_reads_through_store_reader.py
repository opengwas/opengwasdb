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
- scans the 0.2.x Store Variant Table (`iter_variant_records`), or opens it or a
  release's `index.sqlite` through one of opengwasdb's own openers
  (`index_connection`, `connect_index`, `opengwasdb.index.sqlite.connect`,
  `open_index_read_only`, `open_legacy_variant_table`, `TabixVariantTable`).

A name is followed through its import, so `import zarr as z; z.open(p)`,
`from zarr import open_group as og` and `VariantAxis as VA` are each the read
they alias.

The Store Reader's injected 0.2.x implementations count as the Store Reader (#473
ruling 1): the Variant Table's (`opengwasdb.variants.legacy_table`, whose
`open_index_read_only` is the one read-only opener of `index.sqlite`) and the
validator's named `index.sqlite` reads beside it (`opengwasdb.index.release_index`).
Writers, the rewrite command's 0.2.x source side and inputs that are not Store
Releases are exempt, each named below with its reason: a whole module in
`EXEMPT_MODULES`, or one function of a module that also reads in
`WRITER_FUNCTIONS` -- a writer writing, or reading back what it is writing before it
is a release. An exemption nothing needs any more fails the test, so it cannot
excuse the next raw read added there.

`StoreRelease.arrays()` -- a release's `data.zarr` group, read-only -- is the Store
Reader's own accessor, so a call to it is compliant by design (ruling 2 on #473's
decisions). The read-mode calls outside the Store Reader are listed, and checked, in
`COMPLIANT_ARRAYS_CALLS`.

A validation rule may take a raw group from the Store Reader (`Target.group`) only
when it checks the storage itself; a rule that reads stored *values* reads them
through the Store Reader's decoders and readers (ruling on #473's review).
`RAW_GROUP_READERS` classifies every validation module that takes one, with why.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "opengwasdb"

#: The injected 0.2.x implementations (#473 ruling 1, #464 ruling 5): pysam's and
#: SQLite's reads of a release live here until #476. The Store Reader itself is ogquery's
#: (#475); `opengwasdb/store_reader/` holds re-exports only.
STORE_READER = {
    "opengwasdb/variants/axis.py": (
        "the Store Reader's VariantAxis constructed from a path, for opengwasdb-stores' audit; "
        "it also writes the Variant Table"
    ),
    "opengwasdb/variants/legacy_table.py": (
        "the Store Reader's LegacyVariantTable: the 0.2.x table's pysam reads, and "
        "open_index_read_only, the one read-only opener of index.sqlite"
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
        "the writer's release envelope: index_connection, connect_index (read-write) and "
        "arrays(mode) for writes, and open_store, which hands the Store Reader its 0.2.x "
        "Variant Table"
    ),
    "opengwasdb/store/convert.py": (
        "the rewrite command: its 0.2.x source side is the one read the ticket exempts"
    ),
    "opengwasdb/index/sqlite.py": "writes index.sqlite's schema and metadata",
    "opengwasdb/completion/ld_panel.py": "reads an LD panel, not a Store Release",
    "opengwasdb/ld/": "LD Stores and genotype stores, not Store Releases",
}

#: Functions that write a release's index, or read back what their own module is
#: writing before it is a release.
WRITER_FUNCTIONS = {
    ("opengwasdb/store/rewrite.py", "_write_provenance_index"): (
        "writes the rewritten release's index.sqlite, holding the completion_quality it "
        "copies from its source (#471)"
    ),
    ("opengwasdb/completion/fill_shards.py", "shard_checkpoint_fills"): (
        "writes the completion_quality rows of the release it is completing"
    ),
    ("opengwasdb/layouts/dense/complete.py", "_write_union_axis_tables"): (
        "writes the completed release's index.sqlite"
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
    ("opengwasdb/layouts/dense/indexed_subsets.py", "_validate_staged"): (
        "checks the subset group it staged before publishing it"
    ),
    ("opengwasdb/variants/reference.py", "require_written_rsids_match"): (
        "reads back the Variant Table it just wrote, before the release exists"
    ),
    ("opengwasdb/store/legacy_quality.py", "count_missing"): (
        "adds the masked cells to the converted release's completion_quality, which the "
        "rewrite's provenance_index step wrote (#492)"
    ),
    ("opengwasdb/store/rewrite.py", "_records_in_row_order"): (
        "streams the staged copy's Variant Table, before it is a release: its steps are "
        "replacing the search indexes a Variant Axis would open (#468)"
    ),
}

#: Every read-mode `StoreRelease.arrays()` call outside the Store Reader and the
#: writer's envelope (`opengwasdb/store/`), by module and function: compliant by
#: design (ruling 2), listed so a new one is seen.
COMPLIANT_ARRAYS_CALLS = frozenset(
    {
        ("opengwasdb/clumps.py", "_tier_component"),
        ("opengwasdb/layouts/dense/complete.py", "_make_reader.make_reader"),
        ("opengwasdb/layouts/dense/complete.py", "_record_effective_dense_chunk_shape"),
        ("opengwasdb/layouts/dense/complete.py", "_run_completion"),
        ("opengwasdb/layouts/dense/complete.py", "_dense_shard_rows"),
        ("opengwasdb/layouts/dense/rho.py", "build_dense_rho"),
        ("opengwasdb/layouts/hybrid/complete.py", "_crossover_status"),
        ("opengwasdb/store/legacy_conversion.py", "_grid"),
        ("opengwasdb/store/legacy_conversion.py", "_rounded_se"),
        ("opengwasdb/store/legacy_conversion.py", "_refuse_other_planes"),
        ("opengwasdb/store/legacy_expected.py", "_ExpectedStoreQuery.__init__"),
        ("opengwasdb/store/legacy_quality.py", "_on_panel"),
        ("opengwasdb/validation/observed_view.py", "_Planes.__init__"),
        ("opengwasdb/validation/observed_view_hybrid.py", "_open_overflow"),
        ("opengwasdb/validation/observed_view_hybrid.py", "_DenseCodec.of"),
        ("opengwasdb/validation/rules/layout.py", "_validate_hybrid_recordings"),
        ("opengwasdb/validation/rules/top_hits.py", "tier_root"),
        ("opengwasdb/validation/validate.py", "_fidelity_blocks"),
    }
)

#: Every validation module that takes a raw group from the Store Reader
#: (`Target.group`), and why it may: each checks the storage itself, and reads any
#: stored value it judges through the Store Reader's decoders and readers.
RAW_GROUP_READERS = {
    "opengwasdb/validation/rules/associations.py": "the planes' shapes and lengths as stored",
    "opengwasdb/validation/rules/encoding.py": (
        "the planes' and side tables' dtypes and shapes against the declared plan"
    ),
    "opengwasdb/validation/rules/layout.py": "the arrays' chunking against its three recordings",
    "opengwasdb/validation/rules/csr.py": (
        "the CSR's structure: offsets, lengths, a mask's presence"
    ),
    "opengwasdb/validation/rules/side_tables.py": (
        "the exception and overflow tables as stored, read through the Store Reader's tables"
    ),
    "opengwasdb/validation/rules/completion.py": (
        "the completion masks as stored (shape, 0/1 codes); completion_quality via ReleaseIndex"
    ),
    "opengwasdb/validation/rules/sample_size.py": (
        "the n plane's length; its codes are the association scan's"
    ),
    "opengwasdb/validation/rules/by_variant.py": (
        "the By-Variant Index's structure, and that its stored codes are the sequence's"
    ),
    "opengwasdb/validation/rules/subsets.py": (
        "the subsets' groups as stored; their cells decoded by the Store Reader's codec"
    ),
    "opengwasdb/validation/rules/rho.py": (
        "the Rho group's attrs and lengths; its values via store_reader.rho.packed_rho"
    ),
    "opengwasdb/validation/rules/top_hits.py": (
        "the tiers' presence and order; their arrays via store_reader.top_hits.tier_arrays"
    ),
    "opengwasdb/validation/scans.py": (
        "the planes' stored codes (markers, exception positions), decoded by the Store "
        "Reader's plane classes and StoreDecoder, never re-implemented"
    ),
}

REFUSED_IMPORTS = frozenset({"pysam", "sqlite3"})
ZARR_OPENERS = frozenset({"open", "open_group", "open_array", "open_consolidated"})
#: opengwasdb's own openers and scans of a release's 0.2.x files.
OPENGWASDB_OPENERS = frozenset(
    {
        "iter_variant_records",
        "index_connection",
        "connect_index",
        "open_index_read_only",
        "open_legacy_variant_table",
        "TabixVariantTable",
    }
)
#: `opengwasdb.index.sqlite.connect`, by the names it is imported under.
INDEX_CONNECT = frozenset({"opengwasdb.index.sqlite.connect", "opengwasdb.index.connect"})


@dataclass(frozen=True, order=True)
class Read:
    """One raw read of a Store Release outside the Store Reader."""

    path: str
    line: int
    function: str
    what: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} ({self.function}): {self.what}"


def _mode(call: ast.Call, *, position: int = 1) -> str | None:
    """The mode a call opens in: its literal, ``"r"`` when it names none, None when unknown."""
    for keyword in call.keywords:
        if keyword.arg == "mode":
            return keyword.value.value if isinstance(keyword.value, ast.Constant) else None
    if len(call.args) > position:
        value = call.args[position]
        return value.value if isinstance(value, ast.Constant) else None
    return "r"


def _imported(path: str, node: ast.ImportFrom) -> str:
    """The absolute module a `from ... import` names."""
    if not node.level:
        return node.module or ""
    parts = path[: -len(".py")].split("/")
    package = parts[:-1]
    base = package[: len(package) - (node.level - 1)]
    return ".".join([*base, *([node.module] if node.module else [])])


def _aliases(path: str, tree: ast.AST) -> dict[str, str]:
    """Each name the module binds by import, and the dotted name it stands for."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    names[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    names[top] = top
        elif isinstance(node, ast.ImportFrom):
            module = _imported(path, node)
            for alias in node.names:
                names[alias.asname or alias.name] = f"{module}.{alias.name}"
    return names


def _dotted(func: ast.expr, names: dict[str, str]) -> str | None:
    """What a called expression names, through the module's imports."""
    if isinstance(func, ast.Name):
        return names.get(func.id, func.id)
    if isinstance(func, ast.Attribute):
        base = _dotted(func.value, names)
        return f"{base or '?'}.{func.attr}"
    return None


def _call_read(call: ast.Call, names: dict[str, str]) -> str | None:
    dotted = _dotted(call.func, names)
    if dotted is None:
        return None
    last = dotted.rsplit(".", 1)[-1]
    if last == "open_group" or (dotted.startswith("zarr.") and last in ZARR_OPENERS):
        mode = _mode(call)
        return f"{dotted} in mode {mode!r}" if mode in (None, "r") else None
    if last == "VariantAxis":
        return f"{dotted}(...) constructed directly"
    if last in OPENGWASDB_OPENERS or dotted in INDEX_CONNECT:
        return f"{dotted}()"
    return None


def _import_read(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        refused = [a.name for a in node.names if a.name.split(".")[0] in REFUSED_IMPORTS]
        return f"import {', '.join(refused)}" if refused else None
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level == 0 and module.split(".")[0] in REFUSED_IMPORTS:
            return f"from {module} import"
    return None


class _Visitor(ast.NodeVisitor):
    """Every node of a module with the function or class it sits in."""

    def __init__(self) -> None:
        self.scope: list[str] = []
        self.found: list[tuple[ast.AST, str]] = []

    def _enter(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _enter

    def generic_visit(self, node: ast.AST) -> None:
        self.found.append((node, ".".join(self.scope) or "<module>"))
        super().generic_visit(node)


def _scoped(path: str, source: str) -> tuple[ast.AST, Iterator[tuple[ast.AST, str]]]:
    tree = ast.parse(source, filename=path)
    visitor = _Visitor()
    visitor.visit(tree)
    return tree, iter(visitor.found)


def raw_reads(path: str, source: str) -> list[Read]:
    """Every raw read `source` makes, wherever it sits."""
    tree, nodes = _scoped(path, source)
    names = _aliases(path, tree)
    found = []
    for node, function in nodes:
        what = _import_read(node)
        if what is None and isinstance(node, ast.Call):
            what = _call_read(node, names)
        if what is not None:
            found.append(Read(path, getattr(node, "lineno", 0), function, what))
    return sorted(found)


def arrays_calls(path: str, source: str) -> set[tuple[str, str]]:
    """The read-mode `.arrays()` calls `source` makes, by function."""
    _tree, nodes = _scoped(path, source)
    return {
        (path, function)
        for node, function in nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "arrays"
        and _mode(node, position=0) == "r"
    }


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


def _planted(lines: list[str]) -> list[tuple[int, str, str]]:
    source = "\n".join(lines)
    return [(r.line, r.function, r.what) for r in raw_reads("opengwasdb/x/example.py", source)]


def test_the_check_sees_every_kind_of_raw_read():
    found = _planted(
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
            "    open(path)",
            "class C:",
            "    def g(self, store):",
            "        VariantAxis(store.path)",
            "        store.index_connection()",
            "        iter_variant_records(store.variant_table_path)",
            "        store.variant_axis(aliases=False)",
            "        store.arrays()",
        ]
    )
    assert found == [
        (1, "<module>", "import sqlite3"),
        (2, "<module>", "from pysam import"),
        (6, "f", "zarr.open_group in mode 'r'"),
        (7, "f", "zarr.open in mode 'r'"),
        (8, "f", "opengwasdb.store.arrays.open_group in mode None"),
        (14, "C.g", "VariantAxis(...) constructed directly"),
        (15, "C.g", "store.index_connection()"),
        (16, "C.g", "iter_variant_records()"),
    ]


def test_the_check_follows_a_name_through_its_import():
    found = _planted(
        [
            "import zarr as z",
            "import zarr.api.synchronous as zs",
            "from zarr import open_group as og, open_array",
            "from opengwasdb.store_reader.variant_axis import VariantAxis as VA",
            "def f(p):",
            "    z.open(p)",
            "    z.open_array(p, mode='a')",
            "    zs.open_group(p)",
            "    og(p)",
            "    open_array(p)",
            "    VA(p)",
        ]
    )
    assert found == [
        (6, "f", "zarr.open in mode 'r'"),
        (8, "f", "zarr.api.synchronous.open_group in mode 'r'"),
        (9, "f", "zarr.open_group in mode 'r'"),
        (10, "f", "zarr.open_array in mode 'r'"),
        (11, "f", "opengwasdb.store_reader.variant_axis.VariantAxis(...) constructed directly"),
    ]


def test_the_check_refuses_opengwasdbs_own_openers():
    found = _planted(
        [
            "from opengwasdb.store.open import connect_index",
            "from opengwasdb.index import connect as connect_db",
            "from opengwasdb.index import sqlite as idx",
            "from opengwasdb.variants.legacy_table import TabixVariantTable",
            "from opengwasdb.variants import legacy_table",
            "import sqlite3 as s",
            "def f(p):",
            "    connect_index(p)",
            "    connect_db(p)",
            "    idx.connect(p)",
            "    TabixVariantTable(p)",
            "    legacy_table.open_legacy_variant_table(p, aliases=True)",
            "    legacy_table.open_index_read_only(p)",
            "    s.connect(p)",
        ]
    )
    assert found == [
        (6, "<module>", "import sqlite3"),
        (8, "f", "opengwasdb.store.open.connect_index()"),
        (9, "f", "opengwasdb.index.connect()"),
        (10, "f", "opengwasdb.index.sqlite.connect()"),
        (11, "f", "opengwasdb.variants.legacy_table.TabixVariantTable()"),
        (12, "f", "opengwasdb.variants.legacy_table.open_legacy_variant_table()"),
        (13, "f", "opengwasdb.variants.legacy_table.open_index_read_only()"),
    ]


def test_every_exemption_names_a_module_or_function_that_exists():
    sources = engine_sources()
    for key in [*STORE_READER, *EXEMPT_MODULES]:
        assert key in sources or any(path.startswith(key) for path in sources), key
    for path, function in WRITER_FUNCTIONS:
        tree = ast.parse(sources[path])
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef)}
        assert function in names, (path, function)


def test_every_exemption_still_excuses_a_raw_read():
    """An exemption nothing needs any more would excuse the next raw read added there."""
    reads = [read for path, source in engine_sources().items() for read in raw_reads(path, source)]
    used_functions = {(read.path, read.function.split(".")[0]) for read in reads}
    assert sorted(set(WRITER_FUNCTIONS) - used_functions) == []
    unused_modules = [
        key
        for table in (STORE_READER, EXEMPT_MODULES)
        for key in table
        if not any(_covered(read.path, {key: ""}) for read in reads)
    ]
    assert unused_modules == []


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


def test_the_arrays_calls_are_the_ones_listed():
    found = {
        call
        for path, source in engine_sources().items()
        if not _covered(path, {**STORE_READER, **EXEMPT_MODULES})
        for call in arrays_calls(path, source)
    }
    assert sorted(found) == sorted(COMPLIANT_ARRAYS_CALLS)


def test_every_validation_module_taking_a_raw_group_is_classified():
    taking = {
        path
        for path, source in engine_sources().items()
        if path.startswith("opengwasdb/validation/")
        and re.search(r"\btarget\.group\b|\b_target\.group\b|\.reopen\(\)", source)
    }
    assert sorted(taking) == sorted(RAW_GROUP_READERS)
