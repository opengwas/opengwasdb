"""Associations are decoded in one module, with four reviewed exceptions (ADR 0070).

The query facade and the observed-view checks used to decode their own planes --
the imputed mask, then EAF with the panel substitution, then SE from that EAF --
and each copy was a place for the order or the shared read to go wrong (#253).
They now ask `opengwasdb.associations`, and must not touch a plane at all.

`validate` and the Indexed Variant Subset's validation keep four decodes of their
own, each sharing an EAF read the check already holds. Those four call sites are
named below by function; a decode anywhere else in those modules fails.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "opengwasdb"

#: Modules that read associations only through `opengwasdb.associations`.
READERS = (
    "query/facade.py",
    "query/resolve.py",
    "validation/observed_view.py",
    "validation/observed_view_ragged.py",
    "validation/observed_view_hybrid.py",
)

#: Modules that keep reviewed decodes of their own, and only those.
CHECKED = ("validation/validate.py", "layouts/dense/indexed_subsets.py")

#: The reviewed EAF-for-SE decodes, by the function each sits in.
ALLOWED = {
    ("validation/validate.py", "_validate_dense_band"):
        "one frequency read per band, shared by the EAF range check and SE decoding",
    ("validation/validate.py", "_decode_top_hit_band"):
        "the band the Top-Hit Index check already read for its own EAF comparison",
    ("layouts/dense/indexed_subsets.py", "_compare_indexed_band"):
        "the primary block the subset's EAF is compared against",
    ("layouts/dense/indexed_subsets.py", "_sample_decoded_view"):
        "the sampled primary cells the subset's EAF is compared against",
}

#: The plane types whose decode the association readers own.
PLANES = frozenset(
    {"DenseZPlane", "DenseSePlane", "DenseEafPlane", "RaggedSePlane", "RaggedEafPlane", "EafRead"}
)

#: The CSR reader's decode members, which only `CsrReader` calls.
CSR_DECODES = frozenset(
    {
        "get_analysis",
        "se_slice",
        "se_at",
        "eaf_slice_read",
        "eaf_at_read",
        "z_at",
        "n_slice",
        "n_at",
    }
)

#: An SE plane's reads; called with `eaf=`, one decodes SE from a frequency read.
SE_READS = frozenset({"band", "column", "row", "rows", "block", "points", "slice", "at"})


def _in_functions(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """Every node with the name of the innermost function holding it."""
    found: list[tuple[str, ast.AST]] = []

    def visit(node: ast.AST, owner: str) -> None:
        for child in ast.iter_child_nodes(node):
            is_function = isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            name = child.name if is_function else owner
            found.append((name, child))
            visit(child, name)

    visit(tree, "<module>")
    return found


def se_decodes(source: str) -> list[tuple[str, int]]:
    """`(function, line)` of each SE plane read handed an EAF array."""
    return [
        (owner, node.lineno)
        for owner, node in _in_functions(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in SE_READS | {"se_slice", "se_at"}
        and any(keyword.arg == "eaf" for keyword in node.keywords)
    ]


def plane_reads(source: str) -> list[str]:
    """Each way `source` reaches a plane: a plane type imported or named through a
    module (`enc.DenseSePlane`), a CSR decode member called, or an SE decode."""
    found = []
    for _, node in _in_functions(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            found += [f"imports {a.name}" for a in node.names if a.name in PLANES]
        elif isinstance(node, ast.Attribute) and node.attr in PLANES:
            found.append(f"names .{node.attr} at line {node.lineno}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in CSR_DECODES:
                found.append(f"calls .{node.func.attr}() at line {node.lineno}")
    found += [f"decodes SE from EAF in {owner} at line {n}" for owner, n in se_decodes(source)]
    return found


def unreviewed_decodes(module: str, source: str) -> list[tuple[str, int]]:
    return [(owner, line) for owner, line in se_decodes(source) if (module, owner) not in ALLOWED]


def test_the_guard_sees_a_plane_named_through_its_module() -> None:
    source = (
        "import opengwasdb.encoding as enc\n"
        "def read(group, encoding, eaf):\n"
        "    return enc.DenseSePlane.open(group, encoding).band(0, 1, eaf=eaf)\n"
    )
    assert plane_reads(source) == [
        "names .DenseSePlane at line 3",
        "decodes SE from EAF in read at line 3",
    ]


def test_the_guard_sees_a_new_decode_beside_the_reviewed_ones() -> None:
    source = (
        "def _validate_dense_band(se_plane, eaf):\n"
        "    return se_plane.band(0, 1, eaf=eaf)\n"
        "def _validate_something_new(se_plane, eaf):\n"
        "    return se_plane.points(rows, cols, eaf=eaf)\n"
    )
    assert unreviewed_decodes("validation/validate.py", source) == [
        ("_validate_something_new", 4)
    ]


@pytest.mark.parametrize("module", READERS)
def test_readers_reach_no_plane(module: str) -> None:
    assert plane_reads((ROOT / module).read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("module", CHECKED)
def test_only_the_reviewed_decodes_remain(module: str) -> None:
    source = (ROOT / module).read_text(encoding="utf-8")
    assert unreviewed_decodes(module, source) == []
    reviewed = {owner for owner, _ in se_decodes(source)}
    stale = {owner for (path, owner) in ALLOWED if path == module} - reviewed
    assert not stale, f"allowed decodes that no longer exist: {sorted(stale)}"
