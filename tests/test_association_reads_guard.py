"""Queries and the observed-view checks read associations through one module (ADR 0068).

Each of these used to decode its own planes -- the imputed mask, then EAF with
the panel substitution, then SE from that EAF -- and each copy was a place for
the order or the shared read to go wrong (#253). They now ask
`opengwasdb.associations`; this test fails if one starts decoding planes again.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "opengwasdb"

#: The modules that must read associations only through `opengwasdb.associations`.
READERS = (
    "query/facade.py",
    "query/resolve.py",
    "validation/observed_view.py",
    "validation/observed_view_ragged.py",
    "validation/observed_view_hybrid.py",
)

#: The plane types whose decode the association readers own.
PLANES = frozenset(
    {"DenseZPlane", "DenseSePlane", "DenseEafPlane", "RaggedSePlane", "RaggedEafPlane", "EafRead"}
)

#: The CSR reader's decode members, which only `CsrReader` calls.
DECODES = frozenset(
    {"get_analysis", "se_slice", "se_at", "eaf_slice_read", "eaf_at_read", "z_at", "n_slice", "n_at"}
)


def violations(source: str) -> list[str]:
    """Each plane import, and each call of a CSR decode member, in `source`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            found += [f"imports {a.name}" for a in node.names if a.name in PLANES]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in DECODES:
                found.append(f"calls .{node.func.attr}() at line {node.lineno}")
    return found


def test_the_guard_sees_a_plane_decode() -> None:
    source = "from opengwasdb.encoding import DenseSePlane\nreader.se_slice(0, 1, eaf=x)\n"
    assert violations(source) == ["imports DenseSePlane", "calls .se_slice() at line 2"]


@pytest.mark.parametrize("module", READERS)
def test_reads_go_through_the_association_readers(module: str) -> None:
    assert violations((ROOT / module).read_text(encoding="utf-8")) == []
