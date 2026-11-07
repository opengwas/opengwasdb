"""Associations are decoded in one module, with reviewed exceptions (ADR 0070).

The query facade and the observed-view checks used to decode their own planes --
the imputed mask, then EAF with the panel substitution, then SE from that EAF --
and each copy was a place for the order or the shared read to go wrong (#253).
They now ask `opengwasdb.associations`, and must not touch a plane at all.

Validation -- every module under `validation/` that is not a reader, the
Validation Rules and their scans included (ADR 0074) -- and the Indexed Variant
Subset's validation keep decodes of their own, named below by the function each
sits in. SE decoding is recognised by what is
called -- the codec's `decode_se`, an SE plane type named (to open one, or to
annotate a parameter that receives one), or an SE plane read -- so a decode
anywhere else in those modules fails, however it is spelled.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "opengwasdb"

#: Modules that read associations only through `opengwasdb.associations`.
READERS = (
    "layouts/dense/complete.py",
    "layouts/hybrid/complete.py",
    "layouts/ragged/complete.py",
    "query/facade.py",
    "query/resolve.py",
    "validation/observed_view.py",
    "validation/observed_view_ragged.py",
    "validation/observed_view_hybrid.py",
)

#: Modules that use no other module's private names: Hybrid Reference Completion
#: read the Overflow's private arrays and imported the Hybrid builder's table
#: writer by its private name (#391).
OWN_NAMES_ONLY = ("layouts/hybrid/complete.py",)

#: The planes a reader module reaches that hold no association, and why.
READER_PLANES = {
    ("layouts/dense/complete.py", "imports DenseEafPlane"):
        "the frequency pass carries each observed cell's stored frequency across the "
        "row remap, after the statistic pass has read and judged the cell through the reader",
}

#: The one SE decode in a reader module: it reads no plane, but re-encodes the
#: source's crossover cells and decodes them, to predict what completion stored.
READER_ROUND_TRIPS = {("validation/observed_view_hybrid.py", "round_trip")}

#: Every module under `validation/`.
VALIDATION = tuple(
    sorted(str(path.relative_to(ROOT)) for path in (ROOT / "validation").rglob("*.py"))
)

#: Modules that keep reviewed decodes of their own, and only those: every validation
#: module that is not a reader -- so a new rule or scan module cannot decode SE unseen
#: -- and the Indexed Variant Subset's validation.
CHECKED = (
    *(module for module in VALIDATION if module not in READERS),
    "layouts/dense/indexed_subsets.py",
)

#: The reviewed SE decodes, by the function (or class) each sits in.
ALLOWED = {
    # Each shares an EAF read the check already holds.
    ("validation/scans.py", "_dense_band_se"):
        "one SE decode per band, from the codes the band already holds and its own EAF "
        "read, shared by every value rule that folds over the band",
    ("validation/scans.py", "_sequence_window_se"):
        "one SE decode per sequence window, from the codes the window already holds and "
        "its own EAF read, shared by every value rule that folds over the window",
    ("layouts/dense/indexed_subsets.py", "_compare_indexed_band"):
        "the primary block the subset's EAF is compared against",
    ("layouts/dense/indexed_subsets.py", "_sample_decoded_view"):
        "the sampled primary cells the subset's EAF is compared against",
    # The planes those decodes read, and the decoders the readers stand on.
    ("layouts/dense/indexed_subsets.py", "_decode_se_column"):
        "the subset's own decoder, which `SubsetReader` reads through",
    ("layouts/dense/indexed_subsets.py", "_se_band_error"):
        "the publication gate decoding the subset's own SE band against the primary's",
    ("layouts/dense/indexed_subsets.py", "_open_decode_context"):
        "opens the primary SE plane the two comparisons above decode",
    ("layouts/dense/indexed_subsets.py", "_DecodeContext"):
        "carries that plane between them",
}

#: The plane types whose decode the association readers own.
PLANES = frozenset(
    {"DenseZPlane", "DenseSePlane", "DenseEafPlane", "RaggedSePlane", "RaggedEafPlane", "EafRead"}
)

#: The CSR reader's decode members, which only `CsrReader` and the modules named
#: below call.
CSR_DECODES = frozenset(
    {
        "z_slice",
        "z_at",
        "se_slice",
        "se_at",
        "eaf_slice_read",
        "eaf_at_read",
        "eaf_at",
        "n_slice",
        "n_at",
    }
)

#: The modules that may call them, and why.
CSR_DECODE_CALLERS = {
    "associations/csr.py": "the CSR adapter",
    "layouts/ragged/zarr_csr.py": "the reader that defines them",
    "layouts/ragged/top_hits.py": (
        "the Top-Hit harvest, a Layout (layer 3) the builders call, which cannot import "
        "the Association Reader (Query, layer 4) without an upward import "
        "(docs/architecture/components.yaml)"
    ),
}

#: The SE plane types: naming one is how a function comes to decode SE.
SE_PLANES = frozenset({"DenseSePlane", "RaggedSePlane"})

#: The codec's SE decode.
SE_CODEC = frozenset({"decode_se"})

#: An SE plane's reads.
SE_READS = frozenset({"band", "column", "row", "rows", "block", "points", "slice", "at"})


def _in_functions(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """Every node with the name of the innermost function or class holding it."""
    found: list[tuple[str, ast.AST]] = []
    owners = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def visit(node: ast.AST, owner: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, owners) else owner
            found.append((name, child))
            visit(child, name)

    visit(tree, "<module>")
    return found


def _reads_an_se_plane(node: ast.Call) -> bool:
    """A plane read handed an EAF array, or made on an `se_plane` or `_se` receiver."""
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in SE_READS:
        return False
    receiver = ast.unparse(node.func.value)
    handed_eaf = any(keyword.arg == "eaf" for keyword in node.keywords)
    return handed_eaf or receiver.endswith(("se_plane", "._se"))


def se_decodes(source: str) -> list[tuple[str, int]]:
    """`(function, line)` of each place `source` decodes SE, by what it calls."""
    found = set()
    for owner, node in _in_functions(ast.parse(source)):
        named = (isinstance(node, ast.Name) and node.id in SE_PLANES) or (
            isinstance(node, ast.Attribute) and node.attr in SE_PLANES | SE_CODEC
        )
        if named or (isinstance(node, ast.Call) and _reads_an_se_plane(node)):
            found.add((owner, node.lineno))
    return sorted(found)


def plane_reads(source: str, module: str = "") -> list[str]:
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
    found += [
        f"decodes SE in {owner} at line {n}"
        for owner, n in se_decodes(source)
        if (module, owner) not in READER_ROUND_TRIPS
    ]
    return found


def unreviewed_decodes(module: str, source: str) -> list[tuple[str, int]]:
    return [(owner, line) for owner, line in se_decodes(source) if (module, owner) not in ALLOWED]


def test_the_guard_sees_a_plane_named_through_its_module() -> None:
    source = (
        "import opengwasdb.encoding as enc\n"
        "def read(group, encoding, eaf):\n"
        "    return enc.DenseSePlane.open(group, encoding).band(0, 1, eaf=eaf)\n"
    )
    assert plane_reads(source) == ["names .DenseSePlane at line 3", "decodes SE in read at line 3"]


#: New decodes beside the reviewed ones, each spelled a way the guard must still see.
BYPASSES = {
    "an SE read handed EAF": (
        "def _new(se_plane, eaf):\n    return se_plane.points(r, c, eaf=eaf)\n"
    ),
    "the codec's decode": (
        "def _new(codec, raw, eaf, ai, coefficients):\n"
        "    return codec.decode_se(raw, eaf=eaf, analysis_index=ai, coefficients=coefficients)\n"
    ),
    "a plane opened through its module": (
        "import opengwasdb.encoding as enc\n"
        "def _new(root, encoding, rows, cols):\n"
        "    return enc.DenseSePlane.open(root, encoding).block(rows, cols)\n"
    ),
    "a plane received and read without EAF": (
        "def _new(se_plane: DenseSePlane, rows, cols):\n    return se_plane.block(rows, cols)\n"
    ),
}


@pytest.mark.parametrize("module", CHECKED)
@pytest.mark.parametrize("bypass", sorted(BYPASSES))
def test_the_guard_sees_a_new_decode_beside_the_reviewed_ones(module: str, bypass: str) -> None:
    found = unreviewed_decodes(module, BYPASSES[bypass])
    assert found and {owner for owner, _ in found} == {"_new"}, found


def test_the_guard_lets_the_reviewed_functions_be() -> None:
    source = "def _dense_band_se(se_plane, eaf):\n    return se_plane.band(0, 1, eaf=eaf)\n"
    assert unreviewed_decodes("validation/scans.py", source) == []


def test_every_validation_module_is_a_reader_or_checked() -> None:
    readers = {module for module in READERS if module.startswith("validation/")}
    assert readers <= set(VALIDATION), "a reader the guard names no longer exists"
    assert "validation/scans.py" in CHECKED, "the band scan must be checked"
    assert set(VALIDATION) == readers | {m for m in CHECKED if m.startswith("validation/")}


@pytest.mark.parametrize("module", READERS)
def test_readers_reach_no_plane(module: str) -> None:
    found = plane_reads((ROOT / module).read_text(encoding="utf-8"), module)
    assert [read for read in found if (module, read) not in READER_PLANES] == []


@pytest.mark.parametrize(("module", "read"), sorted(READER_PLANES))
def test_each_reviewed_reader_plane_is_still_reached(module: str, read: str) -> None:
    assert read in plane_reads((ROOT / module).read_text(encoding="utf-8"), module)


@pytest.mark.parametrize("module", CHECKED)
def test_only_the_reviewed_decodes_remain(module: str) -> None:
    source = (ROOT / module).read_text(encoding="utf-8")
    assert unreviewed_decodes(module, source) == []
    reviewed = {owner for owner, _ in se_decodes(source)}
    stale = {owner for (path, owner) in ALLOWED if path == module} - reviewed
    assert not stale, f"allowed decodes that no longer exist: {sorted(stale)}"


def private_reaches(source: str) -> list[str]:
    """Each private name `source` imports from a module, or reads off another object."""
    found = []
    for _, node in _in_functions(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            found += [f"imports {a.name}" for a in node.names if a.name.startswith("_")]
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            owner = ast.unparse(node.value)
            if not node.attr.startswith("__") and owner not in ("self", "cls"):
                found.append(f"reads {owner}.{node.attr} at line {node.lineno}")
    return found


def test_the_guard_sees_a_private_name_imported_or_read() -> None:
    source = "from m import _hidden\ndef f(csr):\n    return csr._offsets[:]\n"
    assert private_reaches(source) == ["imports _hidden", "reads csr._offsets at line 3"]


@pytest.mark.parametrize("module", OWN_NAMES_ONLY)
def test_a_module_uses_no_other_module_s_private_names(module: str) -> None:
    assert private_reaches((ROOT / module).read_text(encoding="utf-8")) == []


def csr_decode_calls(source: str) -> list[str]:
    """Each call `source` makes to a CSR decode member, by name."""
    return [
        f"calls .{node.func.attr}() at line {node.lineno}"
        for _, node in _in_functions(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in CSR_DECODES
    ]


def test_only_the_csr_adapter_calls_the_csr_decodes() -> None:
    """A sequence's associations are read through `CsrReader`, which applies the
    presence rule, so no module in the package but those named decodes one itself."""
    found = {
        module: calls
        for path in sorted(ROOT.rglob("*.py"))
        if (module := str(path.relative_to(ROOT))) not in CSR_DECODE_CALLERS
        and (calls := csr_decode_calls(path.read_text(encoding="utf-8")))
    }
    assert found == {}
