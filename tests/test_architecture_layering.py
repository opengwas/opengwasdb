"""The engine's import graph, measured from the AST (issue #359).

Decision 8 on opengwas/opengwasdb-stores#416. `components.yaml` declares six
layers and says a component may import from its own layer or below; the code does
not obey that everywhere. Measuring it starts here: every import in `opengwasdb/`
that crosses a component boundary, as an edge between two components.

The fixtures are module sources written by hand and a model written by hand, and
every expectation is a literal. An expectation computed the way the checker
computes it would pass by construction and could never disagree with it.
"""

from __future__ import annotations

from typing import Any

from architecture_model import ImportEdge, import_graph


def _component(component_id: str, name: str, layer: int, paths: list[str]) -> dict[str, Any]:
    """One fixture component, complete except for the fields the model requires."""
    return {
        "id": component_id,
        "name": name,
        "container": "engine",
        "layer": layer,
        "stages": ["cross"],
        "summary": f"The {name} component.",
        "paths": paths,
    }


MODEL: dict[str, Any] = {
    "repo": "opengwas/opengwasdb",
    "layers": [
        {"layer": 3, "name": "Layouts"},
        {"layer": 1, "name": "Storage and input"},
        {"layer": 0, "name": "Foundations"},
    ],
    "components": [
        _component("model", "Store contract model", 0, ["opengwasdb/model/"]),
        _component("readers", "Source Readers", 1, ["opengwasdb/readers/"]),
        _component("encoding", "Store Encoding", 1, ["opengwasdb/encoding/"]),
        _component("dense", "Dense Layout", 3, ["opengwasdb/layouts/dense/"]),
    ],
}


def test_an_absolute_import_is_one_edge_between_the_two_components() -> None:
    sources = {
        "opengwasdb/model/manifest.py": "from opengwasdb.readers.interface import Kind\n",
        "opengwasdb/readers/interface.py": "Kind = object\n",
    }

    assert import_graph(MODEL, sources) == (
        ImportEdge(
            file="opengwasdb/model/manifest.py",
            name="opengwasdb.readers.interface.Kind",
            source="model",
            destination="readers",
        ),
    )


def test_a_relative_import_is_resolved_against_the_importing_package() -> None:
    """`..readers` from `opengwasdb.model.manifest` is `opengwasdb.readers`; `.enum` is its own."""
    sources = {
        "opengwasdb/model/manifest.py": (
            "from .enum import Row\nfrom ..readers.interface import Kind\n"
        ),
        "opengwasdb/model/enum.py": "Row = object\n",
        "opengwasdb/readers/interface.py": "Kind = object\n",
    }

    assert [edge.name for edge in import_graph(MODEL, sources)] == [
        "opengwasdb.readers.interface.Kind"
    ]


def test_a_function_local_or_type_only_import_counts() -> None:
    """Decision 8: they still couple the code, and #416's numbers count them."""
    sources = {
        "opengwasdb/model/manifest.py": (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from opengwasdb.readers.interface import Kind\n"
            "\n"
            "def load():\n"
            "    from opengwasdb.readers.interface import Reader\n"
            "    return Reader\n"
        ),
        "opengwasdb/readers/interface.py": "Kind = object\nReader = object\n",
    }

    assert [(edge.name, edge.destination) for edge in import_graph(MODEL, sources)] == [
        ("opengwasdb.readers.interface.Kind", "readers"),
        ("opengwasdb.readers.interface.Reader", "readers"),
    ]


def test_a_module_imported_whole_and_a_name_from_it_both_resolve_to_its_component() -> None:
    """`from pkg import module` names the module; `from pkg.module import name` names a
    part of it. Both are imports of the module's component, and the baseline keeps
    them apart because the names differ."""
    sources = {
        "opengwasdb/model/manifest.py": (
            "from opengwasdb.layouts import dense\n"
            "from opengwasdb.layouts.dense.build_vcf import _read_manifest\n"
        ),
        "opengwasdb/layouts/__init__.py": "",
        "opengwasdb/layouts/dense/__init__.py": "",
        "opengwasdb/layouts/dense/build_vcf.py": "def _read_manifest() -> None: ...\n",
    }

    edges = [(edge.name, edge.source, edge.destination) for edge in import_graph(MODEL, sources)]

    assert edges == [
        ("opengwasdb.layouts.dense", "model", "dense"),
        ("opengwasdb.layouts.dense.build_vcf._read_manifest", "model", "dense"),
    ]


def test_an_import_within_one_component_is_not_an_edge() -> None:
    sources = {
        "opengwasdb/model/manifest.py": "from opengwasdb.model.enum import Row\n",
        "opengwasdb/model/enum.py": "Row = object\n",
    }

    assert import_graph(MODEL, sources) == ()
