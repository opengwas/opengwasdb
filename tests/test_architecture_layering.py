"""The engine's layering: the import graph, and the baseline of upward imports.

Issue #359, decision 8 on opengwas/opengwasdb-stores#416. `components.yaml`
declares six layers and says a component may import from its own layer or below.
The code does not obey that everywhere, and enforcing it outright would block
current work, so the imports that go against it are measured and listed in
`docs/architecture/layering-baseline.yaml`. CI fails on an upward import the
baseline does not list -- a new one -- and on a listed import that has gone, so
the list only ever shrinks.

The fixtures are module sources written by hand and a model written by hand, and
every expectation is a literal. An expectation computed the way the checker
computes it would pass by construction and could never disagree with it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from architecture_model import (
    BaselineError,
    ImportEdge,
    check_layering,
    engine_sources,
    import_graph,
    load_baseline,
    load_model,
    upward_imports,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "docs" / "architecture" / "components.yaml"
BASELINE_PATH = ROOT / "docs" / "architecture" / "layering-baseline.yaml"


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


def test_an_upward_import_the_baseline_does_not_list_is_reported() -> None:
    graph = import_graph(
        MODEL,
        {
            "opengwasdb/model/manifest.py": (
                "from opengwasdb.layouts.dense.build_vcf import _read_manifest\n"
            ),
            "opengwasdb/layouts/dense/build_vcf.py": "def _read_manifest() -> None: ...\n",
        },
    )

    assert check_layering(MODEL, graph, baseline=()) == [
        "opengwasdb/model/manifest.py: imports opengwasdb.layouts.dense.build_vcf._read_manifest"
        " from component 'model' (layer 0) into 'dense' (layer 3), which is higher; move the"
        " import or the claim, because docs/architecture/layering-baseline.yaml only shrinks"
    ]


def test_an_upward_import_the_baseline_lists_is_allowed() -> None:
    graph = import_graph(
        MODEL,
        {
            "opengwasdb/model/manifest.py": (
                "from opengwasdb.layouts.dense.build_vcf import _read_manifest\n"
            ),
            "opengwasdb/layouts/dense/build_vcf.py": "def _read_manifest() -> None: ...\n",
        },
    )
    assert len(graph) == 1  # the fixture has the upward import the baseline lists

    baseline = (
        ("opengwasdb/model/manifest.py", "opengwasdb.layouts.dense.build_vcf._read_manifest"),
    )

    assert check_layering(MODEL, graph, baseline) == []


def test_a_same_layer_or_downward_import_is_not_reported() -> None:
    graph = import_graph(
        MODEL,
        {
            "opengwasdb/readers/interface.py": "from opengwasdb.encoding.plan import Plan\n",
            "opengwasdb/encoding/plan.py": "Plan = object\n",
            "opengwasdb/layouts/dense/build_vcf.py": "from opengwasdb.model.enum import Row\n",
            "opengwasdb/model/enum.py": "Row = object\n",
        },
    )
    assert [(edge.source, edge.destination) for edge in graph] == [
        ("dense", "model"),
        ("readers", "encoding"),
    ]

    assert check_layering(MODEL, graph, baseline=()) == []


def test_a_baseline_entry_whose_import_is_gone_is_reported() -> None:
    baseline = (("opengwasdb/model/manifest.py", "opengwasdb.readers.interface.Kind"),)

    assert check_layering(MODEL, (), baseline) == [
        "opengwasdb/model/manifest.py: imports opengwasdb.readers.interface.Kind, listed in"
        " docs/architecture/layering-baseline.yaml, is no longer an upward import; delete the entry"
    ]


def test_a_baseline_entry_whose_import_is_now_downward_is_reported() -> None:
    """A claim move can pay off an entry without touching the import; the entry still goes."""
    graph = import_graph(
        MODEL,
        {
            "opengwasdb/layouts/dense/build_vcf.py": "from opengwasdb.model.enum import Row\n",
            "opengwasdb/model/enum.py": "Row = object\n",
        },
    )
    assert [(edge.source, edge.destination) for edge in graph] == [("dense", "model")]
    baseline = (("opengwasdb/layouts/dense/build_vcf.py", "opengwasdb.model.enum.Row"),)

    assert check_layering(MODEL, graph, baseline) == [
        "opengwasdb/layouts/dense/build_vcf.py: imports opengwasdb.model.enum.Row, listed in"
        " docs/architecture/layering-baseline.yaml, is no longer an upward import; delete the entry"
    ]


BASELINE_YAML = """\
# Reviewed by a person; it only shrinks.
allowed:
  - file: opengwasdb/variants/reference.py
    name: opengwasdb.layouts.dense.build_vcf._read_manifest
  - file: opengwasdb/model/manifest.py
    name: opengwasdb.encoding.StoreEncoding
"""


def _write_baseline(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "layering-baseline.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_baseline_loads_as_importing_file_and_imported_name_pairs(tmp_path: Path) -> None:
    assert load_baseline(_write_baseline(tmp_path, BASELINE_YAML)) == (
        ("opengwasdb/model/manifest.py", "opengwasdb.encoding.StoreEncoding"),
        ("opengwasdb/variants/reference.py", "opengwasdb.layouts.dense.build_vcf._read_manifest"),
    )


def test_an_entry_listed_twice_fails_naming_it(tmp_path: Path) -> None:
    text = BASELINE_YAML + (
        "  - file: opengwasdb/model/manifest.py\n    name: opengwasdb.encoding.StoreEncoding\n"
    )

    path = _write_baseline(tmp_path, text)

    with pytest.raises(BaselineError) as error:
        load_baseline(path)

    assert str(error.value) == (
        f"{path}: opengwasdb/model/manifest.py imports opengwasdb.encoding.StoreEncoding"
        " is listed twice"
    )


def test_an_entry_that_is_not_a_file_and_name_pair_fails(tmp_path: Path) -> None:
    text = "allowed:\n  - file: opengwasdb/model/manifest.py\n"

    with pytest.raises(BaselineError) as error:
        load_baseline(_write_baseline(tmp_path, text))

    assert "entry #0 must be a mapping with only 'file' and 'name'" in str(error.value)


# --------------------------------------------------------------------------
# This repository: the measurement, and the baseline it is held to
# --------------------------------------------------------------------------

#: The 23 upward imports the notes for this ticket expect, name for name
#: (`notes/359-expected-upward.txt`, measured on main at 2ef556c with #358's four
#: claim moves). Nine component edges, seven files, ten of them from one file. A
#: difference here is main having moved, and has to be explained name by name
#: rather than fitted to.
EXPECTED_UPWARD = [
    ("opengwasdb/ancestry/subset.py", "opengwasdb.layouts.dense.constants.DEFAULT_CHUNK_SHAPE"),
    (
        "opengwasdb/ancestry/subset.py",
        "opengwasdb.layouts.hybrid.build.build_hybrid_from_vcf_manifest",
    ),
    ("opengwasdb/build/liftover.py", "opengwasdb.readers.gwas_vcf.project_lifted_alleles"),
    (
        "opengwasdb/layouts/hybrid/checkpoint.py",
        "opengwasdb.completion.checkpoint.checkpoint_dir_for",
    ),
    (
        "opengwasdb/layouts/hybrid/checkpoint.py",
        "opengwasdb.completion.checkpoint.require_fresh_destination",
    ),
    ("opengwasdb/model/info_score_policy.py", "opengwasdb.readers.gwas_ssf.GWAS_SSF_CAPABILITY"),
    (
        "opengwasdb/model/info_score_policy.py",
        "opengwasdb.readers.interface.ImputationScoreDeclaration",
    ),
    ("opengwasdb/model/info_score_policy.py", "opengwasdb.readers.interface.ImputationScoreKind"),
    ("opengwasdb/model/manifest.py", "opengwasdb.encoding.StoreEncoding"),
    ("opengwasdb/model/manifest.py", "opengwasdb.encoding.UnsupportedEncoding"),
    ("opengwasdb/model/manifest.py", "opengwasdb.store.open.check_format_version"),
    ("opengwasdb/store/open.py", "opengwasdb.query.facade.StoreQuery"),
    ("opengwasdb/store/open.py", "opengwasdb.query.facade.query_store"),
    ("opengwasdb/variants/reference.py", "opengwasdb.layouts.dense.build_vcf._ManifestRow"),
    ("opengwasdb/variants/reference.py", "opengwasdb.layouts.dense.build_vcf._Pass1Stats"),
    ("opengwasdb/variants/reference.py", "opengwasdb.layouts.dense.build_vcf._ShardSpec"),
    ("opengwasdb/variants/reference.py", "opengwasdb.layouts.dense.build_vcf._WindowShards"),
    ("opengwasdb/variants/reference.py", "opengwasdb.layouts.dense.build_vcf._alid_sort_key"),
    (
        "opengwasdb/variants/reference.py",
        "opengwasdb.layouts.dense.build_vcf._consume_manifest_shards",
    ),
    ("opengwasdb/variants/reference.py", "opengwasdb.layouts.dense.build_vcf._fork_pool"),
    ("opengwasdb/variants/reference.py", "opengwasdb.layouts.dense.build_vcf._iter_pass1_shard"),
    ("opengwasdb/variants/reference.py", "opengwasdb.layouts.dense.build_vcf._read_manifest"),
    ("opengwasdb/variants/reference.py", "opengwasdb.readers.column_map.ColumnMap"),
]


def test_this_repository_has_no_upward_import_the_baseline_does_not_list() -> None:
    sources = engine_sources(ROOT)
    assert len(sources) > 100  # the listing is the engine's source, not a corner of it
    model = load_model(MODEL_PATH)

    graph = import_graph(model, sources)
    assert len(graph) > 500  # a graph this size is the code's imports, not a few of them

    assert check_layering(model, graph, load_baseline(BASELINE_PATH)) == []


def test_the_measured_upward_imports_are_the_twenty_three_the_notes_expect() -> None:
    model = load_model(MODEL_PATH)
    upward = upward_imports(model, import_graph(model, engine_sources(ROOT)))

    assert [(edge.file, edge.name) for edge in upward] == EXPECTED_UPWARD
    assert len({(edge.source, edge.destination) for edge in upward}) == 9
    assert len({edge.file for edge in upward}) == 7
    assert len(load_baseline(BASELINE_PATH)) == 23
