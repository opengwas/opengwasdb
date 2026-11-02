"""`docs/architecture/components.yaml`: the engine's claims and the checks over them.

Issue #358. The map's ownership half is only worth something if it cannot drift
from the code, so this repository's CI fails here on an engine source file no
component claims, on a claim that matches no tracked file, and on a layer the
model does not declare. The model itself is hand-written; what is checked is
only its agreement with `git ls-files`.

The fixtures are small models written by hand, and every expectation is a
literal. An expectation computed the way the checker computes it would pass by
construction and could never disagree with it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from architecture_model import check_model, claims, component_for_path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "docs" / "architecture" / "components.yaml"

LAYERS = [{"layer": 1, "name": "Storage and input"}, {"layer": 0, "name": "Foundations"}]


def _component(**fields: Any) -> dict[str, Any]:
    """One fixture component, complete except for what the caller overrides."""
    component: dict[str, Any] = {
        "id": "store",
        "name": "Store Release envelope",
        "container": "engine",
        "layer": 1,
        "stages": ["build"],
        "summary": "Paths, manifest, format versions.",
        "paths": ["opengwasdb/store/"],
    }
    component.update(fields)
    return component


def _model(
    *components: dict[str, Any], layers: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "repo": "opengwas/opengwasdb",
        "layers": LAYERS if layers is None else layers,
        "components": list(components),
    }


MODEL_COMPONENT = _component(
    id="model",
    name="Store contract model",
    layer=0,
    stages=["cross"],
    summary="analyses.tsv and the statistic helpers.",
    paths=["opengwasdb/model/", "opengwasdb/stats.py"],
)

TRACKED = [
    "opengwasdb/model/manifest.py",
    "opengwasdb/stats.py",
    "opengwasdb/store/arrays.py",
    "opengwasdb/store/open.py",
]


def test_an_unclaimed_source_file_is_reported_by_path() -> None:
    model = _model(_component(), MODEL_COMPONENT)

    problems = check_model(model, [*TRACKED, "opengwasdb/build/zz.py"])

    assert problems == [
        "opengwasdb/build/zz.py: no component claims this file; add a path for it to a component"
        " in docs/architecture/components.yaml"
    ]


def test_a_directories_nested_files_are_claimed_by_it() -> None:
    problems = check_model(
        _model(_component(), MODEL_COMPONENT), [*TRACKED, "opengwasdb/store/band/arrays.py"]
    )

    assert problems == []


def test_a_tracked_file_that_is_not_python_is_not_a_source_file() -> None:
    problems = check_model(_model(_component(), MODEL_COMPONENT), [*TRACKED, "opengwasdb/NOTES.md"])

    assert problems == []


def test_a_claim_that_matches_no_tracked_file_is_reported() -> None:
    """A claim on code that has been deleted or moved is stale, and says so."""
    maintenance = _component(
        id="maintenance",
        name="Repair and re-encode",
        layer=1,
        stages=["cross"],
        summary="Representation-only repairs.",
        paths=["opengwasdb/repair.py"],
    )

    problems = check_model(
        _model(_component(), MODEL_COMPONENT, maintenance), ["opengwasdb/store/arrays.py"]
    )

    assert problems == [
        "opengwasdb/model/: claimed by component 'model', but no tracked file matches;"
        " fix or remove the claim in docs/architecture/components.yaml",
        "opengwasdb/repair.py: claimed by component 'maintenance', but no tracked file matches;"
        " fix or remove the claim in docs/architecture/components.yaml",
        "opengwasdb/stats.py: claimed by component 'model', but no tracked file matches;"
        " fix or remove the claim in docs/architecture/components.yaml",
    ]


def test_a_component_whose_layer_is_not_declared_is_reported() -> None:
    problems = check_model(_model(_component(layer=7), MODEL_COMPONENT), TRACKED)

    assert problems == [
        "component 'store' declares layer 7, which the model's layers do not declare;"
        " fix one of them in docs/architecture/components.yaml"
    ]


def test_a_prefix_two_components_claim_is_reported() -> None:
    """Sharing a prefix would make ownership a guess, so it is an error."""
    resolution = _component(
        id="resolution", name="Analysis Resolution", stages=["resolve"], paths=["opengwasdb/store/"]
    )

    problems = check_model(
        _model(_component(), resolution), ["opengwasdb/store/arrays.py", "opengwasdb/store/open.py"]
    )

    assert problems == [
        "opengwasdb/store/: claimed by 'resolution' and 'store'; give each component its own"
        " paths in docs/architecture/components.yaml"
    ]


def test_a_file_claim_inside_anothers_directory_claim_belongs_to_the_file() -> None:
    """The directory is shared; the file is not. Longest claimed prefix wins."""
    resolution = _component(
        id="resolution",
        name="Analysis Resolution",
        stages=["resolve"],
        paths=["opengwasdb/store/convert.py"],
    )
    model = _model(_component(), MODEL_COMPONENT, resolution)

    assert component_for_path(claims(model), "opengwasdb/store/convert.py") == "resolution"
    assert component_for_path(claims(model), "opengwasdb/store/open.py") == "store"
    assert check_model(model, [*TRACKED, "opengwasdb/store/convert.py"]) == []


def test_nothing_claims_a_path_outside_every_claim() -> None:
    assert component_for_path(claims(_model(_component())), "opengwasdb/cli/main.py") is None
