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

import pytest
import yaml
from architecture_model import (
    REQUIRED_COMPONENT_KEYS,
    ModelError,
    check_model,
    claims,
    component_for_path,
    load_model,
)

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


YAML = """\
repo: opengwas/opengwasdb
layers:
  - {layer: 1, name: Storage and input}
  - {layer: 0, name: Foundations}
components:
  - id: store
    name: Store Release envelope
    container: engine
    layer: 1
    stages: [build]
    summary: Paths, manifest, format versions.
    paths: [opengwasdb/store/]
  - id: model
    name: Store contract model
    container: engine
    layer: 0
    stages: [cross]
    summary: analyses.tsv and the statistic helpers.
    paths: [opengwasdb/model/, opengwasdb/stats.py]
"""


def _write_model(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "components.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_complete_model_loads(tmp_path: Path) -> None:
    model = load_model(_write_model(tmp_path, YAML))

    assert model["repo"] == "opengwas/opengwasdb"
    assert [(c["id"], c["layer"], c["paths"]) for c in model["components"]] == [
        ("store", 1, ["opengwasdb/store/"]),
        ("model", 0, ["opengwasdb/model/", "opengwasdb/stats.py"]),
    ]


@pytest.mark.parametrize("key", REQUIRED_COMPONENT_KEYS)
def test_a_component_missing_a_required_key_fails_naming_it(tmp_path: Path, key: str) -> None:
    incomplete = _component()
    assert key in incomplete  # the key was there to remove
    del incomplete[key]
    path = _write_model(tmp_path, yaml.safe_dump(_model(incomplete, MODEL_COMPONENT)))

    with pytest.raises(ModelError) as error:
        load_model(path)

    assert f"has no '{key}'" in str(error.value)
    # The component is named by id, or by name when the id is what went missing.
    assert ("'store'" in str(error.value)) or ("Store Release envelope" in str(error.value))


def test_two_components_sharing_an_id_fail_naming_it(tmp_path: Path) -> None:
    text = YAML.replace("- id: model", "- id: store")

    with pytest.raises(ModelError) as error:
        load_model(_write_model(tmp_path, text))

    assert str(error.value) == (
        "component 'store' is declared twice; ids must be unique"
        " in docs/architecture/components.yaml"
    )
