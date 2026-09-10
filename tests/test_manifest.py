import json

from opengwasdb.model.enums import (
    AssociationCoverage,
    CompletionState,
    PrimaryStorageLayout,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store import open_store


def _write_manifest(tmp_path, **overrides):
    """The minimal manifest a release must carry, written to `tmp_path`.

    The `encoding` block is part of the minimum (spec §6a): it is required of
    every release, so a fixture that omits it is not a smaller manifest but an
    unreadable one.
    """
    manifest = {
        "store_id": "example",
        "release_id": "observed-1",
        "format_version": "0.1.0",
        "primary_layout": "dense",
        "association_coverage": "full",
        "completion_state": "observed_only",
        "reference_assembly": "GRCh37",
        "encoding": {
            "version": 3,
            "z": {"kind": "int16_fixed", "scale": 1024},
            "se": {"kind": "float16"},
            "eaf": {"kind": "absent"},
        },
        **overrides,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_manifest_loads_from_store_directory(tmp_path):
    _write_manifest(tmp_path, provenance={"source": "fixture"})

    loaded = StoreManifest.load(tmp_path)

    assert loaded.store_id == "example"
    assert loaded.primary_layout is PrimaryStorageLayout.DENSE
    assert loaded.association_coverage is AssociationCoverage.FULL
    assert loaded.completion_state is CompletionState.OBSERVED_ONLY


def test_open_store_opens_exact_path_supplied(tmp_path):
    _write_manifest(tmp_path)

    store = open_store(tmp_path)

    assert store.path == tmp_path
    assert store.index_path == tmp_path / "index.sqlite"
    assert store.data_path == tmp_path / "data.zarr"
