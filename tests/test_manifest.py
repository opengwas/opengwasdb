import json

import pytest

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
    assert loaded.association_coverage is AssociationCoverage.GWAS
    assert loaded.completion_state is CompletionState.OBSERVED_ONLY


# The four Association Coverages a Store Release can declare (issue #298).
# `full` is not one of them: it is the pre-#298 spelling of `gwas`, and the
# manifest below is the fixture that proves a published Store still opens.
@pytest.mark.parametrize(
    "written",
    ["gwas", "cis", "cis_and_signals", "signals_only"],
)
def test_manifest_round_trips_every_association_coverage(tmp_path, written):
    _write_manifest(tmp_path, association_coverage=written)

    loaded = StoreManifest.load(tmp_path)

    assert loaded.association_coverage.value == written
    assert loaded.to_dict()["association_coverage"] == written


def test_manifest_reading_full_gives_gwas_without_rewriting_it(tmp_path):
    """Every release built before #298 says `full`. It still opens, and reads
    as the value #298 renamed it to, so no published Store needs a rewrite."""
    _write_manifest(tmp_path, association_coverage="full")

    loaded = StoreManifest.load(tmp_path)

    assert loaded.association_coverage is AssociationCoverage.GWAS
    # A build writes `gwas`; the legacy spelling is read, never propagated.
    assert loaded.to_dict()["association_coverage"] == "gwas"


def test_a_provenance_amendment_writes_the_coverage_vocabulary(dense_store_path):
    """`amend_provenance` is the one in-place edit of a built release, and it
    rewrites the manifest as JSON: the coverage it leaves behind is the
    source's spelling unless it normalises. An in-place edit that keeps `full`
    alive is the retired spelling surviving a write this build made (#298)."""
    import json

    from opengwasdb.store import open_store

    manifest_path = dense_store_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["association_coverage"] = "full"
    manifest_path.write_text(json.dumps(manifest))

    open_store(dense_store_path).amend_provenance({"amended_by": "test"})

    written = json.loads(manifest_path.read_text())
    assert written["association_coverage"] == "gwas"
    assert written["provenance"]["amended_by"] == "test"


def test_unknown_association_coverage_is_refused(tmp_path):
    """The vocabulary is closed: a value outside it is an error, never a
    default standing in for it."""
    _write_manifest(tmp_path, association_coverage="something_else")

    with pytest.raises(ValueError, match="something_else"):
        StoreManifest.load(tmp_path)


def test_open_store_opens_exact_path_supplied(tmp_path):
    _write_manifest(tmp_path)

    store = open_store(tmp_path)

    assert store.path == tmp_path
    assert store.index_path == tmp_path / "index.sqlite"
    assert store.data_path == tmp_path / "data.zarr"
