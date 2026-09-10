"""Tests for the reproducible pilot-measurement harness (issue #117 evidence).

The driver (`benchmarks/measure_pilot_releases.py`) is the repository-side
measurement harness ADR 0037's byte tables were missing, and the SE/UKB
benchmark scripts now route their artifacts through the same provenance
plumbing (`benchmarks/_artifact.py`). The tests pin the semantics that would
otherwise be silent:

- a measured Dense release reports grid cells, the encoding plan, compressed
  bytes and a validation result;
- a CSR component's cell count comes from its offsets (associations, not a
  sparse grid);
- a missing or unreadable input stops the run loudly instead of publishing a
  partial artifact;
- SE and UKB artifacts carry `commit` and `measured_at`, and their documented
  regeneration commands use Pixi rather than the retired `uv` invocations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import zarr

from benchmarks import _artifact
from benchmarks.measure_pilot_releases import _csr_cells, main, measure_store
from opengwasdb.model.manifest import StoreManifest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _commit_is_short_sha(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{7,40}", value))


def test_dense_release_measurement_records_cells_encoding_bytes_and_validation(
    dense_store_path,
):
    record = measure_store(dense_store_path)
    manifest = StoreManifest.load(dense_store_path)

    root = zarr.open_group(str(dense_store_path / "data.zarr"), mode="r")
    rows, analyses = (int(n) for n in root["z"].shape)
    assert rows > 0 and analyses > 0, "fixture store must be non-empty to mean anything"

    assert record["store_bytes"] > 0
    assert record["primary_layout"] == "dense"
    # The driver reports the release's own identity and plan, not a guess.
    assert record["format_version"] == manifest.format_version
    assert record["encoding"] == manifest.to_dict().get("encoding")
    assert record["encoding"] is not None  # format >= 1.0 must declare its plan

    (component,) = record["components"]
    assert component["kind"] == "dense_grid"
    assert component["cells"] == {
        "n_variants": rows,
        "n_analyses": analyses,
        "n_cells": rows * analyses,
    }
    assert component["data_bytes"] > 0
    assert {"z", "se"} <= set(component["arrays"])
    assert component["arrays"]["z"]["bytes"] > 0
    assert component["arrays"]["z"]["n_cells"] == rows * analyses
    assert component["arrays"]["z"]["bytes_per_cell"] > 0

    assert record["validation"]["ok"] is True, record["validation"]["errors"]
    assert record["identity"]["manifest_sha256"]
    assert record["identity"]["analyses_tsv_sha256"]


def test_csr_cells_count_stored_associations_not_grid_cells(tmp_path):
    """A Ragged CSR component counts its offsets rows, so a sparse store is not
    charged for the empty grid cells a Dense layout would have filled."""
    group = zarr.open_group(str(tmp_path / "data.zarr" / "ragged"), mode="w")
    group.create_dataset("offsets", data=[0, 2, 5], dtype="int64")
    group.create_dataset("z", data=np.zeros(5, dtype=np.int16), chunks=(3,))

    component = _csr_cells(
        store_path=tmp_path,
        zarr_root=tmp_path / "data.zarr",
        group_name="ragged",
        manifest=_ragged_manifest(),
    )

    assert component["kind"] == "ragged_csr"
    assert component["cells"]["n_analyses"] == 2
    assert component["cells"]["n_cells"] == 5
    assert component["arrays"]["z"]["n_cells"] == 5
    assert component["data_bytes"] > 0


def _ragged_manifest():
    from opengwasdb.model.manifest import StoreManifest

    # A minimal Ragged manifest: the driver reads only the encoding block for
    # a CSR component's record, so the rest of the metadata is not exercised
    # here.
    return StoreManifest.from_dict(
        {
            "store_id": "test",
            "release_id": "test-1",
            "format_version": "0.1.0",
            "primary_layout": "ragged",
            "association_coverage": "cis_and_signals",
            "completion_state": "observed_only",
            "reference_assembly": "GRCh38",
            "encoding": {
                "version": 3,
                "z": {"kind": "int16_fixed", "scale": 1024},
                "se": {"kind": "float16"},
                "eaf": {"kind": "absent"},
            },
        }
    )


def test_artifact_write_carries_commit_and_timestamp(dense_store_path, tmp_path):
    output = tmp_path / "measurements.json"
    assert main([str(dense_store_path), "--output", str(output)]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert _commit_is_short_sha(artifact["commit"])
    assert artifact["measured_at"]  # non-empty ISO-8601 timestamp
    assert len(artifact["stores"]) == 1
    assert artifact["stores"][0]["store_id"] != ""
    # The recorded store release is the one we asked it to measure.
    assert Path(artifact["stores"][0]["path"]) == dense_store_path.resolve()


def test_missing_inputs_fail_loudly(tmp_path):
    with pytest.raises(SystemExit, match="no such Store Release"):
        measure_store(tmp_path / "does-not-exist.opengwasdb")

    not_a_release = tmp_path / "not-a-release"
    not_a_release.mkdir()
    with pytest.raises(SystemExit, match="no manifest.json"):
        measure_store(not_a_release)


def test_shared_provenance_records_commit_and_timestamp():
    fields = _artifact.provenance()
    assert _commit_is_short_sha(fields["commit"])
    assert fields["measured_at"]


def test_se_and_ukb_benchmarks_route_through_shared_provenance():
    import benchmarks.benchmark_se_residual_queries as se
    import benchmarks.benchmark_ukbb_dense as ukb

    # The two benchmark modules write through benchmarks/_artifact.py's
    # provenance and writer, so their artifacts cannot disagree about what
    # `commit` and `measured_at` mean or drift a private copy of the helper.
    assert se.provenance is _artifact.provenance
    assert se.write_artifact is _artifact.write_artifact
    assert ukb.provenance is _artifact.provenance
    assert ukb.write_artifact is _artifact.write_artifact


def test_ukb_benchmark_docstring_uses_pixi_not_uv():
    from benchmarks.benchmark_ukbb_dense import __doc__

    assert "pixi run -e dev python benchmarks/benchmark_ukbb_dense.py" in __doc__
    assert "uv run" not in __doc__


def test_readme_documents_omitted_se_ukb_and_pilot_driver_with_pixi_commands():
    readme = (_REPO_ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")

    assert "### `benchmark_se_residual_queries.py`" in readme
    assert "### `benchmark_ukbb_dense.py`" in readme
    assert "### `measure_pilot_releases.py`" in readme
    assert (
        "pixi run -e dev python benchmarks/measure_pilot_releases.py" in readme
    )
    assert (
        "pixi run -e dev python benchmarks/benchmark_se_residual_queries.py" in readme
    )
    assert "pixi run -e dev python benchmarks/benchmark_ukbb_dense.py" in readme
    assert "uv run" not in readme
