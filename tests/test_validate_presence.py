"""`validate` applies the presence rule to every association sequence (ADR 0070).

The paired-marker check used to run on Reference-Completed Ragged releases only,
and passed a missing Z beside an infinite SE. Every Ragged component -- an
Observed-Only release and a Hybrid's Ragged Overflow included -- now fails a
cell whose two markers disagree, whichever one is set.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
import zarr
from presence_fixtures import CELLS, build_stores, csr_position, plant

from opengwasdb.encoding.codec import Z_MISSING
from opengwasdb.validation.validate import validate_store

CSR_CASES = ["ragged", "hybrid_overflow"]


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return build_stores(tmp_path_factory.mktemp("validate_presence"))


def _plant_missing_z(store: Path, variant: int, analysis: int) -> None:
    """Z's missing marker at one CSR row, in the sequence and in its by-variant copy."""
    ragged = zarr.open_group(store / "data.zarr" / "ragged", mode="r+")
    ragged["z"][csr_position(ragged, variant, analysis)] = Z_MISSING
    index = ragged["by_variant"]
    start, end = (int(x) for x in index["offsets"][variant : variant + 2])
    rows = np.flatnonzero(np.asarray(index["analysis_index"][start:end]) == analysis)
    index["z"][start + int(rows[0])] = Z_MISSING


@pytest.mark.parametrize("name", ["ragged", "hybrid"])
def test_the_unplanted_stores_validate(stores, name) -> None:
    result = validate_store(stores[name])
    assert result.ok, result.errors


@pytest.mark.parametrize("case", CSR_CASES)
def test_a_lone_missing_se_fails_every_csr(stores, case, tmp_path) -> None:
    store = plant(stores, case, np.nan, tmp_path)
    errors = validate_store(store).errors
    assert any("z and se missingness is inconsistent" in e for e in errors), errors


@pytest.mark.parametrize("case", CSR_CASES)
def test_a_missing_z_beside_an_infinite_se_fails(stores, case, tmp_path) -> None:
    store = plant(stores, case, np.inf, tmp_path)
    _, _, variant, analysis = CELLS[case]
    _plant_missing_z(store, variant, analysis)
    errors = validate_store(store).errors
    assert any("missing z rows have a present se" in e for e in errors), errors


@pytest.mark.parametrize("case", CSR_CASES)
def test_a_present_z_beside_an_infinite_se_is_warned_not_failed(
    stores, case, tmp_path
) -> None:
    store = plant(stores, case, np.inf, tmp_path)
    result = validate_store(store)
    assert not any("missingness" in e for e in result.errors), result.errors


#: A float16 NaN other than `np.nan`'s 0x7e00 (spec §15: a non-canonical marker).
PAYLOAD_NAN = np.array([0x7E01], np.uint16).view(np.float16)[0]


def test_a_non_canonical_dense_marker_fails(stores, tmp_path) -> None:
    store = Path(copytree(stores["dense"], tmp_path / "store"))
    grid = zarr.open_group(store / "data.zarr", mode="r+")
    assert np.isnan(np.asarray(grid["se"][2, 0], dtype=np.float32))  # a missing cell
    grid["se"][2, 0] = PAYLOAD_NAN
    result = validate_store(store)
    assert result.failed() == {"associations.markers_canonical"}, result.errors
    assert any("non-canonical missing marker" in e for e in result.errors)


@pytest.mark.parametrize("case", CSR_CASES)
def test_a_non_canonical_csr_marker_fails(stores, case, tmp_path) -> None:
    store = plant(stores, case, PAYLOAD_NAN, tmp_path)
    _, _, variant, analysis = CELLS[case]
    _plant_missing_z(store, variant, analysis)
    errors = validate_store(store).errors
    assert any("non-canonical missing marker" in e for e in errors), errors
