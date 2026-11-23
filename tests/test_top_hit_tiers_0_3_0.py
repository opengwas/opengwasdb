"""A 0.3.0 Top-Hit tier stores z, and the validator recomputes |z| and p from it (#471).

Before 0.3.0 a tier stored `abs_z` and `p_value` beside `z`. Nothing but the validator
read them, so 0.3.0 drops both (ADR 0102). What they let the validator say still holds:

- `top_hits.threshold` fails a tier one of whose rows has a z that does not clear the
  tier's threshold, on a 0.3.0 release of every layout, judged on the |z| recomputed
  from the tier's own z;
- on a 0.2 release, `top_hits.abs_z` recomputes both from z through
  `opengwasdb.store_reader.stats`, and what it recomputes equals what the release
  stored, within the tolerance the rule has always used; it fails a stored `p_value`
  that disagrees;
- a 0.3.0 tier carrying either array fails `top_hits.arrays`.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from format_0_2_3 import as_0_2_3
from scipy.special import erfc

from opengwasdb import top_hits
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store_reader.layout import TOP_HIT_CHUNK_SIZE, ArrayRole
from opengwasdb.store_reader.stats import z_critical
from opengwasdb.validation import validate_store

#: Each layout, and where its tiers are: a Hybrid's Dense Component's and its Overflow's.
TIERS = {
    "dense_observed": ["data.zarr"],
    "ragged_completed": ["data.zarr"],
    "hybrid_completed": ["dense/data.zarr", "data.zarr"],
}


def _copy(digest_releases: dict[str, Path], tmp_path: Path, name: str) -> Path:
    return Path(shutil.copytree(digest_releases[name], tmp_path / name))


def _tier_with_rows(data: Path) -> tuple[str, int]:
    """The loosest tier holding a row, and how many it holds: a fixture asserted meaningful."""
    root = store_arrays.open_group(data, "r")
    tiers = {key: int(root["top_hits"][key]["z"].shape[0]) for key in root["top_hits"]}
    held = {key: n for key, n in tiers.items() if n}
    assert held, f"{data} holds no Top-Hit row: a corrupted tier would corrupt nothing"
    key = max(held, key=lambda k: float(root["top_hits"][k].attrs["threshold"]))
    return key, held[key]


def _rewrite_tier_array(data: Path, key: str, name: str, values: np.ndarray) -> None:
    root = store_arrays.open_group_for_write(data, "a")
    root["top_hits"][key][name][:] = values


@pytest.mark.parametrize("name", sorted(TIERS))
def test_a_clean_0_3_0_release_passes_the_threshold_rule_and_stores_no_p(
    digest_releases: dict[str, Path], name: str
):
    release = digest_releases[name]
    assert json.loads((release / "manifest.json").read_text())["format_version"] == "0.3.0"
    for data in TIERS[name]:
        root = store_arrays.open_group(release / data, "r")
        for key in root["top_hits"]:
            assert not top_hits.STORED_P_ARRAYS & set(root["top_hits"][key].keys()), (data, key)
    result = validate_store(release)
    assert result.ok, result.errors


@pytest.mark.parametrize(("name", "data"), [(n, d) for n, ds in TIERS.items() for d in ds])
def test_the_threshold_rule_fails_a_tier_whose_z_does_not_clear_it(
    digest_releases: dict[str, Path], tmp_path: Path, name: str, data: str
):
    release = _copy(digest_releases, tmp_path, name)
    key, n_rows = _tier_with_rows(release / data)
    root = store_arrays.open_group(release / data, "r")
    z = np.asarray(root["top_hits"][key]["z"][:], dtype=np.float32)
    cutoff = z_critical(float(root["top_hits"][key].attrs["threshold"]))
    assert np.all(np.abs(z) >= cutoff), "the fixture tier was corrupt before the test"
    z[n_rows // 2] = np.float32(cutoff * 0.5)
    _rewrite_tier_array(release / data, key, "z", z)

    result = validate_store(release)

    assert "top_hits.threshold" in result.failed(), result.errors
    assert any(
        f"top-hit index {key} holds 1 row(s) whose z does not clear its threshold" in message
        for message in result.errors
    ), result.errors


def test_the_threshold_rule_fails_a_nan_z(digest_releases: dict[str, Path], tmp_path: Path):
    release = _copy(digest_releases, tmp_path, "ragged_completed")
    key, _ = _tier_with_rows(release / "data.zarr")
    z = np.asarray(store_arrays.open_group(release / "data.zarr", "r")["top_hits"][key]["z"][:])
    z[0] = np.nan
    _rewrite_tier_array(release / "data.zarr", key, "z", z)
    assert "top_hits.threshold" in validate_store(release).failed()


def _as_0_2_3(digest_releases: dict[str, Path], tmp_path: Path, name: str) -> Path:
    return as_0_2_3(_copy(digest_releases, tmp_path, name))


@pytest.mark.parametrize("name", sorted(TIERS))
def test_the_recomputed_abs_z_and_p_value_are_what_0_2_stored(
    digest_releases: dict[str, Path], tmp_path: Path, name: str
):
    release = _as_0_2_3(digest_releases, tmp_path, name)
    checked = 0
    for data in TIERS[name]:
        root = store_arrays.open_group(release / data, "r")
        for key in root["top_hits"]:
            tier = root["top_hits"][key]
            stored_abs_z, stored_p = tier["abs_z"][:], tier["p_value"][:]
            # As `_write_tier` stored them before #471: |z| as float32, and erfc of it.
            assert np.array_equal(stored_abs_z, np.abs(tier["z"][:]))
            np.testing.assert_allclose(
                stored_p, erfc(stored_abs_z.astype("float64") / math.sqrt(2.0)), rtol=1e-12
            )
            abs_z, p_value = top_hits.recomputed_abs_z_and_p(tier["z"][:])
            np.testing.assert_allclose(abs_z, stored_abs_z, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(p_value, stored_p, rtol=1e-3, atol=0.0)
            assert top_hits.stored_p_agrees(stored_abs_z, stored_p, tier["z"][:])
            checked += len(stored_p)
    assert checked, "no tier row was recomputed"
    result = validate_store(release)
    assert result.ok, result.errors


def test_the_abs_z_rule_fails_a_stored_p_value_that_disagrees(
    digest_releases: dict[str, Path], tmp_path: Path
):
    release = _as_0_2_3(digest_releases, tmp_path, "dense_observed")
    key, _ = _tier_with_rows(release / "data.zarr")
    root = store_arrays.open_group(release / "data.zarr", "r")
    p_value = np.asarray(root["top_hits"][key]["p_value"][:])
    p_value[0] *= 10
    _rewrite_tier_array(release / "data.zarr", key, "p_value", p_value)
    assert "top_hits.abs_z" in validate_store(release).failed()


def test_a_0_3_0_tier_that_still_stores_abs_z_fails_its_arrays_rule(
    digest_releases: dict[str, Path], tmp_path: Path
):
    release = _copy(digest_releases, tmp_path, "ragged_completed")
    key, _ = _tier_with_rows(release / "data.zarr")
    root = store_arrays.open_group_for_write(release / "data.zarr", "a")
    tier = root["top_hits"][key]
    abs_z = np.abs(np.asarray(tier["z"][:], dtype=np.float32))
    store_arrays.create_array(
        tier, "p_value", ArrayRole.TOP_HIT_INDEX,
        data=erfc(abs_z.astype("float64") / math.sqrt(2.0)), dtype="float64",
        compressor=store_arrays.compressor(), hint=TOP_HIT_CHUNK_SIZE,
    )
    result = validate_store(release)
    assert "top_hits.arrays" in result.failed()
    assert any("carries p_value, which format 0.3.0 does not store" in e for e in result.errors)


@pytest.mark.parametrize(
    ("name", "command"),
    [("ragged_observed", "build-ragged-top-hits"), ("dense_observed", "build-dense-top-hits")],
)
def test_a_rebuilt_index_keeps_its_release_s_format(
    digest_releases: dict[str, Path], tmp_path: Path, name: str, command: str
):
    """A rebuild writes 0.3.0 tiers into a 0.3.0 release, and a 0.2 release's keep the
    `abs_z` and `p_value` its format stores (#313, #471); both validate."""
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    current = _copy(digest_releases, tmp_path / "current", name)
    legacy = _as_0_2_3(digest_releases, tmp_path / "legacy", name)
    for release, stored in ((current, set()), (legacy, set(top_hits.STORED_P_ARRAYS))):
        result = CliRunner().invoke(app, [command, str(release)])
        assert result.exit_code == 0, result.output
        root = store_arrays.open_group(release / "data.zarr", "r")
        tiers = list(root["top_hits"])
        assert len(tiers) == 3
        for key in tiers:
            assert top_hits.STORED_P_ARRAYS & set(root["top_hits"][key].keys()) == stored
        assert validate_store(release).ok, (release, validate_store(release).errors)
