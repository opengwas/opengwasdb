"""The Rho Matrix of a Hybrid release's Dense Component (#496).

`build-dense-rho` takes the Hybrid release and builds the matrix in its Dense Component,
`dense/data.zarr/rho`, over the component's Analyses; it refuses the Dense Component by
the release's name, however the path names it. An Analysis with no observed association
in the Dense Component -- an Overflow-only Analysis -- has no Rho: the matrix records
exactly which they are, and the Store Reader refuses a Rho query naming one rather than
answer NaN.
"""

from __future__ import annotations

import json
import shutil
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
import zarr
from format_0_2_3 import as_0_2_3
from overflow_only_hybrid import (
    ANALYSES,
    DENSE_ANALYSES,
    OVERFLOW_ONLY,
    RHO_OPTIONS,
    UNTHINNED_ONLY,
    UNTHINNED_VARIANT,
    overflow_only_hybrid,
)
from typer.testing import CliRunner

from opengwasdb.cli.main import app
from opengwasdb.encoding import banding
from opengwasdb.layouts.dense.rho import (
    RhoBuildPathError,
    build_dense_rho,
    estimate_rho_cml,
    select_thinned_variants,
)
from opengwasdb.query import query_store
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store.open import open_store
from opengwasdb.store_reader.analyses import UnknownAnalysisError
from opengwasdb.store_reader.encoding.planes import DenseZPlane
from opengwasdb.store_reader.layout import ArrayRole
from opengwasdb.store_reader.rho import OVERFLOW_ONLY_ATTR, OverflowOnlyAnalysisError
from opengwasdb.validation import validate_store

OVERFLOW_ONLY_INDEX = ANALYSES.index(OVERFLOW_ONLY)
UNTHINNED_ONLY_INDEX = ANALYSES.index(UNTHINNED_ONLY)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The release as built, never written to: copy it to change it."""
    return overflow_only_hybrid(tmp_path_factory.mktemp("overflow-only-hybrid"))


@pytest.fixture
def hybrid(built: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(built, tmp_path / "hybrid.opengwasdb"))


def _rho_group(release: Path, mode: str = "r") -> zarr.Group:
    return zarr.open_group(release / "dense" / "data.zarr" / "rho", mode=mode)


def _thinned_rows(component) -> np.ndarray:
    with closing(component.variant_axis(aliases=False)) as axis:
        records = axis.all()
    return select_thinned_variants(
        np.array([r.chromosome for r in records]),
        np.array([r.position for r in records]),
        RHO_OPTIONS["window_bp"],
    )


def _record(release: Path, overflow_only: list) -> None:
    _rho_group(release, "r+").attrs[OVERFLOW_ONLY_ATTR] = overflow_only


def test_the_fixture_has_each_kind_of_analysis(built: Path):
    """The fixture means something: `trait_c` has no observed cell in the Dense
    Component; `trait_d` has one, at a panel row thinning drops; the two others have one
    at every panel row; and the Overflow holds rows on both chromosomes."""
    component = open_store(built).dense_component()
    z = DenseZPlane.open(component.arrays(), component.manifest.encoding)
    grid = z.band(0, z.shape[0])
    n_panel = z.shape[0]
    assert np.isfinite(grid).sum(axis=0).tolist() == [n_panel, n_panel, 0, 1]

    (unthinned_row,) = np.flatnonzero(np.isfinite(grid[:, UNTHINNED_ONLY_INDEX]))
    with closing(component.variant_axis(aliases=False)) as axis:
        record = axis.by_indices(np.array([unthinned_row]))[int(unthinned_row)]
    assert (record.chromosome, record.position) == UNTHINNED_VARIANT
    assert unthinned_row not in _thinned_rows(component)

    panel = set(np.load(built / "dense" / "dense_to_shared.npy").tolist())
    query = query_store(built)
    try:
        assert [row["analysis_id"] for _, row in sorted(query.analyses_table().items())] == list(
            ANALYSES
        )
        assert len(query.analysis(OVERFLOW_ONLY)["z"]) > 0
        off_panel = [index for index in query.variants_table() if index not in panel]
        chromosomes = {query.variants_table()[index]["chromosome"] for index in off_panel}
    finally:
        query.close()
    # Shared rows are in genome order, so Overflow rows on both chromosomes interleave.
    assert chromosomes == {"1", "2"}
    assert min(off_panel) < max(panel) and max(off_panel) > min(panel)


def test_build_on_the_hybrid_writes_the_dense_components_matrix_and_validates(hybrid: Path):
    build_dense_rho(hybrid, **RHO_OPTIONS)

    assert not (hybrid / "data.zarr" / "rho").exists()
    group = _rho_group(hybrid)
    assert group.attrs["n_analyses"] == len(ANALYSES)
    # `trait_d` is a Dense Analysis: observed at an unthinned row only, it has no null
    # support, but it is not Overflow-only.
    assert group.attrs[OVERFLOW_ONLY_ATTR] == [OVERFLOW_ONLY_INDEX]
    rho = group["rho"][:]
    n_null = group["n_null"][:]
    # Pairs in `np.tril_indices(4, -1)` order: (b, a) first, then every pair with c or d.
    assert np.isfinite(rho[0]) and n_null[0] >= RHO_OPTIONS["min_nulls"]
    assert np.isnan(rho[1:]).all() and (n_null[1:] == 0).all()
    report = validate_store(hybrid)
    assert report.ok, report.errors


def test_the_dense_pair_is_the_estimate_from_the_components_own_z(hybrid: Path):
    """What the Hybrid path stores is what the estimator gives on the Dense Component's
    observed Z at the thinned panel rows: the component is read, not the Overflow."""
    build_dense_rho(hybrid, **RHO_OPTIONS)
    component = open_store(hybrid).dense_component()
    z = DenseZPlane.open(component.arrays(), component.manifest.encoding)
    grid = z.rows(_thinned_rows(component))
    want = estimate_rho_cml(
        grid[:, 1], grid[:, 0], RHO_OPTIONS["z_thresh"], RHO_OPTIONS["min_nulls"]
    )
    assert float(_rho_group(hybrid)["rho"][0]) == pytest.approx(want, abs=1e-3)


def test_the_dense_component_path_is_refused_by_the_hybrids_name(hybrid: Path):
    with pytest.raises(RhoBuildPathError) as refused:
        build_dense_rho(hybrid / "dense", **RHO_OPTIONS)
    assert str(hybrid) in str(refused.value)
    assert "build-dense-rho" in str(refused.value)
    assert not (hybrid / "dense" / "data.zarr" / "rho").exists()


@pytest.mark.parametrize("alias", ["symlink", "dotdot"])
def test_an_alias_of_the_dense_component_is_refused_as_dense_is(
    hybrid: Path, tmp_path: Path, alias: str
):
    """The path is judged where it leads, not by how it is spelt: a symlink named
    otherwise, or `dense/data.zarr/..`, is the Dense Component all the same."""
    if alias == "symlink":
        path = tmp_path / "component"
        path.symlink_to(hybrid / "dense", target_is_directory=True)
    else:
        path = hybrid / "dense" / "data.zarr" / ".."
    with pytest.raises(RhoBuildPathError) as refused:
        build_dense_rho(path, **RHO_OPTIONS)
    assert f"build-dense-rho {hybrid.resolve()}" in str(refused.value)
    assert not (hybrid / "dense" / "data.zarr" / "rho").exists()


def test_a_ragged_release_is_refused(range_releases: dict[str, Path], tmp_path: Path):
    ragged = Path(shutil.copytree(range_releases["ragged"], tmp_path / "ragged"))
    with pytest.raises(RhoBuildPathError, match="Ragged"):
        build_dense_rho(ragged)
    assert not (ragged / "data.zarr" / "rho").exists()


def test_the_cli_takes_the_hybrid_and_refuses_its_dense_component(hybrid: Path):
    options = [
        "--window-bp", str(RHO_OPTIONS["window_bp"]),
        "--z-thresh", str(RHO_OPTIONS["z_thresh"]),
        "--min-nulls", str(RHO_OPTIONS["min_nulls"]),
    ]
    runner = CliRunner()
    refused = runner.invoke(app, ["build-dense-rho", str(hybrid / "dense"), *options])
    assert refused.exit_code == 1
    assert f"build-dense-rho {hybrid.resolve()}" in refused.output
    assert not (hybrid / "dense" / "data.zarr" / "rho").exists()

    done = runner.invoke(app, ["build-dense-rho", str(hybrid), *options])
    assert done.exit_code == 0, done.output
    assert (hybrid / "dense" / "data.zarr" / "rho").is_dir()


def _complete_in_name(release: Path, imputed_columns: list[int] | None) -> None:
    """Declare the Dense Component Reference-Completed; with `imputed_columns`, give it
    an imputed mask flagging every observed cell of those Analyses imputed."""
    manifest_path = release / "dense" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completion_state"] = "reference_completed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if imputed_columns is None:
        return
    component = open_store(release).dense_component()
    z = DenseZPlane.open(component.arrays(), component.manifest.encoding)
    grid = z.band(0, z.shape[0])
    mask = np.zeros(grid.shape, dtype=np.uint8)
    mask[:, imputed_columns] = np.isfinite(grid[:, imputed_columns])
    root = store_arrays.open_group_for_write(release / "dense" / "data.zarr", "a")
    store_arrays.create_array(
        root, "imputed", ArrayRole.DENSE_IMPUTED_MASK, data=mask,
        compressor=store_arrays.compressor(),
    )


def test_a_completed_component_without_its_mask_is_refused(hybrid: Path):
    """Read without its imputed mask, every imputed Z would pass for observed."""
    _complete_in_name(hybrid, None)
    with pytest.raises(ValueError, match="imputed mask"):
        build_dense_rho(hybrid, **RHO_OPTIONS)
    assert not (hybrid / "dense" / "data.zarr" / "rho").exists()


def test_imputed_z_is_never_observed(hybrid: Path):
    """An Analysis whose every Dense cell is imputed has no observed association there:
    it is Overflow-only, and has no null support with anyone."""
    _complete_in_name(hybrid, [0])
    build_dense_rho(hybrid, **RHO_OPTIONS)
    group = _rho_group(hybrid)
    assert group.attrs[OVERFLOW_ONLY_ATTR] == [0, OVERFLOW_ONLY_INDEX]
    assert (group["n_null"][:] == 0).all()


def test_the_reader_answers_for_dense_analyses_and_refuses_an_overflow_only_one(hybrid: Path):
    build_dense_rho(hybrid, **RHO_OPTIONS)
    stored = float(_rho_group(hybrid)["rho"][0])
    query = query_store(hybrid)
    try:
        matrix = query.rho_matrix()
        assert matrix["analysis_id"].tolist() == list(DENSE_ANALYSES)
        assert float(matrix["rho"][0, 1]) == stored
        # `trait_d` is Dense without support: NaN, as for a Dense Store's unsupported pair.
        assert np.isnan(matrix["rho"][2, :2]).all() and (matrix["n_null"][2, :2] == 0).all()

        row = query.rho_row("trait_a")
        assert row["analysis_id"].tolist() == ["trait_b", UNTHINNED_ONLY]
        assert float(row["rho"][0]) == stored

        pair = query.rho("trait_b", "trait_a")
        assert pair["analysis_id_a"].tolist() == ["trait_b"]
        assert float(pair["rho"][0]) == stored
        assert len(query.rho(["trait_a", "trait_b", UNTHINNED_ONLY])["rho"]) == 3

        for ask in (
            lambda: query.rho("trait_a", OVERFLOW_ONLY),
            lambda: query.rho([OVERFLOW_ONLY]),
            lambda: query.rho_row(OVERFLOW_ONLY),
            lambda: query.rho_matrix(["trait_a", OVERFLOW_ONLY]),
        ):
            with pytest.raises(OverflowOnlyAnalysisError) as refused:
                ask()
            assert refused.value.fields == {"analysis_id": OVERFLOW_ONLY}
        # An Analysis the release does not hold is refused as unknown first.
        with pytest.raises(UnknownAnalysisError):
            query.rho(OVERFLOW_ONLY, "absent")
    finally:
        query.close()


def test_without_a_rho_matrix_the_hybrid_answers_empty_and_still_refuses_unknown(hybrid: Path):
    query = query_store(hybrid)
    try:
        assert query.rho_matrix()["rho"].shape == (0, 0)
        assert len(query.rho_row(OVERFLOW_ONLY)["rho"]) == 0
        with pytest.raises(UnknownAnalysisError):
            query.rho_row("absent")
    finally:
        query.close()


def test_a_matrix_that_does_not_record_its_overflow_only_analyses_is_refused(hybrid: Path):
    """A Dense Component's Rho Matrix built before #496 does not say which Analyses have no
    Rho, so a NaN in it could be either; the reader refuses it, and so does validation."""
    build_dense_rho(hybrid, **RHO_OPTIONS)
    del _rho_group(hybrid, "r+").attrs[OVERFLOW_ONLY_ATTR]
    query = query_store(hybrid)
    try:
        with pytest.raises(ValueError, match=OVERFLOW_ONLY_ATTR):
            query.rho_matrix()
    finally:
        query.close()
    assert "rho.overflow_only" in validate_store(hybrid).failed()


@pytest.mark.parametrize(
    "recorded",
    [
        pytest.param([], id="an-overflow-only-analysis-left-out"),
        pytest.param([OVERFLOW_ONLY_INDEX, UNTHINNED_ONLY_INDEX], id="a-dense-analysis-listed"),
        pytest.param([0, OVERFLOW_ONLY_INDEX], id="a-supported-dense-analysis-listed"),
    ],
)
def test_validation_requires_exactly_the_overflow_only_analyses(hybrid: Path, recorded: list):
    """Membership is a fact of the Dense Component -- no observed association there --
    not of null support: `trait_d` has none and is Dense, and a record that leaves
    `trait_c` out would let a query answer NaN for it."""
    build_dense_rho(hybrid, **RHO_OPTIONS)
    _record(hybrid, recorded)
    assert validate_store(hybrid).failed() == {"rho.overflow_only"}


@pytest.mark.parametrize(
    "recorded",
    [["2"], [2.0], [True, 2], [2, 2], [5], -1],
    ids=["string", "float", "bool", "repeated", "out-of-range", "not-a-list"],
)
def test_a_malformed_record_is_refused_not_coerced(hybrid: Path, recorded: object):
    build_dense_rho(hybrid, **RHO_OPTIONS)
    _record(hybrid, recorded)
    query = query_store(hybrid)
    try:
        with pytest.raises(ValueError, match=OVERFLOW_ONLY_ATTR):
            query.rho_row("trait_a")
    finally:
        query.close()
    assert "rho.overflow_only" in validate_store(hybrid).failed()


def test_the_0_2_dense_component_builds_the_same_matrix(hybrid: Path, tmp_path: Path):
    """On 0.2 the component is read through its own Variant Table; on 0.3.0 through the
    shared axis and `dense_to_shared.npy`. Both thin the same rows to the same matrix."""
    legacy = Path(shutil.copytree(hybrid, tmp_path / "legacy.opengwasdb"))
    as_0_2_3(legacy)
    assert open_store(legacy / "dense").manifest.format_version == "0.2.3"
    build_dense_rho(hybrid, **RHO_OPTIONS)
    build_dense_rho(legacy, **RHO_OPTIONS)
    ours, theirs = _rho_group(hybrid), _rho_group(legacy)
    for name in ("rho", "n_null", "variant_index"):
        np.testing.assert_array_equal(ours[name][:], theirs[name][:])
    assert dict(ours.attrs) == dict(theirs.attrs)
    report = validate_store(legacy)
    assert report.ok, report.errors


def _manifests(release: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(release)): path.read_bytes()
        for path in sorted(release.rglob("manifest.json"))
    }


def test_the_manifests_declare_the_matrix_as_a_dense_stores_do(
    hybrid: Path, dense_store_path: Path
):
    """A Dense Store's manifest does not declare its Rho Matrix: the matrix is opt-in,
    add-in-place, and found by its group (ADR 0025). A Hybrid's two manifests are left
    the same way, byte for byte."""
    dense_before = _manifests(dense_store_path)
    hybrid_before = _manifests(hybrid)
    assert set(hybrid_before) == {"manifest.json", "dense/manifest.json"}

    build_dense_rho(dense_store_path)
    build_dense_rho(hybrid, **RHO_OPTIONS)

    assert (dense_store_path / "data.zarr" / "rho").is_dir()
    assert _manifests(dense_store_path) == dense_before
    assert _manifests(hybrid) == hybrid_before


def test_the_existing_rho_rules_judge_the_dense_components_matrix(hybrid: Path):
    """Not skipped, and not passed for want of a target: none of them skips the Dense
    Component, and an out-of-range rho there fails `rho.values`."""
    build_dense_rho(hybrid, **RHO_OPTIONS)
    clean = validate_store(hybrid)
    assert clean.ok, clean.errors
    assert not [f for f in clean.skips if f.rule.startswith("rho.")]

    group = _rho_group(hybrid, "r+")
    rho = group["rho"][:]
    rho[0] = 1.5
    group["rho"][:] = rho
    report = validate_store(hybrid)
    assert report.failed() == {"rho.values"}, report.errors
    assert [f.target for f in report.of("rho.values")] == ["dense/data.zarr"], report.errors


def test_a_build_read_in_many_batches_writes_the_same_matrix(
    hybrid: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The thinned rows, and the scan for Analyses not seen there, are read a batch at a
    time under the decoded-band budget; batches that small change nothing stored."""
    whole = Path(shutil.copytree(hybrid, tmp_path / "whole.opengwasdb"))
    build_dense_rho(whole, **RHO_OPTIONS)
    rows_per_batch = 7
    monkeypatch.setattr(
        banding, "DECODED_BAND_BYTES", rows_per_batch * banding.BYTES_PER_CELL * len(ANALYSES)
    )
    assert banding.band_rows(len(ANALYSES), 0) == rows_per_batch
    assert _rho_group(whole)["variant_index"].shape[0] > 10 * rows_per_batch

    build_dense_rho(hybrid, **RHO_OPTIONS)
    ours, theirs = _rho_group(hybrid), _rho_group(whole)
    for name in ("rho", "n_null", "variant_index"):
        np.testing.assert_array_equal(ours[name][:], theirs[name][:])
    assert dict(ours.attrs) == dict(theirs.attrs)
