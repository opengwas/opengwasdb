from __future__ import annotations

import gzip
import io
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from residual_fixtures import residual_eligible_records

from opengwasdb.build.source import NormalisedAssociation
from opengwasdb.encoding.codec import SeExceptionBuilder, SeExceptionTable, StoreCodec
from opengwasdb.encoding.measure import fit_se_grid
from opengwasdb.encoding.plan import (
    SE_EXCEPTION,
    SE_EXCEPTION_BUDGET,
    EafEncoding,
    EncodingMeasurements,
    SeEncoding,
    SeMeasurements,
    StoreEncoding,
    ZEncoding,
    _decide_se,
)
from opengwasdb.encoding.planes import (
    DenseEafPlane,
    DenseSePlane,
    RaggedSePlane,
    write_se_csr,
    write_se_dense,
)
from opengwasdb.encoding.se import optimise_dense_se_joint
from opengwasdb.layouts.dense.build import build_dense_observed_store
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.dense.top_hits import (
    threshold_key,
    write_top_hit_indexes_for_store,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store
from opengwasdb.validation import validate_store
from opengwasdb.variants import CanonicalVariant


def _plan(residual_range: float = 0.5) -> StoreEncoding:
    return StoreEncoding(
        z=ZEncoding("float16"),
        se=SeEncoding("int8_residual", residual_range),
        eaf=EafEncoding("float32"),
    )


def test_se_manifest_round_trip_and_unknown_kind() -> None:
    encoding = SeEncoding.from_manifest({"kind": "int8_residual", "residual_range": 1.0})
    assert encoding.dtype == "int8"
    assert encoding.to_manifest() == {"kind": "int8_residual", "residual_range": 1.0}
    with pytest.raises(Exception, match="not implemented"):
        SeEncoding.from_manifest({"kind": "surprise"})


def test_se_residual_round_trip_missing_and_exact_exceptions() -> None:
    eaf = np.array([0.1, 0.25, 0.5, 0.8, np.nan], dtype=np.float32)
    coefficients = np.array([[np.log(0.03), -0.5]], dtype=np.float32)
    predictor = np.log(2 * eaf[:4] * (1 - eaf[:4]))
    se = np.array(np.exp(coefficients[0, 0] + coefficients[0, 1] * predictor), dtype=np.float32)
    se = np.concatenate([se, [np.nan]]).astype(np.float32)
    se[1] *= np.float32(np.exp(0.1))
    se[2] = np.float32(0.0)  # exact exception
    se[3] *= np.float32(np.exp(4.0))  # out-of-range exact exception
    positions = np.arange(len(se), dtype=np.int64)
    builder = SeExceptionBuilder()
    codec = StoreCodec(_plan())
    raw = codec.encode_se(
        se,
        eaf=eaf,
        analysis_index=np.zeros(len(se), dtype=np.int64),
        coefficients=coefficients,
        positions=positions,
        exceptions=builder,
    )
    assert raw.dtype == np.int8
    assert raw[-1] == -128
    assert raw[2] == raw[3] == -127
    decoded = StoreCodec(_plan(), se_exceptions=builder.table()).decode_se(
        raw,
        eaf=eaf,
        analysis_index=np.zeros(len(se), dtype=np.int64),
        coefficients=coefficients,
        positions=positions,
    )
    assert np.isnan(decoded[-1])
    assert decoded[2] == se[2]
    assert decoded[3] == se[3]
    np.testing.assert_allclose(decoded[:2], se[:2], rtol=0.01)


def test_se_residual_requires_eaf_and_valid_coefficients() -> None:
    codec = StoreCodec(_plan(), se_exceptions=SeExceptionTable.empty())
    raw = np.array([0], dtype=np.int8)
    with pytest.raises(ValueError, match="EAF"):
        codec.decode_se(
            raw, eaf=np.array([np.nan]), analysis_index=np.array([0]), coefficients=np.ones((1, 2))
        )
    with pytest.raises(ValueError, match="shape"):
        codec.decode_se(
            raw, eaf=np.array([0.2]), analysis_index=np.array([0]), coefficients=np.ones((1, 3))
        )
    with pytest.raises(ValueError, match="finite"):
        codec.decode_se(
            raw,
            eaf=np.array([0.2]),
            analysis_index=np.array([0]),
            coefficients=np.array([[np.nan, 1.0]]),
        )


def test_decision_chooses_smallest_candidate_only_when_all_gates_pass() -> None:
    measured = SeMeasurements(
        eligible=True,
        exception_fraction={0.5: 0.03, 1.0: 0.01, 2.0: 0.0},
        worst_relative_error={0.5: 0.002, 1.0: 0.004, 2.0: 0.008},
        compressed_bytes={0.5: 50, 1.0: 60, 2.0: 70},
        float16_compressed_bytes=100,
    )
    assert StoreEncoding.decide(EncodingMeasurements(1, se=measured)).se == SeEncoding(
        "int8_residual", 1.0
    )
    assert (
        StoreEncoding.decide(EncodingMeasurements(1, se=SeMeasurements(eligible=False))).se.kind
        == "float16"
    )


def test_dense_plane_writer_exposes_only_physical_se(tmp_path) -> None:
    group = zarr.open_group(str(tmp_path / "data.zarr"), mode="w")
    eaf = np.array([[0.1, 0.2], [0.3, 0.4], [0.45, 0.49]], dtype=np.float32)
    coefficients = np.array([[-3.0, -0.5], [-2.5, -0.45]], dtype=np.float32)
    x = np.log(2 * eaf * (1 - eaf))
    se = np.exp(coefficients[None, :, 0] + coefficients[None, :, 1] * x).astype(np.float32)
    se[1, 1] = 0.0
    group.create_dataset("eaf", data=eaf, chunks=(2, 2), dtype="float32")
    group.create_dataset("z", data=np.ones_like(eaf), chunks=(2, 2), dtype="float16")
    plan = _plan(1.0)
    write_se_dense(group, StoreCodec(plan), se, eaf, coefficients, chunks=(2, 2))

    plane = DenseSePlane.open(group, plan)
    np.testing.assert_allclose(plane.band(0, 3), se, rtol=0.01)
    np.testing.assert_allclose(plane.column(1), se[:, 1], rtol=0.01)
    np.testing.assert_allclose(
        plane.points(np.array([0, 1]), np.array([0, 1])),
        np.array([se[0, 0], 0.0]),
        rtol=0.01,
    )
    assert group["se"].dtype == np.dtype("int8")


def test_ragged_plane_uses_csr_ordinals_for_exact_exceptions(tmp_path) -> None:
    group = zarr.open_group(str(tmp_path / "ragged.zarr"), mode="w")
    eaf = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    ai = np.array([0, 0, 1, 1], dtype=np.int64)
    coefficients = np.array([[-3.0, -0.5], [-2.5, -0.45]], dtype=np.float32)
    se = np.exp(coefficients[ai, 0] + coefficients[ai, 1] * np.log(2 * eaf * (1 - eaf)))
    se = se.astype(np.float32)
    se[2] = 0.0
    group.create_dataset("offsets", data=np.array([0, 2, 4]), dtype="int64")
    group.create_dataset("variant_index", data=np.arange(4), dtype="int32")
    group.create_dataset("eaf", data=eaf, dtype="float32")
    write_se_csr(group, StoreCodec(_plan(1.0)), se, eaf, ai, coefficients)

    plane = RaggedSePlane.open(group, _plan(1.0))
    np.testing.assert_allclose(plane.slice(0, 2, analysis_index=0), se[:2], rtol=0.01)
    assert plane.at(np.array([2]), analysis_index=np.array([1]))[0] == 0.0
    assert group["se_exception_index"][:].tolist() == [2]


def _build_residual_dense_store(tmp_path):
    records, expected = residual_eligible_records()
    store = tmp_path / "dense.opengwasdb"
    build_dense_observed_store(
        records,
        store,
        store_id="s",
        release_id="r",
        reference_assembly="GRCh38",
        chunk_shape=(100, 2),
    )
    return store, expected


def test_dense_source_to_query_uses_residual_se_end_to_end(tmp_path) -> None:
    store, expected = _build_residual_dense_store(tmp_path)

    assert StoreManifest.load(store).encoding.se.is_residual
    with query_store(store) as query:
        for analysis_id in expected:
            result = query.analysis(analysis_id)
            np.testing.assert_allclose(result["se"], expected[analysis_id], rtol=0.01)
        assert len(query.top_hits(threshold=5e-8)["se"]) == 12
    assert validate_store(store).ok


def test_validation_rejects_malformed_residual_se_side_arrays(tmp_path) -> None:
    store, _ = _build_residual_dense_store(tmp_path)

    def malformed(name):
        path = tmp_path / f"{name}.opengwasdb"
        shutil.copytree(store, path)
        return path, zarr.open_group(str(path / "data.zarr"), mode="a")

    path, group = malformed("missing-table")
    del group["se_exception_value"]
    assert any("missing required arrays" in error for error in validate_store(path).errors)

    path, group = malformed("bad-coefficients")
    group["se_coefficients"][0, 0] = np.nan
    assert any("non-finite" in error for error in validate_store(path).errors)

    path, group = malformed("stray-exception")
    del group["se_exception_index"]
    del group["se_exception_value"]
    group.create_dataset("se_exception_index", data=np.array([0], dtype=np.int64))
    group.create_dataset("se_exception_value", data=np.array([0.1], dtype=np.float32))
    assert any("not marked" in error for error in validate_store(path).errors)

    path, group = malformed("duplicate-exception")
    del group["se_exception_index"]
    del group["se_exception_value"]
    group.create_dataset("se_exception_index", data=np.array([1, 1], dtype=np.int64))
    group.create_dataset("se_exception_value", data=np.array([0.1, 0.1], dtype=np.float32))
    assert any("duplicates" in error for error in validate_store(path).errors)

    path, group = malformed("dtype-disagreement")
    raw = np.asarray(group["se"][:], dtype=np.float16)
    chunks = group["se"].chunks
    del group["se"]
    group.create_dataset("se", data=raw, chunks=chunks, dtype="float16")
    assert any("dtype float16" in error for error in validate_store(path).errors)


def test_hybrid_joint_selection_streams_dense_and_uses_one_fit(tmp_path) -> None:
    group = zarr.open_group(str(tmp_path / "dense.zarr"), mode="w")
    eaf = np.linspace(0.05, 0.95, 600, dtype=np.float32)[:, None]
    eaf = np.repeat(eaf, 2, axis=1)
    ai = np.broadcast_to(np.arange(2), eaf.shape)
    coefficients = np.array([[-3.0, -0.5], [-2.7, -0.45]], dtype=np.float32)
    dense_se = np.exp(
        coefficients[None, :, 0]
        + coefficients[None, :, 1] * np.log(2 * eaf * (1 - eaf))
        + 0.1 * np.sin(np.arange(len(eaf))[:, None] * 0.1)
    ).astype(np.float32)
    group.create_dataset("eaf", data=eaf, chunks=(100, 2), dtype="float32")
    group.create_dataset("se", data=dense_se, chunks=(100, 2), dtype="float16")
    group.create_dataset("z", data=np.ones_like(eaf), chunks=(100, 2), dtype="float16")
    preliminary = StoreEncoding(
        z=ZEncoding("float16"),
        se=SeEncoding("float16"),
        eaf=EafEncoding("float32"),
    )
    extra_eaf = eaf.ravel().copy()
    extra_ai = ai.ravel().copy()
    extra_se = dense_se.ravel().copy()

    selected, shared_coefficients = optimise_dense_se_joint(
        group,
        preliminary,
        overflow=(extra_se, extra_eaf, extra_ai),
        overflow_chunk=200,
    )

    assert selected.se.is_residual
    assert shared_coefficients is not None
    assert group["se"].dtype == np.dtype("int8")
    np.testing.assert_allclose(
        DenseSePlane.open(group, selected).band(0, len(eaf)),
        dense_se.astype(np.float16).astype(np.float32),
        rtol=0.01,
    )


@pytest.mark.parametrize(
    ("extra_se", "extra_eaf"),
    [
        (np.array([0.1, 0.2], dtype=np.float32), np.array([0.2, 0.3], dtype=np.float32)),
        (
            np.full(600, 0.1, dtype=np.float32),
            np.full(600, np.nan, dtype=np.float32),
        ),
    ],
    ids=["overflow-size-gate", "overflow-missing-eaf"],
)
def test_hybrid_extra_component_can_force_shared_float16_fallback(
    tmp_path, extra_se, extra_eaf
) -> None:
    group = zarr.open_group(str(tmp_path / "dense.zarr"), mode="w")
    eaf = np.linspace(0.05, 0.95, 600, dtype=np.float32)[:, None]
    dense_se = np.exp(-3.0 - 0.5 * np.log(2 * eaf * (1 - eaf))).astype(np.float32)
    group.create_dataset("eaf", data=eaf, chunks=(100, 1), dtype="float32")
    group.create_dataset("se", data=dense_se, chunks=(100, 1), dtype="float16")
    group.create_dataset("z", data=np.ones_like(eaf), chunks=(100, 1), dtype="float16")
    preliminary = StoreEncoding(
        z=ZEncoding("float16"),
        se=SeEncoding("float16"),
        eaf=EafEncoding("float32"),
    )

    selected, shared_coefficients = optimise_dense_se_joint(
        group,
        preliminary,
        overflow=(extra_se, extra_eaf, np.zeros(len(extra_se), dtype=np.int64)),
    )

    assert selected.se == SeEncoding("float16")
    assert shared_coefficients is None
    assert group["se"].dtype == np.dtype("float16")


def test_inline_top_hit_index_carries_plane_decoded_se(tmp_path) -> None:
    """ADR 0040: a derived index carries the values a query reads back.

    An inline VCF/Hybrid build still holds the source's exact SE in memory when
    it writes the index. Once the plane is residual-coded those are no longer
    the same number, and it is the plane a query answers from.
    """
    store, expected = _build_residual_dense_store(tmp_path)
    encoding = StoreManifest.load(store).encoding
    assert encoding.se.is_residual

    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    decoded = DenseSePlane.open(root, encoding).band(0, int(root["se"].shape[0]))
    rows = np.array([0, 100, 200, 300], dtype=np.int64)
    cols = np.zeros(len(rows), dtype=np.int64)
    source_se = expected["a"][rows]
    plane_se = decoded[rows, cols]
    # The fixture only tests anything if quantisation moved the values: were the
    # plane exact, writing either array would give the same index.
    assert not np.array_equal(source_se, plane_se)

    write_top_hit_indexes_for_store(
        store, rows, cols, np.full(len(rows), 8.0, dtype=np.float32), source_se, encoding
    )
    top = zarr.open_group(str(store / "data.zarr" / "top_hits"), mode="r")[threshold_key(5e-8)]
    np.testing.assert_array_equal(top["se"][:].astype(np.float32), plane_se)


def test_one_badly_fitting_analysis_reverts_the_whole_plane() -> None:
    """#118: the plane reverts when *any* Analysis fits worse than the threshold.

    A pooled exception share lets one GCST007320-shaped Analysis hide behind
    its well-fitting neighbours — which is the case the issue was raised about.
    Nineteen clean Analyses against one whose SE is unrelated to its frequency.
    """
    n_variants, n_clean = 400, 19
    frequencies = np.linspace(0.05, 0.95, n_variants, dtype=np.float64)
    predictor = np.log(2 * frequencies * (1 - frequencies))
    clean = np.exp(-3.0 - 0.5 * predictor)[:, None].repeat(n_clean, axis=1)
    rng = np.random.default_rng(0)
    ragged = np.exp(rng.normal(-3.0, 2.0, n_variants))[:, None]
    se = np.concatenate([clean, ragged], axis=1).astype(np.float32)
    eaf = np.broadcast_to(frequencies[:, None].astype(np.float32), se.shape)

    _, measured = fit_se_grid(se, eaf)

    # The fixture is only meaningful if the clean Analyses really do fit: were
    # every column ragged, any gate at all would reject it.
    clean_only = fit_se_grid(se[:, :n_clean], eaf[:, :n_clean])[1]
    assert max(clean_only.exception_fraction.values()) == 0.0

    assert measured.exception_fraction[0.5] > SE_EXCEPTION_BUDGET
    assert _decide_se(measured) == SeEncoding("float16")


def _write_ld_block(
    block_dir: Path,
    name: str,
    variants: list[tuple[str, float, int]],
    *,
    with_eaf: bool = True,
) -> None:
    """One flat-layout LD block: a Variant table and a gzipped correlation matrix.

    `with_eaf=False` drops the frequency column, which is a panel this pipeline
    is required to complete against (issue #113) but cannot supply frequencies
    from. The table's `SNP` column name is the LD panel format's own header, not
    this project's domain vocabulary (see `opengwasdb.completion.ld_panel`).
    """
    block_dir.mkdir(parents=True, exist_ok=True)
    header = "CHR\tSNP\tOA\tEA\tEAF\tBP" if with_eaf else "CHR\tSNP\tOA\tEA\tBP"
    lines = [header]
    for alid, eaf, bp in variants:
        chrom, _, effect, other = alid.split(":")
        frequency = f"{eaf}\t" if with_eaf else ""
        lines.append(f"{chrom}\t{alid}\t{other}\t{effect}\t{frequency}{bp}")
    (block_dir / f"{name}.tsv").write_text("\n".join(lines) + "\n")

    rng = np.random.default_rng(0)
    a = rng.standard_normal((len(variants), len(variants)))
    ld = a @ a.T + np.eye(len(variants)) * len(variants) * 0.1
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
        for row in ld:
            gz.write(("\t".join(f"{v:.6f}" for v in row) + "\n").encode())
    (block_dir / f"{name}.unphased.vcor1.gz").write_bytes(buffer.getvalue())


def _residual_source_and_panel(
    tmp_path: Path, *, panel_has_eaf: bool = True
) -> tuple[Path, Path, dict[str, np.ndarray]]:
    """A residual-SE Dense source, and an LD panel adding four imputation targets."""
    n = 200
    frequencies = np.linspace(0.05, 0.95, n, dtype=np.float32)
    records: list[NormalisedAssociation] = []
    expected: dict[str, np.ndarray] = {}
    for col, analysis_id in enumerate(("a", "b")):
        values = np.exp(
            (-3.0 + col * 0.2)
            - 0.5 * np.log(2 * frequencies * (1 - frequencies))
            + 0.12 * np.sin(np.arange(n) * (0.07 + col * 0.01))
        ).astype(np.float32)
        expected[analysis_id] = values
        records.extend(
            NormalisedAssociation(
                analysis_id=analysis_id,
                variant=CanonicalVariant("1", (row + 1) * 1000, "A", "G"),
                z=8.0 if row % 50 == 0 else 1.0,
                se=float(values[row]),
                eaf=float(frequencies[row]),
            )
            # `b` leaves the last four variants unobserved, so completion has
            # somewhere to impute into a store that already has every row.
            for row in range(n if analysis_id == "a" else n - 4)
        )
    source = tmp_path / "obs.opengwasdb"
    build_dense_observed_store(
        records,
        source,
        store_id="s",
        release_id="obs",
        reference_assembly="GRCh38",
        chunk_shape=(100, 2),
    )

    panel = tmp_path / "ld_panel"
    _write_ld_block(
        panel / "EUR" / "1",
        "1000-200000",
        [
            (f"1:{(row + 1) * 1000}:A:G", float(frequencies[row]), (row + 1) * 1000)
            for row in range(n)
        ],
        with_eaf=panel_has_eaf,
    )
    return source, panel, expected


def test_dense_completion_round_trips_imputed_cells_under_a_residual_plan(tmp_path) -> None:
    """#118: imputed cells round-trip on the same terms as observed ones.

    Completion patches the source's plane in place, so it must reuse the
    source's coefficients: a refit would re-point every carried-over code at a
    new model (ADR 0037 §3).
    """
    source, panel, expected = _residual_source_and_panel(tmp_path)
    assert StoreManifest.load(source).encoding.se.is_residual

    completed = tmp_path / "comp.opengwasdb"
    complete_dense_store(source, completed, panel, ancestry="EUR", min_cor=0.0, release_id="comp")

    manifest = StoreManifest.load(completed)
    assert manifest.encoding.se == StoreManifest.load(source).encoding.se
    source_root = zarr.open_group(str(source / "data.zarr"), mode="r")
    completed_root = zarr.open_group(str(completed / "data.zarr"), mode="r")
    np.testing.assert_array_equal(
        completed_root["se_coefficients"][:], source_root["se_coefficients"][:]
    )

    result = validate_store(completed)
    assert result.ok, result.errors

    imputed = completed_root["imputed"][:]
    # The fixture only tests imputation if completion actually imputed something.
    assert imputed.sum() > 0

    decoded = DenseSePlane.open(completed_root, manifest.encoding).band(
        0, int(completed_root["se"].shape[0])
    )
    assert np.all(np.isfinite(decoded[imputed == 1]))
    assert np.all(decoded[imputed == 1] > 0)

    with query_store(completed) as query:
        observed = query.analysis("a", observed_only=True)
        assert len(observed["se"]) == len(expected["a"])
        np.testing.assert_allclose(observed["se"], expected["a"], rtol=0.01)


def test_validation_catches_a_top_hit_index_left_behind_by_a_migration(tmp_path) -> None:
    """A stale index differs from its plane by the coding's half step, not more.

    Re-encoding an existing store's `se` without rebuilding its top-hit index
    leaves the index holding pre-quantisation values. Measured on the FinnGen
    R13 pilot the gap was 1.970e-03 relative — exactly `expm1(0.5/254)` — and
    5.0e-04 absolute, which an `atol` of 1e-3 waves through. The check has to
    be tight enough to see it.
    """
    store, expected = _build_residual_dense_store(tmp_path)
    assert validate_store(store).ok

    top = zarr.open_group(str(store / "data.zarr" / "top_hits"), mode="a")[threshold_key(5e-8)]
    rows = top["variant_index"][:].astype(np.int64)
    cols = top["analysis_index"][:].astype(np.int64)
    stale = np.array(
        [expected["a" if col == 0 else "b"][row] for row, col in zip(rows, cols, strict=True)],
        dtype=np.float32,
    )
    # The fixture only tests anything if the source and the plane really differ,
    # and only tests the *tolerance* if the gap is small in absolute terms.
    decoded = top["se"][:].astype(np.float32)
    assert not np.array_equal(stale, decoded)
    assert np.max(np.abs(stale - decoded)) < 1e-3

    top["se"][:] = stale
    result = validate_store(store)
    assert not result.ok
    assert any("se value inconsistent" in error for error in result.errors), result.errors


def test_encoding_a_finite_se_without_eaf_is_refused_at_the_source() -> None:
    """A residual plane must not hold a finite `se` at a cell with no EAF (#159).

    The codec used to turn such a cell into an exact exception, which stored the
    value but left the plane outside the contract #118 and #138-#140 describe --
    a residual plane over a complete-EAF store. Encoding is where that has to
    fail: a decode-time refusal rejects data the encoder itself just wrote, and
    a store that only this package can read is the failure this project exists
    to avoid.
    """
    eaf = np.array([0.2, np.nan], dtype=np.float32)
    se = np.array([0.05, 0.04], dtype=np.float32)
    coefficients = np.array([[np.log(0.03), -0.5]], dtype=np.float32)
    codec = StoreCodec(_plan())
    with pytest.raises(ValueError, match="finite EAF"):
        codec.encode_se(
            se,
            eaf=eaf,
            analysis_index=np.zeros(2, dtype=np.int64),
            coefficients=coefficients,
            positions=np.arange(2, dtype=np.int64),
            exceptions=SeExceptionBuilder(),
        )


def test_decoding_refuses_any_non_missing_cell_without_eaf() -> None:
    """The decode-side half of the same rule (#159).

    An exception code is no longer exempt. It cannot arise from a missing
    frequency any more, so a plane that has one there did not come from this
    encoder, and guessing what it meant is worse than refusing it.
    """
    codec = StoreCodec(_plan())
    codec.se_exceptions = SeExceptionTable(
        np.array([1], dtype=np.int64), np.array([0.04], dtype=np.float32)
    )
    raw = np.array([0, SE_EXCEPTION], dtype=np.int8)
    with pytest.raises(ValueError, match="finite EAF"):
        codec.decode_se(
            raw,
            eaf=np.array([0.2, np.nan], dtype=np.float32),
            analysis_index=np.zeros(2, dtype=np.int64),
            coefficients=np.array([[np.log(0.03), -0.5]], dtype=np.float32),
            positions=np.arange(2, dtype=np.int64),
        )


def test_residual_completion_never_writes_se_without_eaf(tmp_path) -> None:
    """Completion cannot produce the cell #159 forbids, and this pins why.

    The concern was that a frequency-less LD panel would leave completion
    writing a finite imputed `se` beside a NaN frequency -- a cell a residual
    plane may not hold, and which completion could not drop to `float16` to
    accommodate, because it writes into the source's own arrays and therefore
    its encoding (ADR 0038 §4).

    It cannot happen, and not by luck: an imputed standard error is *derived
    from* the panel frequency (`se_scale / sqrt(2f(1-f))`, `impute.py`), so a
    cell with no frequency gets no standard error either. Completion therefore
    needs no refusal of its own. This test exists so that stays true -- an
    imputation that ever learned to produce `se` without `eaf` would fail here
    rather than at some later store's decode.
    """
    source, panel, _ = _residual_source_and_panel(tmp_path, panel_has_eaf=False)
    assert StoreManifest.load(source).encoding.se.is_residual

    completed = tmp_path / "no-freq.opengwasdb"
    complete_dense_store(
        source, completed, panel, ancestry="EUR", min_cor=0.0, release_id="no-freq"
    )

    manifest = StoreManifest.load(completed)
    assert manifest.encoding.se.is_residual
    root = zarr.open_group(str(completed / "data.zarr"), mode="r")
    n_rows = int(root["se"].shape[0])
    se = DenseSePlane.open(root, manifest.encoding).band(0, n_rows)
    eaf = DenseEafPlane.open(root, manifest.encoding).band(0, n_rows)

    # The fixture is only meaningful if the panel really supplied no frequency:
    # otherwise every cell has one and the assertion below is vacuous.
    assert not np.any(np.isfinite(eaf[~np.isfinite(se)])) or np.any(~np.isfinite(eaf))
    assert np.all(np.isfinite(eaf[np.isfinite(se)])), "a residual se cell has no eaf"
    assert validate_store(completed).ok
