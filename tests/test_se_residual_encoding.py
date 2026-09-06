from __future__ import annotations

import shutil

import numpy as np
import pytest
import zarr

from opengwasdb.build.source import NormalisedAssociation
from opengwasdb.encoding.codec import SeExceptionBuilder, SeExceptionTable, StoreCodec
from opengwasdb.encoding.plan import (
    EafEncoding,
    EncodingMeasurements,
    SeEncoding,
    SeMeasurements,
    StoreEncoding,
    ZEncoding,
)
from opengwasdb.encoding.planes import (
    DenseSePlane,
    RaggedSePlane,
    write_se_csr,
    write_se_dense,
)
from opengwasdb.encoding.se import optimise_dense_se_joint
from opengwasdb.layouts.dense.build import build_dense_observed_store
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
    records: list[NormalisedAssociation] = []
    expected: dict[str, np.ndarray] = {}
    frequencies = np.linspace(0.05, 0.95, 600, dtype=np.float32)
    for col, analysis_id in enumerate(("a", "b")):
        values = np.exp(
            (-3.0 + col * 0.2)
            - 0.5 * np.log(2 * frequencies * (1 - frequencies))
            + 0.12 * np.sin(np.arange(len(frequencies)) * (0.07 + col * 0.01))
        ).astype(np.float32)
        expected[analysis_id] = values
        records.extend(
            NormalisedAssociation(
                analysis_id=analysis_id,
                variant=CanonicalVariant("1", row + 1, "A", "G"),
                z=8.0 if row % 100 == 0 else 1.0,
                se=float(values[row]),
                eaf=float(frequencies[row]),
            )
            for row in range(len(frequencies))
        )
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
        extra=(extra_se, extra_eaf, extra_ai),
        extra_chunk=200,
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
        extra=(extra_se, extra_eaf, np.zeros(len(extra_se), dtype=np.int64)),
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
