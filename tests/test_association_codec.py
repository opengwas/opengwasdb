"""What a component stores of a batch of associations, and how it encodes them (#350)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest
import zarr

from opengwasdb.admission import NotAdmitted
from opengwasdb.encoding import association_codec
from opengwasdb.encoding.association_codec import (
    SE_FLOAT16_MAX,
    AssociationCodec,
    CellValues,
    Placement,
    log_se_out_of_range,
    storable,
)
from opengwasdb.encoding.codec import EafExceptionBuilder, SeExceptionBuilder, StoreCodec
from opengwasdb.encoding.plan import (
    EAF_EXCEPTION,
    EafEncoding,
    SeEncoding,
    StoreEncoding,
    ZEncoding,
)
from opengwasdb.encoding.planes import (
    DenseEafPlane,
    DenseSePlane,
    DenseZPlane,
    write_dense_cells,
    write_encoded_planes,
)
from opengwasdb.store.arrays import ArrayRole, compressor

NAN = np.float32(np.nan)


def _f32(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


# ── storable: Admission's guarantee, asserted ────────────────────────────────


@pytest.mark.parametrize(
    ("z", "se"),
    [(_f32(1.0, 2.0), _f32(0.1, NAN)), (_f32(1.0, NAN), _f32(0.1, 0.2))],
    ids=["se missing", "z missing"],
)
def test_a_cell_with_one_statistic_missing_is_refused_by_admissions_rule(z, se) -> None:
    message = (
        r"complete-x: cells Association Admission does not admit \(one of z and SE missing: 1\)"
    )
    with pytest.raises(NotAdmitted, match=message + r"; first one of z and SE missing at \(1,\)"):
        storable(CellValues(z, se), 0, 1, where="complete-x")


@pytest.mark.parametrize(
    ("imputed", "rule"),
    [
        (np.array([0, 1], dtype=np.uint8), "imputed flag on a missing cell"),
        (np.array([2, 0], dtype=np.uint8), "imputed flag neither 0 nor 1"),
    ],
)
def test_an_imputed_flag_admission_forbids_is_refused(imputed, rule) -> None:
    cells = CellValues(_f32(1.0, NAN), _f32(0.1, NAN), imputed=imputed)
    with pytest.raises(NotAdmitted, match=rule):
        storable(cells, 0, 1, where="complete-x")


@pytest.mark.parametrize(
    ("z", "se"),
    [(_f32(np.inf), _f32(0.1)), (_f32(1.0), _f32(0.0)), (_f32(1.0), _f32(-0.1))],
    ids=["infinite z", "zero se", "negative se"],
)
def test_a_present_cell_admission_would_not_admit_is_refused(z, se) -> None:
    with pytest.raises(NotAdmitted, match="present with an infinite z"):
        storable(CellValues(z, se), 0, 1, where="complete-x")


#: float16 NaNs other than `np.nan`'s 0x7e00: a payload, a sign and a signalling NaN.
ODD_MARKERS = {"payload": 0x7E01, "negative": 0xFE00, "signalling": 0x7C01}


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("bits", ODD_MARKERS.values(), ids=ODD_MARKERS.keys())
@pytest.mark.parametrize("plane", ["z", "se"])
def test_a_missing_marker_other_than_the_canonical_nan_is_refused(plane, bits, dtype) -> None:
    """Spec §15: a float plane's missing marker is `np.nan`'s NaN, and a builder
    rejects any other. A decoded `float16` plane carries a payload into
    `float32`, and a `float16` plane would store it again as itself."""
    from opengwasdb.associations import non_canonical

    z = np.array([1.0, np.nan], dtype=dtype)
    se = np.array([0.1, np.nan], dtype=dtype)
    {"z": z, "se": se}[plane][1] = np.array([bits], np.uint16).view(np.float16)[0]
    # Fixture first: the cell is missing in both planes, and its marker is odd.
    assert np.isnan(z[1]) and np.isnan(se[1])
    assert non_canonical({"z": z, "se": se}[plane]).tolist() == [False, True]
    with pytest.raises(
        NotAdmitted,
        match=r"complete-x: .*a missing marker other than the canonical NaN: 1\); first .* "
        r"at \(1,\)",
    ):
        storable(CellValues(z, se), 0, 1, where="complete-x")


def test_the_canonical_marker_is_stored() -> None:
    z, se = np.array([1.0, np.nan], np.float16), np.array([0.1, np.nan], np.float16)
    assert int(se[1:].view(np.uint16)[0]) == 0x7E00
    assert storable(CellValues(z, se), 0, 1, where="x").stored.tolist() == [True, True]


def test_an_se_narrowed_to_infinity_is_a_stored_value_and_counted() -> None:
    kept = storable(CellValues(_f32(1.0, NAN), _f32(np.inf, NAN)), 0, 1, where="x")
    assert kept.stored.tolist() == [True, True]
    assert kept.se_out_of_range.tolist() == [1]


@pytest.mark.parametrize(
    ("cells", "named"),
    [
        (CellValues(_f32(1.0, 2.0), _f32(0.1)), "se of length 1"),
        (CellValues(_f32(1.0, 2.0), _f32(0.1, 0.2), _f32(0.3)), "eaf of length 1"),
        (
            CellValues(_f32(1.0, 2.0), _f32(0.1, 0.2), imputed=np.array([1], dtype=np.uint8)),
            "imputed of length 1",
        ),
    ],
    ids=["se", "eaf", "imputed"],
)
def test_a_batch_whose_arrays_are_not_parallel_is_refused(cells, named) -> None:
    """One shape for every array of a batch, and the refusal names each length."""
    with pytest.raises(ValueError, match=f"{named}.*must be parallel"):
        storable(cells, 0, 1, where="complete-x")


def test_the_count_starts_just_above_float16s_largest_value() -> None:
    se = np.array([SE_FLOAT16_MAX, np.nextafter(SE_FLOAT16_MAX, np.inf), 70_000.0, np.inf, 0.5])
    kept = storable(CellValues(np.ones(5), se), np.array([0, 0, 1, 2, 2]), 4, where="x")
    assert SE_FLOAT16_MAX == 65_504.0
    assert kept.se_out_of_range.tolist() == [1, 1, 1, 0]


def test_a_band_counts_each_column_against_its_own_analysis() -> None:
    se = np.full((3, 2), 0.5, dtype=np.float32)
    se[0, 1] = se[2, 1] = np.inf
    kept = storable(CellValues(np.ones((3, 2), np.float32), se), np.array([5, 6]), 7, where="x")
    assert kept.se_out_of_range.tolist() == [0, 0, 0, 0, 0, 0, 2]


def test_storable_changes_nothing_it_keeps_and_is_idempotent() -> None:
    cells = CellValues(_f32(1.0, NAN, 3.0), _f32(0.1, NAN, np.inf), eaf=_f32(0.2, 0.3, NAN))
    once = storable(cells, 0, 1, where="x")
    twice = storable(once.cells, 0, 1, where="x")
    assert once.cells is cells
    assert twice.stored.tolist() == once.stored.tolist() == [True, True, True]
    assert twice.se_out_of_range.tolist() == once.se_out_of_range.tolist() == [1]


def test_a_batch_wider_than_one_window_is_judged_window_by_window(monkeypatch) -> None:
    monkeypatch.setattr(association_codec, "_WINDOW", 4)
    z = np.ones((6, 2), dtype=np.float32)
    se = np.full((6, 2), 0.5, dtype=np.float32)
    se[5, 0] = np.inf
    assert storable(CellValues(z, se), np.arange(2), 2, where="x").se_out_of_range.tolist() == [
        1,
        0,
    ]
    z[4, 1] = np.nan
    with pytest.raises(NotAdmitted, match=r"from index 4 of the batch's first axis.*at \(0, 1\)"):
        storable(CellValues(z, se), np.arange(2), 2, where="x")


def test_the_count_is_logged_once_naming_each_analysis(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="opengwasdb.encoding.association_codec"):
        log_se_out_of_range(np.array([0, 5, 2]), ["a", "b", "c"], where="build-x")
        log_se_out_of_range(np.zeros(3, dtype=np.int64), ["a", "b", "c"], where="build-x")
        log_se_out_of_range(np.array([3]), where="build-y, analysis_index 4")
    tally = "SE value(s) above 65,504, more than a float16 plane holds (spec §15)"
    assert [r.getMessage() for r in caplog.records] == [
        f"build-x: 7 {tally}: b: 5, c: 2",
        f"build-y, analysis_index 4: 3 {tally}",
    ]


# ── Placement: where a batch sits ────────────────────────────────────────────


def test_a_sequence_whose_analysis_index_is_not_parallel_is_refused() -> None:
    """The run's variant index and its Analysis index describe the same cells."""
    with pytest.raises(
        ValueError,
        match="Placement.sequence: variant_index of length 3 and analysis_index of length 2 "
        "must be parallel",
    ):
        Placement.sequence(0, np.arange(3), np.array([0, 1]), 2)


def test_one_analysis_index_of_a_sequence_stands_for_every_cell() -> None:
    """A scalar broadcasts by design: the whole run belongs to that Analysis."""
    at = Placement.sequence(0, np.arange(3), 1, 2)
    kept = storable(
        CellValues(_f32(1.0, 1.0, 1.0), _f32(0.5, np.inf, 0.5)), at.analyses, 2, where="x"
    )
    assert kept.se_out_of_range.tolist() == [0, 1]


# ── AssociationCodec: one encode, in the order the planes allow ──────────────

_RESIDUAL = StoreEncoding(
    z=ZEncoding("int16_fixed", scale=1024),
    se=SeEncoding("int8_residual", residual_range=0.5),
    eaf=EafEncoding("int8_residual", residual_range=0.5),
)
_FLOAT16_SE = _RESIDUAL.with_se(SeEncoding("float16"))
_N_VARIANTS, _N_ANALYSES = 10_000, 4


def _random_cells(n: int, seed: int = 0, *, missing: float = 0.2) -> tuple[CellValues, Placement]:
    """`n` Dense cells at random places, about `missing` of them missing in both statistics."""
    rng = np.random.default_rng(seed)
    rows, cols = np.divmod(rng.choice(_N_VARIANTS * _N_ANALYSES, n, replace=False), _N_ANALYSES)
    eaf = rng.uniform(0.01, 0.99, n).astype(np.float32)
    se = np.exp(-3.0 - 0.5 * np.log(2 * eaf * (1 - eaf)) + rng.normal(0, 0.05, n))
    z = rng.normal(0, 3, n).astype(np.float32)
    gone = rng.random(n) < missing
    z[gone] = np.nan
    se[gone] = np.nan
    cells = CellValues(z, se.astype(np.float32), eaf)
    return cells, Placement.pairs(rows, cols, _N_ANALYSES)


def _codec(plan: StoreEncoding = _RESIDUAL, **kwargs: np.ndarray) -> AssociationCodec:
    rng = np.random.default_rng(1)
    defaults = {
        "eaf_baseline": rng.uniform(0.02, 0.98, _N_VARIANTS).astype(np.float32),
        "se_coefficients": np.tile(np.array([[-3.0, -0.5]], np.float32), (_N_ANALYSES, 1)),
    }
    return AssociationCodec(plan, **{**defaults, **kwargs})


def test_se_is_coded_against_the_frequency_a_reader_decodes_not_the_sources() -> None:
    """The expected codes are built from `StoreCodec`'s per-plane calls, the way the
    writers composed them before the codec existed: EAF encoded against each cell's
    row baseline and decoded with its exception table, then SE against that."""
    cells, at = _random_cells(20_000)
    codec = _codec()
    rows, analyses = np.asarray(at.rows), np.asarray(at.analyses)
    baseline = codec._baseline[rows]
    eaf_entries = EafExceptionBuilder()
    eaf_codes = StoreCodec(_RESIDUAL).encode_eaf(
        cells.eaf, baseline=baseline, positions=at.positions, exceptions=eaf_entries
    )
    decoded_eaf = StoreCodec(_RESIDUAL, eaf_exceptions=eaf_entries.table()).decode_eaf(
        eaf_codes, baseline=baseline, positions=at.positions
    )

    def se_codes(eaf: np.ndarray) -> np.ndarray:
        return StoreCodec(_RESIDUAL).encode_se(
            cells.se,
            eaf=eaf,
            analysis_index=analyses,
            coefficients=codec._model(),
            positions=at.positions,
            exceptions=SeExceptionBuilder(),
        )

    expected, against_source = se_codes(decoded_eaf), se_codes(cells.eaf)
    assert (expected != against_source).sum() > 0, "fixture must make the order matter"

    assert np.array_equal(codec.encode(cells, at, where="x").se, expected)


def test_an_imputed_cell_is_coded_against_the_panels_frequency() -> None:
    panel = np.full(_N_VARIANTS, 0.25, dtype=np.float32)
    codec = _codec(_RESIDUAL.with_eaf_reference(True), eaf_reference=panel)
    cells = CellValues(
        _f32(1.0, 2.0),
        _f32(0.08, 0.09),
        eaf=_f32(NAN, 0.4),
        imputed=np.array([1, 0], dtype=np.uint8),
    )
    at = Placement.pairs(np.array([3, 7]), np.array([0, 1]), _N_ANALYSES)

    back = codec.round_trip(cells, at, where="x")

    assert back.eaf[0] == np.float32(0.25)
    assert abs(back.eaf[1] - 0.4) < 1e-3
    np.testing.assert_allclose(back.se, cells.se, rtol=0.01)


def test_the_baseline_is_gathered_by_each_cells_row_and_tables_keyed_by_its_position() -> None:
    baseline = np.linspace(0.05, 0.95, _N_VARIANTS, dtype=np.float32)
    codec = _codec(eaf_baseline=baseline)
    rows, cols = np.array([40, 2, 17]), np.array([3, 0, 2])
    eaf = np.array([baseline[40], 0.999999, baseline[17]], dtype=np.float32)
    cells = CellValues(_f32(1.0, 200.0, -2.0), _f32(0.1, 0.2, 0.3), eaf=eaf)
    at = Placement.pairs(rows, cols, _N_ANALYSES)

    encoded = codec.encode(cells, at, where="x")
    back = codec.round_trip(cells, at, where="x")

    assert encoded.eaf is not None
    assert encoded.eaf.tolist() == [0, EAF_EXCEPTION, 0], "a cell at its own baseline codes as 0"
    assert encoded.tables.z_overflow.index.tolist() == [2 * _N_ANALYSES + 0]
    assert encoded.tables.eaf_exceptions.index.tolist() == [2 * _N_ANALYSES + 0]
    assert back.z[1] == np.float32(200.0)
    assert back.eaf[1] == np.float32(0.999999)


def test_a_band_encodes_as_its_cells_do() -> None:
    rng = np.random.default_rng(3)
    r0, r1 = 10, 14
    eaf = rng.uniform(0.05, 0.95, (r1 - r0, _N_ANALYSES)).astype(np.float32)
    se = rng.uniform(0.05, 0.2, eaf.shape).astype(np.float32)
    z = rng.normal(0, 40, eaf.shape).astype(np.float32)
    codec = _codec()
    band = codec.encode(CellValues(z, se, eaf), Placement.band(r0, r1, _N_ANALYSES), where="x")
    rows, cols = np.nonzero(np.ones(eaf.shape, bool))
    cells = CellValues(z[rows, cols], se[rows, cols], eaf[rows, cols])
    pairs = codec.encode(cells, Placement.pairs(rows + r0, cols, _N_ANALYSES), where="x")

    assert len(band.tables.z_overflow) > 0, "fixture must overflow z"
    for plane in ("z", "se", "eaf"):
        assert np.array_equal(getattr(band, plane).ravel(), getattr(pairs, plane))
    assert band.tables.z_overflow.index.tolist() == pairs.tables.z_overflow.index.tolist()


def test_z_and_se_are_missing_together_after_a_round_trip() -> None:
    for seed in range(5):
        cells, at = _random_cells(5_000, seed)
        back = _codec().round_trip(cells, at, where="x")
        assert np.array_equal(np.isnan(back.z), np.isnan(cells.z))
        assert np.array_equal(np.isnan(back.se), np.isnan(back.z))


def test_an_se_above_float16s_range_reads_back_as_infinity_until_issue_338() -> None:
    cells = CellValues(_f32(1.0, 2.0), _f32(70_000.0, 0.1), eaf=_f32(0.3, 0.4))
    at = Placement.pairs(np.array([0, 1]), np.array([0, 2]), _N_ANALYSES)

    encoded = _codec(_FLOAT16_SE).encode(cells, at, where="x")
    back = _codec(_FLOAT16_SE).round_trip(cells, at, where="x")

    assert back.se[0] == np.inf and back.z[0] == np.float32(1.0)
    assert encoded.storable.se_out_of_range.tolist() == [1, 0, 0, 0]


def test_a_frequency_the_plan_has_no_plane_for_is_refused() -> None:
    plan = StoreEncoding(z=_RESIDUAL.z, se=SeEncoding("float16"))
    cells = CellValues(_f32(1.0), _f32(0.1), eaf=_f32(0.3))
    at = Placement.pairs(np.array([0]), np.array([0]), _N_ANALYSES)
    with pytest.raises(ValueError, match="x: 1 cell"):
        AssociationCodec(plan).encode(cells, at, where="x")
    assert (
        AssociationCodec(plan).encode(CellValues(_f32(1.0), _f32(0.1)), at, where="x").eaf is None
    )


def test_a_cell_admission_forbids_never_reaches_a_plane() -> None:
    cells = CellValues(_f32(1.0), _f32(NAN), eaf=_f32(0.3))
    at = Placement.pairs(np.array([0]), np.array([0]), _N_ANALYSES)
    with pytest.raises(NotAdmitted, match="one of z and SE missing"):
        _codec().encode(cells, at, where="x")


# ── Writing an encoded batch ─────────────────────────────────────────────────


def _written_band(tmp_path, n_rows: int = 8) -> tuple[Any, AssociationCodec, CellValues]:
    """A residual Dense grid written whole, with entries in every side table."""
    rng = np.random.default_rng(7)
    codec = _codec()
    near = codec._baseline[:n_rows, None] * rng.uniform(0.99, 1.01, (n_rows, _N_ANALYSES))
    eaf = near.astype(np.float32)
    se = np.exp(-3.0 - 0.5 * np.log(2 * eaf * (1 - eaf))).astype(np.float32)
    z = rng.normal(0, 2, eaf.shape).astype(np.float32)
    for row in (0, 5):  # exceptional in every plane
        z[row, 1], eaf[row, 1], se[row, 1] = 300.0, 0.999999, 4000.0
    cells = CellValues(z, se, eaf)
    encoded = codec.encode(cells, Placement.band(0, n_rows, _N_ANALYSES), where="x")
    group = zarr.open_group(str(tmp_path / "grid.zarr"), mode="w", zarr_format=3)
    write_encoded_planes(
        group,
        _RESIDUAL,
        encoded,
        role=ArrayRole.DENSE_STATISTIC_PLANE,
        compressor=compressor(),
        eaf_baseline=codec._baseline,
        se_coefficients=codec._model(),
    )
    for table in ("z_overflow_index", "eaf_exception_index", "se_exception_index"):
        assert group[table][:].tolist() == [1, 5 * _N_ANALYSES + 1], table
    return group, codec, cells


def _read_band(group: Any, n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eaf = DenseEafPlane.open(group, _RESIDUAL).band(0, n_rows)
    se = DenseSePlane.open(group, _RESIDUAL).band(0, n_rows)
    return DenseZPlane.open(group, _RESIDUAL).band(0, n_rows), se, eaf


def test_a_written_grid_reads_back_as_the_codec_round_trips_it(tmp_path) -> None:
    group, codec, cells = _written_band(tmp_path)
    expected = codec.round_trip(cells, Placement.band(0, 8, _N_ANALYSES), where="x")
    for found, want in zip(_read_band(group, 8), expected, strict=True):
        np.testing.assert_array_equal(found, want)


def test_writing_cells_moves_each_cells_table_entries_and_keeps_the_rest(tmp_path) -> None:
    """Row 0's exceptional cell becomes ordinary and row 2's ordinary one exceptional:
    each table loses one entry and gains one, and row 5's entries stay."""
    group, codec, _ = _written_band(tmp_path)
    rows, cols = np.array([0, 2]), np.array([1, 3])
    batch = CellValues(_f32(1.5, 400.0), _f32(0.05, 4500.0), _f32(0.4, 0.999999))
    at = Placement.pairs(rows, cols, _N_ANALYSES)

    write_dense_cells(group, _RESIDUAL, rows, cols, codec.encode(batch, at, where="x"))

    for table in ("z_overflow_index", "eaf_exception_index", "se_exception_index"):
        assert group[table][:].tolist() == [2 * _N_ANALYSES + 3, 5 * _N_ANALYSES + 1], table
    expected = codec.round_trip(batch, at, where="x")
    z, se, eaf = _read_band(group, 8)
    np.testing.assert_array_equal(z[rows, cols], expected.z)
    np.testing.assert_array_equal(se[rows, cols], expected.se)
    np.testing.assert_array_equal(eaf[rows, cols], expected.eaf)
    assert z[5, 1] == np.float32(300.0) and se[5, 1] == np.float32(4000.0)
