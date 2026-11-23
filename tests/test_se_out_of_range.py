"""An SE above float16's range is not stored, and no encoder writes one (issue #338, ADR 0068).

The rule's seams below the builders: `storable`, which every writer holds what
it returns of (ADR 0077); the encoders, which refuse such an SE rather than
narrow it to `+inf`; the `analyses.tsv` column that counts it; and the
`format_version` that column arrives in. The builders are tested in
`test_se_out_of_range_builds.py`, completion and the Observed-View Check in
`test_se_out_of_range_completion.py`.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from opengwasdb.encoding.association_codec import (
    SE_FLOAT16_MAX,
    AssociationCodec,
    CellValues,
    Placement,
    storable,
)
from opengwasdb.encoding.codec import SeOutOfRange, StoreCodec
from opengwasdb.encoding.sd_units import PhenotypeSd
from opengwasdb.model import analyses as model
from opengwasdb.model.analyses import (
    N_MISSING_SE_OUT_OF_RANGE,
    Analysis,
    read_analysis_records,
    to_json_schema,
    with_se_out_of_range,
    write_analysis_records,
)
from opengwasdb.store import open as store_open
from opengwasdb.store_reader import format as store_format
from opengwasdb.store_reader.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding

#: The largest SE a float16 plane holds, and the smallest float32 above it.
EDGE = 65504.0
ABOVE = float(np.nextafter(np.float32(EDGE), np.float32(np.inf)))


# ── storable ──────────────────────────────────────────────────────────────────


def test_storable_keeps_the_edge_and_leaves_out_anything_above_it() -> None:
    """Column `j` is Analysis `j`: 0 holds the edge, 1 a value just above it, 2 `+inf`."""
    assert SE_FLOAT16_MAX == EDGE == float(np.finfo(np.float16).max)
    assert ABOVE == 65504.00390625, "fixture: the smallest float32 above the bound"
    z = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]], dtype=np.float32)
    se = np.array([[EDGE, ABOVE, np.inf], [0.1, np.nan, 0.3]], dtype=np.float32)
    eaf = np.array([[0.2, 0.3, 0.4], [0.5, np.nan, 0.6]], dtype=np.float32)
    imputed = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    assert int(np.count_nonzero(se > EDGE)) == 2, "fixture: two SEs above the bound"

    kept = storable(
        CellValues(z, se, eaf, imputed), np.arange(3)[None, :], 3, where="test",
        phenotype_sd=PhenotypeSd.APPLIED,
    )

    # A missing cell is held as missing; only the two above the bound are left out.
    assert kept.stored.tolist() == [[True, False, False], [True, True, True]]
    assert kept.se_out_of_range.tolist() == [0, 1, 1]
    assert kept.imputed_out_of_range.tolist() == [0, 1, 0]
    cells = kept.cells
    for plane in (cells.z, cells.se, cells.eaf):
        assert np.isnan(np.asarray(plane)[0, 1]) and np.isnan(np.asarray(plane)[0, 2])
    assert np.asarray(cells.imputed)[0].tolist() == [0, 0, 0]
    assert float(np.asarray(cells.se)[0, 0]) == EDGE
    assert float(np.asarray(cells.z)[1, 2]) == 6.0 and int(np.asarray(cells.imputed)[1, 2]) == 1
    # The batch handed over is not edited in place.
    assert float(se[0, 1]) == ABOVE and float(z[0, 2]) == 3.0


def test_storable_is_idempotent_and_counts_nothing_the_second_time() -> None:
    z = np.array([1.0, 2.0], dtype=np.float32)
    se = np.array([0.1, 1e6], dtype=np.float32)
    once = storable(CellValues(z, se), 0, 1, where="test", phenotype_sd=PhenotypeSd.APPLIED)
    twice = storable(once.cells, 0, 1, where="test", phenotype_sd=PhenotypeSd.APPLIED)
    assert once.se_out_of_range.tolist() == [1]
    assert twice.se_out_of_range.tolist() == [0]
    assert twice.stored.tolist() == [True, True]
    assert np.isnan(np.asarray(twice.cells.se)[1])


@pytest.mark.parametrize("kind", ["float16", "int8_residual"])
def test_the_codec_stores_such_an_association_as_missing_under_every_plan(kind: str) -> None:
    """Whether an association is kept does not depend on the plan (ADR 0068 decision 1)."""
    codec = AssociationCodec(
        _plan(kind),
        eaf_baseline=np.full(2, 0.3, np.float32),
        se_coefficients=np.array([[np.log(0.1), -0.5]], np.float32),
    )
    cells = CellValues(
        np.array([1.0, 2.0], np.float32),
        np.array([0.1, 70000.0], np.float32),
        np.array([0.3, 0.3], np.float32),
    )
    decoded = codec.round_trip(cells, Placement.sequence(0, np.arange(2), 0, 1), where="test")
    assert np.isnan(decoded.z[1]) and np.isnan(decoded.se[1]) and np.isnan(decoded.eaf[1])
    assert decoded.z[0] == pytest.approx(1.0, rel=1e-2)
    assert decoded.se[0] == pytest.approx(0.1, rel=1e-2)


# ── the encoders ─────────────────────────────────────────────────────────────


def _plan(kind: str) -> StoreEncoding:
    return StoreEncoding(
        z=ZEncoding("float16"),
        se=SeEncoding(kind, 0.5) if kind == "int8_residual" else SeEncoding(kind),
        eaf=EafEncoding("float32"),
    )


def _encode_se(kind: str, values: list[float]) -> np.ndarray:
    from opengwasdb.encoding.codec import SeExceptionBuilder

    n = len(values)
    return StoreCodec(_plan(kind)).encode_se(
        np.array(values, dtype=np.float32),
        eaf=np.full(n, 0.3, dtype=np.float32),
        analysis_index=np.zeros(n, dtype=np.int64),
        coefficients=np.array([[np.log(0.1), -0.5]], dtype=np.float32),
        positions=np.arange(n),
        exceptions=SeExceptionBuilder(),
    )


@pytest.mark.parametrize("kind", ["float16", "int8_residual"])
@pytest.mark.parametrize("bad", [ABOVE, 1e9, np.inf])
def test_encode_se_refuses_an_se_above_the_bound(kind: str, bad: float) -> None:
    with pytest.raises(SeOutOfRange, match=r"1 standard error.*above 65,504"):
        _encode_se(kind, [0.1, bad])


@pytest.mark.parametrize("kind", ["float16", "int8_residual"])
def test_encode_se_writes_the_edge(kind: str) -> None:
    codes = _encode_se(kind, [0.1, EDGE])
    if kind == "float16":
        assert codes.dtype == np.float16 and float(codes[1]) == EDGE


#: SEs at and just below the bound, the edge last.
NEAR_EDGE = [
    65000.0, 65400.0, 65490.0, 65500.0, 65503.0, float(np.nextafter(np.float32(EDGE), 0)), EDGE,
]


def test_a_residual_code_never_decodes_above_the_bound() -> None:
    """A model predicting 65,520 puts the edge's ordinary code on 65,520.004: an
    admitted SE stored as one no Store may hold (spec §15). The encoder holds each
    such cell as an exact exception, so every decoded SE is at most 65,504."""
    step = SeEncoding("int8_residual", 0.5).step
    coefficients = np.array([[np.log(65520.0), 0.0]], np.float32)
    n = len(NEAR_EDGE)
    codes = np.rint((np.log(np.array(NEAR_EDGE)) - float(coefficients[0, 0])) / step)
    ordinary = np.exp(float(coefficients[0, 0]) + codes * step).astype(np.float32)
    assert (ordinary > EDGE).any(), "fixture: some ordinary code decodes above the bound"
    codec = AssociationCodec(
        _plan("int8_residual"),
        eaf_baseline=np.full(n, 0.3, np.float32),
        se_coefficients=coefficients,
    )
    cells = CellValues(
        np.ones(n, np.float32), np.array(NEAR_EDGE, np.float32), np.full(n, 0.3, np.float32)
    )
    decoded = codec.round_trip(cells, Placement.sequence(0, np.arange(n), 0, 1), where="test")
    assert float(np.max(decoded.se)) <= EDGE, decoded.se.tolist()
    assert float(decoded.se[-1]) == EDGE, "the edge is held exactly"
    assert decoded.se.tolist() == pytest.approx(NEAR_EDGE, rel=0.5 * step + 1e-6)


def _decoded_extremes(seed: int) -> tuple[float, int]:
    """Encode and decode random admitted SEs under random models and ranges: the
    largest decoded SE, and how many cells' prediction moves by a whole unit when
    the frequency is narrowed to float32, as a reader narrows it.

    Half the Analyses are steep, built to defeat the narrowing (issue #338): a float32 slope
    of 1e8 to 1e9 and an intercept, and frequencies within 1e-13 of the one at
    which the float64 prediction is the Analysis' typical SE, so their codes are
    ordinary while narrowing moves the prediction by whole units."""
    from opengwasdb.encoding.codec import SeExceptionBuilder
    from opengwasdb.store_reader.encoding.plan import SE_RANGE_CANDIDATES

    rng = np.random.default_rng(seed)
    n, n_analyses = 2000, 4
    steep = rng.random(n_analyses) < 0.5
    target = np.log(rng.uniform(1.0, EDGE, n_analyses))
    x_wanted = np.log(2 * (f := rng.uniform(0.05, 0.45, n_analyses)) * (1 - f))
    slopes = np.where(
        steep, rng.choice([-1, 1], n_analyses) * 10.0 ** rng.uniform(8, 9, n_analyses),
        rng.uniform(-1.0, 1.0, n_analyses),
    ).astype(np.float32)
    intercepts = (target - slopes.astype(np.float64) * x_wanted).astype(np.float32)
    x0 = (target - intercepts.astype(np.float64)) / slopes.astype(np.float64)
    centre = (1 - np.sqrt(1 - 2 * np.exp(x0))) / 2  # the frequency whose x is x0
    coefficients = np.column_stack([intercepts, slopes])
    ai = rng.integers(0, n_analyses, n)
    spread = np.where(steep, 1e-13, 0.04)
    eaf = centre[ai] + rng.uniform(-1.0, 1.0, n) * spread[ai]
    x = np.log(2 * eaf * (1 - eaf))
    residual_range = float(rng.choice(SE_RANGE_CANDIDATES))
    plan = _plan("int8_residual")
    plan = StoreEncoding(z=plan.z, se=SeEncoding("int8_residual", residual_range), eaf=plan.eaf)
    prediction = coefficients[ai, 0].astype(np.float64) + coefficients[ai, 1] * x
    with np.errstate(over="ignore"):
        se = np.exp(prediction + rng.uniform(-residual_range, residual_range, n) / 2)
    se = np.where(rng.random(n) < 0.1, EDGE, np.clip(se, 1e-3, EDGE)).astype(np.float32)
    codec = StoreCodec(plan)
    exceptions = SeExceptionBuilder()
    codes = codec.encode_se(
        se, eaf=eaf, analysis_index=ai, coefficients=coefficients,
        positions=np.arange(n), exceptions=exceptions,
    )
    decoded = StoreCodec(plan, se_exceptions=exceptions.table()).decode_se(
        codes, eaf=eaf.astype(np.float32), analysis_index=ai, coefficients=coefficients,
        positions=np.arange(n),
    )
    f32 = eaf.astype(np.float32).astype(np.float64)
    narrowed = coefficients[ai, 0].astype(np.float64) + coefficients[ai, 1] * np.log(
        2 * f32 * (1 - f32)
    )
    return float(np.nanmax(decoded)), int(np.count_nonzero(np.abs(narrowed - prediction) > 1.0))


def test_no_stored_cell_decodes_above_the_bound_under_any_model() -> None:
    """A property over random coefficients (slopes up to 1e9, where narrowing the
    frequency to float32 moves a prediction by whole units, as when an
    SE of 10,000 decoded to 11,806,119) and every candidate range."""
    moved = 0
    for seed in range(40):
        largest, shifted = _decoded_extremes(seed)
        assert largest <= EDGE, (seed, largest)
        moved += shifted
    assert moved > 1000, "fixture: some predictions move far when the EAF narrows"


def test_a_cell_far_below_the_bound_is_held_exactly_when_the_eaf_narrows() -> None:
    """SE 10,000 under a model whose float32-EAF prediction differs by whole units."""
    from opengwasdb.encoding.codec import SeExceptionBuilder

    plan = _plan("int8_residual")
    eaf = np.array([0.30000099168380356])
    coefficients = np.array([[867498688.0, 1000000000.0]], np.float32)
    exceptions = SeExceptionBuilder()
    codes = StoreCodec(plan).encode_se(
        np.array([10000.0], np.float32), eaf=eaf, analysis_index=np.zeros(1, np.int64),
        coefficients=coefficients, positions=np.arange(1), exceptions=exceptions,
    )
    decoded = StoreCodec(plan, se_exceptions=exceptions.table()).decode_se(
        codes, eaf=eaf.astype(np.float32), analysis_index=np.zeros(1, np.int64),
        coefficients=coefficients, positions=np.arange(1),
    )
    assert float(decoded[0]) == 10000.0


def test_the_streamed_dense_rewrite_counts_a_crossing_as_its_encode_does(tmp_path) -> None:
    """The edge cell's ordinary code decodes to 65,520 under this model; the
    codes-only count and the rewrite's encode must call it one exception alike,
    or the rewrite refuses its own table (issue #338, ADR 0068)."""
    import zarr

    from opengwasdb.encoding import se as se_module
    from opengwasdb.encoding.timing import PhaseTimer
    from opengwasdb.store_reader.encoding.planes import DenseEafPlane, DenseSePlane

    group = zarr.open_group(str(tmp_path / "data.zarr"), mode="w", zarr_format=2)
    n = 60
    se = np.full((n, 1), 65000.0, np.float32)  # code -2 under this model: ordinary
    se[7, 0] = EDGE
    group.create_array("eaf", data=np.full((n, 1), 0.3, np.float32), chunks=(20, 1))
    group.create_array("se", data=se, chunks=(20, 1))
    plan = _plan("int8_residual")
    coefficients = np.array([[np.log(65520.0), 0.0]], np.float32)

    se_module._rewrite_selected(
        group, group["se"], DenseEafPlane.open(group, plan), coefficients, plan,
        PhaseTimer(), 1,
    )

    decoded = DenseSePlane.open(group, plan).band(0, n)
    assert float(decoded.max()) <= EDGE and float(decoded[7, 0]) == EDGE
    assert group["se_exception_index"].shape[0] == 1


def test_the_float16_narrowing_of_a_dense_scratch_plane_refuses_such_an_se(tmp_path) -> None:
    """The one write of an SE plane that does not go through `encode_se`."""
    from opengwasdb.encoding.se import _narrow_dense_se_to_float16
    from opengwasdb.store.arrays import create_array, open_group_for_write
    from opengwasdb.store_reader.layout import ArrayRole

    def scratch(name: str, last: float):
        group = open_group_for_write(tmp_path / name, "w")
        plane = create_array(
            group, "se", ArrayRole.DENSE_STATISTIC_PLANE, shape=(2, 2), dtype="float32",
            fill_value=np.nan, hint=(2, 2),
        )
        plane[:] = np.array([[0.1, EDGE], [0.2, last]], dtype=np.float32)
        return group

    with pytest.raises(SeOutOfRange):
        _narrow_dense_se_to_float16(scratch("over.zarr", ABOVE))
    group = scratch("edge.zarr", 0.3)
    _narrow_dense_se_to_float16(group)
    assert group["se"].dtype == np.float16 and float(group["se"][0, 1]) == EDGE


# ── the analyses.tsv column ──────────────────────────────────────────────────


def test_a_count_round_trips_and_an_absent_column_reads_as_unknown(tmp_path) -> None:
    """A store-only column (a build manifest never carries it), written by every
    writer, and absent from a release before 0.2.3, where it reads as unknown."""
    assert N_MISSING_SE_OUT_OF_RANGE == "n_missing_se_out_of_range"
    assert N_MISSING_SE_OUT_OF_RANGE in model.ANALYSIS_COLUMNS
    assert model.classify_column(N_MISSING_SE_OUT_OF_RANGE) is model.ColumnClass.STORE_ONLY
    path = tmp_path / "analyses.tsv"
    analyses = with_se_out_of_range([Analysis("a"), Analysis("b")], np.array([0, 3]))
    write_analysis_records(path, analyses)
    assert [a.n_missing_se_out_of_range for a in read_analysis_records(path)] == ["0", "3"]

    header, *rows = path.read_text(encoding="utf-8").splitlines()
    at = header.split("\t").index(N_MISSING_SE_OUT_OF_RANGE)

    def without_column(line: str) -> str:
        cells = line.split("\t")
        return "\t".join(cells[:at] + cells[at + 1 :])

    path.write_text("\n".join(without_column(line) for line in [header, *rows]) + "\n")
    assert [a.n_missing_se_out_of_range for a in read_analysis_records(path)] == ["", ""]


def test_a_carried_count_is_added_and_an_unknown_one_is_refused() -> None:
    carried = with_se_out_of_range([Analysis("a"), Analysis("b")], np.array([2, 0]))
    assert [
        a.n_missing_se_out_of_range
        for a in with_se_out_of_range(carried, np.array([1, 0]), carried_forward=True)
    ] == ["3", "0"]
    with pytest.raises(ValueError, match="unknown"):
        with_se_out_of_range([Analysis("a")], np.array([1]), carried_forward=True)
    with pytest.raises(ValueError, match="2 counts for 1 Analyses"):
        with_se_out_of_range([Analysis("a")], np.array([1, 2]))


def _schema_accepts(value: str) -> bool:
    """Whether the column's JSON Schema accepts `value`: a string matching its
    `pattern`, which JSON Schema searches for, so the pattern anchors itself."""
    column = to_json_schema()["properties"][N_MISSING_SE_OUT_OF_RANGE]
    assert column["type"] == "string"
    pattern = column["pattern"]
    assert pattern.startswith("^") and pattern.endswith("$")
    return re.search(pattern, value) is not None


def test_the_schema_types_the_count_as_a_non_negative_integer_or_unknown() -> None:
    """One table, so the test fails whichever way the pattern is wrong: too loose (a
    string of any kind, as before #338) or too strict (no blank, no 0, no 17)."""
    cases = {
        "0": True, "17": True, "": True,
        "-1": False, "1.5": False, "x": False, "01": False, " 1": False,
    }
    assert {value: _schema_accepts(value) for value in cases} == cases


# ── format_version ───────────────────────────────────────────────────────────


def test_a_build_and_a_completion_write_0_2_3() -> None:
    assert store_format.WRITTEN_FORMAT_VERSIONS == ("0.2.0", "0.2.1", "0.2.2", "0.2.3")
    assert store_format.CURRENT_FORMAT_VERSION == "0.2.3"
    assert dict(store_format.SUPPORTED_FORMAT_VERSIONS)[(0, 2)] == (3,)
    assert store_open.completed_format_version("0.2.2", carries_regions=False) == "0.2.3"
    assert store_open.completed_format_version("0.2.3", carries_regions=True) == "0.2.3"


def test_0_2_3_is_the_newest_release_read_without_a_warning(caplog) -> None:
    """A 0.2.2 and a 0.2.3 release open quietly and complete; a 0.2.4 one is read with
    the newer-remainder warning, and its stamp is not judged: upgrade, not re-stamp."""
    with caplog.at_level("WARNING"):
        store_format.check_format_version("0.2.2")
        store_format.check_format_version("0.2.3")
    assert not caplog.text
    assert store_open.check_writable_format_version("0.2.2") == "0.2.2"
    assert store_open.check_format_version_stamp("0.2.2", carries_regions=False) == "0.2.2"
    assert store_open.check_format_version_stamp("0.2.3", carries_regions=True) == "0.2.3"
    with caplog.at_level("WARNING"):
        store_format.check_format_version("0.2.4")
    assert "0.2.4" in caplog.text and "not visible here" in caplog.text
    with pytest.raises(store_format.UnsupportedFormatVersion, match="Upgrade opengwasdb"):
        store_open.check_format_version_stamp("0.2.4", carries_regions=False)


def test_the_count_is_required_from_0_2_3() -> None:
    assert not store_open.records_se_out_of_range("0.2.2")
    assert store_open.records_se_out_of_range("0.2.3")
