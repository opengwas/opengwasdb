"""Residual-coded `eaf`: the plan, the baseline, the codec and its side table.

Issue #116 / ADR 0037 §2 and §4. The unit-level half of the change -- the
layout round-trips live in `test_eaf_encoding_layouts.py`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from opengwasdb.encoding import (
    EAF_ABSENT,
    EAF_EXCEPTION,
    EafBaselineError,
    EafEncoding,
    EafExceptionBuilder,
    EafExceptionTable,
    EafMeasurements,
    EncodingMeasurements,
    StoreCodec,
    StoreEncoding,
    UnsupportedEncoding,
    eaf_baseline_from_grid,
    eaf_baseline_from_pairs,
    logit,
)


def _measurements(**overrides: object) -> EncodingMeasurements:
    """A build with plenty of EAF-bearing cells per variant and tight residuals."""
    fields: dict[str, object] = {
        "n_cells": 80_000,
        "n_eaf_cells": 80_000,
        "n_variants": 10_000,
        "exception_fraction": {0.5: 0.0, 1.0: 0.0, 2.0: 0.0},
    }
    fields.update(overrides)
    eaf = EafMeasurements(**fields)  # type: ignore[arg-type]
    return EncodingMeasurements(n_analyses=8, eaf=eaf)


def _residual_plan(residual_range: float = 0.5) -> StoreEncoding:
    plan = StoreEncoding.decide(_measurements())
    assert plan.eaf.kind == "int8_residual"
    return StoreEncoding(
        z=plan.z,
        se=plan.se,
        eaf=EafEncoding(kind="int8_residual", residual_range=residual_range),
        version=plan.version,
    )


def _round_trip(
    values: np.ndarray, baseline: np.ndarray, plan: StoreEncoding | None = None
) -> tuple[np.ndarray, np.ndarray, EafExceptionTable]:
    """Encode at flat positions 0..n and decode straight back."""
    plan = plan or _residual_plan()
    exceptions = EafExceptionBuilder()
    positions = np.arange(values.size).reshape(values.shape)
    codes = StoreCodec(plan).encode_eaf(
        values, baseline=baseline, positions=positions, exceptions=exceptions
    )
    table = exceptions.table()
    decoded = StoreCodec(plan, eaf_exceptions=table).decode_eaf(
        codes, baseline=baseline, positions=positions
    )
    return codes, decoded, table


# ── The plan ────────────────────────────────────────────────────────────────


def test_decide_chooses_the_residual_coding_when_the_baseline_amortises():
    encoding = StoreEncoding.decide(_measurements())
    assert encoding.eaf.kind == "int8_residual"
    assert encoding.eaf.residual_range == 0.5


def test_decide_chooses_the_smallest_range_that_fits_the_exception_budget():
    encoding = StoreEncoding.decide(
        _measurements(exception_fraction={0.5: 0.0375, 1.0: 0.0171, 2.0: 0.0087})
    )
    # 3.75% of cells outside ±0.5 is beyond the budget; 1.71% is not. The
    # measured GWAS Catalog case (ADR 0037 §2).
    assert encoding.eaf.residual_range == 1.0


def test_decide_falls_back_to_the_widest_range_when_none_fits():
    encoding = StoreEncoding.decide(
        _measurements(exception_fraction={0.5: 0.4, 1.0: 0.3, 2.0: 0.2})
    )
    assert encoding.eaf.kind == "int8_residual"
    assert encoding.eaf.residual_range == 2.0


def test_decide_keeps_float32_when_the_baseline_costs_more_than_it_saves():
    """One EAF-bearing cell per variant: a `float32` baseline plus an `int8`
    cell is larger than the `float32` cell it replaces (issue #116 review)."""
    encoding = StoreEncoding.decide(
        _measurements(n_cells=10_000, n_eaf_cells=10_000, n_variants=10_000)
    )
    assert encoding.eaf.kind == "float32"


def test_decide_declares_eaf_absent_when_no_analysis_reports_one():
    assert StoreEncoding.decide(EncodingMeasurements(n_analyses=8)).eaf.kind == "absent"
    assert (
        StoreEncoding.decide(_measurements(n_eaf_cells=0)).eaf.kind == "absent"
    )


def test_the_eaf_plan_round_trips_through_the_manifest():
    plan = StoreEncoding.decide(_measurements()).with_eaf_reference(True)
    payload = json.loads(json.dumps(plan.to_manifest()))
    assert StoreEncoding.from_manifest(payload) == plan
    assert payload["eaf"] == {
        "kind": "int8_residual",
        "residual_range": 0.5,
        "reference": True,
    }


def test_a_format_version_1_release_declares_adr_0036s_optional_plane():
    """No `eaf` key means the release predates this encoding -- which is a real
    encoding with a name, not an absence each read site guesses at."""
    plan = StoreEncoding.from_manifest(
        {"version": 1, "z": {"kind": "int16_fixed", "scale": 1024}, "se": {"kind": "float16"}}
    )
    assert plan.eaf.kind == "float32_optional"
    assert plan.eaf.is_optional_plane


def test_reader_rejects_an_eaf_encoding_kind_it_does_not_implement():
    payload = StoreEncoding.decide(_measurements()).to_manifest()
    payload["eaf"]["kind"] = "int16_log_maf"
    with pytest.raises(UnsupportedEncoding, match="int16_log_maf"):
        StoreEncoding.from_manifest(payload)


def test_a_component_cannot_carry_reference_eaf_while_declaring_no_eaf_plane():
    plan = StoreEncoding.decide(EncodingMeasurements(n_analyses=8))
    with pytest.raises(EafBaselineError, match="no eaf plane"):
        plan.with_eaf_reference(True)


# ── The baseline ────────────────────────────────────────────────────────────


def test_the_baseline_is_the_median_in_logit_space():
    baseline = eaf_baseline_from_grid(np.array([[0.1, 0.2]]))
    # median of two logits is their mean, i.e. the geometric mean of the odds:
    # sqrt((1/9) * (1/4)) = 1/6, so f = 1/7.
    assert baseline[0] == pytest.approx(1.0 / 7.0, rel=1e-6)


def test_a_variant_with_no_usable_frequency_has_no_baseline():
    baseline = eaf_baseline_from_grid(
        np.array([[np.nan, np.nan], [0.0, 1.0], [0.0, 0.3]])
    )
    assert np.isnan(baseline[0])
    assert np.isnan(baseline[1])  # monomorphic either way: no logit, no baseline
    assert baseline[2] == pytest.approx(0.3, rel=1e-6)


def test_a_baseline_float32_cannot_represent_is_no_baseline_at_all():
    """`expit` of a large logit rounds to exactly 1.0 in `float32`, and 1.0 has
    no logit. Such a variant gets no baseline and its cells are held exactly,
    rather than being nudged to the nearest representable neighbour -- which
    would move a MAF of 1e-9 to one of 6e-8 without saying so."""
    baseline = eaf_baseline_from_grid(np.array([[1 - 1e-9, 1 - 2e-9], [1e-9, 2e-9]]))
    assert np.isnan(baseline[0])
    assert 0.0 < baseline[1] < 1.0  # the low tail is representable, and kept

    values = np.array([[1 - 1e-9, 1 - 2e-9]])
    codes, decoded, table = _round_trip(values, baseline[:1, None].repeat(2, axis=1))
    assert list(codes[0]) == [EAF_EXCEPTION, EAF_EXCEPTION]
    np.testing.assert_array_equal(decoded[0], values[0].astype(np.float32))


def test_the_csr_baseline_matches_the_grid_baseline_cell_for_cell():
    rng = np.random.default_rng(0)
    grid = rng.uniform(0.01, 0.99, size=(200, 5))
    grid[rng.random(grid.shape) < 0.3] = np.nan
    from_grid = eaf_baseline_from_grid(grid)
    rows, cols = np.nonzero(np.isfinite(grid))
    from_pairs = eaf_baseline_from_pairs(rows, grid[rows, cols], grid.shape[0])
    np.testing.assert_allclose(
        np.nan_to_num(from_grid, nan=-1.0), np.nan_to_num(from_pairs, nan=-1.0), rtol=1e-6
    )


# ── The codec ───────────────────────────────────────────────────────────────


def test_every_unclipped_cell_round_trips_within_half_a_step():
    """The acceptance criterion, stated per range as ADR 0037 §2 requires."""
    rng = np.random.default_rng(7)
    for residual_range in (0.5, 1.0, 2.0):
        plan = _residual_plan(residual_range)
        baseline_values = rng.uniform(0.001, 0.999, size=400)
        residuals = rng.uniform(-residual_range * 0.9, residual_range * 0.9, size=(400, 6))
        values = 1.0 / (1.0 + np.exp(-(logit(baseline_values)[:, None] + residuals)))
        baseline = eaf_baseline_from_grid(values)
        codes, decoded, table = _round_trip(values, baseline[:, None].repeat(6, axis=1), plan)
        kept = codes != EAF_EXCEPTION
        worst = np.maximum(
            np.abs(decoded - values) / values,
            np.abs(decoded - values) / (1.0 - values),
        )
        assert worst[kept].max() <= plan.eaf.worst_case_relative_error * 1.001
        assert plan.eaf.worst_case_relative_error == pytest.approx(residual_range / 254)


def test_an_analysis_with_no_frequency_reads_back_nan_not_a_zero_residual():
    values = np.array([[0.3, np.nan, 0.3]])
    baseline = eaf_baseline_from_grid(values)
    codes, decoded, _ = _round_trip(values, baseline[:, None].repeat(3, axis=1))
    assert codes[0, 1] == EAF_ABSENT
    assert codes[0, 0] == 0  # a residual of zero, which must not read alike
    assert np.isnan(decoded[0, 1])
    assert decoded[0, 0] == pytest.approx(0.3, rel=1e-5)


def test_a_clipped_cell_resolves_to_its_exact_value_not_to_the_baseline():
    values = np.array([[0.30, 0.31, 0.32, 0.9999]])
    baseline = eaf_baseline_from_grid(values)
    codes, decoded, table = _round_trip(values, baseline[:, None].repeat(4, axis=1))
    assert codes[0, 3] == EAF_EXCEPTION
    assert len(table) == 1
    assert decoded[0, 3] == np.float32(0.9999)
    assert decoded[0, 3] != pytest.approx(float(baseline[0]), rel=1e-3)


def test_frequencies_of_exactly_zero_and_one_are_stored_exactly():
    values = np.array([[0.0, 0.4, 1.0]])
    baseline = eaf_baseline_from_grid(values)
    codes, decoded, table = _round_trip(values, baseline[:, None].repeat(3, axis=1))
    assert list(codes[0]) == [EAF_EXCEPTION, 0, EAF_EXCEPTION]
    assert decoded[0, 0] == 0.0
    assert decoded[0, 2] == 1.0


def test_a_variant_with_no_baseline_keeps_its_cells_exactly():
    """Every Analysis monomorphic: no logit, so no baseline -- and the cells go
    to the table rather than being coded against an invented one."""
    values = np.array([[0.0, 1.0]])
    baseline = eaf_baseline_from_grid(values)
    assert np.isnan(baseline[0])
    codes, decoded, table = _round_trip(values, baseline[:, None].repeat(2, axis=1))
    assert list(codes[0]) == [EAF_EXCEPTION, EAF_EXCEPTION]
    np.testing.assert_array_equal(decoded[0], [0.0, 1.0])


def test_a_value_outside_zero_to_one_fails_the_build():
    with pytest.raises(ValueError, match="not a frequency"):
        _round_trip(np.array([[1.4]]), np.array([[0.5]], dtype=np.float32))


def test_decoding_a_residual_plane_without_its_baseline_fails_loudly():
    plan = _residual_plan()
    with pytest.raises(EafBaselineError, match="eaf_baseline"):
        StoreCodec(plan).decode_eaf(np.zeros(4, dtype=np.int8))


def test_decoding_a_plane_whose_dtype_contradicts_the_manifest_fails():
    plan = _residual_plan()
    with pytest.raises(ValueError, match="manifest disagree"):
        StoreCodec(plan).decode_eaf(
            np.zeros(4, dtype=np.float32), baseline=np.full(4, 0.3, dtype=np.float32)
        )


def test_an_exception_cell_with_no_table_entry_is_an_error_not_a_number():
    plan = _residual_plan()
    codes = np.array([EAF_EXCEPTION], dtype=np.int8)
    codec = StoreCodec(plan, eaf_exceptions=EafExceptionTable.empty())
    with pytest.raises(ValueError, match="eaf exception table has no entry"):
        codec.decode_eaf(
            codes, baseline=np.array([0.3], dtype=np.float32), positions=np.array([0])
        )


def test_decoding_and_re_encoding_against_the_same_baseline_changes_nothing():
    """The invariant Reference Completion rests on: it decodes a source plane
    and re-encodes it onto a new variant axis, carrying the baselines across
    rather than recomputing them. If that round trip moved a code, a completed
    release would be less accurate than the release it was completed from."""
    rng = np.random.default_rng(3)
    values = 1.0 / (
        1.0 + np.exp(-(rng.uniform(-9, 9, (500, 1)) + rng.normal(0, 0.15, (500, 6))))
    )
    values[rng.random(values.shape) < 0.1] = np.nan
    baseline = eaf_baseline_from_grid(values)
    per_cell = baseline[:, None].repeat(6, axis=1)
    codes, decoded, table = _round_trip(values, per_cell)
    assert len(table) > 0  # the fixture really does exercise the table

    again = EafExceptionBuilder()
    recoded = StoreCodec(_residual_plan()).encode_eaf(
        decoded,
        baseline=per_cell,
        positions=np.arange(values.size).reshape(values.shape),
        exceptions=again,
    )
    np.testing.assert_array_equal(codes, recoded)
    np.testing.assert_array_equal(table.index, again.table().index)
    np.testing.assert_array_equal(table.value, again.table().value)


# ── Reference EAF for imputed cells (ADR 0037 §4) ───────────────────────────


def test_imputed_cells_read_the_panel_frequency_and_observed_cells_do_not():
    plan = _residual_plan().with_eaf_reference(True)
    values = np.array([[0.30, np.nan, 0.31]])
    baseline = eaf_baseline_from_grid(values)
    per_cell = baseline[:, None].repeat(3, axis=1)
    exceptions = EafExceptionBuilder()
    positions = np.arange(3).reshape(1, 3)
    codes = StoreCodec(plan).encode_eaf(
        values, baseline=per_cell, positions=positions, exceptions=exceptions
    )
    reference = np.array([[0.55, 0.55, 0.55]], dtype=np.float32)
    imputed = np.array([[False, True, False]])
    decoded = StoreCodec(plan, eaf_exceptions=exceptions.table()).decode_eaf(
        codes, baseline=per_cell, positions=positions, imputed=imputed, reference=reference
    )
    assert decoded[0, 1] == np.float32(0.55)          # imputed: the panel's
    assert decoded[0, 0] == pytest.approx(0.30, rel=1e-4)  # observed: the cohort's
    assert decoded[0, 2] == pytest.approx(0.31, rel=1e-4)


def test_an_eaf_less_observed_cell_stays_nan_and_never_takes_the_panel_value():
    """The central negative result: FinnGen's frequencies differ from the EUR
    panel by up to 3000x, so a missing cohort frequency must not read as the
    panel's (ADR 0037 §4)."""
    plan = _residual_plan().with_eaf_reference(True)
    values = np.array([[np.nan, 0.31]])
    baseline = eaf_baseline_from_grid(values)
    per_cell = baseline[:, None].repeat(2, axis=1)
    codes = StoreCodec(plan).encode_eaf(
        values, baseline=per_cell, positions=np.arange(2).reshape(1, 2),
        exceptions=EafExceptionBuilder(),
    )
    decoded = StoreCodec(plan).decode_eaf(
        codes,
        baseline=per_cell,
        positions=np.arange(2).reshape(1, 2),
        imputed=np.array([[False, False]]),
        reference=np.array([[0.55, 0.55]], dtype=np.float32),
    )
    assert np.isnan(decoded[0, 0])


def test_decoding_a_reference_bearing_release_without_the_mask_fails_loudly():
    plan = _residual_plan().with_eaf_reference(True)
    with pytest.raises(EafBaselineError, match="imputed mask"):
        StoreCodec(plan).decode_eaf(
            np.zeros(2, dtype=np.int8), baseline=np.full(2, 0.3, dtype=np.float32)
        )
