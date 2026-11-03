"""What a component stores of a batch of associations, and how it encodes them (#350)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from opengwasdb.admission import NotAdmitted
from opengwasdb.encoding import association_codec
from opengwasdb.encoding.association_codec import (
    SE_FLOAT16_MAX,
    CellValues,
    log_se_out_of_range,
    storable,
)

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


def test_an_se_narrowed_to_infinity_is_a_stored_value_and_counted() -> None:
    kept = storable(CellValues(_f32(1.0, NAN), _f32(np.inf, NAN)), 0, 1, where="x")
    assert kept.stored.tolist() == [True, False]
    assert kept.se_out_of_range.tolist() == [1]


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
    assert twice.stored.tolist() == once.stored.tolist() == [True, False, True]
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
    assert [r.getMessage() for r in caplog.records] == [
        "build-x: 7 SE value(s) above 65,504, stored as +inf (spec §15): b: 5, c: 2"
    ]
