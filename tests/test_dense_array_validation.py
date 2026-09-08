"""Seam tests for the split Dense array-validation pass (issue 130).

The row-band validation of a Dense store was one 59-cyclomatic, 185-line
function (``_validate_dense_arrays``). It is now a set of guard, latch and
report seams around a ``_DenseBandState``, so each independent flag and each
side-table position list has one function that owns it. These tests pin each
seam to the malformed-input diagnostic it exists to produce, so a future merge
that folds two seams back together -- or drops one -- fails here rather than
passing review as "simpler". The store-level companions live beside the store
fixtures they corrupt (``TestValidation`` in test_dense_completion.py and the
shape/side-length tests in test_validation.py / test_eaf_encoding_layouts.py).
"""

from __future__ import annotations

import numpy as np
import zarr

from opengwasdb.encoding import EAF_BASELINE, EAF_REFERENCE
from opengwasdb.validation.validate import (
    _band_marked_positions,
    _dense_eaf_side_lengths,
    _dense_plane_shape_errors,
    _dense_required_planes,
    _DenseBandState,
    _flatten_band_positions,
    _latch_dense_eaf_range,
    _latch_dense_imputed_content,
    _latch_dense_imputed_missingness,
    _latch_dense_missingness,
    _latch_dense_off_panel_imputed,
    _latch_negative_se,
    _report_dense_band_flags,
    _report_dense_eaf_range,
)


def _band_state(n_analyses: int = 2) -> _DenseBandState:
    return _DenseBandState(n_analyses=n_analyses)


# ── Guard seams: presence, shapes, per-variant side lengths ───────────────


def test_required_planes_names_every_missing_plane():
    root = zarr.group()
    root.create_dataset("z", data=np.zeros((3, 2), dtype="float16"))
    errors: list[str] = []

    planes = _dense_required_planes(root, errors)

    assert planes is None
    assert errors == ["missing data.zarr/se"]


def test_required_planes_reports_z_before_se_and_stops_the_pass():
    root = zarr.group()
    errors: list[str] = []

    planes = _dense_required_planes(root, errors)

    assert planes is None
    assert errors == ["missing data.zarr/z", "missing data.zarr/se"]


def test_required_planes_hands_back_present_planes():
    root = zarr.group()
    z = root.create_dataset("z", data=np.zeros((3, 2), dtype="float16"))
    se = root.create_dataset("se", data=np.zeros((3, 2), dtype="float16"))

    planes = _dense_required_planes(root, [])

    assert planes is not None
    z_arr, se_arr = planes
    assert z_arr.shape == z.shape
    assert se_arr.shape == se.shape


def test_plane_shape_errors_record_every_misshaped_plane_in_order():
    root = zarr.group()
    z = root.create_dataset("z", data=np.zeros((3, 2), dtype="int16"))
    se = root.create_dataset("se", data=np.zeros((2, 2), dtype="float16"))
    root.create_dataset("eaf", data=np.zeros((2, 2), dtype="float32"))
    errors: list[str] = []

    eaf_arr = _dense_plane_shape_errors(root, z, se, 3, 2, errors)

    assert eaf_arr is None  # the mis-shaped eaf plane must not reach a decode seam
    assert errors == [
        "se shape (2, 2) does not match (3, 2)",
        "eaf shape (2, 2) does not match (3, 2)",
    ]


def test_plane_shape_errors_keep_an_aligned_eaf_plane():
    root = zarr.group()
    z = root.create_dataset("z", data=np.zeros((3, 2), dtype="int16"))
    se = root.create_dataset("se", data=np.zeros((3, 2), dtype="float16"))
    eaf = root.create_dataset("eaf", data=np.zeros((3, 2), dtype="float32"))

    eaf_arr = _dense_plane_shape_errors(root, z, se, 3, 2, [])

    assert eaf_arr is not None
    assert eaf_arr.shape == eaf.shape


def test_eaf_side_lengths_reject_short_per_variant_side_arrays():
    root = zarr.group()
    root.create_dataset(EAF_BASELINE, data=np.zeros(4, dtype="float32"))
    root.create_dataset(EAF_REFERENCE, data=np.zeros(5, dtype="float32"))
    errors: list[str] = []

    _dense_eaf_side_lengths(root, 6, errors)

    assert errors == [
        f"{EAF_BASELINE} has 4 entries but the variant axis has 6",
        f"{EAF_REFERENCE} has 5 entries but the variant axis has 6",
    ]


# ── Latch seams: one seam owns one independent flag ───────────────────────


def test_negative_se_latch_fires_on_one_bad_band():
    findings = _band_state()
    _latch_negative_se(findings, np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    assert findings.neg_se is False

    _latch_negative_se(
        findings, np.array([[0.1, -0.2], [np.nan, 0.4]], dtype=np.float32)
    )

    assert findings.neg_se is True


def test_missingness_latch_needs_z_and_se_to_agree_cell_by_cell():
    se = np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float32)
    z_missing = np.array([[True, False], [False, True]])
    findings = _band_state()

    _latch_dense_missingness(findings, z_missing, se)

    assert findings.missingness is True

    consistent = _band_state()
    se_ok = np.array([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float32)
    _latch_dense_missingness(consistent, z_missing, se_ok)
    assert consistent.missingness is False


def test_imputed_content_latches_non_binary_and_counts_imputed_cells():
    findings = _band_state()
    band = np.array([[0, 1], [2, 1]], dtype=np.uint8)

    imp_mask = _latch_dense_imputed_content(findings, band, 0, 2)

    assert findings.imp_not_binary is True
    assert findings.imputed_per_analysis.tolist() == [0, 2]
    assert imp_mask is not None
    assert imp_mask.tolist() == [[False, True], [False, True]]


def test_imputed_counts_accumulate_across_bands():
    findings = _band_state()
    _latch_dense_imputed_content(findings, np.array([[1, 0]], dtype=np.uint8), 0, 1)
    _latch_dense_imputed_content(findings, np.array([[0, 1]], dtype=np.uint8), 0, 1)

    assert findings.imputed_per_analysis.tolist() == [1, 1]
    assert findings.imp_not_binary is False


def test_imputed_missingness_latches_on_missing_z_or_nan_se():
    findings = _band_state()
    imp_mask = np.array([[True, True]])
    z_missing = np.array([[True, False]])
    se = np.array([[1.0, np.nan]], dtype=np.float32)

    _latch_dense_imputed_missingness(findings, z_missing, se, imp_mask)

    assert findings.imp_nan_z is True
    assert findings.imp_nan_se is True

    clean = _band_state()
    se_ok = np.array([[1.0, 2.0]], dtype=np.float32)
    z_present = np.array([[False, False]])
    _latch_dense_imputed_missingness(clean, z_present, se_ok, imp_mask)
    assert clean.imp_nan_z is False
    assert clean.imp_nan_se is False


def test_off_panel_latch_requires_an_off_panel_row_with_an_imputed_cell():
    findings = _band_state()
    on_panel = np.array([1, 0], dtype=np.uint8)
    imp_mask = np.array([[False, False], [True, True]])

    _latch_dense_off_panel_imputed(findings, on_panel, imp_mask, 0, 2)

    assert findings.off_panel_imputed is True

    fine = _band_state()
    imp_on_panel_only = np.array([[True, True], [False, False]])
    _latch_dense_off_panel_imputed(fine, on_panel, imp_on_panel_only, 0, 2)
    assert fine.off_panel_imputed is False


class _FakeEafPlane:
    """A plane stub: decodes a band, or raises, and counts decode attempts."""

    def __init__(self, values: np.ndarray | None = None, raises: bool = False) -> None:
        self.values = values
        self.raises = raises
        self.calls = 0

    def band(self, r0: int, r1: int) -> np.ndarray:
        self.calls += 1
        if self.raises:
            raise ValueError("no table entry for an exception cell")
        assert self.values is not None
        return self.values


def test_eaf_range_latch_marks_out_of_range_decoded_value():
    findings = _band_state()
    plane = _FakeEafPlane(values=np.array([[0.3, 1.4]], dtype=np.float32))

    _latch_dense_eaf_range(findings, plane, 0, 1)

    assert findings.eaf_out_of_range is True
    assert findings.eaf_undecodable is False


def test_eaf_range_latch_stops_decoding_once_a_flag_latches():
    findings = _band_state()
    plane = _FakeEafPlane(values=np.array([[0.3, 1.4]], dtype=np.float32))

    _latch_dense_eaf_range(findings, plane, 0, 1)
    _latch_dense_eaf_range(findings, plane, 1, 2)

    assert plane.calls == 1  # later bands skip the decode entirely


def test_eaf_range_latch_marks_undecodable_and_stops_further_decodes():
    findings = _band_state()
    plane = _FakeEafPlane(raises=True)

    _latch_dense_eaf_range(findings, plane, 0, 1)
    _latch_dense_eaf_range(findings, plane, 1, 2)

    assert findings.eaf_undecodable is True
    assert findings.eaf_out_of_range is False
    assert plane.calls == 1


# ── Report seams: fixed diagnostic order, one seam per concern ─────────────


def test_band_flag_reporter_emits_latched_flags_in_fixed_order():
    findings = _band_state()
    findings.neg_se = True
    findings.missingness = True
    findings.imp_not_binary = True
    findings.imp_nan_z = True
    findings.imp_nan_se = True
    findings.off_panel_imputed = True
    errors: list[str] = []

    _report_dense_band_flags(errors, findings)

    assert errors == [
        "se contains negative finite values",
        "z and se missingness is inconsistent",
        "data.zarr/imputed contains values other than 0 and 1",
        "imputed=1 cells have missing z-scores",
        "imputed=1 cells have NaN se values",
        "off-panel (on_panel=0) rows have imputed=1 cells — off-panel is never imputable",
    ]


def test_eaf_range_reporter_is_a_separate_seam_from_the_band_flags():
    findings = _band_state()
    findings.eaf_out_of_range = True
    errors: list[str] = []

    _report_dense_eaf_range(errors, findings)

    assert errors == ["data.zarr/eaf contains finite values outside [0, 1]"]


# ── Position seams: the side-table lists every report seam reconciles ──────


def test_band_marked_positions_are_flat_plane_offsets():
    raw = np.array([[1, -127, 1], [-127, 1, 1]], dtype=np.int8)

    positions = _band_marked_positions(raw, -127, r0=2, n_analyses=3)

    assert positions.tolist() == [2 * 3 + 1, 2 * 3 + 3]


def test_flatten_band_positions_concatenates_in_band_order_or_returns_empty():
    assert _flatten_band_positions([]).tolist() == []

    combined = _flatten_band_positions([np.array([2]), np.array([5, 6])])

    assert combined.tolist() == [2, 5, 6]
