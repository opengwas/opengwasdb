"""Seam tests for the Dense value rules and the band scan they fold over (issues 130, #55).

The row-band validation of a Dense store was one 59-cyclomatic, 185-line function,
then a set of latch and report seams around a band state. Each check is now a
Validation Rule (ADR 0074): a fold measures every band and is judged on the total,
and the band scan reads each plane once for all of them. These tests hold each rule
to the malformed input it exists to report, measured on bands built from arrays
(`Block.of`) and judged as the runner would add them up, and the scan to its
memory budget. The store-level companions live beside the store fixtures they
corrupt (test_validation.py, test_validate_presence.py, test_dense_completion.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from opengwasdb import top_hits
from opengwasdb.encoding import (
    EAF_BASELINE,
    EAF_REFERENCE,
    Z_OVERFLOW,
    SeEncoding,
    StoreEncoding,
    ZEncoding,
)
from opengwasdb.model.enums import CompletionState
from opengwasdb.store import open_store
from opengwasdb.top_hits import slices_are_ordered
from opengwasdb.validation import RULES, validate_store
from opengwasdb.validation.rule import Fold, Kind, Target
from opengwasdb.validation.rules import associations
from opengwasdb.validation.run import evaluate
from opengwasdb.validation.scans import BYTES_PER_CELL, Block, CellIndex, band_limits

FOLDS: dict[str, Fold[Any, Any]] = {rule.name: rule for rule in RULES if isinstance(rule, Fold)}

FLOAT16 = StoreEncoding(z=ZEncoding(kind="float16"), se=SeEncoding(kind="float16"))


def _grid(group: Any = None, n_variants: int = 3, n_analyses: int = 2) -> Target:
    target = Target(Kind.GRID, "data.zarr", group=group, encoding=FLOAT16)
    target.remember("n_variants", lambda: n_variants)
    target.remember("n_analyses", lambda: n_analyses)
    return target


def _judged(name: str, blocks: Sequence[Block], target: Target | None = None) -> list[str]:
    """What rule `name` reports on `blocks`, added up from its zero as the runner does."""
    fold = FOLDS[name]
    target = target if target is not None else _grid()
    total = fold.zero
    for block in blocks:
        total = total + fold.measure(target, block)
    return [finding.message for finding in fold.judged(target, total)]


def _checked(name: str, target: Target) -> list[str]:
    rule = next(rule for rule in associations.CHECKS if rule.name == name)
    return [finding.message for finding in evaluate([rule], [target]).findings]


def test_top_hit_arrays_remain_in_their_compact_contract_dtypes() -> None:
    group = zarr.group()
    arrays = {
        "variant_index": np.array([1, 2], dtype=np.uint32),
        "analysis_index": np.array([0, 1], dtype=np.uint32),
        "z": np.array([7.0, 8.0], dtype=np.float32),
        "se": np.array([0.1, 0.2], dtype=np.float32),
        "abs_z": np.array([7.0, 8.0], dtype=np.float32),
        "p_value": np.array([1e-10, 1e-12], dtype=np.float64),
        "analysis_offsets": np.array([0, 1, 2], dtype=np.uint64),
        "imputed": np.array([0, 1], dtype=np.uint8),
        "eaf": np.array([0.3, 0.4], dtype=np.float32),
    }
    for name, values in arrays.items():
        group.create_array(name, data=values)

    loaded = top_hits.load_tier_arrays(group)

    assert loaded["rows"].dtype == np.uint32
    assert loaded["cols"].dtype == np.uint32
    assert loaded["offsets"].dtype == np.uint64
    assert sum(value.nbytes for value in loaded.values() if value is not None) == 74


def test_strict_top_hit_slice_order_proves_cell_keys_are_unique() -> None:
    offsets = np.array([0, 3], dtype=np.uint64)
    analyses = np.zeros(3, dtype=np.uint32)

    assert slices_are_ordered(
        analyses, np.array([1, 2, 3], dtype=np.uint32), offsets, 1
    )
    assert not slices_are_ordered(
        analyses, np.array([1, 1, 3], dtype=np.uint32), offsets, 1
    )
    assert not slices_are_ordered(
        analyses, np.array([2, 1, 3], dtype=np.uint32), offsets, 1
    )


def test_a_band_never_exceeds_the_decoded_memory_budget() -> None:
    """The pre-#254 fixed 250,000-row band handed a wide store 506 million cells."""
    n_variants, n_analyses = 250_001, 2_024
    bands = list(band_limits(n_variants, n_analyses, 1_000))
    assert bands[0][0] == 0 and bands[-1][1] == n_variants, "every row must be covered"
    assert all(b[0] == a[1] for a, b in zip(bands, bands[1:], strict=False))
    assert max((r1 - r0) * n_analyses for r0, r1 in bands) * BYTES_PER_CELL <= 1024**3


# ── The grid's planes: presence, shapes, per-variant side lengths ─────────


def test_grid_planes_name_every_missing_plane_z_first() -> None:
    assert _checked("grid.planes", _grid(zarr.group())) == [
        "missing data.zarr/z",
        "missing data.zarr/se",
    ]
    root = zarr.group()
    root.create_array("z", data=np.zeros((3, 2), dtype="float16"))
    assert _checked("grid.planes", _grid(root)) == ["missing data.zarr/se"]


def test_grid_planes_record_every_misshaped_plane_in_order() -> None:
    root = zarr.group()
    root.create_array("z", data=np.zeros((3, 2), dtype="int16"))
    root.create_array("se", data=np.zeros((2, 2), dtype="float16"))
    root.create_array("eaf", data=np.zeros((2, 2), dtype="float32"))
    assert _checked("grid.planes", _grid(root)) == [
        "se shape (2, 2) does not match (3, 2)",
        "eaf shape (2, 2) does not match (3, 2)",
    ]


def test_grid_planes_pass_an_aligned_eaf_plane() -> None:
    root = zarr.group()
    for name, dtype in (("z", "int16"), ("se", "float16"), ("eaf", "float32")):
        root.create_array(name, data=np.zeros((3, 2), dtype=dtype))
    assert _checked("grid.planes", _grid(root)) == []


def test_eaf_side_lengths_reject_short_per_variant_side_arrays() -> None:
    root = zarr.group()
    root.create_array(EAF_BASELINE, data=np.zeros(4, dtype="float32"))
    root.create_array(EAF_REFERENCE, data=np.zeros(5, dtype="float32"))
    assert _checked("grid.eaf_side_lengths", _grid(root, n_variants=6)) == [
        f"{EAF_BASELINE} has 4 entries but the variant axis has 6",
        f"{EAF_REFERENCE} has 5 entries but the variant axis has 6",
    ]


# ── Value rules: one fold, one finding ─────────────────────────────────────

_PRESENT = np.ones((1, 2), dtype=np.float32)


def test_one_band_with_a_negative_se_fails_the_rule() -> None:
    clean = Block.of(_PRESENT, [[0.1, 0.2]])
    bad = Block.of(_PRESENT, [[0.1, -0.2]], offset=2)
    assert _judged("associations.se_negative", [clean]) == []
    assert _judged("associations.se_negative", [clean, bad]) == [
        "se contains negative finite values"
    ]
    infinite_and_missing = Block.of(_PRESENT, [[-np.inf, np.nan]])
    assert _judged("associations.se_negative", [infinite_and_missing]) == []


def test_z_and_se_markers_must_agree_cell_by_cell() -> None:
    z = np.array([[np.nan, 1.0], [1.0, np.nan]], dtype=np.float32)
    crossed = Block.of(z, [[1.0, np.nan], [np.nan, 2.0]])
    agreeing = Block.of(z, [[np.nan, 1.0], [2.0, np.nan]])
    assert _judged("associations.markers_paired", [crossed]) == [
        "z and se missingness is inconsistent"
    ]
    assert _judged("associations.markers_paired", [agreeing]) == []


def test_a_mask_value_other_than_0_or_1_fails_wherever_it_sits() -> None:
    """A 2 beside a partial cell still reports the mask's own finding."""
    flags = np.array([[0, 2]], dtype=np.uint8)
    block = Block.of([[1.0, np.nan]], [[1.0, 1.0]], imputed=flags)
    assert _judged("completion.mask_binary", [block]) == [
        "data.zarr/imputed contains values other than 0 and 1"
    ]
    assert _judged("associations.markers_paired", [block]) == [
        "z and se missingness is inconsistent"
    ]


def test_an_imputed_cell_missing_z_or_se_is_reported_by_each() -> None:
    flags = np.array([[1, 1]], dtype=np.uint8)
    bad = Block.of([[np.nan, 1.0]], [[np.nan, np.nan]], imputed=flags)
    assert _judged("completion.imputed_present", [bad]) == [
        "imputed=1 cells have missing z-scores",
        "imputed=1 cells have NaN se values",
    ]
    present = Block.of(_PRESENT, [[1.0, 2.0]], imputed=flags)
    assert _judged("completion.imputed_present", [present]) == []


def test_an_infinite_imputed_se_is_present() -> None:
    """An infinite value is a value (spec §15, ADR 0070); the warning counts it."""
    block = Block.of([[1.0]], [[np.inf]], imputed=np.array([[1]], dtype=np.uint8))
    assert _judged("completion.imputed_present", [block]) == []
    assert _judged("associations.markers_paired", [block]) == []


def test_imputed_cells_are_counted_per_analysis_across_bands() -> None:
    fold = FOLDS["completion.declaration"]
    first = Block.of(_PRESENT, _PRESENT, imputed=np.array([[1, 0]], dtype=np.uint8))
    second = Block.of(_PRESENT, _PRESENT, imputed=np.array([[0, 1]], dtype=np.uint8), offset=2)
    total = fold.zero + fold.measure(_grid(), first) + fold.measure(_grid(), second)
    assert total.tolist() == [1, 1]


def test_an_off_panel_row_with_an_imputed_cell_fails() -> None:
    imputed = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    on_panel = np.array([1, 0], dtype=np.uint8)
    z = np.ones((2, 2), dtype=np.float32)
    off_panel = Block.of(z, z, imputed=imputed, on_panel=on_panel)
    assert _judged("completion.off_panel", [off_panel]) == [
        "off-panel (on_panel=0) rows have imputed=1 cells — off-panel is never imputable"
    ]
    on_panel_only = Block.of(z, z, imputed=imputed[::-1].copy(), on_panel=on_panel)
    assert _judged("completion.off_panel", [on_panel_only]) == []


def test_a_decoded_frequency_outside_0_1_fails() -> None:
    outside = Block.of(_PRESENT, _PRESENT, eaf=[[0.3, 1.4]])
    missing = Block.of(_PRESENT, _PRESENT, eaf=[[0.3, np.nan]])
    assert _judged("associations.eaf_range", [outside]) == [
        "data.zarr/eaf contains finite values outside [0, 1]"
    ]
    assert _judged("associations.eaf_range", [missing]) == []


def _codes_block(codes: np.ndarray, offset: int) -> Block:
    zeros = np.zeros(codes.shape, dtype=np.float32)
    return Block(
        offset=offset,
        n_analyses=int(codes.shape[1]),
        z_codes=codes,
        se_codes=zeros,
        z_missing=np.zeros(codes.shape, dtype=bool),
    )


def test_marked_cells_are_flat_plane_positions_in_block_order() -> None:
    fold = FOLDS["side_tables.z_overflow"]
    fixed = StoreEncoding(
        z=ZEncoding(kind="int16_fixed", scale=1024), se=SeEncoding(kind="float16")
    )
    target = Target(Kind.GRID, "data.zarr", encoding=fixed)
    codes = np.array([[Z_OVERFLOW, 0], [0, Z_OVERFLOW]], dtype=np.int16)

    total = fold.zero
    for block in (_codes_block(codes, 0), _codes_block(codes[::-1].copy(), 4)):
        total = total + fold.measure(target, block)

    assert total.positions.tolist() == [0, 3, 5, 6]
    assert fold.zero.positions.dtype == np.int64 and len(fold.zero.positions) == 0


def test_a_band_locates_the_cells_it_holds_by_their_grid_coordinates() -> None:
    zeros = np.zeros((2, 3), dtype=np.float32)
    band = Block.of(zeros, zeros, offset=2 * 3)  # rows 2 and 3 of a three-Analysis grid
    rows = np.array([1, 2, 3, 3, 4], dtype=np.uint32)
    cols = np.array([0, 2, 0, 1, 0], dtype=np.uint32)

    found, flat = band.locate(CellIndex(rows, cols))

    assert found.tolist() == [1, 2, 3]
    assert flat.tolist() == [2, 3, 4]


def test_a_tier_flag_is_held_to_a_stored_mask_whatever_the_completion_state(
    dense_store_path: Path,
) -> None:
    """An Observed-Only grid's mask flags no imputed cell, but a tier copies the mask
    as stored, so a tier flag of 0 against a stored 1 is still a wrong flag."""
    store = open_store(dense_store_path)
    assert store.manifest.completion_state is CompletionState.OBSERVED_ONLY
    root = store.arrays(mode="r+")
    tiers = top_hits.index_group(root)
    keys = list(tiers)
    row, col = int(tiers[keys[0]]["variant_index"][0]), int(tiers[keys[0]]["analysis_index"][0])
    mask = np.zeros(root["z"].shape, dtype=np.uint8)
    mask[row, col] = 1
    root.create_array("imputed", shape=mask.shape, dtype="uint8")[:] = mask
    for key in keys:
        n = int(tiers[key]["variant_index"].shape[0])
        assert row in tiers[key]["variant_index"][:].tolist()
        tiers[key].create_array("imputed", shape=(n,), dtype="uint8")[:] = 0

    result = validate_store(dense_store_path, rules=["top_hits.imputed"])

    assert "top_hits.imputed" not in result.skipped
    assert sorted(f.message for f in result.of("top_hits.imputed")) == sorted(
        f"top-hit index {key} contains imputed value inconsistent with imputed array"
        for key in keys
    )
