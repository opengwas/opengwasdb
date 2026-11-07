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

import pathlib
from collections.abc import Sequence
from itertools import permutations
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr
from store_reads import element_reads, times_read

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
from opengwasdb.validation.rule import Fold, Kind, Target, ValidationOptions
from opengwasdb.validation.rules import associations
from opengwasdb.validation.rules import top_hits as top_hit_rules
from opengwasdb.validation.run import evaluate
from opengwasdb.encoding.banding import BYTES_PER_CELL, band_limits
from opengwasdb.validation.scans import Block, CellIndex
from opengwasdb.validation.validate import release_targets

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


def test_the_tiers_are_enumerated_in_threshold_order_not_store_order() -> None:
    """A Zarr group hands its keys back in its store's order, and that order is not
    stable: the same release listed its three tiers in three orders over six runs.
    Every `top_hits.*` rule enumerates them through the one listing, which sorts them
    by threshold, as `TOP_HIT_THRESHOLDS` does (#378).
    """

    def tier(threshold: float) -> Any:
        return SimpleNamespace(attrs={"threshold": threshold})

    # Held in the store's order, which is neither the threshold's nor the key's.
    index = {top_hits.threshold_key(t): tier(t) for t in (5e-4, 5e-8, 5e-6)}

    assert top_hit_rules._keys(_grid({"top_hits": index})) == [
        "p_5e_08",
        "p_5e_06",
        "p_5e_04",
    ]


def test_the_tier_listing_does_not_depend_on_the_order_the_tiers_are_in() -> None:
    """Every order the group could list its keys in gives the one listing.

    A Zarr group lists its keys in its store's order and promises nothing about it,
    so what a rule reports must not follow it. Two tiers written for one threshold
    are ordered by name, and a tier with no threshold to read sorts after every tier
    that has one, by name (#378).
    """

    def tier(threshold: float | None) -> Any:
        return SimpleNamespace(attrs={} if threshold is None else {"threshold": threshold})

    tiers = {
        "p_5e_08": tier(5e-8),
        "p_5e_06": tier(5e-8),  # a tie with p_5e_08, for the name to break
        "p_5e_04": tier(5e-4),
        "broken_a": tier(None),  # no threshold, and no key that spells one
        "broken_b": tier(None),
    }
    listed = ["p_5e_06", "p_5e_08", "p_5e_04", "broken_a", "broken_b"]

    for order in permutations(tiers):
        index = {key: tiers[key] for key in order}
        assert top_hit_rules._keys(_grid({"top_hits": index})) == listed, order


def test_malformed_tiers_are_diagnosed_and_listed_after_the_well_formed_ones(
    dense_store_path: pathlib.Path,
) -> None:
    """A tier whose `threshold` will not parse must not stop the rules that report it.

    `broken_a` and `broken_b` carry no arrays, so `top_hits.arrays` says what each is
    missing and the value rules go on to diagnose the tiers that do hold cells. A
    listing that parsed every threshold to sort spent the whole rule set on a float
    conversion instead (#378 review 2).
    """
    root = open_store(dense_store_path).arrays(mode="a")
    root["se"][0, 0] = np.float16(-0.1)
    root["se"][0, 1] = np.nan
    for name in ("broken_a", "broken_b"):
        root["top_hits"].create_group(name)

    result = validate_store(dense_store_path)

    missing = "abs_z, analysis_index, analysis_offsets, p_value, se, variant_index, z"
    assert result.failed() == {
        "associations.se_negative",
        "associations.markers_paired",
        "top_hits.arrays",
        "top_hits.se",
    }
    assert result.errors == [
        "se contains negative finite values",
        "z and se missingness is inconsistent",
        f"top-hit index broken_a is missing {missing}",
        f"top-hit index broken_b is missing {missing}",
        "top-hit index p_5e_08 contains se value inconsistent with decoded se plane",
        "top-hit index p_5e_06 contains se value inconsistent with decoded se plane",
        "top-hit index p_5e_04 contains se value inconsistent with decoded se plane",
    ]


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
    with_frequencies = _grid({"eaf": outside.eaf})
    assert _judged("associations.eaf_range", [outside], with_frequencies) == [
        "data.zarr/eaf contains finite values outside [0, 1]"
    ]
    assert _judged("associations.eaf_range", [missing], with_frequencies) == []


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


def test_cells_held_in_row_order_are_located_without_a_second_order() -> None:
    zeros = np.zeros((2, 3), dtype=np.float32)
    band = Block.of(zeros, zeros, offset=2 * 3)  # rows 2 and 3 of a three-Analysis grid
    cells = CellIndex(np.array([1, 2, 3, 3, 4], dtype=np.uint32), np.array([0, 2, 0, 1, 0]))

    found, flat = band.locate(cells)

    assert cells.by_row[0] is None
    assert found.tolist() == [1, 2, 3]
    assert flat.tolist() == [2, 3, 4]


def test_a_validation_reads_each_tier_array_once_and_holds_the_tiers_by_row(
    dense_store_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The structure rules run after the band scan has released the tiers, from what
    they learned in the same load; the scan holds each tier sorted by row."""
    tiers = top_hits.index_group(open_store(dense_store_path).arrays(mode="r"))
    arrays = [tiers[key][name] for key in tiers for name in ("variant_index", "z", "abs_z")]
    assert all(array.size for array in arrays), "the fixture's tiers must hold cells"

    with element_reads(monkeypatch) as reads:
        result = validate_store(dense_store_path)

    assert result.ok, result.errors
    assert [times_read(reads, array) for array in arrays] == [1.0] * len(arrays)
    grid = Target(Kind.GRID, "data.zarr", open_store(dense_store_path))
    assert all(tier.cells.by_row[0] is None for tier in top_hit_rules.sound_tiers(grid))


def _top_hit_holdings(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Target, str]]:
    """Every `top_hits/…` value a target is asked to hold that is more than a fact.

    The structure rules' small facts (a tier's problem, whether its `abs_z` agrees)
    are kept for the run; what the association scan reads to check the planes
    against, the tiers' arrays, is not.
    """
    held: list[tuple[Target, str]] = []
    remember = Target.remember

    def recording(self: Target, key: str, read: Any, *, until: str | None = None) -> Any:
        value = remember(self, key, read, until=until)
        if key.startswith("top_hits/") and not isinstance(value, (bool, str, type(None))):
            held.append((self, key))
        return value

    monkeypatch.setattr(Target, "remember", recording)
    return held


@pytest.mark.parametrize("interrupted", [False, True], ids=["finished", "interrupted"])
def test_the_association_scan_holds_the_top_hit_tiers_only_for_its_pass(
    dense_store_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, interrupted: bool
) -> None:
    """The real Dense scan, through `evaluate`: the tiers it held for its folds do not
    outlive its pass, whether the pass ends or is interrupted part-way. An
    interruption is no rule's failure: it reaches the caller."""
    store = open_store(dense_store_path)
    keys = list(top_hits.index_group(store.arrays(mode="r")))
    assert keys, "the fixture's grid must carry a Top-Hit Index"
    targets = release_targets(store, ValidationOptions())
    held = _top_hit_holdings(monkeypatch)
    located: list[int] = []
    if interrupted:

        def interrupt(self: Block, cells: CellIndex) -> tuple[np.ndarray, np.ndarray]:
            located.append(len(cells.rows))
            raise KeyboardInterrupt

        monkeypatch.setattr(Block, "locate", interrupt)
        with pytest.raises(KeyboardInterrupt):
            evaluate(RULES, targets)
        assert located, "the interruption must come from inside the scan's pass"
    else:
        result = evaluate(RULES, targets)
        assert result.ok, result.errors

    assert {key for _, key in held} >= {"top_hits/sound", *(f"top_hits/{key}" for key in keys)}
    assert [key for target, key in held if target.forget(key) is not None] == []


def test_a_band_locates_the_cells_it_holds_by_their_grid_coordinates() -> None:
    zeros = np.zeros((2, 3), dtype=np.float32)
    band = Block.of(zeros, zeros, offset=2 * 3)  # rows 2 and 3 of a three-Analysis grid
    rows = np.array([1, 2, 3, 3, 4], dtype=np.uint32)
    cols = np.array([0, 2, 0, 1, 0], dtype=np.uint32)

    found, flat = band.locate(CellIndex(rows, cols))

    assert found.tolist() == [1, 2, 3]
    assert flat.tolist() == [2, 3, 4]


def test_a_tier_flag_is_held_to_a_stored_mask_whatever_the_completion_state(
    dense_store_path: pathlib.Path,
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


def test_a_top_hit_rule_lists_its_tiers_in_threshold_order(
    dense_store_path: pathlib.Path,
) -> None:
    """The same three failing tiers come out in one order, every run.

    A Zarr group's key order is its store's, which a store does not promise: this
    release reported its three tiers in three orders over six runs, so `errors`
    did too (#378).
    """
    root = open_store(dense_store_path).arrays(mode="a")
    root["se"][0, 0] = np.float16(-0.1)
    root["se"][0, 1] = np.nan
    tiers = top_hits.index_group(open_store(dense_store_path).arrays(mode="r"))
    assert sorted(tiers) == ["p_5e_04", "p_5e_06", "p_5e_08"], "three tiers to fail"

    result = validate_store(dense_store_path)

    assert [f.message for f in result.of("top_hits.se")] == [
        f"top-hit index {key} contains se value inconsistent with decoded se plane"
        for key in ("p_5e_08", "p_5e_06", "p_5e_04")
    ]


def test_a_release_without_frequencies_records_the_range_rule_as_skipped(
    dense_store_path: pathlib.Path,
) -> None:
    """No frequency to judge is a skip, which a caller can tell from a range that held."""
    store = open_store(dense_store_path)
    assert "eaf" not in store.arrays(mode="r") and not store.manifest.encoding.eaf.reference

    result = validate_store(dense_store_path)

    assert result.ok, result.errors
    assert result.skipped["associations.eaf_range"] == ("no frequencies",)
