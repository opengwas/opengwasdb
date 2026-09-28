"""The bounded per-variant EAF baseline for a Ragged Overflow plane (issue #226).

`eaf_baseline_from_pairs` sizes every intermediate to the plane, so the Hybrid
joint SE fit asked for ~750 GiB on OGS-00011's 15,078,327,210-cell Overflow and
died. `eaf_baseline_from_sorted_runs` walks variant-aligned blocks of the
per-Analysis runs instead. What matters is that it is *exactly* the same
baseline -- a store whose baselines shifted would decode every residual
differently -- so the equivalence tests here compare the two paths cell for
cell, and the memory test pins the property that motivated the change.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from opengwasdb.encoding.codec import (
    EafBaselineError,
    eaf_baseline_from_pairs,
    eaf_baseline_from_sorted_runs,
)


def _runs(
    n_analyses: int,
    n_variants: int,
    seed: int,
    *,
    nan_fraction: float = 0.0,
    per_analysis: int | None = None,
):
    """Per-Analysis `(variant_index, eaf)` runs, each sorted by variant index.

    `per_analysis` fixes every run's length, so a test can scale the cell count
    by the Analysis count exactly rather than on average.
    """
    rng = np.random.default_rng(seed)
    index_runs, value_runs = [], []
    for _ in range(n_analyses):
        size = per_analysis if per_analysis is not None else int(rng.integers(0, n_variants + 1))
        index = np.sort(rng.choice(n_variants, size=size, replace=False)).astype(np.int32)
        values = rng.uniform(0.001, 0.999, size=size).astype(np.float32)
        if nan_fraction:
            values[rng.random(size) < nan_fraction] = np.nan
        index_runs.append(index)
        value_runs.append(values)
    return index_runs, value_runs


def _flat(index_runs, value_runs):
    return (
        np.concatenate(index_runs) if index_runs else np.empty(0, dtype=np.int32),
        np.concatenate(value_runs) if value_runs else np.empty(0, dtype=np.float32),
    )


def _assert_identical(got, want):
    """Bit-identical, counting NaN as equal -- a variant with no baseline must
    stay without one rather than becoming a number."""
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(got[~np.isnan(got)], want[~np.isnan(want)])


# ── Equivalence with the unbounded path ──────────────────────────────────────


@pytest.mark.parametrize("seed", range(6))
def test_bounded_baseline_is_bit_identical_to_the_flat_path(seed):
    n_variants = 400
    index_runs, value_runs = _runs(12, n_variants, seed, nan_fraction=0.2)
    vi, values = _flat(index_runs, value_runs)

    _assert_identical(
        eaf_baseline_from_sorted_runs(index_runs, value_runs, n_variants),
        eaf_baseline_from_pairs(vi, values, n_variants),
    )


@pytest.mark.parametrize("cell_budget", [1, 2, 3, 7, 64, 10_000])
def test_every_block_size_gives_the_same_answer(cell_budget):
    """A variant's cells must never be split across blocks: if they were, each
    half would take its own median and the result would drift with the budget."""
    n_variants = 250
    index_runs, value_runs = _runs(9, n_variants, 11, nan_fraction=0.15)
    vi, values = _flat(index_runs, value_runs)

    _assert_identical(
        eaf_baseline_from_sorted_runs(index_runs, value_runs, n_variants, cell_budget=cell_budget),
        eaf_baseline_from_pairs(vi, values, n_variants),
    )


def test_even_and_odd_counts_take_the_same_median_as_numpy():
    """Two reporters average their logits, three take the middle one."""
    index_runs = [np.array([0, 1], dtype=np.int32)] * 3
    value_runs = [
        np.array([0.10, 0.20], dtype=np.float32),
        np.array([0.30, 0.40], dtype=np.float32),
        np.array([0.50, 0.90], dtype=np.float32),
    ]
    three = eaf_baseline_from_sorted_runs(index_runs, value_runs, 2)
    two = eaf_baseline_from_sorted_runs(index_runs[:2], value_runs[:2], 2)

    _assert_identical(three, eaf_baseline_from_pairs(*_flat(index_runs, value_runs), 2))
    _assert_identical(two, eaf_baseline_from_pairs(*_flat(index_runs[:2], value_runs[:2]), 2))


# ── Degenerate planes ────────────────────────────────────────────────────────


def test_a_variant_no_analysis_reports_gets_no_baseline():
    index_runs = [np.array([0, 2], dtype=np.int32)]
    value_runs = [np.array([0.4, 0.6], dtype=np.float32)]
    baseline = eaf_baseline_from_sorted_runs(index_runs, value_runs, 4)
    assert np.isnan(baseline[1]) and np.isnan(baseline[3])
    assert np.isfinite(baseline[0]) and np.isfinite(baseline[2])


@pytest.mark.parametrize(
    ("index_runs", "value_runs"),
    [
        ([], []),
        ([np.empty(0, dtype=np.int32)], [np.empty(0, dtype=np.float32)]),
        ([np.array([0, 1], dtype=np.int32)], [np.array([np.nan, np.nan], dtype=np.float32)]),
        ([np.array([0, 1], dtype=np.int32)], [np.array([0.0, 1.0], dtype=np.float32)]),
    ],
    ids=["no runs", "empty run", "all NaN", "outside the open interval"],
)
def test_a_plane_with_nothing_usable_yields_no_baselines(index_runs, value_runs):
    baseline = eaf_baseline_from_sorted_runs(index_runs, value_runs, 2)
    assert baseline.shape == (2,)
    assert np.isnan(baseline).all()


def test_zero_variants_is_an_empty_baseline():
    assert eaf_baseline_from_sorted_runs([], [], 0).shape == (0,)


# ── The contract the bounded walk rests on ───────────────────────────────────


def test_an_unsorted_run_is_refused_rather_than_silently_miscomputed():
    index_runs = [np.array([2, 0, 1], dtype=np.int32)]
    value_runs = [np.array([0.2, 0.4, 0.6], dtype=np.float32)]
    with pytest.raises(EafBaselineError, match="not sorted by variant index"):
        eaf_baseline_from_sorted_runs(index_runs, value_runs, 3)


def test_mismatched_run_lengths_are_refused():
    with pytest.raises(EafBaselineError, match="does not match"):
        eaf_baseline_from_sorted_runs(
            [np.array([0, 1], dtype=np.int32)], [np.array([0.5], dtype=np.float32)], 2
        )


def test_a_different_number_of_index_and_value_runs_is_refused():
    with pytest.raises(EafBaselineError, match="does not match"):
        eaf_baseline_from_sorted_runs([np.array([0], dtype=np.int32)], [], 1)


# ── Memory: bounded by the block, not by the plane ───────────────────────────


def _peak_bytes(fn) -> int:
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_peak_memory_does_not_grow_with_the_cell_count():
    """Eight times the cells at a fixed budget must not cost eight times the
    peak -- that ratio is exactly what made the OGS-00011 Overflow unbuildable."""
    n_variants = 20_000
    budget = 4096
    small_index, small_values = _runs(4, n_variants, 3, per_analysis=n_variants // 2)
    large_index, large_values = _runs(32, n_variants, 3, per_analysis=n_variants // 2)
    small_cells = sum(r.size for r in small_index)
    large_cells = sum(r.size for r in large_index)
    assert large_cells == 8 * small_cells, (small_cells, large_cells)

    peak_small = _peak_bytes(
        lambda: eaf_baseline_from_sorted_runs(
            small_index, small_values, n_variants, cell_budget=budget
        )
    )
    peak_large = _peak_bytes(
        lambda: eaf_baseline_from_sorted_runs(
            large_index, large_values, n_variants, cell_budget=budget
        )
    )

    growth = peak_large / peak_small
    assert growth < 2.0, f"peak grew {growth:.1f}x for {large_cells / small_cells:.1f}x the cells"


def test_peak_memory_follows_the_budget():
    """The knob is the budget, so raising it is what raises the working set."""
    n_variants = 20_000
    index_runs, value_runs = _runs(32, n_variants, 5, per_analysis=n_variants // 2)
    lean = _peak_bytes(
        lambda: eaf_baseline_from_sorted_runs(index_runs, value_runs, n_variants, cell_budget=2048)
    )
    fat = _peak_bytes(
        lambda: eaf_baseline_from_sorted_runs(
            index_runs, value_runs, n_variants, cell_budget=1 << 20
        )
    )
    assert fat > lean, (lean, fat)
