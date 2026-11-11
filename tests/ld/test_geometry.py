"""Row and band geometry (spec §2, §4, §5) against loops over positions, one pair at a time:
what the spec says in words, written the slow way, never by calling the code under test."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ogld.format import FAR_BP, NEAR_BP
from ogld.format.geometry import (
    blocks,
    chromosome_index,
    chromosome_ranges,
    global_rows,
    indptr_of,
    local_rows,
    near_address,
    near_pair_of,
    near_row_lengths,
    owners,
    position_rows,
    row_pairs,
    spans,
    spans_at,
    start_within,
    stop_within,
    windows,
    within,
)


@st.composite
def ascending(draw: st.DrawFn, distance: int, *, most: int = 30) -> list[int]:
    """Ascending positions, empty and one variant included, whose gaps are 0 (two variants at
    one position), exactly `distance`, one more, or anything: where an off-by-one shows."""
    gap = st.one_of(
        st.just(0), st.just(distance), st.just(distance + 1), st.integers(0, 2 * distance + 2)
    )
    gaps = draw(st.lists(gap, max_size=most))
    first = draw(st.integers(1, 5))
    return [first + sum(gaps[:i]) for i in range(len(gaps))]


distances = st.integers(1, 12)


def brute_stop(position: list[int], distance: int) -> list[int]:
    """Row l's stop: how many variants m, before or after, have p_m − p_l ≤ distance."""
    return [sum(1 for q in position if q - p <= distance) for p in position]


@given(data=st.data(), distance=distances)
def test_a_stop_counts_every_variant_within_the_distance(
    data: st.DataObject, distance: int
) -> None:
    position = data.draw(ascending(distance))
    assert stop_within(position, distance).tolist() == brute_stop(position, distance)


@given(data=st.data(), distance=distances)
def test_a_near_row_holds_the_later_variants_within_near_bp(
    data: st.DataObject, distance: int
) -> None:
    position = data.draw(ascending(distance))
    expected = [
        sum(1 for m in range(row + 1, len(position)) if position[m] - position[row] <= distance)
        for row in range(len(position))
    ]
    got = near_row_lengths(position, distance)
    assert got.tolist() == expected
    assert got.dtype == np.int64


@given(data=st.data(), distance=distances)
def test_the_near_start_is_the_first_variant_within_near_bp_before(
    data: st.DataObject, distance: int
) -> None:
    position = data.draw(ascending(distance))
    expected = [
        min(m for m in range(len(position)) if position[row] - position[m] <= distance)
        for row in range(len(position))
    ]
    assert start_within(position, distance).tolist() == expected
    some = [row for row in range(len(position)) if row % 2 == 0][::-1]
    assert start_within(position, distance, some).tolist() == [expected[row] for row in some]


@pytest.mark.parametrize(("distance", "name"), [(NEAR_BP, "near_bp"), (FAR_BP, "far_bp")])
def test_exactly_the_distance_apart_is_within_and_one_more_is_not(distance: int, name: str) -> None:
    position = [100, 100 + distance, 100 + distance + 1]
    # Row 0 reaches the variant exactly `distance` away, not the one 1 bp farther.
    assert stop_within(position, distance).tolist() == [2, 3, 3], name
    assert near_row_lengths(position, distance).tolist() == [1, 1, 0], name
    assert start_within(position, distance).tolist() == [0, 0, 1], name


def test_two_variants_at_one_position_see_each_other() -> None:
    position = [5, 5, 5, 9]
    assert stop_within(position, 3).tolist() == [3, 3, 3, 4]
    assert near_row_lengths(position, 3).tolist() == [2, 1, 0, 0]
    assert start_within(position, 3).tolist() == [0, 0, 0, 3]


def test_a_one_variant_chromosome_has_one_empty_row() -> None:
    assert stop_within([7], NEAR_BP).tolist() == [1]
    assert near_row_lengths([7], NEAR_BP).tolist() == [0]
    assert start_within([7], NEAR_BP).tolist() == [0]


def test_the_last_row_is_empty_and_its_stop_is_the_end() -> None:
    position = [1, 2, 3, 100]
    assert near_row_lengths(position, 10).tolist() == [2, 1, 0, 0]
    assert stop_within(position, 10).tolist()[-1] == len(position)
    assert start_within(position, 10).tolist()[0] == 0


def test_an_empty_chromosome_has_no_rows() -> None:
    for got in (stop_within([], 5), start_within([], 5), near_row_lengths([], 5)):
        assert got.tolist() == []
        assert got.dtype == np.int64
    lo, hi = position_rows([], [1, 4], [9, 6])
    assert (lo.tolist(), hi.tolist()) == ([0, 0], [0, 0])
    lo, hi = position_rows([], 1, 9)
    assert (int(lo), int(hi)) == (0, 0)


@given(
    data=st.data(),
    distance=distances,
    start=st.integers(0, 40),
    width=st.integers(0, 40),
)
def test_a_position_range_holds_the_rows_between_both_ends_inclusive(
    data: st.DataObject, distance: int, start: int, width: int
) -> None:
    position = data.draw(ascending(distance))
    stop = start + width
    lo, hi = position_rows(position, start, stop)
    inside = [m for m, p in enumerate(position) if start <= p <= stop]
    assert list(range(int(lo), int(hi))) == inside


@given(data=st.data(), distance=distances)
def test_a_pair_is_stored_where_the_band_enumerates_it(data: st.DataObject, distance: int) -> None:
    position = data.draw(ascending(distance))
    band = [
        (row, m)
        for row in range(len(position))
        for m in range(row + 1, len(position))
        if position[m] - position[row] <= distance
    ]
    indptr = [0]  # each row's first address, counted off the enumerated band
    for row in range(len(position)):
        indptr.append(indptr[-1] + sum(1 for first, _ in band if first == row))
    assert len(band) == indptr[-1]
    lo = np.array([row for row, _ in band], dtype=np.int64)
    hi = np.array([m for _, m in band], dtype=np.int64)
    assert near_address(indptr, lo, hi).tolist() == list(range(len(band)))
    got_lo, got_hi = near_pair_of(indptr, np.arange(len(band)))
    assert (got_lo.tolist(), got_hi.tolist()) == (lo.tolist(), hi.tolist())


def test_the_address_of_a_pair_is_its_rows_start_plus_its_distance_in_rows() -> None:
    indptr = [0, 3, 5, 6, 6]  # rows of 3, 2, 1 and 0 codes: pairs (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)
    assert near_address(indptr, [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]).tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
    assert near_address(indptr, [], []).tolist() == []


# ------------------------------------------------------------------ chromosomes

labels = st.sampled_from(["1", "2", "10", "X"])


def runs_of(chrom: list[str]) -> dict[str, tuple[int, int]] | None:
    """Each label's [first row, last row + 1) by walking the rows; None when a label returns."""
    runs: dict[str, tuple[int, int]] = {}
    for row, name in enumerate(chrom):
        if name not in runs:
            runs[name] = (row, row + 1)
        elif runs[name][1] == row:
            runs[name] = (runs[name][0], row + 1)
        else:
            return None
    return runs


@given(st.lists(labels, max_size=12))
def test_chromosome_ranges_are_the_runs_of_equal_labels(chrom: list[str]) -> None:
    runs = runs_of(chrom)
    if runs is not None:
        assert chromosome_ranges(np.array(chrom, dtype=np.str_)) == runs
    else:
        with pytest.raises(ValueError, match="not contiguous"):
            chromosome_ranges(np.array(chrom, dtype=np.str_))


def test_a_one_variant_chromosome_and_the_first_and_last_rows_are_ranges() -> None:
    ranges = chromosome_ranges(np.array(["1", "1", "2", "X"], dtype=np.str_))
    assert ranges == {"1": (0, 2), "2": (2, 3), "X": (3, 4)}


def test_no_variants_have_no_chromosomes() -> None:
    assert chromosome_ranges(np.array([], dtype=np.str_)) == {}


@given(st.lists(st.integers(1, 5), min_size=1, max_size=6))
def test_the_chromosome_index_of_a_row_is_its_place_among_the_stops(sizes: list[int]) -> None:
    stops = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    expected = [
        i for i, size in enumerate(sizes) for _ in range(size)
    ]  # the chromosome of each row
    total = stops[-1]
    assert chromosome_index(stops, np.arange(total)).tolist() == expected
    assert chromosome_index(stops, [0, total - 1]).tolist() == [0, len(sizes) - 1]


@pytest.mark.parametrize("row", [-1, 6, 7])
def test_a_row_outside_the_variant_set_has_no_chromosome(row: int) -> None:
    # −1 is what an Absent ALID reads as: it is not the first chromosome, nor the last row.
    with pytest.raises(ValueError, match="not in the Variant Set of 6 rows"):
        chromosome_index([2, 6], [0, row])
    with pytest.raises(ValueError, match="of 0 rows"):
        chromosome_index([], [0])


def test_local_and_global_rows_convert_both_ways() -> None:
    assert local_rows([10, 11, 14], 10).tolist() == [0, 1, 4]
    assert global_rows([0, 1, 4], 10).tolist() == [10, 11, 14]
    assert local_rows([], 3).tolist() == []


@given(
    gap=st.lists(st.integers(-30, 30), max_size=20),
    distance=st.integers(0, 12),
)
def test_within_is_inclusive_either_way_round(gap: list[int], distance: int) -> None:
    assert within(gap, distance).tolist() == [abs(g) <= distance for g in gap]


def test_a_pair_exactly_at_the_distance_is_within_and_one_more_is_not() -> None:
    assert within([NEAR_BP, NEAR_BP + 1, -NEAR_BP, -NEAR_BP - 1], NEAR_BP).tolist() == [
        True,
        False,
        True,
        False,
    ]
    assert within([FAR_BP, FAR_BP + 1], FAR_BP).tolist() == [True, False]


# ------------------------------------------------------------------ ragged ranges


def walked(first: list[int], length: list[int]) -> tuple[list[int], list[int]]:
    """Each element of the spans, in order: the span it belongs to and its value."""
    span, value = [], []
    for i, (start, count) in enumerate(zip(first, length, strict=True)):
        for k in range(count):
            span.append(i)
            value.append(start + k)
    return span, value


@pytest.mark.parametrize("seed", range(6))
def test_spans_at_inverts_spans_element_by_element(seed: int) -> None:
    rng = np.random.default_rng(seed)
    length = rng.integers(0, 6, size=40)
    length[[0, 7, 8, -1]] = 0  # empty spans first, together and last
    first = rng.integers(-5, 50, size=40)
    span, value = walked(first.tolist(), length.tolist())
    assert len(value) > 50, "the spans must hold many elements for this to mean anything"
    np.testing.assert_array_equal(spans(first, length), value)
    at = rng.permutation(len(value))
    found_span, found_value = spans_at(first, length, at)
    np.testing.assert_array_equal(found_span, np.asarray(span)[at])
    np.testing.assert_array_equal(found_value, np.asarray(value)[at])


@given(st.lists(st.integers(0, 5), max_size=12))
def test_owners_name_the_span_of_each_element_and_indptr_counts_them_off(length: list[int]) -> None:
    named = [i for i, count in enumerate(length) for _ in range(count)]
    assert owners(length).tolist() == named
    assert owners(length, [100 + i for i in range(len(length))]).tolist() == [
        100 + i for i in named
    ]
    running = [0]
    for count in length:
        running.append(running[-1] + count)
    assert indptr_of(length).tolist() == running
    assert indptr_of(length).dtype == np.int64


def test_no_spans_are_no_elements() -> None:
    assert spans([], []).tolist() == []
    assert spans([4, 9], [0, 0]).tolist() == []


# ------------------------------------------------------------------ bounded iteration


@given(start=st.integers(0, 20), size=st.integers(0, 40), step=st.integers(1, 9))
def test_windows_tile_the_range_in_steps(start: int, size: int, step: int) -> None:
    covered = [i for lo, hi in windows(start, start + size, step) for i in range(lo, hi)]
    assert covered == list(range(start, start + size))
    assert all(0 < hi - lo <= step for lo, hi in windows(start, start + size, step))


@given(
    lengths=st.lists(st.tuples(st.integers(0, 6), st.integers(0, 6)), min_size=1, max_size=15),
    limits=st.tuples(st.integers(1, 10), st.integers(1, 10)),
    both=st.booleans(),
)
def test_blocks_are_the_longest_runs_of_rows_within_every_limit(
    lengths: list[tuple[int, int]], limits: tuple[int, int], both: bool
) -> None:
    rows = len(lengths)
    indptrs = [
        np.concatenate([[0], np.cumsum([row[k] for row in lengths])]).astype(np.int64)
        for k in range(2 if both else 1)
    ]
    given_limits = list(zip(indptrs, limits, strict=False))
    expected, a = [], 0
    while a < rows:
        b = a + 1
        while b < rows and all(p[b + 1] - p[a] <= most for p, most in given_limits):
            b += 1
        expected.append((a, b))
        a = b
    assert list(blocks(*given_limits)) == expected


def test_a_row_holding_more_than_the_limit_is_a_block_of_its_own() -> None:
    indptr = np.array([0, 1, 11, 12], dtype=np.int64)
    assert list(blocks((indptr, 4))) == [(0, 1), (1, 2), (2, 3)]
    assert list(blocks((np.array([0], dtype=np.int64), 4))) == []


@given(
    first=st.lists(st.integers(0, 8), max_size=12),
    reach=st.lists(st.integers(-2, 6), max_size=12),
    per_step=st.integers(1, 8),
)
def test_row_pairs_are_every_pair_of_a_rows_range_in_bounded_steps(
    first: list[int], reach: list[int], per_step: int
) -> None:
    n = min(len(first), len(reach))
    low = np.array(first[:n], dtype=np.int64)
    stop = low + np.array(reach[:n], dtype=np.int64)
    expected = [(i, j) for i in range(n) for j in range(int(low[i]), int(stop[i]))]
    steps = list(row_pairs(low, stop, per_step))
    got = [(int(i), int(j)) for rows, cols in steps for i, j in zip(rows, cols, strict=True)]
    assert got == expected
    for rows, _ in steps:
        # A step is whole rows; it exceeds per_step only when one row alone does.
        assert len(rows) <= per_step or len(set(rows.tolist())) == 1


def test_the_inverse_of_an_address_reads_a_slice_of_rows() -> None:
    # Rows 1 and 2 of the band above, whose codes sit at addresses 3, 4 and 5: the pairs are
    # (1, 2), (1, 3) and (2, 3), so in the slice's own rows (0, 1), (0, 2) and (1, 2).
    lo, hi = near_pair_of([3, 5, 6], [3, 4, 5])
    assert (lo.tolist(), hi.tolist()) == ([0, 0, 1], [1, 2, 2])


def test_chromosome_ranges_reads_numpy_variable_width_labels() -> None:
    # zarr returns a string array as numpy's StringDType, not a fixed-width `<U`.
    labels = np.array(["21", "21", "22", "22", "22", "X"], dtype=np.dtypes.StringDType())
    assert chromosome_ranges(labels) == {"21": (0, 2), "22": (2, 5), "X": (5, 6)}
    assert chromosome_ranges(labels[:0]) == {}
