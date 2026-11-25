"""Row and band geometry (ld-store §2, §4, §5) against loops over positions, one pair at a time:
what the spec says in words, written the slow way, never by calling the code under test."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from opengwasdb.ld.format.geometry import (
    chromosome_ranges,
    chromosome_ranges_from_counts,
    near_address,
    near_pair_of,
    spans,
    spans_at,
)


def test_closed_chromosome_counts_form_joined_row_ranges() -> None:
    assert chromosome_ranges_from_counts([("1", 3), ("2", 3), ("22", 2)]) == {
        "1": (0, 3),
        "2": (3, 6),
        "22": (6, 8),
    }
    assert chromosome_ranges_from_counts([]) == {}


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_joined_chromosome_counts_must_be_positive_integers(count) -> None:
    with pytest.raises(ValueError, match=r"chromosome 2.*positive integer"):
        chromosome_ranges_from_counts([("1", 3), ("2", count)])


def test_joined_chromosome_counts_cannot_repeat_a_chromosome() -> None:
    with pytest.raises(ValueError, match="chromosome 1 is not contiguous"):
        chromosome_ranges_from_counts([("1", 3), ("2", 3), ("1", 2)])


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


# ------------------------------------------------------------------ bounded iteration


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


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype("U3"),
        np.dtype("S3"),
        np.dtype(object),
        np.dtypes.StringDType(),
        np.dtypes.StringDType(na_object=None),
    ],
    ids=["U", "S", "object", "StringDType", "StringDType-na"],
)
def test_chromosome_ranges_names_are_str_whatever_the_labels_dtype(dtype: np.dtype) -> None:
    labels = np.array(["21", "21", "22", "X"], dtype=dtype)
    ranges = chromosome_ranges(labels)
    assert ranges == {"21": (0, 2), "22": (2, 3), "X": (3, 4)}
    assert all(type(name) is str for name in ranges)
