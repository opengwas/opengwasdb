"""Tests for the Dense-only query shapes a Hybrid store adds.

A Hybrid store answers a lookup in three parts -- the Dense Component's cells,
the shared-axis identifier resolve and each requested Analysis's Overflow
segment -- and only the first is what a reader who queries the reference axis
alone pays. The `_dense_only` twins exist to time that first part on its own,
so the two silent failures they can have are tested here: drawing their
variants from the wrong axis (which asks a smaller question), and reaching
through the Hybrid instead of the component (which times the very work they
exist to separate out).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from benchmarks import _query_shapes, benchmark_store_comparison

DENSE_ONLY_SHAPES = {
    "regional_dense_only",
    "regional_one_analysis_dense_only",
    "random_lookup_10_variants_100_analyses_dense_only",
    "random_lookup_100_variants_10_analyses_dense_only",
}
COMMON_SHAPES = set((
    "bulk", "phewas", "regional", "regional_one_analysis", "tophits",
    "random_lookup_10_variants_100_analyses", "random_lookup_100_variants_10_analyses",
))


def _result() -> dict[str, np.ndarray]:
    return {
        "variant_index": np.zeros(1, dtype="int32"),
        "analysis_index": np.zeros(1, dtype="int32"),
        "z": np.ones(1, dtype="float32"),
        "se": np.full(1, 0.1, dtype="float32"),
        "eaf": np.full(1, 0.3, dtype="float32"),
        "association_status": np.array(["observed"], dtype=object),
    }


class _Axis:
    """A variant axis that names every record after its own index."""

    def __init__(self, n_variants: int, in_window: list[int] | None = None) -> None:
        self.n_variants = n_variants
        self.in_window = [0, 1, 2] if in_window is None else in_window
        self.by_index_calls: list[int] = []

    def by_index(self, index: int) -> Any:
        self.by_index_calls.append(int(index))
        return SimpleNamespace(alid=f"1:{int(index)}:A:C", variant_index=int(index))

    def range_indices(self, chromosome: str, start: int, end: int) -> np.ndarray:
        return np.asarray(self.in_window)


class _PartialAxis(_Axis):
    """A Dense axis that cannot resolve every drawn index."""

    def by_index(self, index: int) -> Any:
        return None if int(index) % 4 == 0 else super().by_index(index)


class _Dense:
    """The Dense Component stand-in: records the calls its patterns make."""

    def __init__(self, n_variants: int) -> None:
        self._variant_axis = _Axis(n_variants)
        self.calls: list[tuple[Any, ...]] = []

    def lookup(self, alids: list[str], analyses: list[str]) -> dict[str, np.ndarray]:
        self.calls.append(("lookup", list(alids), list(analyses)))
        return _result()

    def range_phewas(self, chromosome: str, start: int, end: int) -> dict[str, np.ndarray]:
        self.calls.append(("range_phewas", chromosome, start, end))
        return _result()


class _Hybrid:
    """A Hybrid stand-in whose shared-axis surface fails if a pattern reaches it."""

    def __init__(self, n_panel: int) -> None:
        self._dense = _Dense(n_panel)
        self._dense_to_shared = np.arange(n_panel, dtype="int32")

    @property
    def _variant_axis(self) -> Any:
        raise AssertionError("a Dense-only pattern reached the shared axis")

    def lookup(self, *args: Any) -> dict[str, np.ndarray]:
        raise AssertionError("a Dense-only pattern called the Hybrid's lookup")

    def range_phewas(self, *args: Any) -> dict[str, np.ndarray]:
        raise AssertionError("a Dense-only pattern called the Hybrid's range_phewas")


def test_no_dense_component_means_no_dense_only_shapes():
    dense_store = SimpleNamespace(_variant_axis=_Axis(1000))
    ragged_store = SimpleNamespace(_variant_axis=_Axis(1000))

    for store in (dense_store, ragged_store):
        assert _query_shapes.dense_component_query_patterns(
            store, exposure="e", region=("1", 0, 10), random_analyses=["a"]
        ) == {}


def test_dense_only_shapes_query_the_component_and_draw_from_its_axis():
    """Four shapes, the component's own methods, the given Analyses, seeded draws."""
    hybrid = _Hybrid(n_panel=1000)
    patterns = _query_shapes.dense_component_query_patterns(
        hybrid, exposure="e", region=("1", 100, 200), random_analyses=["a", "b"]
    )

    assert set(patterns) == DENSE_ONLY_SHAPES
    drawn = [
        record.alid
        for record in (
            hybrid._dense._variant_axis.by_index(int(v))
            for v in np.random.default_rng(0).choice(1000, size=100, replace=False)
        )
    ]
    assert patterns["random_lookup_100_variants_10_analyses_dense_only"]() == _result()
    assert patterns["random_lookup_10_variants_100_analyses_dense_only"]() == _result()
    assert patterns["regional_one_analysis_dense_only"]() == _result()
    assert patterns["regional_dense_only"]() == _result()

    assert hybrid._dense.calls == [
        # The regional ALIDs come from the Dense axis' own window, not the shared one.
        ("lookup", drawn, ["a", "b"]),
        ("lookup", drawn[:10], ["a", "b"]),
        ("lookup", ["1:0:A:C", "1:1:A:C", "1:2:A:C"], ["e"]),
        ("range_phewas", "1", 100, 200),
    ]
    # Determinism: a second call draws the identical variants, in the same order.
    repeat = _query_shapes.dense_component_query_patterns(
        hybrid, exposure="e", region=("1", 100, 200), random_analyses=["a", "b"]
    )
    repeat["random_lookup_100_variants_10_analyses_dense_only"]()
    assert hybrid._dense.calls[-1] == ("lookup", drawn, ["a", "b"])


def test_dense_only_shapes_use_the_pinned_alids_when_given_them():
    """The generic harness resolves the draw once; a second store must not redraw."""
    hybrid = _Hybrid(n_panel=1000)
    patterns = _query_shapes.dense_component_query_patterns(
        hybrid, exposure="e", region=("1", 100, 200), random_analyses=["a"],
        dense_random_alids=["1:7:A:C", "1:8:A:C"],
    )
    patterns["random_lookup_100_variants_10_analyses_dense_only"]()

    assert hybrid._dense.calls == [("lookup", ["1:7:A:C", "1:8:A:C"], ["a"])]


class _ComparisonQuery:
    """A store stand-in for `benchmark_store_comparison._patterns_for_store`."""

    def __init__(self, dense: _Dense | None = None) -> None:
        self._variant_axis = _Axis(1000, in_window=[])
        self._dense = dense
        if dense is not None:
            self._dense_to_shared = np.arange(1000, dtype="int32")
        self.calls: list[tuple[Any, ...]] = []

    def lookup(self, alids: list[str], analyses: list[str]) -> dict[str, np.ndarray]:
        self.calls.append(("lookup", list(alids), list(analyses)))
        return _result()

    def range_phewas(self, *args: Any) -> dict[str, np.ndarray]:
        self.calls.append(("range_phewas", *args))
        return _result()

    def analysis(self, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        return _result()

    def phewas(self, *args: Any) -> dict[str, np.ndarray]:
        return _result()

    def top_hits(self, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        return _result()


_SELECTION = {
    "exposure_analysis_id": "e",
    "phewas_alid": "1:5:A:C",
    "region": {"chrom": "1", "start": 1, "end": 10},
    "random_alids": ["1:2:A:G"],
    "random_analyses": ["a", "b"],
}


def test_a_dense_store_times_the_same_seven_shapes_as_before():
    """A Dense or Ragged store has no Dense component to single out."""
    patterns = benchmark_store_comparison._patterns_for_store(
        _ComparisonQuery(), _SELECTION
    )

    assert set(patterns) == COMMON_SHAPES


def test_a_hybrid_store_adds_the_four_dense_only_shapes_from_the_selection():
    dense = _Dense(1000)
    query = _ComparisonQuery(dense=dense)
    analyses = [f"a{index}" for index in range(12)]
    pinned = [f"1:{index}:A:C" for index in range(100)]
    selection = {
        **_SELECTION,
        "random_analyses": analyses,
        "dense_random_alids": pinned,
    }

    patterns = benchmark_store_comparison._patterns_for_store(query, selection)

    assert set(patterns) == COMMON_SHAPES | DENSE_ONLY_SHAPES
    patterns["random_lookup_100_variants_10_analyses_dense_only"]()
    patterns["random_lookup_10_variants_100_analyses_dense_only"]()
    assert dense.calls == [
        ("lookup", pinned, analyses[:10]),
        ("lookup", pinned[:10], analyses),
    ]
    assert query.calls == []  # the Dense-only shapes did not go through the Hybrid


def test_dense_only_alids_are_empty_for_a_store_without_a_dense_component():
    assert benchmark_store_comparison.dense_only_alids(_ComparisonQuery()) is None


def test_dense_only_alids_resolve_against_the_panel_and_refuse_a_partial_selection():
    hybrid = _Hybrid(n_panel=1000)
    alids = benchmark_store_comparison.dense_only_alids(hybrid)

    assert len(alids) == _query_shapes.RANDOM_AXIS_SIZE
    partial = _Hybrid(n_panel=1000)
    partial._dense._variant_axis = _PartialAxis(1000)
    with pytest.raises(SystemExit, match="partial selection"):
        benchmark_store_comparison.dense_only_alids(partial)
