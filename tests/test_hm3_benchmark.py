"""Contracts for HM3 extraction in the generic benchmark workload (issue #272).

The benchmark measures a default *derived* artifact, so its central contract is
what it does when the artifact is not there: it must report a non-failing
status, never fail the run, and never confuse "not measured" with a zero-row
measurement. The tests here pin all four states -- measured, missing,
not_applicable, unsupported -- and the one failure class that must stay loud:
a subset that is present and structurally valid whose extraction then fails is
*not* silently downgraded to "missing".

The Dense fixture is the rich Observed-Only release the indexed-subset suites
share, so the measured path reaches a subset that spans overflow, missing, EAF
exception and SE exception cells rather than a uniform grid. The Hybrid case
proves discovery resolves the root to the nested Dense Component (issue #271).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import test_indexed_subsets as subsets
import zarr
from test_hybrid_build import HG38_ALID_1, HG38_ALID_3
from test_indexed_subset_query import _SUBSET_VARIANTS, _write_alid_list
from test_query_metadata_reads import _hybrid
from test_ragged_build_ssf import _make_fixture

from benchmarks import _hm3, _query_shapes
from opengwasdb.layouts.dense.indexed_subsets import (
    INDEXED_SUBSETS_GROUP,
    IndexedSubsetError,
    list_indexed_subsets,
)
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.query import query_store

#: The Hybrid fixture's on-panel ALIDs (its Dense Component's reference axis).
_ON_PANEL = [HG38_ALID_1, HG38_ALID_3]

_TIMING_KEYS = frozenset(
    {"first_read_ms", "median_ms", "p95_ms", "result_count", "analysis_id"}
)


class _RaisingQuery:
    """A query handle whose indexed extraction always fails.

    Only ``store.path`` and ``analysis`` are part of the contract
    ``measure_hm3_extract`` uses, so this stands in for a *valid* subset whose
    answer path blows up -- the case that must propagate.
    """

    def __init__(self, path: Path) -> None:
        self.store = SimpleNamespace(path=path)

    def analysis(self, *_args: Any, **_kwargs: Any) -> dict[str, np.ndarray]:
        raise RuntimeError("indexed extraction failed")


class _UnqueriedHandle:
    """Fails if its query path is touched: a non-measured status must not query."""

    def __init__(self, path: Path) -> None:
        self.store = SimpleNamespace(path=path)

    def analysis(self, *_args: Any, **_kwargs: Any) -> dict[str, np.ndarray]:
        raise AssertionError("a non-measured HM3 status must not run a query")


def _dense_group(store: Path, name: str) -> zarr.Group:
    return zarr.open_group(
        str(store / "data.zarr" / INDEXED_SUBSETS_GROUP / name), mode="r+", zarr_format=3
    )


@pytest.fixture(scope="module")
def dense_without_subset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return subsets.build_rich_store(tmp_path_factory.mktemp("hm3_missing") / "rich.opengwasdb")


@pytest.fixture(scope="module")
def dense_with_hm3(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("hm3_measured")
    store = subsets.build_rich_store(tmp / "rich.opengwasdb")
    alids = [subsets.alid(v) for v in _SUBSET_VARIANTS]
    subsets._build(store, "hm3", _write_alid_list(tmp / "hm3.alid.txt", alids))
    return store


@pytest.fixture
def dense_with_hm3_copy(dense_with_hm3: Path, tmp_path: Path) -> Path:
    return shutil.copytree(dense_with_hm3, tmp_path / "store.opengwasdb")


@pytest.fixture(scope="module")
def ragged_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("hm3_ragged")
    manifest, filtered = _make_fixture(tmp)
    store = tmp / "ragged.opengwasdb"
    build_ragged_from_ssf(manifest, filtered, store, store_id="test", release_id="v1")
    build_ragged_top_hit_indexes(store)
    return store


@pytest.fixture(scope="module")
def hybrid_without_subset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("hm3_hybrid_missing")
    return _hybrid(tmp, tmp)


@pytest.fixture(scope="module")
def hybrid_with_hm3(
    hybrid_without_subset: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    # A copy, so building the index here cannot turn the shared "without
    # subset" fixture into a "measured" one for a test that runs later.
    tmp = tmp_path_factory.mktemp("hm3_hybrid_measured")
    store = shutil.copytree(hybrid_without_subset, tmp / "hybrid.opengwasdb")
    subsets._build(store, "hm3", _write_alid_list(tmp / "hm3.alid.txt", _ON_PANEL))
    return store


# ── Measured ────────────────────────────────────────────────────────────────


def test_dense_hm3_is_measured_with_timing_and_result_count(dense_with_hm3: Path) -> None:
    with query_store(dense_with_hm3) as query:
        ordinary = query.analysis("a0")
        record = _hm3.measure_hm3_extract(query, "a0", store_path=dense_with_hm3, reps=3)

    # The fixture must reach the indexed path with real rows for any of this to
    # mean anything.
    assert list_indexed_subsets(dense_with_hm3) == ("hm3",)
    assert len(ordinary["z"]) > len(_SUBSET_VARIANTS), "ordinary result must exceed the subset"
    assert record["status"] == "measured", record
    assert record["result_count"] > 0, record
    assert record["result_count"] < len(ordinary["z"]), (
        "an indexed result must be the subset, not the primary matrix"
    )
    assert _TIMING_KEYS <= set(record)
    assert record["median_ms"] >= 0
    assert record["p95_ms"] >= record["median_ms"]
    assert record["subset_n_variants"] == len(_SUBSET_VARIANTS)
    assert record["subset_resolved_count"] == len(_SUBSET_VARIANTS)


def test_shared_seam_emits_the_same_record(dense_with_hm3: Path) -> None:
    with query_store(dense_with_hm3) as query:
        record = _query_shapes.measure_hm3_extract(
            query, "a0", store_path=dense_with_hm3, reps=1
        )
    assert record["status"] == "measured", record
    assert record["subset"] == "hm3"
    assert record["result_count"] > 0, record


def test_hybrid_root_discovers_its_dense_components_hm3(hybrid_with_hm3: Path) -> None:
    with query_store(hybrid_with_hm3) as query:
        record = _hm3.measure_hm3_extract(query, "trait_a", store_path=hybrid_with_hm3, reps=2)

    assert record["status"] == "measured", record
    assert record["result_count"] > 0, record
    assert record["subset_n_variants"] == len(_ON_PANEL)


# ── Missing / not applicable / unsupported ──────────────────────────────────


def test_dense_without_subset_is_missing_and_never_queries(
    dense_without_subset: Path,
) -> None:
    state = _hm3.classify_hm3(dense_without_subset)
    assert state.status == "missing", state
    assert "hm3" in state.detail

    record = _hm3.measure_hm3_extract(
        _UnqueriedHandle(dense_without_subset), "a0", store_path=dense_without_subset
    )
    assert record["status"] == "missing"
    assert not (_TIMING_KEYS & set(record)), "missing must carry no measurement"


def test_ragged_layout_is_not_applicable(ragged_store: Path) -> None:
    state = _hm3.classify_hm3(ragged_store)
    assert state.status == "not_applicable", state
    assert "Dense-only" in state.detail

    record = _hm3.measure_hm3_extract(
        _UnqueriedHandle(ragged_store), "trait_a", store_path=ragged_store
    )
    assert record["status"] == "not_applicable"
    assert not (_TIMING_KEYS & set(record)), "not applicable must carry no measurement"


def test_hybrid_without_subset_is_missing(hybrid_without_subset: Path) -> None:
    assert _hm3.classify_hm3(hybrid_without_subset).status == "missing"


def test_forward_schema_subset_is_unsupported_not_missing(
    dense_with_hm3_copy: Path,
) -> None:
    _dense_group(dense_with_hm3_copy, "hm3").attrs["indexed_subset_schema"] = 999

    state = _hm3.classify_hm3(dense_with_hm3_copy)
    assert state.status == "unsupported", state
    assert "schema" in state.detail

    record = _hm3.measure_hm3_extract(
        _UnqueriedHandle(dense_with_hm3_copy), "a0", store_path=dense_with_hm3_copy
    )
    assert record["status"] == "unsupported"
    assert not (_TIMING_KEYS & set(record)), "unsupported must carry no measurement"


def test_forward_encoding_subset_is_unsupported_not_missing(
    dense_with_hm3_copy: Path,
) -> None:
    group = _dense_group(dense_with_hm3_copy, "hm3")
    encoding = dict(group.attrs["encoding"])
    encoding["z"] = {"kind": "quantum"}
    group.attrs["encoding"] = encoding

    state = _hm3.classify_hm3(dense_with_hm3_copy)
    assert state.status == "unsupported", state
    assert "encoding" in state.detail


# ── Present-but-broken stays loud ───────────────────────────────────────────


def test_present_but_damaged_subset_raises_rather_than_reading_missing(
    dense_with_hm3_copy: Path,
) -> None:
    group = _dense_group(dense_with_hm3_copy, "hm3")
    del group["se"]
    group.create_array(
        "se",
        shape=(int(group.attrs["n_analyses"]), int(group.attrs["n_subset_variants"])),
        chunks=(1, int(group.attrs["n_subset_variants"])),
        shards=(1, int(group.attrs["n_subset_variants"])),
        dtype="float16",
    )

    with pytest.raises(IndexedSubsetError):
        _hm3.classify_hm3(dense_with_hm3_copy)


def test_valid_present_subset_extraction_error_propagates(dense_with_hm3: Path) -> None:
    # classify_hm3 reads the filesystem and sees a valid subset...
    assert _hm3.classify_hm3(dense_with_hm3).status == "measured"
    # ...so a failure on the extraction path is a real failure, not "missing".
    with pytest.raises(RuntimeError, match="indexed extraction failed"):
        _hm3.measure_hm3_extract(
            _RaisingQuery(dense_with_hm3), "a0", store_path=dense_with_hm3, reps=1
        )
