"""Selected-Analysis queries through a named Indexed Variant Subset (issue #265).

Issue #264 built the storage artifact; this module pins the query contract ADR
0053 specifies on top of it. A Dense selected-Analysis query may name a subset
and must then answer from it -- same six parallel arrays, same meanings, dtypes,
allele orientation, finite Z/SE filtering and Store ordering as filtering the
ordinary ``analysis()`` result to the subset's Variant Indices. An explicit
request never falls back to the primary matrix: an unknown, incomplete, stale,
invalid or unsupported subset raises instead.

The fixture is the same rich one ``test_indexed_subsets`` uses, because the
equivalence it asserts is only meaningful if the release actually reaches a
missing cell, a Z overflow, an SE exception and an EAF exception, and the
subset carries them. Those are asserted before anything is asserted about the
query result.

The primary-plane instrument (patching ``DenseZPlane.column`` and friends) is
how "never fall back" is tested: if the indexed path touches the authoritative
matrices at all, the patched read raises rather than quietly returning the slow
path's answer.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr  # corruption tests open the indexed group directly
from test_indexed_subsets import (
    EAF_EXCEPTION_VARIANT,
    MISSING_VARIANT,
    N_ANALYSES,
    OVERFLOW_VARIANT,
    SE_EXCEPTION_VARIANT,
    _build,
    alid,
    build_rich_store,
)
from test_query_metadata_reads import _hybrid
from test_ragged_build_ssf import _make_fixture
from typer.testing import CliRunner

from opengwasdb.cli.main import app
from opengwasdb.encoding import DenseEafPlane, DenseSePlane, DenseZPlane
from opengwasdb.layouts.dense.indexed_subsets import (
    INDEXED_SUBSETS_GROUP,
    IndexedSubsetError,
    IndexedSubsetLayoutError,
    IndexedSubsetStaleError,
    list_indexed_subsets,
    open_indexed_subset,
)
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.query import query_store
from opengwasdb.stats import beta_from_z_se, p_value_from_z

#: The subset's variant slots, chosen so the index carries every class the query
#: path has to decode: overflow, missing, EAF exception, SE exception and two
#: ordinary finite cells. Store order is ascending, so the list is not sorted.
_SUBSET_VARIANTS = [
    SE_EXCEPTION_VARIANT,
    OVERFLOW_VARIANT,
    200,
    MISSING_VARIANT,
    EAF_EXCEPTION_VARIANT,
    42,
]
_SUBSET_ALIDS = [alid(v) for v in _SUBSET_VARIANTS]
_RESULT_KEYS = {"variant_index", "analysis_index", "z", "se", "eaf", "association_status"}


def _write_alid_list(path: Path, alids: list[str]) -> Path:
    path.write_text("\n".join(alids) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def indexed_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The rich Observed-Only Dense release with one ``hm3`` subset built."""
    tmp = tmp_path_factory.mktemp("indexed_query")
    store = build_rich_store(tmp / "rich.opengwasdb")
    _build(store, "hm3", _write_alid_list(tmp / "hm3.alid.txt", _SUBSET_ALIDS))
    return store


@pytest.fixture
def indexed_copy(indexed_store: Path, tmp_path: Path) -> Path:
    """A per-test copy so a destructive failure test cannot poison the module."""
    return shutil.copytree(indexed_store, tmp_path / "store.opengwasdb")


def _analyse(store: Path, analysis_id: str, **kwargs: object) -> dict[str, np.ndarray]:
    with query_store(store) as query:
        return query.analysis(analysis_id, **kwargs)


def _ordinary_filtered(ordinary: dict[str, np.ndarray], variants: np.ndarray) -> dict:
    keep = np.isin(ordinary["variant_index"], variants)
    return {key: value[keep] for key, value in ordinary.items()}


def _assert_results_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    assert left.keys() == right.keys() == _RESULT_KEYS
    for key in left:
        a, b = np.asarray(left[key]), np.asarray(right[key])
        assert a.dtype == b.dtype, f"{key}: dtypes differ ({a.dtype} != {b.dtype})"
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), f"{key} differs"


# ── Fixture is meaningful, before anything is asserted about a query ─────────


def test_fixture_reaches_every_indexed_class_through_the_fast_path(
    indexed_store: Path,
) -> None:
    subset = open_indexed_subset(indexed_store, "hm3")
    assert subset.n_subset_variants == len(_SUBSET_VARIANTS), (
        "the index must carry the special cells for the equivalence below to mean anything"
    )
    assert subset.n_analyses == N_ANALYSES
    for variant in (OVERFLOW_VARIANT, MISSING_VARIANT, EAF_EXCEPTION_VARIANT, SE_EXCEPTION_VARIANT):
        assert variant in subset.variant_index.tolist()
    assert subset.n_z_overflow > 0, "fixture must force Z overflow into the index"
    assert subset.n_se_exceptions > 0, "fixture must force an SE exception"
    assert subset.n_eaf_exceptions > 0, "fixture must force an EAF exception"
    # The primary release must still be the source of those classes.
    root = zarr.open_group(str(indexed_store / "data.zarr"), mode="r")
    assert len(root["z_overflow_index"][:]) > 0
    assert len(root["se_exception_index"][:]) > 0
    assert len(root["eaf_exception_index"][:]) > 0


def test_indexed_result_is_non_empty_and_carries_the_special_cells(
    indexed_store: Path,
) -> None:
    result = _analyse(indexed_store, "a0", indexed_subset="hm3")
    assert len(result["z"]) > 0, "the fast path returned nothing; it is not being exercised"
    variants = set(result["variant_index"].tolist())
    assert OVERFLOW_VARIANT in variants, "the Z overflow cell must survive the fast path"
    assert SE_EXCEPTION_VARIANT in variants, "the SE exception cell must survive the fast path"
    assert EAF_EXCEPTION_VARIANT in variants, "the EAF exception cell must survive the fast path"
    assert MISSING_VARIANT not in variants, "a missing cell must be filtered out like ordinary"


# ── Equivalence with filtering ordinary analysis() to the subset ─────────────


@pytest.mark.parametrize("analysis_id", ["a0", "a1", "a2", "a3"])
def test_indexed_equals_ordinary_filtered_for_every_field_and_dtype(
    indexed_store: Path, analysis_id: str
) -> None:
    ordinary = _analyse(indexed_store, analysis_id)
    indexed = _analyse(indexed_store, analysis_id, indexed_subset="hm3")
    subset = open_indexed_subset(indexed_store, "hm3")
    expected = _ordinary_filtered(ordinary, subset.variant_index)
    assert len(expected["z"]) > 0, "the filtered ordinary result must be non-empty"
    _assert_results_equal(indexed, expected)


@pytest.mark.parametrize("analysis_id", ["a0", "a1"])
def test_observed_only_selector_is_equivalent_too(indexed_store: Path, analysis_id: str) -> None:
    ordinary = _analyse(indexed_store, analysis_id, observed_only=True)
    indexed = _analyse(
        indexed_store, analysis_id, observed_only=True, indexed_subset="hm3"
    )
    subset = open_indexed_subset(indexed_store, "hm3")
    _assert_results_equal(indexed, _ordinary_filtered(ordinary, subset.variant_index))


def test_derived_beta_and_p_match_the_primary_result(indexed_store: Path) -> None:
    """Beta and p stay derived from the returned Z/SE, never stored separately."""
    ordinary = _analyse(indexed_store, "a1")
    indexed = _analyse(indexed_store, "a1", indexed_subset="hm3")
    subset = open_indexed_subset(indexed_store, "hm3")
    expected = _ordinary_filtered(ordinary, subset.variant_index)
    assert len(expected["z"]) > 0, (
        "the fixture must return derived values for this to mean anything"
    )
    indexed_beta = [
        beta_from_z_se(float(z), float(se))
        for z, se in zip(indexed["z"], indexed["se"], strict=True)
    ]
    expected_beta = [
        beta_from_z_se(float(z), float(se))
        for z, se in zip(expected["z"], expected["se"], strict=True)
    ]
    assert indexed_beta == expected_beta
    indexed_p = [p_value_from_z(float(z)) for z in indexed["z"]]
    expected_p = [p_value_from_z(float(z)) for z in expected["z"]]
    assert indexed_p == expected_p


def test_reversed_input_order_produces_the_same_result(indexed_copy: Path) -> None:
    """The writer stores Store order, so a caller's list order cannot leak out."""
    reversed_list = _write_alid_list(
        indexed_copy.parent / "reversed.alid.txt", list(reversed(_SUBSET_ALIDS))
    )
    _build(indexed_copy, "rev", reversed_list)
    forward = _analyse(indexed_copy, "a2", indexed_subset="hm3")
    backward = _analyse(indexed_copy, "a2", indexed_subset="rev")
    _assert_results_equal(forward, backward)


# ── Failure semantics: loud, named, and never a fallback ─────────────────────


def test_unknown_subset_raises_naming_the_subset(indexed_store: Path) -> None:
    with pytest.raises(IndexedSubsetError, match="no Indexed Variant Subset 'nope'"):
        _analyse(indexed_store, "a0", indexed_subset="nope")


def test_unknown_analysis_keeps_the_ordinary_empty_result(indexed_store: Path) -> None:
    """Only the subset selector fails loudly; an unknown Analysis ID does not."""
    result = _analyse(indexed_store, "does-not-exist", indexed_subset="hm3")
    assert result.keys() == _RESULT_KEYS
    assert all(len(values) == 0 for values in result.values())


def test_incomplete_group_is_refused(indexed_copy: Path) -> None:
    namespace = indexed_copy / "data.zarr" / INDEXED_SUBSETS_GROUP
    incomplete = namespace / "half"
    zarr.open_group(str(incomplete), mode="w", zarr_format=3)
    with pytest.raises(IndexedSubsetError, match="half"):
        _analyse(indexed_copy, "a0", indexed_subset="half")


def test_staging_group_is_not_published_and_is_refused(indexed_copy: Path) -> None:
    namespace = indexed_copy / "data.zarr" / INDEXED_SUBSETS_GROUP
    (namespace / ".hm3.tmp.abc").mkdir()
    (namespace / ".half.tmp.abc").mkdir()
    # A staging sibling is inert: it shadows nothing and is never listed.
    assert list_indexed_subsets(indexed_copy) == ("hm3",)
    result = _analyse(indexed_copy, "a0", indexed_subset="hm3")
    assert len(result["z"]) > 0, "a staging sibling must not disturb a published subset"
    # A name that exists only as a staging group is unknown, not read.
    with pytest.raises(IndexedSubsetError, match="'half'"):
        _analyse(indexed_copy, "a0", indexed_subset="half")


def test_stale_index_is_refused_while_the_primary_planes_stay_readable(
    indexed_copy: Path,
) -> None:
    group = zarr.open_group(
        str(indexed_copy / "data.zarr" / INDEXED_SUBSETS_GROUP / "hm3"),
        mode="r+",
        zarr_format=3,
    )
    group.attrs["source_release_id"] = "another-release"
    with pytest.raises(IndexedSubsetStaleError, match="stale"):
        _analyse(indexed_copy, "a0", indexed_subset="hm3")
    # The authoritative matrix is untouched and still answers.
    ordinary = _analyse(indexed_copy, "a0")
    assert len(ordinary["z"]) > 0


def test_shape_mismatch_is_refused(indexed_copy: Path) -> None:
    group = zarr.open_group(
        str(indexed_copy / "data.zarr" / INDEXED_SUBSETS_GROUP / "hm3"),
        mode="r+",
        zarr_format=3,
    )
    n_analyses = int(group.attrs["n_analyses"])
    n_subset = int(group.attrs["n_subset_variants"])
    del group["z"]
    group.create_array(
        "z",
        shape=(n_analyses, n_subset - 1),
        chunks=(1, n_subset - 1),
        shards=(1, n_subset - 1),
        dtype="int16",
    )
    with pytest.raises(IndexedSubsetError, match="shape"):
        _analyse(indexed_copy, "a0", indexed_subset="hm3")


def test_malformed_metadata_is_refused(indexed_copy: Path) -> None:
    """A non-integer attribute must name the subset, not escape as a ValueError."""
    group = zarr.open_group(
        str(indexed_copy / "data.zarr" / INDEXED_SUBSETS_GROUP / "hm3"),
        mode="r+",
        zarr_format=3,
    )
    group.attrs["n_analyses"] = "many"
    with pytest.raises(IndexedSubsetError, match="non-integer n_analyses"):
        _analyse(indexed_copy, "a0", indexed_subset="hm3")


def test_indexed_query_never_reads_the_primary_planes(
    indexed_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("the primary statistic planes were read; the path fell back")

    monkeypatch.setattr(DenseZPlane, "column", boom)
    monkeypatch.setattr(DenseSePlane, "column", boom)
    monkeypatch.setattr(DenseEafPlane, "read_column", boom)
    result = _analyse(indexed_store, "a0", indexed_subset="hm3")
    assert len(result["z"]) > 0, "the fixture must return rows for this to prove anything"


def test_ragged_release_refuses_a_subset_selector(tmp_path: Path) -> None:
    manifest, filtered = _make_fixture(tmp_path)
    store = tmp_path / "ragged.opengwasdb"
    build_ragged_from_ssf(manifest, filtered, store, store_id="ragged-test", release_id="v1")
    with query_store(store) as query:
        analysis_id = next(iter(query.analyses_table().values()))["analysis_id"]
        with pytest.raises(IndexedSubsetLayoutError, match="Dense"):
            query.analysis(analysis_id, indexed_subset="hm3")


def test_hybrid_release_refuses_a_subset_selector(tmp_path: Path) -> None:
    store = _hybrid(tmp_path, tmp_path)
    with query_store(store) as query:
        analysis_id = next(iter(query.analyses_table().values()))["analysis_id"]
        with pytest.raises(IndexedSubsetLayoutError, match="Dense"):
            query.analysis(analysis_id, indexed_subset="hm3")


# ── CLI: same schemas, selector passthrough, loud failures ───────────────────


def _invoke(store: Path, *args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, ["query-analysis", str(store), *args])
    return result.exit_code, result.output


def _json_rows(store: Path, *args: str) -> list[dict[str, object]]:
    code, output = _invoke(store, *args, "--format", "json")
    assert code == 0, output
    return json.loads(output)


def _tsv_rows(store: Path, *args: str) -> list[list[str]]:
    code, output = _invoke(store, *args, "--format", "tsv")
    assert code == 0, output
    return list(csv.reader(output.splitlines(), delimiter="\t"))


def test_cli_json_schema_is_unchanged_with_the_selector(indexed_store: Path) -> None:
    without = _json_rows(indexed_store, "a0")
    with_selector = _json_rows(indexed_store, "a0", "--indexed-subset", "hm3")
    assert without and with_selector
    assert set(without[0]) == set(with_selector[0]) == {
        "variant_index",
        "analysis_index",
        "z",
        "se",
    }
    indexed_variants = {row["variant_index"] for row in with_selector}
    assert indexed_variants.issubset({row["variant_index"] for row in without})
    ordinary_by_variant = {row["variant_index"]: row for row in without}
    for row in with_selector:
        assert row == ordinary_by_variant[row["variant_index"]], (
            "the indexed JSON row must be byte-identical to the ordinary one"
        )


def test_cli_tsv_schema_is_unchanged_and_rows_match_the_subset(indexed_store: Path) -> None:
    without = _tsv_rows(indexed_store, "a0")
    with_selector = _tsv_rows(indexed_store, "a0", "--indexed-subset", "hm3")
    assert without[0] == with_selector[0], "the TSV header must not change"
    subset = open_indexed_subset(indexed_store, "hm3")
    ordinary_rows = {row[4]: row for row in without[1:]}  # alid is column 4
    expected = [
        ordinary_rows[alid(v)]
        for v in subset.variant_index.tolist()
        if alid(v) in ordinary_rows
    ]
    assert with_selector[1:] == expected, (
        "TSV rows must be the ordinary rows restricted to the subset, in Store order"
    )
    assert len(with_selector) > 1, "the selector must return rows for this to mean anything"


def test_cli_unknown_subset_fails_loudly_without_falling_back(indexed_store: Path) -> None:
    code, output = _invoke(indexed_store, "a0", "--indexed-subset", "nope")
    assert code == 1
    assert "error:" in output
    assert "nope" in output


def test_cli_unsupported_layout_fails_loudly(tmp_path: Path) -> None:
    manifest, filtered = _make_fixture(tmp_path)
    store = tmp_path / "ragged.opengwasdb"
    build_ragged_from_ssf(manifest, filtered, store, store_id="ragged-test", release_id="v1")
    with query_store(store) as query:
        analysis_id = next(iter(query.analyses_table().values()))["analysis_id"]
    code, output = _invoke(store, analysis_id, "--indexed-subset", "hm3")
    assert code == 1
    assert "error:" in output
    assert "Dense" in output
