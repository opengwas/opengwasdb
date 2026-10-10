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
from store_reads import chunk_reads, duplicate_chunk_keys
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


def _corrupt_group(store: Path) -> zarr.Group:
    """Open the fixture's ``hm3`` group for in-place damage (failure tests only)."""
    return zarr.open_group(
        str(store / "data.zarr" / INDEXED_SUBSETS_GROUP / "hm3"), mode="r+", zarr_format=3
    )


def _replace_se_with_float16(store: Path) -> None:
    """Damage the `se` plane's dtype, leaving its shape and every other array intact."""
    group = _corrupt_group(store)
    n_analyses = int(group.attrs["n_analyses"])
    n_subset = int(group.attrs["n_subset_variants"])
    del group["se"]
    group.create_array(
        "se",
        shape=(n_analyses, n_subset),
        chunks=(1, n_subset),
        shards=(1, n_subset),
        dtype="float16",
    )


def _push_variant_index_out_of_bounds(store: Path) -> None:
    """Point the last subset slot beyond the fixture's 320-variant Store axis."""
    group = _corrupt_group(store)
    values = np.asarray(group["variant_index"][:])
    values[-1] = 10_000_000  # still sorted, so only the bounds rule can refuse it
    group["variant_index"][:] = values


#: On-disk content for the corrupt-group cases: invalid JSON, a valid JSON
#: document with an unsupported `zarr_format`, and a valid JSON list (a non-object).
_DAMAGE_CONTENT = {
    "corrupt": "{not json",
    "badformat": json.dumps({"zarr_format": 99, "node_type": "group"}),
    "jsonlist": json.dumps([1, 2, 3]),
}


def _damage_subset_entry(store: Path, name: str, damage: str) -> None:
    """Put a plain directory or an unreadable/invalid Zarr group at the name.

    All of these pass ``Path.is_dir()`` and reach the group-open boundary, which
    is the corruption the read seam must normalise rather than let zarr, json or
    a plain ValueError/TypeError leak.
    """
    group_dir = store / "data.zarr" / INDEXED_SUBSETS_GROUP / name
    group_dir.mkdir(parents=True)
    content = _DAMAGE_CONTENT.get(damage)
    if content is not None:
        (group_dir / "zarr.json").write_text(content, encoding="utf-8")


def _assert_refused(store: Path, subset: str, match: str) -> str:
    """The API refuses `subset` with a message naming the Store and the subset."""
    with pytest.raises(IndexedSubsetError, match=match) as excinfo:
        _analyse(store, "a0", indexed_subset=subset)
    message = str(excinfo.value)
    assert str(store) in message, f"the failure must name the Store: {message}"
    assert subset in message, f"the failure must name the subset: {message}"
    return message


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


def test_unknown_subset_raises_naming_store_and_subset(indexed_store: Path) -> None:
    _assert_refused(indexed_store, "nope", "no Indexed Variant Subset 'nope'")


def test_unknown_analysis_keeps_the_ordinary_empty_result(indexed_store: Path) -> None:
    """Only the subset selector fails loudly; an unknown Analysis ID does not."""
    result = _analyse(indexed_store, "does-not-exist", indexed_subset="hm3")
    assert result.keys() == _RESULT_KEYS
    assert all(len(values) == 0 for values in result.values())


def test_incomplete_group_is_refused(indexed_copy: Path) -> None:
    namespace = indexed_copy / "data.zarr" / INDEXED_SUBSETS_GROUP
    zarr.open_group(str(namespace / "half"), mode="w", zarr_format=3)
    _assert_refused(indexed_copy, "half", "half")


def test_staging_group_is_not_published_and_is_refused(indexed_copy: Path) -> None:
    namespace = indexed_copy / "data.zarr" / INDEXED_SUBSETS_GROUP
    (namespace / ".hm3.tmp.abc").mkdir()
    (namespace / ".half.tmp.abc").mkdir()
    # A staging sibling is inert: it shadows nothing and is never listed.
    assert list_indexed_subsets(indexed_copy) == ("hm3",)
    result = _analyse(indexed_copy, "a0", indexed_subset="hm3")
    assert len(result["z"]) > 0, "a staging sibling must not disturb a published subset"
    # A name that exists only as a staging group is unknown, not read.
    _assert_refused(indexed_copy, "half", "no Indexed Variant Subset 'half'")


def test_stale_index_is_refused_while_the_primary_planes_stay_readable(
    indexed_copy: Path,
) -> None:
    _corrupt_group(indexed_copy).attrs["source_release_id"] = "another-release"
    _assert_refused(indexed_copy, "hm3", "stale")
    # The authoritative matrix is untouched and still answers.
    ordinary = _analyse(indexed_copy, "a0")
    assert len(ordinary["z"]) > 0


def test_shape_mismatch_is_refused(indexed_copy: Path) -> None:
    group = _corrupt_group(indexed_copy)
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
    _assert_refused(indexed_copy, "hm3", "shape")


def test_wrong_plane_dtype_is_refused(indexed_copy: Path) -> None:
    """A readable primary store still refuses a plane whose bytes are the wrong kind."""
    _replace_se_with_float16(indexed_copy)
    _assert_refused(indexed_copy, "hm3", "dtype")


def test_out_of_bounds_variant_index_is_refused(indexed_copy: Path) -> None:
    """An index that names a variant the release does not have is not decoded."""
    _push_variant_index_out_of_bounds(indexed_copy)
    _assert_refused(indexed_copy, "hm3", "out of bounds")


@pytest.mark.parametrize(
    ("attr", "value", "match"),
    [
        ("indexed_subset_name", "other", "records its name"),
        ("indexed_subset_schema", 999, "declares schema"),
        ("statistic_profile", "z_only", "statistic_profile"),
        ("order", "variant,analysis", "may be transposed"),
        ("n_analyses", 99, "but this release has"),
    ],
)
def test_invalid_declared_metadata_is_refused(
    indexed_copy: Path, attr: str, value: object, match: str
) -> None:
    """Name/schema/profile/order/counts are read-path contracts, not just validation ones."""
    _corrupt_group(indexed_copy).attrs[attr] = value
    _assert_refused(indexed_copy, "hm3", match)


def test_inconsistent_requested_counts_are_refused(indexed_copy: Path) -> None:
    group = _corrupt_group(indexed_copy)
    group.attrs["requested_count"] = int(group.attrs["requested_count"]) + 1
    _assert_refused(indexed_copy, "hm3", "do not add up")


@pytest.mark.parametrize(
    "encoding",
    [
        5,
        {"version": 3, "z": {}, "se": {}, "eaf": {}},
        {"version": 3, "z": {"kind": "int16_fixed", "scale": 1024}, "se": {"kind": "float16"}},
    ],
)
def test_malformed_encoding_metadata_is_refused(indexed_copy: Path, encoding: object) -> None:
    """Every malformed encoding shape surfaces as an IndexedSubsetError, not a traceback."""
    _corrupt_group(indexed_copy).attrs["encoding"] = encoding
    _assert_refused(indexed_copy, "hm3", "encoding")


def test_malformed_integer_metadata_is_refused(indexed_copy: Path) -> None:
    """A non-integer attribute must name the Store and subset, not raise ValueError."""
    _corrupt_group(indexed_copy).attrs["n_analyses"] = "many"
    _assert_refused(indexed_copy, "hm3", "non-integer n_analyses")


@pytest.mark.parametrize("damage", ["plain", "corrupt", "badformat", "jsonlist"])
def test_unreadable_subset_group_is_refused(indexed_copy: Path, damage: str) -> None:
    """A plain directory or unreadable/invalid metadata is an IndexedSubsetError on both surfaces.

    This is the open/read boundary: zarr's GroupNotFoundError and ContainsArrayError,
    a metadata JSONDecodeError, and the plain ValueError/TypeError a parseable but
    invalid document raises must not escape the facade or the CLI.
    """
    name = f"{damage}idx"
    _damage_subset_entry(indexed_copy, name, damage)
    _assert_refused(indexed_copy, name, "not a readable Zarr group")
    _assert_cli_refused(indexed_copy, name, "not a readable Zarr group")


def test_variant_index_is_read_once_per_indexed_query(
    indexed_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam validates the axis once; decode reuses it rather than re-reading (#265)."""
    with chunk_reads(monkeypatch, ("variant_index",)) as reads:
        result = _analyse(indexed_store, "a0", indexed_subset="hm3")
    assert len(result["z"]) > 0, "the fixture must return rows for this to mean anything"
    assert reads["variant_index"], "the subset's axis must be read for this to mean anything"
    assert duplicate_chunk_keys(reads) == {}, "variant_index was read more than once in one query"


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


def _assert_cli_refused(store: Path, subset: str, match: str) -> str:
    """The CLI exits 1 and names the Store, the subset and the reason."""
    code, output = _invoke(store, "a0", "--indexed-subset", subset, "--format", "json")
    assert code == 1, output
    assert "error:" in output, output
    assert str(store) in output, output
    assert subset in output, output
    assert match in output, output
    return output


def test_cli_unknown_subset_fails_loudly_without_falling_back(indexed_store: Path) -> None:
    _assert_cli_refused(indexed_store, "nope", "no Indexed Variant Subset")


def test_cli_malformed_encoding_fails_loudly_naming_store_and_subset(
    indexed_copy: Path,
) -> None:
    _corrupt_group(indexed_copy).attrs["encoding"] = 5
    _assert_cli_refused(indexed_copy, "hm3", "encoding")


def test_cli_wrong_plane_dtype_fails_loudly(indexed_copy: Path) -> None:
    _replace_se_with_float16(indexed_copy)
    _assert_cli_refused(indexed_copy, "hm3", "dtype")


def test_cli_transposed_order_fails_loudly(indexed_copy: Path) -> None:
    _corrupt_group(indexed_copy).attrs["order"] = "variant,analysis"
    _assert_cli_refused(indexed_copy, "hm3", "transposed")


def test_cli_out_of_bounds_variant_index_fails_loudly(indexed_copy: Path) -> None:
    _push_variant_index_out_of_bounds(indexed_copy)
    _assert_cli_refused(indexed_copy, "hm3", "out of bounds")


def test_cli_unsupported_layout_fails_loudly(tmp_path: Path) -> None:
    manifest, filtered = _make_fixture(tmp_path)
    store = tmp_path / "ragged.opengwasdb"
    build_ragged_from_ssf(manifest, filtered, store, store_id="ragged-test", release_id="v1")
    with query_store(store) as query:
        analysis_id = next(iter(query.analyses_table().values()))["analysis_id"]
    code, output = _invoke(store, analysis_id, "--indexed-subset", "hm3")
    assert code == 1
    assert "error:" in output
    assert str(store) in output
    assert "hm3" in output
    assert "Dense" in output
