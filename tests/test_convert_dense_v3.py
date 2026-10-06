"""The Dense 0.1.0 -> 0.2.0 converter and the 0.2.0 validation rules (#245).

The converter's job is a physical one: re-write every array in a Dense
Observed-Only release as Zarr v3 with the sharding codec, holding the same
stored codes.  The tests therefore assert two things a wrong conversion would
still satisfy only by accident -- that every query shape returns identical
results on source and converted, and that the stored values are bit-identical
(so a NaN payload counts) -- and one thing a *plausible* conversion would fail:
that nothing is published when a destination shard disagrees with the source.

`converted_dense_store` is session-scoped: the fixture store is 1005 x 120, so
its converted planes have more than one inner chunk **and** more than one shard
on both axes, which is what makes the assertions below non-vacuous.  The module
asserts that geometry before asserting anything about the conversion.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from legacy_fixtures import relayout_dense_as_0_1_0

from benchmarks import _query_shapes
from benchmarks.benchmark_store_comparison import assert_identical, result_digests
from opengwasdb.build.observed import build_dense_observed_from_sources
from opengwasdb.query import query_store
from opengwasdb.store import open as store_open
from opengwasdb.store.arrays import (
    TOP_HIT_SHARD_CHUNKS,
    ArrayRole,
    compressor,
    create_array,
    create_group,
    inner_chunk_of,
    open_group,
    open_group_for_write,
    shard_layout,
    sharded_compressor,
)
from opengwasdb.store.convert import (
    ConversionError,
    ConversionVerificationError,
    _attrs_differ_only_where_expected,
    _plan_arrays,
    convert_dense_release,
    verify_conversion,
)
from opengwasdb.validation import validate_store

#: The fixture's geometry: more than 1,000 variants so the Dense variant-axis
#: inner chunk (fixed at 1,000) splits, and more than 100 Analyses so the
#: standard random-lookup shapes can draw their 100 from it.
N_VARIANTS = 1005
N_ANALYSES = 120
DENSE_ANALYSIS_CHUNK = 4
DENSE_SHARD = (1000, 8)

SOURCE_HEADER = (
    "analysis_id\tphenotype_id\tphenotype_label\tanalysis_label\tchromosome\tposition"
    "\teffect_allele\tother_allele\tz\tse\trsid\tstored_effect_scale"
)

SOURCE_HEADER_WITH_EAF = (
    "analysis_id\tphenotype_id\tphenotype_label\tanalysis_label\tchromosome\tposition"
    "\teffect_allele\tother_allele\tz\tse\teaf\trsid\tstored_effect_scale"
)


def _write_source(path: Path, *, with_eaf: bool = True) -> Path:
    """A Dense Observed-Only source with the geometry the tests need.

    `with_eaf=False` omits the frequency column, which makes every finite-SE
    cell one with no EAF -- trigger 1 of the SE eligibility gate -- so the
    builder's SE plan is `float16` (ADR 0037 §3, #229).  The NaN-fill fixture
    needs that; the plan is otherwise chunk-shape dependent and cannot be
    assumed.
    """
    rng = np.random.default_rng(7)
    lines = [SOURCE_HEADER_WITH_EAF if with_eaf else SOURCE_HEADER]
    for a in range(N_ANALYSES):
        analysis_id = f"a{a + 1:03d}"
        for v in range(N_VARIANTS):
            position = 100_000 + v * 137
            z = float(rng.normal(0, 1.5))
            if v % 97 == 0:
                z = 5.5 + (a % 3)
            se = 0.05 + 0.001 * (v % 7)
            eaf = 0.05 + 0.9 * ((v * 7 + a) % 100) / 100.0
            columns = [
                analysis_id,
                f"p{a + 1:03d}",
                f"Trait {a + 1}",
                f"Trait {a + 1} primary",
                "1",
                str(position),
                "A",
                "G",
                f"{z:.6f}",
                f"{se:.6f}",
            ]
            if with_eaf:
                columns.append(f"{eaf:.6f}")
            columns.extend([f"rs{v}", "sd"])
            lines.append("\t".join(columns))
    source = path / "associations.tsv"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return source


@pytest.fixture(scope="session")
def dense_conversion(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(source, converted) Dense releases, built once for the session.

    Since #247 the builders write only 0.2.0, so the 0.1.0 source is produced by
    relaying the built release out as Zarr v2 (`tests/legacy_fixtures.py`) -- a
    genuine 0.1.0 store, which is what the converter is defined against.
    """
    root = tmp_path_factory.mktemp("convert-dense")
    source = _write_source(root)
    built = root / "built-0.2.0.opengwasdb"
    build_dense_observed_from_sources(
        [source],
        built,
        store_id="fixture-dense",
        release_id="source-v1",
        reference_assembly="GRCh37",
    )
    source_store = root / "source.opengwasdb"
    relayout_dense_as_0_1_0(built, source_store)
    converted = root / "converted.opengwasdb"
    convert_dense_release(
        source_store,
        converted,
        dense_analysis_chunk=DENSE_ANALYSIS_CHUNK,
        dense_shard=DENSE_SHARD,
        workers=2,
    )
    return source_store, converted


@pytest.fixture(scope="session")
def dense_source(dense_conversion: tuple[Path, Path]) -> Path:
    return dense_conversion[0]


@pytest.fixture(scope="session")
def converted_dense_store(dense_conversion: tuple[Path, Path]) -> Path:
    return dense_conversion[1]


def _copy_release(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


#: Rows each deterministic query shape must return from the fixture.  A shape
#: whose query silently returned nothing would still "match" its converted
#: store; these make the identity check non-vacuous (CONTRIBUTING.md).
MEANINGFUL_ROWS = {
    "bulk": N_VARIANTS,
    "phewas": N_ANALYSES,
    "regional": N_VARIANTS * N_ANALYSES,
    "regional_one_analysis": N_VARIANTS,
    "random_lookup_10_variants_100_analyses": 10 * 100,
    "random_lookup_100_variants_10_analyses": 100 * 10,
}


def _assert_meaningful(shape: str, result: dict[str, Any]) -> None:
    """The source result must have rows, and the expected count where fixed."""
    rows = len(result["z"])
    assert rows > 0, f"{shape} returned no rows; the identity check would be vacuous"
    expected = MEANINGFUL_ROWS.get(shape)
    if expected is not None:
        assert rows == expected, f"{shape} returned {rows} rows, expected {expected}"


@pytest.fixture(scope="session")
def nan_fill_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 0.1.0 Dense release whose `se` is float16 with a **NaN** fill.

    `NaN != NaN`, so an ordinary equality check would reject a faithful
    conversion of a store whose fill is NaN.  The source is built without EAF,
    which makes the SE plan `float16` deterministically (trigger 1 of the
    eligibility gate, #229), then re-laid out as 0.1.0 and its `se` rewritten
    with a NaN fill -- the values unchanged, so the top-hit index stays
    consistent with the plane.
    """
    root = tmp_path_factory.mktemp("convert-dense-nan")
    source = _write_source(root, with_eaf=False)
    built = root / "built-0.2.0.opengwasdb"
    build_dense_observed_from_sources(
        [source],
        built,
        store_id="fixture-dense-nan",
        release_id="source-v1",
        reference_assembly="GRCh37",
    )
    legacy = relayout_dense_as_0_1_0(built, root / "source.opengwasdb")
    group = open_group(legacy / "data.zarr", "r+")
    se = group["se"]
    assert str(se.dtype) == "float16", str(se.dtype)
    values = np.asarray(se[:])
    create_array(
        group,
        "se",
        ArrayRole.DENSE_STATISTIC_PLANE,
        data=values,
        dtype="float16",
        fill_value=np.nan,
        compressor=compressor(),
        hint=tuple(int(size) for size in se.chunks),
        overwrite=True,
    )
    assert str(group["se"].dtype) == "float16"
    assert np.isnan(group["se"].fill_value)
    return legacy


# ── the fixture is meaningful before anything is asserted about it ───────────


def test_the_fixture_has_more_than_one_inner_chunk_and_shard_on_both_axes(
    converted_dense_store: Path,
):
    """Without this, a conversion that wrote one chunk per shard could still
    pass every equality below, because there would be nothing to shard."""
    root = open_group(converted_dense_store / "data.zarr", "r")
    for name in ("z", "se", "eaf"):
        array = root[name]
        assert array.shards is not None, name
        inner = inner_chunk_of(array)
        shard = tuple(int(size) for size in array.shards)
        shape = tuple(int(size) for size in array.shape)
        assert shape[0] > inner[0] and shape[0] > shard[0], (name, shape, inner, shard)
        assert shape[1] > inner[1] and shape[1] > shard[1], (name, shape, inner, shard)
        # The shard is a whole multiple of the inner chunk on each axis.
        for outer, inner_axis in zip(shard, inner, strict=True):
            assert outer % inner_axis == 0


def test_the_fixture_source_is_zarr_v2_and_the_conversion_is_zarr_v3(
    dense_source: Path, converted_dense_store: Path
):
    assert (dense_source / "data.zarr" / "z" / ".zarray").is_file()
    assert not (dense_source / "data.zarr" / "z" / "zarr.json").exists()
    assert (converted_dense_store / "data.zarr" / "z" / "zarr.json").is_file()
    assert not (converted_dense_store / "data.zarr" / "z" / ".zarray").exists()
    assert (
        json.loads((converted_dense_store / "data.zarr" / "zarr.json").read_text())[
            "zarr_format"
        ]
        == 3
    )


# ── the top-hit shard width is a parameter (#246) ────────────────────────────

#: A top-hit tier with more than one shard's worth of hits: 2,000,000 hits is
#: 123 inner chunks of 16,384, so the default 64-chunk shard and the one-chunk
#: shard are observably different.  The fixture store's own tiers are one inner
#: chunk each (a 1005 x 120 grid has far fewer hits), which is why the planning
#: input is built here.
TOP_HIT_TIER_HITS = 2_000_000


def test_shard_layout_takes_a_top_hit_shard_width_override():
    """The seam's top-hit policy honours `top_hit_shard_chunks`; `1` is one chunk."""
    default = shard_layout(
        ArrayRole.TOP_HIT_INDEX, (TOP_HIT_TIER_HITS,), inner_chunk=(16_384,)
    )
    one = shard_layout(
        ArrayRole.TOP_HIT_INDEX,
        (TOP_HIT_TIER_HITS,),
        inner_chunk=(16_384,),
        top_hit_shard_chunks=1,
    )
    assert default == (TOP_HIT_SHARD_CHUNKS * 16_384,)
    assert one == (16_384,)
    with pytest.raises(ValueError, match="at least one inner chunk"):
        shard_layout(
            ArrayRole.TOP_HIT_INDEX,
            (TOP_HIT_TIER_HITS,),
            inner_chunk=(16_384,),
            top_hit_shard_chunks=0,
        )


def test_a_conversion_records_the_requested_top_hit_shard_chunks(
    dense_source: Path, converted_dense_store: Path, tmp_path: Path
):
    """The manifest must say which top-hit width the release was written with.

    The session fixture is the default path, and a fresh conversion asks for 1;
    both must land in `provenance.zarr_v3_conversion`, together with the width
    the seam's policy would use.  A release whose recorded width is missing or
    wrong is the silent failure this pins.
    """
    default_manifest = json.loads(
        (converted_dense_store / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        default_manifest["provenance"]["zarr_v3_conversion"]["top_hit_shard_chunks"]
        == TOP_HIT_SHARD_CHUNKS
    )

    converted = tmp_path / "tops.opengwasdb"
    convert_dense_release(
        dense_source,
        converted,
        dense_analysis_chunk=DENSE_ANALYSIS_CHUNK,
        dense_shard=DENSE_SHARD,
        top_hit_shard_chunks=1,
        workers=2,
    )
    manifest = json.loads((converted / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["zarr_v3_conversion"]["top_hit_shard_chunks"] == 1


def test_the_converter_plans_top_hit_shards_at_the_width_it_is_asked_for(tmp_path: Path):
    """`--top-hit-shard-chunks` reaches every top-hit array and nothing else.

    A real end-to-end conversion cannot show this on a unit-test fixture: a
    store large enough to have a multi-shard top-hit tier is the 33 GB
    OGS-00009, which is what #246 converts and measures.  This drives the
    converter's own planning seam instead, so the parameter's route to
    `shard_layout` is tested without one.
    """
    root = open_group_for_write(tmp_path / "data.zarr", "w", zarr_format=2)
    create_array(
        root,
        "z",
        ArrayRole.DENSE_STATISTIC_PLANE,
        shape=(2000, 8),
        dtype="int16",
        fill_value=-1,
    )
    create_group(root, "top_hits")
    create_group(root["top_hits"], "p_5e_04")
    create_array(
        root["top_hits"]["p_5e_04"],
        "z",
        ArrayRole.TOP_HIT_INDEX,
        shape=(TOP_HIT_TIER_HITS,),
        dtype="float32",
        fill_value=0.0,
    )

    def planned(**kwargs: int) -> dict[str, Any]:
        return {
            plan.path: plan
            for plan in _plan_arrays(
                root, dense_analysis_chunk=4, dense_shard=(1000, 8), **kwargs
            )
        }

    wide = planned()
    narrow = planned(top_hit_shard_chunks=1)
    assert wide["top_hits/p_5e_04/z"].inner_chunk == (16_384,)
    assert wide["top_hits/p_5e_04/z"].shard_shape == (64 * 16_384,)
    assert narrow["top_hits/p_5e_04/z"].shard_shape == (16_384,)
    # The Dense plane's own shard is untouched by the top-hit parameter.
    assert wide["z"].shard_shape == narrow["z"].shard_shape == (1000, 8)


# ── every query shape returns identical results ──────────────────────────────


def _patterns(store: Path) -> dict[str, Any]:
    query = query_store(store)
    analyses = query.analyses_table()
    random_alids, random_analyses = _query_shapes.resolve_axis_selections(
        query._variant_axis,
        analyses,
        int(query._root["z"].shape[0]),
        len(analyses),
    )
    first = query._variant_axis.by_index(0)
    assert first is not None
    return _query_shapes.common_query_patterns(
        query,
        exposure="a001",
        phewas_alid=first.alid,
        region=("1", 0, 200_000_000),
        random_alids=random_alids,
        random_analyses=random_analyses,
    )


def test_every_query_shape_returns_identical_results(
    dense_source: Path, converted_dense_store: Path
):
    source_patterns = _patterns(dense_source)
    converted_patterns = _patterns(converted_dense_store)
    assert set(source_patterns) == set(converted_patterns)
    assert len(source_patterns) == 7
    source_results: dict[str, dict[str, str]] = {}
    for name, pattern in source_patterns.items():
        result = pattern()
        _assert_meaningful(name, result)
        source_results[name] = result_digests(result)
    converted_results = {
        name: result_digests(fn()) for name, fn in converted_patterns.items()
    }
    assert_identical("source", source_results, "converted", converted_results)


# ── the verifier ─────────────────────────────────────────────────────────────


def test_a_float16_se_with_a_nan_fill_converts(
    nan_fill_source: Path, tmp_path: Path
):
    """A NaN fill value must survive conversion; `NaN != NaN` must not reject it."""
    converted = tmp_path / "converted.opengwasdb"
    convert_dense_release(
        nan_fill_source,
        converted,
        dense_analysis_chunk=DENSE_ANALYSIS_CHUNK,
        dense_shard=DENSE_SHARD,
        workers=2,
    )
    verify_conversion(nan_fill_source, converted)
    converted_se = open_group(converted / "data.zarr", "r")["se"]
    assert str(converted_se.dtype) == "float16"
    assert np.isnan(converted_se.fill_value)
    assert validate_store(converted).ok


def test_attribute_key_sets_are_compared_before_values():
    """A source `None` and a destination with no key are not the same thing."""
    assert _attrs_differ_only_where_expected({}, {"a": None}, root=False) is not None
    assert _attrs_differ_only_where_expected({"a": None}, {}, root=False) is not None
    assert _attrs_differ_only_where_expected({"a": None}, {"a": None}, root=False) is None
    # A rewritten root key may legitimately be absent from either side.
    assert _attrs_differ_only_where_expected({}, {"chunk_shape": [1]}, root=True) is None


def test_a_none_valued_attribute_only_on_the_destination_fails_the_verifier(
    dense_source: Path, converted_dense_store: Path, tmp_path: Path
):
    corrupted = _copy_release(converted_dense_store, tmp_path / "attrs.opengwasdb")
    root = open_group(corrupted / "data.zarr", "r+")
    root["top_hits"].attrs["none_marker"] = None

    with pytest.raises(ConversionVerificationError, match="only on the destination"):
        verify_conversion(dense_source, corrupted)


def test_the_verifier_accepts_a_faithful_conversion(
    dense_source: Path, converted_dense_store: Path
):
    verify_conversion(dense_source, converted_dense_store)


def test_the_verifier_catches_a_corrupted_shard(
    dense_source: Path, converted_dense_store: Path, tmp_path: Path
):
    """Corrupt one destination shard's values after conversion, through zarr."""
    corrupted = _copy_release(converted_dense_store, tmp_path / "corrupted.opengwasdb")
    root = open_group(corrupted / "data.zarr", "r+")
    array = root["z"]
    assert tuple(array.shards) == DENSE_SHARD
    # Write inside exactly one shard, so the corruption is local and the verifier
    # must find it by value comparison rather than by a shape difference.
    array[0:DENSE_SHARD[0] // 2, 0 : DENSE_SHARD[1] // 2] = 12345

    with pytest.raises(ConversionVerificationError, match="not bit-identical"):
        verify_conversion(dense_source, corrupted)


# ── refusals ─────────────────────────────────────────────────────────────────


def _fake_release(root: Path, **manifest: object) -> Path:
    """A directory with only a manifest.json, enough for a manifest-level refusal."""
    release = root / "store.opengwasdb"
    release.mkdir(parents=True)
    payload = {
        "format_version": "0.1.0",
        "primary_layout": "dense",
        "completion_state": "observed_only",
        "release_id": "fake",
        "store_id": "fake",
    }
    payload.update(manifest)
    (release / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return release


def test_source_and_destination_may_not_be_the_same_path(dense_source: Path):
    with pytest.raises(ConversionError, match="same path"):
        convert_dense_release(dense_source, dense_source)


def test_an_existing_destination_is_refused(
    dense_source: Path, converted_dense_store: Path
):
    with pytest.raises(ConversionError, match="already exists"):
        convert_dense_release(dense_source, converted_dense_store)


@pytest.mark.parametrize("layout", ["ragged", "hybrid"])
def test_a_non_dense_layout_is_refused_by_name(layout: str, tmp_path: Path):
    source = _fake_release(tmp_path, primary_layout=layout)
    with pytest.raises(ConversionError, match="opengwas/opengwasdb#248"):
        convert_dense_release(source, tmp_path / "out.opengwasdb")


def test_a_reference_completed_source_is_refused_by_name(tmp_path: Path):
    source = _fake_release(tmp_path, completion_state="reference_completed")
    with pytest.raises(ConversionError, match="Reference-Completed"):
        convert_dense_release(source, tmp_path / "out.opengwasdb")


def test_an_already_converted_source_is_refused(tmp_path: Path):
    source = _fake_release(tmp_path, format_version="0.2.0")
    with pytest.raises(ConversionError, match="already converted"):
        convert_dense_release(source, tmp_path / "out.opengwasdb")


def test_a_source_of_another_format_is_refused(tmp_path: Path):
    source = _fake_release(tmp_path, format_version="0.1.5")
    with pytest.raises(ConversionError, match="not a general migration tool"):
        convert_dense_release(source, tmp_path / "out.opengwasdb")


def test_an_unknown_array_fails_the_conversion(
    dense_source: Path, tmp_path: Path
):
    """A path the seam cannot name a role for is never copied with a guess."""
    source = _copy_release(dense_source, tmp_path / "source.opengwasdb")
    root = open_group(source / "data.zarr", "r+")
    create_array(
        root,
        "mystery_plane",
        ArrayRole.TOP_HIT_INDEX,
        shape=(10,),
        dtype="int32",
        fill_value=0,
    )

    with pytest.raises(ConversionError, match="mystery_plane"):
        convert_dense_release(source, tmp_path / "out.opengwasdb")


def test_an_unknown_empty_group_fails_the_conversion(
    dense_source: Path, tmp_path: Path
):
    """A group the format does not define is refused, even when it is empty.

    An empty group has no array for `role_for_array_path` to reject, so without
    the group check it would be recreated in the 0.2.0 tree unnoticed.
    """
    source = _copy_release(dense_source, tmp_path / "source.opengwasdb")
    root = open_group(source / "data.zarr", "r+")
    create_group(root, "mystery_empty_group")
    assert "mystery_empty_group" in root

    with pytest.raises(ConversionError, match="mystery_empty_group"):
        convert_dense_release(source, tmp_path / "out.opengwasdb")


# ── validation: the Zarr format matches format_version ───────────────────────


def _set_format_version(root: Path, version: str) -> None:
    path = root / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["format_version"] = version
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _set_dense_chunk_shape(root: Path, chunk_shape: list[int]) -> None:
    path = root / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["provenance"]["dense"]["chunk_shape"] = chunk_shape
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_a_0_1_0_manifest_over_v3_arrays_is_invalid(
    converted_dense_store: Path, tmp_path: Path
):
    store = _copy_release(converted_dense_store, tmp_path / "mislabelled.opengwasdb")
    _set_format_version(store, "0.1.0")

    result = validate_store(store)

    assert not result.ok
    assert any("Zarr v3 metadata" in error for error in result.errors), result.errors


def test_a_0_2_0_manifest_over_v2_arrays_is_invalid(dense_source: Path, tmp_path: Path):
    store = _copy_release(dense_source, tmp_path / "mislabelled.opengwasdb")
    _set_format_version(store, "0.2.0")

    result = validate_store(store)

    assert not result.ok
    assert any("Zarr v2 metadata" in error for error in result.errors), result.errors


# ── validation: the recorded layout matches the arrays ───────────────────────


def test_a_manifest_chunk_shape_that_disagrees_with_the_arrays_is_invalid(
    converted_dense_store: Path, tmp_path: Path
):
    store = _copy_release(converted_dense_store, tmp_path / "wrong-chunk.opengwasdb")
    _set_dense_chunk_shape(store, [500, DENSE_ANALYSIS_CHUNK])
    result = validate_store(store)

    assert not result.ok
    assert any("manifest.json provenance.dense" in error for error in result.errors), (
        result.errors
    )


def test_index_metadata_that_disagrees_with_the_arrays_is_invalid(
    converted_dense_store: Path, tmp_path: Path
):
    store = _copy_release(converted_dense_store, tmp_path / "wrong-index.opengwasdb")
    connection = store_open.open_store(store).index_connection()
    with connection:
        blob = json.loads(
            connection.execute(
                "SELECT value FROM metadata WHERE key = 'dense'"
            ).fetchone()["value"]
        )
        blob["shard_shape"] = [1000, 16]
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('dense', ?)",
            (json.dumps(blob),),
        )
        connection.commit()

    result = validate_store(store)

    assert not result.ok
    assert any("index.sqlite dense metadata" in error for error in result.errors), (
        result.errors
    )


def test_root_attrs_that_disagree_with_the_arrays_are_invalid(
    converted_dense_store: Path, tmp_path: Path
):
    store = _copy_release(converted_dense_store, tmp_path / "wrong-attrs.opengwasdb")
    root = open_group(store / "data.zarr", "r+")
    root.attrs["chunk_shape"] = [500, DENSE_ANALYSIS_CHUNK]

    result = validate_store(store)

    assert not result.ok
    assert any("data.zarr root attrs" in error for error in result.errors), result.errors


def test_root_attrs_that_name_the_actual_layout_pass(converted_dense_store: Path):
    """The converted store's recordings describe its real arrays."""
    root = open_group(converted_dense_store / "data.zarr", "r")
    plane = root["z"]
    assert root.attrs["chunk_shape"] == list(inner_chunk_of(plane))
    assert root.attrs["shard_shape"] == list(int(size) for size in plane.shards)
    assert validate_store(converted_dense_store).ok


def test_a_non_z_plane_with_a_disagreeing_layout_is_invalid(
    converted_dense_store: Path, tmp_path: Path
):
    """Every present Dense plane is judged, not only `z`.

    `se` is re-written with an inner chunk half the recorded 1000, so the
    manifest describes a layout `se` does not have while `z` still matches.
    """
    store = _copy_release(converted_dense_store, tmp_path / "wrong-se.opengwasdb")
    root = open_group(store / "data.zarr", "r+")
    se = root["se"]
    values = np.asarray(se[:])
    assert inner_chunk_of(se) != (500, DENSE_ANALYSIS_CHUNK)
    create_array(
        root,
        "se",
        ArrayRole.DENSE_STATISTIC_PLANE,
        data=values,
        dtype=str(se.dtype),
        fill_value=se.fill_value,
        compressor=sharded_compressor(),
        inner_chunk=(500, DENSE_ANALYSIS_CHUNK),
        shards=(1000, DENSE_SHARD[1]),
        overwrite=True,
    )

    result = validate_store(store)

    assert not result.ok
    assert any(
        "data.zarr/se" in error and "chunk_shape" in error for error in result.errors
    ), result.errors


# ── validation: the per-variant rule applies to the inner chunk ──────────────


def _rewrite_baseline(root: Any, values: np.ndarray, inner: int, shard: int) -> None:
    """Re-write `eaf_baseline` with the given inner chunk and shard, same values."""
    create_array(
        root,
        "eaf_baseline",
        ArrayRole.PER_VARIANT,
        data=values,
        dtype="float32",
        fill_value=0.0,
        inner_chunk=(inner,),
        shards=(shard,),
        compressor=None,
        overwrite=True,
    )


def test_the_per_variant_rule_judges_the_inner_chunk_not_the_shard(
    converted_dense_store: Path, tmp_path: Path
):
    """A sharded array's `chunks` is the inner chunk and `shards` the file unit.

    The rule must read the inner chunk: a shard larger than the bound is
    allowed (it is only a file), an inner chunk larger than it is not.
    """
    store = _copy_release(converted_dense_store, tmp_path / "coarse.opengwasdb")
    root = open_group(store / "data.zarr", "r+")
    expected = int(inner_chunk_of(root["z"])[0])
    values = np.asarray(root["eaf_baseline"][:])
    length = int(values.shape[0])
    assert expected == min(1000, length)  # the bound is the plane's variant chunk
    assert length > expected  # so an inner chunk of expected * 2 really exceeds it

    # Inner chunk at the bound but a much larger shard: still valid.
    _rewrite_baseline(root, values, expected, expected * 3)
    assert validate_store(store).ok

    # Now an inner chunk above the bound: invalid, and named.
    _rewrite_baseline(root, values, expected * 2, expected * 2)
    result = validate_store(store)

    assert not result.ok
    assert any(
        "eaf_baseline" in error and "inner chunk" in error for error in result.errors
    ), result.errors

    assert not result.ok
    assert any(
        "eaf_baseline" in error and "inner chunk" in error for error in result.errors
    ), result.errors


# ── validation: 0.2.0 is readable but not yet writable ───────────────────────


def test_completion_accepts_a_0_2_0_source(converted_dense_store: Path):
    """Since #247 the current format is writable, so the guard does not fire.

    Completion preserves its source's format *and* writes into its arrays; that
    is only honest while this build writes that format, and from #247 it writes
    0.2.0.  Until #247 the converted release was refused here (ADR 0038 §4);
    now a 0.1.0 source is the one refused, with the conversion tool named
    (`tests/test_format_version.py`).
    """
    version = store_open.open_store(converted_dense_store).manifest.format_version
    assert version == "0.2.0"
    assert store_open.check_writable_format_version(version) == version


# ── the seam's shard policy is the one authority ─────────────────────────────


def test_shard_layout_refuses_a_shard_that_is_not_a_whole_multiple_of_the_inner_chunk():
    with pytest.raises(ValueError, match="not a whole multiple"):
        shard_layout(
            ArrayRole.DENSE_STATISTIC_PLANE,
            (10_000, 200),
            inner_chunk=(1000, 4),
            dense_shard=(1500, 8),
        )


def test_shard_layout_matches_the_seam_policy_for_the_fixture(converted_dense_store: Path):
    root = open_group(converted_dense_store / "data.zarr", "r")
    plane = root["z"]
    expected = shard_layout(
        ArrayRole.DENSE_STATISTIC_PLANE,
        tuple(int(size) for size in plane.shape),
        inner_chunk=inner_chunk_of(plane),
        dense_shard=DENSE_SHARD,
    )
    assert expected == tuple(int(size) for size in plane.shards)


def test_the_converted_arrays_read_through_the_seam(converted_dense_store: Path):
    """zarr 3 opens a v3 sharded store through the seam's read path unchanged."""
    import zarr

    assert zarr.__version__.startswith("3")
    root = open_group(converted_dense_store / "data.zarr", "r")
    assert np.asarray(root["z"][:2, :2]).shape == (2, 2)
