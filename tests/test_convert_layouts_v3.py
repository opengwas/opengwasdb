"""The remaining-layout converters (#248): Dense Reference-Completed, Ragged, Hybrid.

#245 converted a Dense Observed-Only release.  This module checks the layouts it
refused -- Dense Reference-Completed, Ragged (Observed-Only and
Reference-Completed) and Hybrid (whose outer release and nested Dense Component
are two Store Releases) -- convert bit-exactly and keep every query answer.

The fixtures come from the builder suites rather than being copied, so the
store each layout is converted from is the same one those suites validate.  The
Hybrid fixture is asserted to span both components before anything is asserted
about its conversion, and every identity check asserts the *source* result is
non-empty before comparing digests, so an empty result cannot pass as an
unchanged one.

Layouts whose arrays the seam cannot name a role for are still refused; the
module checks the unknown-array and unknown-group refusal per new layout, and a
half-converted Hybrid (one manifest moved to 0.2.0, the other left behind).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from legacy_fixtures import relayout_as_0_1_0
from test_dense_completion import SOURCE_HEADER, SOURCE_ROWS
from test_dense_completion import _make_ld_panel as _make_dense_ld_panel
from test_hybrid_build import (
    HG19_POS_1,
    HG19_POS_2,
    HG19_POS_3,
    _make_manifest,
    _make_vcf,
    _panel,
)
from test_ragged_build_ssf import _make_fixture as _make_ragged_ssf_fixture
from test_ragged_completion import _make_besd_fixture
from test_ragged_completion import _make_ld_panel as _make_ragged_ld_panel

from benchmarks.benchmark_store_comparison import assert_identical, result_digests
from opengwasdb.build.observed import build_dense_observed_from_sources
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.complete import complete_ragged_store
from opengwasdb.model.analyses import read_analyses
from opengwasdb.query import query_store
from opengwasdb.store.arrays import (
    ArrayRole,
    create_array,
    create_group,
    open_group,
    role_for_array_path,
    shard_layout,
)
from opengwasdb.store.convert import ConversionError, convert_release, verify_conversion
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store

DENSE_ANALYSIS_CHUNK = 4
DENSE_SHARD = (1000, 8)


# ── fixtures, reused from the builder suites ─────────────────────────────────


@pytest.fixture(scope="session")
def ragged_observed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ragged-observed")
    manifest, filtered = _make_ragged_ssf_fixture(root)
    built = root / "ragged-obs-built.opengwasdb"
    build_ragged_from_ssf(manifest, filtered, built, store_id="ragged-test", release_id="obs-v1")
    return relayout_as_0_1_0(built, root / "ragged-obs.opengwasdb")


@pytest.fixture(scope="session")
def ragged_completed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ragged-completed")
    prefix = _make_besd_fixture(root)
    observed = root / "ragged-obs.opengwasdb"
    build_ragged_from_besd(
        prefix, observed, store_id="ragged-rc", release_id="obs-v1", tissue="Blood"
    )
    panel = _make_ragged_ld_panel(root, "1", 900_000, 1_300_000)
    completed = root / "ragged-rc-built.opengwasdb"
    complete_ragged_store(
        observed,
        completed,
        panel,
        ancestry="EUR",
        cis_window_bp=500_000,
        min_cor=0.0,
        release_id="rc-v1",
    )
    return relayout_as_0_1_0(completed, root / "ragged-rc.opengwasdb")


@pytest.fixture(scope="session")
def dense_completed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dense-completed")
    source = root / "associations.tsv"
    source.write_text(SOURCE_HEADER + "\n" + "\n".join(SOURCE_ROWS) + "\n", encoding="utf-8")
    observed = root / "dense-obs.opengwasdb"
    build_dense_observed_from_sources(
        [source], observed, store_id="dense-rc", release_id="obs-v1", reference_assembly="GRCh38"
    )
    panel = _make_dense_ld_panel(root)
    completed = root / "dense-rc-built.opengwasdb"
    complete_dense_store(
        observed, completed, panel, ancestry="EUR", min_cor=0.0, release_id="rc-v1"
    )
    return relayout_as_0_1_0(completed, root / "dense-rc.opengwasdb")


@pytest.fixture(scope="session")
def small_dense_observed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Dense store with 1,005 variants and 9 Analyses.

    Nine Analyses is deliberate: `--dense-analysis-chunk 64` clips to 9, and the
    default Dense shard's 1,024 Analyses is then not a whole multiple of the
    actual inner chunk.  That is the geometry that broke the first OGS-00004
    conversion.
    """
    root = tmp_path_factory.mktemp("small-dense")
    lines = [SOURCE_HEADER]
    for a in range(9):
        for v in range(1005):
            lines.append(
                f"a{a + 1:03d}\tp{a + 1:03d}\tTrait {a + 1}\tTrait {a + 1} primary\t1\t"
                f"{100_000 + v * 137}\tA\tG\t{1.0 + 0.01 * (v % 11):.6f}\t0.1\trs{v}\tsd"
            )
    source = root / "associations.tsv"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    store = root / "small-dense-built.opengwasdb"
    build_dense_observed_from_sources(
        [source], store, store_id="small-dense", release_id="v1", reference_assembly="GRCh37"
    )
    return relayout_as_0_1_0(store, root / "small-dense.opengwasdb")


@pytest.fixture(scope="session")
def hybrid_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("hybrid")
    vcf1 = _make_vcf(
        root,
        "trait_a",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t2.0:0.5:0.2\n",
            f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE:AF\t1.5:0.3:0.3\n",
            f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t0.6:0.2:0.4\n",
        ],
    )
    vcf2 = _make_vcf(
        root,
        "trait_b",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t6.0:0.5:0.25\n",
            f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t1.2:0.3:0.45\n",
        ],
    )
    manifest = _make_manifest(root, [("trait_a", vcf1, "Trait A"), ("trait_b", vcf2, "Trait B")])
    store = root / "hybrid-built.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest,
        store,
        reference_panel=_panel(root),
        store_id="hybrid-test",
        release_id="v1",
        n_workers=1,
    )
    return relayout_as_0_1_0(store, root / "hybrid.opengwasdb")


def _convert(source: Path, destination: Path, *, zarr_rel: str = "data.zarr") -> Path:
    """Convert with a shard that fits the fixture, then return the release.

    A Ragged release has no Dense plane, so the shard parameter is unused there
    and the seam's default is harmless; a Dense one gets the fixture's own shape
    so the shard is trivially a whole multiple of the clipped inner chunk.
    """
    root = open_group(source / zarr_rel, "r")
    dense_shard = (
        (int(root["z"].shape[0]), int(root["z"].shape[1]))
        if "z" in root
        else (1000, 8)
    )
    convert_release(
        source,
        destination,
        dense_analysis_chunk=DENSE_ANALYSIS_CHUNK,
        dense_shard=dense_shard,
        workers=2,
    )
    return destination


# ── identity: every query shape the layout supports, source vs converted ─────


def _queries(store: Path) -> dict[str, Callable[[], dict[str, np.ndarray]]]:
    q = query_store(store)
    analyses = q.analyses_table()
    assert analyses, "fixture has no Analyses; the identity check would be vacuous"
    analysis_id = analyses[min(analyses)]["analysis_id"]
    axis = q._variant_axis
    first = axis.by_index(0)
    assert first is not None, "fixture has no variants; the identity check would be vacuous"
    return {
        "analysis": lambda: q.analysis(analysis_id),
        "phewas": lambda: q.phewas(first.alid),
        "regional": lambda: q.range_phewas("1", 0, 10_000_000),
        "lookup": lambda: q.lookup([first.alid], [analysis_id]),
        # No `analysis_id`: the coarsest tier is non-empty for some fixture but
        # not every Analysis, and an empty per-Analysis result would be no test
        # at all.  The whole-release form still exercises the index read.
        "top_hits": lambda: q.top_hits(threshold=5e-4),
    }


def _shaped_results(
    store: Path,
) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for name, fn in _queries(store).items():
        result = fn()
        assert len(result["z"]) > 0, (
            f"{name} returned no rows on {store}; an identity check over empty results "
            "would be vacuous"
        )
        out[name] = result_digests(result)
    return out


def _assert_identity(source: Path, converted: Path) -> None:
    source_results = _shaped_results(source)
    converted_results = _shaped_results(converted)
    assert set(source_results) == set(converted_results)
    assert_identical("source", source_results, "converted", converted_results)


def test_a_dense_store_with_fewer_analyses_than_the_chunk_converts(
    small_dense_observed: Path, tmp_path: Path
):
    """The default Dense shard must work when the Analysis axis clips the chunk.

    OGS-00004's Dense Component has nine Analyses, so `--dense-analysis-chunk 64`
    clips to 9 and the default shard of 1,024 Analyses is not a whole multiple of
    the inner chunk.  The converter reconciles the hint with the array's actual
    inner chunk (the whole nine-Analysis axis) rather than refusing a valid
    source.
    """
    converted = tmp_path / "small-dense-0.2.0.opengwasdb"
    convert_release(small_dense_observed, converted, dense_analysis_chunk=64)
    plane = open_group(converted / "data.zarr", "r")["z"]
    inner = tuple(int(size) for size in plane.chunks)
    shard = tuple(int(size) for size in plane.shards)
    assert inner == (1000, 9), inner
    for outer, inner_axis in zip(shard, inner, strict=True):
        assert outer % inner_axis == 0, (shard, inner)
    manifest = json.loads((converted / "manifest.json").read_text())
    assert manifest["provenance"]["dense"]["shard_shape"] == list(shard)
    assert manifest["provenance"]["dense"]["chunk_shape"] == list(inner)
    assert validate_store(converted).ok
    assert verify_conversion(small_dense_observed, converted) is None


def test_dense_reference_completed_identity(dense_completed: Path, tmp_path: Path):
    converted = _convert(dense_completed, tmp_path / "dense-rc-0.2.0.opengwasdb")
    verify_conversion(dense_completed, converted)
    _assert_identity(dense_completed, converted)


def test_ragged_observed_identity(ragged_observed: Path, tmp_path: Path):
    converted = _convert(ragged_observed, tmp_path / "ragged-obs-0.2.0.opengwasdb")
    verify_conversion(ragged_observed, converted)
    _assert_identity(ragged_observed, converted)


def test_ragged_reference_completed_identity(ragged_completed: Path, tmp_path: Path):
    converted = _convert(ragged_completed, tmp_path / "ragged-rc-0.2.0.opengwasdb")
    verify_conversion(ragged_completed, converted)
    _assert_identity(ragged_completed, converted)


def test_ragged_reference_completed_keeps_the_completion_arrays(
    ragged_completed: Path, tmp_path: Path
):
    """A Ragged RC conversion must carry the imputed mask and eaf_reference.

    The source is asserted to be a panel axis first: an `eaf_reference` that was
    absent, or all-NaN, would make the equality below true whatever the
    converter did with the completion arrays.
    """
    source_root = open_group(ragged_completed / "data.zarr", "r")
    assert "imputed" in source_root["ragged"]
    assert "eaf_reference" in source_root["ragged"], "fixture is not a reference panel axis"
    source_mask = np.asarray(source_root["ragged"]["imputed"][:])
    source_reference = np.asarray(source_root["ragged"]["eaf_reference"][:])
    assert source_reference.size > 0 and np.isfinite(source_reference).any(), (
        "fixture has no panel frequencies"
    )

    converted = _convert(ragged_completed, tmp_path / "ragged-rc-0.2.0.opengwasdb")
    converted_root = open_group(converted / "data.zarr", "r")
    assert np.array_equal(
        source_mask, np.asarray(converted_root["ragged"]["imputed"][:])
    )
    assert np.array_equal(
        source_reference,
        np.asarray(converted_root["ragged"]["eaf_reference"][:]),
        equal_nan=True,
    )


# ── Hybrid: the fixture spans both components before anything else ───────────


def _assert_hybrid_spans_both_components(store: Path) -> None:
    """Assert the fixture is a real Hybrid before asserting anything about it.

    A "Hybrid" whose overflow is empty would convert and pass every equality
    below while never exercising the Ragged Overflow path, so the two
    components and their arrays are asserted present and non-empty first.
    """
    from opengwasdb.store.open import open_store

    assert (store / "dense" / "manifest.json").is_file(), "nested Dense Component missing"
    outer = open_group(store / "data.zarr", "r")
    nested = open_group(store / "dense" / "data.zarr", "r")
    assert "ragged" in outer and "z" in nested
    assert int(outer["ragged"]["z"].shape[0]) > 0, "Ragged Overflow is empty"
    assert int(nested["z"].shape[0]) > 0, "Dense Component is empty"
    assert open_store(store).manifest.primary_layout.value == "hybrid"


def test_hybrid_fixture_spans_both_components(hybrid_source: Path):
    _assert_hybrid_spans_both_components(hybrid_source)


def test_hybrid_identity(hybrid_source: Path, tmp_path: Path):
    _assert_hybrid_spans_both_components(hybrid_source)
    converted = _convert(
        hybrid_source, tmp_path / "hybrid-0.2.0.opengwasdb", zarr_rel="dense/data.zarr"
    )
    _assert_hybrid_spans_both_components(converted)
    verify_conversion(hybrid_source, converted)
    _assert_identity(hybrid_source, converted)
    assert json.loads((converted / "manifest.json").read_text())["format_version"] == "0.2.0"
    nested = json.loads((converted / "dense" / "manifest.json").read_text())
    assert nested["format_version"] == "0.2.0"
    assert (converted / "dense" / "data.zarr" / "z" / "zarr.json").is_file()


def test_a_hybrid_conversion_records_the_component_layout(hybrid_source: Path, tmp_path: Path):
    """The nested component records its layout, closing #245's reported gap."""
    converted = _convert(
        hybrid_source, tmp_path / "hybrid-0.2.0.opengwasdb", zarr_rel="dense/data.zarr"
    )
    nested_manifest = json.loads((converted / "dense" / "manifest.json").read_text())
    block = nested_manifest["provenance"]["dense"]
    nested_root = open_group(converted / "dense" / "data.zarr", "r")
    plane = nested_root["z"]
    assert block["chunk_shape"] == list(int(size) for size in plane.chunks)
    assert block["shard_shape"] == list(int(size) for size in plane.shards)
    assert block["zarr_format"] == 3
    outer_manifest = json.loads((converted / "manifest.json").read_text())
    hybrid = outer_manifest["provenance"]["hybrid"]
    assert hybrid["chunk_shape"] == block["chunk_shape"]
    assert hybrid["shard_shape"] == block["shard_shape"]
    assert validate_store(converted).ok


def test_a_nested_component_with_a_disagreeing_layout_is_invalid(
    hybrid_source: Path, tmp_path: Path
):
    """A nested manifest describing another shape is a silent failure class."""
    converted = _convert(
        hybrid_source, tmp_path / "hybrid-0.2.0.opengwasdb", zarr_rel="dense/data.zarr"
    )
    path = converted / "dense" / "manifest.json"
    data = json.loads(path.read_text())
    assert data["provenance"]["dense"]["chunk_shape"]
    data["provenance"]["dense"]["chunk_shape"] = [1, 1]
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = validate_store(converted)

    assert not result.ok
    assert any("manifest.json provenance.dense" in error for error in result.errors), result.errors


def test_an_unknown_top_hit_leaf_fails_the_conversion(
    ragged_observed: Path, tmp_path: Path
):
    """Only the leaves the format defines are accepted under `top_hits/<tier>`.

    Before #248 review round 1, every leaf became `TOP_HIT_INDEX`, so an
    unrecognised member was copied with a guessed role.
    """
    source = _copy(ragged_observed, tmp_path / "source.opengwasdb")
    _add_unknown_array(open_group(source / "data.zarr", "r+"), "top_hits/p_5e_04")

    with pytest.raises(ConversionError, match="mystery_plane"):
        convert_release(source, tmp_path / "out.opengwasdb")


def test_an_unknown_top_hit_leaf_in_the_nested_component_fails(
    hybrid_source: Path, tmp_path: Path
):
    source = _copy(hybrid_source, tmp_path / "source.opengwasdb")
    _add_unknown_array(
        open_group(source / "dense" / "data.zarr", "r+"), "top_hits/p_5e_04"
    )

    with pytest.raises(ConversionError, match="mystery_plane"):
        convert_release(source, tmp_path / "out.opengwasdb")


def test_an_unknown_rho_leaf_fails_the_conversion(dense_completed: Path, tmp_path: Path):
    """Only `rho`, `n_null` and `variant_index` are accepted under `rho/`."""
    source = _copy(dense_completed, tmp_path / "source.opengwasdb")
    root = open_group(source / "data.zarr", "r+")
    create_array(
        create_group(root, "rho"),
        "mystery_plane",
        ArrayRole.RHO_ARRAY,
        shape=(4,),
        dtype="float32",
        fill_value=0.0,
    )

    with pytest.raises(ConversionError, match="mystery_plane"):
        convert_release(source, tmp_path / "out.opengwasdb")


def test_the_group_leaf_maps_accept_the_defined_members_and_refuse_the_rest():
    """`top_hits/<tier>/<leaf>` and `rho/<leaf>` are explicit allow-lists."""
    assert role_for_array_path("top_hits/p_5e_04/analysis_offsets") is (
        ArrayRole.TOP_HIT_ANALYSIS_OFFSETS
    )
    for leaf in (
        "variant_index",
        "analysis_index",
        "abs_z",
        "z",
        "se",
        "p_value",
        "eaf",
        "imputed",
    ):
        assert role_for_array_path(f"top_hits/p_5e_04/{leaf}") is ArrayRole.TOP_HIT_INDEX
    for leaf in ("rho", "n_null", "variant_index"):
        assert role_for_array_path(f"rho/{leaf}") is ArrayRole.RHO_ARRAY
    # An unknown leaf, and a tier that is not exactly one segment, are refused.
    assert role_for_array_path("top_hits/p_5e_04/mystery_plane") is None
    assert role_for_array_path("top_hits/p_5e_04/extra/leaf") is None
    assert role_for_array_path("rho/mystery_plane") is None


# ── the outer Hybrid recordings must agree with the nested component ─────────


def _stale_outer_manifest_chunk_shape(store: Path) -> None:
    path = store / "manifest.json"
    data = json.loads(path.read_text())
    data["provenance"]["hybrid"]["chunk_shape"] = [1, 1]
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_a_hybrid_with_a_stale_outer_manifest_layout_is_invalid(
    hybrid_source: Path, tmp_path: Path
):
    """The outer `provenance.hybrid` must describe the nested component's arrays."""
    converted = _convert(
        hybrid_source, tmp_path / "hybrid-0.2.0.opengwasdb", zarr_rel="dense/data.zarr"
    )
    assert validate_store(converted).ok
    _stale_outer_manifest_chunk_shape(converted)

    result = validate_store(converted)

    assert not result.ok
    assert any(
        "provenance.hybrid" in error and "dense/data.zarr" in error for error in result.errors
    ), result.errors


def test_a_hybrid_with_a_stale_outer_index_blob_is_invalid(
    hybrid_source: Path, tmp_path: Path
):
    """The outer `index.sqlite` dense blob must describe the nested arrays too."""
    converted = _convert(
        hybrid_source, tmp_path / "hybrid-0.2.0.opengwasdb", zarr_rel="dense/data.zarr"
    )
    connection = open_store(converted).index_connection()
    with connection:
        blob = json.loads(
            connection.execute("SELECT value FROM metadata WHERE key = 'dense'").fetchone()["value"]
        )
        blob["chunk_shape"] = [1, 1]
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('dense', ?)",
            (json.dumps(blob),),
        )
        connection.commit()

    result = validate_store(converted)

    assert not result.ok
    assert any("index.sqlite dense metadata" in error for error in result.errors), result.errors


def _edit_outer_hybrid(store: Path, mutate: Callable[[dict], None]) -> None:
    path = store / "manifest.json"
    data = json.loads(path.read_text())
    mutate(data["provenance"]["hybrid"])
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_a_hybrid_with_a_stale_outer_zarr_format_is_invalid(
    hybrid_source: Path, tmp_path: Path
):
    """The outer recording's `zarr_format` must agree with the nested component."""
    converted = _convert(
        hybrid_source, tmp_path / "hybrid-0.2.0.opengwasdb", zarr_rel="dense/data.zarr"
    )
    assert validate_store(converted).ok
    _edit_outer_hybrid(converted, lambda block: block.__setitem__("zarr_format", 2))

    result = validate_store(converted)

    assert not result.ok
    assert any(
        "provenance.hybrid" in error and "zarr_format" in error for error in result.errors
    ), result.errors


def test_a_hybrid_with_the_outer_compressor_removed_is_invalid(
    hybrid_source: Path, tmp_path: Path
):
    """`provenance.hybrid.compressor` is required by spec §10a, not optional."""
    converted = _convert(
        hybrid_source, tmp_path / "hybrid-0.2.0.opengwasdb", zarr_rel="dense/data.zarr"
    )
    assert validate_store(converted).ok
    _edit_outer_hybrid(converted, lambda block: block.pop("compressor"))

    result = validate_store(converted)

    assert not result.ok
    assert any(
        "provenance.hybrid" in error and "compressor" in error for error in result.errors
    ), result.errors


# ── a 0.2.0 Dense component MUST carry all three recordings ──────────────────


def _drop_manifest_dense_provenance(store: Path) -> None:
    path = store / "manifest.json"
    data = json.loads(path.read_text())
    data["provenance"].pop("dense", None)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _drop_index_dense_blob(store: Path) -> None:
    connection = open_store(store).index_connection()
    with connection:
        connection.execute("DELETE FROM metadata WHERE key = 'dense'")
        connection.commit()


def _set_manifest_dense_zarr_format(store: Path, value: Any) -> None:
    path = store / "manifest.json"
    data = json.loads(path.read_text())
    data["provenance"]["dense"]["zarr_format"] = value
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_a_0_2_0_dense_component_without_its_manifest_recording_is_invalid(
    dense_completed: Path, tmp_path: Path
):
    """`provenance.dense` is required for 0.2.0, not skipped when absent."""
    converted = _convert(dense_completed, tmp_path / "dense-rc-0.2.0.opengwasdb")
    assert validate_store(converted).ok
    _drop_manifest_dense_provenance(converted)

    result = validate_store(converted)

    assert not result.ok
    assert any(
        "provenance.dense" in error and "no chunk_shape" in error for error in result.errors
    ), result.errors


def test_a_0_2_0_dense_component_without_its_index_blob_is_invalid(
    dense_completed: Path, tmp_path: Path
):
    """The `index.sqlite` dense blob is required for 0.2.0, not skipped when absent."""
    converted = _convert(dense_completed, tmp_path / "dense-rc-0.2.0.opengwasdb")
    assert validate_store(converted).ok
    _drop_index_dense_blob(converted)

    result = validate_store(converted)

    assert not result.ok
    assert any(
        "index.sqlite dense metadata" in error and "no dense blob" in error
        for error in result.errors
    ), result.errors


@pytest.mark.parametrize("bad", ["3", 3.5, "bogus", True])
def test_a_bad_zarr_format_value_is_a_validation_error_not_an_exception(
    dense_completed: Path, bad: Any, tmp_path: Path
):
    """Only the integers 2 and 3 are valid; no lossy `int(...)` coercion."""
    converted = _convert(dense_completed, tmp_path / "dense-rc-0.2.0.opengwasdb")
    assert validate_store(converted).ok
    _set_manifest_dense_zarr_format(converted, bad)

    result = validate_store(converted)  # must not raise

    assert not result.ok
    assert any("zarr_format" in error for error in result.errors), result.errors


# ── external completion data survives the conversion ─────────────────────────


def _completion_quality(store: Path) -> list[tuple[Any, ...]]:
    connection = open_store(store).index_connection()
    try:
        return [tuple(row) for row in connection.execute("SELECT * FROM completion_quality")]
    finally:
        connection.close()


def _completion_rollups(store: Path) -> dict[str, list[str]]:
    rows = read_analyses(store / "analyses.tsv").rows
    columns = (
        "completion_median_pearson_r",
        "completion_n_imputed_total",
        "completion_n_missing_total",
        "completed_against",
    )
    return {column: [str(row.get(column, "")) for row in rows] for column in columns}


@pytest.mark.parametrize("fixture", ["dense_completed", "ragged_completed"])
def test_completion_quality_and_rollups_survive_the_conversion(
    fixture: str, request: pytest.FixtureRequest, tmp_path: Path
):
    """External completion data is walked, not only the Zarr arrays.

    `verify_conversion` compares only `data.zarr`; a converter that rewrote a
    valid-looking `completion_quality` value or an `analyses.tsv` rollup would
    pass it.  The source is asserted to carry the data first, so the equality is
    not vacuous.
    """
    source = request.getfixturevalue(fixture)
    source_quality = _completion_quality(source)
    source_rollups = _completion_rollups(source)
    assert source_quality, "fixture has no completion_quality rows"
    assert any(any(value != "" for value in column) for column in source_rollups.values()), (
        "fixture has no completion rollups"
    )

    converted = _convert(source, tmp_path / "converted.opengwasdb")

    assert _completion_quality(converted) == source_quality
    assert _completion_rollups(converted) == source_rollups


# ── a half-converted Hybrid is invalid ───────────────────────────────────────


def _set_format_version(manifest: Path, version: str) -> None:
    data = json.loads(manifest.read_text())
    data["format_version"] = version
    manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_a_hybrid_with_an_unconverted_outer_manifest_is_invalid(
    hybrid_source: Path, tmp_path: Path
):
    converted = _convert(
        hybrid_source, tmp_path / "hybrid-0.2.0.opengwasdb", zarr_rel="dense/data.zarr"
    )
    _set_format_version(converted / "manifest.json", "0.1.0")

    result = validate_store(converted)

    assert not result.ok
    assert any("Zarr v3 metadata" in error for error in result.errors), result.errors


def test_a_hybrid_with_an_unconverted_nested_manifest_is_invalid(
    hybrid_source: Path, tmp_path: Path
):
    converted = _convert(
        hybrid_source, tmp_path / "hybrid-0.2.0.opengwasdb", zarr_rel="dense/data.zarr"
    )
    _set_format_version(converted / "dense" / "manifest.json", "0.1.0")

    result = validate_store(converted)

    assert not result.ok
    assert any(
        "dense/data.zarr" in error and "Zarr v3 metadata" in error for error in result.errors
    ), result.errors


# ── refusals: unknown arrays and groups, per layout ──────────────────────────


def _copy(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _add_unknown_array(root: Any, group: str, name: str = "mystery_plane") -> None:
    target = root if not group else root[group]
    create_array(
        target,
        name,
        ArrayRole.TOP_HIT_INDEX,
        shape=(4,),
        dtype="int32",
        fill_value=0,
    )


@pytest.mark.parametrize(
    "fixture,group",
    [
        ("ragged_observed", "ragged"),
        ("ragged_completed", "ragged"),
    ],
)
def test_an_unknown_ragged_array_fails_the_conversion(
    fixture: str, group: str, request: pytest.FixtureRequest, tmp_path: Path
):
    source = _copy(request.getfixturevalue(fixture), tmp_path / "source.opengwasdb")
    _add_unknown_array(open_group(source / "data.zarr", "r+"), group)

    with pytest.raises(ConversionError, match="mystery_plane"):
        convert_release(source, tmp_path / "out.opengwasdb")


def test_an_unknown_empty_group_fails_a_ragged_conversion(
    ragged_observed: Path, tmp_path: Path
):
    source = _copy(ragged_observed, tmp_path / "source.opengwasdb")
    root = open_group(source / "data.zarr", "r+")
    create_group(root["ragged"], "mystery_empty_group")
    assert "mystery_empty_group" in root["ragged"]

    with pytest.raises(ConversionError, match="mystery_empty_group"):
        convert_release(source, tmp_path / "out.opengwasdb")


def test_an_unknown_array_in_the_nested_hybrid_component_fails(
    hybrid_source: Path, tmp_path: Path
):
    source = _copy(hybrid_source, tmp_path / "source.opengwasdb")
    _add_unknown_array(open_group(source / "dense" / "data.zarr", "r+"), "")

    with pytest.raises(ConversionError, match="mystery_plane"):
        convert_release(source, tmp_path / "out.opengwasdb")


def test_an_unknown_group_in_the_hybrid_overflow_fails(hybrid_source: Path, tmp_path: Path):
    source = _copy(hybrid_source, tmp_path / "source.opengwasdb")
    root = open_group(source / "data.zarr", "r+")
    create_group(root["ragged"], "mystery_empty_group")

    with pytest.raises(ConversionError, match="mystery_empty_group"):
        convert_release(source, tmp_path / "out.opengwasdb")


def test_dense_reference_completed_records_the_effective_chunk(
    dense_completed: Path, tmp_path: Path
):
    """The completed source's recorded chunk is rewritten to the axis written.

    #245 review fixed completion to record the chunk it wrote; a conversion must
    carry the effective recording, so the converted manifest's `chunk_shape` is
    the new inner chunk, and it matches the arrays.
    """
    converted = _convert(dense_completed, tmp_path / "dense-rc-0.2.0.opengwasdb")
    manifest = json.loads((converted / "manifest.json").read_text())
    root = open_group(converted / "data.zarr", "r")
    planes = [name for name in ("z", "se", "eaf", "imputed") if name in root]
    assert {"z", "se", "imputed"} <= set(planes), planes
    for name in planes:
        assert manifest["provenance"]["dense"]["chunk_shape"] == list(
            int(size) for size in root[name].chunks
        ), name
        assert manifest["provenance"]["dense"]["shard_shape"] == list(
            int(size) for size in root[name].shards
        ), name
    assert validate_store(converted).ok


# ── the Ragged and Overflow shard policy, decided here ───────────────────────


def test_the_ragged_sequence_shard_gives_tens_of_mb_files_for_ogs_00011():
    """OGS-00011's overflow sequences are 3,085,080,783 entries at 200,000 chunks."""
    shape = (3_085_080_783,)
    inner = (200_000,)
    shard = shard_layout(ArrayRole.ASSOCIATION_SEQUENCE, shape, inner_chunk=inner)
    assert shard == (50_000_000,)
    files = -(-shape[0] // shard[0])
    assert files == 62
    # The widest sequence dtype is 4 bytes, so the largest file is ~200 MB
    # uncompressed -- the same per-worker ceiling the Dense shard sets.
    assert shard[0] * 4 <= 205_000_000


def test_the_ragged_side_shard_bounds_a_large_overflow_table():
    """OGS-00011's Ragged `eaf_exception_index` is 180,396,687 entries."""
    shape = (180_396_687,)
    inner = (200_000,)
    shard = shard_layout(ArrayRole.RAGGED_EXCEPTION_TABLE, shape, inner_chunk=inner)
    assert shard == (10_000_000,)
    assert -(-shape[0] // shard[0]) == 19
    # An int64 exception index is 80 MB per shard, not one 1.4 GB file.
    assert shard[0] * 8 == 80_000_000
