"""Indexed Variant Subsets over a Hybrid release (issue #271).

A Hybrid release's Indexed Variant Subset belongs to its self-contained Dense
Component, so it is built and stored at
``<root>/dense/data.zarr/indexed_subsets/<name>``.  A Hybrid-root selector is
answered from that index alone: the Dense Component's panel-local Variant
Indices are remapped through ``dense_to_shared.npy``, the Ragged Overflow is
never read or concatenated, and the ordinary Hybrid query is unchanged.  A
requested canonical ALID that belongs only to the shared/overflow axis fails the
build rather than being silently counted absent -- the plausible-wrong-answer
class this project refuses.

The observed-only fixture is the same union/overflow hybrid
``test_query_metadata_reads`` uses; the reference-completed one reuses
``test_hybrid_completion``'s builder so the imputed mask and reference EAF are
exercised exactly as a completed Hybrid release produces them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

# Third-party imports and the shared fixtures this suite reuses from siblings.
import numpy as np
import pytest
import zarr
from test_hybrid_build import HG38_ALID_1, HG38_ALID_2, HG38_ALID_3
from test_hybrid_completion import (
    _residual_hybrid_crossover_source,
    _residual_ld_panel_with_crossover,
)
from test_indexed_subset_query import _assert_results_equal, _ordinary_filtered
from test_query_metadata_reads import _hybrid
from typer.testing import CliRunner

# Project imports.
from opengwasdb.cli.main import app
from opengwasdb.encoding import DenseEafPlane, DenseSePlane, DenseZPlane
from opengwasdb.layouts.dense.indexed_subsets import (
    INDEXED_SUBSETS_GROUP,
    IndexedSubsetError,
    IndexedSubsetExistsError,
    IndexedSubsetLayoutError,
    IndexedSubsetStaleError,
    build_indexed_subset,
    indexed_subset_namespace_path,
    list_indexed_subsets,
    open_indexed_subset,
    remove_indexed_subset,
)
from opengwasdb.layouts.dense.overview import write_overview_html
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.layouts.hybrid.layout import dense_component_path, dense_to_shared_path
from opengwasdb.model.analyses import read_analyses
from opengwasdb.query import query_store
from opengwasdb.validation import validate_store

#: On-panel ALIDs of the ``_hybrid`` fixture (its reference panel).
ON_PANEL = [HG38_ALID_1, HG38_ALID_3]
OFF_PANEL_ALID = HG38_ALID_2
ABSENT_ALID = "1:99999999:A:G"
ASSEMBLY = "GRCh38"
#: On-panel ALIDs for the completed fixture: one observed and the four panel
#: variants trait_b never observed, which completion imputes.
REF_SUBSET = [
    "1:1000:A:G",
    "1:197000:A:G",
    "1:198000:A:G",
    "1:199000:A:G",
    "1:200000:A:G",
]

_RESULT_KEYS = {"variant_index", "analysis_index", "z", "se", "eaf", "association_status"}


def _write_list(path: Path, alids: list[str]) -> Path:
    path.write_text("\n".join(alids) + "\n", encoding="utf-8")
    return path


def _build(store: Path, name: str, variant_list: Path, **kwargs: object) -> object:
    return build_indexed_subset(
        store, name, variant_list, reference_assembly=ASSEMBLY, **kwargs
    )


def _analyse(store: Path, analysis_id: str, **kwargs: object) -> dict[str, np.ndarray]:
    with query_store(store) as query:
        return query.analysis(analysis_id, **kwargs)


def _subset_group(store: Path, name: str = "hm3") -> zarr.Group:
    """Open the nested Dense Component's published group for in-place damage."""
    path = dense_component_path(store) / "data.zarr" / INDEXED_SUBSETS_GROUP / name
    return zarr.open_group(str(path), mode="r+", zarr_format=3)


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for entry in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(entry.relative_to(path)).encode())
        digest.update(entry.read_bytes())
    return digest.hexdigest()


def _damage_subset_entry(store: Path, name: str) -> None:
    """Put a plain non-Zarr directory at a published subset name."""
    group_dir = dense_component_path(store) / "data.zarr" / INDEXED_SUBSETS_GROUP / name
    group_dir.mkdir(parents=True)


def _assert_scoped_to_root(store: Path, exc: IndexedSubsetError) -> None:
    """The failure names the Hybrid root exactly, never the nested Dense path.

    A plain substring check passes for `<root>/dense` because the root is its
    prefix; this asserts the whole `store <root>: ` prefix, so a failure scoped
    to `<root>/dense` (or double-prefixed) fails the test.
    """
    message = str(exc)
    dense = dense_component_path(store)
    assert message.startswith(f"store {store}: "), (
        f"failure must be scoped to the caller-facing Hybrid root {store}: {message}"
    )
    assert not message.startswith(f"store {dense}: "), (
        f"failure must not be scoped to the nested Dense Component: {message}"
    )
    assert not message.startswith(f"store {store}: store "), (
        f"failure must not be double-prefixed: {message}"
    )


def _shared_indices(store: Path, variant_index: np.ndarray) -> np.ndarray:
    """Map a subset's dense-local Variant Indices onto the shared index space."""
    panel = np.load(dense_to_shared_path(store))
    return panel[np.asarray(variant_index, dtype="int64")].astype("int32")


def _interrupt_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the staged writer fail, to exercise interrupted publication."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        "opengwasdb.layouts.dense.indexed_subsets.write_indexed_subset_group", boom
    )


# ── Observed-Only Hybrid fixtures ───────────────────────────────────────────


@pytest.fixture(scope="module")
def observed_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Observed-Only Hybrid release with no index yet."""
    tmp = tmp_path_factory.mktemp("hybrid_indexed")
    return _hybrid(tmp, tmp)


@pytest.fixture(scope="module")
def indexed_root(observed_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The Hybrid root with one ``hm3`` index over its on-panel ALIDs only."""
    store = observed_root
    _build(store, "hm3", _write_list(tmp_path_factory.mktemp("list") / "hm3.txt", ON_PANEL))
    return store


@pytest.fixture
def root_copy(indexed_root: Path, tmp_path: Path) -> Path:
    """A per-test copy so a destructive failure test cannot poison the module."""
    return shutil.copytree(indexed_root, tmp_path / "hybrid.opengwasdb")


# ── Build placement and discovery ───────────────────────────────────────────


def test_hybrid_build_places_the_index_in_the_nested_dense_component(
    observed_root: Path, tmp_path: Path
) -> None:
    store = shutil.copytree(observed_root, tmp_path / "build.opengwasdb")
    _build(store, "hm3", _write_list(tmp_path / "hm3.txt", ON_PANEL))

    nested = dense_component_path(store) / "data.zarr" / INDEXED_SUBSETS_GROUP / "hm3"
    assert nested.is_dir(), "a Hybrid index must live in the nested Dense Component"
    assert not (store / "data.zarr" / INDEXED_SUBSETS_GROUP).exists(), (
        "the shared/overflow data.zarr must not carry an Indexed Variant Subset"
    )
    # Discovery and reading resolve the root without the operator naming /dense.
    assert list_indexed_subsets(store) == ("hm3",)
    subset = open_indexed_subset(store, "hm3")
    assert subset.n_subset_variants == len(ON_PANEL)
    # The index records the identity of the release that physically owns it,
    # which is the nested Dense Component (its store_id is `<root>-dense`).
    dense_manifest_store_id = open_indexed_subset(store, "hm3").source_store_id
    assert dense_manifest_store_id.endswith("-dense")


def test_direct_nested_dense_build_remains_compatible(tmp_path: Path) -> None:
    """Building on `<root>/dense` directly still works and is discoverable by root."""
    store = _hybrid(tmp_path, tmp_path)
    dense = dense_component_path(store)
    _build(dense, "hm3", _write_list(tmp_path / "hm3.txt", ON_PANEL))
    assert list_indexed_subsets(dense) == ("hm3",)
    assert list_indexed_subsets(store) == ("hm3",)
    result = _analyse(store, "trait_a", indexed_subset="hm3")
    assert len(result["z"]) > 0
    # Removal through the nested path is visible through the root too.
    assert remove_indexed_subset(dense, "hm3") is True
    assert list_indexed_subsets(store) == ()


def test_direct_nested_dense_build_treats_overflow_only_as_absent(tmp_path: Path) -> None:
    """A direct Dense target keeps the pre-#271 absent handling; it is not a Hybrid request."""
    store = _hybrid(tmp_path, tmp_path)
    dense = dense_component_path(store)
    result = _build(
        dense, "hm3", _write_list(tmp_path / "hm3.txt", [*ON_PANEL, OFF_PANEL_ALID])
    )
    assert result.resolved_count == len(ON_PANEL)
    assert result.absent_count == 1
    assert (dense / "data.zarr" / INDEXED_SUBSETS_GROUP / "hm3").is_dir()


# ── Overflow-only rejection (the load-bearing #271 rule) ────────────────────


def test_hybrid_build_refuses_an_overflow_only_alid(
    observed_root: Path, tmp_path: Path
) -> None:
    store = shutil.copytree(observed_root, tmp_path / "reject.opengwasdb")
    with pytest.raises(IndexedSubsetError) as excinfo:
        _build(store, "hm3", _write_list(tmp_path / "hm3.txt", [*ON_PANEL, OFF_PANEL_ALID]))
    message = str(excinfo.value)
    assert OFF_PANEL_ALID in message, message
    assert "shared/overflow" in message, message
    assert str(store) in message, (
        "the refusal must name the release the operator passed, not /dense: " + message
    )
    # Nothing was staged or published by the failed build.
    assert list_indexed_subsets(store) == ()
    namespace = indexed_subset_namespace_path(store)
    assert not namespace.is_dir() or list(namespace.iterdir()) == []


def test_hybrid_build_counts_a_truly_absent_alid_but_does_not_fail(
    observed_root: Path, tmp_path: Path
) -> None:
    store = shutil.copytree(observed_root, tmp_path / "absent.opengwasdb")
    result = _build(
        store, "hm3", _write_list(tmp_path / "hm3.txt", [*ON_PANEL, ABSENT_ALID])
    )
    assert result.requested_count == len(ON_PANEL) + 1
    assert result.resolved_count == len(ON_PANEL)
    assert result.absent_count == 1
    assert list_indexed_subsets(store) == ("hm3",)


def test_hybrid_build_list_with_an_overflow_only_alid_writes_nothing(
    observed_root: Path, tmp_path: Path
) -> None:
    """The failure happens before staging, so a published index is never disturbed."""
    store = shutil.copytree(observed_root, tmp_path / "disturb.opengwasdb")
    _build(store, "hm3", _write_list(tmp_path / "good.txt", ON_PANEL))
    before = (dense_component_path(store) / "data.zarr" / INDEXED_SUBSETS_GROUP / "hm3")
    before_snapshot = sorted(p.name for p in before.rglob("*"))

    with pytest.raises(IndexedSubsetError):
        _build(
            store,
            "hm3",
            _write_list(tmp_path / "bad.txt", [*ON_PANEL, OFF_PANEL_ALID]),
            overwrite=True,
        )
    assert sorted(p.name for p in before.rglob("*")) == before_snapshot
    assert list_indexed_subsets(store) == ("hm3",)


def test_ragged_release_is_still_refused_for_build_and_query(tmp_path: Path) -> None:
    from test_ragged_build_ssf import _make_fixture

    from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf

    manifest, filtered = _make_fixture(tmp_path)
    store = tmp_path / "ragged.opengwasdb"
    build_ragged_from_ssf(manifest, filtered, store, store_id="ragged-test", release_id="v1")
    with pytest.raises(IndexedSubsetError, match="Dense"):
        _build(store, "hm3", _write_list(tmp_path / "hm3.txt", ON_PANEL))
    with query_store(store) as query:
        analysis_id = next(iter(query.analyses_table().values()))["analysis_id"]
        with pytest.raises(IndexedSubsetLayoutError, match="Dense-only"):
            query.analysis(analysis_id, indexed_subset="hm3")


# ── Query: remap, no overflow, no fallback ──────────────────────────────────


def test_hybrid_indexed_query_is_the_ordinary_result_remapped_and_filtered(
    indexed_root: Path,
) -> None:
    subset = open_indexed_subset(indexed_root, "hm3")
    shared = _shared_indices(indexed_root, subset.variant_index)
    for analysis_id in ("trait_a", "trait_b"):
        ordinary = _analyse(indexed_root, analysis_id)
        expected = _ordinary_filtered(ordinary, shared)
        assert len(expected["z"]) > 0, "the filtered ordinary result must be non-empty"
        _assert_results_equal(_analyse(indexed_root, analysis_id, indexed_subset="hm3"), expected)
    # trait_a has an overflow row the subset excludes, so the filter is meaningful.
    assert len(_analyse(indexed_root, "trait_a")["z"]) > len(shared)


def test_hybrid_indexed_query_never_includes_the_overflow(indexed_root: Path) -> None:
    with query_store(indexed_root) as query:
        off_shared = next(
            idx for idx, v in query.variants_table().items() if v["alid"] == OFF_PANEL_ALID
        )
    ordinary = _analyse(indexed_root, "trait_a")
    assert off_shared in ordinary["variant_index"], (
        "the fixture must route this ALID to the overflow for this test to mean anything"
    )
    indexed = _analyse(indexed_root, "trait_a", indexed_subset="hm3")
    assert off_shared not in indexed["variant_index"]


def test_hybrid_indexed_query_never_reads_the_primary_planes_or_the_overflow(
    indexed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("the authoritative planes were read; the path fell back")

    monkeypatch.setattr(DenseZPlane, "column", boom)
    monkeypatch.setattr(DenseSePlane, "column", boom)
    monkeypatch.setattr(DenseEafPlane, "read_column", boom)
    from opengwasdb.query.facade import HybridStoreQuery

    monkeypatch.setattr(HybridStoreQuery, "_overflow_for_analysis", boom)
    result = _analyse(indexed_root, "trait_a", indexed_subset="hm3")
    assert len(result["z"]) > 0, "the fixture must return rows for this to prove anything"


def test_hybrid_ordinary_query_is_unchanged_by_the_index(indexed_root: Path) -> None:
    ordinary = _analyse(indexed_root, "trait_a")
    assert len(ordinary["z"]) > 0
    # The overflow still appears on the ordinary path.
    with query_store(indexed_root) as query:
        off_shared = next(
            idx for idx, v in query.variants_table().items() if v["alid"] == OFF_PANEL_ALID
        )
    assert off_shared in ordinary["variant_index"]


def test_hybrid_observed_only_selector_is_equivalent(indexed_root: Path) -> None:
    subset = open_indexed_subset(indexed_root, "hm3")
    shared = _shared_indices(indexed_root, subset.variant_index)
    ordinary = _analyse(indexed_root, "trait_a", observed_only=True)
    expected = _ordinary_filtered(ordinary, shared)
    _assert_results_equal(
        _analyse(indexed_root, "trait_a", observed_only=True, indexed_subset="hm3"), expected
    )


# ── Failure semantics: unknown, stale, corrupt, no fallback ─────────────────


def test_hybrid_unknown_subset_fails_loudly_naming_the_root(indexed_root: Path) -> None:
    with pytest.raises(IndexedSubsetError) as excinfo:
        _analyse(indexed_root, "trait_a", indexed_subset="nope")
    _assert_scoped_to_root(indexed_root, excinfo.value)
    assert "nope" in str(excinfo.value)
    assert not isinstance(excinfo.value, IndexedSubsetStaleError)


def test_hybrid_stale_index_is_refused_while_the_ordinary_path_still_answers(
    root_copy: Path,
) -> None:
    _subset_group(root_copy).attrs["source_release_id"] = "another-release"
    with pytest.raises(IndexedSubsetStaleError) as excinfo:
        _analyse(root_copy, "trait_a", indexed_subset="hm3")
    _assert_scoped_to_root(root_copy, excinfo.value)
    assert "stale" in str(excinfo.value)
    assert len(_analyse(root_copy, "trait_a")["z"]) > 0


def test_hybrid_corrupt_index_is_refused_naming_the_root(root_copy: Path) -> None:
    group = _subset_group(root_copy)
    values = np.asarray(group["variant_index"][:])
    values[-1] = 10_000_000  # sorted, so only the bounds rule can refuse it
    group["variant_index"][:] = values
    with pytest.raises(IndexedSubsetError) as excinfo:
        _analyse(root_copy, "trait_a", indexed_subset="hm3")
    _assert_scoped_to_root(root_copy, excinfo.value)
    assert "out of bounds" in str(excinfo.value)


def test_hybrid_unreadable_index_is_refused_naming_the_root(root_copy: Path) -> None:
    _damage_subset_entry(root_copy, "broken")
    with pytest.raises(IndexedSubsetError) as excinfo:
        _analyse(root_copy, "trait_a", indexed_subset="broken")
    _assert_scoped_to_root(root_copy, excinfo.value)
    assert "not a readable Zarr group" in str(excinfo.value)


@pytest.mark.parametrize("damage", ["unknown", "stale", "corrupt", "unreadable"])
def test_hybrid_invalid_selector_precedes_unknown_analysis_and_names_the_root(
    root_copy: Path, damage: str
) -> None:
    """The selector is resolved and validated before the Analysis ID is looked up.

    A typo'd, stale or corrupt selector must fail even when the Analysis is also
    unknown -- and the message must name the Hybrid root, not `<root>/dense`.
    """
    subset = "hm3"
    if damage == "unknown":
        subset = "nope"
    elif damage == "stale":
        _subset_group(root_copy).attrs["source_release_id"] = "another-release"
    elif damage == "corrupt":
        group = _subset_group(root_copy)
        values = np.asarray(group["variant_index"][:])
        values[-1] = 10_000_000
        group["variant_index"][:] = values
    elif damage == "unreadable":
        _damage_subset_entry(root_copy, "broken")
        subset = "broken"
    with pytest.raises(IndexedSubsetError) as excinfo:
        _analyse(root_copy, "does-not-exist", indexed_subset=subset)
    _assert_scoped_to_root(root_copy, excinfo.value)


def test_direct_nested_dense_failure_names_the_nested_path(tmp_path: Path) -> None:
    """A direct Dense-Component query keeps naming `<root>/dense` (no re-scope)."""
    store = _hybrid(tmp_path, tmp_path)
    dense = dense_component_path(store)
    with query_store(dense) as query:
        analysis_id = next(iter(query.analyses_table().values()))["analysis_id"]
        with pytest.raises(IndexedSubsetError) as excinfo:
            query.analysis(analysis_id, indexed_subset="nope")
    message = str(excinfo.value)
    assert message.startswith(f"store {dense}: "), message
    assert not message.startswith(f"store {store}: "), message


def test_hybrid_unknown_analysis_keeps_the_ordinary_empty_result(indexed_root: Path) -> None:
    result = _analyse(indexed_root, "does-not-exist", indexed_subset="hm3")
    assert result.keys() == _RESULT_KEYS
    assert all(len(values) == 0 for values in result.values())


# ── Reference-Completed Hybrid fixture ──────────────────────────────────────


@pytest.fixture(scope="module")
def completed_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Reference-Completed Hybrid release with a guaranteed imputed cell."""
    tmp = tmp_path_factory.mktemp("hybrid_completed")
    src, crossover_alid, _se, crossover_eaf = _residual_hybrid_crossover_source(tmp)
    ld = _residual_ld_panel_with_crossover(tmp, crossover_alid, crossover_eaf)
    dst = tmp / "completed.opengwasdb"
    result = complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)
    assert result.n_imputed > 0, (
        "the fixture must impute a cell for the index tests to mean anything"
    )
    return dst


@pytest.fixture(scope="module")
def indexed_completed_root(
    completed_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    _build(
        completed_root,
        "hm3",
        _write_list(tmp_path_factory.mktemp("ref_list") / "hm3.txt", REF_SUBSET),
    )
    return completed_root


def test_completed_hybrid_index_carries_the_imputed_mask(indexed_completed_root: Path) -> None:
    subset = open_indexed_subset(indexed_completed_root, "hm3")
    assert subset.has_imputed is True, "the fixture must be Reference-Completed"
    assert subset.encoding.eaf.reference is True, "a completed release carries reference EAF"


def test_completed_hybrid_indexed_query_matches_the_ordinary_result(
    indexed_completed_root: Path,
) -> None:
    subset = open_indexed_subset(indexed_completed_root, "hm3")
    shared = _shared_indices(indexed_completed_root, subset.variant_index)
    for analysis_id in ("trait_a", "trait_b"):
        ordinary = _analyse(indexed_completed_root, analysis_id)
        expected = _ordinary_filtered(ordinary, shared)
        assert len(expected["z"]) > 0, "the filtered ordinary result must be non-empty"
        _assert_results_equal(
            _analyse(indexed_completed_root, analysis_id, indexed_subset="hm3"), expected
        )


def test_completed_hybrid_indexed_query_returns_an_imputed_status(
    indexed_completed_root: Path,
) -> None:
    indexed = _analyse(indexed_completed_root, "trait_b", indexed_subset="hm3")
    assert "imputed" in set(indexed["association_status"]), (
        "the fixture must route an imputed cell through the Hybrid index for this to mean "
        "anything"
    )


def test_completed_hybrid_observed_only_matches_the_ordinary_result(
    indexed_completed_root: Path,
) -> None:
    subset = open_indexed_subset(indexed_completed_root, "hm3")
    shared = _shared_indices(indexed_completed_root, subset.variant_index)
    ordinary = _analyse(indexed_completed_root, "trait_b", observed_only=True)
    expected = _ordinary_filtered(ordinary, shared)
    _assert_results_equal(
        _analyse(indexed_completed_root, "trait_b", observed_only=True, indexed_subset="hm3"),
        expected,
    )
    assert "imputed" not in set(expected["association_status"])


def test_completed_hybrid_index_validates(indexed_completed_root: Path) -> None:
    result = validate_store(indexed_completed_root)
    assert result.ok, result.errors


# ── Lifecycle: overwrite, removal, atomicity ────────────────────────────────


def test_hybrid_overwrite_and_remove_resolve_the_root(indexed_root: Path, tmp_path: Path) -> None:
    store = shutil.copytree(indexed_root, tmp_path / "lifecycle.opengwasdb")
    with pytest.raises(IndexedSubsetExistsError):
        _build(store, "hm3", _write_list(tmp_path / "again.txt", ON_PANEL))
    _build(store, "hm3", _write_list(tmp_path / "again.txt", ON_PANEL), overwrite=True)
    assert list_indexed_subsets(store) == ("hm3",)
    assert remove_indexed_subset(store, "hm3") is True
    assert list_indexed_subsets(store) == ()
    assert remove_indexed_subset(store, "hm3") is False
    # The authoritative Hybrid path is untouched.
    assert len(_analyse(store, "trait_a")["z"]) > 0


def test_hybrid_interrupted_build_leaves_no_published_group_or_temp(
    root_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _interrupt_writer(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _build(root_copy, "newidx", _write_list(tmp_path / "new.txt", ON_PANEL))
    assert "newidx" not in list_indexed_subsets(root_copy)
    namespace = indexed_subset_namespace_path(root_copy)
    assert not [p for p in namespace.iterdir() if p.name.startswith(".")]


def test_hybrid_failed_overwrite_preserves_the_published_group(
    root_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = dense_component_path(root_copy) / "data.zarr" / INDEXED_SUBSETS_GROUP / "hm3"
    before = _tree_digest(published)
    _interrupt_writer(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _build(root_copy, "hm3", _write_list(tmp_path / "again.txt", ON_PANEL), overwrite=True)

    assert _tree_digest(published) == before, "a failed overwrite must change nothing"
    assert not [p for p in published.parent.iterdir() if p.name.startswith(".")]


# ── Validation and overview through the Hybrid root ─────────────────────────


def test_hybrid_validation_accepts_the_nested_index(indexed_root: Path) -> None:
    result = validate_store(indexed_root)
    assert result.ok, result.errors


def test_hybrid_validation_refuses_a_corrupt_nested_index(root_copy: Path) -> None:
    _subset_group(root_copy).attrs["statistic_profile"] = "z_only"
    result = validate_store(root_copy)
    assert not result.ok
    assert any("hm3" in error for error in result.errors), result.errors


def test_hybrid_overview_lists_the_nested_index(indexed_root: Path, tmp_path: Path) -> None:
    store = shutil.copytree(indexed_root, tmp_path / "overview.opengwasdb")
    table = read_analyses(store / "analyses.tsv")
    write_overview_html(store, table)
    html = (store / "overview.html").read_text(encoding="utf-8")
    assert "hm3" in html, "the nested Dense Component's index must appear in the Hybrid overview"


# ── CLI ─────────────────────────────────────────────────────────────────────


def _invoke(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, list(args))
    return result.exit_code, result.output


def _invoke_build(store: Path, alid_list: Path) -> tuple[int, str]:
    return _invoke(
        "build-indexed-subset",
        str(store),
        "hm3",
        "--variant-list",
        str(alid_list),
        "--reference-assembly",
        ASSEMBLY,
    )


def test_cli_build_and_query_indexed_subset_on_a_hybrid_root(tmp_path: Path) -> None:
    store = _hybrid(tmp_path, tmp_path)
    alid_list = _write_list(tmp_path / "hm3.txt", ON_PANEL)
    code, output = _invoke_build(store, alid_list)
    assert code == 0, output
    assert "dense" in output, output
    assert list_indexed_subsets(store) == ("hm3",)

    code, output = _invoke(
        "query-analysis", str(store), "trait_a", "--indexed-subset", "hm3", "--format", "json"
    )
    assert code == 0, output
    rows = json.loads(output)
    assert rows, "the indexed CLI query must return rows"
    ordinary_code, ordinary_output = _invoke(
        "query-analysis", str(store), "trait_a", "--format", "json"
    )
    assert ordinary_code == 0
    assert set(rows[0]) == set(json.loads(ordinary_output)[0])


def test_cli_hybrid_overflow_only_alid_fails_loudly(tmp_path: Path) -> None:
    store = _hybrid(tmp_path, tmp_path)
    alid_list = _write_list(tmp_path / "bad.txt", [*ON_PANEL, OFF_PANEL_ALID])
    code, output = _invoke_build(store, alid_list)
    assert code == 1, output
    assert OFF_PANEL_ALID in output, output
    assert "shared/overflow" in output, output
    assert list_indexed_subsets(store) == ()


def test_cli_help_describes_hybrid_root_support() -> None:
    runner = CliRunner()
    build_help = runner.invoke(app, ["build-indexed-subset", "--help"])
    query_help = runner.invoke(app, ["query-analysis", "--help"])
    assert build_help.exit_code == 0, build_help.output
    assert query_help.exit_code == 0, query_help.output
    assert "Hybrid" in build_help.output
    assert "Hybrid" in query_help.output
