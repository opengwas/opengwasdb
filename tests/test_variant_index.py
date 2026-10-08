"""#252 step 5: the variant-centric `by_variant/` index (ADR 0060).

The index is a `(variant_index, analysis_index)`-ordered duplicate of a Ragged
component's cell-keyed arrays, with `analysis_index` derived from the source's
offsets and the exception/overflow tables re-keyed to the duplicate's ordinals.
These tests pin four things that must not drift:

* the **path/role mapper** learns `ragged/by_variant/...` before a builder
  writes it, or a 0.2.0 conversion refuses the group;
* the **build** reproduces every cell -- values, ordering, the imputed mask and
  the exact exception values -- and the reader answers as the step-3 scan does;
* **validation** rejects a damaged or stale index;
* the **augment command** installs one atomically and refreshes the manifest and
  any consolidated-metadata record with it.

The wrong versions live in the damage tests: each writes a specific defect into
a valid index and asserts `validate` names it, so the digest and the structural
rules are known to have teeth.
"""

from __future__ import annotations

from json import loads
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
from test_ragged_build_besd import _make_besd_fixture
from test_ragged_build_ssf import _make_fixture
from test_ragged_completion import _RESULT_KEYS
from test_ragged_residual_completion import RaggedResidualScenario

from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.by_variant import (
    BY_VARIANT_GROUP,
    ByVariantReader,
    VariantIndexError,
    add_variant_index,
    has_variant_index,
)
from opengwasdb.query import query_store
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store import open_store
from opengwasdb.validation import validate_store

#: The arrays a `by_variant` group always carries beside its offsets.
_CORE_LEAVES = ("analysis_index", "z", "se")


# ── fixtures ────────────────────────────────────────────────────────────────


def _build_ssf_store(
    root: Path, *, name: str = "ragged.opengwasdb", **kwargs: object
) -> Path:
    """Build the two-Analysis SSF fixture into `root`, returning the store path."""
    root.mkdir(parents=True, exist_ok=True)
    manifest, filtered = _make_fixture(root)
    kwargs.setdefault("store_id", "idx")
    kwargs.setdefault("release_id", "v1")
    store = root / name
    build_ragged_from_ssf(manifest, filtered, store, **kwargs)
    return store


@pytest.fixture(scope="module")
def observed_ssf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny observed Ragged store built by the SSF builder, index included."""
    return _build_ssf_store(tmp_path_factory.mktemp("variant_index_ssf"))


@pytest.fixture(scope="module")
def observed_besd(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A BESD-built Ragged store: no EAF array at all (ADR 0036)."""
    root = tmp_path_factory.mktemp("variant_index_besd")
    prefix = _make_besd_fixture(root)
    store = root / "ragged.opengwasdb"
    build_ragged_from_besd(prefix, store, store_id="idx-besd", release_id="v1", tissue="Blood")
    return store


@pytest.fixture(scope="module")
def residual(tmp_path_factory: pytest.TempPathFactory) -> RaggedResidualScenario:
    """A completed fixture with residual-coded SE and imputed cells.

    The one fixture whose index must carry `imputed` and a re-keyed SE
    exception table, so a duplicate that dropped either is caught.
    """
    return RaggedResidualScenario(tmp_path_factory)


# ── the strict path/role mapper (red before it existed) ─────────────────────


def test_path_mapper_learns_the_index_leaves() -> None:
    assert store_arrays.role_for_array_path("ragged/by_variant/offsets") is (
        store_arrays.ArrayRole.RAGGED_PER_VARIANT
    )
    for leaf in _CORE_LEAVES + ("eaf", "imputed"):
        assert store_arrays.role_for_array_path(f"ragged/by_variant/{leaf}") is (
            store_arrays.ArrayRole.ASSOCIATION_SEQUENCE
        )
    for leaf in (
        "z_overflow_index",
        "z_overflow_value",
        "eaf_exception_index",
        "eaf_exception_value",
        "se_exception_index",
        "se_exception_value",
    ):
        assert store_arrays.role_for_array_path(f"ragged/by_variant/{leaf}") is (
            store_arrays.ArrayRole.RAGGED_EXCEPTION_TABLE
        )
    assert store_arrays.role_for_array_path("ragged/by_variant/mystery") is None
    assert store_arrays.role_for_array_path("ragged/by_variant/deeper/leaf") is None


def test_group_mapper_records_the_index_group() -> None:
    assert store_arrays.is_recorded_group_path("ragged/by_variant")
    assert not store_arrays.is_recorded_group_path("ragged/by_variant/deeper")
    assert not store_arrays.is_recorded_group_path("ragged/other")


def test_offsets_hint_is_a_path_fact() -> None:
    assert (
        store_arrays.inner_chunk_hint_for_path("ragged/by_variant/offsets")
        == store_arrays.BY_VARIANT_OFFSETS_CHUNK
    )
    assert store_arrays.inner_chunk_hint_for_path("ragged/eaf_baseline") is None
    # The role's default follows the plane (200,000), which is why the path
    # hint is needed: without it a query reading one variant decompresses
    # 200,000 offsets.
    default = store_arrays.chunk_layout(
        store_arrays.ArrayRole.RAGGED_PER_VARIANT, (3,), component_chunk=200_000
    )
    hinted = store_arrays.chunk_layout(
        store_arrays.ArrayRole.RAGGED_PER_VARIANT,
        (3,),
        hint=store_arrays.inner_chunk_hint_for_path("ragged/by_variant/offsets"),
    )
    assert default == (3,)
    assert hinted == (3,)


# ── the builder writes the layout ADR 0060 decides ─────────────────────────


def test_builder_writes_the_decided_layout(observed_ssf: Path) -> None:
    assert has_variant_index(observed_ssf), "the SSF builder must write the index by default"
    root = open_store(observed_ssf).arrays(mode="r")["ragged"]
    index = root[BY_VARIANT_GROUP]
    n_axis = open_store(observed_ssf).manifest.provenance["n_variants"]
    assert root["offsets"].shape[0] - 1 == 2, "fixture must have two Analyses"
    assert index["offsets"].shape[0] == n_axis + 1
    assert set(index.array_keys()) >= set(_CORE_LEAVES)
    # The offsets array takes the path hint, clipped to its own length...
    assert int(index["offsets"].chunks[0]) == min(store_arrays.BY_VARIANT_OFFSETS_CHUNK, n_axis + 1)
    assert int(index["offsets"].shards[0]) % int(index["offsets"].chunks[0]) == 0
    # ...and a sequence leaf keeps the role's declared inner chunk.
    for leaf in _CORE_LEAVES:
        assert int(index[leaf].chunks[0]) == 200_000
    # The decided shards for a full-size array (ADR 0060): 10 M for the offset
    # axis, 50 M for the sequences.  A tiny fixture clips them to its own
    # extent, which is why the roles are checked on a large synthetic shape.
    assert store_arrays.shard_layout(
        store_arrays.ArrayRole.RAGGED_PER_VARIANT, (200_000_000,), inner_chunk=(1_000,)
    ) == (10_000_000,)
    assert store_arrays.shard_layout(
        store_arrays.ArrayRole.ASSOCIATION_SEQUENCE,
        (200_000_000,),
        inner_chunk=(200_000,),
    ) == (50_000_000,)
    # The variant axis is the store's own for a standalone Ragged component.
    assert int(index["offsets"][-1]) == int(root["offsets"][-1])


def test_eaf_is_absent_on_a_besd_build_like_the_component(observed_besd: Path) -> None:
    root = open_store(observed_besd).arrays(mode="r")["ragged"]
    assert "eaf" not in root
    assert "eaf" not in root[BY_VARIANT_GROUP]
    assert "z" in root[BY_VARIANT_GROUP]


def test_imputed_and_se_exceptions_are_duplicated(residual: RaggedResidualScenario) -> None:
    root = open_store(residual.completed).arrays(mode="r")["ragged"]
    index = root[BY_VARIANT_GROUP]
    assert "imputed" in root and "imputed" in index
    assert int(index["imputed"][:].sum()) > 0, "fixture must carry imputed cells"
    assert "se_exception_index" in root and "se_exception_index" in index
    assert len(index["se_exception_index"][:]) > 0, "fixture must carry SE exceptions"


# ── identity: indexed vs the step-3 scan, and the reader vs the CSR ─────────


def _indexed_and_scanned(store: Path):
    """Two query facades over one store: with the index, and forced to scan."""
    indexed = query_store(store)
    scanned = query_store(store)
    assert scanned._by_variant is not None
    scanned._by_variant = None
    return indexed, scanned


def _canonical(result: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    order = np.lexsort(
        (np.asarray(result["analysis_index"]), np.asarray(result["variant_index"]))
    )
    return {key: np.asarray(result[key])[order] for key in result}


@pytest.mark.parametrize("fixture_name", ["observed_ssf", "observed_besd", "residual"])
def test_variant_shapes_are_identical_indexed_and_scanned(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    fixture = request.getfixturevalue(fixture_name)
    store = fixture.completed if hasattr(fixture, "completed") else fixture
    indexed, scanned = _indexed_and_scanned(store)
    with indexed, scanned:
        table = indexed.variants_table()
        alid = str(next(iter(table.values()))["alid"])
        analysis_ids = [
            str(row["analysis_id"]) for _, row in sorted(indexed.analyses_table().items())
        ]
        calls = {
            "phewas": lambda q: q.phewas(alid),
            "range_phewas": lambda q: q.range_phewas("1", 0, 2_000_000),
            "lookup": lambda q: q.lookup([alid], analysis_ids),
        }
        for name, call in calls.items():
            got, want = call(indexed), call(scanned)
            assert len(got["z"]) == len(want["z"]), f"{name}: row count differs"
            for key in _RESULT_KEYS:
                np.testing.assert_array_equal(
                    _canonical(got)[key], _canonical(want)[key], err_msg=f"{name}:{key}"
                )


def test_the_reader_recovers_each_rows_variant_from_the_offsets(observed_ssf: Path) -> None:
    reader = ByVariantReader(observed_ssf)
    for variant in range(reader.n_axis):
        first, last = reader.rows_for_variant(variant)
        if first == last:
            continue
        decoded = reader.decode(variant, variant, first, last)
        assert np.all(decoded["variant_index"] == variant)
        ordered = np.asarray(decoded["analysis_index"])
        assert np.all(np.diff(ordered) >= 0), "a variant's rows must be Analysis-ascending"


def test_a_range_block_holds_only_the_ranges_variants(observed_ssf: Path) -> None:
    reader = ByVariantReader(observed_ssf)
    low, high = 0, reader.n_axis - 1
    first, last = reader.rows_for_variant_range(low, high)
    decoded = reader.decode(low, high, first, last)
    assert len(decoded["z"]) == last - first
    assert decoded["variant_index"].min() >= low
    assert decoded["variant_index"].max() <= high


# ── validation (red: a damaged index passed silently before) ────────────────


def _copy_and_validate(source: Path, tmp_path: Path) -> Path:
    store = tmp_path / f"{source.name}.copy"
    copytree(source, store)
    return store


def _damage(index: Path, name: str, values: np.ndarray) -> None:
    """Write `values` into a `by_variant/<name>` array, whole, in place."""
    root = store_arrays.open_group_for_write(index, "a")
    root[name][...] = values


def test_validation_accepts_the_built_index(
    observed_ssf: Path, residual: RaggedResidualScenario
) -> None:
    assert validate_store(observed_ssf).ok, validate_store(observed_ssf).errors
    assert validate_store(residual.completed).ok, validate_store(residual.completed).errors


def test_validation_rejects_a_wrong_code(observed_ssf: Path, tmp_path: Path) -> None:
    store = _copy_and_validate(observed_ssf, tmp_path)
    index = store / "data.zarr" / "ragged" / BY_VARIANT_GROUP
    z = np.asarray(store_arrays.open_group(index)["z"][:])
    z[0] = z[0] + 1
    _damage(index, "z", z)
    result = validate_store(store)
    assert not result.ok
    assert any("digest" in error for error in result.errors), result.errors


def test_validation_rejects_a_swapped_analysis(observed_ssf: Path, tmp_path: Path) -> None:
    store = _copy_and_validate(observed_ssf, tmp_path)
    index = store / "data.zarr" / "ragged" / BY_VARIANT_GROUP
    root = store_arrays.open_group(index)
    offsets = np.asarray(root["offsets"][:], dtype=np.int64)
    analysis = np.asarray(root["analysis_index"][:])
    # Flip the two rows of a variant that genuinely holds two Analyses.
    widths = np.diff(offsets)
    variant = int(np.argmax(widths >= 2))
    assert widths[variant] >= 2, "fixture must have a multi-Analysis variant"
    start = int(offsets[variant])
    analysis[start], analysis[start + 1] = analysis[start + 1], analysis[start]
    _damage(index, "analysis_index", analysis.astype("int32"))
    result = validate_store(store)
    assert not result.ok
    assert any(BY_VARIANT_GROUP in error for error in result.errors), result.errors


def test_validation_rejects_a_broken_offsets(observed_ssf: Path, tmp_path: Path) -> None:
    store = _copy_and_validate(observed_ssf, tmp_path)
    index = store / "data.zarr" / "ragged" / BY_VARIANT_GROUP
    offsets = np.asarray(store_arrays.open_group(index)["offsets"][:])
    offsets[0] = 1  # no longer spans [0, N)
    _damage(index, "offsets", offsets)
    result = validate_store(store)
    assert not result.ok
    assert any("offsets" in error for error in result.errors), result.errors


def test_validation_rejects_a_stale_index_after_a_row_changes(
    residual: RaggedResidualScenario, tmp_path: Path
) -> None:
    """A valid index made stale by changing the Analysis-sorted plane fails."""
    store = _copy_and_validate(residual.completed, tmp_path)
    root = store_arrays.open_group_for_write(store / "data.zarr" / "ragged", "a")
    z = np.asarray(root["z"][:])
    changed = z.copy()
    changed[0] = np.int16(changed[0] + 1)
    root["z"][...] = changed
    result = validate_store(store)
    assert not result.ok
    assert any(
        "digest" in error or "counts" in error or BY_VARIANT_GROUP in error
        for error in result.errors
    )


# ── the augment command ─────────────────────────────────────────────────────


def test_augment_adds_the_index_and_records_provenance(tmp_path: Path) -> None:
    store = _build_ssf_store(tmp_path / "aug", store_id="aug", write_variant_index=False)
    assert not has_variant_index(store)
    result = add_variant_index(store)
    assert result.n_rows == int(np.asarray(open_store(store).arrays()["ragged"]["offsets"][-1]))
    assert has_variant_index(store)
    block = open_store(store).manifest.provenance["ragged"]["by_variant"]
    assert block["n_rows"] == result.n_rows
    assert validate_store(store).ok, validate_store(store).errors
    with pytest.raises(VariantIndexError):
        add_variant_index(store)


def test_augment_is_idempotent_through_the_identity_harness(tmp_path: Path) -> None:
    """The committed identity harness returns identical answers on a fresh store."""
    from benchmarks.variant_index_identity import check_store

    store = _build_ssf_store(tmp_path / "identity", store_id="idx")
    shapes = check_store(store)
    assert "phewas" in shapes and shapes["phewas"]["identical"]
    assert any(shape["rows"] > 0 for shape in shapes.values())


def test_scaling_harness_splits_phases_and_matches_counts(tmp_path: Path) -> None:
    """The step-2 harness reports three phases per side with equal counts."""
    from benchmarks.variant_side_scaling import measure

    store = _build_ssf_store(tmp_path / "scaling", store_id="idx")
    measured = measure(store, ["phewas"])
    sides = measured["phewas"]
    assert set(sides) == {"indexed", "scanned"}
    for side in sides.values():
        assert {"elapsed_s", "match_s", "read_s", "gather_s"} <= set(side)
        assert side["result_count"] == sides["indexed"]["result_count"]
    assert sides["indexed"]["match_s"] <= sides["scanned"]["match_s"] + 0.05


def test_completion_rebuilds_the_index(residual: RaggedResidualScenario) -> None:
    """A completed release's index describes the completed planes (ruling Q3).

    The observed source carries an index too; the completion remaps the variant
    axis and adds imputed cells, so carrying the source's index forward would
    be stale -- exactly what these two checks reject.
    """
    source_root = open_store(residual.source).arrays(mode="r")["ragged"]
    completed_root = open_store(residual.completed).arrays(mode="r")["ragged"]
    assert BY_VARIANT_GROUP in source_root and BY_VARIANT_GROUP in completed_root
    source_rows = int(source_root[BY_VARIANT_GROUP]["offsets"][-1])
    completed_rows = int(completed_root[BY_VARIANT_GROUP]["offsets"][-1])
    assert completed_rows == int(completed_root["offsets"][-1])
    assert completed_rows != source_rows, (
        "the completion added rows; a carried-forward index would still hold the "
        "source's count"
    )
    result = validate_store(residual.completed)
    assert result.ok, result.errors


def test_augment_rebuilds_with_force(tmp_path: Path) -> None:
    store = _build_ssf_store(tmp_path / "aug-force", store_id="aug")
    add_variant_index(store, force=True)
    assert validate_store(store).ok, validate_store(store).errors


def test_augment_rolls_back_when_the_install_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _build_ssf_store(tmp_path / "aug-rollback", store_id="aug", write_variant_index=False)
    before = (store / "manifest.json").read_bytes()

    import opengwasdb.layouts.ragged.by_variant as module

    def _boom(_store_path: str | Path, _result: object) -> None:
        raise RuntimeError("simulated crash after the group was installed")

    monkeypatch.setattr(module, "_record_index_in_manifest", _boom)
    with pytest.raises(RuntimeError):
        add_variant_index(store)
    assert not (store / "data.zarr" / "ragged" / BY_VARIANT_GROUP).exists()
    assert not (store / "data.zarr" / "ragged" / f"{BY_VARIANT_GROUP}.building").exists()
    assert (store / "manifest.json").read_bytes() == before


def test_augment_refreshes_consolidated_metadata(tmp_path: Path) -> None:
    import zarr

    store = _build_ssf_store(
        tmp_path / "aug-consolidated", store_id="aug", write_variant_index=False
    )
    data_root = store / "data.zarr"
    zarr.consolidate_metadata(str(data_root), zarr_format=3)
    assert zarr.open_group(str(data_root), mode="r").metadata.consolidated_metadata is not None

    add_variant_index(store)

    assert has_variant_index(store)
    record = loads((data_root / "zarr.json").read_text())
    assert record.get("consolidated_metadata") is not None, "the record must be refreshed"
    # The refreshed record describes the group: a fresh open sees the new array.
    group = zarr.open_group(str(data_root), mode="r")
    assert "by_variant" in group["ragged"]
