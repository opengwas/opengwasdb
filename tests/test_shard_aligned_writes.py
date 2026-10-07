"""No build path writes a partial shard (#247).

A shard is the unit a 0.2.0 release stores as one file (ADR 0057), and a write
that covers part of one turns it into a read-modify-write of the whole shard.
That is correct but loses the throughput the shard exists for, and it is
invisible: the store validates either way, and a small fixture's shard is the
whole array, so nothing else in the suite can show it.

`opengwasdb.store.arrays.require_whole_shard_writes` is the test-time hook.  It
patches every public Zarr method that writes a selection of cells -- sync
`__setitem__`, the five `set_*_selection` methods (which `oindex`, `vindex` and
`array.blocks[...]` delegate to), and async `AsyncArray.setitem` -- and refuses
a write that does not start and end on a shard boundary.  Every multi-shard
array is judged, 1-D Ragged association sequences included (#249); a 1-D array
whose shard already spans it (the Dense SE exception table, preallocated and
filled band by band by the rewrite) is not, and neither is an unsharded array.
`resize` and attribute writes are not region writes and are not covered.
Production pays nothing: the hook is entered only by tests or when
`OPEN_GWASDB_REQUIRE_WHOLE_SHARD_WRITES=1`, which the real-data pilot sets so a
genuinely multi-shard build proves its writers are aligned.

`count_shard_writes` is the counting companion: it records the real bytes and
shard key of every storage write, so a test can assert a shard is written
**exactly once** rather than only that each write starts and ends on a boundary.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest
import zarr

from opengwasdb.encoding.se import _row_block_of
from opengwasdb.layouts.dense.build_vcf import _eaf_row_band, _flush_band, _shard_columns
from opengwasdb.layouts.dense.complete import _completion_band_rows
from opengwasdb.store.arrays import (
    ArrayRole,
    PartialShardWriteError,
    _block_selection_to_elements,
    count_shard_writes,
    create_array,
    open_group,
    open_group_for_write,
    require_whole_shard_write,
    require_whole_shard_writes,
    sharded_compressor,
    write_shard_cells,
)

#: A plane small enough to build in a test but with **two** shards on the
#: Analysis axis and two on the variant axis, so an unaligned selection is
#: observably different from an aligned one.
SHAPE = (20, 16)
INNER = (10, 4)
SHARD = (20, 8)


@pytest.fixture
def plane(tmp_path: Path) -> tuple[object, object]:
    """A v3 Dense plane with a 2x2 grid of shards, and a band buffer for it."""
    root = open_group_for_write(tmp_path / "data.zarr", "w", zarr_format=3)
    array = create_array(
        root,
        "z",
        ArrayRole.DENSE_STATISTIC_PLANE,
        shape=SHAPE,
        dtype="float32",
        fill_value=np.nan,
        compressor=sharded_compressor(),
        inner_chunk=INNER,
        shards=SHARD,
    )
    assert tuple(int(size) for size in array.shards) == SHARD
    assert SHAPE[1] > SHARD[1]  # more than one shard on the Analysis axis
    return array, np.zeros((SHAPE[0], SHARD[1]), dtype="float32")


def test_an_aligned_band_write_passes_the_guard(plane: tuple[object, object]) -> None:
    array, band = plane
    with require_whole_shard_writes():
        _flush_band(array, (0, SHARD[1], band), "test", SHAPE[1], time.monotonic(), SHARD[1])
    assert np.isfinite(np.asarray(array[:, : SHARD[1]])).all()


def test_the_guard_covers_every_write_api(plane: tuple[object, object]) -> None:
    """Every Zarr write route is refused for a sub-shard selection (#247 r1/r3).

    `__setitem__`, `oindex`, `vindex` and `blocks` all delegate to one of the
    `set_*_selection` methods, so patching those covers the sync API, and
    `AsyncArray.setitem` covers the async one.  A whole-shard `blocks` write is
    aligned by construction (`array.blocks` indexes the **shard** grid); a
    sub-shard element selection through any route still raises.
    """
    array, band = plane
    whole = (slice(None), slice(0, SHARD[1]))
    partial = (slice(None), slice(0, INNER[1]))
    one = (np.array([0]), np.array([0]))
    with require_whole_shard_writes():
        array[whole] = band  # aligned, must pass
        asyncio.run(array.async_array.setitem(whole, band))  # aligned async
        # `array.blocks[0, 0]` is exactly one whole shard (the block grid is the
        # shard grid: `(20, 8)`), the minimal aligned block write.
        array.blocks[0, 0] = 1.0
        # A stepped block selection addresses disjoint whole shards, so the guard
        # must not reject it for being non-contiguous.  zarr 3.4's block indexing
        # itself refuses `step != 1`, so the translation is checked directly.
        stepped = _block_selection_to_elements(
            array, (slice(None), slice(None, None, 2))
        )
        require_whole_shard_write(array, stepped)
        for write in (
            lambda: array.__setitem__(partial, band[:, : INNER[1]]),
            lambda: array.set_basic_selection(partial, band[:, : INNER[1]]),
            lambda: array.set_orthogonal_selection(one, np.array([1.0])),
            lambda: array.set_mask_selection(np.zeros(SHAPE, dtype=bool), 1.0),
            lambda: array.set_coordinate_selection(one, np.array([1.0])),
            lambda: array.oindex.__setitem__(one, np.array([1.0])),
            lambda: array.vindex.__setitem__(one, np.array([1.0])),
            # The async route: `AsyncArray.setitem`.
            lambda: asyncio.run(
                array.async_array.setitem((slice(None), slice(0, INNER[1])), 1.0)
            ),
        ):
            with pytest.raises(PartialShardWriteError):
                write()


def test_write_shard_cells_writes_one_band_per_touched_shard(
    plane: tuple[object, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cell patch is grouped by shard, not a per-cell read-modify-write.

    Two cells in the same shard and one in another must produce **two** shard
    writes, not three `vindex` cell writes.
    """
    array, _band = plane
    calls: list[object] = []
    original = zarr.Array.__setitem__

    def counting(self: object, selection: object, value: object) -> None:
        calls.append(selection)
        original(self, selection, value)

    monkeypatch.setattr(zarr.Array, "__setitem__", counting)
    # shard_rows=20, shard_cols=8: rows 0 and 5 share shard (0,0); row 0 col 9 is
    # shard (0,1).
    write_shard_cells(array, np.array([0, 5, 0]), np.array([0, 1, 9]), np.array([1.0, 2.0, 3.0]))
    assert len(calls) == 2, calls
    assert float(array[0, 0]) == 1.0
    assert float(array[5, 1]) == 2.0
    assert float(array[0, 9]) == 3.0
    assert np.isnan(float(array[19, 15]))  # untouched cell keeps the fill


def test_a_misaligned_band_write_fails_loudly(plane: tuple[object, object]) -> None:
    """The deliberately misaligned writer: a band of one inner chunk, not a shard.

    Before #247 this write is exactly what the Dense VCF band writer did (it
    used the inner chunk as the band width), so this is the regression the guard
    exists to catch, not a hypothetical one.
    """
    array, band = plane
    with require_whole_shard_writes(), pytest.raises(PartialShardWriteError, match="whole shard"):
        _flush_band(
            array,
            (0, INNER[1], band[:, : INNER[1]]),
            "test",
            SHAPE[1],
            time.monotonic(),
            INNER[1],
        )


def test_a_write_that_starts_off_a_shard_boundary_fails(plane: tuple[object, object]) -> None:
    array, band = plane
    with require_whole_shard_writes(), pytest.raises(PartialShardWriteError):
        # Starts at column 4, inside the first shard.
        array[:, INNER[1] : SHARD[1]] = band[:, INNER[1] : SHARD[1]]


def test_a_whole_array_write_is_aligned(plane: tuple[object, object]) -> None:
    """The whole-array write `create_array(data=...)` makes must stay allowed."""
    array, _band = plane
    with require_whole_shard_writes():
        array[...] = np.arange(SHAPE[0] * SHAPE[1], dtype="float32").reshape(SHAPE)
    assert np.isfinite(np.asarray(array[:])).all()


def test_a_dense_vcf_build_writes_whole_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The **production** band writer, not `_flush_band` called by hand (#247 r1).

    A 16-variant × 4-Analysis fixture with the decided shard replaced by
    `(8, 4)` has two shards on the variant axis and one on the Analysis axis, so
    the production band width (the shard's 4 Analyses) differs from the inner
    chunk's 2.  With the guard on, reverting `_write_dense_bands` to the inner
    chunk makes the first band write fail, and reverting the row-block writer
    (`_row_block_of`, in the SE narrowing) does the same; this test would pass
    either way without the guard.
    """
    import test_dense_vcf_build as tdv

    import opengwasdb.store.arrays as arrays_module
    from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest

    monkeypatch.setattr(arrays_module, "DENSE_SHARD_SHAPE", (8, 4))
    entries = []
    for a in range(4):
        rows = [
            f"1\t{10_000 + v * 10}\t.\t{'ACGT'[v % 4]}\t{'TGCA'[v % 4]}\t.\tPASS\t.\tES:SE"
            f"\t1.0:0.5\n"
            for v in range(16)
        ]
        entries.append((f"trait_{a}", tdv._make_vcf(tmp_path, f"trait_{a}", rows), f"Trait {a}"))
    manifest = tdv._make_manifest(tmp_path, entries)
    store = tmp_path / "store.opengwasdb"
    with require_whole_shard_writes():
        build_dense_from_vcf_manifest(
            manifest,
            store,
            store_id="s",
            release_id="r",
            n_workers=1,
            chunk_shape=(4, 2),
            source_assembly="hg38",
        )
    root = open_group(store / "data.zarr", "r")
    shape = tuple(int(size) for size in root["z"].shape)
    assert shape[0] > 8 and shape[1] > 2, shape  # more than one shard on each axis's worth
    assert tuple(int(size) for size in root["z"].chunks) == (4, 2)
    assert tuple(int(size) for size in root["z"].shards) == (8, 4)


def test_a_dense_completion_writes_whole_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The completion band writer's production row block, under the guard (#247 r1).

    With the decided shard replaced by `(4, 2)` and the completion's memory
    target by 6 rows, the new row block is a whole 4-row shard; the old
    `max(inner rows, _BAND_ROWS)` block of 6 is not a multiple of 4 and reverted
    code writes `z[r0:r0+6]`, which the guard refuses.
    """
    import test_dense_completion as tdc

    import opengwasdb.layouts.dense.complete as complete_module
    import opengwasdb.store.arrays as arrays_module
    from opengwasdb.build.source import stream_normalised_associations
    from opengwasdb.layouts.dense.build import build_dense_observed_store
    from opengwasdb.layouts.dense.complete import complete_dense_store

    monkeypatch.setattr(arrays_module, "DENSE_SHARD_SHAPE", (4, 2))
    monkeypatch.setattr(arrays_module, "DENSE_CHUNK_SHAPE", (2, 1))
    monkeypatch.setattr(complete_module, "DEFAULT_CHUNK_SHAPE", (2, 1))
    monkeypatch.setattr(complete_module, "_BAND_ROWS", 6)

    rows = [
        f"a1\tp1\tTrait\tTrait primary\t1\t{900_000 + i * 1_000}\tA\tG\t2.0\t0.1\trs{i}\tsd"
        for i in range(12)
    ]
    source = tmp_path / "associations.tsv"
    source.write_text(tdc.SOURCE_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    observed = tmp_path / "obs.opengwasdb"
    build_dense_observed_store(
        stream_normalised_associations([source]),
        observed,
        store_id="s",
        release_id="obs",
        reference_assembly="GRCh38",
        chunk_shape=(2, 1),
    )
    completed = tmp_path / "comp.opengwasdb"
    with require_whole_shard_writes():
        complete_dense_store(
            observed, completed, tdc._make_ld_panel(tmp_path), ancestry="EUR", min_cor=0.0
        )
    root = open_group(completed / "data.zarr", "r")
    assert tuple(int(size) for size in root["z"].chunks) == (2, 1)
    shards = tuple(int(size) for size in root["z"].shards)
    shape = tuple(int(size) for size in root["z"].shape)
    assert shards == (4, 1)
    assert shape[0] > shards[0]  # more than one shard on the variant axis


def test_a_hybrid_completion_patches_whole_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hybrid completion's cell patches are whole-shard writes (#247 r1).

    The crossover fold patched `z`/`se`/`eaf`/`imputed` one cell at a time
    through `vindex`, which the extended guard now catches.  This counts the
    whole-shard writes the rewritten paths make: one band per touched shard per
    array, against one shard read-modify-write per cell per array before.
    """
    import test_hybrid_completion as thc

    import opengwasdb.encoding.planes as planes_module
    import opengwasdb.layouts.hybrid.complete as hybrid_complete_module
    from opengwasdb.layouts.hybrid.complete import complete_hybrid_store

    bands: list[int] = []
    cells: list[int] = []

    def counting(array: object, rows: object, cols: object, values: object) -> None:
        import numpy as np

        shards = getattr(array, "shards", None)
        shape = array.shape
        if shards is not None and len(shape) == 2:
            sr, sc = int(shards[0]), int(shards[1])
            n_col_shards = -(-int(shape[1]) // sc)
            keys = (np.asarray(rows) // sr) * n_col_shards + (np.asarray(cols) // sc)
            bands.append(int(np.unique(keys).size))
        cells.append(int(np.asarray(rows).size))
        original(array, rows, cols, values)

    original = planes_module.write_shard_cells
    monkeypatch.setattr(planes_module, "write_shard_cells", counting)
    monkeypatch.setattr(hybrid_complete_module, "write_shard_cells", counting)

    src = thc._build_source(tmp_path)
    ld = thc._make_ld_panel_with_crossover(tmp_path)
    dst = tmp_path / "dst.opengwasdb"
    with require_whole_shard_writes():
        complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)

    assert bands, "the crossover fold patched no cells"
    # Before the rewrite each patched cell was its own `vindex` shard
    # read-modify-write; now it is one band per touched shard.
    before = sum(cells)
    after = sum(bands)
    assert after <= before
    print(f"hybrid crossover shard writes: before {before}, after {after} ({bands})")


def test_builder_and_converter_agree_on_a_tail_shard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A tail shard is laid out identically by a build and a conversion (#247 r1).

    A 10-variant × 3-Analysis release at inner `(2, 1)` and shard `(4, 2)` has a
    full shard and a 2-row tail on the variant axis; the `v2` source is a
    relayout of the built release and is converted with the same shapes, so
    every array's chunk and shard must match.
    """
    from legacy_fixtures import relayout_as_0_1_0

    import opengwasdb.store.arrays as arrays_module
    import opengwasdb.store.convert as convert_module
    from opengwasdb.build.source import stream_normalised_associations
    from opengwasdb.layouts.dense.build import build_dense_observed_store
    from opengwasdb.store.convert import convert_release

    monkeypatch.setattr(arrays_module, "DENSE_CHUNK_SHAPE", (2, 1))
    monkeypatch.setattr(arrays_module, "DENSE_SHARD_SHAPE", (4, 2))
    monkeypatch.setattr(convert_module, "DENSE_CHUNK_SHAPE", (2, 1))

    header = (
        "analysis_id\tphenotype_id\tphenotype_label\tanalysis_label\tchromosome\tposition"
        "\teffect_allele\tother_allele\tz\tse\trsid\tstored_effect_scale"
    )
    rows = [
        f"a{a}\tp{a}\tTrait {a}\tTrait {a} primary\t1\t{1000 + i * 10}\tA\tG\t2.0\t0.1"
        f"\trs{i}\tsd"
        for a in range(3)
        for i in range(10)
    ]
    source = tmp_path / "associations.tsv"
    source.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    built = tmp_path / "built.opengwasdb"
    build_dense_observed_store(
        stream_normalised_associations([source]),
        built,
        store_id="s",
        release_id="r",
        reference_assembly="GRCh37",
        chunk_shape=(2, 1),
    )
    legacy = relayout_as_0_1_0(built, tmp_path / "legacy.opengwasdb")
    converted = tmp_path / "converted.opengwasdb"
    convert_release(
        legacy, converted, dense_analysis_chunk=1, dense_shard=(4, 2), workers=1
    )

    left = open_group(built / "data.zarr", "r")
    right = open_group(converted / "data.zarr", "r")
    z = left["z"]
    assert tuple(int(size) for size in z.shards) == (4, 2)
    assert int(z.shape[0]) % 4 != 0, "no tail shard in the fixture"
    for name in ("z", "se", "eaf"):
        if name not in left:
            continue
        assert tuple(int(s) for s in left[name].chunks) == tuple(
            int(s) for s in right[name].chunks
        ), name
        assert tuple(int(s) for s in left[name].shards) == tuple(
            int(s) for s in right[name].shards
        ), name
        assert tuple(int(s) for s in left[name].shape) == tuple(
            int(s) for s in right[name].shape
        ), name


def test_an_inner_chunk_that_does_not_tile_the_dense_shard_is_refused() -> None:
    """A build cannot derive a different shard from its inner chunk (#247 r1).

    `chunk_layout`/`shard_layout` are the seam's one authority: an inner chunk
    that does not divide an axis of the decided `[100000, 1024]` shard is
    refused, with the values that do named, instead of writing `[100000, 1000]`.
    A small array still clips.
    """
    from opengwasdb.store.arrays import chunk_layout, shard_layout

    shape = (200_000, 2_000)
    inner = chunk_layout(ArrayRole.DENSE_STATISTIC_PLANE, shape, hint=(1000, 1000))
    with pytest.raises(ValueError, match="does not tile") as excinfo:
        shard_layout(ArrayRole.DENSE_STATISTIC_PLANE, shape, inner_chunk=inner)
    assert "1024" in str(excinfo.value) and "1000" in str(excinfo.value)
    assert "Allowed Analysis-axis inner chunks" in str(excinfo.value)

    # A divisor of 1024 is accepted and produces the decided shard.
    tiling = chunk_layout(ArrayRole.DENSE_STATISTIC_PLANE, shape, hint=(1000, 512))
    assert shard_layout(ArrayRole.DENSE_STATISTIC_PLANE, shape, inner_chunk=tiling) == (
        100_000,
        1_024,
    )
    # A small array clips instead of refusing.
    small = chunk_layout(ArrayRole.DENSE_STATISTIC_PLANE, (11, 3), hint=(1000, 1000))
    assert shard_layout(ArrayRole.DENSE_STATISTIC_PLANE, (11, 3), inner_chunk=small) == (11, 3)


def test_a_partial_shard_write_without_the_guard_is_silent(
    plane: tuple[object, object]
) -> None:
    """The guard is what makes the defect visible; without it, nothing objects.

    This is the "observed failing" half: with the hook absent the misaligned
    write that `test_a_misaligned_band_write_fails_loudly` refuses simply
    succeeds, which is the silent read-modify-write #247 exists to prevent.
    """
    array, band = plane
    array[0 : INNER[0], 0 : INNER[1]] = band[: INNER[0], : INNER[1]]
    assert np.isfinite(np.asarray(array[:INNER[0], :INNER[1]])).all()


# ── the row-block writers choose whole-shard multiples ───────────────────────


def test_the_eaf_row_band_is_a_whole_number_of_shards() -> None:
    for shard_rows in (1, 7, 100_000):
        band = _eaf_row_band(shard_rows, 10)
        assert band % shard_rows == 0, (shard_rows, band)
        assert band >= shard_rows


def test_the_completion_band_rows_are_a_whole_number_of_shards() -> None:
    for shard_rows in (1, 7, 100_000):
        band = _completion_band_rows(shard_rows)
        assert band % shard_rows == 0, (shard_rows, band)
        assert band >= shard_rows


def test_the_dense_vcf_band_width_is_the_shard_analysis_width(plane: tuple[object, object]) -> None:
    """The band writer's width is the shard's Analysis extent, not the inner chunk.

    A band one inner chunk wide is the pre-#247 behaviour and turns every band
    write into a read-modify-write of the shard it ends inside; this pins the
    fix at the function the band writer calls.
    """
    array, _band = plane
    assert _shard_columns(array, INNER) == SHARD[1]
    assert _shard_columns(array, INNER) != INNER[1]


def test_the_se_rewrite_row_block_is_the_shard_rows(plane: tuple[object, object]) -> None:
    array, _band = plane
    assert _row_block_of(array) == SHARD[0]


def test_the_se_rewrite_row_block_falls_back_to_the_inner_chunk_without_a_shard(
    tmp_path: Path,
) -> None:
    root = open_group_for_write(tmp_path / "v2.zarr", "w", zarr_format=2)
    array = create_array(
        root, "se", ArrayRole.DENSE_STATISTIC_PLANE, shape=(7, 3), hint=(7, 3), dtype="float32"
    )
    assert array.shards is None
    assert _row_block_of(array) == 7


# ── 1-D Ragged sequences (#249) ──────────────────────────────────────────────


def test_the_guard_judges_a_one_dimensional_sequence(tmp_path: Path) -> None:
    """A 1-D Ragged sequence is judged on the same whole-shard rule (#249).

    Before #249 the guard returned early for any array that was not 2-D, so a
    writer could rewrite a 50,000,000-element sequence shard once per region
    with nothing objecting.  The whole shard (including the short final one) is
    aligned; a selection inside a shard raises.
    """
    root = open_group_for_write(tmp_path / "data.zarr", "w", zarr_format=3)
    array = create_array(
        root,
        "z",
        ArrayRole.ASSOCIATION_SEQUENCE,
        shape=(9,),
        dtype="int16",
        compressor=sharded_compressor(),
        inner_chunk=(3,),
        shards=(6,),
    )
    assert tuple(int(size) for size in array.shards) == (6,)
    with require_whole_shard_writes():
        array[0:6] = np.arange(6, dtype="int16")  # the whole first shard
        array[6:9] = np.arange(3, dtype="int16")  # the whole short final shard
    assert list(np.asarray(array[:])) == list(range(6)) + [0, 1, 2]
    with require_whole_shard_writes(), pytest.raises(PartialShardWriteError):
        array[0:3] = np.zeros(3, dtype="int16")  # inside the first shard
    with require_whole_shard_writes(), pytest.raises(PartialShardWriteError):
        array[3:6] = np.zeros(3, dtype="int16")  # the second half of the first shard
    with require_whole_shard_writes(), pytest.raises(PartialShardWriteError):
        array[1:9] = np.zeros(8, dtype="int16")  # starts off the shard boundary


def test_a_whole_array_shard_one_dimensional_table_may_be_filled_incrementally(
    tmp_path: Path,
) -> None:
    """The Dense SE rewrite fills its exception table band by band (#249).

    `se_exception_index` is a 1-D array whose shard policy is "one shard holds
    the whole array"; `encoding/se.py` preallocates it to the exact count the
    codes-only pass produced and writes each row band's run in order.  That is
    by design, so the guard must allow it while still refusing a partial write
    of a multi-shard sequence (the test above).
    """
    root = open_group_for_write(tmp_path / "data.zarr", "w", zarr_format=3)
    table = create_array(
        root,
        "se_exception_index",
        ArrayRole.EXCEPTION_TABLE,
        shape=(9,),
        dtype="int64",
        compressor=sharded_compressor(),
    )
    assert int(table.shards[0]) >= 9  # one shard holds the whole table
    with require_whole_shard_writes():
        table[0:4] = np.arange(4, dtype="int64")
        table[4:9] = np.arange(5, dtype="int64")
    assert list(np.asarray(table[:])) == list(range(4)) + list(range(5))


def test_a_ragged_sequence_shard_is_written_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Ragged flush writes each sequence shard exactly once (#249).

    At the decided 50,000,000-element shard and the 4,194,304-cell region this
    was about twelve rewrites of every shard; the fixture uses an 800,000-element
    shard (four inner chunks) so three shards fit in a test.  The writer's cells
    span all three.  If the region is not shard-aligned the same shard key is
    written more than once, which is the amplification, not a correctness bug.
    """
    import opengwasdb.store.arrays as store_arrays
    from opengwasdb.encoding import EncodingMeasurements, StoreEncoding
    from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter

    monkeypatch.setattr(
        store_arrays,
        "RAGGED_SEQUENCE_SHARD_ELEMENTS",
        4 * store_arrays.ASSOCIATION_SEQUENCE_CHUNK,
    )
    encoding = StoreEncoding.decide(EncodingMeasurements(n_analyses=3))
    writer = RaggedCSRWriter(50_000)
    rng = np.random.default_rng(0)
    for _ in range(3):
        writer.add_analysis(
            np.sort(rng.integers(0, 50_000, size=700_000)).astype(np.int32),
            rng.standard_normal(700_000).astype(np.float32),
            np.abs(rng.standard_normal(700_000)).astype(np.float32),
            rng.random(700_000).astype(np.float32),
        )
    expected_shards = 3  # 2,100,000 cells / 800,000
    with count_shard_writes() as recorder:
        writer.flush(tmp_path, encoding, region_cells=200_000)
    written = recorder.chunk_writes()
    sequences = {path: keys for path, keys in written.items() if path in ("z", "se")}
    assert sequences, written  # the fixture actually wrote sequences
    for path, keys in sequences.items():
        assert len(keys) == expected_shards, (path, keys)
        assert set(keys.values()) == {1}, (path, keys)
    assert len(written["variant_index"]) == expected_shards
    assert set(written["variant_index"].values()) == {1}
