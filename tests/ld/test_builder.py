"""ogld#5's builder against the Test Panel's independent reference r, read back through the store
layer (`ogld.store.StoreReader`), and the inputs that must not change what it builds.

The contract tests (tests/contract/test_build.py) read through `ogld.LDStore.matrix`, which
ogld#4 builds; these read through the reader the walking skeleton built, so they hold the
builder to the reference whether or not ogld#4 has landed.

- **The Near Band** at k = 10 and k = 0 is the reference r signed to A1, within half a
  quantisation step, and the Far Pairs are exactly the reference's above the far code floor.
- **Nothing about how the inputs are laid out changes the store:** the PC File's row order,
  splitting the genotypes into two filesets whose psams order the samples differently, and
  the number of threads.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import build_fixtures as fixtures
import numpy as np
import numpy.typing as npt
import pytest
import testdata
from store_copies import FLOOR, codes_at

import ogld

N, K = 771, 10


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    """The Test Panel's store at k, shared (build_fixtures.test_panel_build) and built when a
    test first asks for that k, so no one setup pays for two builds."""
    return Built(tmp_path_factory)


class Built:
    def __init__(self, factory: pytest.TempPathFactory) -> None:
        self._factory = factory

    def __getitem__(self, k: int) -> Path:
        return fixtures.test_panel_build(k, self._factory)[1]


def passing() -> npt.NDArray[np.int64]:
    return np.flatnonzero(testdata.variants().passes_filters).astype(np.int64)


def reference_at(k: int, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """The reference r among `rows`, signed to A1; NaN where the reference holds no pair."""
    i, j, r = fixtures.reference_pairs(k)
    where = np.full(len(testdata.variants()), -1)
    where[rows] = np.arange(len(rows))
    a, b = where[i], where[j]
    out = np.full((len(rows), len(rows)), np.nan)
    out[a, b] = out[b, a] = r
    np.fill_diagonal(out, 1.0)
    return out


@pytest.mark.parametrize("k", [10, 0])
def test_the_near_band_is_the_reference_r_signed_to_a1(built: Built, k: int) -> None:
    rows = passing()
    ours, theirs = fixtures.stored_r(built[k], rows), reference_at(k, rows)
    v = testdata.variants()
    pos, chrom = v.pos[rows], v.chrom[rows]
    near = (np.abs(pos[:, None] - pos[None, :]) <= 3_000_000) & (chrom[:, None] == chrom[None, :])
    assert near.sum() > 100_000, "the fixture must hold many Near Band pairs"
    assert not np.isnan(theirs[near]).any()
    worst = np.abs(ours[near] - theirs[near]).max()
    assert worst <= fixtures.HALF_STEP + fixtures.SLACK, (
        f"a Near Band pair is {worst} from the reference"
    )
    # The store's r is signed to A1: the fixture must hold pairs of one A1 = REF and one A1 = ALT
    # variant, where r with the sign dropped would be off by twice |r|.
    sign = fixtures.a1_signs()[rows]
    assert (near & (sign[:, None] != sign[None, :]) & (np.abs(theirs) > 0.1)).sum() > 100


def test_the_far_pairs_are_the_reference_s_above_the_floor(built: Built) -> None:
    rows = passing()
    ours, theirs = fixtures.stored_r(built[K], rows), reference_at(K, rows)
    v = testdata.variants()
    pos, chrom = v.pos[rows], v.chrom[rows]
    d = np.abs(pos[:, None] - pos[None, :])
    far = (d > 3_000_000) & (d <= 10_000_000) & (chrom[:, None] == chrom[None, :])
    expected = np.abs(codes_at(theirs[far], fixtures.R_SCALE)) >= FLOOR
    kept = ~np.isnan(ours[far])
    assert expected.sum() == 14, "the Test Panel's 7 Far Pairs, both ways round"
    assert (~expected).sum() > 70_000, "and tens of thousands of 3–10 Mb pairs below t_far"
    assert np.array_equal(kept, expected)
    assert np.abs(ours[far][kept] - theirs[far][kept]).max() <= fixtures.HALF_STEP + fixtures.SLACK


def test_the_pc_file_s_row_order_does_not_matter(built: Built, tmp_path: Path) -> None:
    # Covariates are matched to genotypes by IID, never by row: a PC File in another order
    # builds the same store.
    lines = testdata.panel().pcs.read_text().splitlines()
    shuffled = tmp_path / "pcs.eigenvec"
    shuffled.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
    ogld.build(fixtures.inputs(testdata.panel().pgen, K, pcs=shuffled), tmp_path / "store")
    fixtures.same_store(built[K], tmp_path / "store")


def test_two_filesets_in_two_sample_orders_build_one_store(built: Built, tmp_path: Path) -> None:
    # chr21 and chr22 in a fileset each, chr22's psam listing the samples backwards: pgenlib
    # returns each fileset's samples in its own psam order, and the build puts them in one.
    v = testdata.variants()
    on21, on22 = np.flatnonzero(v.chrom == "21"), np.flatnonzero(v.chrom == "22")
    first = fixtures.write_fileset(tmp_path / "chr21", on21)
    backwards = np.arange(N)[::-1]
    second = fixtures.write_fileset(tmp_path / "chr22", on22, samples=backwards)
    ogld.build(fixtures.inputs(first, K, pgen=(first, second)), tmp_path / "store")
    fixtures.same_store(built[K], tmp_path / "store")


def test_the_thread_count_does_not_change_the_store(built: Built, tmp_path: Path) -> None:
    provenance = ogld.build(
        fixtures.inputs(testdata.panel().pgen, K), tmp_path / "store", threads=3
    )
    assert provenance["build"]["threads"] == 3
    fixtures.same_store(built[K], tmp_path / "store")


def test_a_copied_fileset_is_the_same_input(built: Built, tmp_path: Path) -> None:
    # The store names its genotype files by name and hash, not by where they were read.
    provenance = ogld.build(
        fixtures.inputs(fixtures.copy_panel(tmp_path / "copy"), 0), tmp_path / "store"
    )
    assert set(provenance["source"]["sha256"]) == {
        "copy.pgen",
        "copy.pvar",
        "copy.psam",
        "panel.keep",
    }
    fixtures.same_store(built[0], tmp_path / "store")


def test_blas_runs_one_thread_in_each_worker_while_tiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review of #5 (standards a4): `threads` must bound the build, but each tile's matrix product
    # ran on as many BLAS threads as the caller's environment allowed (OpenBLAS here starts 64).
    # Inside a build BLAS runs one thread, so `threads` workers are `threads` threads.
    from threadpoolctl import threadpool_info, threadpool_limits

    from ogld.build import tiles

    seen: list[int] = []
    real = tiles._tile

    def spy(inputs: tiles.Inputs, work: tiles.Work) -> tiles.Tiled:
        pools = [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"]
        seen.append(max(pools))
        return real(inputs, work)

    monkeypatch.setattr(tiles, "_tile", spy)
    pgen = fixtures.write_fileset(
        tmp_path / "chr22", np.flatnonzero(testdata.variants().chrom == "22")
    )
    with threadpool_limits(limits=8, user_api="blas"):
        assert max(p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas") == 8
        provenance = ogld.build(fixtures.inputs(pgen), tmp_path / "store", threads=2)
    assert seen, "the spy must see the build's tiles"
    assert max(seen) == 1 <= provenance["build"]["threads"]


def test_the_store_is_validated_on_the_build_s_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Orchestrator note 6: StoreWriter.finish validates on threads (#4's follow-up, c862ea1), and
    # the build passes it its own, which bound it: the whole EUR panel's validation took 1 h 58 min
    # on one thread and about 11 min on 16.
    from ogld.store import StoreWriter

    seen: list[int] = []
    real = StoreWriter.finish

    def spy(self: StoreWriter, provenance: ogld.Provenance, *, threads: int = 1) -> ogld.Provenance:
        seen.append(threads)
        return real(self, provenance, threads=threads)

    monkeypatch.setattr(StoreWriter, "finish", spy)
    pgen = fixtures.write_fileset(
        tmp_path / "chr22", np.flatnonzero(testdata.variants().chrom == "22")
    )
    ogld.build(fixtures.inputs(pgen), tmp_path / "store", threads=3)
    assert seen == [3]


def test_zarr_s_pool_is_held_to_the_build_s_threads_and_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review round 2 (standards a1): zarr's default pool added 28 threads to the write, so a
    # build at threads = 4 reached 1187% CPU. Inside a build zarr's pool is capped at `threads`,
    # and the caller's own setting is back when the build returns.
    import zarr

    from ogld.store import Cut, StoreWriter

    seen: list[object] = []
    real = StoreWriter.write

    def spy(self: StoreWriter, cut: Cut) -> None:
        seen.append(zarr.config.get("threading.max_workers"))
        real(self, cut)

    monkeypatch.setattr(StoreWriter, "write", spy)
    before = zarr.config.get("threading.max_workers")
    pgen = fixtures.write_fileset(
        tmp_path / "chr22", np.flatnonzero(testdata.variants().chrom == "22")
    )
    ogld.build(fixtures.inputs(pgen), tmp_path / "store", threads=3)
    assert len(seen) > 1, "the spy must see every cut the build writes"
    assert set(seen) == {3}
    assert zarr.config.get("threading.max_workers") == before


#: 64 rows of three-sample unit vectors, each row its own number, and the unadjusted ones negated.
WINDOW_SOURCE = np.column_stack([np.arange(64.0), np.arange(64.0) + 0.5, -np.arange(64.0)])


def window_blocks(firsts: range | list[int] = range(0, 64, 8)) -> Iterator[tuple[int, Any, Any]]:
    """WINDOW_SOURCE as the build reads it, 8 rows a block, starting at each of `firsts`."""
    for first in firsts:
        rows = WINDOW_SOURCE[first : first + 8]
        yield first, rows.copy(), -rows


def test_the_window_of_unit_vectors_moves_the_rows_still_needed_and_spares_a_tile_s_view() -> None:
    # The builder holds a window of a chromosome's unit vectors, never the chromosome (ogld#17).
    # When its array is full, the rows from the oldest tile in flight on move to a new one; a tile
    # still reading the old array must see it unchanged.
    from ogld.build import tiles

    source = WINDOW_SOURCE
    window = tiles.Window(window_blocks(), capacity=20)
    window.reach(10, keep=0)
    in_flight = window.rows(4, 12)
    np.testing.assert_array_equal(in_flight, source[4:12])
    window.reach(24, keep=8)  # 24 rows from row 0 do not fit 20: rows 8 … 15 move
    np.testing.assert_array_equal(window.rows(8, 24), source[8:24])
    np.testing.assert_array_equal(window.rows(8, 24, unadjusted=True), -source[8:24])
    np.testing.assert_array_equal(in_flight, source[4:12])
    window.reach(36, keep=20)
    np.testing.assert_array_equal(window.rows(20, 36), source[20:36])
    np.testing.assert_array_equal(in_flight, source[4:12])


@pytest.mark.parametrize(("first", "stop"), [(4, 12), (16, 26), (12, 10)])
def test_the_window_refuses_rows_it_does_not_hold(first: int, stop: int) -> None:
    # Review of #17 (standards a5): an assert guarded this, which `python -O` drops, and the
    # wrong rows of the array would then be read as unit vectors: plausible r, never a refusal.
    from ogld.build import tiles

    window = tiles.Window(window_blocks(), capacity=20)
    window.reach(10, keep=0)
    window.reach(24, keep=8)  # holds rows 8 … 23
    np.testing.assert_array_equal(window.rows(8, 24), WINDOW_SOURCE[8:24])
    with pytest.raises(ValueError, match=f"rows {first}–{stop - 1} .*rows 8–23"):
        window.rows(first, stop)


def test_the_window_refuses_a_block_that_does_not_start_where_the_last_ended() -> None:
    # A gap would leave rows of the array unwritten, and an overlap would write rows twice; either
    # way some tile would read rows that are not the chromosome's.
    from ogld.build import tiles

    window = tiles.Window(window_blocks([0, 8, 24]), capacity=40)
    window.reach(16, keep=0)
    with pytest.raises(ValueError, match="starts at row 24, not 16"):
        window.reach(30, keep=0)


@pytest.mark.parametrize(("k", "threads"), [(0, 1), (0, 4), (K, 1), (K, 4)])
def test_a_build_whose_window_of_unit_vectors_moves_writes_the_same_store(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, k: int, threads: int
) -> None:
    # Review of #17 (standards a4): no Test Panel chromosome is long enough to move the window at
    # the build's own sizes. With 16-row tiles and 40-row genotype blocks (468 and 588 rows end in
    # a partial block of 28) it moves, and the store must be the one built without a move.
    from ogld.build import tiles

    moves: list[int] = []
    real = tiles.Window._move

    def spy(self: tiles.Window, keep: int) -> None:
        moves.append(keep)
        real(self, keep)

    monkeypatch.setattr(tiles, "TILE_ROWS", 16)
    monkeypatch.setattr(tiles, "UNIT_ROWS", 40)
    monkeypatch.setattr(tiles.Window, "_move", spy)
    ogld.build(fixtures.inputs(testdata.panel().pgen, k), tmp_path / "store", threads=threads)
    assert len(moves) >= 2, f"the window must move more than once, not {len(moves)} times"
    fixtures.same_store(built[k], tmp_path / "store")
