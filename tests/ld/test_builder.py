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

from pathlib import Path

import build_fixtures as fixtures
import numpy as np
import numpy.typing as npt
import pytest
import testdata

import ogld
from ogld.format import far_code_floor, quantise, t_far

N, K = 771, 10


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[int, Path]:
    """The Test Panel built at k = 10 and k = 0, shared (build_fixtures.test_panel_build)."""
    return {k: fixtures.test_panel_build(k, tmp_path_factory)[1] for k in (10, 0)}


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
def test_the_near_band_is_the_reference_r_signed_to_a1(built: dict[int, Path], k: int) -> None:
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


def test_the_far_pairs_are_the_reference_s_above_the_floor(built: dict[int, Path]) -> None:
    rows = passing()
    ours, theirs = fixtures.stored_r(built[K], rows), reference_at(K, rows)
    v = testdata.variants()
    pos, chrom = v.pos[rows], v.chrom[rows]
    d = np.abs(pos[:, None] - pos[None, :])
    far = (d > 3_000_000) & (d <= 10_000_000) & (chrom[:, None] == chrom[None, :])
    floor = far_code_floor(t_far(N, K))
    expected = np.abs(quantise(theirs[far]).astype(np.int64)) >= floor
    kept = ~np.isnan(ours[far])
    assert expected.sum() == 14, "the Test Panel's 7 Far Pairs, both ways round"
    assert (~expected).sum() > 70_000, "and tens of thousands of 3–10 Mb pairs below t_far"
    assert np.array_equal(kept, expected)
    assert np.abs(ours[far][kept] - theirs[far][kept]).max() <= fixtures.HALF_STEP + fixtures.SLACK


def test_the_pc_file_s_row_order_does_not_matter(built: dict[int, Path], tmp_path: Path) -> None:
    # Covariates are matched to genotypes by IID, never by row: a PC File in another order
    # builds the same store.
    lines = testdata.panel().pcs.read_text().splitlines()
    shuffled = tmp_path / "pcs.eigenvec"
    shuffled.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
    ogld.build(fixtures.inputs(testdata.panel().pgen, K, pcs=shuffled), tmp_path / "store")
    fixtures.same_store(built[K], tmp_path / "store")


def test_two_filesets_in_two_sample_orders_build_one_store(
    built: dict[int, Path], tmp_path: Path
) -> None:
    # chr21 and chr22 in a fileset each, chr22's psam listing the samples backwards: pgenlib
    # returns each fileset's samples in its own psam order, and the build puts them in one.
    v = testdata.variants()
    on21, on22 = np.flatnonzero(v.chrom == "21"), np.flatnonzero(v.chrom == "22")
    first = fixtures.write_fileset(tmp_path / "chr21", on21)
    backwards = np.arange(N)[::-1]
    second = fixtures.write_fileset(tmp_path / "chr22", on22, samples=backwards)
    ogld.build(fixtures.inputs(first, K, pgen=(first, second)), tmp_path / "store")
    fixtures.same_store(built[K], tmp_path / "store")


def test_the_thread_count_does_not_change_the_store(built: dict[int, Path], tmp_path: Path) -> None:
    provenance = ogld.build(
        fixtures.inputs(testdata.panel().pgen, K), tmp_path / "store", threads=3
    )
    assert provenance["build"]["threads"] == 3
    fixtures.same_store(built[K], tmp_path / "store")


def test_a_copied_fileset_is_the_same_input(built: dict[int, Path], tmp_path: Path) -> None:
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


def test_the_proxy_index_layout_is_both_directions_sorted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The layout places each half where it goes instead of sorting 2N entries; it must equal
    # the sort, across several placement chunks.
    from ogld.build import tiles

    rng = np.random.default_rng(9)
    rows = 300
    pairs = {(int(a), int(b)) for a, b in rng.integers(0, rows, (4_000, 2)) if a < b}
    lo, hi = (np.array(x, dtype=np.int32) for x in zip(*sorted(pairs), strict=True))
    codes = rng.integers(-32767, 32768, len(lo)).astype(np.dtype("int16"))
    monkeypatch.setattr(tiles, "PLACE_CHUNK", 97)
    indptr, partner, placed = tiles.Entries.of([lo], [hi], [codes]).both_directions(rows, 1_000)
    row = np.concatenate([lo, hi]).astype(np.int64)
    other = np.concatenate([hi, lo]).astype(np.int64)
    order = np.lexsort((other, row))
    assert np.array_equal(partner, other[order] + 1_000)
    assert np.array_equal(placed, np.concatenate([codes, codes])[order])
    assert np.array_equal(
        indptr, np.concatenate([[0], np.cumsum(np.bincount(row, minlength=rows))])
    )
    with pytest.raises(ValueError, match="not in"):
        tiles.Entries.of([lo[::-1]], [hi[::-1]], [codes])


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

    def spy(inputs: tiles.Inputs, first: int, last: int) -> tiles.Tile:
        pools = [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"]
        seen.append(max(pools))
        return real(inputs, first, last)

    monkeypatch.setattr(tiles, "_tile", spy)
    pgen = fixtures.write_fileset(
        tmp_path / "chr22", np.flatnonzero(testdata.variants().chrom == "22")
    )
    with threadpool_limits(limits=8, user_api="blas"):
        assert max(p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas") == 8
        provenance = ogld.build(fixtures.inputs(pgen), tmp_path / "store", threads=2)
    assert seen, "the spy must see the build's tiles"
    assert max(seen) == 1 <= provenance["build"]["threads"]
