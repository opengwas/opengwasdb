"""The refusal of a Variant Set with no 100 kb pair is checked a tile of rows at a time, not
over a whole chromosome (opengwas/ogld#25).

`diagnostic.require_change_pairs` listed every pair within 100 kb of a whole chromosome before
it asked whether any lay outside the Long-Range LD Regions: 9.4 GB on EUR chr6, the build's peak
once the PC Check's genotypes were windowed. It now asks per tile of rows and stops at the first
pair. The answer is the same: a refusal if and only if no pair is clear.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import numpy.typing as npt
import pytest

from opengwasdb.ld.build import diagnostic, tiles
from opengwasdb.ld.format import InputError

MHC = (28_510_020, 33_480_577)
WIDE_MHC_END = 35_032_223  # price2008_12_MHC_wide, the region that ends last on chromosome 6


def _spread(rows: int, step: int, first: int = 1_000_000) -> npt.NDArray[np.int64]:
    return first + np.arange(rows, dtype=np.int64) * step


def test_a_chromosome_is_checked_a_tile_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # 6,000 variants 200 bp apart: 500 partners within 100 kb each, 3 M pairs in all, which a
    # whole-chromosome pass holds at once (about 70 MB) and a 256-row tile holds 1/23 of.
    # The only clear pair is the last two rows, so every earlier tile finds nothing: the check
    # must go on to the partial last tile (6,000 = 23 × 256 + 112) and not stop early.
    inside = _spread(5_998, 200, MHC[0])
    position = np.concatenate([inside, [WIDE_MHC_END + 1_000, WIDE_MHC_END + 1_100]])
    assert not diagnostic.outside_regions("6", position)[:5_998].any(), "fixture must be inside"
    assert diagnostic.outside_regions("6", position)[5_998:].all()
    calls: list[int] = []
    real = tiles.change_pairs

    def spy(
        band: tiles.Band, outside: npt.NDArray[np.bool_], first: int, last: int
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        calls.append(last - first)
        return real(band, outside, first, last)

    monkeypatch.setattr(tiles, "change_pairs", spy)
    tracemalloc.start()
    try:
        diagnostic.require_change_pairs([("6", position)], "test-panel")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert max(calls) <= tiles.TILE_ROWS
    assert sum(calls) == 6_000, "every row is checked once"
    assert calls[-1] == 6_000 % tiles.TILE_ROWS, "the partial last tile is checked"
    # The rows' pairs that fall inside the regions are listed too: 3 M pairs at 24 B each.
    assert peak < 20_000_000, f"{peak / 1e6:.0f} MB held: a chromosome's pairs, not a tile's"


def test_the_first_clear_pair_ends_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    position = _spread(5_000, 200)  # chr1 here: 1.0-2.0 Mb, outside every region
    assert diagnostic.outside_regions("1", position).all()
    seen: list[int] = []
    real = tiles.change_pairs
    monkeypatch.setattr(
        tiles,
        "change_pairs",
        lambda b, o, first, last: seen.append(last - first) or real(b, o, first, last),
    )
    diagnostic.require_change_pairs([("1", position)], "test-panel")
    assert seen == [tiles.TILE_ROWS], "it read on after a tile held a clear pair"


@pytest.mark.parametrize("rows", [255, 256, 257, 1_000])
def test_a_variant_set_with_no_clear_pair_is_refused(rows: int) -> None:
    position = _spread(rows, 5, MHC[0])  # all inside the MHC, whatever the tile edges
    assert not diagnostic.outside_regions("6", position).any()
    with pytest.raises(InputError, match="no two variants within 100 kb"):
        diagnostic.require_change_pairs([("6", position)], "test-panel")


def test_a_pair_split_across_chromosomes_is_not_a_pair() -> None:
    far = np.array([1_000_000, 1_500_000], dtype=np.int64)  # > 100 kb apart on one chromosome
    with pytest.raises(InputError, match="no two variants within 100 kb"):
        diagnostic.require_change_pairs([("1", far[:1]), ("2", far[1:])], "test-panel")
