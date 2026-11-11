"""The PC Check's pass holds a window's genotypes, not a chromosome's (opengwas/ogld#25).

`pc_check.stream_windows` finds the windows `pc_check.windows` finds, but reads each window's
thinned variants when it is scanned. So the scan is the same answer, and what is held is bounded
by `MAX_VARIANTS` rows a window and `threads` windows at once, whatever the chromosome's rows.
Expected values are `pc_check.windows` over the whole genotype matrix, the reference
implementation the stream must reproduce.
"""

from __future__ import annotations

import tracemalloc
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest
import testdata

from ogld.build import passes, pc_check
from ogld.build.genotypes import Genotypes
from ogld.build.variant_set import Settings, variant_set

#: A window thins to at most this many variants here, so a small fixture is thinned too.
THIN: Final = 40
SAMPLES: Final = 300


@pytest.fixture
def thin(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(pc_check, "MAX_VARIANTS", THIN)
    return THIN


def _chromosome(
    rows: int, span: int = 17_000_000, seed: int = 3
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """`rows` variants spread over `span` bp (the last window partial), float32 genotypes of
    SAMPLES samples; a tenth of them rare (MAF < 0.05), so the selection matters."""
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(np.arange(1, span), rows, replace=False)).astype(np.int64)
    freq = np.where(rng.random(rows) < 0.1, 0.005, rng.uniform(0.1, 0.5, rows))
    return pos, rng.binomial(2, freq[:, None], (rows, SAMPLES)).astype(np.float32)


class Reads:
    """A `read` over genotypes `g` that remembers how many rows each call asked for."""

    def __init__(self, g: npt.NDArray[np.float32]) -> None:
        self.g, self.sizes = g, []

    def __call__(self, rows: npt.NDArray[np.intp]) -> npt.NDArray[np.float32]:
        self.sizes.append(len(rows))
        return self.g[rows]

    sizes: list[int]


def _streamed(
    pos: npt.NDArray[np.int64], read: Reads, g: npt.NDArray[np.float32], threads: int
) -> pc_check.Windows:
    return pc_check.stream_windows("5", pos, pc_check.common(g), read, g.shape[1], threads=threads)


@pytest.mark.parametrize("threads", [2, 4, 7])
def test_the_chunking_of_windows_by_threads_changes_no_window(thin: int, threads: int) -> None:
    pos, g = _chromosome(3_000)
    alone = _streamed(pos, Reads(g), g, 1)
    reads = Reads(g)
    streamed = _streamed(pos, reads, g, threads)
    # The fixture must reach what the chunking does: thinned windows, rare rows left out, the
    # partial last window, and a window count that no thread count above one divides.
    assert max(reads.sizes) == thin, "no window is thinned"
    assert not pc_check.common(g).all(), "no rare row to leave out"
    assert len(alone.where) == 9, "nine windows: no thread count above one divides them"
    assert alone.where[-1][1] + pc_check.WINDOW_BP > int(pos[-1]) + pc_check.STEP_BP
    assert streamed.where == alone.where
    np.testing.assert_array_equal(streamed.basis, alone.basis)


def test_windows_over_the_whole_matrix_is_the_stream_per_chromosome(thin: int) -> None:
    first, g1 = _chromosome(3_000)
    second, g2 = _chromosome(2_000, seed=9)
    chrom = np.array(["5"] * len(first) + ["7"] * len(second))
    pos, g = np.concatenate([first, second]), np.concatenate([g1, g2])
    both = pc_check.windows(chrom, pos, g, threads=3)
    parts = [
        _streamed(first, Reads(g1), g1, 1),
        pc_check.stream_windows("7", second, pc_check.common(g2), Reads(g2), SAMPLES),
    ]
    expected = pc_check.merge(parts)
    assert {c for c, _ in both.where} == {"5", "7"}, "one chromosome only proves little"
    assert both.where == expected.where
    np.testing.assert_array_equal(both.basis, expected.basis)


def test_a_chromosome_with_no_common_variant_has_no_windows(thin: int) -> None:
    pos, g = _chromosome(300)
    rare = np.zeros_like(g)
    rare[:, :2] = 1
    assert not pc_check.common(rare).any()
    streamed = pc_check.stream_windows("5", pos, pc_check.common(rare), Reads(rare), SAMPLES)
    assert streamed.where == []
    assert streamed.basis.shape == (SAMPLES, 0)
    both = pc_check.windows(
        np.array(["5"] * 300 + ["6"] * len(pos)), np.tile(pos, 2), np.concatenate([rare, g])
    )
    assert {c for c, _ in both.where} == {"6"}, "the chromosome with no common variant is skipped"


def test_the_common_rule_at_its_edges() -> None:
    n = 100
    exactly = np.zeros((1, n), dtype=np.float32)
    exactly[0, :10] = 1  # MAF exactly 0.05: common, since the rule is >=
    # Non-integer dosages summing to 9.999999999999998 in float64, whose float64 mean is below
    # 0.05 and whose float32 mean is not: the scan's float32 decides.
    dosages = np.zeros((1, n))
    dosages[0, :20] = [
        0.7,
        0.5,
        0.4,
        0.7,
        0.2,
        0.1,
        0.3,
        0.7,
        0.4,
        0.5,
        0.3,
        0.7,
        0.5,
        0.5,
        0.7,
        0.2,
        0.5,
        0.7,
        0.7,
        0.7,
    ]
    assert dosages[0].mean() / 2 < pc_check.COMMON_MAF, "the fixture must straddle the threshold"
    below = np.zeros((1, n), dtype=np.float32)
    below[0, :9] = 1  # MAF 0.045
    assert pc_check.common(exactly).tolist() == [True]
    assert pc_check.common(dosages).tolist() == [True]
    assert pc_check.common(below).tolist() == [False]
    assert pc_check.common(dosages.astype(np.float32)).tolist() == [True]


def test_a_window_reads_only_its_thinned_rows(thin: int) -> None:
    pos, g = _chromosome(3_000)
    reads = Reads(g)
    _streamed(pos, reads, g, 3)
    assert reads.sizes, "the stream read nothing"
    assert max(reads.sizes) <= thin
    assert sum(reads.sizes) < pc_check.common(g).sum() * 2, "it read more than the windows need"


def test_a_chromosome_with_no_window_reads_nothing(thin: int) -> None:
    pos, g = _chromosome(4, span=1_000_000)
    reads = Reads(g)
    streamed = _streamed(pos, reads, g, 2)
    assert streamed.where == []
    assert streamed.basis.shape == (SAMPLES, 0)
    assert reads.sizes == []


def _peak(rows: int, threads: int) -> int:
    """The most memory the stream holds over a chromosome of `rows` variants, whose genotypes
    are made on demand, so that nothing but the stream keeps them."""
    pos, _ = _chromosome(rows)
    seed = np.random.default_rng(1).integers(0, 2**31, rows)

    def read(wanted: npt.NDArray[np.intp]) -> npt.NDArray[np.float32]:
        return np.stack(
            [np.random.default_rng(seed[w]).binomial(2, 0.3, SAMPLES) for w in wanted]
        ).astype(np.float32)

    selected = np.ones(rows, dtype=np.bool_)
    tracemalloc.start()
    try:
        pc_check.stream_windows("5", pos, selected, read, SAMPLES, threads=threads)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("threads", [1, 3])
def test_memory_does_not_grow_with_the_chromosome_s_rows(thin: int, threads: int) -> None:
    small_rows, large_rows = 4_000, 16_000
    small, large = _peak(small_rows, threads), _peak(large_rows, threads)
    # What each extra row costs: its position and index entries, never its genotypes, which
    # are SAMPLES float32 values (a whole-chromosome pass costs at least that a row).
    per_row = (large - small) / (large_rows - small_rows)
    assert per_row < SAMPLES * 4 / 4, f"{per_row:.0f} bytes a row: peak {small} -> {large}"


def test_the_build_s_pass_reads_windows_and_finds_the_whole_scan(
    thin: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The build's pass over the Test Panel, through `variant_set`: the scan equals the one made
    from every passing row at once, and no read is larger than a window's thinned variants."""
    panel = testdata.panel()
    plan = passes.plan((panel.pgen,), panel.keep, panel.pcs, "test-panel")
    asked: list[int] = []
    real = Genotypes.rows

    def spy(self: Genotypes, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        asked.append(len(rows))
        return real(self, rows)

    monkeypatch.setattr(Genotypes, "rows", spy)
    settings = Settings("test-panel", 20, None, True, threads=3)
    found = variant_set(plan.filesets, settings, plan.samples.n)
    assert asked, "the pass read no window"
    assert max(asked) <= thin
    # The expected scan: every kept variant's genotypes in memory, in the Variant Set's order.
    with Genotypes(plan.filesets[0]) as genotypes:
        for chromosome in found.chromosomes:
            g = genotypes.rows(chromosome.rows).astype(np.float32)
            expected = pc_check.windows(
                np.full(len(chromosome), chromosome.name), chromosome.position, g
            )
            got = found.windows[chromosome.name]
            assert len(expected.where) > 1, "a chromosome with one window proves little"
            assert got.where == expected.where
            np.testing.assert_array_equal(got.basis, expected.basis)
    assert len(found.chromosomes) == 2
