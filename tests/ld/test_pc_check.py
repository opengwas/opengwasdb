"""The PC Check's reference rule, `scripts/test_panel/pc_check.py` (build-inputs.md §8).

Hand-built genotypes, because the case that matters is one real data lacks: a noise PC in a small
panel. With 60 samples a random vector's best 4 Mb window explains well over 10% of it by
chance, and the rest of the genome explains it exactly by chance, so a rule with no window-side
null drops it as local. A window's own genotype axis is the opposite case, and must be dropped.
"""

from __future__ import annotations

import numpy as np
import pytest
from script_modules import load_script

pc_check = load_script("test_panel/pc_check")

N, CHROMS, STEP_BP = 60, 3, 20_000


def _genome(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unlinked genotypes: 2,000 variants every 20 kb on each of three chromosomes."""
    rng = np.random.default_rng(seed)
    per = 2_000
    chrom = np.repeat([str(c) for c in range(1, CHROMS + 1)], per)
    pos = np.tile(np.arange(1, per + 1) * STEP_BP, CHROMS)
    freq = rng.uniform(0.1, 0.5, len(pos))
    return chrom, pos, rng.binomial(2, freq[:, None], (len(pos), N)).astype(np.int8)


def _local_axis(g: np.ndarray, pos: np.ndarray, chrom: np.ndarray) -> np.ndarray:
    """The top sample-space PC of one 4 Mb window's genotypes, chromosome 2, 10-14 Mb."""
    rows = (chrom == "2") & (pos >= 10_000_000) & (pos < 14_000_000)
    x = g[rows].astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    return np.linalg.svd(x.T, full_matrices=False)[0][:, 0]


@pytest.fixture(scope="module")
def scanned() -> tuple[object, np.ndarray, np.ndarray]:
    chrom, pos, g = _genome()
    windows = pc_check.windows(chrom, pos, g)
    noise = np.random.default_rng(11).standard_normal((N, 19))
    return windows, _local_axis(g, pos, chrom), noise


def test_a_noise_pc_is_kept_though_a_window_explains_over_10_percent(scanned) -> None:
    windows, local, noise = scanned
    scan = pc_check.check(windows, np.column_stack([local, noise]))["scan"]
    noisy = [s for s in scan[1:] if s["window_share"] >= pc_check.MIN_SHARE]
    assert len(noisy) >= 5, "the fixture must give noise PCs a window share over 10%"
    assert not any(s["local"] for s in noisy), "a chance window is not a local PC"
    assert all(s["window_share"] <= s["window_null_q99"] for s in noisy)


def test_a_window_s_own_axis_is_local(scanned) -> None:
    windows, local, noise = scanned
    first = pc_check.check(windows, np.column_stack([local, noise]))["scan"][0]
    # Windows step by 2 Mb from each chromosome's first variant, here at 20 kb.
    assert first["local"] and first["window"] == "2:10020000-14019999"
    assert first["window_share"] > 0.9 and first["window_share"] > first["window_null_q99"]


def test_covariates_are_the_first_ten_kept(scanned) -> None:
    windows, local, noise = scanned
    record = pc_check.check(windows, np.column_stack([local, noise]))
    assert record["dropped"] == ["PC1"]
    assert record["covariates"] == [f"PC{n}" for n in range(2, 12)]
    assert (record["k_target"], record["k_min"], record["k"]) == (10, 5, 10)


def test_fewer_kept_pcs_than_the_minimum_is_refused(scanned) -> None:
    windows, local, noise = scanned
    many_local = np.column_stack([local + 1e-3 * noise[:, n] for n in range(16)] + [noise[:, :4]])
    with pytest.raises(SystemExit, match="fewer than the minimum 5"):
        pc_check.check(windows, many_local)
