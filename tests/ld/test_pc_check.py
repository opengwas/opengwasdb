"""The PC Check, `opengwasdb.ld.build.pc_check` (build-inputs.md §8): the one implementation,
adopted from
lane 2d's reference `scripts/test_panel/pc_check.py` by ogld#5.

The owner's rule (7 November 2026): a PC is local, and dropped, when one 4 Mb window explains at
least 10% of it, more than a random vector's best window does (the window-side null), and the
rest of the genome explains at most 0.05 of what that window does (rest share / window share).

Two kinds of evidence. Real scans from the prototype, whose verdicts the rule was chosen on
(`proto/1-build-inputs@c1c1639:prototypes/build-inputs/results/rule_power.tsv`), and
hand-built genotypes for the case real data lacks: a noise PC in a small panel, whose best window
explains well over 10% of it by chance at 60 samples. And the Test Panel's own scan, which the
reference made before the adoption and the manifest holds: the adopted module must give it again.
"""

from __future__ import annotations

import numpy as np
import pytest
import testdata
from ld_script_modules import load_script
from ogquery.ld.format import InputError

from opengwasdb.ld.build import pc_check

CHROMS, STEP_BP = 3, 20_000
# A small panel, where chance alone gives a noise PC a best window over 10%; and a panel the size
# of AMR's, where a local axis's rest share (about 5 / n per window, by chance) is small enough
# against its window share for the ratio to drop it. Below n of about 100 the ratio cannot drop
# any PC: chance alone puts every window at 5 / n.
SMALL, PANEL = 60, 300


def _genome(n: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unlinked genotypes for n samples: 2,000 variants every 20 kb on each of 3 chromosomes."""
    rng = np.random.default_rng(seed)
    per = 2_000
    chrom = np.repeat([str(c) for c in range(1, CHROMS + 1)], per)
    pos = np.tile(np.arange(1, per + 1) * STEP_BP, CHROMS)
    freq = rng.uniform(0.1, 0.5, len(pos))
    return chrom, pos, rng.binomial(2, freq[:, None], (len(pos), n)).astype(np.int8)


def _local_axis(g: np.ndarray, pos: np.ndarray, chrom: np.ndarray) -> np.ndarray:
    """The top sample-space PC of one 4 Mb window's genotypes, chromosome 2, 10-14 Mb."""
    rows = (chrom == "2") & (pos >= 10_000_000) & (pos < 14_000_000)
    x = g[rows].astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    return np.linalg.svd(x.T, full_matrices=False)[0][:, 0]


# window_share, window_null_q99, rest_share, rest_null_q99, and the owner's verdict.
REAL_SCANS = {
    # 93% one window, but H1/H2 frequency gives the rest of the genome a trace of ancestry:
    # rest share above its null, so the earlier "no better than chance" rule kept it.
    "EUR 17q21.31 karyotype score": (0.934288, 0.0398293, 0.00725251, 0.00710721, True),
    # EUR's north-south axis, LCT its best window: must be kept.
    "EUR PC2": (0.195118, 0.038234, 0.0203655, 0.0069868, False),
    # AMR's noisiest-looking PC, rest share below its null: must be kept.
    "AMR PC7": (0.0852417, 0.0981663, 0.0168256, 0.0235512, False),
}


@pytest.mark.parametrize("name", sorted(REAL_SCANS))
def test_the_owner_s_rule_on_real_scans(name: str) -> None:
    window_share, window_null, rest_share, _rest_null, local = REAL_SCANS[name]
    assert pc_check.is_local(window_share, window_null, rest_share) is local


def test_the_rest_null_no_longer_decides() -> None:
    window_share, window_null, rest_share, rest_null, _ = REAL_SCANS["EUR 17q21.31 karyotype score"]
    assert rest_share > rest_null, "the fixture must beat its rest null for this to mean anything"
    assert pc_check.is_local(window_share, window_null, rest_share)


def _scanned(n: int) -> tuple[object, np.ndarray, np.ndarray]:
    chrom, pos, g = _genome(n)
    windows = pc_check.windows(chrom, pos, g)
    noise = np.random.default_rng(11).standard_normal((n, 19))
    return windows, _local_axis(g, pos, chrom), noise


@pytest.fixture(scope="module")
def small() -> tuple[object, np.ndarray, np.ndarray]:
    return _scanned(SMALL)


@pytest.fixture(scope="module")
def scanned() -> tuple[object, np.ndarray, np.ndarray]:
    return _scanned(PANEL)


def test_a_noise_pc_is_kept_though_a_window_explains_over_10_percent(small) -> None:
    windows, local, noise = small
    scan = pc_check.check(windows, np.column_stack([local, noise])).record["scan"]
    noisy = [s for s in scan[1:] if s["window_share"] >= pc_check.MIN_SHARE]
    assert len(noisy) >= 5, "the fixture must give noise PCs a window share over 10%"
    assert not any(s["local"] for s in noisy), "a chance window is not a local PC"
    assert all(s["window_share"] <= s["window_null_q99"] for s in noisy)
    assert all(s["ratio"] == pytest.approx(s["rest_share"] / s["window_share"]) for s in scan)


def test_a_window_s_own_axis_is_local(scanned) -> None:
    windows, local, noise = scanned
    first = pc_check.check(windows, np.column_stack([local, noise])).record["scan"][0]
    # Windows step by 2 Mb from each chromosome's first variant, here at 20 kb.
    assert first["local"]
    assert first["window"] == "2:10020000-14019999"
    assert first["window_share"] > 0.85
    assert first["window_share"] > first["window_null_q99"]
    assert first["ratio"] <= pc_check.MAX_RATIO


def test_covariates_are_the_first_ten_kept(scanned) -> None:
    windows, local, noise = scanned
    checked = pc_check.check(windows, np.column_stack([local, noise]))
    assert checked.record["dropped"] == ["PC1"]
    assert checked.covariates == [f"PC{n}" for n in range(2, 12)]
    assert (checked.k_target, checked.k_min, checked.k, checked.kept) == (10, 5, 10, 19)


def test_fewer_kept_pcs_than_the_minimum_is_refused(scanned) -> None:
    windows, local, noise = scanned
    many_local = np.column_stack([local + 1e-3 * noise[:, n] for n in range(16)] + [noise[:, :4]])
    with pytest.raises(InputError, match="fewer than the minimum 5") as refused:
        pc_check.check(windows, many_local)
    # build-inputs.md §8: the refusal names the dropped PCs, so a person can see which.
    message = str(refused.value)
    assert "keeps 4 PCs" in message
    assert "dropped as local: PC1, PC2, PC3" in message
    assert message.rstrip().endswith("PC16")


def test_too_few_pcs_with_none_local_says_none_was_dropped(small) -> None:
    windows, _, noise = small
    # Four noise PCs: none is local (as above), so the refusal has no PC to name.
    with pytest.raises(InputError, match="keeps 4 PCs, fewer than the minimum 5") as refused:
        pc_check.check(windows, noise[:, :4])
    assert str(refused.value).endswith("; none dropped as local")


def test_the_null_records_its_permutations_and_seed_as_integers(small) -> None:
    windows, _, noise = small
    null = pc_check.check(windows, noise[:, :5]).record["null"]
    assert (null["permutations"], null["seed"]) == (200, 405)
    assert all(type(null[key]) is int for key in ("permutations", "seed"))


def test_the_test_panel_s_scan_is_the_one_its_manifest_holds() -> None:
    # The manifest's scan was made by lane 2d's reference before ogld#5 adopted it; the adopted
    # module, given every passing row, selects the common ones itself and gives it again.
    v = testdata.variants()
    passing = np.flatnonzero(v.passes_filters)
    g = load_script("make_test_store").alt_counts(passing.astype(np.int64))
    windows = pc_check.windows(v.chrom[passing], v.pos[passing], g, threads=2)
    checked = pc_check.check(windows, testdata.pcs().values)
    held = testdata.manifest()["pc_check"]
    assert len(windows.where) == held["windows"] == 33
    assert checked.report().keys() == held.keys()
    for ours, theirs in zip(checked.record["scan"], held["scan"], strict=True):
        assert ours["window"] == theirs["window"]
        assert ours["local"] is theirs["local"]
        for key in ("window_share", "rest_share", "ratio", "window_null_q99", "rest_null_q99"):
            assert ours[key] == pytest.approx(theirs[key], rel=0, abs=1e-12), (ours["pc"], key)
    assert checked.covariates == list(held["covariates"])


def test_only_common_variants_are_scanned() -> None:
    chrom, pos, g = _genome(SMALL)
    rare = np.zeros_like(g[:1000])
    rare[:, :2] = 1  # MAF 1/60 < 0.05: these rows must not enter a window
    mixed = np.concatenate([g[:1000], rare])
    alone = pc_check.windows(chrom[:1000], pos[:1000], g[:1000])
    both = pc_check.windows(
        np.concatenate([chrom[:1000]] * 2), np.concatenate([pos[:1000]] * 2), mixed
    )
    assert pc_check.common(rare).sum() == 0
    assert alone.where == both.where
    np.testing.assert_allclose(np.abs(alone.basis), np.abs(both.basis), atol=1e-12)


def test_a_scan_without_windows_apart_from_the_best_is_refused() -> None:
    chrom, pos, g = _genome(SMALL)
    # One chromosome's first 4 Mb: two windows, starting 2 Mb apart, so each overlaps the other.
    on_one = (pos < 4_000_000) & (chrom == "1")
    windows = pc_check.windows(chrom[on_one], pos[on_one], g[on_one])
    assert len(windows.where) == 2
    noise = np.random.default_rng(3).standard_normal((SMALL, 5))
    with pytest.raises(InputError, match="rest share is undefined"):
        pc_check.check(windows, noise)


def test_a_scan_with_no_window_is_refused() -> None:
    chrom, pos, g = _genome(SMALL)
    few = np.flatnonzero(chrom == "1")[:4]
    windows = pc_check.windows(chrom[few], pos[few], g[few])
    assert windows.where == []
    with pytest.raises(InputError, match="no 4 Mb window"):
        pc_check.check(windows, np.random.default_rng(3).standard_normal((SMALL, 5)))


def test_a_window_of_rank_below_five_has_no_made_up_direction() -> None:
    # Five common variants, two identical: the window's genotypes span four directions, and its
    # basis holds those four and a zero column, not an arbitrary fifth (build-inputs §8).
    g = np.random.default_rng(5).binomial(2, 0.3, (5, PANEL)).astype(np.int8)
    g[4] = g[3]
    windows = pc_check.windows(np.full(5, "1"), np.arange(1, 6) * 1_000, g)
    assert len(windows.where) == 1
    lengths = np.sort(np.linalg.norm(windows.basis, axis=0))
    np.testing.assert_allclose(lengths, [0, 1, 1, 1, 1], atol=1e-9)
