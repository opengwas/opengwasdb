"""Locus boundaries from the association profile (prototypes/locus_leads, #409).

Rows are taken in the Locus Lead ranking. A row joins the strongest accepted
lead within 10 Mb whose col, the lowest envelope (max |z| of the 5e-4 tier per
bin of B bp, 0 in a bin with no tier row) over the bins from one to the other
inclusive, stays above the threshold; otherwise it becomes a lead.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from prototype_modules import load_prototype

profile = load_prototype("locus_leads", "profile")
rule = load_prototype("locus_leads", "rule")

BIN_BP = 10_000
HALF = profile.Threshold("relative", 0.5)


def _tier(position: list[int] | np.ndarray, abs_z: list[float] | np.ndarray,
          chromosome: str = "1") -> Any:
    position = np.asarray(position, dtype="int64")
    return profile.Profile(np.full(position.size, chromosome), position,
                           np.asarray(abs_z, dtype="float64"))


def _leads(position: list[int], abs_z: list[float], tier: Any,
           threshold: Any = HALF, bin_bp: int = BIN_BP) -> np.ndarray:
    result = profile.prominence_leads(
        np.full(len(position), "1"), np.array(position), np.array(abs_z),
        np.arange(len(position)), tier, bin_bp=bin_bp, threshold=threshold)
    lead_of: np.ndarray = result.lead_of
    return lead_of


def _bp_window_leads(position: list[int], abs_z: list[float]) -> np.ndarray:
    lead_of: np.ndarray = rule.locus_leads(
        np.full(len(position), "1"), np.array(position), np.zeros(len(position)),
        np.array(abs_z), np.arange(len(position)), d_cm=float("inf"), w_bp=1_000_000)
    return lead_of


def _envelope_between(tier: Any, start: int, end: int) -> list[float]:
    """The envelope of each bin from `start`'s to `end`'s, worked out row by row."""
    bins = range(start // BIN_BP, end // BIN_BP + 1)
    return [max((z for p, z in zip(tier.position, tier.abs_z, strict=True)
                 if p // BIN_BP == b), default=0.0) for b in bins]


def test_a_strong_peaks_shoulder_15_mb_out_stays_in_its_locus_where_a_1_mb_window_splits_it(
) -> None:
    # A |z| 100 peak at 5 Mb whose LD shadow carries |z| 40 rows every 5 kb out to a
    # |z| 12 shoulder at 6.5 Mb.
    position, abs_z = [5_000_000, 6_500_000], [100.0, 12.0]
    filler = np.arange(5_005_000, 6_500_000, 5_000)
    tier = _tier(np.r_[position, filler], np.r_[abs_z, np.full(filler.size, 40.0)])
    between = _envelope_between(tier, *position)
    assert position[1] - position[0] == 1_500_000, "the shoulder must lie beyond a 1 Mb window"
    assert min(between) == abs_z[1] > HALF.value * abs_z[1], "the profile must not dip between"
    np.testing.assert_array_equal(_bp_window_leads(position, abs_z), [0, 1])

    np.testing.assert_array_equal(_leads(position, abs_z, tier), [0, 0])


def test_two_peaks_400_kb_apart_across_an_empty_valley_stay_apart_where_a_1_mb_window_merges(
) -> None:
    position, abs_z = [20_000_000, 20_400_000], [15.0, 12.0]
    shoulders = np.r_[np.arange(19_950_000, 20_050_000, 5_000),
                      np.arange(20_355_000, 20_450_000, 5_000)]
    tier = _tier(np.r_[position, shoulders], np.r_[abs_z, np.full(shoulders.size, 8.0)])
    between = _envelope_between(tier, *position)
    assert position[1] - position[0] == 400_000, "the peaks must lie inside a 1 Mb window"
    assert min(between) == 0.0 and max(between) == 15.0, "an empty bin must separate them"
    np.testing.assert_array_equal(_bp_window_leads(position, abs_z), [0, 0])

    np.testing.assert_array_equal(_leads(position, abs_z, tier), [0, 1])


def _chain(span: list[int], abs_z: float, *, gaps: tuple[int, ...] = (),
           low: dict[int, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One tier row in every 10 kb bin across `span`, at the bin's midpoint: none in the
    bins of `gaps`, and the |z| of `low` in its bins."""
    low = low or {}
    bins = [b for b in range(min(span) // BIN_BP, max(span) // BIN_BP + 1) if b not in gaps]
    return (np.array([b * BIN_BP + BIN_BP // 2 for b in bins]),
            np.array([low.get(b, abs_z) for b in bins]))


def _with(position: list[int], abs_z: list[float],
          chain: tuple[np.ndarray, np.ndarray]) -> Any:
    return _tier(np.r_[position, chain[0]], np.r_[abs_z, chain[1]])


def test_the_col_runs_over_both_rows_bins_and_every_bin_between_and_no_further() -> None:
    # Bins 0 and 3 are empty; the rows sit at the inner edges of bins 1 and 2.
    tier = _tier([19_999, 20_000], [20.0, 10.0])
    assert _envelope_between(tier, 0, 39_999) == [0.0, 20.0, 10.0, 0.0]

    np.testing.assert_array_equal(_leads([19_999, 20_000], [20.0, 10.0], tier), [0, 0])
    tier = _tier([20_000, 19_999], [20.0, 10.0])
    np.testing.assert_array_equal(_leads([20_000, 19_999], [20.0, 10.0], tier), [0, 0])
    # One empty bin between them splits them, at 10,001 bp.
    tier = _tier([19_999, 30_000], [20.0, 10.0])
    np.testing.assert_array_equal(_leads([19_999, 30_000], [20.0, 10.0], tier), [0, 1])


def test_a_bin_with_no_tier_row_is_a_valley_of_zero_for_either_threshold() -> None:
    position, abs_z = [1_005_000, 1_055_000], [20.0, 10.0]
    empty = _with(position, abs_z, _chain(position, 30.0, gaps=(103,)))
    weak = _with(position, abs_z, _chain(position, 30.0, low={103: 3.5}))
    assert _envelope_between(empty, *position)[3] == 0.0
    assert _envelope_between(weak, *position)[3] == 3.5

    for threshold in (profile.Threshold("relative", 0.25), profile.Threshold("fixed", 0.0)):
        np.testing.assert_array_equal(_leads(position, abs_z, empty, threshold), [0, 1])
        np.testing.assert_array_equal(_leads(position, abs_z, weak, threshold), [0, 0])
    np.testing.assert_array_equal(_leads(position, abs_z, weak, HALF), [0, 1])


def test_a_relative_threshold_scales_with_the_candidates_own_abs_z_and_must_be_exceeded() -> None:
    position = [1_005_000, 1_055_000]
    chain = _chain(position, 30.0, low={103: 5.0})
    quarter = profile.Threshold("relative", 0.25)
    # 0.25 x 12 = 3 < 5: one locus. Scaled by the lead's 20 it would be 5, not above 5.
    tier = _with(position, [20.0, 12.0], chain)
    np.testing.assert_array_equal(_leads(position, [20.0, 12.0], tier, quarter), [0, 0])
    # 0.25 x 20 = 5: the col equals the threshold, so they split.
    tier = _with(position, [30.0, 20.0], chain)
    np.testing.assert_array_equal(_leads(position, [30.0, 20.0], tier, quarter), [0, 1])


def test_a_candidate_joins_the_strongest_lead_it_reaches_not_the_nearest() -> None:
    # A (|z| 50) and B (|z| 30) sit either side of a |z| 8 valley that B cannot cross
    # (0.5 x 30 = 15) but the weaker C, beside B, can (0.5 x 10 = 5).
    position, abs_z = [1_005_000, 1_205_000, 1_155_000], [50.0, 30.0, 10.0]
    tier = _with(position, abs_z, _chain([1_005_000, 1_205_000], 40.0, low={110: 8.0}))
    assert min(_envelope_between(tier, 1_005_000, 1_205_000)) == 8.0

    np.testing.assert_array_equal(_leads(position, abs_z, tier), [0, 1, 0])


def test_a_fixed_threshold_of_zero_merges_any_chain_of_occupied_bins_however_low() -> None:
    position, abs_z = [3_005_000, 5_005_000], [40.0, 30.0]
    tier = _with(position, abs_z, _chain(position, 3.5))
    assert min(_envelope_between(tier, *position)) == 3.5

    np.testing.assert_array_equal(
        _leads(position, abs_z, tier, profile.Threshold("fixed", 0.0)), [0, 0])
    np.testing.assert_array_equal(
        _leads(position, abs_z, tier, profile.Threshold("relative", 0.25)), [0, 1])


def test_no_merge_spans_more_than_10_mb() -> None:
    cases = ((10_000_000, [0, 0]), (10_000_001, [0, 1]))
    for (gap, expected), sign in itertools.product(cases, (1, -1)):
        # The lead at 15 Mb, its candidate either side of it.
        position, abs_z = [15_000_000, 15_000_000 + sign * gap], [40.0, 30.0]
        tier = _with(position, abs_z, _chain(position, 35.0))
        assert min(_envelope_between(tier, *sorted(position))) == 35.0

        np.testing.assert_array_equal(_leads(position, abs_z, tier), expected)


def _prominence(position: list[int], abs_z: list[float],
                tier: Any) -> Any:
    result: Any = profile.prominence_leads(
        np.full(len(position), "1"), np.array(position), np.array(abs_z),
        np.arange(len(position)), tier, bin_bp=BIN_BP, threshold=HALF)
    return result


def test_a_split_records_the_valley_against_its_nearest_lead_nearest_the_row_first() -> None:
    # A (bin 100) leads; B (bin 150) and C (bin 140) are split off. Bins 105, 120 and 145
    # hold no tier row; a row 20 Mb away has no lead within reach, so it is no split.
    position, abs_z = [1_005_000, 1_505_000, 1_405_000, 21_505_000], [30.0, 25.0, 10.0, 9.0]
    tier = _with(position, abs_z, _chain([1_005_000, 1_505_000], 40.0, gaps=(105, 120, 145)))
    assert [b for b, e in enumerate(_envelope_between(tier, 1_005_000, 1_505_000), 100)
            if e == 0.0] == [105, 120, 145]

    result = _prominence(position, abs_z, tier)

    np.testing.assert_array_equal(result.lead_of, [0, 1, 2, 3])
    splits = result.splits
    # B against A: of three empty bins, the one nearest B. C against B, the nearer lead.
    np.testing.assert_array_equal(splits.row, [1, 2])
    np.testing.assert_array_equal(splits.lead, [0, 1])
    np.testing.assert_array_equal(splits.bin, [145, 145])
    np.testing.assert_array_equal(splits.kind, ["no_row", "no_row"])


def test_a_valley_whose_lowest_bin_holds_a_tier_row_is_low_abs_z() -> None:
    position, abs_z = [1_005_000, 1_055_000], [20.0, 10.0]
    tier = _with(position, abs_z, _chain(position, 30.0, low={102: 4.0, 103: 3.5}))

    splits = _prominence(position, abs_z, tier).splits

    np.testing.assert_array_equal(splits.bin, [103])
    np.testing.assert_array_equal(splits.envelope, [3.5])
    np.testing.assert_array_equal(splits.kind, ["low_z"])
