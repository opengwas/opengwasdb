"""The semantics core: how r is stored, which pairs a store keeps, and which requests it answers.

Every expected value is independent of the code under test. They are literals from the
spec's tables (`docs/spec/ld-store.md` §3, §8, §9), from the epic's t_far table, and from
the walkthroughs of `proto/1-ld-store-layout@65f94ca:prototypes/answer-semantics.html`,
which computed them in JavaScript. None is recomputed the way `ogld.format` computes it.
Hypothesis holds the properties: the round-trip bound, sign symmetry, and monotonicity at
the boundaries.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import ogld.format as fmt

# r in the closed interval the format stores.
R = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)
# t_far's range for any panel the formula admits: the 0.001 floor up to just below 1.
T = st.floats(min_value=0.001, max_value=0.999, allow_nan=False)


# ------------------------------------------------------------------ §3 quantisation


def test_the_scale_reserved_code_and_error_bound_are_the_spec_s() -> None:
    assert fmt.R_SCALE == 32767
    assert fmt.RESERVED_CODE == -32768
    assert pytest.approx(1.53e-5, abs=5e-8) == fmt.QUANTISATION_ERROR
    assert fmt.QUANTISATION_ERROR == 1 / 65534


@pytest.mark.parametrize(
    ("r", "code"),
    [(1.0, 32767), (-1.0, -32767), (0.0, 0), (0.5, 16384), (-0.5, -16384)],
)
def test_quantise_literals(r: float, code: int) -> None:
    out = fmt.quantise(r)
    assert out.dtype == np.int16
    assert int(out) == code


def test_a_tie_rounds_to_the_even_code() -> None:
    # 5 / 65534 × 32767 is exactly 2.5: half to even gives 2, half away from zero 3.
    r = 5 / 65534
    assert r * 32767 == 2.5, "the fixture must be an exact tie for this to test anything"
    assert int(fmt.quantise(r)) == 2
    assert int(fmt.quantise(-r)) == -2
    # 3 / 65534 × 32767 is exactly 1.5, whose even neighbour is above.
    assert 3 / 65534 * 32767 == 1.5
    assert int(fmt.quantise(3 / 65534)) == 2


def test_rounding_onto_a_threshold_matches_the_spec_example() -> None:
    # §3: a computed r² of 0.0099994 is stored as code 3277 and reads back as 0.0100018.
    code = int(fmt.quantise(math.sqrt(0.0099994)))
    assert code == 3277
    assert float(fmt.dequantise(code)) ** 2 == pytest.approx(0.0100018, abs=5e-8)


def test_the_reserved_code_is_never_written_and_never_read() -> None:
    assert int(fmt.quantise(-1.0)) == -32767
    with pytest.raises(ValueError, match="-32768"):
        fmt.dequantise(np.array([0, fmt.RESERVED_CODE], dtype=np.int16))


@pytest.mark.parametrize("r", [math.nan, math.inf, -math.inf, 1.001, -1.5])
def test_a_value_that_is_not_a_correlation_is_refused_not_clamped(r: float) -> None:
    with pytest.raises(ValueError, match="not a correlation"):
        fmt.quantise(np.array([0.1, r]))


def test_quantise_accepts_float_error_just_past_one() -> None:
    # A dot product of standardised vectors can land a hair past ±1; the spec clamps it.
    assert int(fmt.quantise(1.0 + 1e-12)) == 32767
    assert int(fmt.quantise(-1.0 - 1e-12)) == -32767


def test_dequantise_returns_float64() -> None:
    out = fmt.dequantise(np.array([32767, -32767, 0], dtype=np.int16))
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, -1.0, 0.0]


@given(R)
def test_round_trip_is_within_half_a_step(r: float) -> None:
    assert abs(float(fmt.dequantise(fmt.quantise(r))) - r) <= fmt.QUANTISATION_ERROR * (1 + 1e-9)


@given(R)
def test_flipping_an_allele_negates_the_code(r: float) -> None:
    assert int(fmt.quantise(-r)) == -int(fmt.quantise(r))


@given(R, R)
def test_quantise_is_monotone(a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    assert int(fmt.quantise(lo)) <= int(fmt.quantise(hi))


# ------------------------------------------------------------------ §3 t_far and its floor

# (n, k, t_far as the spec's §3 table prints it, far code floor, code of a pair at exactly
# t_far, whether that pair is stored). From docs/spec/ld-store.md §3 and
# proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/t_far_edge.tsv.
T_FAR_TABLE = [
    ("UK Biobank EUR", 49766, 0, 0.001, 1037, 1036, False),
    ("UK Biobank EUR", 49766, 10, 0.001, 1037, 1036, False),
    ("HGDP+1kGP EUR", 771, 0, 0.0270649, 5391, 5391, True),
    ("HGDP+1kGP EUR", 771, 10, 0.0274211, 5426, 5426, True),
    ("HGDP+1kGP AFR", 804, 0, 0.0259527, 5279, 5279, True),
    ("HGDP+1kGP AFR", 804, 10, 0.0262799, 5312, 5312, True),
    ("HGDP+1kGP EAS", 817, 0, 0.0255392, 5237, 5236, False),
    ("HGDP+1kGP EAS", 817, 10, 0.0258561, 5269, 5269, True),
    ("HGDP+1kGP SAS", 783, 0, 0.0266496, 5350, 5349, False),
    ("HGDP+1kGP SAS", 783, 10, 0.0269948, 5384, 5384, True),
    ("HGDP+1kGP AMR", 345, 0, 0.0605814, 8066, 8065, False),
    ("HGDP+1kGP AMR", 345, 10, 0.0623952, 8185, 8185, True),
]


@pytest.mark.parametrize(
    ("panel", "n", "k", "printed", "floor", "exact_code", "stored"),
    T_FAR_TABLE,
    ids=[f"{row[0]}-k{row[2]}" for row in T_FAR_TABLE],
)
def test_t_far_and_its_floor_match_the_spec_table(
    panel: str, n: int, k: int, printed: float, floor: int, exact_code: int, stored: bool
) -> None:
    del panel
    tf = fmt.t_far(n, k)
    assert tf == pytest.approx(printed, abs=5e-8)
    assert fmt.far_code_floor(tf) == floor
    exact = int(fmt.quantise(math.sqrt(tf)))
    assert exact == exact_code
    assert (exact >= floor) is stored


def test_the_epic_s_t_far_table_at_k_10() -> None:
    # Epic #1, "Decisions already taken", decision 3: rounded to the epic's digits.
    printed = {49766: 0.001, 771: 0.027, 804: 0.026, 817: 0.026, 783: 0.027, 345: 0.062}
    for n, value in printed.items():
        assert round(fmt.t_far(n, 10), 3) == value


def test_the_constants_of_t_far() -> None:
    assert fmt.T_FAR_Q == 20.84
    assert fmt.T_FAR_MIN == 0.001
    # The floor binds from n − k − 1 ≥ 20,840: UK Biobank EUR's raw value is 0.00042.
    assert fmt.t_far(49766, 10) == 0.001
    assert fmt.t_far(20841 + 10, 10) == 0.001


@pytest.mark.parametrize(("n", "k"), [(21, 0), (31, 10), (10, 20), (771, -1), (0, 0)])
def test_a_panel_too_small_for_t_far_below_one_is_refused(n: int, k: int) -> None:
    # §12 far.threshold: n − k − 1 > q, so t_far < 1; and k counts covariates, so k ≥ 0.
    with pytest.raises(ValueError, match="n"):
        fmt.t_far(n, k)


def test_a_pair_at_the_floor_is_stored_and_one_code_below_is_the_bound() -> None:
    # "A pair at exactly t_far" walkthrough: UK Biobank, k = 10.
    assert int(fmt.quantise(math.sqrt(0.0010006))) == 1036  # still the Bound
    assert int(fmt.quantise(math.sqrt(0.0010007))) == 1037  # stored
    assert fmt.far_code_floor(0.001) == 1037


@pytest.mark.parametrize(("t", "floor"), [(0.0, 0), (1.0, 32767)])
def test_far_code_floor_at_the_ends(t: float, floor: int) -> None:
    assert fmt.far_code_floor(t) == floor


@pytest.mark.parametrize("t", [-0.1, 1.01, math.nan])
def test_far_code_floor_refuses_a_value_that_is_not_an_r_squared(t: float) -> None:
    with pytest.raises(ValueError, match="r²"):
        fmt.far_code_floor(t)


@given(T)
def test_the_floor_is_the_smallest_code_that_reaches_t(t: float) -> None:
    floor = fmt.far_code_floor(t)
    assert (floor / 32767) ** 2 >= t
    assert ((floor - 1) / 32767) ** 2 < t


@given(T, T)
def test_the_floor_is_monotone_in_t(a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    assert fmt.far_code_floor(lo) <= fmt.far_code_floor(hi)


# ------------------------------------------------------------------ §8 pair answers

EUR_K10 = fmt.t_far(771, 10)
AMR_K10 = fmt.t_far(345, 10)


def kind(distance: int, r_hat: float, tf: float = EUR_K10, *, same: bool = True) -> fmt.AnswerKind:
    return fmt.answer_kind(
        same_chromosome=same,
        distance=distance,
        r_hat=r_hat,
        t_far=tf,
        near_bp=fmt.NEAR_BP,
        far_bp=fmt.FAR_BP,
    )


def test_the_default_distances_are_adr_0001_s() -> None:
    assert fmt.NEAR_BP == 3_000_000
    assert fmt.FAR_BP == 10_000_000


def test_exactly_3_mb_is_the_near_band_and_one_more_base_is_not() -> None:
    # The walkthrough "Exactly 3 Mb", EUR, k = 10.
    weak = float(fmt.dequantise(fmt.quantise(0.05)))
    assert kind(3_000_000, weak) is fmt.AnswerKind.STORED
    assert kind(3_000_001, weak) is fmt.AnswerKind.BOUND
    assert kind(3_000_001, float(fmt.dequantise(fmt.quantise(0.30)))) is fmt.AnswerKind.STORED


def test_exactly_10_mb_is_measured_and_one_more_base_is_not() -> None:
    strong = float(fmt.dequantise(fmt.quantise(0.30)))
    assert kind(10_000_000, strong) is fmt.AnswerKind.STORED
    assert kind(10_000_001, strong) is fmt.AnswerKind.NOT_MEASURED
    assert kind(10_000_001, 1.0) is fmt.AnswerKind.NOT_MEASURED


def test_another_chromosome_is_not_measured_at_any_distance() -> None:
    for distance in (0, 1, 3_000_000, 5_000_000):
        assert kind(distance, 1.0, same=False) is fmt.AnswerKind.NOT_MEASURED


def test_two_records_at_one_position_are_a_near_band_pair() -> None:
    assert kind(0, 0.0) is fmt.AnswerKind.STORED


def test_r_squared_equal_to_t_far_is_stored_one_code_below_is_the_bound() -> None:
    # The walkthrough "r² equal to t_far": EUR k = 10's floor is 5426 (§3).
    assert kind(5_000_000, 5426 / 32767) is fmt.AnswerKind.STORED
    assert kind(5_000_000, 5425 / 32767) is fmt.AnswerKind.BOUND
    assert kind(5_000_000, -5426 / 32767) is fmt.AnswerKind.STORED


def test_amr_s_t_far_bounds_a_pair_eur_stores() -> None:
    # The walkthrough "AMR's t_far of 0.062": r = 0.24 (r² ≈ 0.0576), 5 Mb apart.
    r_hat = float(fmt.dequantise(fmt.quantise(0.24)))
    assert kind(5_000_000, r_hat, AMR_K10) is fmt.AnswerKind.BOUND
    assert kind(5_000_000, r_hat, EUR_K10) is fmt.AnswerKind.STORED


def test_a_store_that_measured_less_than_10_mb_says_so() -> None:
    # An imported store may record far_bp below 10 Mb (spec §7); the reader uses the record.
    out = fmt.answer_kind(
        same_chromosome=True,
        distance=9_000_000,
        r_hat=1.0,
        t_far=0.001,
        near_bp=3_000_000,
        far_bp=8_000_000,
    )
    assert out is fmt.AnswerKind.NOT_MEASURED


@pytest.mark.parametrize("bad", [{"distance": -1}, {"r_hat": math.nan}, {"r_hat": 1.5}])
def test_answer_kind_refuses_impossible_inputs(bad: dict[str, float]) -> None:
    args: dict[str, float] = {"distance": 5_000_000, "r_hat": 0.5} | bad
    with pytest.raises(ValueError, match=r"distance|r̂"):
        fmt.answer_kind(
            same_chromosome=True,
            distance=int(args["distance"]),
            r_hat=args["r_hat"],
            t_far=EUR_K10,
            near_bp=fmt.NEAR_BP,
            far_bp=fmt.FAR_BP,
        )


@given(R, T, st.integers(min_value=3_000_001, max_value=10_000_000))
def test_a_far_pair_is_stored_exactly_when_its_code_reaches_the_floor(
    r: float, tf: float, d: int
) -> None:
    code = int(fmt.quantise(r))
    stored = kind(d, code / 32767, tf) is fmt.AnswerKind.STORED
    assert stored == (abs(code) >= fmt.far_code_floor(tf))


@given(R, T, st.integers(min_value=0, max_value=12_000_000), st.booleans())
def test_the_answer_does_not_depend_on_the_sign(r: float, tf: float, d: int, same: bool) -> None:
    r_hat = float(fmt.dequantise(fmt.quantise(r)))
    assert kind(d, r_hat, tf, same=same) is kind(d, -r_hat, tf, same=same)


def test_answer_kinds_reads_the_kind_of_every_cell_of_a_matrix() -> None:
    # Five variants: three on chr1 at 1, 2, and 4.5 Mb and 12 Mb, one on chr2.
    chrom = np.array(["1", "1", "1", "1", "2"])
    position = np.array([1_000_000, 2_000_000, 4_500_000, 12_000_000, 1_000_000])
    nan = math.nan
    r = np.array(
        [
            [1.0, 0.2, nan, nan, nan],
            [0.2, 1.0, 0.3, nan, nan],
            [nan, 0.3, 1.0, nan, nan],
            [nan, nan, nan, 1.0, nan],
            [nan, nan, nan, nan, 1.0],
        ]
    )
    s, b, m = (
        fmt.AnswerKind.STORED.value,
        fmt.AnswerKind.BOUND.value,
        fmt.AnswerKind.NOT_MEASURED.value,
    )
    expected = [
        [s, s, b, m, m],  # 1→4.5 Mb is 3.5 Mb: unstored, so the Bound; 1→12 Mb is 11 Mb
        [s, s, s, b, m],  # 2→12 Mb is 10 Mb exactly: unstored, so the Bound
        [b, s, s, b, m],  # 4.5→12 Mb is 7.5 Mb
        [m, b, b, s, m],
        [m, m, m, m, s],
    ]
    out = fmt.answer_kinds(
        r, chrom=chrom, position=position, t_far=EUR_K10, near_bp=fmt.NEAR_BP, far_bp=fmt.FAR_BP
    )
    assert out.tolist() == expected


def test_answer_kinds_refuses_an_unstored_near_band_pair() -> None:
    # Every pair within near_bp is stored, so a NaN there is a broken read, not an answer.
    with pytest.raises(ValueError, match="Near Band"):
        fmt.answer_kinds(
            np.array([[1.0, math.nan], [math.nan, 1.0]]),
            chrom=np.array(["1", "1"]),
            position=np.array([10, 20]),
            t_far=EUR_K10,
            near_bp=fmt.NEAR_BP,
            far_bp=fmt.FAR_BP,
        )


def test_answer_kinds_after_a_fill_still_reads_the_bound() -> None:
    # fill = 0 replaces NaN. A Far Pair has r̂² ≥ t_far, so a 3–10 Mb cell below t_far,
    # 0 included, is the Bound; and every Near Band pair is stored, so a 0 there is r.
    out = fmt.answer_kinds(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.2], [0.0, 0.2, 1.0]]),
        chrom=np.array(["1", "1", "1"]),
        position=np.array([0, 5_000_000, 5_000_100]),
        t_far=EUR_K10,
        near_bp=fmt.NEAR_BP,
        far_bp=fmt.FAR_BP,
    )
    s, b = fmt.AnswerKind.STORED.value, fmt.AnswerKind.BOUND.value
    assert out.tolist() == [[s, b, b], [b, s, s], [b, s, s]]


# ------------------------------------------------------------------ §9 refusals


def judge(r2: float, kb: float, tf: float = EUR_K10, far_bp: int = fmt.FAR_BP) -> fmt.Decision:
    return fmt.judge_request(r2=r2, kb=kb, t_far=tf, near_bp=fmt.NEAR_BP, far_bp=far_bp)


UKB = fmt.t_far(49766, 10)


@pytest.mark.parametrize(
    ("r2", "kb", "tf", "refusal"),
    [
        # "Exactly 10 Mb"
        (0.1, 10_000, EUR_K10, None),
        (0.1, 10_000.001, EUR_K10, fmt.RefusalReason.WINDOW_OVER_FAR),
        # "A window over 10 Mb": refused on every panel, whatever r2 is
        (0.8, 10_001, EUR_K10, fmt.RefusalReason.WINDOW_OVER_FAR),
        (0.8, 10_001, UKB, fmt.RefusalReason.WINDOW_OVER_FAR),
        (0.001, 10_000, UKB, None),
        (0.001, 10_000, EUR_K10, fmt.RefusalReason.THRESHOLD_BELOW_T_FAR),
        # "r² equal to t_far"
        (EUR_K10, 5_000, EUR_K10, None),
        (EUR_K10 - 1e-6, 5_000, EUR_K10, fmt.RefusalReason.THRESHOLD_BELOW_T_FAR),
        # "AMR's t_far of 0.062"
        (0.05, 5_000, AMR_K10, fmt.RefusalReason.THRESHOLD_BELOW_T_FAR),
        (0.05, 5_000, EUR_K10, None),
        (0.05, 3_000, AMR_K10, None),
        # §9: W ≤ 3 Mb is answered from the Near Band at any r2
        (0.0, 3_000, EUR_K10, None),
        (0.0, 3_000.001, EUR_K10, fmt.RefusalReason.THRESHOLD_BELOW_T_FAR),
        (1.0, 0, EUR_K10, None),
        # §9: invalid
        (-0.01, 1_000, EUR_K10, fmt.RefusalReason.INVALID),
        (1.01, 1_000, EUR_K10, fmt.RefusalReason.INVALID),
        (math.nan, 1_000, EUR_K10, fmt.RefusalReason.INVALID),
        (0.5, -1, EUR_K10, fmt.RefusalReason.INVALID),
        (0.5, math.nan, EUR_K10, fmt.RefusalReason.INVALID),
        (0.5, math.inf, EUR_K10, fmt.RefusalReason.INVALID),
    ],
)
def test_the_refusal_table(
    r2: float, kb: float, tf: float, refusal: fmt.RefusalReason | None
) -> None:
    decision = judge(r2, kb, tf)
    assert decision.refusal is refusal
    assert decision.answered is (refusal is None)


def test_windows_are_compared_with_the_distance_the_store_measured() -> None:
    # A store whose far_bp is 8 Mb refuses a 9 Mb window that a 10 Mb store answers.
    assert judge(0.5, 9_000, far_bp=10_000_000).answered
    assert judge(0.5, 9_000, far_bp=8_000_000).refusal is fmt.RefusalReason.WINDOW_OVER_FAR


def test_a_refusal_names_its_reason() -> None:
    decision = judge(0.001, 10_000)
    with pytest.raises(fmt.Refusal) as caught:
        decision.raise_if_refused()
    assert caught.value.reason is fmt.RefusalReason.THRESHOLD_BELOW_T_FAR
    assert "t_far" in str(caught.value)
    judge(0.5, 1_000).raise_if_refused()  # an answered decision raises nothing


def test_an_answered_decision_carries_its_window_in_bp() -> None:
    assert judge(0.8, 10_000).window_bp == 10_000_000
    assert judge(0.8, 2.5).window_bp == 2_500


@given(
    st.floats(min_value=0, max_value=1),
    st.floats(min_value=0, max_value=1),
    st.floats(min_value=0, max_value=12_000),
    st.floats(min_value=0, max_value=12_000),
    st.sampled_from([EUR_K10, AMR_K10, UKB]),
)
def test_a_stricter_request_is_answered_whenever_a_looser_one_is(
    r2_a: float, r2_b: float, kb_a: float, kb_b: float, tf: float
) -> None:
    # Raising r2 or narrowing the window never turns an answered request into a refusal.
    loose_r2, strict_r2 = sorted((r2_a, r2_b))
    narrow, wide = sorted((kb_a, kb_b))
    if judge(loose_r2, wide, tf).answered:
        assert judge(strict_r2, narrow, tf).answered


@pytest.mark.parametrize(
    ("radius", "far_bp", "refusal"),
    [
        (5_000_000, 10_000_000, None),
        (5_000_001, 10_000_000, fmt.RefusalReason.REGION_OVER_FAR),
        (0, 10_000_000, None),
        (4_000_001, 8_000_000, fmt.RefusalReason.REGION_OVER_FAR),
        (-1, 10_000_000, fmt.RefusalReason.INVALID),
    ],
)
def test_a_region_is_at_most_the_far_distance_wide(
    radius: int, far_bp: int, refusal: fmt.RefusalReason | None
) -> None:
    # Owner, 7 November 2026: a region's width, 2 × radius, is capped at 10 Mb.
    assert fmt.judge_region(radius=radius, far_bp=far_bp).refusal is refusal
