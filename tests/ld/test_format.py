"""The semantics core: how r is stored, which pairs a store keeps, and which requests it answers.

Every expected value is independent of the code under test. They are literals from the
spec's tables (`docs/spec/ld-store.md` §3, §8, §9), from the epic's t_far table, and from
the walkthroughs of `proto/1-ld-store-layout@65f94ca:prototypes/answer-semantics.html`,
which computed them in JavaScript. None is recomputed the way `opengwasdb.ld.format` computes it.
Hypothesis holds the properties: the round-trip bound, sign symmetry, and monotonicity at
the boundaries.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import opengwasdb.ld.format as fmt

# r in the closed interval the format stores.
R = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)
# t_far's range for any panel the formula admits: the 0.001 floor up to just below 1.
T = st.floats(min_value=0.001, max_value=0.999, allow_nan=False)


# ------------------------------------------------------------------ §3 quantisation

#: The scales the decisions name: HGDP+1kGP's, UK Biobank EUR's and the container's widest
#: (owner, 10 November 2026; ADR 0090).
SCALES = [511, 2047, 32767]


@pytest.mark.parametrize(("scale", "printed"), [(511, 9.78e-4), (2047, 2.44e-4), (32767, 1.53e-5)])
def test_the_step_and_the_error_bound_are_the_store_s_own(scale: int, printed: float) -> None:
    # The issue's table: max error in r 9.8×10⁻⁴ at × 511, 2.4×10⁻⁴ at × 2047.
    encoding = fmt.REncoding(scale)
    assert encoding.step == 1 / scale
    assert encoding.max_error == 1 / (2 * scale)
    assert pytest.approx(printed, rel=5e-3) == encoding.max_error  # printed to 3 figures


@pytest.mark.parametrize(
    ("scale", "r", "code"),
    [
        (511, 1.0, 511),
        (511, -1.0, -511),
        (511, 0.0, 0),
        # 0.5 × 511 is exactly 255.5: half to even gives 256, as at every odd scale.
        (511, 0.5, 256),
        (511, -0.5, -256),
        (2047, 1.0, 2047),
        (2047, 0.5, 1024),
        (32767, 1.0, 32767),
        (32767, -1.0, -32767),
        (32767, 0.5, 16384),
    ],
)
def test_quantise_literals(scale: int, r: float, code: int) -> None:
    out = fmt.REncoding(scale).quantise(r)
    assert out.dtype == np.dtype("int16")
    assert int(out) == code


@pytest.mark.parametrize("scale", SCALES)
def test_a_tie_rounds_to_the_even_code(scale: int) -> None:
    encoding = fmt.REncoding(scale)
    # 5 / (2 × scale) × scale is exactly 2.5: half to even gives 2, half away from zero 3.
    r = 5 / (2 * scale)
    assert r * scale == 2.5, "the fixture must be an exact tie for this to test anything"
    assert int(encoding.quantise(r)) == 2
    assert int(encoding.quantise(-r)) == -2
    # 3 / (2 × scale) × scale is exactly 1.5, whose even neighbour is above.
    assert 3 / (2 * scale) * scale == 1.5
    assert int(encoding.quantise(3 / (2 * scale))) == 2


def test_rounding_onto_a_threshold_matches_the_spec_examples() -> None:
    # §3 at × 32767: a computed r² of 0.0099994 is stored as code 3277 and reaches 0.01.
    widest = fmt.REncoding(32767)
    code = int(widest.quantise(math.sqrt(0.0099994)))
    assert code == 3277
    assert float(widest.dequantise(code)) ** 2 == pytest.approx(0.0100018, abs=5e-8)
    # §3 at × 511: a computed r² of 0.0101, above 0.01, is stored as code 51, which reads back
    # as 0.0099609 and does not reach it; 0.0102 is code 52, 0.0103554, and does.
    hgdp = fmt.REncoding(511)
    assert int(hgdp.quantise(math.sqrt(0.0101))) == 51
    assert float(hgdp.dequantise(51)) ** 2 == pytest.approx(0.0099609, abs=5e-8)
    assert int(hgdp.quantise(math.sqrt(0.0102))) == 52
    assert float(hgdp.dequantise(52)) ** 2 == pytest.approx(0.0103554, abs=5e-8)


@pytest.mark.parametrize("scale", SCALES)
def test_r_of_one_is_exact_at_every_scale(scale: int) -> None:
    encoding = fmt.REncoding(scale)
    codes = encoding.quantise(np.array([1.0, -1.0]))
    assert codes.tolist() == [scale, -scale]
    assert encoding.dequantise(codes).tolist() == [1.0, -1.0]


@pytest.mark.parametrize("scale", SCALES)
def test_the_reserved_code_is_never_written_and_never_read(scale: int) -> None:
    encoding = fmt.REncoding(scale)
    assert int(encoding.quantise(-1.0)) == -scale
    with pytest.raises(ValueError, match="-32768"):
        encoding.dequantise(np.array([0, -32768], dtype=np.int16))


@pytest.mark.parametrize("r", [math.nan, math.inf, -math.inf, 1.001, -1.5])
def test_a_value_that_is_not_a_correlation_is_refused_not_clamped(r: float) -> None:
    with pytest.raises(ValueError, match="not a correlation"):
        fmt.REncoding(511).quantise(np.array([0.1, r]))


@pytest.mark.parametrize("scale", SCALES)
def test_quantise_accepts_float_error_just_past_one(scale: int) -> None:
    # A dot product of standardised vectors can land a hair past ±1; the spec clamps it.
    assert int(fmt.REncoding(scale).quantise(1.0 + 1e-12)) == scale
    assert int(fmt.REncoding(scale).quantise(-1.0 - 1e-12)) == -scale


@pytest.mark.parametrize("scale", SCALES)
@given(r=R)
def test_round_trip_is_within_half_the_store_s_step(scale: int, r: float) -> None:
    encoding = fmt.REncoding(scale)
    back = float(encoding.dequantise(encoding.quantise(r)))
    assert abs(back - r) <= 1 / (2 * scale) * (1 + 1e-9)


@pytest.mark.parametrize("scale", SCALES)
@given(r=R)
def test_flipping_an_allele_negates_the_code(scale: int, r: float) -> None:
    encoding = fmt.REncoding(scale)
    assert int(encoding.quantise(-r)) == -int(encoding.quantise(r))


@pytest.mark.parametrize("scale", SCALES)
@given(a=R, b=R)
def test_quantise_is_monotone(scale: int, a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    encoding = fmt.REncoding(scale)
    assert int(encoding.quantise(lo)) <= int(encoding.quantise(hi))


# ------------------------------------------------------------------ §3, §7 the record

#: ld-store §7's `encoding` of the HGDP+1kGP stores.
HGDP = {"r_dtype": "int16", "r_scale": 511, "r_reserved_code": -32768, "rounding": "half_to_even"}


@pytest.mark.parametrize("scale", SCALES)
def test_an_encoding_reads_back_from_its_record(scale: int) -> None:
    record = fmt.REncoding(scale).record()
    assert record == {**HGDP, "r_scale": scale}
    assert fmt.REncoding.of(record) == fmt.REncoding(scale)


UNIMPLEMENTED: dict[str, object] = {
    "an int8 container": {**HGDP, "r_dtype": "int8"},
    "another reserved code": {**HGDP, "r_reserved_code": 0},
    "rounding half away from zero": {**HGDP, "rounding": "half_away_from_zero"},
    "a scale past the container": {**HGDP, "r_scale": 32768},
    "a scale of 0": {**HGDP, "r_scale": 0},
    "a fractional scale": {**HGDP, "r_scale": 511.5},
    "a scale written as text": {**HGDP, "r_scale": "511"},
    "a key ogld does not know": {**HGDP, "r_offset": 0},
    "no scale": {k: v for k, v in HGDP.items() if k != "r_scale"},
    "no record": None,
}


# ------------------------------------------------------------------ §3 t_far and its floor

# (n, k, the store's scale, t_far as the spec's §3 table prints it, far code floor, code of a
# pair at exactly t_far, whether that pair is stored, the true r² cut). From docs/spec/ld-store.md
# §3, each panel at its own scale (ADR 0090), on the samples each build keeps: the unrelated ones
# (ADR 0088, decision 11), EUR 665, AFR 608, EAS 724, SAS 672, AMR 341. The Test Panel keeps all
# 771 EUR members (owner's ruling), so the Test Store's rows are its own. The values were computed
# in 50-digit decimal arithmetic, outside ogld.
T_FAR_TABLE = [
    ("UK Biobank EUR (Pan-UKB)", 420542, 15, 2047, 0.001, 65, 65, True, 0.0009929),
    ("HGDP+1kGP EUR", 665, 0, 511, 0.0313855, 91, 91, True, 0.0313657),
    ("HGDP+1kGP EUR", 665, 10, 511, 0.0318654, 92, 91, False, 0.0320627),
    ("HGDP+1kGP AFR", 608, 0, 511, 0.0343328, 95, 95, True, 0.0341997),
    ("HGDP+1kGP AFR", 608, 10, 511, 0.0349079, 96, 95, False, 0.0349273),
    ("HGDP+1kGP EAS", 724, 0, 511, 0.0288243, 87, 87, True, 0.0286543),
    ("HGDP+1kGP EAS", 724, 10, 511, 0.0292286, 88, 87, False, 0.0293207),
    ("HGDP+1kGP SAS", 672, 0, 511, 0.0310581, 91, 90, False, 0.0313657),
    ("HGDP+1kGP SAS", 672, 10, 511, 0.0315280, 91, 91, True, 0.0313657),
    ("HGDP+1kGP AMR", 341, 0, 511, 0.0612941, 127, 127, True, 0.0612829),
    ("HGDP+1kGP AMR", 341, 10, 511, 0.0631515, 129, 128, False, 0.0632360),
    ("the Test Store (n = 771)", 771, 10, 511, 0.0274211, 85, 85, True, 0.0273446),
]


@pytest.mark.parametrize(
    ("panel", "n", "k", "scale", "printed", "floor", "exact_code", "stored", "cut"),
    T_FAR_TABLE,
    ids=[f"{row[0]}-k{row[2]}" for row in T_FAR_TABLE],
)
def test_t_far_and_its_floor_match_the_spec_table(
    panel: str,
    n: int,
    k: int,
    scale: int,
    printed: float,
    floor: int,
    exact_code: int,
    stored: bool,
    cut: float,
) -> None:
    del panel
    encoding = fmt.REncoding(scale)
    tf = fmt.t_far(n, k)
    assert tf == pytest.approx(printed, abs=5e-8)
    assert encoding.far_code_floor(tf) == floor
    exact = int(encoding.quantise(math.sqrt(tf)))
    assert exact == exact_code
    assert (exact >= floor) is stored
    # A true r² past the cut quantises to the floor; short of it, one code below. The cut is
    # printed to 7 places, so 1×10⁻⁷ either side is clear of it.
    assert int(encoding.quantise(math.sqrt(cut + 1e-7))) >= floor
    assert int(encoding.quantise(math.sqrt(cut - 1e-7))) < floor


# The floors at × 32767, the one scale before ogld#16: the floor is the encoding's, so one t_far
# gives each scale its own. 5850 is what the real EUR chr22 build at × 32767 records (ld-store §6);
# the rows at 771, 816 and 345 are ld-store §3's former table, on all of each panel's v2 members.
WIDEST_FLOORS = [
    (49766, 10, 1037),
    (665, 10, 5850),
    (771, 0, 5391),
    (771, 10, 5426),
    (816, 10, 5273),
    (345, 0, 8066),
]


@pytest.mark.parametrize(("n", "k", "floor"), WIDEST_FLOORS)
def test_the_floor_at_the_widest_scale_is_the_former_table_s(n: int, k: int, floor: int) -> None:
    assert fmt.REncoding(32767).far_code_floor(fmt.t_far(n, k)) == floor


def test_the_epic_s_t_far_table_at_k_10() -> None:
    # Epic opengwas/ogld#1, "Decisions already taken", decision 3: rounded to the epic's digits.
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
    # §12 provenance.threshold: n − k − 1 > q, so t_far < 1; and k counts covariates, so k ≥ 0.
    with pytest.raises(ValueError, match="n"):
        fmt.t_far(n, k)


def test_a_pair_at_the_floor_is_stored_and_one_code_below_is_the_bound() -> None:
    # "A pair at exactly t_far" on UK Biobank EUR at × 2047: t_far = 0.001, floor 65, and a true
    # r² of 0.0009929, below t_far, already reaches it.
    ukb = fmt.REncoding(2047)
    assert ukb.far_code_floor(0.001) == 65
    assert int(ukb.quantise(math.sqrt(0.0009928))) == 64  # the Bound
    assert int(ukb.quantise(math.sqrt(0.0009930))) == 65  # stored
    # And at × 32767, where the walkthrough was first done: floor 1037.
    widest = fmt.REncoding(32767)
    assert int(widest.quantise(math.sqrt(0.0010006))) == 1036  # still the Bound
    assert int(widest.quantise(math.sqrt(0.0010007))) == 1037  # stored
    assert widest.far_code_floor(0.001) == 1037


@pytest.mark.parametrize("scale", SCALES)
def test_far_code_floor_at_the_ends(scale: int) -> None:
    assert fmt.REncoding(scale).far_code_floor(0.0) == 0
    assert fmt.REncoding(scale).far_code_floor(1.0) == scale


@pytest.mark.parametrize("t", [-0.1, 1.01, math.nan])
def test_far_code_floor_refuses_a_value_that_is_not_an_r_squared(t: float) -> None:
    with pytest.raises(ValueError, match="r²"):
        fmt.REncoding(511).far_code_floor(t)


@pytest.mark.parametrize("scale", SCALES)
@given(t=T)
def test_the_floor_is_the_smallest_code_that_reaches_t(scale: int, t: float) -> None:
    floor = fmt.REncoding(scale).far_code_floor(t)
    assert (floor / scale) ** 2 >= t
    assert ((floor - 1) / scale) ** 2 < t


@pytest.mark.parametrize("scale", SCALES)
@given(a=T, b=T)
def test_the_floor_is_monotone_in_t(scale: int, a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    encoding = fmt.REncoding(scale)
    assert encoding.far_code_floor(lo) <= encoding.far_code_floor(hi)


@pytest.mark.parametrize("scale", SCALES)
@given(r=R, t=T)
def test_every_r_whose_code_reaches_the_floor_is_above_least_r(
    scale: int, r: float, t: float
) -> None:
    # The builder judges only |r| ≥ least_r(floor): no pair it skips could have reached it.
    encoding = fmt.REncoding(scale)
    floor = encoding.far_code_floor(t)
    if abs(int(encoding.quantise(r))) >= floor:
        assert abs(r) >= encoding.least_r(floor)


# ------------------------------------------------------------------ §8 pair answers

#: t_far at k = 10 on the samples the EUR and AMR builds keep (ADR 0088, decision 11).
EUR_K10 = fmt.t_far(665, 10)
#: The HGDP+1kGP stores' encoding, in which §8's walkthroughs read r (ADR 0090).
HGDP_R = fmt.REncoding(511)


def stored(r: float) -> float:
    """r as a × 511 store holds it."""
    return float(HGDP_R.dequantise(HGDP_R.quantise(r)))


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


@pytest.mark.parametrize("scale", SCALES)
@given(r=R, tf=T, d=st.integers(min_value=3_000_001, max_value=10_000_000))
def test_a_far_pair_is_stored_exactly_when_its_code_reaches_the_floor(
    scale: int, r: float, tf: float, d: int
) -> None:
    encoding = fmt.REncoding(scale)
    code = int(encoding.quantise(r))
    held = kind(d, code / scale, tf) is fmt.AnswerKind.STORED
    assert held == (abs(code) >= encoding.far_code_floor(tf))


# ------------------------------------------------------------------ §9 refusals


def judge(
    r2: float,
    kb: float,
    tf: float = EUR_K10,
    far_bp: int = fmt.FAR_BP,
    t_partner: float | None = None,
) -> fmt.Decision:
    """The judgement `clump` and `proxies` make: with a store's t_partner it is
    `judge_partner_request`, and without one `judge_request`, which is a 0.1.0 store's table."""
    if t_partner is not None:
        return fmt.judge_partner_request(
            r2=r2,
            kb=kb,
            t_far=tf,
            near_bp=fmt.NEAR_BP,
            far_bp=far_bp,
            t_partner=t_partner,
            store="a store",
        )
    return fmt.judge_request(r2=r2, kb=kb, t_far=tf, near_bp=fmt.NEAR_BP, far_bp=far_bp)


UKB = fmt.t_far(49766, 10)
