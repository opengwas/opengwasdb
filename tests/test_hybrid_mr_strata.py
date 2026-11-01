"""Tests for the shared IVW estimate and the observed/imputed MR strata.

A stratum and the whole-set estimate are only comparable if they are the same
quantity, so both are `_hybrid_report.ivw` and the tests hold it to the formula
independently -- a weighted least-squares fit through the origin, computed here
with numpy rather than with the function under test. The split itself is tested
because a wrong one, or a stratum of two instruments reported as an estimate,
would be read as a finding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks import _hybrid_report


def _instruments(beta_exp: list[float], beta_out: list[float], se_out: list[float],
                 status_exp: str = "observed", status_out: str = "observed") -> list[dict]:
    return [
        {"beta_exp": bx, "beta_out": by, "se_out": so, "se_exp": 0.01,
         "status_exp": status_exp, "status_out": status_out}
        for bx, by, so in zip(beta_exp, beta_out, se_out, strict=True)
    ]


def test_ivw_recovers_a_proportional_effect():
    instruments = _instruments(
        beta_exp=[0.10, 0.20, 0.30], beta_out=[0.05, 0.10, 0.15], se_out=[0.01, 0.03, 0.02]
    )

    assert _hybrid_report.ivw(instruments)["beta"] == pytest.approx(0.5)


def test_ivw_se_and_p_match_a_weighted_least_squares_fit_through_the_origin():
    instruments = _instruments(
        beta_exp=[0.10, 0.20, 0.30, 0.05], beta_out=[0.10, 0.02, 0.20, 0.06],
        se_out=[0.01, 0.01, 0.02, 0.01],
    )
    design = np.array([[i["beta_exp"]] for i in instruments])
    outcome = np.array([i["beta_out"] for i in instruments])
    weights = 1.0 / np.array([i["se_out"] for i in instruments]) ** 2
    normal_matrix = design.T @ (design * weights[:, None])
    beta = float(np.linalg.solve(normal_matrix, design.T @ (outcome * weights))[0])
    se = float(np.sqrt(np.linalg.inv(normal_matrix)[0, 0]))

    residuals = (outcome - beta * design[:, 0]) * np.sqrt(weights)
    q_stat = float(residuals @ residuals)
    se_random = se * max(1.0, math.sqrt(q_stat / 3))

    estimate = _hybrid_report.ivw(instruments)
    assert estimate["beta"] == pytest.approx(beta)
    assert estimate["se_fixed"] == pytest.approx(se)
    assert estimate["cochran_q"] == pytest.approx(q_stat)
    assert estimate["q_df"] == 3
    assert estimate["se"] == pytest.approx(se_random)
    assert estimate["se"] > estimate["se_fixed"]  # these instruments are heterogeneous
    assert estimate["p"] == pytest.approx(math.erfc(abs(beta / se_random) / math.sqrt(2.0)))
    assert estimate["n"] == 4


def test_ivw_se_is_the_fixed_se_when_instruments_agree():
    """Q/df below 1 never shrinks the SE below the fixed-effect one."""
    instruments = _instruments(
        beta_exp=[0.10, 0.20, 0.30], beta_out=[0.05, 0.10, 0.15], se_out=[0.01, 0.03, 0.02]
    )
    estimate = _hybrid_report.ivw(instruments)
    assert estimate["cochran_q"] == pytest.approx(0.0, abs=1e-12)
    assert estimate["se"] == pytest.approx(estimate["se_fixed"])


def test_mr_strata_split_by_status_and_a_small_stratum_is_none():
    instruments = (
        _instruments([0.1] * 4, [0.05] * 4, [0.01] * 4, "observed", "observed")
        + _instruments([0.1] * 5, [0.05] * 5, [0.01] * 5, "imputed", "imputed")
        + _instruments([0.1] * 2, [0.05] * 2, [0.01] * 2, "observed", "imputed")
        + _instruments([0.1], [0.05], [0.01], "missing", "observed")
    )
    strata = _hybrid_report.mr_strata(instruments)

    assert sorted(strata) == ["both_imputed", "both_observed", "mixed"]
    assert strata["both_observed"]["n"] == 4
    assert strata["both_imputed"]["n"] == 5
    assert strata["mixed"] is None  # two instruments is not an estimate
    # The two whole-set estimates the strata are read against cover every row.
    assert _hybrid_report.ivw(instruments)["n"] == len(instruments)


def test_neglog10_p_stays_significant_where_a_naive_tail_would_not():
    """`1 - ndtr(z)` is already 0 at z = 40; the log-domain p is not."""
    from scipy.special import ndtr

    assert 1.0 - ndtr(40.0) == 0.0
    assert _hybrid_report.neglog10_p(40.0) == pytest.approx(350.0, abs=1.0)
