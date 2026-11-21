"""Literal covariance oracle, including invalid-diagonal loss classes (#93)."""

import numpy as np

from ogld.build.importing.matrix import correlation, diagonal_mask


def test_covariance_becomes_a1_signed_correlation() -> None:
    # sqrt(4*1)=2, sqrt(4*9)=6, sqrt(1*9)=3; the middle A1 flips ALT.
    raw = np.array([[4, 1, -3], [1, 1, 0.75], [-3, 0.75, 9]])
    diagonal = np.array([4.0, 1, 9])
    signs = np.array([1, -1, 1])
    np.testing.assert_array_equal(
        correlation(raw, diagonal, diagonal, signs, signs),
        [[1, -0.5, -0.5], [-0.5, 1, -0.25], [-0.5, -0.25, 1]],
    )


def test_invalid_diagonal_drops_have_separate_loss_counts() -> None:
    mask, losses = diagonal_mask(np.array([4, 0, -1, np.nan, np.inf, -np.inf, 9]))
    assert mask.any()
    assert not mask.all()
    np.testing.assert_array_equal(mask, [True, False, False, False, False, False, True])
    assert losses == {"nonfinite_diagonal": 3, "nonpositive_diagonal": 2}
