"""The in-memory SE measurement shares the Dense sampling contract (issue #147).

`fit_se` -- the measurement behind Ragged stores and the Hybrid Overflow
Component -- had the same shape as the old Dense survey: for each of three
candidate ranges it coded every cell and compressed the result, plus a
`float16` baseline. It samples through the same `sample_chunks` rule as the
Dense path (#146), so a Ragged store and a Dense store cannot get different
answers from two different sampling implementations.
"""

from __future__ import annotations

import numpy as np

from opengwasdb.encoding.measure import (
    SeMeasurementRecord,
    fit_se,
    fit_se_grid,
)
from opengwasdb.encoding.plan import (
    EncodingMeasurements,
    SeEncoding,
    SeMeasurements,
    StoreEncoding,
)


def _decide(measured: SeMeasurements) -> SeEncoding:
    return StoreEncoding.decide(EncodingMeasurements(1, se=measured)).se


def _csr_data(*, n: int, well_fitted: bool, seed: int = 3):
    """A flat association sequence whose SE either follows or ignores its EAF."""
    rng = np.random.default_rng(seed)
    eaf = rng.uniform(0.05, 0.95, n).astype(np.float32)
    predictor = np.log(2 * eaf.astype(np.float64) * (1 - eaf.astype(np.float64)))
    if well_fitted:
        se = np.exp(-3.0 - 0.5 * predictor + 0.05 * rng.standard_normal(n)).astype(np.float32)
    else:
        se = np.exp(rng.normal(-3.0, 2.0, n)).astype(np.float32)
    ai = np.zeros(n, dtype=np.int64)
    return se, eaf, ai


def test_fit_se_sampled_matches_exhaustive_on_csr(tmp_path=None) -> None:
    se, eaf, ai = _csr_data(n=400_000, well_fitted=True)

    full = fit_se(se, eaf, ai, n_analyses=1, chunks=10_000, measure_max_chunks=10**6)[1]
    sampled = fit_se(se, eaf, ai, n_analyses=1, chunks=10_000, measure_max_chunks=4)[1]

    assert _decide(full).is_residual
    assert _decide(sampled) == _decide(full)


def test_fit_se_sampled_catches_a_bad_analysis(tmp_path=None) -> None:
    """Sampling must not hide the GCST007320 case from the Ragged path."""
    se, eaf, ai = _csr_data(n=400_000, well_fitted=False)

    full = fit_se(se, eaf, ai, n_analyses=1, chunks=10_000, measure_max_chunks=10**6)[1]
    sampled = fit_se(se, eaf, ai, n_analyses=1, chunks=10_000, measure_max_chunks=4)[1]

    assert _decide(full) == SeEncoding("float16")
    assert _decide(sampled) == _decide(full)


def test_fit_se_grid_sampled_matches_exhaustive(tmp_path=None) -> None:
    """The in-memory grid shape (tiny Dense builder) shares the same rule."""
    rng = np.random.default_rng(11)
    n_rows, n_cols = 200_000, 1
    eaf = rng.uniform(0.05, 0.95, (n_rows, n_cols)).astype(np.float32)
    predictor = np.log(2 * eaf.astype(np.float64) * (1 - eaf.astype(np.float64)))
    se = np.exp(-3.0 - 0.5 * predictor + 0.05 * rng.standard_normal(eaf.shape)).astype(np.float32)

    full = fit_se_grid(se, eaf, chunks=(10_000, 1), measure_max_chunks=10**6)[1]
    sampled = fit_se_grid(se, eaf, chunks=(10_000, 1), measure_max_chunks=4)[1]

    assert _decide(full).is_residual
    assert _decide(sampled) == _decide(full)


def test_fit_se_record_says_what_was_measured() -> None:
    se, eaf, ai = _csr_data(n=400_000, well_fitted=True)
    record = SeMeasurementRecord()

    fit_se(se, eaf, ai, n_analyses=1, chunks=10_000, record=record, measure_max_chunks=4)

    assert record.csr is not None
    assert record.csr.sampled_chunks == 4
    assert record.csr.total_chunks == 40
    assert record.to_manifest() == {
        "dense": "none",
        "csr": {"sampled_chunks": 4, "total_chunks": 40},
    }
