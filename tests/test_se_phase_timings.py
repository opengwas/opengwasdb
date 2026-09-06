from __future__ import annotations

import time

import numpy as np
import pytest
import zarr

from opengwasdb.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding
from opengwasdb.encoding.se import optimise_dense_se_joint
from opengwasdb.encoding.timing import PhaseTimer


def _preliminary() -> StoreEncoding:
    return StoreEncoding(
        z=ZEncoding("float16"),
        se=SeEncoding("float16"),
        eaf=EafEncoding("float32"),
    )


def _dense_group(tmp_path, *, name: str, well_fitted: bool):
    """A Dense scratch plane whose SE either follows the MAF model or defies it."""
    group = zarr.open_group(str(tmp_path / name), mode="w")
    eaf = np.linspace(0.05, 0.95, 600, dtype=np.float32)[:, None]
    predictor = np.log(2 * eaf * (1 - eaf))
    if well_fitted:
        se = np.exp(-3.0 - 0.5 * predictor)
    else:
        # Residuals far outside +-2, so every cell is an exception and the
        # coding cannot earn its bytes.
        se = np.exp(-3.0 - 0.5 * predictor + 6.0 * np.sin(np.arange(len(eaf))[:, None]))
    group.create_dataset("eaf", data=eaf, chunks=(100, 1), dtype="float32")
    group.create_dataset("se", data=se.astype(np.float32), chunks=(100, 1), dtype="float32")
    group.create_dataset("z", data=np.ones_like(eaf), chunks=(100, 1), dtype="float16")
    return group


def test_phase_timer_accumulates_repeated_visits_and_reports_shares(monkeypatch) -> None:
    # The timer reads a wall clock; script it rather than sleep, so the habit
    # check (no fixed sleeps in tests) stays satisfied and the arithmetic is
    # exact instead of approximate.
    ticks = iter([0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.035])
    monkeypatch.setattr("opengwasdb.encoding.timing.time.perf_counter", lambda: next(ticks))

    timer = PhaseTimer()
    for _ in range(3):
        with timer.phase("chunk"):
            pass
    with timer.phase("once"):
        pass

    assert timer.seconds["chunk"] == pytest.approx(0.03)
    assert timer.seconds["once"] == pytest.approx(0.005)
    report = timer.report()
    assert [name for name, _, _ in report] == ["chunk", "once"]
    assert sum(share for _, _, share in report) == pytest.approx(1.0)
    assert timer.total() == pytest.approx(sum(timer.seconds.values()))


def test_optimise_dense_se_charges_the_fit_the_measurement_and_the_rewrite(tmp_path) -> None:
    group = _dense_group(tmp_path, name="dense.zarr", well_fitted=True)
    timer = PhaseTimer()

    started = time.perf_counter()
    selected, _ = optimise_dense_se_joint(group, _preliminary(), timer=timer)
    wall = time.perf_counter() - started

    assert selected.se.is_residual
    charged = timer.seconds
    assert charged["fit"] > 0
    assert any(name.startswith("measure.") for name in charged)
    assert any(name.startswith("rewrite.") for name in charged)
    # Phases must partition the work rather than nest, or a share of the total
    # says nothing about which pass to optimise.
    assert timer.total() <= wall


def test_a_float16_outcome_still_charges_the_measurement_it_paid_for(tmp_path) -> None:
    group = _dense_group(tmp_path, name="bad-fit.zarr", well_fitted=False)
    timer = PhaseTimer()

    selected, coefficients = optimise_dense_se_joint(group, _preliminary(), timer=timer)

    assert not selected.se.is_residual
    assert coefficients is None
    # The survey is paid for whether or not it selects the coding; that is the
    # cost issue #144 exists to expose.
    assert any(name.startswith("measure.") for name in timer.seconds)
    assert not any(name.startswith("rewrite.") for name in timer.seconds)
