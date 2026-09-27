"""Issue #229: a plane that cannot carry residual SE is known to be so early.

Residual SE is all-or-nothing per component -- one Analysis with a finite SE and
no frequency, or one Analysis the fit cannot solve, condemns the whole plane --
and that verdict used to be read only after the coefficient fit, with both byte
measurement passes running regardless. The gate decides it from the fit's own
bounded pass instead, so the rejection costs the read the fit would have made
and saves the two measurements, and the fallback says which trigger fired and
how many Analyses were responsible.

What a rejected plane ends up as must not change: the same `float16` narrowing,
the same absence of coefficients and side tables, and the same plan in the
manifest. An eligible plane must be fitted, measured, chosen and rewritten
exactly as it was -- one `fit` pass, then the measurement and rewrite passes.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from opengwasdb.encoding.measure import fit_se_grid
from opengwasdb.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding
from opengwasdb.encoding.planes import DenseSePlane
from opengwasdb.encoding.se import OverflowCells, optimise_dense_se_joint
from opengwasdb.encoding.timing import PhaseTimer

_N_ROWS = 600
_ROW_CHUNK = 100
_PRELIMINARY = StoreEncoding(
    z=ZEncoding("float16"), se=SeEncoding("float16"), eaf=EafEncoding("float32")
)
#: Fixtures with one deliberate defect, and the control they are read against.
_DEFECTS = ("", "no-eaf", "one-cell", "one-frequency")


class _RecordingTimer(PhaseTimer):
    """A timer that also keeps the order its phases were entered in.

    `PhaseTimer.seconds` sums a phase's visits, which cannot say whether the
    plane was walked once or twice, nor whether a pass ran at all. The gate's
    whole claim is about *which* passes run, so the tests count entries.
    """

    def __init__(self) -> None:
        super().__init__()
        self.entered: list[str] = []

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self.entered.append(name)
        with super().phase(name):
            yield

    def passes(self, prefix: str) -> int:
        return sum(1 for name in self.entered if name.startswith(prefix))


def _plane(n_analyses: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """A grid whose SE follows the MAF model closely in every Analysis."""
    eaf = np.linspace(0.05, 0.95, _N_ROWS, dtype=np.float32)[:, None]
    eaf = np.repeat(eaf, n_analyses, axis=1)
    se = np.exp(-3.0 - 0.5 * np.log(2 * eaf * (1 - eaf))).astype(np.float32)
    return eaf, se


def _defective(eaf: np.ndarray, se: np.ndarray, defect: str) -> None:
    """Break Analysis 1 the way one trigger's fixture requires, in place.

    Each defect leaves Analysis 1 carrying *finite* standard errors, so the
    fixtures cannot pass for the wrong reason:
    `no-eaf` gives its cells no frequency at all; `one-cell` leaves one usable
    cell, which is fewer than the two parameters being fitted; `one-frequency`
    leaves two cells at one frequency, so the design has no spread and the
    least-squares denominator is exactly zero. That last one is why the two
    cells share a row chunk: identical frequencies, added to identical partial
    sums, leave `count*sxx - sx*sx` zero rather than merely small.
    """
    if defect == "no-eaf":
        eaf[200:, 1] = np.nan
    elif defect == "one-cell":
        eaf[1:, 1] = np.nan
        se[1:, 1] = np.nan
    elif defect == "one-frequency":
        eaf[2:, 1] = np.nan
        se[2:, 1] = np.nan
        eaf[:2, 1] = np.float32(0.5)
    elif defect:
        raise AssertionError(f"unknown defect {defect!r}")


def _group(tmp_path: Path, name: str, eaf: np.ndarray, se: np.ndarray) -> Any:
    """A Dense scratch group: the `float32` plane, its EAF, and its `z`."""
    chunks = (_ROW_CHUNK, eaf.shape[1])
    group = zarr.open_group(str(tmp_path / name), mode="w")
    group.create_dataset("eaf", data=eaf, chunks=chunks, dtype="float32")
    group.create_dataset("se", data=se, chunks=chunks, dtype="float32")
    group.create_dataset(
        "z", data=np.ones(eaf.shape, dtype=np.float16), chunks=chunks, dtype="float16"
    )
    return group


def _overflow(eaf: np.ndarray, se: np.ndarray) -> OverflowCells:
    """The same cells as a Ragged Overflow Component bundle.

    Analysis by Analysis, the way a CSR holds them and the way the fit's
    Analysis-aligned batching expects -- so a defect planted in one column is
    that column's cells here too.
    """
    n_analyses = eaf.shape[1]
    return OverflowCells(
        se_values=np.concatenate([se[:, analysis] for analysis in range(n_analyses)]),
        eaf_values=np.concatenate([eaf[:, analysis] for analysis in range(n_analyses)]),
        analysis_indices=np.repeat(np.arange(n_analyses, dtype=np.int64), eaf.shape[0]),
        n_analyses=n_analyses,
    )


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def _eligibility_line(caplog: pytest.LogCaptureFixture) -> str:
    """The one line the gate logs when it rejects the plane.

    The assertion is on the line itself rather than on the presence of some
    message: a fallback that says nothing, or says it twice, is the failure this
    feature exists to remove.
    """
    lines = [message for message in _messages(caplog) if message.startswith("SE eligibility:")]
    assert len(lines) == 1, lines
    return lines[0]


def _rejected(
    tmp_path: Path,
    name: str,
    eaf: np.ndarray,
    se: np.ndarray,
    caplog: pytest.LogCaptureFixture,
    *,
    overflow: OverflowCells | None = None,
) -> _RecordingTimer:
    """Run the optimiser over an ineligible fixture, and return what it paid for.

    Both triggers are asserted the same way -- the plane comes back as
    `float16`, no byte measurement ran, and one line names the Analyses -- so
    the text that differs between the tests is only the reason.
    """
    group = _group(tmp_path, name, eaf, se)
    timer = _RecordingTimer()
    with caplog.at_level(logging.INFO, logger="opengwasdb.encoding.se"):
        selected, coefficients = optimise_dense_se_joint(
            group,
            _PRELIMINARY,
            overflow=_overflow(eaf, se) if overflow is None else overflow,
            overflow_chunk=200,
            timer=timer,
        )
    assert selected.se == SeEncoding("float16")
    assert coefficients is None
    assert timer.passes("fit") == 1
    assert timer.passes("measure.") == 0, "no byte measurement may be paid for"
    return timer


# ── The eligible path is untouched ───────────────────────────────────────────


def test_an_eligible_plane_is_fitted_measured_and_rewritten_as_before(tmp_path) -> None:
    """One pass decides eligibility, and an eligible plane still pays for the
    measurement and rewrite it always did -- the gate adds no pass of its own."""
    eaf, se = _plane()
    group = _group(tmp_path, "eligible.zarr", eaf, se)
    timer = _RecordingTimer()

    selected, coefficients = optimise_dense_se_joint(
        group, _PRELIMINARY, overflow=_overflow(eaf, se), overflow_chunk=200, timer=timer
    )

    assert selected.se.is_residual, "fixture is meaningful only if the coding is chosen"
    assert coefficients is not None
    assert timer.passes("fit") == 1, "the plane is walked once, not once per question asked of it"
    assert timer.passes("measure.read") >= 1
    assert timer.passes("measure.overflow") == 1
    assert timer.passes("rewrite.") >= 1
    decoded = DenseSePlane.open(group, selected).band(0, _N_ROWS)
    np.testing.assert_allclose(decoded, se.astype(np.float16).astype(np.float32), rtol=0.01)


# ── Trigger 1: a finite SE whose cell has no EAF ─────────────────────────────


def test_the_control_fixture_is_eligible_so_the_verdicts_below_mean_something(tmp_path) -> None:
    """Without this, every assertion about a rejection could hold vacuously."""
    eaf, se = _plane(n_analyses=3)
    assert fit_se_grid(se, eaf)[1].eligible
    group = _group(tmp_path, "control.zarr", eaf, se)
    selected, _ = optimise_dense_se_joint(group, _PRELIMINARY)
    assert selected.se.is_residual


def test_analyses_without_eaf_are_rejected_before_the_fit_or_a_measurement(
    tmp_path, caplog
) -> None:
    """Two of three Analyses without frequencies condemn all three (issue #229).

    The trigger is the count of Analyses, not the count of cells: the cell with
    a finite SE and no EAF cannot be stored in a residual plane at all, so every
    other Analysis loses the coding with it.
    """
    eaf, se = _plane(n_analyses=3)
    _defective(eaf, se, "no-eaf")
    eaf[200:, 2] = np.nan
    assert not fit_se_grid(se, eaf)[1].eligible, "the whole-array fit must agree it is ineligible"
    timer = _rejected(tmp_path, "no-eaf.zarr", eaf, se, caplog)

    assert timer.passes("rewrite.count") == 0 and timer.passes("rewrite.encode") == 0
    line = _eligibility_line(caplog)
    assert "2 of 3 Analyses" in line and "no EAF" in line


# ── Trigger 2: an Analysis the fit cannot solve ──────────────────────────────


@pytest.mark.parametrize("defect", ["one-cell", "one-frequency"])
def test_an_unfittable_analysis_is_rejected_before_the_fit_or_a_measurement(
    tmp_path, caplog, defect
) -> None:
    """Too few cells, or two cells at one frequency, is no fit rather than a bad one.

    Both leave Analysis 1's coefficients non-finite, and a single such Analysis
    sends the plane to `float16` without a measurement being taken.
    """
    eaf, se = _plane()
    _defective(eaf, se, defect)
    assert not fit_se_grid(se, eaf)[1].eligible, "the whole-array fit must agree it is ineligible"
    _rejected(tmp_path, f"{defect}.zarr", eaf, se, caplog)

    line = _eligibility_line(caplog)
    assert "1 of 2 Analyses" in line and "too few or degenerate cells" in line


def test_the_overflow_alone_can_reject_the_shared_plane(tmp_path, caplog) -> None:
    """The gate reads both components, and the Dense plane never sees a measurement.

    A Hybrid release's two components partition one Analysis's associations, so
    an Overflow Analysis without frequencies must condemn the Dense plane too --
    which is the veto path a real build hit.
    """
    eaf, se = _plane()
    overflow_eaf, overflow_se = _plane()
    _defective(overflow_eaf, overflow_se, "no-eaf")
    _rejected(
        tmp_path,
        "overflow-veto.zarr",
        eaf,
        se,
        caplog,
        overflow=_overflow(overflow_eaf, overflow_se),
    )

    assert "no EAF" in _eligibility_line(caplog)


# ── What a rejected plane is left as ─────────────────────────────────────────


@pytest.mark.parametrize("defect", ["no-eaf", "one-cell", "one-frequency"])
def test_a_rejected_plane_is_left_exactly_as_the_float16_fallback_left_it(
    tmp_path, defect
) -> None:
    """The verdict moved earlier; nothing the verdict produces did.

    The plane is the `float16` narrowing of the scratch, no coefficients or
    exception tables are invented beside it, and the returned plan declares
    `float16` for `se` even when the caller's plan declared a residual one --
    which is what the pre-#229 decision returned for a rejected plane, and what
    a reader needs the manifest to say about the array actually written.
    """
    eaf, se = _plane()
    _defective(eaf, se, defect)
    group = _group(tmp_path, f"left-as-{defect}.zarr", eaf, se)

    selected, coefficients = optimise_dense_se_joint(
        group,
        StoreEncoding(
            z=ZEncoding("float16"),
            se=SeEncoding("int8_residual", 0.5),
            eaf=EafEncoding("float32"),
        ),
    )

    assert selected.se == SeEncoding("float16")
    assert coefficients is None
    assert group["se"].dtype == np.dtype("float16")
    np.testing.assert_array_equal(
        np.asarray(group["se"][:], dtype=np.float32),
        se.astype(np.float16).astype(np.float32),
    )
    assert "se_coefficients" not in group
    assert "se_exception_index" not in group
    assert "se_exception_value" not in group


# ── The gate is the same rule the whole-array fit implements ─────────────────


@pytest.mark.parametrize("defect", _DEFECTS)
def test_the_streamed_gate_agrees_with_the_whole_array_fit(tmp_path, defect) -> None:
    """`fit_se_grid` is the other implementation of the same two triggers.

    The Ragged layout's whole-array fit and this streaming one are two
    allocations of the same rule, and a plane one calls eligible and the other
    does not is a store whose encoding depends on which builder wrote it. On
    these fixtures the byte comparison cannot reject a plane the fit accepts
    (the control below codes residually), so what the optimiser selected is the
    gate's verdict made visible.
    """
    eaf, se = _plane()
    _defective(eaf, se, defect)
    whole_array = fit_se_grid(se, eaf)[1]
    group = _group(tmp_path, f"agree-{defect or 'control'}.zarr", eaf, se)

    selected, _ = optimise_dense_se_joint(group, _PRELIMINARY)

    assert whole_array.eligible is selected.se.is_residual


def test_the_gate_costs_one_pass_over_the_plane(tmp_path) -> None:
    """No second exhaustive walk: the sums and the evidence come from one read.

    A rejection still reads the plane once -- the evidence for the second
    trigger is only complete once every Analysis has been summed -- but the
    `fit` phase is entered exactly once, which is the read the pre-#229 path
    made before its measurements as well.
    """
    eaf, se = _plane()
    _defective(eaf, se, "one-cell")
    group = _group(tmp_path, "one-pass.zarr", eaf, se)
    timer = _RecordingTimer()

    optimise_dense_se_joint(group, _PRELIMINARY, timer=timer)

    assert timer.passes("fit") == 1
    assert timer.passes("measure.") == 0


# ── Nothing the fallback does is left unsaid ─────────────────────────────────


def test_the_fallback_names_the_trigger_and_counts_the_analyses(tmp_path, caplog) -> None:
    """A bare "falling back to float16" cannot tell the two triggers apart.

    The count is the difference between a source that reported no frequency and
    a plane that cannot be fitted at all, and it is the number that says whether
    closing the gap would recover the coding. The narrowing the fallback then
    performs is logged in its own right, as it was before this gate.
    """
    eaf, se = _plane(n_analyses=3)
    _defective(eaf, se, "one-cell")
    _rejected(tmp_path, "reason.zarr", eaf, se, caplog)

    line = _eligibility_line(caplog)
    assert "1 of 3 Analyses" in line and "too few or degenerate cells" in line
    messages = _messages(caplog)
    assert not any(message.startswith("SE fit: no eligible model") for message in messages)
    assert any(message.startswith("SE float16 narrowing: start") for message in messages)
