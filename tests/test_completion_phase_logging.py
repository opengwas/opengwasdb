"""Reference Completion says where its wall time goes (#309).

A completion on a Store the size of OGS-00009 runs for most of a day and, until
now, printed only a line per 200 LD blocks. Each of its five phases logs a start
line and an end line with the elapsed time, the LD-block phase logs progress with
an ETA, and the run closes with a report that says how the accounted time split.
Hybrid completes its Dense Component through the same function, so it logs the
same phases without a second copy of the logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from dense_sources import build_dense_fixture
from test_dense_completion import SOURCE_HEADER, SOURCE_ROWS, _make_ld_panel
from test_hybrid_completion import _build_source as build_hybrid_source
from test_hybrid_completion import _make_ld_panel as make_hybrid_panel

from opengwasdb.completion.parallel import run_block_tasks
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store

PHASES = (
    "Axis construction",
    "LD-block completion",
    "Checkpoint merge",
    "Band write",
    "Top-Hit Index",
)


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def _assert_every_phase_started_and_finished(messages: list[str]) -> None:
    for phase in PHASES:
        starts = [i for i, m in enumerate(messages) if m == f"{phase}: start"]
        ends = [i for i, m in enumerate(messages) if m.startswith(f"{phase}: done in ")]
        assert len(starts) == 1, f"{phase}: expected one start line in {messages}"
        assert len(ends) == 1, f"{phase}: expected one end line in {messages}"
        assert starts[0] < ends[0]
    order = [next(i for i, m in enumerate(messages) if m == f"{p}: start") for p in PHASES]
    assert order == sorted(order), "the phases ran out of order"


def _assert_report_names_every_phase(messages: list[str]) -> None:
    reports = [m for m in messages if m.startswith("Reference Completion phase report")]
    assert len(reports) == 1, messages
    for phase in PHASES:
        assert phase in reports[0]
    assert "total accounted" in reports[0]


@pytest.fixture
def dense_inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "associations.tsv"
    source.write_text(SOURCE_HEADER + "\n" + "\n".join(SOURCE_ROWS) + "\n", encoding="utf-8")
    observed = tmp_path / "obs.opengwasdb"
    build_dense_fixture(
        [source], observed, store_id="test", release_id="obs-v1", reference_assembly="GRCh38"
    )
    panel = _make_ld_panel(tmp_path)
    assert len(list(panel.glob("EUR/*/*.tsv"))) > 1, "the fixture must have several LD blocks"
    return observed, panel


def test_complete_dense_logs_each_phase_progress_and_a_report(
    dense_inputs: tuple[Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    observed, panel = dense_inputs
    with caplog.at_level(logging.INFO):
        complete_dense_store(
            observed, tmp_path / "comp.opengwasdb", panel,
            ancestry="EUR", min_cor=0.0, release_id="comp-v1",
        )
    messages = _messages(caplog)
    _assert_every_phase_started_and_finished(messages)
    _assert_report_names_every_phase(messages)
    progress = [m for m in messages if m.startswith("LD blocks:")]
    assert progress, messages
    assert "2/2" in progress[-1] and "ETA" in progress[-1]


def test_complete_hybrid_logs_the_dense_components_phases(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src = build_hybrid_source(tmp_path)
    with caplog.at_level(logging.INFO):
        complete_hybrid_store(
            src, tmp_path / "dst.opengwasdb", make_hybrid_panel(tmp_path), min_cor=0.0, thresh=0.9
        )
    messages = _messages(caplog)
    _assert_every_phase_started_and_finished(messages)
    _assert_report_names_every_phase(messages)


@pytest.mark.parametrize("n_workers", [1, 2])
def test_block_progress_reports_every_block_with_an_eta(
    n_workers: int, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        run_block_tasks(list(range(5)), n_workers, abs, every=2)
    lines = [m for m in _messages(caplog) if m.startswith("LD blocks:")]
    assert [line.split(" — ")[0] for line in lines] == [
        "LD blocks: 2/5", "LD blocks: 4/5", "LD blocks: 5/5",
    ]
    assert all("ETA" in line for line in lines)
