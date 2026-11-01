"""Reference Completion and an SE above float16's range (issue #338, ADR 0068).

A completed release never holds an SE above 65,504: observed cells that do are
masked, imputed candidates that would are left missing, and the Analysis's
`n_missing_se_out_of_range` carries the source's count forward and adds both.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
import test_dense_completion as tdc
from test_check_observed_view import HEADER, ROWS, drop_n_missing_se_column, plant_infinite_se
from test_dense_completion import (
    _SIGNAL_MISSING_IDX,
    _SIGNAL_POSITIONS,
    _SIGNAL_Z_TRUE,
    write_signal_panel,
)

import opengwasdb.completion.block as block_module
from opengwasdb.build.observed import build_dense_observed_from_sources
from opengwasdb.build.source import NormalisedAssociation
from opengwasdb.layouts.dense.build import build_dense_observed_store
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.model.analyses import (
    AnalysesTable,
    read_analyses,
    read_analysis_records,
    write_analyses,
)
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store
from opengwasdb.variants import CanonicalVariant


def _recorded(store: Path) -> dict[str, str]:
    return {
        a.analysis_id: a.n_missing_se_out_of_range
        for a in read_analysis_records(store / "analyses.tsv")
    }


def _imputed(store: Path) -> np.ndarray:
    return np.asarray(open_store(store).arrays(mode="r")["imputed"][:])


def _quality(store: Path) -> tuple[int, int]:
    import sqlite3

    with sqlite3.connect(store / "index.sqlite") as db:
        return db.execute(
            "SELECT COALESCE(SUM(n_imputed), 0), COALESCE(SUM(n_missing), 0) "
            "FROM completion_quality"
        ).fetchone()


def _set_recorded(store: Path, counts: dict[str, str]) -> None:
    table = read_analyses(store / "analyses.tsv")
    rows = tuple(
        {**row, "n_missing_se_out_of_range": counts[row["analysis_id"]]} for row in table.rows
    )
    fieldnames = (
        table.fieldnames
        if "n_missing_se_out_of_range" in table.fieldnames
        else (*table.fieldnames, "n_missing_se_out_of_range")
    )
    write_analyses(store / "analyses.tsv", AnalysesTable(fieldnames, rows))


@pytest.fixture
def source(tmp_path: Path) -> Path:
    rows = tmp_path / "associations.tsv"
    # The observed-view fixture: the same rows with a frequency each, which an imputed SE needs.
    rows.write_text(HEADER + "\n" + "\n".join(ROWS) + "\n", encoding="utf-8")
    out = tmp_path / "obs.opengwasdb"
    build_dense_observed_from_sources(
        [rows], out, store_id="test", release_id="obs-v1", reference_assembly="GRCh38"
    )
    return out


@pytest.fixture
def signal_observed_store(tmp_path: Path) -> Path:
    """One Analysis observing every position of the AR(1) signal block but the held-out one."""
    records = [
        NormalisedAssociation(
            analysis_id="a1",
            variant=CanonicalVariant("1", pos, "A", "G"),
            z=float(z),
            se=0.2,
            eaf=0.3,
        )
        for i, (pos, z) in enumerate(zip(_SIGNAL_POSITIONS, _SIGNAL_Z_TRUE, strict=True))
        if i != _SIGNAL_MISSING_IDX
    ]
    out = tmp_path / "signal_obs.opengwasdb"
    build_dense_observed_store(
        records, out, store_id="test", release_id="obs-v1", reference_assembly="GRCh38"
    )
    return out


@pytest.fixture
def signal_panel_npz_only(tmp_path: Path) -> Path:
    return write_signal_panel(tmp_path / "signal_panel")


@pytest.fixture
def panel(tmp_path: Path) -> Path:
    return tdc._make_ld_panel(tmp_path)


def _complete(source: Path, panel: Path, tmp_path: Path, name: str) -> Path:
    out = tmp_path / name
    complete_dense_store(source, out, panel, ancestry="EUR", min_cor=0.0, release_id="comp")
    return out


def test_dense_completion_carries_the_count_forward_and_leaves_oversized_imputed_cells_missing(
    tmp_path, signal_observed_store, signal_panel_npz_only, monkeypatch
):
    """One Analysis, one held-out position: baseline imputes exactly that one cell."""
    source, panel = signal_observed_store, signal_panel_npz_only
    baseline = _complete(source, panel, tmp_path, "baseline.opengwasdb")
    assert int(_imputed(baseline).sum()) == 1, "fixture must impute the one held-out cell"
    assert _recorded(baseline) == {"a1": "0"}
    assert _quality(baseline)[0] == 1

    # Imputing an SE above float16's range needs an unrealistic fixture, so the imputation's
    # SE output is patched at its public seam (completion.block's `scalar_n_se`).
    real = block_module.scalar_n_se
    monkeypatch.setattr(
        block_module, "scalar_n_se", lambda *a, **k: np.asarray(real(*a, **k)) * 1e9
    )
    _set_recorded(source, {"a1": "3"})  # what the observed build recorded
    completed = _complete(source, panel, tmp_path, "oversized.opengwasdb")

    assert int(_imputed(completed).sum()) == 0
    assert _recorded(completed) == {"a1": "4"}  # 3 carried forward + 1 left missing
    # `completion_quality` agrees with the arrays: nothing imputed, the cell missing instead.
    assert _quality(completed)[0] == 0
    assert _quality(completed)[1] == _quality(baseline)[1] + 1
    table = read_analyses(completed / "analyses.tsv")
    assert int(table.rows[0]["completion_n_imputed_total"]) == 0
    se = np.asarray(open_store(completed).arrays(mode="r")["se"][:], dtype=np.float32)
    assert not np.isinf(se).any()


def test_dense_completion_masks_a_source_infinite_se_and_counts_it(tmp_path, source, panel):
    plant_infinite_se(source, "1:1000000:A:G", 0)  # a 0.2.0-era plane: +inf, no count column
    baseline_after = _complete(source, panel, tmp_path, "completed.opengwasdb")

    # No encoder guard was hit: the cell is stored missing, in z and se together, ...
    root = open_store(baseline_after).arrays(mode="r")
    assert np.isfinite(np.asarray(root["se"][:], dtype=np.float32)).sum() > 0
    assert not np.isinf(np.asarray(root["se"][:], dtype=np.float32)).any()
    # ... and the count is the one cell completion found in a release with no column.
    recorded = _recorded(baseline_after)
    assert recorded["a1"] == "1" and recorded["a2"] == "0"
    result = validate_store(baseline_after)
    assert result.ok, result.errors
    assert not [w for w in result.warnings if "infinite" in w.lower()]


def test_dense_completion_without_a_source_column_counts_from_zero(tmp_path, source, panel):
    drop_n_missing_se_column(source)
    completed = _complete(source, panel, tmp_path, "completed.opengwasdb")
    assert _recorded(completed) == {"a1": "0", "a2": "0"}  # a written 0, not unknown


def test_the_count_survives_a_copy_of_the_pair(tmp_path, source):
    # Guards the fixture helper itself: copying a store keeps its recorded count.
    copied = copytree(source, tmp_path / "copy.opengwasdb")
    assert _recorded(copied) == _recorded(source) == {"a1": "0", "a2": "0"}


def test_ragged_completion_leaves_oversized_imputed_rows_missing_and_counts_them(
    tmp_path, monkeypatch
):
    import test_ragged_completion as trc
    from test_observed_view_ragged import SOURCES, _build_source

    from opengwasdb.layouts.ragged.complete import complete_ragged_store

    observed = _build_source(tmp_path / "src", SOURCES)
    panel = trc._make_rich_ld_panel(tmp_path)

    def run(name: str) -> Path:
        out = tmp_path / name
        complete_ragged_store(
            observed, out, panel, ancestry="EUR", cis_window_bp=500_000, min_cor=0.0,
            release_id="comp",
        )
        return out

    baseline = run("baseline.opengwasdb")
    n_imputed = int(open_store(baseline).arrays(mode="r")["ragged"]["imputed"][:].sum())
    assert n_imputed > 0, "fixture must impute something, or there is nothing to leave missing"
    assert set(_recorded(baseline).values()) == {"0"}

    real = block_module.scalar_n_se
    monkeypatch.setattr(
        block_module, "scalar_n_se", lambda *a, **k: np.asarray(real(*a, **k)) * 1e9
    )
    completed = run("oversized.opengwasdb")

    group = open_store(completed).arrays(mode="r")["ragged"]
    assert int(np.asarray(group["imputed"][:]).sum()) == 0
    assert sum(int(v) for v in _recorded(completed).values()) == n_imputed
    assert _quality(completed)[0] == 0
    assert _quality(baseline)[0] == n_imputed
    assert not np.isinf(np.asarray(group["se"][:], dtype=np.float32)).any()
    result = validate_store(completed)
    assert result.ok, result.errors
