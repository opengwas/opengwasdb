"""Completion's quality counts describe stored cells, not merely planned fills (#361)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from shutil import copyfile, copytree, rmtree

import numpy as np
import pytest
import test_completion_identity as identity
import test_completion_long_alids as long
from completion_counts import quality_counts
from test_completion_long_alids import LONG, _dense, _ragged
from validation_bands import band_rows

from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store

RULE = "completion.quality_counts"
overlapping_completion = identity.overlapping_completion


def test_agreeing_dense_overlaps_validate_without_the_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = long._panel

    def overlapping_panel(root: Path, alids: dict[str, str]) -> Path:
        panel = original(root, alids)
        chromosome = panel / "EUR" / "1"
        for path in list(chromosome.iterdir()):
            copyfile(path, path.with_name(path.name.replace("900000-1300000", "second-block")))
        return panel

    monkeypatch.setattr(long, "_panel", overlapping_panel)
    completed = _dense(tmp_path, LONG)
    assert list(quality_counts(completed).values()) == [(4, 0), (4, 0)]
    assert np.asarray(open_store(completed).arrays(mode="r")["imputed"][:]).sum() == 4
    rmtree(tmp_path / "ld_panel")
    band_rows(monkeypatch, 1)
    for workers in (1, 3):
        result = validate_store(completed, n_workers=workers)
        assert result.ok, result.errors


def test_agreeing_and_conflicting_ragged_overlaps_validate_without_the_panel(
    overlapping_completion: tuple[Path, dict[str, int]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed, totals = overlapping_completion
    assert list(quality_counts(completed).values()) == [(2, 0), (1, 0), (2, 0)]
    assert (totals["n_imputed"], totals["n_missing"]) == (1, 1)
    rmtree(completed.parent / "panel")
    band_rows(monkeypatch, 1)
    for workers in (1, 3):
        result = validate_store(completed, n_workers=workers)
        assert result.ok, result.errors


@pytest.fixture(scope="module", params=[_dense, _ragged], ids=["dense", "ragged"])
def quality_seed(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return request.param(tmp_path_factory.mktemp("quality"), LONG)


@pytest.fixture
def completed(quality_seed: Path, tmp_path: Path) -> Path:
    return copytree(quality_seed, tmp_path / "completed.opengwasdb")


@pytest.mark.parametrize("column", ["n_imputed", "n_missing"])
def test_quality_counts_must_describe_the_cells_written(completed: Path, column: str) -> None:
    assert validate_store(completed).ok, "the unmodified completed fixture must validate"
    with sqlite3.connect(completed / "index.sqlite") as db:
        rows = db.execute("SELECT n_imputed, n_missing FROM completion_quality").fetchall()
        assert len(rows) == 1 and rows[0][0] >= 2, "one quality row must record several fills"
        # The two column names are this test's closed parameter set, not input SQL.
        db.execute(f"UPDATE completion_quality SET {column} = {column} + 1")
    result = validate_store(completed)
    assert result.failed() == {RULE}, result.errors
    assert any(column in message for message in result.errors)


@pytest.mark.parametrize("workers", [1, 3])
def test_a_lost_fill_is_found_even_when_other_fills_survive(
    completed: Path, monkeypatch: pytest.MonkeyPatch, workers: int,
) -> None:
    store = open_store(completed)
    group = store.arrays(mode="r+")
    if "ragged" in group:
        group = group["ragged"]
    mask = np.asarray(group["imputed"][:])
    filled = np.flatnonzero(mask)
    assert len(filled) >= 2, "losing one fill must not trigger the categorical declaration rule"
    mask.flat[filled[0]] = 0
    group["imputed"][:] = mask
    band_rows(monkeypatch, 1, mask.shape[1] if mask.ndim == 2 else 1)
    result = validate_store(completed, rules={RULE}, n_workers=workers)
    assert result.failed() == {RULE}, result.errors
    assert any("n_imputed" in message for message in result.errors)


def test_erasing_failed_attempts_does_not_establish_ancestry_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opengwasdb.completion import block

    monkeypatch.setattr(
        block, "impute_z_block", lambda z, *args, **kwargs: (np.full(len(z), np.nan), -1.0)
    )
    completed = _dense(tmp_path, LONG)
    assert validate_store(completed).ok, "the rejected-fit completion must initially validate"
    with sqlite3.connect(completed / "index.sqlite") as db:
        assert db.execute(
            "SELECT n_imputed, n_missing FROM completion_quality"
        ).fetchall() == [(0, 4)], "four failed attempts must actually be recorded"
        db.execute("DELETE FROM completion_quality")
    result = validate_store(completed)
    assert result.failed() == {RULE}, result.errors
    assert "n_imputed + n_missing=0" in result.errors[0]


def test_a_selected_count_rule_cannot_pass_without_its_quality_table(completed: Path) -> None:
    with sqlite3.connect(completed / "index.sqlite") as db:
        db.execute("DROP TABLE completion_quality")
    result = validate_store(completed, rules={RULE})
    assert result.failed() == {RULE}, result.errors
    assert "cannot be counted" in result.errors[0]


def test_a_clean_completed_release_agrees_with_its_counts(
    completed: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    band_rows(monkeypatch, 1)
    result = validate_store(completed, rules={RULE})
    assert result.ok, result.errors
    assert RULE not in result.skipped
