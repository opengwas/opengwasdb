"""From 0.3.0 the envelope is a function of layout, version and completion state (#471).

`index.sqlite` holds only `completion_quality` from format 0.3.0 (ADR 0102), so it ships
only in a Reference-Completed release, a Hybrid's in its Dense Component. Each of the
six fixture releases, as this build writes them, is held to that; then each way a
release can get it wrong is shown to fail the rule that names it.
"""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from format_0_2_3 import as_0_2_3

from opengwasdb.store.open import (
    DENSE_ENVELOPE,
    HYBRID_DENSE_COMPONENT_ENVELOPE,
    HYBRID_ENVELOPE,
    RAGGED_ENVELOPE,
    envelope_at,
    ships_index,
)
from opengwasdb.store_reader.enums import CompletionState
from opengwasdb.validation import validate_store

NAMES = [
    f"{layout}_{state}"
    for layout in ("dense", "hybrid", "ragged")
    for state in ("completed", "observed")
]
OO, RC = CompletionState.OBSERVED_ONLY, CompletionState.REFERENCE_COMPLETED


def _tables(index: Path) -> set[str]:
    with closing(sqlite3.connect(index)) as connection:
        listed = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {row[0] for row in listed}


def test_index_sqlite_ships_by_version_layout_and_completion_state():
    every = (DENSE_ENVELOPE, RAGGED_ENVELOPE, HYBRID_ENVELOPE, HYBRID_DENSE_COMPONENT_ENVELOPE)
    for envelope in every:
        assert "index.sqlite" not in envelope
        for state in (OO, RC):
            assert "index.sqlite" in envelope_at(envelope, "0.2.3", state)
            assert ships_index(envelope, "0.2.3", state)
        assert "index.sqlite" not in envelope_at(envelope, "0.3.0", OO)
    for envelope in (DENSE_ENVELOPE, RAGGED_ENVELOPE, HYBRID_DENSE_COMPONENT_ENVELOPE):
        assert "index.sqlite" in envelope_at(envelope, "0.3.0", RC)
    # A Hybrid's outer release holds no completion_quality: its Dense Component does.
    assert "index.sqlite" not in envelope_at(HYBRID_ENVELOPE, "0.3.0", RC)


@pytest.mark.parametrize("name", NAMES)
def test_a_0_3_0_build_carries_index_sqlite_only_to_hold_completion_quality(
    digest_releases: dict[str, Path], name: str
):
    release = digest_releases[name]
    completed = name.endswith("_completed")
    hybrid = name.startswith("hybrid")
    assert (release / "index.sqlite").exists() is (completed and not hybrid)
    if hybrid:
        assert (release / "dense" / "index.sqlite").exists() is completed
    for index in [release / "index.sqlite", release / "dense" / "index.sqlite"]:
        if index.exists():
            assert _tables(index) == {"completion_quality"}, index
    assert validate_store(release).ok


def _copy(digest_releases: dict[str, Path], tmp_path: Path, name: str) -> Path:
    return Path(shutil.copytree(digest_releases[name], tmp_path / name))


@pytest.mark.parametrize("name", ["dense_observed", "ragged_observed", "hybrid_completed"])
def test_an_index_sqlite_where_0_3_0_carries_none_fails_the_closed_envelope(
    digest_releases: dict[str, Path], tmp_path: Path, name: str
):
    release = _copy(digest_releases, tmp_path, name)
    assert not (release / "index.sqlite").exists()
    (release / "index.sqlite").touch()
    result = validate_store(release)
    assert "envelope.closed" in result.failed()
    assert any("unexpected store entry 'index.sqlite'" in e for e in result.errors)


@pytest.mark.parametrize(
    ("name", "component"), [("dense_completed", "."), ("hybrid_completed", "dense")]
)
def test_a_0_3_0_index_holding_a_dropped_table_fails_provenance_only(
    digest_releases: dict[str, Path], tmp_path: Path, name: str, component: str
):
    release = _copy(digest_releases, tmp_path, name)
    with closing(sqlite3.connect(release / component / "index.sqlite")) as connection:
        connection.execute("CREATE TABLE variant_aliases (alias TEXT, variant_index INTEGER)")
        connection.commit()
    result = validate_store(release)
    assert "sqlite.provenance_only" in result.failed()
    assert any("index.sqlite holds variant_aliases" in e for e in result.errors)


def test_a_completed_0_3_0_release_without_its_index_fails_required_entries(
    digest_releases: dict[str, Path], tmp_path: Path
):
    release = _copy(digest_releases, tmp_path, "ragged_completed")
    (release / "index.sqlite").unlink()
    result = validate_store(release)
    assert "envelope.required" in result.failed()
    assert "missing index.sqlite" in result.errors


@pytest.mark.parametrize("name", ["dense_observed", "ragged_observed"])
def test_a_0_2_3_release_still_requires_its_index(
    digest_releases: dict[str, Path], tmp_path: Path, name: str
):
    release = as_0_2_3(_copy(digest_releases, tmp_path, name))
    assert validate_store(release).ok
    (release / "index.sqlite").unlink()
    assert "missing index.sqlite" in validate_store(release).errors
