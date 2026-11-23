"""`variant_table.position_sidecar` fails each way a 0.3.0 position sidecar can be wrong (#468).

Each corruption below leaves the three files present and loadable, so only the rule
can see it: positions out of order, a row moved, a chromosome boundary moved,
chromosomes out of canonical order or named twice, offsets that do not end at the row
count or do not match the chromosomes, a position of 0, the wrong dtype, a chromosome
misnamed, and a sidecar a row short. Each is asserted by identity: the whole validator
runs, and this rule alone fails. A clean release passes it and a 0.2 release is not
judged; the version, not the files present, decides whether a release carries one.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from format_0_2_3 import as_0_2_3
from test_range_phewas_0_3_0 import LAYOUTS

from opengwasdb.query import query_store
from opengwasdb.store_reader.position_sidecar import (
    CHROMOSOME_OFFSETS_FILENAME,
    CHROMOSOMES_FILENAME,
    POSITION_FILENAME,
    POSITION_SIDECAR_FILENAMES,
    MissingPositionSidecar,
    PositionSidecar,
    write_position_sidecar,
)
from opengwasdb.validation import validate_store

RULE = "variant_table.position_sidecar"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Each layout's 0.3.0 release over three chromosomes. Never written to."""
    return {name: build(tmp_path_factory.mktemp(name)) for name, build in LAYOUTS.items()}


@pytest.fixture
def dense(built: dict[str, Path], tmp_path: Path) -> Path:
    return Path(shutil.copytree(built["dense"], tmp_path / "dense.opengwasdb"))


def _rows(store: Path) -> tuple[list[str], list[int]]:
    sidecar = PositionSidecar.open(store)
    assert sidecar is not None
    chromosomes = sidecar.chromosomes_of(np.arange(sidecar.n_rows))
    return [str(c) for c in chromosomes], np.asarray(sidecar.positions).tolist()


def _edit(store: Path, name: str, change: Callable[[np.ndarray], np.ndarray]) -> None:
    np.save(store / name, change(np.load(store / name)))


def _set(index: int | slice, value: object) -> Callable[[np.ndarray], np.ndarray]:
    def change(array: np.ndarray) -> np.ndarray:
        array[index] = value
        return array

    return change


def positions_out_of_order(store: Path) -> None:
    """Chromosome 1's first and last rows trade positions."""
    _edit(store, POSITION_FILENAME, lambda a: a[[4, 1, 2, 3, 0, 5, 6, 7]])


def a_row_moved(store: Path) -> None:
    """Row 0 a base earlier: still in order, but not where the table puts it."""
    _edit(store, POSITION_FILENAME, _set(0, 999))


def a_boundary_moved(store: Path) -> None:
    """Chromosome 1's last row counted as chromosome 2's first."""
    _edit(store, CHROMOSOME_OFFSETS_FILENAME, _set(1, 4))


def chromosomes_out_of_order(store: Path) -> None:
    _edit(store, CHROMOSOMES_FILENAME, lambda a: a[[1, 0, 2]])


def a_chromosome_named_twice(store: Path) -> None:
    _edit(store, CHROMOSOMES_FILENAME, _set(1, b"1"))


def a_chromosome_misnamed(store: Path) -> None:
    """Chromosome 2's rows named chromosome 3: in canonical order, but not the table's."""
    _edit(store, CHROMOSOMES_FILENAME, _set(1, b"3"))


def offsets_short_of_the_rows(store: Path) -> None:
    _edit(store, CHROMOSOME_OFFSETS_FILENAME, lambda a: np.append(a[:-1], a[-1] - 1))


def offsets_for_another_count_of_chromosomes(store: Path) -> None:
    _edit(store, CHROMOSOME_OFFSETS_FILENAME, lambda a: np.insert(a, 1, 2))


def a_position_of_zero(store: Path) -> None:
    _edit(store, POSITION_FILENAME, _set(0, 0))


def positions_of_another_dtype(store: Path) -> None:
    _edit(store, POSITION_FILENAME, lambda a: a.astype("<i8"))


def a_sidecar_one_row_short(store: Path) -> None:
    chromosomes, positions = _rows(store)
    write_position_sidecar(store, chromosomes[:-1], positions[:-1], chromosome_key=None)


CORRUPTIONS: dict[str, tuple[Callable[[Path], None], str]] = {
    "positions_out_of_order": (positions_out_of_order, "not in genomic order: row 1 is"),
    "a_row_moved": (a_row_moved, "row 0: the sidecar places it at 1:999"),
    "a_boundary_moved": (a_boundary_moved, "row 5 is at position 300, before row 4"),
    "chromosomes_out_of_order": (chromosomes_out_of_order, "out of the canonical"),
    "a_chromosome_named_twice": (a_chromosome_named_twice, "resumes at row 5"),
    "a_chromosome_misnamed": (a_chromosome_misnamed, "row 5: the sidecar places it at 3:300"),
    "offsets_short_of_the_rows": (offsets_short_of_the_rows, "runs from 0 to 7, not from 0"),
    "offsets_for_another_count_of_chromosomes": (
        offsets_for_another_count_of_chromosomes,
        "has 5 offsets for 3 chromosome(s)",
    ),
    "a_position_of_zero": (a_position_of_zero, "holds position 0 at row 0"),
    "positions_of_another_dtype": (positions_of_another_dtype, "int64 of 1 dimension(s)"),
    "a_sidecar_one_row_short": (a_sidecar_one_row_short, "holds 7 rows but variants.tsv.gz has 8"),
}


def test_the_fixture_is_clean_and_judged(dense: Path) -> None:
    chromosomes, positions = _rows(dense)
    # Every corruption above names these rows.
    assert chromosomes == ["1"] * 5 + ["2"] * 2 + ["X"]
    assert positions == [1000, 5000, 5000, 5000, 9000, 300, 7000, 2500]
    result = validate_store(dense)
    assert result.ok, result.errors
    assert result.of(RULE) == [], "a clean 0.3.0 release passes without a finding"
    assert RULE not in {f.rule for f in result.skips}


@pytest.mark.parametrize("corruption", sorted(CORRUPTIONS))
def test_each_corruption_fails_the_rule_alone(dense: Path, corruption: str) -> None:
    corrupt, says = CORRUPTIONS[corruption]
    corrupt(dense)

    result = validate_store(dense)

    assert result.failed() == {RULE}, result.errors
    assert any(says in error for error in result.errors), result.errors


@pytest.mark.parametrize("layout", ["ragged", "hybrid"])
def test_the_rule_judges_every_layout(built: dict[str, Path], tmp_path: Path, layout: str) -> None:
    store = Path(shutil.copytree(built[layout], tmp_path / layout))
    assert validate_store(store).ok
    positions_out_of_order(store)
    assert validate_store(store).failed() == {RULE}


def test_the_rule_judges_a_hybrids_dense_component(built: dict[str, Path], tmp_path: Path) -> None:
    store = Path(shutil.copytree(built["hybrid"], tmp_path / "hybrid"))
    nested = store / "dense"
    _edit(nested, POSITION_FILENAME, _set(0, 999))
    result = validate_store(store)
    assert result.failed() == {RULE}, result.errors
    assert any("dense" in f.target for f in result.findings if f.rule == RULE), result.errors


def test_a_0_2_release_is_not_judged_and_a_stray_sidecar_is_not_read(dense: Path) -> None:
    as_0_2_3(dense)
    result = validate_store(dense)
    assert result.ok, result.errors
    assert all(f.severity.value != "error" for f in result.of(RULE))

    # A wrong sidecar beside a 0.2.3 release: read, it would answer chromosome 1's
    # ranges with nothing. The version says tabix answers, and the envelope reports it.
    write_position_sidecar(dense, ["2"] * 8, list(range(1, 9)), chromosome_key=None)
    with query_store(dense) as q:
        assert q._variant_axis.position_sidecar is None
        assert len(q._variant_axis.range_indices("1", 1, 10**6)) == 5
    result = validate_store(dense)
    assert result.failed() == {"envelope.closed"}, result.errors
    stray = sorted(e.split("'")[1] for e in result.errors if "unexpected store entry" in e)
    assert stray == sorted(POSITION_SIDECAR_FILENAMES)


@pytest.mark.parametrize("missing", [CHROMOSOMES_FILENAME, "all"])
def test_a_0_3_0_release_missing_its_sidecar_is_refused(dense: Path, missing: str) -> None:
    names = POSITION_SIDECAR_FILENAMES if missing == "all" else (missing,)
    for name in names:
        (dense / name).unlink()
    refused = FileNotFoundError if missing != "all" else MissingPositionSidecar
    with pytest.raises(refused, match="position sidecar is incomplete|carries a position"):
        with query_store(dense) as q:
            q.range_phewas("1", 1, 10**6)
    assert "envelope.required" in validate_store(dense).failed()
