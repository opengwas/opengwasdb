"""`query --format json` carries the record's `eaf` and status, as RFC 8259 JSON.

Each row is the association record (spec §19) keyed by index: a cell that is not
there is `null`, and `association_status` says the association is `missing`;
±inf is the string `"Infinity"`/`"-Infinity"`, because it is a value (spec §15)
that RFC 8259 has no token for (issue #394).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_ragged_residual_completion import RaggedResidualScenario
from typer.testing import CliRunner

import opengwasdb.cli.main as cli
from opengwasdb.associations import Associations
from opengwasdb.cli.main import _emit_json, app

#: A row is the record's mapping, key by key (spec §19).
JSON_KEYS = {"variant_index", "analysis_index", "z", "se", "eaf", "association_status"}


def _strict(output: str) -> list[dict[str, object]]:
    """Parse `output` as RFC 8259, refusing the tokens it has no room for."""

    def refuse(token: str) -> object:
        raise AssertionError(f"{token} is not an RFC 8259 token")

    return json.loads(output, parse_constant=refuse)


def _rows(command: str, store: Path, *args: str) -> list[dict[str, object]]:
    result = CliRunner().invoke(app, [command, str(store), *args, "--format", "json"])
    assert result.exit_code == 0, result.output
    return _strict(result.output)


@pytest.fixture(scope="module")
def dense_store(dense_fixture_store: Path) -> Path:
    """The shared fixture Store, which every command in this module only reads."""
    return dense_fixture_store


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("query-phewas", ("rs1",)),
        ("query-range-phewas", ("1", "1", "400")),
        ("query-analysis", ("a1",)),
        ("query-lookup", ("rs1,rs3", "a1,a2")),
        ("query-top-hits", ()),
    ],
)
def test_every_query_command_returns_the_record(
    dense_store: Path, command: str, args: tuple[str, ...]
) -> None:
    rows = _rows(command, dense_store, *args)
    assert rows, f"{command} must return rows for this to mean anything"
    for row in rows:
        assert set(row) == JSON_KEYS, command


def test_the_rows_keep_the_values_they_printed_before(dense_store: Path) -> None:
    """The four keys the CLI printed before this change keep their names and
    their values, with `eaf` and `association_status` added beside them.

    This Store's cells are all present and finite and it has no frequency plane,
    so the rows are the old ones exactly, with `"eaf": null` on each."""
    assert _rows("query-phewas", dense_store, "rs1") == [
        {
            "variant_index": 0,
            "analysis_index": 0,
            "z": 2.0,
            "se": 0.0999755859375,
            "eaf": None,
            "association_status": "observed",
        },
        {
            "variant_index": 0,
            "analysis_index": 1,
            "z": 6.0,
            "se": 0.199951171875,
            "eaf": None,
            "association_status": "observed",
        },
    ]


def test_a_missing_cell_is_null_and_the_status_says_why(
    residual_ragged_scenario: RaggedResidualScenario,
) -> None:
    """One sequence holding observed, imputed and missing rows, so a `null` can be
    read against a value in the same output."""
    rows = _rows("query-analysis", residual_ragged_scenario.completed, "a")
    first: dict[str, dict[str, object]] = {}
    for row in rows:
        first.setdefault(str(row["association_status"]), row)
    assert set(first) == {"observed", "imputed", "missing"}
    missing = first["missing"]
    assert missing["z"] is None
    assert missing["se"] is None
    assert missing["eaf"] is None
    for status in ("observed", "imputed"):
        assert first[status]["z"] is not None, status
        assert first[status]["se"] is not None, status


def test_an_imputed_row_carries_the_reference_eaf_beside_an_observed_null(
    panel_only_eaf_ragged_store: Path,
) -> None:
    """This release has no association-frequency plane, only the Reference EAF, so
    an observed cell's `eaf` is unavailable and an imputed one's is the panel's
    (spec §9, ADR 0036). A `null` therefore says the frequency is not available,
    not that the association is: `association_status` says that."""
    rows = _rows("query-analysis", panel_only_eaf_ragged_store, "a1")
    observed = [row for row in rows if row["association_status"] == "observed"]
    imputed = [row for row in rows if row["association_status"] == "imputed"]
    assert observed and imputed, "the fixture must hold both for this to mean anything"
    assert all(row["eaf"] is None for row in observed)
    assert [row["eaf"] for row in imputed] == [pytest.approx(0.4, abs=1e-6)]


def test_an_infinite_se_keeps_its_value_and_its_sign(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An SE above `float16`'s range is stored `+inf` beside a finite Z (spec §15,
    issue #338): a present cell, which may not print as the `null` a missing one
    prints as. The record is the one every reader returns."""
    _emit_json(
        Associations.from_decoded(
            0,
            0,
            z=np.array([2.0, 3.0, np.nan], dtype=np.float32),
            se=np.array([np.inf, -np.inf, np.nan], dtype=np.float32),
            eaf=np.array([0.25, np.nan, np.nan], dtype=np.float32),
            imputed=np.array([0, 1, 0], dtype=np.uint8),
        )
    )
    rows = _strict(capsys.readouterr().out)
    assert [row["se"] for row in rows] == ["Infinity", "-Infinity", None]
    assert [row["z"] for row in rows] == [2.0, 3.0, None]
    assert [row["association_status"] for row in rows] == ["observed", "imputed", "missing"]
    assert rows[0]["eaf"] == 0.25


def test_n_keeps_its_integer_form(capsys: pytest.CaptureFixture[str]) -> None:
    """ADR 0067's `n` rides the row as an integer, `null` where the build stored
    none, and a NaN there must not reach JSON, which has no token for it."""
    _emit_json(
        {
            "variant_index": np.array([0, 1], dtype=np.int32),
            "analysis_index": np.array([0, 0], dtype=np.int32),
            "z": np.array([2.0, 6.0], dtype=np.float32),
            "se": np.array([0.5, 0.25], dtype=np.float32),
            "eaf": np.array([np.nan, 0.25], dtype=np.float32),
            "n": np.array([100.0, np.nan], dtype=np.float64),
            "association_status": np.array(["observed", "observed"], dtype=object),
        }
    )
    rows = _strict(capsys.readouterr().out)
    assert rows[0] == {
        "variant_index": 0,
        "analysis_index": 0,
        "z": 2.0,
        "se": 0.5,
        "eaf": None,
        "association_status": "observed",
        "n": 100,
    }
    assert rows[1]["n"] is None
    assert rows[1]["eaf"] == 0.25


def test_a_nan_that_escapes_the_row_mapping_stops_the_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`allow_nan=False`: a value the mapping did not turn into `null` stops the
    command rather than printing a token no JSON parser reads."""
    monkeypatch.setattr(cli, "_json_cell", float)
    result = {
        "variant_index": np.array([0], dtype=np.int32),
        "analysis_index": np.array([0], dtype=np.int32),
        "z": np.array([np.nan], dtype=np.float32),
        "se": np.array([0.5], dtype=np.float32),
        "eaf": np.array([np.nan], dtype=np.float32),
        "association_status": np.array(["missing"], dtype=object),
    }
    with pytest.raises(ValueError, match="Out of range float values"):
        _emit_json(result)
