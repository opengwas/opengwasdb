"""The ancestry parameter on the draw of Analyses for a sweep (#405).

On a synthetic pool: another ancestry draws only its own Analyses, the target
and the |z| quotas floor at what the pool holds rather than refusing, and the
record of a draw other than EUR's names the ancestry and the floors while the
EUR record keeps the shape #403 committed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from prototype_modules import load_prototype

select = load_prototype("locus_leads", "select")


# (store, analysis, rows, max |z|, ancestry) for the draw, on a synthetic pool:
# EAS holds two Analyses at |z| >= 100, AFR and AMR one each.
MIXED = [("OGS-X", "eas1", 5_000, 150.0, "EAS"), ("OGS-X", "eas2", 50_000, 200.0, "EAS"),
         ("OGS-X", "eas3", 200_000, 40.0, "EAS"), ("OGS-X", "eas4", 300, 10.0, "EAS"),
         ("OGS-Y", "afr1", 5_000, 150.0, "AFR"), ("OGS-Y", "afr2", 20_000, 20.0, "AFR"),
         ("OGS-Y", "amr1", 5_000, 120.0, "AMR"), ("OGS-Y", "amr2", 20_000, 25.0, "AMR")]


def _mixed() -> list[Any]:
    return [select.PoolEntry(store, analysis, rows, z, ancestry)
            for store, analysis, rows, z, ancestry in MIXED]


def _rule(**overrides: Any) -> Any:
    settings = {"seed": 405, "target": 40, "min_z100": 5, "min_p0": 5, "max_rows": 1_000_000,
                "ancestry": "EAS"}
    return select.SelectionRule(**(settings | overrides))


def test_an_ancestry_draws_only_that_ancestrys_analyses() -> None:
    rule, _ = select.with_floors(_mixed(), _rule())
    result = select.draw(_mixed(), rule)

    assert {e.ancestry for e in result.chosen} == {"EAS"}
    assert {e.ancestry for e in result.excluded + result.empty} <= {"EAS"}
    assert sum(result.pool_by_stratum.values()) == 4
    assert sorted(e.analysis_id for e in result.chosen) == ["eas1", "eas2", "eas3", "eas4"]


def test_the_quotas_and_target_floor_at_what_the_pool_holds_and_are_recorded() -> None:
    rule, floors = select.with_floors(_mixed(), _rule())

    # EAS holds four eligible Analyses: two at |z| >= 100, three at |z| >= 38.5.
    assert (rule.target, rule.min_z100, rule.min_p0) == (4, 2, 3)
    assert floors["min_abs_z_100"] == {"asked": 5, "floor": 2, "in_pool": 2}
    assert floors["target"] == {"asked": 40, "floor": 4, "in_pool": 4}
    assert len(select.draw(_mixed(), rule).chosen) == 4


def test_the_record_of_another_ancestry_names_it_and_its_floors() -> None:
    rule, floors = select.with_floors(_mixed(), _rule())
    record = select.selection_record(rule, [], _mixed(), select.draw(_mixed(), rule),
                                     floors=floors)

    assert record["ancestry"] == "EAS"
    assert record["quota_floors"] == floors
    assert "assigned_ancestry == EAS" in record["rule"]["pool"]
    assert record["pool_by_stratum"] == {">=100k x 38.5-100": 1, "10k-100k x >=100": 1,
                                         "1k-10k x >=100": 1, "<1k x <38.5": 1}


def test_the_eur_record_keeps_the_shape_403_committed() -> None:
    rule, floors = select.with_floors(_mixed(), _rule(ancestry="EUR", min_z100=0, min_p0=0))
    record = select.selection_record(rule, [], _mixed(), select.draw(_mixed(), rule),
                                     floors=floors)

    assert "ancestry" not in record
    assert "quota_floors" not in record
    assert "assigned_ancestry == EUR" in record["rule"]["pool"]


def test_a_short_eur_pool_still_refuses_as_403s_draw_did() -> None:
    """EUR keeps #403's refusal: one Analysis cannot be drawn as a 40-Analysis rule."""
    pool = [select.PoolEntry("OGS-X", "only", 5_000, 150.0, "EUR")]

    with pytest.raises(select.QuotaUnmet):
        select.draw_with_floors(pool, _rule(ancestry="EUR"))
    # The same pool for another ancestry is swept, with the floors recorded.
    drawn_by, result, floors = select.draw_with_floors(
        [replace(pool[0], ancestry="EAS")], _rule())
    assert (drawn_by.target, drawn_by.min_z100, drawn_by.min_p0) == (1, 1, 1)
    assert len(result.chosen) == 1
    assert floors is not None and floors["target"]["floor"] == 1


def test_the_draw_routes_on_the_exact_assigned_ancestry(monkeypatch: Any) -> None:
    """A Store's metadata, not the pool's shape: exact labels only, and Unassigned out."""
    rows = [("eur-1", "EUR"), ("afr-1", "AFR"), ("unassigned", ""), ("look-alike", "NOT_AFR"),
            ("afr-2", "AFR"), ("amr-1", "AMR")]
    opened: list[_FakeQuery] = []

    def _open(path: Path) -> _FakeQuery:
        opened.append(_FakeQuery(rows))
        return opened[-1]

    monkeypatch.setattr(select, "query_store", _open)
    store = Path("/stores/OGS-X/store.opengwasdb")

    entries = select.survey_store(store, "AFR")

    assert [e.analysis_id for e in entries] == ["afr-1", "afr-2"]
    assert {e.ancestry for e in entries} == {"AFR"}
    assert [e.observed_rows for e in entries] == [3, 3]
    assert all(query.closed for query in opened), "the Store query must be closed"

    for ancestry, expected in [("EUR", ["eur-1"]), ("AMR", ["amr-1"]), ("EAS", [])]:
        assert [e.analysis_id for e in select.survey_store(store, ancestry)] == expected


class _Hits:
    """One Analysis's Observed Associations at the tier: three rows of |z| 6."""

    count = 3
    rows = {"z": [6.0, -6.0, 2.0]}

    def __getitem__(self, key: str) -> list[float]:
        return self.rows[key]


class _FakeQuery:
    """A Store query stand-in: an analyses table and one tier per Analysis."""

    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self._rows = rows
        self.closed = False

    def analyses_table(self) -> dict[int, dict[str, str]]:
        return {i: {"analysis_id": analysis_id, "assigned_ancestry": ancestry}
                for i, (analysis_id, ancestry) in enumerate(self._rows)}

    def top_hits(self, *, analysis_id: str, threshold: float, observed_only: bool) -> _Hits:
        return _Hits()

    def close(self) -> None:
        self.closed = True
