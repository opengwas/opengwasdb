"""The ancestry parameter on the draw of Analyses for a sweep (#405).

On a synthetic pool: another ancestry draws only its own Analyses, the target
and the |z| quotas floor at what the pool holds rather than refusing, and the
record of a draw other than EUR's names the ancestry and the floors while the
EUR record keeps the shape #403 committed.
"""

from __future__ import annotations

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
