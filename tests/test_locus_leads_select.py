"""The stratified draw of EUR Analyses for the sweep (#403), on a synthetic pool."""

from __future__ import annotations

from typing import Any

import pytest
from prototype_modules import load_prototype

select = load_prototype("locus_leads", "select")

# (rows, max |z|) per Analysis: 6 with |z| >= 100, 6 in [38.5, 100), the rest below.
SHAPES = ([(500, 120.0)] * 3 + [(50_000, 150.0)] * 3 + [(2_000, 40.0)] * 3
          + [(200_000, 60.0)] * 3 + [(300, 10.0)] * 6 + [(5_000, 20.0)] * 6
          + [(50_000, 30.0)] * 6 + [(500_000, 35.0)] * 6 + [(2_000_000, 80.0)] * 2)


def _pool() -> list[Any]:
    return [select.PoolEntry("OGS-X", f"a{i:02d}", rows, z) for i, (rows, z) in enumerate(SHAPES)]


def _rule(**overrides: Any) -> Any:
    settings = {"seed": 7, "target": 12, "min_z100": 3, "min_p0": 5, "max_rows": 1_000_000}
    return select.SelectionRule(**(settings | overrides))


def test_the_strata_are_row_count_by_max_abs_z() -> None:
    assert select.stratum(999, 38.4) == ("<1k", "<38.5")
    assert select.stratum(1_000, 38.5) == ("1k-10k", "38.5-100")
    assert select.stratum(99_999, 99.9) == ("10k-100k", "38.5-100")
    assert select.stratum(100_000, 100.0) == (">=100k", ">=100")


def test_the_same_seed_draws_the_same_analyses_and_another_seed_does_not() -> None:
    first = select.draw(_pool(), _rule())
    assert select.draw(list(reversed(_pool())), _rule()).chosen == first.chosen
    assert select.draw(_pool(), _rule(seed=8)).chosen != first.chosen
    assert len(first.chosen) == 12


def test_the_quotas_are_met_and_the_exclusion_is_recorded() -> None:
    result = select.draw(_pool(), _rule())
    z = {e.analysis_id: e.max_abs_z for e in result.chosen}
    assert sum(v >= 100 for v in z.values()) >= 3
    assert sum(v >= 38.5 for v in z.values()) >= 5
    assert [(e.analysis_id, e.observed_rows) for e in result.excluded] == [
        ("a36", 2_000_000), ("a37", 2_000_000)]
    assert not {"a36", "a37"} & set(z)
    assert sum(result.pool_by_stratum.values()) == 36


def test_analyses_with_no_observed_rows_are_counted_and_never_drawn() -> None:
    pool = [*_pool(), select.PoolEntry("OGS-X", "empty", 0, 0.0)]
    result = select.draw(pool, _rule())
    assert result.empty == 1
    assert "empty" not in {e.analysis_id for e in result.chosen}
    assert sum(result.pool_by_stratum.values()) == 36


def test_a_quota_that_cannot_be_met_refuses_rather_than_drawing_fewer() -> None:
    with pytest.raises(select.QuotaUnmet, match=r"\|z\| >= 100.*6"):
        select.draw(_pool(), _rule(min_z100=7))
    with pytest.raises(select.QuotaUnmet, match="target"):
        select.draw(_pool(), _rule(target=100))
