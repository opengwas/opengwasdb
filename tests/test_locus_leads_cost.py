"""What genetic-distance pruning costs (#406).

Three things are held here directly: that a slice's timed record names every
stage the measurement asks for and carries the machine, date, commit and load
it was taken under; that a child process's peak RSS is seen (plink2's, when
clumping); and that the faster rule gives the same leads as `rule.locus_leads`.
"""

from __future__ import annotations

import datetime
import re
import sys

import numpy as np
import pytest
from prototype_modules import load_prototype

cost = load_prototype("locus_leads", "cost")
rule = load_prototype("locus_leads", "rule")

_SPIN = """
total = 0
for i in range(4_000_000):
    total += i
"""
_ALLOCATE_AND_SPIN = _SPIN + """
block = bytearray(200 * 1024 * 1024)
for at in range(0, len(block), 4096):
    block[at] = 1
"""


def test_the_timer_reports_every_stage_and_the_record_carries_its_provenance() -> None:
    called: list[str] = []

    def stage(name: str) -> object:
        def run() -> object:
            called.append(name)
            return name

        return run

    record = cost.time_stages({name: stage(name) for name in cost.STAGES}, repeats=2)

    assert tuple(record) == cost.STAGES
    assert sorted(called) == sorted(list(cost.STAGES) * 2)
    for entry in record.values():
        assert len(entry["samples_s"]) == 2
        assert entry["min_s"] <= entry["median_s"] <= entry["max_s"]
        assert isinstance(entry["median_s"], float)
        assert set(entry["load"]) == {"one", "five", "fifteen"}
        assert entry["peak_source"] in {"process", "children"}
        assert isinstance(entry["peak_mb"], float) and entry["peak_mb"] > 0

    provenance = cost.provenance()
    assert re.fullmatch(r"[0-9a-f]{7,40}", provenance["commit"])
    measured_at = datetime.datetime.fromisoformat(provenance["measured_at"])
    assert measured_at.utcoffset() == datetime.timedelta(0), "the date is recorded in UTC"
    machine = provenance["machine"]
    assert isinstance(machine["hostname"], str) and machine["hostname"]
    assert isinstance(machine["cpu_model"], str) and machine["cpu_model"]
    assert isinstance(machine["cores"], int) and machine["cores"] > 0
    assert isinstance(machine["memory_gb"], float) and machine["memory_gb"] > 0
    assert provenance["environment"]["plink2"], "plink2's version is part of the record"


def test_the_child_rss_sampler_sees_a_childs_peak_and_not_a_small_childs() -> None:
    small = cost.peak_child_rss([sys.executable, "-c", _SPIN])
    big = cost.peak_child_rss([sys.executable, "-c", _ALLOCATE_AND_SPIN])

    assert small < 100.0, f"a child that allocates nothing peaked at {small:.1f} MB"
    assert 190.0 <= big <= 330.0, f"a child holding 200 MB peaked at {big:.1f} MB"


def test_the_faster_rule_gives_the_same_leads_on_ties_a_hotspot_and_several_chromosomes() -> None:
    leads = _leads(
        chromosome=["1"] * 5 + ["2"] * 3 + ["X"] * 2,
        position=[1_000, 1_000, 1_450_000, 1_550_000, 1_600_000, 900_000, 950_000, 9_000_000,
                  10_000, 10_000],
        # A hotspot adds 2 cM between 1,499,000 and 1,501,000; everything else is cold.
        cm=[0.1, 0.1, 0.145, 2.155, 2.165, 0.9, 0.95, 9.9, 0.01, 0.01],
        abs_z=[9.0, 9.0, 8.0, 8.0, 7.0, 9.0, 9.0, 9.0, 6.0, 6.0],
        variant_index=[3, 1, 7, 2, 5, 9, 4, 8, 6, 0],
    )

    # Tied rows (at 1,000 and at 10,000) are visited by ascending variant index;
    # the pair across the hotspot keeps two leads, the cold pair at 900/950 kb one.
    assert leads.tolist() == [1, 1, 2, 3, 3, 6, 6, 7, 9, 9]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_faster_rule_gives_the_same_leads_on_clustered_rows_with_many_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    chromosome, position, cm = _clustered_rows(rng)

    leads = _leads(
        chromosome=chromosome.tolist(), position=position.tolist(), cm=cm.tolist(),
        # Stored z is quantised, so equal abs_z is the common case, not the exception.
        abs_z=(rng.integers(30, 60, position.size) / 4.0).tolist(),
        variant_index=rng.permutation(position.size).tolist(),
    )

    left = leads.tolist()
    assert 0 < sum(left[row] == row for row in range(len(left))) < len(left) // 2, \
        "the fixture must both keep leads and merge rows"


def _clustered_rows(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three chromosomes of dense clusters, with a hotspot on chromosome 2."""
    labels, positions = [], []
    for name, count in (("1", 900), ("2", 700), ("3", 400)):
        centres = rng.integers(1_000_000, 9_000_000, 10)
        near = centres[rng.integers(0, centres.size, count)]
        drawn = np.where(rng.random(count) < 0.7,
                         near + rng.integers(-9_000, 9_000, count),
                         rng.integers(1, 9_500_000, count))
        labels.append(np.repeat(name, count))
        positions.append(np.clip(drawn, 1, 9_500_000))
    position = np.concatenate(positions)
    chromosome = np.concatenate(labels)
    cm = np.where(chromosome == "2",
                  position / 4_000_000 + 2.0 * (position > 4_500_000),
                  position / 10_000_000)
    return chromosome, position, cm


def test_a_map_whose_cm_falls_is_refused_rather_than_silently_reordered() -> None:
    with pytest.raises(ValueError, match="falls"):
        _leads(chromosome=["1", "1"], position=[1_000, 2_000], cm=[0.5, 0.4], abs_z=[9.0, 8.0],
               variant_index=[0, 1])


def test_the_budget_divides_the_pruning_stages_by_the_query_they_follow() -> None:
    record = cost.budget_record({
        "tier": {"median_s": 10.0}, "positions": {"median_s": 4.0},
        "positions_array": {"median_s": 0.1}, "cm": {"median_s": 0.5},
        "sort": {"median_s": 1.0}, "rule": {"median_s": 8.0}, "rule_fast": {"median_s": 1.0},
    })

    assert record["query_s"] == 10.0
    assert record["pruning_s"] == pytest.approx(13.5)
    assert record["ratio"] == pytest.approx(1.35)
    assert record["fast_pruning_s"] == pytest.approx(2.6)
    assert record["fast_ratio"] == pytest.approx(0.26)
    assert record["within_budget"] is True, "2.6 s of pruning on a 10 s query"

    too_slow = cost.budget_record({
        "tier": {"median_s": 10.0}, "positions": {"median_s": 16.0},
        "positions_array": {"median_s": 0.1}, "cm": {"median_s": 0.1},
        "sort": {"median_s": 0.6}, "rule": {"median_s": 15.0}, "rule_fast": {"median_s": 1.5},
    })
    assert too_slow["ratio"] > 1.0 and too_slow["fast_ratio"] < 1.0


class _Leads:
    """The rule's leads, asserting that the faster rule returned the same ones."""

    def __init__(self, slow: np.ndarray, fast: np.ndarray) -> None:
        self._slow = slow
        self._fast = fast

    def tolist(self) -> list[int]:
        differing = np.flatnonzero(self._slow != self._fast)
        assert differing.size == 0, f"the faster rule differs from the rule at rows {differing[:5]}"
        return self._slow.tolist()


def _leads(*, chromosome: list[str], position: list[int], cm: list[float],
           abs_z: list[float], variant_index: list[int]) -> _Leads:
    arguments = (np.array(chromosome), np.array(position), np.array(cm, dtype="float64"),
                 np.array(abs_z, dtype="float64"), np.array(variant_index))
    return _Leads(rule.locus_leads(*arguments, d_cm=0.5, w_bp=1_000_000),
                  cost.fast_locus_leads(*arguments, d_cm=0.5, w_bp=1_000_000))
