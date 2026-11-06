"""What genetic-distance pruning costs (#406).

Held here directly: that a slice's timed record names every stage the
measurement asks for and carries the machine, date, commit and load it was taken
under, with one load record per repeat; that a child process's peak RSS is seen
and a peak nobody observed is missing rather than zero; that no total charges a
sort the rule already does; that pruning is divided by three different clumping
bills; and that the faster rule gives the same leads as `rule.locus_leads`,
including where a lead from one block owns rows in the next.
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
        assert len(entry["loads"]) == 2, "one load record per repeat, not one per stage"
        for load in entry["loads"]:
            assert set(load) == {"one", "five", "fifteen"}
        assert entry["peak_source"] in {"process", "children"}
        assert isinstance(entry["peak_mb"], float) and entry["peak_mb"] > 0
        assert isinstance(entry["rss_start_mb"], float) and entry["rss_start_mb"] > 0
        assert entry["rss_observations"] >= 1

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


def test_the_child_rss_sampler_sees_a_childs_peak_and_not_a_fork_clones() -> None:
    small = cost.peak_child_rss([sys.executable, "-c", _SPIN])
    big = cost.peak_child_rss([sys.executable, "-c", _ALLOCATE_AND_SPIN])

    assert small["peak_mb"] < 100.0, (
        f"a child that allocates nothing peaked at {small['peak_mb']:.1f} MB")
    assert 190.0 <= big["peak_mb"] <= 330.0, (
        f"a child holding 200 MB peaked at {big['peak_mb']:.1f} MB")
    assert big["observations"] > 0


def test_a_child_nobody_polled_is_a_missing_peak_rather_than_a_zero() -> None:
    # The poll interval is longer than the child's life, so no poll can see it.
    unobserved = cost.peak_child_rss([sys.executable, "-c", "pass"], interval=10.0)

    assert unobserved["peak_mb"] is None, "no observation is not 0 MB"
    assert unobserved["observations"] == 0
    assert unobserved["interval_s"] == 10.0


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


def test_the_faster_rule_carries_a_leads_ownership_across_block_boundaries() -> None:
    # Thirty clusters 3 Mb apart, so each is its own lead and W = 1 Mb keeps them
    # apart. Every row shares abs_z, so the rule's order is by Store Variant Index,
    # which interleaves the clusters: the 30 leads are accepted inside the first
    # block and own rows in every block after it.
    clusters, width = 30, 400
    position = np.concatenate([k * 3_000_000 + np.arange(width) for k in range(clusters)])
    chromosome = np.repeat("1", position.size)
    cm = position / 10_000_000
    variant_index = np.concatenate([np.arange(width) * clusters + k for k in range(clusters)])
    assert position.size > cost.BLOCK_MIN, "the fixture must span more than one block"
    assert np.all(np.diff(cm) >= 0), "fixture cM must be monotone"

    leads = _leads(chromosome=chromosome.tolist(), position=position.tolist(),
                   cm=cm.tolist(), abs_z=[40.0] * position.size,
                   variant_index=variant_index.tolist())

    kept = leads.tolist()
    lead_rows = [row for row in range(len(kept)) if kept[row] == row]
    assert len(lead_rows) == clusters, "one lead per cluster"
    owned_beyond_the_first_block = [
        row for row in range(cost.BLOCK_MIN, len(kept))
        if kept[row] != row and kept[row] in lead_rows
    ]
    assert len(owned_beyond_the_first_block) > cost.BLOCK_MIN, (
        "rows after the first block must be owned by a lead accepted before it")


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


def test_the_budget_does_not_charge_a_sort_the_rule_already_does() -> None:
    record = cost.budget_record({
        "tier": {"median_s": 10.0}, "resolved_query": {"median_s": 20.0},
        "positions": {"median_s": 4.0}, "positions_array": {"median_s": 0.1},
        "cm": {"median_s": 0.5}, "sort": {"median_s": 1.0}, "rule": {"median_s": 8.0},
        "rule_fast": {"median_s": 1.0},
    })

    assert record["query_s"] == 10.0
    assert record["resolved_query_s"] == 20.0
    # The separate sort stage is not added: both rules sort inside their own time.
    assert record["pruning_s"] == pytest.approx(12.5)
    assert record["ratio"] == pytest.approx(1.25)
    assert record["fast_pruning_s"] == pytest.approx(1.6)
    assert record["fast_ratio"] == pytest.approx(0.16)
    assert record["fast_ratio_of_resolved"] == pytest.approx(0.08)
    assert record["charged"]["as_written"] == ["positions", "cm", "rule"]
    assert record["charged"]["fast"] == ["positions_array", "cm", "rule_fast"]
    assert "sort" in record["sort_note"]
    assert record["within_budget"] is True, "1.6 s of pruning on a 10 s query"

    too_slow = cost.budget_record({
        "tier": {"median_s": 10.0}, "resolved_query": {"median_s": 20.0},
        "positions": {"median_s": 16.0}, "positions_array": {"median_s": 0.1},
        "cm": {"median_s": 0.1}, "sort": {"median_s": 0.6}, "rule": {"median_s": 15.0},
        "rule_fast": {"median_s": 1.5},
    })
    assert too_slow["ratio"] > 1.0 and too_slow["fast_ratio"] < 1.0


def test_the_ratio_reports_clumping_end_to_end_matched_and_plink2_alone() -> None:
    record = cost.ratio_record(
        pruning={"cm": {"median_s": 0.5}, "sort": {"median_s": 0.25},
                 "rule": {"median_s": 4.0}, "rule_fast": {"median_s": 1.0}},
        clumping={"matching_s": 3.0, "shared_cm_s": 2.0, "ranked_s": 0.5, "plink2_s": 10.0},
        rows=1_000,
    )

    assert record["rows"] == 1_000
    # The sort is not charged: the rule's own time already includes one.
    assert record["pruning_s"] == pytest.approx(4.5)
    assert record["fast_pruning_s"] == pytest.approx(1.5)
    assert record["clumping"] == {
        "plink2_s": 10.0, "matched_s": 13.5, "end_to_end_s": 15.5,
    }
    assert record["shared_set_split_s"] == pytest.approx(5.0)
    # Ratios are rounded to four places, as every timed figure in the artifact is.
    assert record["ratios"] == pytest.approx(
        {"plink2_only": 0.45, "matched": 0.3333, "end_to_end": 0.2903})
    assert record["ratios_fast"] == pytest.approx(
        {"plink2_only": 0.15, "matched": 0.1111, "end_to_end": 0.0968})
    assert "sort" in record["sort_note"]


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
