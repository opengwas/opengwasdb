"""One Analysis through the sweep, on a synthetic tier and a tiny pgen panel (#403)."""

from __future__ import annotations

from pathlib import Path

import locus_leads_panel
import numpy as np
import pytest
from prototype_modules import load_prototype

genetic_map = load_prototype("locus_leads", "genetic_map")
sweep = load_prototype("locus_leads", "sweep")
panel = load_prototype("locus_leads", "panel")
run = load_prototype("locus_leads", "run")

MAP = genetic_map.LoadedMap(
    genetic_map.GeneticMap({"1": (np.array([1, 10_000_000]), np.array([0.0, 100.0]))}),
    {"assembly": "GRCh38"})
# Rows 1-3 carry one signal (identical genotypes); row 4 is a second, unlinked one.
SIGNAL = ["0/0", "1/1", "0/0", "1/1"] * 10
OTHER = ["0/0", "0/0", "1/1", "1/1"] * 10  # r^2 = 0 with SIGNAL


def _tier(positions: list[int], abs_z: list[float]) -> object:
    n = len(positions)
    return run.TierRows(
        np.array(["1"] * n), np.array(positions),
        np.array([f"1:{p}:A:G" for p in positions]), np.array(["A"] * n), np.array(["G"] * n),
        np.array(abs_z), np.arange(n))


def _identity(rows: int) -> dict:
    return {"store": {"reference_assembly": "GRCh38", "store_id": "S"},
            "analysis": {"analysis_id": "a"}, "query": {"threshold": 5e-8, "observed": rows}}


@pytest.fixture(scope="module")
def panel_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("sweep_panel")
    variants = [(p, "A", "G") for p in (100_000, 200_000, 300_000, 5_000_000)]
    locus_leads_panel.write_panel(directory, "1", variants, [SIGNAL, SIGNAL, SIGNAL, OTHER],
                                  assembly="gnomAD_GRCh38")
    return directory


# HapMap3 holds the first, third and fourth variants of the panel.
HM3 = frozenset({"1:100000:A:G", "1:300000:A:G", "1:5000000:A:G"})


def _analyse(directory: Path, tier: object, work: Path, **extra: object) -> dict:
    return sweep.analyse(tier, _identity(tier.chromosome.size), MAP, panel.Panel(directory),
                         work_dir=work, threads=1, spikes=sweep.EUR_SPIKES,
                         configs=sweep.grid_configs(), hm3=HM3, **extra)


def test_every_config_is_compared_on_the_same_rows_and_pairs_carry_their_r2(
    panel_dir: Path, tmp_path: Path,
) -> None:
    tier = _tier([100_000, 200_000, 300_000, 5_000_000, 9_000_000], [9.0, 8.0, 7.0, 6.0, 5.0])
    record = _analyse(panel_dir, tier, tmp_path)
    assert record["rows"]["shared_set"] == 4  # the row at 9,000,000 is off the panel
    assert len(record["configs"]) == 24
    # The map runs 10 cM per Mb: rows 100 kb apart are 1 cM apart, beyond D = 0.1 cM.
    cm = next(c for c in record["configs"] if c["d_cm"] == 0.1 and c["w_bp"] == 250_000)
    bp = next(c for c in record["configs"] if c["d_cm"] == "bp_only" and c["w_bp"] == 250_000)
    assert bp["by_r2"]["0.01"]["locus_leads"] == 2
    assert cm["by_r2"]["0.01"]["locus_leads"] == 4
    # plink2 keeps rows 1-3 in one clump (r^2 = 1), so rows 2 and 3 are splits, at r^2 = 1.
    assert cm["by_r2"]["0.01"]["spurious_splits"] == 2
    histogram = cm["by_r2"]["0.01"]["spurious_split_pairs"]["r2_histogram"]
    assert histogram["[0.5,inf)"] == 2
    chance = record["chance_ld"]
    assert (chance["variants"], chance["pairs"]) == (2, 0)  # both index variants on chr1


def test_an_analysis_with_too_few_rows_in_the_shared_set_is_refused_with_its_breakdown(
    panel_dir: Path, tmp_path: Path,
) -> None:
    tier = _tier([100_000, 200_000, 7_000_000, 8_000_000, 9_000_000], [9.0, 8.0, 7.0, 6.0, 5.0])
    with pytest.raises(sweep.SharedSetTooSmall, match="40.0%") as error:
        _analyse(panel_dir, tier, tmp_path)
    assert error.value.record["partition"]["absent_at_position"] == 3


def test_leads_are_recomputed_on_the_hapmap3_rows_without_reclumping(
    panel_dir: Path, tmp_path: Path,
) -> None:
    tier = _tier([100_000, 200_000, 300_000, 5_000_000], [9.0, 8.0, 7.0, 6.0])
    record = _analyse(panel_dir, tier, tmp_path)
    assert record["density"] == {"rows_full": 4, "rows_hm3": 3}
    at = {(c["d_cm"], c["w_bp"]): c["density"] for c in record["configs"]}
    # D 0.1 cM at W 250 kb: all four rows are leads; on HM3 rows (1, 3, 4) all three are.
    assert at[(0.1, 250_000)] == {"leads_full": 4, "leads_hm3": 3}
    # Bp-only at W 250 kb: rows 1-2 are 100 kb apart, so row 2 merges; rows 3 and 4 stay leads.
    assert at[("bp_only", 250_000)] == {"leads_full": 2, "leads_hm3": 2}
    # Bp-only at W 10 Mb merges everything into the strongest row, with or without row 2.
    assert at[("bp_only", 10_000_000)] == {"leads_full": 1, "leads_hm3": 1}


def test_the_pair_records_hold_each_disagreement_of_the_chosen_configs_with_its_r2(
    panel_dir: Path, tmp_path: Path,
) -> None:
    tier = _tier([100_000, 200_000, 300_000, 5_000_000], [9.0, 8.0, 7.0, 6.0])
    record = _analyse(panel_dir, tier, tmp_path, pair_configs=((0.1, 250_000),))
    rows = record["pair_records"]
    assert {(r["d_cm"], r["w_bp"]) for r in rows} == {(0.1, 250_000)}
    assert [(r["kind"], r["alid"], r["stronger_alid"]) for r in rows] == [
        ("spurious_split", "1:200000:A:G", "1:100000:A:G"),
        ("spurious_split", "1:300000:A:G", "1:100000:A:G")]
    assert [r["r2"] for r in rows] == pytest.approx([1.0, 1.0])
    assert [r["distance_bp"] for r in rows] == [100_000, 200_000]
    assert rows[0]["abs_z"] == 8.0 and rows[0]["stronger_abs_z"] == 9.0
    assert rows[0]["distance_cm"] == pytest.approx(1.0)
    assert {r["analysis"] for r in rows} == {"a"} and {r["store"] for r in rows} == {"S"}
