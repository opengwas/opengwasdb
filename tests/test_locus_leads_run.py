"""One run of the Locus Lead prototype, from the shared set onward (#402).

The rows are synthetic, so no Store is needed; the panel is a tiny pgen per
chromosome built by plink2, so matching and clumping run for real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import locus_leads_panel
import numpy as np
import pytest
from prototype_modules import load_prototype

genetic_map = load_prototype("locus_leads", "genetic_map")
panel = load_prototype("locus_leads", "panel")
run = load_prototype("locus_leads", "run")

# Chromosome 1's map; chromosome 2 is on the panel but not on the map.
MAP = genetic_map.GeneticMap({"1": (np.array([1, 10_000_000]), np.array([0.0, 1.0]))})

# Nine rows: four the panel and map both hold, then one of each way to miss.
ROWS = run.TierRows(
    chromosome=np.array(["1", "1", "1", "1", "2", "X", "1", "1", "1"]),
    position=np.array([1_000, 2_000, 3_000, 4_000, 1_000, 100, 5_000, 2_000, 6_000]),
    alid=np.array(["1:1000:A:G", "1:2000:C:T", "1:3000:A:G", "1:4000:C:T", "2:1000:A:G",
                   "X:100:C:T", "1:5000:A:G", "1:2000:A:C", "1:6000:C:CAATA"]),
    allele_1=np.array(["A", "C", "A", "C", "A", "C", "A", "A", "C"]),
    allele_2=np.array(["G", "T", "G", "T", "G", "T", "G", "C", "CAATA"]),
    abs_z=np.array([39.5, 41.25, 40.0, 39.0, 6.0, 7.0, 6.5, 6.2, 6.1]),
    variant_index=np.array([12, 5, 9, 20, 30, 40, 50, 60, 70]),
)


def _panel_directory(directory: Path, assembly: str = "GRCh38") -> Path:
    """chr1: three variants with identical genotypes, one (4,000) uncorrelated with
    them (r^2 = 0 by construction), and an insertion and a deletion sharing one allele
    set (6,000); chr2: one variant."""
    linked = ["0/0", "1/1", "0/0", "1/1"] * 10
    unlinked = ["0/0", "0/0", "1/1", "1/1"] * 10
    chr1 = [(1_000, "A", "G"), (2_000, "C", "T"), (3_000, "G", "A"), (4_000, "T", "C"),
            (6_000, "C", "CAATA"), (6_000, "CAATA", "C")]
    calls = [linked, linked, linked, unlinked, linked, unlinked]
    locus_leads_panel.write_panel(directory, "1", chr1, calls, assembly=assembly)
    locus_leads_panel.write_panel(directory, "2", [(1_000, "A", "G")], [linked], assembly=assembly)
    return directory


@pytest.fixture(scope="module")
def panel_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _panel_directory(tmp_path_factory.mktemp("panel"))


def test_every_tier_row_falls_in_exactly_one_category_and_the_shared_rows_are_returned(
    panel_dir: Path,
) -> None:
    shared, accounting = run.build_shared_set(ROWS, panel.Panel(panel_dir), MAP)

    assert accounting["partition"] == {
        "shared_set": 4, "matched_no_cm": 1, "off_panel_chromosome": 1,
        "absent_at_position": 1, "allele_mismatch": 1, "ambiguous": 1,
    }
    assert sum(accounting["partition"].values()) == accounting["tier_rows"] == 9
    assert list(shared.rows.position) == [1_000, 2_000, 3_000, 4_000]
    assert list(shared.panel_id) == ["1:1000:A:G", "1:2000:C:T", "1:3000:G:A", "1:4000:T:C"]


MAP_IDENTITY = {
    "map": "EUR.tsv.gz", "sha256": "ab" * 32, "assembly": "GRCh38",
    "archive": {"path": "original/hg38_maps.tar.gz", "md5": "45ebb10c4acf939ea03a3932c483f638"},
}


def _identity(store_assembly: str = "GRCh38") -> dict[str, dict[str, Any]]:
    return {
        "store": {"store_id": "OGS-TEST", "reference_assembly": store_assembly},
        "analysis": {"analysis_id": "synthetic-analysis"},
        "query": {"threshold": 5e-8, "observed_only": True, "rows": 9, "observed": 9,
                  "imputed": 0},
    }


def _record(panel_directory: Path, work_dir: Path, store_assembly: str = "GRCh38",
            map_assembly: str = "GRCh38") -> dict[str, Any]:
    loaded = genetic_map.LoadedMap(MAP, MAP_IDENTITY | {"assembly": map_assembly})
    record: dict[str, Any] = run.compare_tier(
        ROWS, _identity(store_assembly), loaded, panel.Panel(panel_directory), d_cm=0.5,
        w_bp=1_000_000, r2_values=[0.01], work_dir=work_dir,
    )
    return record


def test_the_record_names_its_inputs_parameters_partition_and_rates(
    panel_dir: Path, tmp_path: Path
) -> None:
    record = _record(panel_dir, tmp_path)

    assert record["store"] == _identity()["store"]
    assert record["analysis"] == {"analysis_id": "synthetic-analysis"}
    assert record["genetic_map"]["sha256"] == "ab" * 32
    assert record["genetic_map"]["archive"]["md5"] == "45ebb10c4acf939ea03a3932c483f638"
    assert record["parameters"] == {
        "d_cm": 0.5, "w_bp": 1_000_000, "r2": [0.01], "clump_kb": 10_000,
        "clump_p1": 5e-8, "clump_p2": 5e-8, "clump_r2": "unphased",
    }
    partition = record["rows"]["partition"]
    assert sum(partition.values()) == record["query"]["observed"] == 9
    # The rule keeps one lead, 2,000 (|z| 41.25); plink2 also indexes 4,000,
    # which is unlinked to it, so 4,000 is a lost signal and nothing is split.
    at_r2 = record["comparison"]["by_r2"]["0.01"]
    assert {key: at_r2[key] for key in ("rows", "locus_leads", "index_variants", "agreed",
                                        "spurious_splits", "lost_signals")} == {
        "rows": partition["shared_set"], "locus_leads": 1, "index_variants": 2, "agreed": 1,
        "spurious_splits": 0, "lost_signals": 1,
    }
    assert (at_r2["spurious_split_rate"], at_r2["lost_signal_rate"]) == (0.0, 0.5)
