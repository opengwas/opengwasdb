"""One run of the Locus Lead prototype, from the shared set onward (#402).

The rows are synthetic, so no Store is needed; the panel is a tiny pgen per
chromosome built by plink2, so matching and clumping run for real.
"""

from __future__ import annotations

from pathlib import Path

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
