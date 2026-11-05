"""Pair r^2 from exported dosages, against plink2's own `--r2-unphased` (#403)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import locus_leads_panel
import numpy as np
import pytest
from prototype_modules import load_prototype

ld = load_prototype("locus_leads", "ld")
panel = load_prototype("locus_leads", "panel")

# Eight samples; one missing call in the third variant. Chosen so the four pairs
# below have distinct r^2 (0, 1, and two in between).
CALLS = [
    ["0/0", "0/0", "0/1", "0/1", "1/1", "1/1", "0/0", "0/1"],
    ["0/0", "0/0", "0/1", "0/1", "1/1", "1/1", "0/0", "0/1"],
    ["0/1", "0/0", "./.", "0/1", "1/1", "0/0", "0/0", "1/1"],
    ["0/0", "1/1", "0/0", "1/1", "0/1", "0/1", "0/0", "1/1"],
]
VARIANTS = [(100, "A", "G"), (200, "C", "T"), (300, "A", "C"), (400, "G", "T")]
IDS = [f"1:{p}:{r}:{a}" for p, r, a in VARIANTS]


@pytest.fixture(scope="module")
def panel_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("ld_panel")
    locus_leads_panel.write_panel(directory, "1", VARIANTS, CALLS)
    return directory


def _plink2_r2(directory: Path, first: str, second: str, work: Path) -> float:
    out = work / f"{first}_{second}".replace(":", "_")
    subprocess.run([panel.plink2(), "--pfile", str(directory / "chr1"), "vzs", "--r2-unphased",
                    "--ld-window-r2", "0", "--ld-snp", first, "--ld-window-kb", "1000",
                    "--ld-window", "99999", "--out", str(out)], check=True, capture_output=True)
    for line in out.with_suffix(".vcor").read_text().splitlines()[1:]:
        cells = line.split("\t")
        if cells[5] == second:
            return float(cells[-1])
    raise AssertionError(f"plink2 reported no r2 for {first} and {second}")


def test_the_fixture_has_pairs_that_differ_and_a_missing_call() -> None:
    assert "./." in CALLS[2] and len({IDS[i] for i in range(4)}) == 4


def test_pair_r2_matches_plink2_unphased_including_the_pair_with_a_missing_call(
    panel_dir: Path, tmp_path: Path,
) -> None:
    dosage = ld.export_dosages(panel.Panel(panel_dir), "1", IDS, tmp_path)
    assert dosage.shape == (4, 8)
    for first, second in [(0, 1), (0, 2), (0, 3), (2, 3)]:
        expected = _plink2_r2(panel_dir, IDS[first], IDS[second], tmp_path)
        assert ld.pair_r2(dosage[first], dosage[second]) == pytest.approx(expected, abs=1e-5)
    assert ld.pair_r2(dosage[0], dosage[1]) == pytest.approx(1.0)
    assert 0.0 < ld.pair_r2(dosage[0], dosage[3]) < 1.0


def test_the_r2_matrix_agrees_with_pair_r2_and_leaves_the_diagonal_at_one(
    panel_dir: Path, tmp_path: Path,
) -> None:
    dosage = ld.export_dosages(panel.Panel(panel_dir), "1", IDS, tmp_path)
    matrix = ld.r2_matrix(dosage)
    assert matrix[0, 2] == pytest.approx(ld.pair_r2(dosage[0], dosage[2]))
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T)
