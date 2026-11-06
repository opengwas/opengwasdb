"""One Locus Lead per LD block (prototypes/locus_leads, #409).

Blocks are half-open, so a row at a shared endpoint belongs to the block that
starts there. A block's lead is its strongest row in the Locus Lead ranking.
Rows outside every block are counted, and each stretch between blocks holds
its own lead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from prototype_modules import load_prototype

blocks = load_prototype("locus_leads", "blocks")

TSV = """chr\tstart\tend\tancestry
1\t100\t200\tEUR
1\t200\t300\tEUR
1\t400\t500\tEUR
1\t0\t1000\tAFR
2\t100\t200\tEUR
"""


def _blocks(tmp_path: Path, text: str = TSV) -> object:
    path = tmp_path / "ld_regions.tsv"
    path.write_text(text)
    return blocks.read_blocks(path, "EUR")


def test_a_row_on_a_shared_endpoint_belongs_to_the_block_that_starts_there(tmp_path: Path) -> None:
    eur = _blocks(tmp_path)
    position = [199, 200, 299]
    assert position[1] == 200 and "1\t100\t200\tEUR\n1\t200\t300\tEUR" in TSV

    result = blocks.block_leads(np.array(["1"] * 3), np.array(position),
                                np.array([5.0, 6.0, 4.0]), np.arange(3), eur)

    np.testing.assert_array_equal(result.lead_of, [0, 1, 1])
    assert result.outside_rows == 0


def test_a_blocks_lead_is_its_strongest_row_ties_to_the_lower_variant_index(
        tmp_path: Path) -> None:
    eur = _blocks(tmp_path)

    result = blocks.block_leads(np.array(["1"] * 4), np.array([410, 420, 430, 440]),
                                np.array([2.0, 7.0, 7.0, 3.0]), np.array([1, 9, 3, 2]), eur)

    np.testing.assert_array_equal(result.lead_of, [2, 2, 2, 2])


def test_rows_outside_every_block_are_counted_and_each_stretch_between_blocks_leads_once(
        tmp_path: Path) -> None:
    eur = _blocks(tmp_path)
    # Before chr1's first block, between its second and third, at and after its last
    # block's end, and on chromosome 3, which has no EUR block. The AFR block covering
    # chr1 0-1000 is not read.
    chromosome = ["1", "1", "1", "1", "1", "3", "3"]
    position = [50, 350, 360, 500, 900, 150, 5_000]
    abs_z = [6.0, 5.0, 8.0, 7.0, 9.0, 4.0, 3.0]

    result = blocks.block_leads(np.array(chromosome), np.array(position), np.array(abs_z),
                                np.arange(7), eur)

    np.testing.assert_array_equal(result.lead_of, [0, 2, 2, 4, 4, 5, 5])
    assert result.outside_rows == 7
    assert result.outside_stretches == 4


def test_blocks_that_overlap_or_an_ancestry_with_none_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overlap"):
        _blocks(tmp_path, TSV + "2\t150\t250\tEUR\n")
    with pytest.raises(ValueError, match="no SAS block"):
        blocks.read_blocks(tmp_path / "ld_regions.tsv", "SAS")
