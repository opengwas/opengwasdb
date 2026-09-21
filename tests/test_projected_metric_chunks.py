"""Parity between the row-wise and blocked metrics projections (issue #209).

`stream_projected_metric_chunks` exists only to be faster than
`stream_projected_metrics`; the moment it also answers differently it is worse
than the slow thing it replaces, because nothing downstream would notice. Every
test here asserts the two produce the same rows, in the same order, with the
row-wise projection as the reference.

The fixtures deliberately reuse the shapes issue #179 and #209 already found
worth covering -- reordered columns, the legacy `hm_*` layout, ragged and quoted
rows, invalid alleles, orientation and duplicates -- plus the cases the blocked
path can get wrong on its own: a chromosome spelled like a missing-value token,
a position that is a float literal, and a block boundary falling mid-file.
"""

from __future__ import annotations

import gzip
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from opengwasdb.readers.gwas_ssf import _METRICS_COLUMNS
from opengwasdb.readers.gwas_vcf import is_palindromic
from opengwasdb.readers.tabular import (
    MetricsChunk,
    stream_projected_metric_chunks,
    stream_projected_metrics,
)

_HEADER = (
    "chromosome base_pair_location effect_allele other_allele beta standard_error"
    " effect_allele_frequency"
).split()


def _write(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")
    return path


def _number(value: float | None) -> float | None:
    """One statistic as a comparable value; `NaN` and `None` are both absent."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _row_wise(path: Path) -> list[tuple[object, ...]]:
    return [
        (
            row.alid,
            row.flipped,
            is_palindromic(row.ref, row.alt),
            _number(row.af_alt),
            _number(row.beta),
            _number(row.se),
        )
        for row in stream_projected_metrics(path, _METRICS_COLUMNS)
    ]


def _blocked(path: Path, chunk_rows: int = 1_000_000) -> list[tuple[object, ...]]:
    signatures: list[tuple[object, ...]] = []
    for chunk in stream_projected_metric_chunks(path, _METRICS_COLUMNS, chunk_rows=chunk_rows):
        assert isinstance(chunk, MetricsChunk)
        for index in range(len(chunk)):
            signatures.append(
                (
                    chunk.alid[index],
                    bool(chunk.flipped[index]),
                    bool(chunk.palindromic[index]),
                    _number(chunk.af_alt[index]),
                    _number(chunk.beta[index]),
                    _number(chunk.se[index]),
                )
            )
    return signatures


def _assert_parity(path: Path, *, chunk_rows: int = 1_000_000, expected_rows: int) -> None:
    reference = _row_wise(path)
    assert len(reference) == expected_rows, (
        "the fixture must project the number of rows the test claims, or it is "
        "asserting parity over a file neither path really reads"
    )
    assert _blocked(path, chunk_rows) == reference


def _ordinary_rows() -> list[list[str]]:
    """Normalisation, orientation, missing values and dropped rows."""
    return [
        ["1", "100", "A", "G", "0.2", "0.1", "0.05"],
        ["chr2", "200", "C", "G", "0.3", "NA", "0.08"],  # palindromic, se absent
        ["3", "300", "a", "c", "0.4", "0.2", "0.03"],  # lower case, flipped identity
        ["4", "400", "AT", "A", "0.5", "0.3", "0.04"],  # indel via normalise_allele
        ["5", "500", "G", "G", ".", "", "inf"],  # identical alleles: dropped
        ["6", "0", "A", "C", "0.1", "0.1", "0.1"],  # non-positive position: dropped
        ["X", "600", "T", "A", "0.6", "0.1", "0.02"],
        ["7", "700", "N", "A", "0.1", "0.1", "0.1"],  # invalid allele: dropped
        ["8", "800", "A", "C", "nan", "0.1", "0.0"],
        ["8", "800", "A", "C", "0.25", "0.1", "0.0"],  # duplicate ALID
    ]


def test_ordinary_projection_parity(tmp_path: Path) -> None:
    path = _write(tmp_path / "ordinary.tsv.gz", _HEADER, _ordinary_rows())

    _assert_parity(path, expected_rows=7)


def test_reordered_and_extra_columns_parity(tmp_path: Path) -> None:
    """Columns are resolved by name, and unread columns are not read."""
    header = [
        "unused_a", "effect_allele_frequency", "other_allele", "chromosome", "beta",
        "base_pair_location", "effect_allele", "unused_b", "standard_error",
    ]
    rows = [
        ["x", "0.2", "A", "1", "0.1", "100", "G", "y", "0.05"],
        ["x", "0.3", "C", "2", "0.2", "200", "T", "y", "0.06"],
    ]
    path = _write(tmp_path / "reordered.tsv.gz", header, rows)

    _assert_parity(path, expected_rows=2)


def test_legacy_hm_prefixed_layout_parity(tmp_path: Path) -> None:
    """The `hm_*` layout carries the same readable plain columns."""
    header = [
        "hm_variant_id", "hm_rsid", "hm_chrom", "hm_pos", "hm_other_allele",
        "hm_effect_allele", "hm_beta", "hm_effect_allele_frequency", "hm_code",
        "chromosome", "base_pair_location", "variant_id", "effect_allele", "beta",
        "standard_error", "other_allele", "effect_allele_frequency",
    ]
    rows = [
        ["10_1_A_G", "rs1", "10", "1", "A", "G", "0.1", "0.2", "7",
         "10", "1", "rs1", "G", "0.1", "0.03", "A", "0.2"],
        ["10_2_C_T", "rs2", "10", "2", "C", "T", "0.2", "0.3", "7",
         "10", "2", "rs2", "T", "0.2", "0.04", "C", "0.3"],
    ]
    path = _write(tmp_path / "legacy.tsv.gz", header, rows)

    _assert_parity(path, expected_rows=2)


def test_ragged_and_quoted_rows_parity(tmp_path: Path) -> None:
    """Short rows drop or lose statistics; a quoted tab stays one cell."""
    rows = [
        ["1", "100", "A", "G", "0.1", "0.05", "0.2"],
        ["1", "200", "A"],  # too few identity cells: dropped
        ['1', "300", '"A\tG"', "C", "0.1", "0.05", "0.2"],  # quoted tab: invalid allele
        ["1", "400", "A", "C", "0.1", "0.05"],  # frequency cell absent
    ]
    path = _write(tmp_path / "ragged.tsv.gz", _HEADER, rows)

    _assert_parity(path, expected_rows=2)


def test_chromosome_spelled_like_a_missing_value_is_kept(tmp_path: Path) -> None:
    """`NA` names a chromosome the row-wise projection keeps, so this one must.

    The blocked path reads the identity columns through pandas, whose default
    missing-value tokens include `NA`, `NULL` and `None`. Letting those become
    missing would silently drop rows the reference projection reports.
    """
    rows = [
        ["NA", "100", "A", "G", "0.1", "0.05", "0.2"],
        ["NULL", "200", "A", "C", "0.1", "0.05", "0.2"],
        ["None", "300", "A", "C", "0.1", "0.05", "0.2"],
        ["1", "400", "A", "C", "0.1", "0.05", "0.2"],
    ]
    path = _write(tmp_path / "na_chromosome.tsv.gz", _HEADER, rows)

    reference = _row_wise(path)
    assert [row[0] for row in reference] == [
        "NA:100:A:G", "NULL:200:A:C", "None:300:A:C", "1:400:A:C"
    ], "the fixture only means something if the row-wise projection keeps all four"
    assert _blocked(path) == reference


def test_position_that_is_not_an_integer_literal_is_dropped(tmp_path: Path) -> None:
    """`int()` rejects `100.5` and `1e5`; a float parser would accept both."""
    rows = [
        ["1", "100.5", "A", "G", "0.1", "0.05", "0.2"],
        ["1", "1e5", "A", "C", "0.1", "0.05", "0.2"],
        ["1", "-300", "A", "C", "0.1", "0.05", "0.2"],
        ["1", " 400 ", "A", "C", "0.1", "0.05", "0.2"],  # int() strips whitespace
        ["1", "00500", "A", "C", "0.1", "0.05", "0.2"],  # int() drops leading zeros
    ]
    path = _write(tmp_path / "positions.tsv.gz", _HEADER, rows)

    reference = _row_wise(path)
    assert [row[0] for row in reference] == ["1:400:A:C", "1:500:A:C"], (
        "the fixture only means something if the row-wise projection drops the "
        "float, exponent and negative positions and keeps the other two"
    )
    assert _blocked(path) == reference


def test_palindrome_test_uses_the_verbatim_labels(tmp_path: Path) -> None:
    """A padded `C`/`G` pair is not palindromic row-wise, so it must not be here.

    `is_palindromic` is handed the source's own cell text, which
    `normalise_allele` would have stripped. Deciding strand ambiguity from the
    normalised allele instead is tidier and disagrees, and disagreeing silently
    changes which sites the ancestry fit is allowed to read.
    """
    rows = [
        ["1", "100", " C", "G", "0.1", "0.05", "0.2"],
        ["1", "200", "C", "G", "0.1", "0.05", "0.2"],
    ]
    path = _write(tmp_path / "padded.tsv.gz", _HEADER, rows)

    reference = _row_wise(path)
    assert [(row[0], row[2]) for row in reference] == [
        ("1:100:C:G", False), ("1:200:C:G", True)
    ], "the fixture must pair a padded allele against an unpadded one to mean anything"
    assert _blocked(path) == reference


def test_statistic_usability_rules_match(tmp_path: Path) -> None:
    """A frequency outside [0, 1], a non-finite beta and a non-positive SE are absent."""
    rows = [
        ["1", "100", "A", "G", "0.1", "0.05", "1.5"],  # frequency out of range
        ["1", "200", "A", "C", "inf", "0.05", "0.2"],  # non-finite beta
        ["1", "300", "A", "C", "0.1", "0", "0.2"],  # non-positive SE
        ["1", "400", "A", "C", "0.1", "-0.05", "0.2"],  # negative SE
        ["1", "500", "A", "C", "0.1", "0.05", "0"],  # zero is a usable frequency
    ]
    path = _write(tmp_path / "statistics.tsv.gz", _HEADER, rows)

    reference = _row_wise(path)
    assert [(row[3], row[4], row[5]) for row in reference] == [
        (None, 0.1, 0.05), (0.2, None, 0.05), (0.2, 0.1, None),
        (0.2, 0.1, None), (0.0, 0.1, 0.05),
    ], "the fixture must exercise each usability rule before parity means anything"
    assert _blocked(path) == reference


def test_long_decimals_keep_the_value_float_would_give(tmp_path: Path) -> None:
    """A statistic must be the same double `float()` returns, to the last place.

    pandas' default C float converter is a digit less accurate than `float()`.
    `0.37411300000000003` is a real `effect_allele_frequency` from
    `GCST90502916`, one of nine rows in 1,145,324 where the two disagreed; the
    difference is invisible in every report the value reaches.
    """
    rows = [
        ["1", "100", "A", "G", "0.29344000000000003", "0.0565002", "0.37411300000000003"],
        ["1", "200", "A", "C", "-0.0102029", "0.119662", "0.4778369999999999"],
    ]
    path = _write(tmp_path / "precision.tsv.gz", _HEADER, rows)

    reference = _row_wise(path)
    assert [row[3] for row in reference] == [
        float("0.37411300000000003"), float("0.4778369999999999")
    ], "the fixture must carry decimals whose nearest double needs every digit"
    assert _blocked(path) == reference


@pytest.mark.parametrize("chunk_rows", [1, 2, 3, 7, 1000])
def test_block_size_does_not_change_the_projection(tmp_path: Path, chunk_rows: int) -> None:
    """Blocking is a memory bound, not a semantic one.

    Allele and chromosome normalisation run per *category*, and a block carries
    only the categories its own rows use -- so a boundary that splits a file
    between two blocks is exactly where a category-indexed projection would go
    wrong.
    """
    rows = [
        [str(1 + index % 22), str(100 + index), allele, "C", "0.1", "0.05", "0.2"]
        for index, allele in enumerate(["A", "G", "T", "AT", "N", "a", "c"] * 3)
    ]
    path = _write(tmp_path / "blocked.tsv.gz", _HEADER, rows)

    _assert_parity(path, chunk_rows=chunk_rows, expected_rows=15)


def test_repeated_reads_are_deterministic(tmp_path: Path) -> None:
    path = _write(tmp_path / "deterministic.tsv.gz", _HEADER, _ordinary_rows())

    assert _blocked(path) == _blocked(path)


def test_duplicate_projected_column_raises(tmp_path: Path) -> None:
    """Two columns with one projected name would silently read the wrong one."""
    header = [*_HEADER, "beta"]
    rows = [["1", "100", "A", "G", "0.1", "0.05", "0.2", "0.9"]]
    path = _write(tmp_path / "duplicate.tsv.gz", header, rows)

    with pytest.raises(ValueError, match="appears 2 times"):
        list(stream_projected_metric_chunks(path, _METRICS_COLUMNS))
