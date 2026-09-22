"""Parity fixtures for the issue #209 external-decompressor parser prototype.

The prototype in `benchmarks/benchmark_resolver_evidence_scan.py` decompresses a
source with an argv-safe external `gzip -dc`/`pigz -dc` and projects the same
bytes the production `stream_projected_metrics` projection does. A parser that
silently changes eligibility, normalization, orientation or missing-value
semantics is worse than a slow one, so every fixture here asserts the two
produce identical rows -- the production path being the reference.

`gzip` is required (the prototype's decompressor); `pigz` is exercised when it
is on PATH rather than assumed.
"""

from __future__ import annotations

import gzip
import shutil
from collections.abc import Sequence
from pathlib import Path

from benchmarks.benchmark_resolver_evidence_scan import (
    _external_rows,
    _row_signature,
    _take,
)
from opengwasdb.readers.gwas_ssf import _METRICS_COLUMNS
from opengwasdb.readers.tabular import stream_projected_metrics

_BINARIES = ["gzip"]
if shutil.which("pigz"):
    _BINARIES.append("pigz")
_PROJECTION = _METRICS_COLUMNS


def _write(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(row))
            fh.write("\n")
    return path


def _ordinary_rows() -> list[list[str]]:
    """Rows exercising normalization, orientation and missing values."""
    return [
        ["1", "100", "A", "G", "0.2", "0.1", "0.05"],
        ["chr2", "200", "C", "T", "0.3", "NA", "0.08"],  # palindromic: unusable
        ["3", "300", "a", "c", "0.4", "0.2", "0.03"],  # lower case, flipped identity
        ["4", "400", "AT", "A", "0.5", "0.3", "0.04"],  # indel via normalise_allele
        ["5", "500", "G", "G", ".", "", "inf"],  # identical alleles: dropped
        ["6", "0", "A", "C", "0.1", "0.1", "0.1"],  # non-positive position: dropped
        ["X", "600", "T", "A", "0.6", "0.1", "0.02"],
        ["7", "700", "N", "A", "0.1", "0.1", "0.1"],  # invalid allele
        ["8", "800", "A", "C", "nan", "0.1", "0.0"],
        ["8", "800", "A", "C", "0.25", "0.1", "0.0"],  # duplicate ALID
    ]


def _assert_parity(path: Path, limit: int | None = None) -> None:
    assert _BINARIES, "the parity fixtures require gzip on PATH"
    expected = [
        _row_signature(row)
        for row in _take(stream_projected_metrics(path, _PROJECTION), limit)
    ]
    assert expected, "the fixture must yield at least one projected row to mean anything"
    for binary in _BINARIES:
        actual = [_row_signature(row) for row in _take(_external_rows(path, binary), limit)]
        assert actual == expected, f"{binary} projection differs from production"


def test_ordinary_projection_parity(tmp_path: Path) -> None:
    header = list(
        "chromosome base_pair_location effect_allele other_allele beta standard_error"
        " effect_allele_frequency".split()
    )
    path = _write(tmp_path / "ordinary.tsv.gz", header, _ordinary_rows())

    _assert_parity(path)


def test_reordered_and_extra_columns_parity(tmp_path: Path) -> None:
    """Columns are resolved by name, never by position."""
    header = [
        "unused_a",
        "effect_allele_frequency",
        "other_allele",
        "chromosome",
        "beta",
        "base_pair_location",
        "effect_allele",
        "unused_b",
        "standard_error",
    ]
    rows = [
        ["x", "0.2", "A", "1", "0.1", "100", "G", "y", "0.05"],
        ["x", "0.3", "C", "2", "0.2", "200", "T", "y", "0.06"],
    ]
    path = _write(tmp_path / "reordered.tsv.gz", header, rows)

    _assert_parity(path)


def test_legacy_hm_prefixed_layout_parity(tmp_path: Path) -> None:
    """The `hm_*` GWAS-SSF layout carries the same readable plain columns."""
    header = [
        "hm_variant_id",
        "hm_rsid",
        "hm_chrom",
        "hm_pos",
        "hm_other_allele",
        "hm_effect_allele",
        "hm_beta",
        "hm_effect_allele_frequency",
        "hm_code",
        "chromosome",
        "base_pair_location",
        "variant_id",
        "effect_allele",
        "beta",
        "standard_error",
        "other_allele",
        "effect_allele_frequency",
    ]
    rows = [
        [
            "10_1_A_G", "rs1", "10", "1", "A", "G", "0.1", "0.2", "7",
            "10", "1", "rs1", "G", "0.1", "0.03", "A", "0.2",
        ],
        [
            "10_2_C_T", "rs2", "10", "2", "C", "T", "0.2", "0.3", "7",
            "10", "2", "rs2", "T", "0.2", "0.04", "C", "0.3",
        ],
    ]
    path = _write(tmp_path / "legacy.tsv.gz", header, rows)

    _assert_parity(path)


def test_ragged_and_quoted_rows_parity(tmp_path: Path) -> None:
    """Short rows are dropped by both paths; quoted cells are re-split by both."""
    header = list(
        "chromosome base_pair_location effect_allele other_allele beta standard_error"
        " effect_allele_frequency".split()
    )
    rows = [
        ["1", "100", "A", "G", "0.1", "0.05", "0.2"],
        ["1", "200", "A"],  # ragged: too few identity cells
        ['1', "300", '"A\tG"', "C", "0.1", "0.05", "0.2"],  # quoted tab inside a cell
        ["1", "400", "A", "C", "0.1", "0.05"],
    ]
    path = _write(tmp_path / "ragged.tsv.gz", header, rows)

    _assert_parity(path)


def test_early_stop_parity_and_decompressor_reaping(tmp_path: Path) -> None:
    """A prefix parse stops early and reaps the decompressor without an error."""
    header = list(
        "chromosome base_pair_location effect_allele other_allele beta standard_error"
        " effect_allele_frequency".split()
    )
    rows = [[str(chrom), str(index), "A", "C", "0.1", "0.05", "0.2"] for chrom, index in
            ((chrom, 100 + i) for chrom in range(1, 23) for i in range(50))]
    path = _write(tmp_path / "prefix.tsv.gz", header, rows)

    _assert_parity(path, limit=25)


def test_determinism_across_repeated_full_scans(tmp_path: Path) -> None:
    header = list(
        "chromosome base_pair_location effect_allele other_allele beta standard_error"
        " effect_allele_frequency".split()
    )
    path = _write(tmp_path / "deterministic.tsv.gz", header, _ordinary_rows())

    first = [_row_signature(row) for row in stream_projected_metrics(path, _PROJECTION)]
    second = [_row_signature(row) for row in stream_projected_metrics(path, _PROJECTION)]

    assert first == second
    _assert_parity(path)
