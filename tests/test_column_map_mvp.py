"""dbGaP MVP raw files read through a Column Map (issue #281, stores ADR 0036).

opengwasdb-stores rewrites every dbGaP phs002453 deposit file into EBI's GWAS-SSF
before the GWAS-SSF reader can read it (`acquire-dbgap-phs002453.py`'s
`CONVERTER_AWK`). A Column Map is meant to make that rewrite unnecessary, so the
bar is that the raw file read through the map gives exactly the rows the
rewritten file gives through the GWAS-SSF reader.

`fixtures/dbgap_mvp/raw/` holds one file per deposit shape: quantitative,
binary and META. `fixtures/dbgap_mvp/gwas_ssf/` is each file passed through the
converter itself -- the awk program extracted from opengwasdb-stores `dev` at
56465a2 and run as `awk -F '\\t' -v OFS='\\t' "$CONVERTER_AWK" raw > gwas_ssf`
-- not a re-implementation of it, so the comparison is against the real rewrite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from column_map_fixtures import ssf_streams

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.column_map import ColumnMap, load_column_map
from opengwasdb.readers.column_map_reader import ColumnMapReader
from opengwasdb.readers.gwas_ssf import GwasSsfReader

_FIXTURES = Path(__file__).parent / "fixtures" / "dbgap_mvp"
_SHAPES = {
    "continuous": StoredEffectScale.SD,
    "binary": StoredEffectScale.LOG_OR,
    "meta": StoredEffectScale.SD,
}


def _mvp_map() -> ColumnMap:
    return load_column_map(_FIXTURES / "mvp-gia.toml")


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_raw_files_read_through_the_map_give_the_rewrite_s_rows(shape: str) -> None:
    scale = _SHAPES[shape]
    rewritten = GwasSsfReader(_FIXTURES / "gwas_ssf" / f"{shape}.tsv", scale, chunk_rows=2)
    reference = ssf_streams(rewritten)
    assert reference["associations"] != "[]", "the fixture must yield associations"

    raw = ColumnMapReader(_FIXTURES / "raw" / f"{shape}.tsv", _mvp_map(), scale, chunk_rows=2)

    assert ssf_streams(raw) == reference


def test_the_fixture_reaches_every_convention_the_rewrite_handles() -> None:
    """Parity over rows that never exercise a convention would prove nothing."""
    continuous = list(
        GwasSsfReader(_FIXTURES / "gwas_ssf" / "continuous.tsv").stream_metrics()
    )
    binary = list(
        GwasSsfReader(
            _FIXTURES / "gwas_ssf" / "binary.tsv", StoredEffectScale.LOG_OR
        ).stream_metrics()
    )
    positions = {row.position for row in continuous}
    assert 24_000_000 in positions, "an exactly integral scientific position"
    assert 712_300 in positions, "a scientific position with a fractional mantissa"
    assert 9_000 in positions, "a signed integer position"
    assert any(row.chromosome == "X" for row in continuous), "a non-autosome"
    assert any(row.ref == "AT" for row in continuous), "an indel"
    assert any(row.flipped for row in continuous), "an effect allele that is ref"
    assert any(row.se_from_p_value for row in continuous), "a standard error from p"
    assert any(row.se_from_ci for row in binary), "a standard error from the interval"
    assert any(row.se_from_p_value for row in binary), "an interval with a missing bound"


def _raw(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "raw.tsv"
    path.write_text(text, encoding="utf-8")
    return path


_HEADER = "SNP_ID\tchrom\tpos\tref\talt\tea\taf\tnum_samples\tbeta\tsebeta\tpval\n"


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ("rs1\t1\t100\tA\tG\tC\t0.2\t1000\t0.1\t0.01\t0.5\n", "neither ref"),
        ("rs1\t1\t100\tA\tG\tG\t.\t1000\t0.1\t0.01\t0.5\n", "declared missing"),
        ("rs1\t1\t100\tA\tG\tG\t0.2\t1000\t0.1\t0.01\t0.5\textra\n", "has 12 fields"),
    ],
)
def test_a_row_the_rewrite_would_abort_on_fails_loudly(
    tmp_path: Path, row: str, reason: str
) -> None:
    reader = ColumnMapReader(_raw(tmp_path, _HEADER + row), _mvp_map())

    with pytest.raises(ValueError, match=reason):
        list(reader.stream_associations())


def test_an_interval_that_is_not_a_pair_fails_loudly(tmp_path: Path) -> None:
    text = (
        "SNP_ID\tchrom\tpos\tref\talt\tea\taf\tnum_samples\tor\tci\tpval\n"
        "rs1\t1\t100\tA\tG\tG\t0.2\t1000\t1.1\t1.05\t0.5\n"
    )
    reader = ColumnMapReader(_raw(tmp_path, text), _mvp_map(), StoredEffectScale.LOG_OR)

    with pytest.raises(ValueError, match="not a lower,upper pair"):
        list(reader.stream_metric_chunks())


def test_a_deposit_column_the_map_does_not_know_fails_loudly(tmp_path: Path) -> None:
    row = "rs1\t1\t100\tA\tG\tG\t0.2\t1000\t0.1\t0.01\t0.5\tx\n"
    text = _HEADER.replace("\n", "\tnew_column\n") + row
    reader = ColumnMapReader(_raw(tmp_path, text), _mvp_map())

    with pytest.raises(ValueError, match="new_column"):
        list(reader.stream_variants())
