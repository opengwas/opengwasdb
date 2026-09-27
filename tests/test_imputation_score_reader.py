"""Provider-declared imputation quality in GWAS-SSF's three public streams (#175)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from opengwasdb.readers.gwas_ssf import GwasSsfReader
from opengwasdb.readers.interface import (
    ImputationScoreDeclaration,
    ImputationScoreKind,
    ImputationScoreStatus,
)
from opengwasdb.readers.tabular import metrics_chunks_from_rows

_HEADER = (
    "chromosome\tbase_pair_location\tother_allele\teffect_allele\tbeta\t"
    "standard_error\teffect_allele_frequency"
)


def _file(path: Path, headers: str, scores: list[str]) -> Path:
    path.write_text(
        _HEADER
        + "\t"
        + headers
        + "\n"
        + "".join(
            f"1\t{100 + i}\tA\tG\t0.2\t0.1\t0.1\t{score}\n" for i, score in enumerate(scores)
        ),
        encoding="utf-8",
    )
    return path


def _declared(
    name: str = "info", kind: ImputationScoreKind = ImputationScoreKind.IMPUTATION_INFO
) -> ImputationScoreDeclaration:
    return ImputationScoreDeclaration(
        name, kind, "provider documentation: imputation INFO for this Analysis"
    )


def test_declared_score_same_on_associations_rows_and_bounded_chunks(tmp_path: Path) -> None:
    values = ["0", "0.6", "1", "", "NA", "bogus", "inf", "-0.1", "1.01"]
    path = _file(tmp_path / "quality.tsv", "info", values)
    reader = GwasSsfReader(path, chunk_rows=2, imputation_score_declaration=_declared())
    expected = [
        ImputationScoreStatus.USABLE,
        ImputationScoreStatus.USABLE,
        ImputationScoreStatus.USABLE,
        ImputationScoreStatus.MISSING,
        ImputationScoreStatus.MISSING,
        ImputationScoreStatus.MALFORMED,
        ImputationScoreStatus.NONFINITE,
        ImputationScoreStatus.OUT_OF_RANGE,
        ImputationScoreStatus.OUT_OF_RANGE,
    ]
    associations = list(reader.stream_associations())
    rows = list(reader.stream_metrics())
    chunks = list(reader.stream_metric_chunks())
    assert len(associations) == len(rows) == sum(map(len, chunks)) == len(values)
    assert [row.imputation_score.status for row in rows] == expected
    assert [a.imputation_score for a in associations] == [row.imputation_score for row in rows]
    assert [s for chunk in chunks for s in chunk.imputation_score_status] == expected
    assert [row.imputation_score.value for row in rows[:3]] == [0.0, 0.6, 1.0]
    assert [float(v) for chunk in chunks for v in chunk.imputation_score][:3] == [0.0, 0.6, 1.0]
    assert all(row.imputation_score.value is None for row in rows[3:])
    assert all(
        math.isnan(float(v)) for v in [v for chunk in chunks for v in chunk.imputation_score][3:]
    )
    fallback_chunks = list(metrics_chunks_from_rows(rows, chunk_rows=2))
    assert [s for chunk in fallback_chunks for s in chunk.imputation_score_status] == expected


def test_score_remains_on_row_with_unusable_effect(tmp_path: Path) -> None:
    path = _file(tmp_path / "bad_effect.tsv", "info", ["0.8"])
    path.write_text(path.read_text().replace("\t0.2\t0.1\t", "\tNA\t0.1\t"))
    reader = GwasSsfReader(path, imputation_score_declaration=_declared())
    assert list(reader.stream_associations()) == []
    row = next(reader.stream_metrics())
    assert row.imputation_score.value == 0.8
    assert next(reader.stream_metric_chunks()).imputation_score.tolist() == [0.8]


def test_no_declaration_never_uses_lookalikes(tmp_path: Path) -> None:
    path = _file(tmp_path / "unmapped.tsv", "info\tINFO\tR2\tinfo_score", ["0.9\t0.8\t0.7\t0.6"])
    reader = GwasSsfReader(path)
    assert reader.imputation_score_column is None
    assert (
        next(reader.stream_associations()).imputation_score.status
        is ImputationScoreStatus.UNDECLARED
    )
    assert next(reader.stream_metrics()).imputation_score.status is ImputationScoreStatus.UNDECLARED
    chunk = next(reader.stream_metric_chunks())
    assert chunk.imputation_score_status.tolist() == [ImputationScoreStatus.UNDECLARED]
    assert math.isnan(float(chunk.imputation_score[0]))


@pytest.mark.parametrize(
    "header", ["INFO", "info\tinfo", "info \tinfo", "info\tinfo ", "info_score", "R2"]
)
def test_declared_header_must_be_exact_and_unique(tmp_path: Path, header: str) -> None:
    path = _file(tmp_path / "ambiguous.tsv", header, ["0.9"])
    reader = GwasSsfReader(path, imputation_score_declaration=_declared())
    with pytest.raises(ValueError, match="imputation score column"):
        list(reader.stream_associations())
    with pytest.raises(ValueError, match="imputation score column"):
        list(reader.stream_metrics())
    with pytest.raises(ValueError, match="imputation score column"):
        list(reader.stream_metric_chunks())


def test_explicit_r2_is_not_inferred_from_name(tmp_path: Path) -> None:
    path = _file(tmp_path / "r2.tsv", "R2\tINFO", ["0.65\t0.95"])
    reader = GwasSsfReader(
        path, imputation_score_declaration=_declared("R2", ImputationScoreKind.IMPUTATION_R2)
    )
    assert reader.imputation_score_column == "R2"
    assert next(reader.stream_associations()).imputation_score.value == 0.65
    assert next(reader.stream_metrics()).imputation_score.value == 0.65
    assert next(reader.stream_metric_chunks()).imputation_score.tolist() == [0.65]


@pytest.mark.parametrize(
    "name,kind,provenance",
    [
        ("", ImputationScoreKind.IMPUTATION_INFO, "provider"),
        ("info ", ImputationScoreKind.IMPUTATION_INFO, "provider"),
        ("info", "other", "provider"),
        ("info", ImputationScoreKind.IMPUTATION_INFO, "  "),
    ],
)
def test_declaration_requires_exact_name_kind_and_provenance(
    name: str, kind: ImputationScoreKind, provenance: str
) -> None:
    with pytest.raises(ValueError):
        ImputationScoreDeclaration(name, kind, provenance)
