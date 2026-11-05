"""The source's own p-value travels through every metrics projection (#345).

The resolver compares the z-score a build will store against the p-value its
source reported, so the p-value has to reach it exactly as the row-wise reader
reads it -- blanks, `NA`, 0, 1 and values below 1e-300 included. A projection
that disagreed would make the check measure the reader, not the file.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.readers.gwas_ssf import _METRICS_COLUMNS, stream_full_row_metrics
from opengwasdb.readers.tabular import (
    metrics_chunks_from_rows,
    stream_projected_metric_chunks,
    stream_projected_metrics,
)

#: Every spelling of a p-value the check must treat identically on each path.
_P_CELLS = ["0.05", "", "NA", "0", "1", "1e-310", "1e-400", "2.5e-300", "5e-324", "1.5", "nan", "abc"]


def _file(tmp_path: Path) -> Path:
    path = tmp_path / "ssf.tsv"
    header = "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\tstandard_error\tp_value"
    lines = [header]
    for index, cell in enumerate(_P_CELLS):
        lines.append(f"1\t{100 + index}\tA\tC\t0.1\t0.02\t{cell}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _nan_to_none(values: np.ndarray) -> list[float | None]:
    return [None if math.isnan(v) else float(v) for v in values]


def test_fixture_exercises_every_awkward_p_value(tmp_path: Path):
    rows = list(stream_projected_metrics(_file(tmp_path), _METRICS_COLUMNS))
    assert len(rows) == len(_P_CELLS)
    present = [r.p_value for r in rows if r.p_value is not None]
    assert 0.0 in present and 1.0 in present and 0.05 in present
    assert any(0.0 < p < 1e-300 for p in present)
    assert sum(r.p_value is None for r in rows) >= 3


@pytest.mark.parametrize("chunk_rows", [1, 4, 1_000])
def test_row_wise_and_blocked_projections_give_identical_p_values(tmp_path: Path, chunk_rows: int):
    path = _file(tmp_path)
    rows = list(stream_projected_metrics(path, _METRICS_COLUMNS))
    row_wise = [None if r.p_value is None else r.p_value for r in rows]

    blocked = np.concatenate(
        [c.p_value for c in stream_projected_metric_chunks(path, _METRICS_COLUMNS, chunk_rows=chunk_rows)]
    )
    via_rows = np.concatenate(
        [c.p_value for c in metrics_chunks_from_rows(iter(rows), chunk_rows=chunk_rows)]
    )
    assert _nan_to_none(blocked) == row_wise
    assert _nan_to_none(via_rows) == row_wise


def test_the_full_row_parser_carries_the_projections_p_values(tmp_path: Path):
    """The GWAS-SSF row parser reads `p_value` on every row, not only on a row
    whose standard error it must recover: every fixture row has a usable SE, so
    a parser that read the cell only for recovery would report none."""
    path = _file(tmp_path)
    projected = [r.p_value for r in stream_projected_metrics(path, _METRICS_COLUMNS)]
    full_rows = list(stream_full_row_metrics(path))
    assert all(r.se is not None for r in full_rows)

    assert [r.p_value for r in full_rows] == projected
    via_rows = np.concatenate([c.p_value for c in metrics_chunks_from_rows(iter(full_rows))])
    assert _nan_to_none(via_rows) == projected
