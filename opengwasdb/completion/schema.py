"""The ``completion_quality`` SQLite schema shared by Dense and Ragged Reference
Completion (issue #22) and checked by the validator -- one definition instead
of three independently typed-out copies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

COMPLETION_QUALITY_COLUMNS = frozenset(
    {"analysis_index", "block_id", "pearson_r", "n_imputed", "n_missing"}
)


@dataclass(frozen=True)
class CompletionQualityRollup:
    """One Analysis's completion-rollup columns, named rather than positional
    so Dense/Ragged's differing use of ``n_missing_total`` (see
    ``completion_quality_rollup`` below) can't be mixed up by index."""

    median_pearson_r: str
    n_imputed_total: str
    n_missing_total: str

CREATE_COMPLETION_QUALITY_SQL = """
    CREATE TABLE completion_quality (
        analysis_index INTEGER NOT NULL,
        block_id       TEXT    NOT NULL,
        pearson_r      REAL,
        n_imputed      INTEGER NOT NULL,
        n_missing      INTEGER NOT NULL,
        PRIMARY KEY (analysis_index, block_id)
    )
"""


def create_completion_quality_table(connection: Any) -> None:
    connection.execute(CREATE_COMPLETION_QUALITY_SQL)


def completion_quality_rollup(
    connection: Any, n_analyses: int, *, imputed: np.ndarray, missing: np.ndarray
) -> dict[int, CompletionQualityRollup]:
    """Per-Analysis completion-rollup columns: the LD-block-by-Analysis
    ``completion_quality`` table's median Pearson r (issue #22) -- the large,
    fine-grained table stays SQLite-only per ADR 0030; only this rollup travels
    into ``analyses.tsv`` -- and the counts the release holds.

    A quality row is one block's attempts, so a cell two overlapping blocks attempt is
    in both rows and the rows' sums are not the Analysis's totals. ``imputed`` and
    ``missing`` (per Analysis) are required: the cells the layout counted where it wrote
    them, the totals `analyses.tsv` declares (#362). Dense's missing count is its
    off-panel cells (variants never attempted), Ragged's the rows it holds no value at.
    """
    per_analysis_r: dict[int, list[float]] = {}
    for row in connection.execute(
        "SELECT analysis_index, pearson_r FROM completion_quality WHERE pearson_r IS NOT NULL"
    ):
        per_analysis_r.setdefault(int(row["analysis_index"]), []).append(float(row["pearson_r"]))
    rollup: dict[int, CompletionQualityRollup] = {}
    for idx in range(n_analyses):
        r_values = per_analysis_r.get(idx, [])
        median_r = f"{float(np.median(r_values)):.6g}" if r_values else ""
        rollup[idx] = CompletionQualityRollup(
            median_pearson_r=median_r,
            n_imputed_total=str(int(imputed[idx])),
            n_missing_total=str(int(missing[idx])),
        )
    return rollup
