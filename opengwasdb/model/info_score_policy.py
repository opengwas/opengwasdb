"""Per-Analysis provider-backed INFO filter input policy (stores #175).

A source header is not evidence of a score's meaning. This policy only parses
explicit manifest values; it does not inspect source data or apply the filter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY
from opengwasdb.readers.interface import ImputationScoreDeclaration, ImputationScoreKind

INFO_SCORE_COLUMNS: tuple[str, ...] = (
    "info_score_threshold",
    "imputation_score_column",
    "imputation_score_kind",
    "imputation_score_provenance",
)


@dataclass(frozen=True)
class InfoScorePolicy:
    """A numeric cut-off with a provider declaration, or no filter."""

    info_score_threshold: float | None = None
    imputation_score_declaration: ImputationScoreDeclaration | None = None


def parse_info_score_policy(
    row: Mapping[str, str], *, reader_capability: str | None = None
) -> InfoScorePolicy:
    """Parse one analyses.tsv row; fail on incomplete or unsupported evidence.

    `reader_capability` is the already-resolved per-Analysis capability (row,
    caller default, or existing resolver fallback). The public manifest validator
    uses the explicit row value; it cannot infer a reader from a source header.
    NaN is an explicit unavailable threshold only when no mapping is declared.
    """
    values = {column: (row.get(column) or "").strip() for column in INFO_SCORE_COLUMNS[1:]}
    threshold = _parse_threshold(row.get("info_score_threshold"), any(values.values()))
    if threshold is None:
        return InfoScorePolicy()
    return InfoScorePolicy(threshold, _parse_declaration(row, values, reader_capability))


def _parse_threshold(raw: str | None, has_declaration: bool) -> float | None:
    if raw is None or raw.strip() == "NaN":
        if has_declaration:
            state = "NaN" if raw is not None else "missing"
            raise ValueError(f"imputation score declaration with {state} info_score_threshold")
        return None
    try:
        threshold = float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid info_score_threshold {raw!r}; expected [0, 1] or NaN") from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"invalid info_score_threshold {raw!r}; expected [0, 1] or NaN")
    return threshold


def _parse_declaration(
    row: Mapping[str, str], values: dict[str, str], reader_capability: str | None
) -> ImputationScoreDeclaration:
    missing = [column for column in INFO_SCORE_COLUMNS[1:] if not values[column]]
    if missing:
        raise ValueError(f"numeric info_score_threshold requires {', '.join(missing)}")
    capability = reader_capability or row.get("source_reader_capability")
    if capability != GWAS_SSF_CAPABILITY:
        raise ValueError(
            f"numeric info_score_threshold requires source_reader_capability "
            f"{GWAS_SSF_CAPABILITY!r}; got {capability!r}"
        )
    column_name = values["imputation_score_column"]
    if column_name != row["imputation_score_column"]:
        raise ValueError("imputation_score_column must have exact unpadded spelling")
    provenance = values["imputation_score_provenance"]
    if provenance == column_name:
        raise ValueError("imputation_score_provenance must be independent provider evidence")
    return ImputationScoreDeclaration(
        column_name, _parse_score_kind(values["imputation_score_kind"]), provenance
    )


def _parse_score_kind(value: str) -> ImputationScoreKind:
    try:
        return ImputationScoreKind(value)
    except ValueError as exc:
        raise ValueError(
            f"invalid imputation_score_kind {value!r}; expected imputation_info or imputation_r2"
        ) from exc
