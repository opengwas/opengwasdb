"""Per-Analysis provider-backed INFO filter input policy (stores #175).

A source header is not evidence of a score's meaning. This policy only parses
explicit manifest values; it does not inspect source data or apply the filter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY
from opengwasdb.readers.interface import ImputationScoreDeclaration, ImputationScoreKind

INFO_SCORE_COLUMNS: tuple[str, ...] = (
    "info_score_threshold",
    "imputation_score_column",
    "imputation_score_kind",
    "imputation_score_provenance",
)


class InfoScoreState(StrEnum):
    """Why an Analysis does or does not apply its declared INFO threshold.

    `NO_USABLE_SCORES` is the one member a *policy* never carries: it is the
    outcome a declared score with no usable value is recorded as (stores #176).
    `InfoScorePolicy` validates against the policy states, so it can never be
    constructed with this one.
    """

    LEGACY_ABSENT = "legacy_absent"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"
    FILTERED = "filtered"
    NO_USABLE_SCORES = "no_usable_scores"


#: The states a policy with no threshold and no declaration may carry: only
#: whether the manifest carried the threshold column at all separates them.
_NO_THRESHOLD_STATES = (InfoScoreState.LEGACY_ABSENT, InfoScoreState.UNAVAILABLE)


def _numeric_state(threshold: float) -> InfoScoreState:
    """The state a validated finite threshold in [0, 1] means (stores #175)."""
    return InfoScoreState.DISABLED if threshold == 0 else InfoScoreState.FILTERED


def _expected_states(
    threshold: float | None, declaration: ImputationScoreDeclaration | None
) -> tuple[InfoScoreState, ...]:
    """The states one threshold/declaration pair allows (stores #175).

    A numeric threshold requires its declaration and fixes its own state; no
    threshold requires the absence of one and leaves legacy and explicit
    unavailability to the manifest.
    """
    if threshold is None:
        if declaration is not None:
            raise ValueError(
                "an imputation score declaration requires a numeric info_score_threshold"
            )
        return _NO_THRESHOLD_STATES
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"info_score_threshold must be finite in [0, 1], got {threshold!r}")
    if declaration is None:
        raise ValueError(
            f"info_score_threshold {threshold!r} requires an imputation score declaration"
        )
    return (_numeric_state(threshold),)


@dataclass(frozen=True)
class InfoScorePolicy:
    """A numeric cut-off with a provider declaration, or no filter.

    `state` separates four situations callers must not collapse into one
    absence: no threshold was declared (`LEGACY_ABSENT`), the manifest says the
    score is unavailable (`UNAVAILABLE`), a zero threshold with a usable score
    asked for no filtering (`DISABLED`), or a positive threshold filters rows
    (`FILTERED`). It is derived by `parse_info_score_policy` -- never from a
    source header -- and asserted here so a policy that contradicts its own
    threshold and declaration cannot be constructed and silently applied.
    """

    info_score_threshold: float | None = None
    imputation_score_declaration: ImputationScoreDeclaration | None = None
    state: InfoScoreState = InfoScoreState.LEGACY_ABSENT

    def __post_init__(self) -> None:
        allowed = _expected_states(self.info_score_threshold, self.imputation_score_declaration)
        if self.state not in allowed:
            names = " or ".join(state.value for state in allowed)
            raise ValueError(
                f"info_score_state {self.state.value!r} contradicts info_score_threshold "
                f"{self.info_score_threshold!r}; expected {names}"
            )


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
        state = (
            InfoScoreState.LEGACY_ABSENT
            if row.get("info_score_threshold") is None
            else InfoScoreState.UNAVAILABLE
        )
        return InfoScorePolicy(state=state)
    declaration = _parse_declaration(row, values, reader_capability)
    return InfoScorePolicy(threshold, declaration, _numeric_state(threshold))


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
