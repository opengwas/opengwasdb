"""The Association Admission rule set: which of an Analysis's rows a Store admits.

One rule set, judged row by row and in a fixed precedence, so each dropped
row carries exactly one `Reason`:

1. the declared INFO filter (`info_below_threshold`): a usable score below a
   `FILTERED` threshold; equality passes;
2. the declared MAF filter, under a `FILTERED` policy: a row with no usable
   frequency (`maf_missing`), then a MAF below the threshold
   (`maf_below_threshold`); equality passes.

MAF is `min(af, 1 - af)` for a finite `af` inside `[0, 1]`. A row whose `af` is
missing, non-finite or outside `[0, 1]` carries no frequency evidence, and a
common-variant Store cannot admit a row whose MAF it cannot read
(opengwasdb#331). Exactly `0.0` and `1.0` *are* frequencies here, a monomorphic
site of MAF 0, so a `FILTERED` policy drops them for the filter's reason. (The
reader's `parse_af` still stores no EAF for them, ADR 0036; whether a Store
holds a frequency is a different question from whether a filter admits a row.)

INFO comes first: a row both filters would drop is counted under INFO. The
rows with no usable frequency are also counted on their own, whatever INFO
did, under any policy that reads the column (`DISABLED` and `FILTERED`), which
is the `maf_missing` a Store's provenance reports.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from opengwasdb.build.info_score_filter import (
    InfoScoreCounts,
    count_info_scores,
    retained_mask,
)
from opengwasdb.model.info_score_policy import InfoScorePolicy
from opengwasdb.model.maf_policy import MafPolicy, MafState

__all__ = ["AdmissionCounts", "AnalysisRules", "Reason", "SourceRows", "judge"]


class Reason(StrEnum):
    """Why Association Admission dropped a row: one per dropped row."""

    INFO_BELOW_THRESHOLD = "info_below_threshold"
    MAF_MISSING = "maf_missing"
    MAF_BELOW_THRESHOLD = "maf_below_threshold"


#: A row's reason code: 0 is admitted, ``i + 1`` is ``_REASONS[i]``.
_REASONS: tuple[Reason, ...] = tuple(Reason)
_CODE = {reason: np.uint8(i + 1) for i, reason in enumerate(_REASONS)}


@dataclass(frozen=True)
class AnalysisRules:
    """What one Analysis declared that Admission judges its rows by."""

    analysis_id: str
    info: InfoScorePolicy = field(default_factory=InfoScorePolicy)
    maf: MafPolicy = field(default_factory=MafPolicy)


@dataclass(frozen=True)
class SourceRows:
    """One batch of an Analysis's rows, before routing, in the reader's orientation.

    Every array is positionally parallel. ``eaf`` and ``imputation_score`` are
    NaN where the source reports none; ``imputation_status`` holds each row's
    `ImputationScoreStatus`.
    """

    eaf: np.ndarray
    imputation_score: np.ndarray
    imputation_status: np.ndarray

    def __post_init__(self) -> None:
        lengths = {len(values) for values in vars(self).values()}
        if len(lengths) != 1:
            raise ValueError(
                f"source rows hold {sorted(lengths)} row(s) across their columns; every "
                "column must hold the batch's rows"
            )

    def __len__(self) -> int:
        return len(self.eaf)


@dataclass(frozen=True)
class AdmissionCounts:
    """One Analysis's admission, over the rows it examined.

    ``dropped`` holds each reason's rows, one reason per row. ``info`` is the
    declared-score tally the Stores' provenance reports. ``maf_missing``
    counts every row with no usable frequency, whatever INFO did, unless the
    Analysis declared no MAF filter; it is not ``dropped[MAF_MISSING]``, which
    counts only the rows dropped for it.
    """

    examined: int = 0
    admitted: int = 0
    dropped: Mapping[Reason, int] = field(default_factory=dict)
    info: InfoScoreCounts = field(default_factory=InfoScoreCounts)
    maf_missing: int = 0

    def __add__(self, other: AdmissionCounts) -> AdmissionCounts:
        dropped = {
            reason: self.dropped.get(reason, 0) + other.dropped.get(reason, 0)
            for reason in _REASONS
            if reason in self.dropped or reason in other.dropped
        }
        return AdmissionCounts(
            examined=self.examined + other.examined,
            admitted=self.admitted + other.admitted,
            dropped=dropped,
            info=self.info + other.info,
            maf_missing=self.maf_missing + other.maf_missing,
        )

    @property
    def maf_below_threshold(self) -> int:
        """The rows the MAF threshold dropped that INFO did not."""
        return self.dropped.get(Reason.MAF_BELOW_THRESHOLD, 0)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready form, reasons in their declared order."""
        return {
            "examined": self.examined,
            "admitted": self.admitted,
            "dropped": {reason.value: n for reason, n in self._ordered_drops()},
            "info": asdict(self.info),
            "maf_missing": self.maf_missing,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AdmissionCounts:
        """Rebuild what `as_dict` wrote."""
        return cls(
            examined=int(data["examined"]),
            admitted=int(data["admitted"]),
            dropped={Reason(name): int(n) for name, n in data["dropped"].items()},
            info=InfoScoreCounts(**data["info"]),
            maf_missing=int(data["maf_missing"]),
        )

    def _ordered_drops(self) -> list[tuple[Reason, int]]:
        return [(reason, self.dropped[reason]) for reason in _REASONS if reason in self.dropped]


@dataclass(frozen=True)
class Verdict:
    """One batch judged: each row's reason code (0 admitted) and the batch's counts."""

    codes: np.ndarray
    counts: AdmissionCounts

    @property
    def keep(self) -> np.ndarray:
        keep: np.ndarray = self.codes == 0
        return keep


def judge(rows: SourceRows, rules: AnalysisRules) -> Verdict:
    """Each row's reason, by the rule set's precedence, and what the batch counts."""
    codes = np.zeros(len(rows), dtype=np.uint8)
    info_keep = retained_mask(rows.imputation_score, rows.imputation_status, rules.info)
    codes[~info_keep] = _CODE[Reason.INFO_BELOW_THRESHOLD]
    usable_frequency = np.isfinite(rows.eaf) & (rows.eaf >= 0.0) & (rows.eaf <= 1.0)
    if rules.maf.state is MafState.FILTERED:
        _judge_maf(codes, rows.eaf, usable_frequency, rules.maf)
    # A policy that declared no MAF filter records no MAF dispositions, exactly as
    # an undeclared INFO policy records none; `DISABLED` still reads the column.
    maf_missing = 0 if rules.maf.state is MafState.UNAVAILABLE else int(
        np.count_nonzero(~usable_frequency)
    )
    return Verdict(
        codes=codes,
        counts=AdmissionCounts(
            examined=len(rows),
            admitted=int(np.count_nonzero(codes == 0)),
            dropped=_tally(codes),
            info=count_info_scores(rows.imputation_score, rows.imputation_status, rules.info),
            maf_missing=maf_missing,
        ),
    )


def _judge_maf(
    codes: np.ndarray, eaf: np.ndarray, usable_frequency: np.ndarray, policy: MafPolicy
) -> None:
    """Mark the rows a `FILTERED` MAF policy drops that no earlier rule did."""
    threshold = policy.maf_threshold
    assert threshold is not None, "a FILTERED MAF policy carries its threshold"
    open_rows = codes == 0
    # Unusable frequencies are dropped, not retained: retaining them let
    # monomorphic and blank-frequency rows bypass the filter entirely, which is
    # how rare variation flooded a common-variant Store.
    codes[open_rows & ~usable_frequency] = _CODE[Reason.MAF_MISSING]
    with np.errstate(invalid="ignore"):
        below = usable_frequency & (np.minimum(eaf, 1.0 - eaf) < threshold)
    codes[open_rows & below] = _CODE[Reason.MAF_BELOW_THRESHOLD]


def _tally(codes: np.ndarray) -> dict[Reason, int]:
    """Each reason's rows in one batch, reasons with none left out."""
    counts = np.bincount(codes, minlength=len(_REASONS) + 1)
    return {reason: int(counts[i + 1]) for i, reason in enumerate(_REASONS) if counts[i + 1]}
