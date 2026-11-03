"""The Association Admission rule set: which of an Analysis's rows a Store admits.

One rule set, judged row by row and in a fixed precedence, so each dropped
row carries exactly one `Reason`. First the rules that decide whether a row is
an association a Store can hold at all, then the filters the Analysis declared:

0. a malformed identity, which no ALID can spell: a missing chromosome
   (`chromosome_missing`: an empty cell, `.`, `NA`), then an allele outside
   A, C, G and T (`allele_invalid`). The Source Reader finds these as it forms
   the row's identity (`variants.normalise.MalformedIdentity`), drops the row
   and reports it, the same on every reader;
1. an unusable reported SE (`se_unusable`): missing, not positive, or not
   finite;
2. no effect (`effect_missing`): a usable SE beside a z the source gave no
   effect to compute;
3. a non-finite z (`z_nonfinite`): an infinite effect, or z, beside a usable
   SE;
4. the declared INFO filter (`info_below_threshold`): a usable score below a
   `FILTERED` threshold; equality passes;
5. the declared MAF filter, under a `FILTERED` policy: a row with no usable
   frequency (`maf_missing`), then a MAF below the threshold
   (`maf_below_threshold`); equality passes.

The declared filters judge, and count, only the rows the first rules admit:
those are the associations a Source Reader yielded, which is the population a
Store's `info_score` and `maf` provenance has always described.

MAF is `min(af, 1 - af)` for a finite `af` inside `[0, 1]`. A row whose `af` is
missing, non-finite or outside `[0, 1]` carries no frequency evidence, and a
common-variant Store cannot admit a row whose MAF it cannot read
(opengwasdb#331). Exactly `0.0` and `1.0` *are* frequencies here, a monomorphic
site of MAF 0, so a `FILTERED` policy drops them for the filter's reason. (The
reader's `parse_af` still stores no EAF for them, ADR 0036; whether a Store
holds a frequency is a different question from whether a filter admits a row.)

INFO comes before MAF: a row both filters would drop is counted under INFO. The
rows with no usable frequency are also counted on their own, whatever INFO
did, under any policy that reads the column (`DISABLED` and `FILTERED`), which
is the `maf_missing` a Store's provenance reports.

After the rules, a row is judged against the other rows that reach its place
on the Variant Index. Rows that agree on z and SE are one association and
collapse to the first (`duplicate_collapsed`); rows that disagree are all
dropped (`duplicate_conflicting`), since nothing says which is right. The one
exception is a reversed indel: an insertion and a deletion at one position
share an ALID, because the ALID sorts its alleles, so when the source declares
which allele is the reference the row whose reference allele the ALID spells
first is kept and the other dropped (`duplicate_reversed_indel`). A row whose
variant no axis of the Store holds is dropped too (`off_axis`), where a build
has nowhere else to put it.

A row dropped by a declared filter, or collapsed into its first, is counted
only; a row dropped for any other reason is also logged, by its source key,
because nothing but the source can say which rows those were.

A frequency is not a reason to drop a row. One the source reported outside
(0, 1) -- exactly 0 or 1, beyond the interval, or not finite -- is absent in the
Store, and counted (`eaf_made_absent`) over the rows the Source Reader read.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
from opengwasdb.variants.normalise import InvalidAllele, MissingChromosome

__all__ = [
    "LOGGED",
    "AdmissionCounts",
    "AnalysisRules",
    "Reason",
    "SourceRows",
    "Verdict",
    "judge",
]


class Reason(StrEnum):
    """Why Association Admission dropped a row: one per dropped row, in precedence order."""

    #: A row whose chromosome the source does not give (#365).
    CHROMOSOME_MISSING = MissingChromosome.reason
    #: A row with an allele outside the ALID alphabet (#365).
    ALLELE_INVALID = InvalidAllele.reason
    SE_UNUSABLE = "se_unusable"
    EFFECT_MISSING = "effect_missing"
    Z_NONFINITE = "z_nonfinite"
    INFO_BELOW_THRESHOLD = "info_below_threshold"
    MAF_MISSING = "maf_missing"
    MAF_BELOW_THRESHOLD = "maf_below_threshold"
    #: A row that agreed with an earlier row for its (variant, Analysis), whose
    #: association it is: not lost, so counted only.
    DUPLICATE_COLLAPSED = "duplicate_collapsed"
    #: An indel row whose declared reference allele its ALID spells second, at
    #: a (variant, Analysis) that keeps the row spelling it first.
    DUPLICATE_REVERSED_INDEL = "duplicate_reversed_indel"
    #: A row for a (variant, Analysis) whose rows disagree on z or SE.
    DUPLICATE_CONFLICTING = "duplicate_conflicting"
    #: A row whose variant no axis of the Store holds, so routing places it nowhere.
    OFF_AXIS = "off_axis"


#: The reasons whose rows are logged by source key, not only counted. A declared
#: filter's drops are reproducible from the source and the policy, and logging
#: them would cost more than the Store they filter.
LOGGED: frozenset[Reason] = frozenset(
    {
        Reason.CHROMOSOME_MISSING,
        Reason.ALLELE_INVALID,
        Reason.SE_UNUSABLE,
        Reason.EFFECT_MISSING,
        Reason.Z_NONFINITE,
        Reason.DUPLICATE_REVERSED_INDEL,
        Reason.DUPLICATE_CONFLICTING,
        Reason.OFF_AXIS,
    }
)

#: A row's reason code: 0 is admitted, ``i + 1`` is ``_REASONS[i]``.
_REASONS: tuple[Reason, ...] = tuple(Reason)
_CODE = {reason: np.uint8(i + 1) for i, reason in enumerate(_REASONS)}
_LOGGED_CODES = np.array(sorted(int(_CODE[reason]) for reason in LOGGED), dtype=np.uint8)


@dataclass(frozen=True)
class AnalysisRules:
    """What one Analysis declared that Admission judges its rows by."""

    analysis_id: str
    info: InfoScorePolicy = field(default_factory=InfoScorePolicy)
    maf: MafPolicy = field(default_factory=MafPolicy)


#: The source keys of the rows at some positions of one batch, for the log.
KeysAt = Callable[[np.ndarray], Sequence[str]]


@dataclass(frozen=True)
class SourceRows:
    """One batch of an Analysis's rows, before routing, in the reader's orientation.

    Every array is positionally parallel. ``eaf`` is NaN where the source
    reports none. ``imputation_score`` and ``imputation_status`` are what the
    declared score column parsed to, or ``None`` where no score was declared,
    which is every row `UNDECLARED`. ``se`` is the reported SE, before any
    standardisation, and ``z`` the row's z, NaN where the source reports no
    effect; either is ``None`` from a caller whose reader has already judged
    it. ``keys_at`` names rows for the log; a batch that can drop a logged row
    must carry it.
    """

    eaf: np.ndarray
    imputation_score: np.ndarray | None = None
    imputation_status: np.ndarray | None = None
    se: np.ndarray | None = None
    z: np.ndarray | None = None
    keys_at: KeysAt | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        # One row judged against another row's statistics would keep the wrong
        # rows in silence, so a ragged batch is refused (stores #175).
        columns = (self.eaf, self.imputation_score, self.imputation_status, self.se, self.z)
        lengths = {len(values) for values in columns if values is not None}
        if len(lengths) != 1:
            raise ValueError(
                f"source rows hold {sorted(lengths)} row(s) across their columns; every "
                "column must hold the batch's rows"
            )
        if (self.imputation_score is None) != (self.imputation_status is None):
            raise ValueError("a declared score needs both its values and their statuses")

    def __len__(self) -> int:
        return len(self.eaf)


@dataclass(frozen=True)
class AdmissionCounts:
    """One Analysis's admission, over the rows it examined.

    ``dropped`` holds each reason's rows, one reason per row. ``info`` is the
    declared-score tally the Stores' provenance reports, over the rows the
    rules before the declared filters admitted. ``maf_missing`` counts every
    one of those rows with no usable frequency, whatever INFO did, unless the
    Analysis declared no MAF filter; it is not ``dropped[MAF_MISSING]``, which
    counts only the rows dropped for it. ``eaf_made_absent`` counts the reported
    frequencies outside (0, 1) the Store holds as absent, over every row read
    (D7); a row is never dropped for one.
    """

    examined: int = 0
    admitted: int = 0
    dropped: Mapping[Reason, int] = field(default_factory=dict)
    info: InfoScoreCounts = field(default_factory=InfoScoreCounts)
    maf_missing: int = 0
    eaf_made_absent: int = 0

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
            eaf_made_absent=self.eaf_made_absent + other.eaf_made_absent,
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
            "eaf_made_absent": self.eaf_made_absent,
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
            eaf_made_absent=int(data["eaf_made_absent"]),
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

    def logged(self) -> tuple[np.ndarray, list[Reason]]:
        """The positions of the rows dropped for a logged reason, and each one's reason."""
        positions = np.flatnonzero(np.isin(self.codes, _LOGGED_CODES))
        return positions, [_REASONS[int(code) - 1] for code in self.codes[positions]]


def judge(rows: SourceRows, rules: AnalysisRules) -> Verdict:
    """Each row's reason, by the rule set's precedence, and what the batch counts."""
    codes = _value_codes(rows)
    associations = np.flatnonzero(codes == 0)
    info = _judge_info(codes, rows, associations, rules.info)
    eaf = rows.eaf[associations]
    usable_frequency = np.isfinite(eaf) & (eaf >= 0.0) & (eaf <= 1.0)
    if rules.maf.state is MafState.FILTERED:
        _judge_maf(codes, associations, eaf, usable_frequency, rules.maf)
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
            info=info,
            maf_missing=maf_missing,
        ),
    )


def _value_codes(rows: SourceRows) -> np.ndarray:
    """Each row's code from the rules that decide whether it is an association."""
    codes = np.zeros(len(rows), dtype=np.uint8)
    if rows.se is not None:
        with np.errstate(invalid="ignore"):
            codes[~(rows.se > 0) | ~np.isfinite(rows.se)] = _CODE[Reason.SE_UNUSABLE]
    if rows.z is not None:
        codes[(codes == 0) & np.isnan(rows.z)] = _CODE[Reason.EFFECT_MISSING]
        codes[(codes == 0) & np.isinf(rows.z)] = _CODE[Reason.Z_NONFINITE]
    return codes


def _judge_info(
    codes: np.ndarray, rows: SourceRows, associations: np.ndarray, policy: InfoScorePolicy
) -> InfoScoreCounts:
    """Mark the associations a declared INFO policy drops, and count their scores.

    With no score declared every row is `UNDECLARED`: none is dropped, and the
    tally is the associations, all retained.
    """
    if rows.imputation_score is None or rows.imputation_status is None:
        n = len(associations)
        return InfoScoreCounts(observed=n, retained=n)
    scores = rows.imputation_score[associations]
    statuses = rows.imputation_status[associations]
    codes[associations[~retained_mask(scores, statuses, policy)]] = _CODE[
        Reason.INFO_BELOW_THRESHOLD
    ]
    return count_info_scores(scores, statuses, policy)


def _judge_maf(
    codes: np.ndarray,
    associations: np.ndarray,
    eaf: np.ndarray,
    usable_frequency: np.ndarray,
    policy: MafPolicy,
) -> None:
    """Mark the associations a `FILTERED` MAF policy drops that INFO did not."""
    threshold = policy.maf_threshold
    assert threshold is not None, "a FILTERED MAF policy carries its threshold"
    open_rows = codes[associations] == 0
    # Unusable frequencies are dropped, not retained: retaining them let
    # monomorphic and blank-frequency rows bypass the filter entirely, which is
    # how rare variation flooded a common-variant Store.
    codes[associations[open_rows & ~usable_frequency]] = _CODE[Reason.MAF_MISSING]
    with np.errstate(invalid="ignore"):
        below = usable_frequency & (np.minimum(eaf, 1.0 - eaf) < threshold)
    codes[associations[open_rows & below]] = _CODE[Reason.MAF_BELOW_THRESHOLD]


def _tally(codes: np.ndarray) -> dict[Reason, int]:
    """Each reason's rows in one batch, reasons with none left out."""
    counts = np.bincount(codes, minlength=len(_REASONS) + 1)
    return {reason: int(counts[i + 1]) for i, reason in enumerate(_REASONS) if counts[i + 1]}
