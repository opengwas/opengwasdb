"""Association Admission's rule set, through a counting gate (stores #175, #176).

Every builder and the resolver judge rows through one `Gate`, so these tests
exercise the rules at that seam: which rows a batch keeps, and the reason each
dropped row is counted under.
"""

from __future__ import annotations

import numpy as np

from opengwasdb.admission import (
    AdmissionCounts,
    AnalysisRules,
    Reason,
    SourceRows,
    counting_gate,
)
from opengwasdb.model.info_score_policy import InfoScorePolicy, InfoScoreState
from opengwasdb.model.maf_policy import MafPolicy, MafState
from opengwasdb.readers.interface import (
    ImputationScoreDeclaration,
    ImputationScoreKind,
    ImputationScoreStatus,
)

USABLE = ImputationScoreStatus.USABLE
DECLARATION = ImputationScoreDeclaration(
    "info", ImputationScoreKind.IMPUTATION_INFO, "Provider Table 2 INFO"
)
FILTERED_INFO = InfoScorePolicy(0.7, DECLARATION, InfoScoreState.FILTERED)
FILTERED_MAF = MafPolicy(0.005, MafState.FILTERED)


def _rows(af: list[float], scores: list[float] | None = None) -> SourceRows:
    score = np.full(len(af), 0.9) if scores is None else np.asarray(scores, dtype=np.float64)
    return SourceRows(
        eaf=np.asarray(af, dtype=np.float64),
        imputation_score=score,
        imputation_status=np.array([USABLE] * len(af), dtype=object),
    )


def _judged(
    rows: SourceRows, info: InfoScorePolicy, maf: MafPolicy
) -> tuple[list[bool], AdmissionCounts]:
    gate = counting_gate(AnalysisRules("A1", info=info, maf=maf))
    keep = gate.source_rows(rows)
    assert keep.dtype == np.bool_ and keep.shape == (len(rows),)
    return keep.tolist(), gate.counts


def test_monomorphic_and_boundary_frequencies_are_judged_under_filtered():
    """A frequency of exactly 0 or exactly 1 is a frequency, and its MAF is 0.

    Opengwas #331: a monomorphic row under a `filtered` policy carries a MAF of
    0, which is below any positive threshold, so it must be dropped -- and
    counted below-threshold, because the threshold was compared. Treating it as
    an absent frequency let tens of millions of monomorphic rows into a
    common-variant Store. The boundary stays: `0.0049` drops, `0.005` passes.
    """
    keep, counts = _judged(_rows([0.0, 1.0, 0.0049, 0.005, 0.5]), InfoScorePolicy(), FILTERED_MAF)

    assert keep == [False, False, False, True, True]
    assert counts.admitted == 2
    assert counts.dropped == {Reason.MAF_BELOW_THRESHOLD: 3}
    assert counts.maf_below_threshold == 3
    assert counts.maf_missing == 0


def test_unusable_frequencies_are_dropped_as_missing_under_filtered():
    """Under `filtered`, a row with no usable frequency cannot bypass admission.

    A NaN, an infinity and an out-of-range value are all "no frequency
    evidence": a common-variant Store cannot admit a row whose MAF it cannot
    read, or the filter is defeated by blanks. They are dropped and counted
    `maf_missing` -- not below-threshold, because no threshold was compared.
    """
    keep, counts = _judged(
        _rows([np.nan, np.inf, -0.1, 1.5, 0.001, 0.01]), InfoScorePolicy(), FILTERED_MAF
    )

    assert keep == [False, False, False, False, False, True]
    assert counts.dropped == {Reason.MAF_MISSING: 4, Reason.MAF_BELOW_THRESHOLD: 1}
    assert counts.maf_below_threshold == 1
    assert counts.maf_missing == 4


def test_no_filter_keeps_every_frequency():
    """`disabled` and `unavailable` drop no row.

    Only a declared positive threshold admits the "no frequency evidence"
    reading; `0` asked for no filtering, though it still counts the rows with no
    usable frequency, and an undeclared column records no MAF dispositions.
    """
    rows = _rows([0.0, 1.0, 0.001, np.nan])

    keep, disabled = _judged(rows, InfoScorePolicy(), MafPolicy(0.0, MafState.DISABLED))
    assert keep == [True, True, True, True]
    assert disabled.dropped == {}
    assert disabled.maf_missing == 1

    keep, undeclared = _judged(rows, InfoScorePolicy(), MafPolicy())
    assert keep == [True, True, True, True]
    assert undeclared.dropped == {}
    assert undeclared.maf_missing == 0


def test_info_comes_before_maf_and_each_row_has_one_reason():
    """A row both filters would drop is counted under INFO, and once.

    `maf_missing` still counts every row with no usable frequency, INFO-dropped
    or not, because that is what a Store's provenance reports.
    """
    rows = _rows(
        [0.001, np.nan, 0.3, 0.001, np.nan],
        scores=[0.5, 0.5, 0.5, 0.9, 0.9],
    )
    keep, counts = _judged(rows, FILTERED_INFO, FILTERED_MAF)

    assert keep == [False, False, False, False, False]
    assert counts.dropped == {
        Reason.INFO_BELOW_THRESHOLD: 3,
        Reason.MAF_MISSING: 1,
        Reason.MAF_BELOW_THRESHOLD: 1,
    }
    assert counts.examined == 5
    assert counts.admitted == 0
    assert counts.info.below_threshold == 3
    assert counts.maf_missing == 2


def test_a_gate_adds_up_its_batches():
    """A gate fed in batches reports what one batch of every row would."""
    af = [0.3, 0.001, np.nan, 0.2, 0.004, 0.5]
    scores = [0.9, 0.9, 0.5, 0.5, 0.9, 0.9]
    whole = counting_gate(AnalysisRules("A1", info=FILTERED_INFO, maf=FILTERED_MAF))
    whole.source_rows(_rows(af, scores))
    split = counting_gate(AnalysisRules("A1", info=FILTERED_INFO, maf=FILTERED_MAF))
    split.source_rows(_rows(af[:2], scores[:2]))
    split.source_rows(_rows(af[2:], scores[2:]))

    assert split.counts == whole.counts
    assert AdmissionCounts.from_dict(whole.counts.as_dict()) == whole.counts
