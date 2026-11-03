"""Association Admission's rule set, through a counting gate (stores #175, #176).

Every builder and the resolver judge rows through one `Gate`, so these tests
exercise the rules at that seam: which rows a batch keeps, and the reason each
dropped row is counted under.
"""

from __future__ import annotations

import numpy as np
import pytest

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


def _with_se(af: list[float], se: list[float], scores: list[float]) -> SourceRows:
    rows = _rows(af, scores)
    keys = [f"1:{100 + i}:A:G" for i in range(len(af))]
    return SourceRows(
        eaf=rows.eaf,
        imputation_score=rows.imputation_score,
        imputation_status=rows.imputation_status,
        se=np.asarray(se, dtype=np.float64),
        keys_at=lambda positions: [keys[i] for i in positions.tolist()],
    )


def test_an_unusable_se_is_dropped_before_any_declared_filter_judges_the_row():
    """Missing, zero, negative and infinite SEs are one reason, and it comes
    first: those rows are not associations, so the declared filters neither
    drop nor count them, and INFO's tally is over the associations alone."""
    rows = _with_se(
        [0.3, 0.3, 0.001, 0.3, 0.3, 0.3],
        [np.nan, 0.0, -0.1, np.inf, 0.2, 0.2],
        scores=[0.5, 0.9, 0.9, 0.9, 0.5, 0.9],
    )
    gate = counting_gate(AnalysisRules("A1", info=FILTERED_INFO, maf=FILTERED_MAF))

    keep = gate.source_rows(rows)

    assert keep.tolist() == [False, False, False, False, False, True]
    assert gate.counts.dropped == {Reason.SE_UNUSABLE: 4, Reason.INFO_BELOW_THRESHOLD: 1}
    assert gate.counts.info.observed == 2
    assert gate.counts.examined == 6
    assert gate.dropped == [(f"1:{100 + i}:A:G", Reason.SE_UNUSABLE) for i in range(4)]


def test_a_declared_filter_drop_is_counted_but_not_logged():
    gate = counting_gate(AnalysisRules("A1", info=FILTERED_INFO, maf=FILTERED_MAF))
    gate.source_rows(_with_se([0.001, 0.3], [0.2, 0.2], scores=[0.9, 0.5]))

    assert gate.counts.dropped == {
        Reason.INFO_BELOW_THRESHOLD: 1,
        Reason.MAF_BELOW_THRESHOLD: 1,
    }
    assert gate.dropped == []


def test_a_batch_that_drops_a_logged_row_must_name_its_keys():
    rows = SourceRows(eaf=np.array([0.3]), se=np.array([0.0]))

    with pytest.raises(ValueError, match="names no source keys"):
        counting_gate(AnalysisRules("A1")).source_rows(rows)


def test_a_batch_whose_columns_disagree_in_length_is_refused():
    with pytest.raises(ValueError, match=r"\[2, 3\] row"):
        SourceRows(eaf=np.zeros(3), se=np.zeros(2))


def test_a_row_with_no_effect_is_dropped_after_the_se_rule_and_logged():
    """A NaN z beside a usable SE is a row the source gave no effect: it is not
    an association, so it is dropped and logged before any declared filter. A
    row with neither is counted under the SE rule, which comes first."""
    keys = ["1:100:A:G", "1:101:A:G", "1:102:A:G"]
    rows = SourceRows(
        eaf=np.array([0.3, 0.3, 0.3]),
        se=np.array([0.2, np.nan, 0.2]),
        z=np.array([np.nan, np.nan, 1.5]),
        keys_at=lambda positions: [keys[i] for i in positions.tolist()],
    )
    gate = counting_gate(AnalysisRules("A1", maf=FILTERED_MAF))

    assert gate.source_rows(rows).tolist() == [False, False, True]
    assert gate.counts.dropped == {Reason.SE_UNUSABLE: 1, Reason.EFFECT_MISSING: 1}
    assert gate.dropped == [
        ("1:100:A:G", Reason.EFFECT_MISSING),
        ("1:101:A:G", Reason.SE_UNUSABLE),
    ]


def test_a_completion_candidate_is_admissible_only_with_finite_z_and_positive_se():
    from opengwasdb.admission.cells import admissible

    z = np.array([1.0, np.nan, np.inf, 2.0, 2.0, 2.0])
    se = np.array([0.1, 0.1, 0.1, 0.0, -0.1, np.nan])

    assert admissible(z, se).tolist() == [True, False, False, False, False, False]


def test_require_admitted_accepts_what_admission_stores():
    from opengwasdb.admission import require_admitted

    require_admitted(
        np.array([1.0, np.nan, -2.0]),
        np.array([0.1, np.nan, 0.2]),
        eaf=np.array([0.3, np.nan, 0.9]),
        imputed=np.array([0, 0, 1], dtype=np.uint8),
        where="test",
    )


@pytest.mark.parametrize(
    ("z", "se", "eaf", "imputed", "rule"),
    [
        ([1.0], [np.nan], None, None, "one of z and SE missing"),
        ([np.nan], [0.1], None, None, "one of z and SE missing"),
        ([np.inf], [0.1], None, None, "infinite z"),
        ([1.0], [0.0], None, None, "SE not finite and positive"),
        ([1.0], [-0.1], None, None, "SE not finite and positive"),
        ([1.0], [0.1], [1.0], None, "frequency outside"),
        ([1.0], [0.1], [0.0], None, "frequency outside"),
        ([np.nan], [np.nan], None, [1], "imputed flag on a missing cell"),
        ([1.0], [0.1], None, [2], "imputed flag neither 0 nor 1"),
    ],
)
def test_require_admitted_refuses_each_broken_rule(z, se, eaf, imputed, rule):
    from opengwasdb.admission import NotAdmitted, require_admitted

    with pytest.raises(NotAdmitted, match=rf"codec: .*{rule}"):
        require_admitted(
            np.array(z),
            np.array(se),
            eaf=None if eaf is None else np.array(eaf),
            imputed=None if imputed is None else np.array(imputed, dtype=np.uint8),
            where="codec",
        )
