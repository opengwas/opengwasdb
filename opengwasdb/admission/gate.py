"""One Analysis's admission gate: judge each batch, keep the counts.

A `Gate` is opened once per Analysis and fed its rows batch by batch. It
returns each batch's keep mask and adds what it judged to the Analysis's
`AdmissionCounts`, so the rows a caller keeps and the counts it reports come
from one judgement and cannot disagree (stores #176).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from opengwasdb.admission.rules import AdmissionCounts, AnalysisRules, SourceRows, judge

__all__ = ["Gate", "RoutedBatch", "counting_gate"]


class RoutedBatch(Protocol):
    """What Variant Routing's ``admit=`` hands a gate: one batch, before matching."""

    @property
    def eaf(self) -> np.ndarray: ...

    @property
    def imputation_score(self) -> np.ndarray: ...

    @property
    def imputation_status(self) -> np.ndarray: ...


class Gate:
    """One Analysis's rows judged batch by batch, and what it dropped counted."""

    def __init__(self, rules: AnalysisRules) -> None:
        self.rules = rules
        self._counts = AdmissionCounts()

    @property
    def counts(self) -> AdmissionCounts:
        """Everything this gate has judged so far."""
        return self._counts

    def source_rows(self, rows: SourceRows) -> np.ndarray:
        """The rows of one batch Admission keeps; the rest are counted by reason."""
        verdict = judge(rows, self.rules)
        self._counts += verdict.counts
        return verdict.keep

    def admit(self, batch: RoutedBatch) -> np.ndarray:
        """`source_rows` for one of Variant Routing's batches: its ``admit=``."""
        return self.source_rows(
            SourceRows(
                eaf=batch.eaf,
                imputation_score=batch.imputation_score,
                imputation_status=batch.imputation_status,
            )
        )


def counting_gate(rules: AnalysisRules) -> Gate:
    """A gate that only counts: the resolver's, and the tests'."""
    return Gate(rules)
