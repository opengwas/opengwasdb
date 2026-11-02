"""One Analysis's admission gate: judge each batch, keep the counts and the log.

A `Gate` is opened once per Analysis and fed its rows batch by batch. It
returns each batch's keep mask, adds what it judged to the Analysis's
`AdmissionCounts`, and keeps the source key and reason of every row it drops
for a logged reason, so the rows a caller keeps and the counts and log it
reports come from one judgement and cannot disagree (stores #176).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from opengwasdb.admission.rules import AdmissionCounts, AnalysisRules, Reason, SourceRows, judge

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
    """One Analysis's rows judged batch by batch, and what it dropped counted and logged."""

    def __init__(self, rules: AnalysisRules) -> None:
        self.rules = rules
        self._counts = AdmissionCounts()
        self._dropped: list[tuple[str, Reason]] = []

    @property
    def counts(self) -> AdmissionCounts:
        """Everything this gate has judged so far."""
        return self._counts

    @property
    def dropped(self) -> list[tuple[str, Reason]]:
        """The logged drops so far, ``(source key, reason)``, in the order judged."""
        return list(self._dropped)

    def source_rows(self, rows: SourceRows) -> np.ndarray:
        """The rows of one batch Admission keeps; the rest are counted by reason."""
        verdict = judge(rows, self.rules)
        positions, reasons = verdict.logged()
        if len(positions):
            if rows.keys_at is None:
                raise ValueError(
                    f"Analysis {self.rules.analysis_id}: {len(positions)} row(s) were dropped "
                    "for a logged reason from a batch that names no source keys"
                )
            self._dropped.extend(zip(rows.keys_at(positions), reasons, strict=True))
        self._counts += verdict.counts
        return verdict.keep

    def admit(self, batch: RoutedBatch) -> np.ndarray:
        """`source_rows` for one of Variant Routing's batches: its ``admit=``.

        An Analysis that declared no score was read without one, so its rows'
        scores are not handed on.
        """
        declared = self.rules.info.imputation_score_declaration is not None
        return self.source_rows(
            SourceRows(
                eaf=batch.eaf,
                imputation_score=batch.imputation_score if declared else None,
                imputation_status=batch.imputation_status if declared else None,
            )
        )


def counting_gate(rules: AnalysisRules) -> Gate:
    """A gate that writes no record and keeps only its own counts: the resolver's."""
    return Gate(rules)
