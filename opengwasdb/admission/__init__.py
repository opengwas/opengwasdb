"""Association Admission: which of an Analysis's source rows a Store admits.

One rule set for every builder and the resolver. A caller opens a `Gate` per
Analysis and hands it each batch of rows before they are matched to the
Variant Index; the gate returns the rows to keep and counts the rest by
`Reason`. A build's gates come from its `AdmissionRecord`, which writes what
each Analysis examined, admitted and dropped beside the Store Release; the
resolver's come from `counting_gate`, which only counts.
`opengwasdb.admission.rules` states the rules and their precedence.
"""

from __future__ import annotations

from opengwasdb.admission.gate import Gate, RoutedBatch, counting_gate
from opengwasdb.admission.record import (
    ADMISSION_RULES,
    AdmissionRecord,
    AdmissionSummary,
    RecordIncomplete,
    admission_record_for,
)
from opengwasdb.admission.rules import AdmissionCounts, AnalysisRules, Reason, SourceRows

__all__ = [
    "ADMISSION_RULES",
    "AdmissionCounts",
    "AdmissionRecord",
    "AdmissionSummary",
    "AnalysisRules",
    "Gate",
    "Reason",
    "RecordIncomplete",
    "RoutedBatch",
    "SourceRows",
    "admission_record_for",
    "counting_gate",
]
