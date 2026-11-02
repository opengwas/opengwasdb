"""Association Admission: which of an Analysis's source rows a Store admits.

One rule set for every builder and the resolver. A caller opens a `Gate` per
Analysis, from `counting_gate`, and hands it each batch of rows before they are
matched to the Variant Index; the gate returns the rows to keep and counts the
rest by `Reason`. `opengwasdb.admission.rules` states the rules and their
precedence.
"""

from __future__ import annotations

from opengwasdb.admission.gate import Gate, RoutedBatch, counting_gate
from opengwasdb.admission.rules import AdmissionCounts, AnalysisRules, Reason, SourceRows

__all__ = [
    "AdmissionCounts",
    "AnalysisRules",
    "Gate",
    "Reason",
    "RoutedBatch",
    "SourceRows",
    "counting_gate",
]
