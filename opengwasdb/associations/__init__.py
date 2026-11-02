"""Association reads: one decode, one record and one presence rule for every layout.

Three adapters answer the same two questions about a region of one component
(`Cells`): `read` -- the region's associations as an `Associations` record,
refusing a cell spec §15 forbids -- and `inspect` -- every cell with its
`State`, for the checks that count what a read refuses. `DenseGridReader` reads
a Dense grid, `CsrReader` an association sequence and `SubsetReader` an Indexed
Variant Subset; each hides its planes' decode order (imputed mask, then EAF with
the panel substitution, then SE) and its routing. Presence is `states()`, in
`record`, and nowhere else (ADR 0070).
"""

from opengwasdb.associations.cells import Cells
from opengwasdb.associations.csr import CsrReader
from opengwasdb.associations.dense import DenseGridReader
from opengwasdb.associations.record import (
    STATUS_LABELS,
    Associations,
    Inspection,
    InvalidAssociations,
    State,
    Status,
    Tally,
    classify,
    non_canonical,
    present,
    states,
)
from opengwasdb.associations.subset import SubsetReader

__all__ = [
    "STATUS_LABELS",
    "Associations",
    "Cells",
    "CsrReader",
    "DenseGridReader",
    "Inspection",
    "InvalidAssociations",
    "State",
    "Status",
    "SubsetReader",
    "Tally",
    "classify",
    "non_canonical",
    "present",
    "states",
]
