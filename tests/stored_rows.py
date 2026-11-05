"""Reading back what an association sequence stores, in a test.

A test that checks what a writer stored reads it through the Association
Reader's `inspect`: every row of an Analysis, decoded, invalid ones included and
none refused. It reads whatever imputed mask the sequence carries and owes none.
"""

from __future__ import annotations

from pathlib import Path

from opengwasdb.associations import Cells, CsrReader, Inspection
from opengwasdb.encoding import StoreEncoding


def stored_rows(
    store: str | Path, analysis: int, encoding: StoreEncoding | None = None
) -> Inspection:
    """Every stored row of one Analysis, variants ascending."""
    reader = CsrReader.open(store, encoding, requires_mask=False)
    return reader.inspect(Cells.analysis(analysis))
