"""Rules that read an association sequence: a Ragged release's, or a Hybrid's Overflow (ADR 0074).

`offsets`, `variant_index`, `z` and `se` travel together in every Ragged component
-- a Ragged release's sequence and a Hybrid's Ragged Overflow alike -- so their
absence is one rule, reported the same way for both.
"""

from __future__ import annotations

from collections.abc import Iterator

from opengwasdb.validation.rule import Check, Kind, Rule, Target
from opengwasdb.validation.rules.release import REQUIRED

#: The rule every rule that reads a sequence's parallel arrays runs after.
ARRAYS = "csr.arrays"


def arrays_present(target: Target) -> Iterator[str]:
    """The four parallel arrays every association sequence carries are present."""
    group = target.group
    for name in ("offsets", "variant_index", "z", "se"):
        if name not in group:
            yield f"missing data.zarr/ragged/{name}"


RULES: tuple[Rule, ...] = (
    Check(name=ARRAYS, kinds=Kind.CSR, after=(REQUIRED,), run=arrays_present),
)
