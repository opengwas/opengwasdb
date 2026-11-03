"""Rules that read an association sequence: a Ragged release's, or a Hybrid's Overflow (ADR 0074).

`offsets`, `variant_index`, `z` and `se` travel together in every Ragged component
-- a Ragged release's sequence and a Hybrid's Ragged Overflow alike -- so their
absence, and their lengths, are one rule each, reported the same way for both.
What the arrays hold is judged by folds over the association scan, which reads
them only once these rules have passed.
"""

from __future__ import annotations

from collections.abc import Iterator

from opengwasdb.model.analyses import read_analyses
from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.store.arrays import array_length
from opengwasdb.validation.rule import Check, Kind, Rule, Target
from opengwasdb.validation.rules.release import REQUIRED
from opengwasdb.validation.scans import judges_frequencies, sequence_offsets

#: The rule every rule that reads a sequence's parallel arrays runs after.
ARRAYS = "csr.arrays"
#: The rule every rule that reads a sequence cell by cell runs after.
LENGTHS = "csr.lengths"
#: The rule an Overflow's values are judged after: it holds no imputed cell.
OBSERVED_ONLY = "csr.observed_only"


def is_overflow(target: Target) -> bool:
    """Whether the sequence is a Hybrid release's Ragged Overflow."""
    return target.kind is Kind.CSR and target.layout is PrimaryStorageLayout.HYBRID


def is_ragged(target: Target) -> bool:
    """Whether the sequence is a Ragged release's own."""
    return target.layout is PrimaryStorageLayout.RAGGED


def prefix(target: Target) -> str:
    """What a value message about an Overflow starts with, to say which component it is."""
    return "Ragged Overflow " if is_overflow(target) else ""


def arrays_present(target: Target) -> Iterator[str]:
    """The four parallel arrays every association sequence carries are present."""
    group = target.group
    for name in ("offsets", "variant_index", "z", "se"):
        if name not in group:
            yield f"missing data.zarr/ragged/{name}"


def lengths(target: Target) -> Iterator[str]:
    """Each parallel array holds one entry per association `offsets` bounds.

    `eaf` is optional (ADR 0036); when present it is a fourth parallel array, and
    one entry out of step hands every later cell another association's frequency.
    """
    group = target.group
    n_assoc = int(sequence_offsets(target)[-1])
    names = ["variant_index", "z", "se"]
    if "eaf" in group and judges_frequencies(target):
        names.append("eaf")
    for name in names:
        if array_length(group[name]) != n_assoc:
            yield (
                f"data.zarr/ragged/{name} has {array_length(group[name])} entries "
                f"but offsets imply {n_assoc}"
            )


def analyses_match(target: Target) -> Iterator[str]:
    """`analyses.tsv` describes the Analyses the sequence's offsets bound.

    `analyses.tsv` is the sole source of truth for Analytical Metadata (ADR 0030,
    ADR 0034), so its row count must equal the number of segments.
    """
    n_tsv = len(read_analyses(target.store.analyses_path).rows)
    n_csr = len(sequence_offsets(target)) - 1
    if n_tsv != n_csr:
        yield f"analyses.tsv has {n_tsv} rows but zarr CSR offsets imply {n_csr} analyses"


def observed_only(target: Target) -> Iterator[str]:
    """A Ragged Overflow is off-panel, so nothing in it can have been imputed."""
    if "imputed" in target.group:
        yield (
            "Ragged Overflow has an imputed array — the overflow is off-panel and "
            "must never be imputed (always observed)"
        )


RULES: tuple[Rule, ...] = (
    Check(name=ARRAYS, kinds=Kind.CSR, after=(REQUIRED,), run=arrays_present),
    Check(name=LENGTHS, kinds=Kind.CSR, after=(ARRAYS,), run=lengths),
    Check(
        name="csr.analyses",
        kinds=Kind.CSR,
        when=is_ragged,
        after=(ARRAYS, "analyses.readable"),
        run=analyses_match,
    ),
    Check(name=OBSERVED_ONLY, kinds=Kind.CSR, when=is_overflow, after=(ARRAYS,), run=observed_only),
)
