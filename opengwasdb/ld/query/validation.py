"""Kind dispatch for public validation, from the one root read `opengwasdb.ld.open` also uses
(#91)."""

from __future__ import annotations

import os
from pathlib import Path

from opengwasdb.ld.genotypes import validation as genotypes
from opengwasdb.ld.query.kind import GENOTYPE, open_kind
from opengwasdb.ld.store import Finding
from opengwasdb.ld.store.validation import validate_group


def validate(location: str | os.PathLike[str]) -> dict[str, list[Finding]]:
    """Run the named rules of the store kind recorded at the root (ld-store §12)."""
    return _validate(location, threads=1)


def validate_at(location: str | os.PathLike[str], *, threads: int = 1) -> dict[str, list[Finding]]:
    """`validate` on `threads` threads, which is what `opengwasdb ld validate --threads` asks
    for."""
    return _validate(location, threads=threads)


def _validate(location: str | os.PathLike[str], *, threads: int) -> dict[str, list[Finding]]:
    """The root is opened once, and its kind decides which named rules run.

    Validation reports an envelope it cannot read as a finding of its own rule, rather than
    raising: a Zarr v2 hierarchy, or a root with no provenance, is what `envelope.arrays` and
    `provenance.complete` are for (ld-store §12).
    """
    kinded = open_kind(location)
    root = kinded.root
    if kinded.kind == GENOTYPE:
        record = kinded.record()
        if not isinstance(record, dict):
            return {
                "genotypes.provenance": [
                    Finding("genotypes.provenance", str(location), "not a Genotype Store")
                ]
            }
        return genotypes.validate_sidecar(Path(location), root, record, threads=threads)
    return validate_group(root, kinded.record(), threads=threads)
