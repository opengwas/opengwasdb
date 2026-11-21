"""Kind dispatch for public validation, from the one root read `ogld.open` also uses (#91)."""

from __future__ import annotations

import os
from pathlib import Path

from ogld.genotypes import validation as genotypes
from ogld.genotypes.layout import KEY
from ogld.query.kind import GENOTYPE, open_kind
from ogld.store import Finding
from ogld.store.validation import validate_group


def validate(location: str | os.PathLike[str]) -> dict[str, list[Finding]]:
    """Run the named rules of the store kind recorded at the root (spec §12)."""
    return _validate(location, threads=1)


def validate_at(location: str | os.PathLike[str], *, threads: int = 1) -> dict[str, list[Finding]]:
    """`validate` on `threads` threads, which is what `ogld validate --threads` asks for."""
    return _validate(location, threads=threads)


def _validate(location: str | os.PathLike[str], *, threads: int) -> dict[str, list[Finding]]:
    """The root is opened once, and its kind decides which named rules run."""
    kinded = open_kind(location)
    root = kinded.opened.root
    if kinded.kind == GENOTYPE:
        record = root.attrs.get(KEY)
        if not isinstance(record, dict):
            return {
                "genotypes.provenance": [
                    Finding("genotypes.provenance", str(location), "not a Genotype Store")
                ]
            }
        return genotypes.validate_sidecar(Path(location), root, record, threads=threads)
    return validate_group(root, kinded.opened.provenance, threads=threads)
