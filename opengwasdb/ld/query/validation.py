"""Kind dispatch for public validation, beside the query reader dispatch (#91)."""

from __future__ import annotations

import os

from ogld.genotypes import validation as genotypes
from ogld.genotypes.layout import KEY
from ogld.store import Finding
from ogld.store.location import open_group
from ogld.store.validation import validate_at as validate_ld


def validate_at(location: str | os.PathLike[str], *, threads: int = 1) -> dict[str, list[Finding]]:
    if KEY in open_group(location).attrs:
        return genotypes.validate(location, threads=threads)
    return validate_ld(location, threads=threads)


def validate(location: str | os.PathLike[str]) -> dict[str, list[Finding]]:
    """Run the named rules of the store kind recorded at the root."""
    return validate_at(location)
