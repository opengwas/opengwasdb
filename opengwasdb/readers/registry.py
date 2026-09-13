"""Source Reader Capability resolution (issue #19; opengwasdb-stores ADR-0009).

The single, documented point that maps a Source Collection's
`source_reader_capability` string to a concrete `SourceReader`. Existing
builders do not yet call through this resolver -- wiring them is out of
scope here (see `opengwasdb.readers.interface`'s module docstring); this
ticket only introduces the seam.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.finngen import FINNGEN_R13_CAPABILITY, FinnGenR13Reader
from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY, GwasSsfReader
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY, GwasVcfReader
from opengwasdb.readers.interface import SourceReader

_READERS: dict[str, Callable[[str | Path, StoredEffectScale], SourceReader]] = {
    FINNGEN_R13_CAPABILITY: FinnGenR13Reader,
    GWAS_VCF_CAPABILITY: GwasVcfReader,
    GWAS_SSF_CAPABILITY: GwasSsfReader,
}


def known_capabilities() -> tuple[str, ...]:
    """Return all registered source reader capabilities in sorted order."""
    return tuple(sorted(_READERS))


def resolve_reader(
    capability: str, path: str | Path, stored_effect_scale: StoredEffectScale
) -> SourceReader:
    """Return a SourceReader for `path`, given its Source Collection's
    `source_reader_capability` string.

    `stored_effect_scale` is the Analysis's manifest-resolved effect scale
    (issue #17): every reader needs it, since no Source Format's own header
    is authoritative for it.

    Raises ValueError for an unknown capability, naming the ones this
    package knows about.
    """
    try:
        reader_factory = _READERS[capability]
    except KeyError:
        known = ", ".join(known_capabilities()) or "(none registered)"
        raise ValueError(
            f"unknown source reader capability {capability!r}; known: {known}"
        ) from None
    return reader_factory(path, stored_effect_scale)
