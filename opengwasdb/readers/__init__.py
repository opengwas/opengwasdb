"""Source Reader Capability seam (issue #19): resolves a Source Collection's
`source_reader_capability` string to a concrete `SourceReader`.
"""

from __future__ import annotations

from opengwasdb.readers.effect_source import (
    EffectSource,
    EffectSourceKind,
    resolve_effect_source,
)
from opengwasdb.readers.fake import FakeReader
from opengwasdb.readers.finngen import FINNGEN_R13_CAPABILITY, FinnGenR13Reader
from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY, GwasSsfReader
from opengwasdb.readers.gwas_vcf import (
    GWAS_VCF_CAPABILITY,
    GwasVcfReader,
    is_palindromic,
    load_liftover,
    write_regions_file,
)
from opengwasdb.readers.interface import (
    ReaderAssociation,
    SiteMetrics,
    SourceReader,
    af_only,
    site_metrics_arrays,
)
from opengwasdb.readers.registry import known_capabilities, resolve_reader

__all__ = [
    "FINNGEN_R13_CAPABILITY",
    "GWAS_SSF_CAPABILITY",
    "GWAS_VCF_CAPABILITY",
    "EffectSource",
    "EffectSourceKind",
    "FakeReader",
    "FinnGenR13Reader",
    "GwasSsfReader",
    "GwasVcfReader",
    "ReaderAssociation",
    "SiteMetrics",
    "SourceReader",
    "af_only",
    "is_palindromic",
    "known_capabilities",
    "load_liftover",
    "resolve_effect_source",
    "resolve_reader",
    "site_metrics_arrays",
    "write_regions_file",
]
