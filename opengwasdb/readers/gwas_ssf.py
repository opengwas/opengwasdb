"""GWAS-SSF SourceReader (issue #84).

Adapts the orientation and beta/se -> z parsing already proven by
`opengwasdb.layouts.ragged.build_ssf._read_filtered` to the `SourceReader`
interface (issue #19), so a filtered/harmonised GWAS-Catalog-SSF file can
route through `opengwasdb.readers.registry.resolve_reader` into the Dense
and Hybrid builders (issue #20), not only the Ragged-only path that module
serves. `stream_associations` and `extract_at_sites` share the full row parser;
`stream_variants` has a projection-aware path that reads only identity and
alias columns while using the same normalization rules (issue #179).
`rsid`/`variant_id` are read into each `SourceVariant` (issue #109) so a Dense
or Hybrid store built through this reader is queryable by rsid, exactly as the
Ragged path already was; `ReaderAssociation` still has no rsid field -- an
rsid names a variant, not an association.

`ref`/`alt` on each `ReaderAssociation`/`stream_variants` tuple are the
source's own `other_allele`/`effect_allele` labelling (mirroring GWAS-VCF's
REF/ALT, where ALT is likewise the effect allele) -- not reordered to
canonical A1/A2, per the interface's contract.

`stream_metrics` (issue #207) is the third projection: identity plus the
effect allele's frequency, `beta` and `standard_error`, read by column index in
one pass. It exists because the pre-build annotation stages both want a
statistic `ReaderAssociation` does not carry or does not keep -- it holds a
`z` rather than the beta behind it, and it drops rows with an unusable beta that
ancestry assignment could still read a frequency from.

`extract_at_sites` has no GWAS-VCF/bcftools equivalent to call into (issue
#21 built that combined AF+SE lookup around bcftools -R specifically): it
scans the file once, reading `effect_allele_frequency` where the file
carries that column and dropping requested sites the file has no AF for
(SiteMetrics never fabricates an AF), oriented to canonical A1 and excluding
palindromic (A/T, C/G) variants like `GwasVcfReader.extract_at_sites` does,
since neither this reader nor its callers have strand information to
resolve them.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.interface import ReaderAssociation, SiteMetrics, SourceVariant
from opengwasdb.readers.tabular import (
    DEFAULT_CHUNK_ROWS,
    MetricsChunk,
    MetricsProjectionColumns,
    TabularMetricsRow,
    TabularRow,
    VariantProjectionColumns,
    extract_at_sites,
    parse_af,
    parse_finite_float,
    parse_positive_float,
    stream_associations,
    stream_projected_metric_chunks,
    stream_projected_metrics,
    stream_projected_variants,
    stream_variants,
)
from opengwasdb.variants.normalise import VariantNormalisationError, orient_to_canonical

GWAS_SSF_CAPABILITY = "opengwasdb.gwas-ssf"
_VARIANT_COLUMNS = VariantProjectionColumns(
    chromosome=(b"chromosome",),
    position=b"base_pair_location",
    ref=b"other_allele",
    alt=b"effect_allele",
    aliases=(b"rsid", b"variant_id"),
)
_METRICS_COLUMNS = MetricsProjectionColumns(
    chromosome=(b"chromosome",),
    position=b"base_pair_location",
    ref=b"other_allele",
    alt=b"effect_allele",
    frequency=b"effect_allele_frequency",
    beta=b"beta",
    standard_error=b"standard_error",
)


def _rsid(rsid: str | None, variant_id: str | None) -> str:
    """The row's rs identifier, or "" if it names none.

    Harmonised GWAS-SSF carries a dedicated `rsid` column; `variant_id` is the
    harmonised (usually non-rs) identifier and is only a fallback, mirroring
    `opengwasdb.layouts.ragged.build_ssf._read_filtered`. Anything that is not
    an rs identifier is dropped: it is not something a user can look the
    variant up by (issue #109).
    """
    for candidate in (rsid, variant_id):
        value = (candidate or "").strip()
        if value.startswith("rs"):
            return value
    return ""


def _iter_rows(path: str | Path) -> Iterator[TabularRow]:
    """Parse each row of a filtered/harmonised GWAS-SSF file once.

    A row with an unparseable chromosome, position, or allele pair cannot be
    represented as a variant at all and is dropped from every stream; a row
    with a valid identity but an unusable `beta`/`standard_error` still
    yields a `TabularRow` (its `beta`/`se` are `None`) so `stream_variants`
    can still see it per the interface's superset contract.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            effect_allele = row.get("effect_allele")
            other_allele = row.get("other_allele")
            if effect_allele is None or other_allele is None:
                continue
            try:
                ori = orient_to_canonical(
                    row.get("chromosome", ""),
                    row.get("base_pair_location", ""),
                    effect_allele,
                    other_allele,
                )
            except VariantNormalisationError:
                continue
            se = parse_positive_float(row.get("standard_error"))
            beta = parse_finite_float(row.get("beta"))
            yield TabularRow(
                chromosome=ori.variant.chromosome,
                position=ori.variant.position,
                alid=ori.variant.alid,
                ref=other_allele,
                alt=effect_allele,
                flipped=ori.flipped,
                beta=beta,
                se=se,
                af_alt=parse_af(row.get("effect_allele_frequency")),
                rsid=_rsid(row.get("rsid"), row.get("variant_id")),
            )


def _iter_variants(path: str | Path) -> Iterator[SourceVariant]:
    yield from stream_projected_variants(path, _VARIANT_COLUMNS)


def stream_full_row_variants(path: str | Path) -> Iterator[SourceVariant]:
    """Reference variant semantics retained for parity benchmarks (issue #179)."""
    yield from stream_variants(_iter_rows(path))


def stream_full_row_metrics(path: str | Path) -> Iterator[TabularMetricsRow]:
    """Reference metrics semantics: the full-row parser's own rows (issue #207).

    Kept so the projection `stream_metrics` replaces can be compared against it
    field for field on the same file -- the arrangement
    `stream_full_row_variants` provides for the variant projection. A
    `TabularRow` *is* a `TabularMetricsRow`; it adds only the identifier this
    seam never reads.
    """
    yield from _iter_rows(path)


@dataclass(frozen=True)
class GwasSsfReader:
    """SourceReader for one filtered/harmonised GWAS-Catalog-SSF file.

    `stored_effect_scale` is Analytical Metadata for this file's Analysis,
    resolved by the caller from the build manifest (issue #16's schema) --
    the file's own columns carry no effect-scale concept, so it is never
    derived from `path` itself.

    `chunk_rows` bounds what `stream_metric_chunks` holds at once, and so what a
    resolver worker costs against a genome-wide source: a block's ALIDs, not the
    file's (issue #209). It is on the reader because that is where the caller
    running sixty-four of these can reach it.
    """

    path: str | Path
    stored_effect_scale: StoredEffectScale = StoredEffectScale.SD
    chunk_rows: int = DEFAULT_CHUNK_ROWS

    def stream_associations(self) -> Iterator[ReaderAssociation]:
        yield from stream_associations(_iter_rows(self.path), self.stored_effect_scale)

    def stream_variants(self) -> Iterator[SourceVariant]:
        yield from _iter_variants(self.path)

    def stream_metrics(self) -> Iterator[TabularMetricsRow]:
        """Yield each row's identity and statistics from one projected scan.

        The seam issue #207's one-pass resolver reads: ancestry assignment wants
        a frequency and phenotype-SD estimation wants a standard error, and this
        yields both -- plus the beta the beta-distribution tier needs -- for
        every row in one pass over the file. `ReaderAssociation` cannot serve
        that: it carries a `z` rather than the beta behind it, and it drops every
        row whose beta is unusable, which is a row ancestry assignment can still
        read a frequency from.
        """
        yield from stream_projected_metrics(self.path, _METRICS_COLUMNS)

    def stream_metric_chunks(self) -> Iterator[MetricsChunk]:
        """The same projection `stream_metrics` yields, a block at a time.

        What the one-pass resolver actually reads (issue #209): the projection
        is identical row for row, and normalising a block's distinct alleles
        once rather than every row's separately roughly halves the time a
        genome-wide source takes.
        """
        yield from stream_projected_metric_chunks(
            self.path, _METRICS_COLUMNS, chunk_rows=self.chunk_rows
        )

    def extract_at_sites(self, alids: Iterable[str]) -> dict[str, SiteMetrics]:
        return extract_at_sites(_iter_rows(self.path), alids)
