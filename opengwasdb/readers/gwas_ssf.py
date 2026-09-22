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

The effect is read from whichever permitted column the file carries -- `beta`,
`odds_ratio` (as `log(odds_ratio)`), or a signed `z_score` -- resolved by
`opengwasdb.readers.effect_source` and reported through `effect_source` rather
than assumed (issues #213-#215). A header naming a candidate column twice
raises, because `GCST006329` carries `beta ` and `beta` and a last-wins lookup
would silently read one of two columns. A z-score source derives
`beta = z * se` with `se = 1 / sqrt(2 f (1 - f) (N + z^2))` from the row's own
EAF and per-row N, and carries `assumes_standardised` because that formula
assumes `var(Y) = 1`.

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
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.effect_source import (
    EffectSource,
    EffectSourceKind,
    derive_z_score_effect,
    refuse_case_control_z_score,
    resolve_effect_source,
    resolve_sample_size_column,
)
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
    require_signed_z_score,
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


def _row_effect(
    row: dict[str, str], effect_source: EffectSource | None, sample_size_column: str | None
) -> tuple[float | None, float | None]:
    """One row's ``(beta, se)``, from whichever effect column the file resolved to.

    `beta = log(odds_ratio)`; a signed z derives both by the #215 formula. A
    non-positive or unparseable `odds_ratio`/`z` is unusable in exactly the way
    an unparseable `beta` is, so it yields `(None, None)` and the row drops from
    the association stream. `standard_error` is read verbatim for a beta or
    odds-ratio source -- GWAS-SSF reports it on the log scale already (#213).
    """
    if effect_source is None:
        return None, parse_positive_float(row.get("standard_error"))
    if effect_source.kind is EffectSourceKind.Z_SCORE:
        derived = derive_z_score_effect(
            parse_finite_float(row.get(effect_source.column_name)),
            parse_af(row.get("effect_allele_frequency")),
            parse_positive_float(row.get(sample_size_column)) if sample_size_column else None,
        )
        return derived if derived is not None else (None, None)
    if effect_source.kind is EffectSourceKind.ODDS_RATIO:
        raw = parse_positive_float(row.get(effect_source.column_name))
        beta = math.log(raw) if raw is not None else None
    else:
        beta = parse_finite_float(row.get(effect_source.column_name))
    return beta, parse_positive_float(row.get("standard_error"))


def _tabular_row(
    row: dict[str, str], source: EffectSource | None, sample_size_column: str | None
) -> TabularRow | None:
    """One source row as a `TabularRow`, or `None` when it names no canonical variant.

    A row whose chromosome, position or allele pair cannot be represented is
    dropped from every stream; a row with a valid identity but an unusable
    effect/`standard_error` still yields a row (its `beta`/`se` are `None`) so
    `stream_variants` can still see it per the interface's superset contract.
    """
    effect_allele = row.get("effect_allele")
    other_allele = row.get("other_allele")
    if effect_allele is None or other_allele is None:
        return None
    try:
        ori = orient_to_canonical(
            row.get("chromosome", ""),
            row.get("base_pair_location", ""),
            effect_allele,
            other_allele,
        )
    except VariantNormalisationError:
        return None
    beta, se = _row_effect(row, source, sample_size_column)
    return TabularRow(
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


def _sample_size_column(
    fieldnames: Sequence[str] | None, source: EffectSource | None
) -> str | None:
    """The per-row sample-size column, resolved only for a z-score source.

    Only a z-score derivation needs a sample size, so only it pays for the
    lookup and its duplicate/ambiguity rule (issue #215).
    """
    if source is not None and source.kind is EffectSourceKind.Z_SCORE:
        return resolve_sample_size_column(fieldnames or ())
    return None


def _iter_rows(
    path: str | Path, *, effect_source: EffectSource | None = None
) -> Iterator[TabularRow]:
    """Parse each row of a filtered/harmonised GWAS-SSF file once.

    `effect_source` is the caller's resolution for this file, so the property
    and the stream cannot disagree; when omitted it is resolved here, which is
    what keeps `stream_full_row_metrics` a faithful reference for the
    projection's effect handling.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        source = effect_source or resolve_effect_source(reader.fieldnames or ())
        sample_size_column = _sample_size_column(reader.fieldnames, source)
        for row in reader:
            parsed = _tabular_row(row, source, sample_size_column)
            if parsed is not None:
                yield parsed


def _iter_variants(path: str | Path) -> Iterator[SourceVariant]:
    yield from stream_projected_variants(path, _VARIANT_COLUMNS)


def _header(path: str | Path) -> list[str]:
    """A GWAS-SSF file's header names, reading no data row.

    Shares `_iter_rows`' opener rule so the effect source `GwasSsfReader`
    reports is the one its row parser would resolve for the same file.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh, delimiter="\t"), [])


def require_z_score_usable(
    path: str | Path,
    effect_source: EffectSource | None,
    stored_effect_scale: StoredEffectScale,
) -> None:
    """Refuse a z-score source that cannot be honestly derived from (issue #215).

    This is the policy half of the guard, and it lives with the reader because
    only the reader knows the Analysis's `stored_effect_scale`. A signed z
    derives a phenotype-SD-standardised beta, so a case-control scale is refused
    outright; and the column must be signed, which is a fact about its values
    and not its header (`require_signed_z_score` reads them).
    """
    if effect_source is None or effect_source.kind is not EffectSourceKind.Z_SCORE:
        return
    refuse_case_control_z_score(effect_source, stored_effect_scale)
    require_signed_z_score(path, effect_source.column_name)


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

    @property
    def effect_source(self) -> EffectSource | None:
        """Which effect column this file's Analysis resolves to (issue #213).

        Reported to the caller rather than assumed: a file may carry `beta`,
        `odds_ratio`, or neither, and which one it uses is a fact the caller
        needs (for a manifest record, a diagnostic, or to know a beta is
        derived). A header naming a candidate effect column twice raises
        `ValueError` here -- `GCST006329` carries `beta` twice, and a
        last-wins lookup would silently read one of two columns.
        """
        return resolve_effect_source(_header(self.path))

    def stream_associations(self) -> Iterator[ReaderAssociation]:
        source = self.effect_source
        # A z-score source is derived, not read: refuse it on a case-control
        # scale and refuse an unsigned column before yielding any row (#215).
        require_z_score_usable(self.path, source, self.stored_effect_scale)
        yield from stream_associations(
            _iter_rows(self.path, effect_source=source), self.stored_effect_scale
        )

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
        require_z_score_usable(self.path, self.effect_source, self.stored_effect_scale)
        yield from stream_projected_metrics(self.path, _METRICS_COLUMNS)

    def stream_metric_chunks(self) -> Iterator[MetricsChunk]:
        """The same projection `stream_metrics` yields, a block at a time.

        What the one-pass resolver actually reads (issue #209): the projection
        is identical row for row, and normalising a block's distinct alleles
        once rather than every row's separately roughly halves the time a
        genome-wide source takes.
        """
        require_z_score_usable(self.path, self.effect_source, self.stored_effect_scale)
        yield from stream_projected_metric_chunks(
            self.path, _METRICS_COLUMNS, chunk_rows=self.chunk_rows
        )

    def extract_at_sites(self, alids: Iterable[str]) -> dict[str, SiteMetrics]:
        return extract_at_sites(_iter_rows(self.path), alids)
