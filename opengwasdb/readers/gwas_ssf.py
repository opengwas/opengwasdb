"""GWAS-SSF SourceReader (issues #84, #56).

Reads a filtered/harmonised GWAS-Catalog-SSF file for every builder that
stores one -- Dense, Hybrid and, since issue #56, Ragged SSF -- through
`opengwasdb.readers.registry`. `stream_associations`, `stream_batches` and
`extract_at_sites` share the full row parser;
`stream_variants` has a projection-aware path that reads only identity and
alias columns while using the same normalization rules (issue #179).
`rsid`/`variant_id` are read into each `SourceVariant` and each batch row
(issue #109), so every store built through this reader is queryable by rsid;
`ReaderAssociation` has no rsid field -- an rsid names a variant, not an
association.

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

Imputation quality is not inferred from the header: only an explicit per-Analysis
`ImputationScoreDeclaration` backed by independent provider provenance enables
`imputation_score` on associations, projected rows and bounded chunks (#175).
An undeclared file always reports UNDECLARED, regardless of `info`, `INFO`,
`info_score`, or `R2` columns; the caller must carry the same declaration into
release metadata and filtering.

The effect is read from whichever permitted column the file carries -- `beta`,
`odds_ratio` (as `log(odds_ratio)`), or a signed `z_score` -- resolved by
`opengwasdb.readers.effect_source` and reported through `effect_source` rather
than assumed (issues #213-#215). A header naming a candidate column twice
raises, because `GCST006329` carries `beta ` and `beta` and a last-wins lookup
would silently read one of two columns. A z-score source derives
`beta = z * se` with `se = 1 / sqrt(2 f (1 - f) (N + z^2))` from the row's own
EAF and per-row N, and carries `assumes_standardised` because that formula
assumes `var(Y) = 1`.

Which column a *row* was read from is a second, per-row question (stores #176):
a file may carry `beta` and `odds_ratio` and populate only one column, and it
may report its precision as a 95% interval or a p-value instead of a
`standard_error`. `effect_source.row_statistics` is that rule -- the first usable
column wins, the effect's own scale decides the scale an interval is read on,
and nothing is approximated -- and every GWAS-SSF reading of it (the full-row
parser, the row-wise projection and the blocked projection) goes through it, so
the rows a build retains and the rows its resolver record counts cannot
disagree.

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
import io
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.effect_source import (
    EffectSource,
    EffectSourceKind,
    RowCells,
    RowStatistics,
    refuse_case_control_z_score,
    resolve_effect_source,
    resolve_sample_size_column,
    row_statistics,
)
from opengwasdb.readers.interface import (
    ImputationScore,
    ImputationScoreDeclaration,
    ReaderAssociation,
    ReferenceAllele,
    RegionsFile,
    SiteMetrics,
    SourceVariant,
    StreamTally,
    parse_imputation_score,
)
from opengwasdb.readers.rows import ReaderBatch, stream_batches
from opengwasdb.readers.tabular import (
    DEFAULT_CHUNK_ROWS,
    DeclaredReference,
    MetricsChunk,
    MetricsProjectionColumns,
    SourcePath,
    SourceRow,
    TabularMetricsRow,
    TabularSource,
    VariantProjectionColumns,
    declared_reference,
    extract_at_sites,
    parse_af,
    parse_effect,
    parse_finite_float,
    require_signed_z_score,
    resolve_imputation_score_column,
    stream_associations,
    stream_projected_metric_chunks,
    stream_projected_metrics,
    stream_projected_variants,
    stream_variants,
)
from opengwasdb.stats import FrequencyTally
from opengwasdb.variants.normalise import (
    HmCodeIndel,
    HmColumnsIndel,
    IdentityFaults,
    Orientation,
    UnharmonisedIndel,
    orient_or_drop,
)
from opengwasdb.variants.rsid import normalise_rsid

GWAS_SSF_CAPABILITY = "opengwasdb.gwas-ssf"
#: GWAS-SSF names an effect and an other allele, and the format declares
#: neither the genome's reference, harmonised or not (ADR 0072).
_REFERENCE = ReferenceAllele.UNDECLARED
#: Which rows of a GWAS-SSF file declare their `other_allele` the reference, by
#: the file's layout, read from its header (ADR 0079; owner, 7 Nov 2026):
#:
#: - harmonised alleles in the main columns, `hm_code` beside them: a row whose
#:   harmoniser code is 10 or 11. At FinnGen R13 indel sites read one way,
#:   `other_allele` is FinnGen's `ref` on 99.38% of 37,894,632 code-10 rows and
#:   98.32% of 19,267,774 code-11 rows (197 files); the residual error, a few
#:   files whose minority code disagrees, is accepted;
#: - the study's alleles in the main columns, harmonised ones in `hm_*` columns:
#:   none, so every indel is dropped. Code-11 deletions there agree with
#:   FinnGen on 87.42% of rows over 35 files;
#: - no harmoniser column: none.
MAIN_COLUMNS_REFERENCE = DeclaredReference(
    column=b"hm_code", codes=frozenset({b"10", b"11"}), fault=HmCodeIndel
)
HM_COLUMNS_REFERENCE = DeclaredReference(fault=HmColumnsIndel)
UNHARMONISED_REFERENCE = DeclaredReference(fault=UnharmonisedIndel)
_HM_ALLELE_COLUMNS = ("hm_other_allele", "hm_effect_allele")
#: Each harmonised layout, by whether the header names `hm_code` and how many
#: `hm_*` allele columns it names.
_LAYOUTS = {(True, 0): MAIN_COLUMNS_REFERENCE, (True, 2): HM_COLUMNS_REFERENCE}
#: A row of an Analysis that declares no imputation score (#175).
_UNDECLARED_SCORE = ImputationScore()
#: GWAS-SSF's identity and alias columns, the variant projection's declaration.
#: Public because a Column Map's translation is read in these columns (#281).
VARIANT_COLUMNS = VariantProjectionColumns(
    chromosome=(b"chromosome",),
    position=b"base_pair_location",
    ref=b"other_allele",
    alt=b"effect_allele",
    aliases=(b"rsid", b"variant_id"),
    reference=UNHARMONISED_REFERENCE,
)
_METRICS_COLUMNS = MetricsProjectionColumns(
    chromosome=(b"chromosome",),
    position=b"base_pair_location",
    ref=b"other_allele",
    alt=b"effect_allele",
    frequency=b"effect_allele_frequency",
    standard_error=b"standard_error",
    # The recovery columns (stores #176). A harmonised file may report its
    # effect as `odds_ratio` beside an empty `beta`, or its precision as a 95%
    # interval or a p-value instead of a `standard_error`; declaring them here
    # and nowhere else is what keeps the recovery GWAS-SSF's.
    effect_fallback=b"odds_ratio",
    ci_lower=b"ci_lower",
    ci_upper=b"ci_upper",
    p_value=b"p_value",
    reference=UNHARMONISED_REFERENCE,
)


def _rsid(rsid: str | None, variant_id: str | None) -> str:
    """The row's rs identifier, or "" if it names none.

    Harmonised GWAS-SSF carries a dedicated `rsid` column; `variant_id` is the
    harmonised (usually non-rs) identifier and is only a fallback. Anything that is not
    an rs identifier is dropped: it is not something a user can look the
    variant up by (issue #109). A compound or listed id is read through
    `normalise_rsid`, exactly as the variant projection reads it (issue #306).
    """
    for candidate in (rsid, variant_id):
        value = normalise_rsid(candidate or "")
        if value:
            return value
    return ""


def _row_cells(
    row: dict[str, str],
    source: EffectSource | None,
    sample_size_column: str | None,
    p_value: float | None,
) -> RowCells:
    """One row's candidate effect and precision cells, parsed by the shared rules.

    The column names are GWAS-SSF's own: `effect_allele_frequency` is the
    frequency of the effect allele, and `ci_lower`/`ci_upper` are the 95%
    interval's bounds. The second effect column is looked up by the spelling
    `resolve_effect_source` matched, padding included.

    A cell is parsed only where `row_statistics` reads it: the second effect
    column for a beta the row does not give, the interval and p-value for an
    SE it does not give, and the frequency and sample size for a signed z.
    The rest are left unread, which is what `row_statistics` makes of them.
    ``p_value`` is the row's `p_value` cell, which the caller parses for every
    row because the row carries it.
    """
    standard_error = parse_finite_float(row.get("standard_error"))
    if source is None:
        return RowCells(standard_error=standard_error)
    effect = parse_finite_float(row.get(source.column_name))
    if source.kind is EffectSourceKind.Z_SCORE:
        return _z_score_cells(row, effect, standard_error, sample_size_column)
    fallback_column = source.fallback_column_name
    odds_ratio = (
        parse_finite_float(row.get(fallback_column))
        if fallback_column and effect is None and source.kind is EffectSourceKind.BETA
        else None
    )
    if standard_error is not None and standard_error > 0.0:
        return RowCells(effect=effect, odds_ratio=odds_ratio, standard_error=standard_error)
    return RowCells(
        effect=effect,
        odds_ratio=odds_ratio,
        standard_error=standard_error,
        ci_lower=parse_finite_float(row.get("ci_lower")),
        ci_upper=parse_finite_float(row.get("ci_upper")),
        p_value=p_value,
    )


def _z_score_cells(
    row: dict[str, str],
    z: float | None,
    standard_error: float | None,
    sample_size_column: str | None,
) -> RowCells:
    """A signed z's row: its z, frequency and sample size, from which `row_statistics`
    derives the effect and SE (#215)."""
    return RowCells(
        effect=z,
        standard_error=standard_error,
        frequency=parse_af(row.get("effect_allele_frequency")),
        sample_size=(
            parse_finite_float(row.get(sample_size_column)) if sample_size_column else None
        ),
    )


def _tabular_row(
    row: dict[str, str],
    source: EffectSource | None,
    sample_size_column: str | None,
    score_column: str | None = None,
    frequencies: FrequencyTally | None = None,
    faults: IdentityFaults | None = None,
    infinite_effects: bool = False,
    *,
    declaration: DeclaredReference,
) -> SourceRow | None:
    """One source row as a `SourceRow`, or `None` when it names no canonical variant.

    A row whose chromosome, position or allele pair cannot be represented is
    dropped from every stream; a row with a valid identity but an unusable
    effect/`standard_error` still yields a row (its `beta`/`se` are `None`) so
    `stream_variants` can still see it per the interface's superset contract.
    A row too short to reach an allele has a missing allele, which `faults`
    names (#365), and so is an indel the row does not declare the reference
    of (``declaration``, ADR 0079). With ``infinite_effects`` (the association
    stream), an effect the row reports as infinite is its beta
    (`_infinite_effect`).
    """
    effect_allele = row.get("effect_allele")
    other_allele = row.get("other_allele")
    ori = orient_or_drop(
        row.get("chromosome"),
        row.get("base_pair_location"),
        effect_allele,
        other_allele,
        faults,
        other_is_reference=declaration.of_row(row),
        undeclared=declaration.fault,
    )
    # An allele a row does not reach never orients, so the two None checks
    # only say so to the type checker.
    if ori is None or effect_allele is None or other_allele is None:
        return None
    # The p-value is read on every row, so the row says what the source says
    # (#345); a row with no usable SE recovers its SE from the same parse.
    p_value = parse_finite_float(row.get("p_value"))
    statistics = row_statistics(source, _row_cells(row, source, sample_size_column, p_value))
    beta, from_fallback = statistics.beta, statistics.effect_from_odds_ratio_fallback
    if infinite_effects:
        beta, from_fallback = _reported_effect(row, source, beta, from_fallback)
    return _built_row(
        ori,
        (other_allele, effect_allele),
        statistics,
        beta,
        from_fallback,
        parse_af(row.get("effect_allele_frequency"), frequencies),
        (
            _UNDECLARED_SCORE
            if score_column is None
            else parse_imputation_score(row.get(score_column), declared=True)
        ),
        _rsid(row.get("rsid"), row.get("variant_id")),
        p_value,
    )


def _built_row(
    ori: Orientation,
    alleles: tuple[str, str],
    statistics: RowStatistics,
    beta: float | None,
    from_fallback: bool,
    af: float | None,
    score: ImputationScore,
    rsid: str,
    p_value: float | None,
) -> SourceRow:
    """One parsed row's `SourceRow`, built without its frozen ``__init__``.

    Every value is parsed and checked already, and a genome-wide file builds
    millions of these, so one dictionary update sets exactly the values the
    ``__init__`` would. ``alleles`` is the row's ``(other, effect)``.
    """
    variant = ori.variant
    built = object.__new__(SourceRow)
    object.__setattr__(
        built,
        "__dict__",
        {
            "chromosome": variant.chromosome,
            "position": variant.position,
            "ref": alleles[0],
            "alt": alleles[1],
            "alid": variant.alid,
            "flipped": ori.flipped,
            "af_alt": af,
            "beta": beta,
            "se": statistics.se,
            "imputation_score": score,
            "effect_from_odds_ratio_fallback": from_fallback,
            "se_from_ci": statistics.se_from_ci,
            "se_from_p_value": statistics.se_from_p_value,
            "p_value": p_value,
            "rsid": rsid,
            "z": None,
            "source_id": "",
        },
    )
    return built


def _reported_effect(
    row: dict[str, str], source: EffectSource | None, beta: float | None, from_fallback: bool
) -> tuple[float | None, bool]:
    """The row's beta for its association, and whether the fallback gave it.

    A beta read from the effect column itself is finite; only a row whose
    effect came from the fallback, or that has none, can report an infinity.
    """
    if beta is not None and not from_fallback:
        return beta, from_fallback
    infinite = _infinite_effect(row, source, recovered=beta is not None)
    return (beta, from_fallback) if infinite is None else (infinite, False)


def _infinite_effect(
    row: dict[str, str], source: EffectSource | None, *, recovered: bool
) -> float | None:
    """The infinite effect a row reports, for its association (D4).

    `row_statistics` reads an infinite effect as unusable and recovers the
    effect from the odds-ratio column instead, which the metrics projections
    need (ADR 0055). For the association an infinity the effect column reports
    is the row's effect, whatever the fallback holds, so its z is infinite and
    Admission drops it as `z_nonfinite` rather than admit the fallback. An
    infinite odds ratio is the effect only when nothing was ``recovered``. A
    z-score row derives its SE from the z, so an infinite z leaves it no SE,
    and `se_unusable` comes first.
    """
    if source is None:
        return None
    columns = [source.column_name]
    if not recovered and source.fallback_column_name:
        columns.append(source.fallback_column_name)
    for column in columns:
        effect = parse_effect(row.get(column))
        if effect is not None and math.isinf(effect):
            return effect
    return None


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


def _validated_score_column(fieldnames: Sequence[str] | None, name: str | None) -> str | None:
    if name is not None:
        resolve_imputation_score_column(
            [field.encode("utf-8") for field in fieldnames or ()], name
        )
    return name


def _open_text(path: SourcePath) -> IO[str]:
    """One source as text: a file by its own opener rule, a translated source as UTF-8.

    The file rule is unchanged by issue #281 -- only `.gz` is gunzipped here --
    so a file read by name gives exactly what it always gave.
    """
    if isinstance(path, TabularSource):
        return io.TextIOWrapper(path.open_binary(), encoding="utf-8", newline="")
    opener = gzip.open if str(path).endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8", newline="")


def _iter_rows(
    path: SourcePath,
    *,
    effect_source: EffectSource | None = None,
    score_column: str | None = None,
    tally: StreamTally | None = None,
    infinite_effects: bool = False,
) -> Iterator[SourceRow]:
    """Parse each row of a filtered/harmonised GWAS-SSF file once.

    `effect_source` is the caller's resolution for this file, so the property
    and the stream cannot disagree; when omitted it is resolved here, which is
    what keeps `stream_full_row_metrics` a faithful reference for the
    projection's effect handling. ``infinite_effects`` keeps an effect a row
    reports as infinite, which only the association stream asks for.
    """
    with _open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        source = effect_source or resolve_effect_source(reader.fieldnames or ())
        sample_size_column = _sample_size_column(reader.fieldnames, source)
        score_column = _validated_score_column(reader.fieldnames, score_column)
        tally = tally or StreamTally()
        faults, frequencies = tally.faults, tally.frequencies
        declaration = _declaration(path, reader.fieldnames)
        for row in reader:
            parsed = _tabular_row(
                row, source, sample_size_column, score_column, frequencies, faults,
                infinite_effects, declaration=declaration,
            )
            if parsed is None:
                continue
            tally.indels_kept_by_hm_code += _kept_by_code(declaration, parsed)
            yield parsed


def _kept_by_code(declaration: DeclaredReference, row: SourceRow) -> bool:
    """Whether ``row`` is an indel its harmoniser code placed."""
    return declaration.column is not None and len(row.ref.strip()) != len(row.alt.strip())


def _declaration(path: SourcePath, header: Sequence[str] | None = None) -> DeclaredReference:
    """Which of ``path``'s rows declare their `other_allele` the reference.

    A Column Map's translation declares by its allele convention, of every row
    or none. A GWAS-SSF file declares by its layout (`ssf_layout_reference`),
    read from ``header`` or, when not given, from the file.
    """
    if isinstance(path, TabularSource):
        return declared_reference(path.reference)
    return ssf_layout_reference(_header(path) if header is None else header, path)


def ssf_layout_reference(header: Sequence[str], source: object) -> DeclaredReference:
    """The reference declaration of a GWAS-SSF file whose header is ``header``.

    Its layout decides it (`MAIN_COLUMNS_REFERENCE`): `hm_code` with no `hm_*`
    allele column is the main-column layout; `hm_code` with both `hm_*` allele
    columns the `hm_*` layout; no harmoniser column at all an unharmonised
    file. Any other mix fits neither harmonised layout, and nothing says which
    columns hold its harmonised alleles, so it is refused rather than read
    under one of them.
    """
    harmoniser = sorted(name for name in header if name.startswith("hm_"))
    layout = ("hm_code" in harmoniser, sum(name in harmoniser for name in _HM_ALLELE_COLUMNS))
    declaration = _LAYOUTS.get(layout)
    if declaration is not None and (harmoniser or declaration is not UNHARMONISED_REFERENCE):
        return declaration
    if not harmoniser:
        return UNHARMONISED_REFERENCE
    raise ValueError(
        f"{source}: a header whose harmoniser columns are {harmoniser} fits neither "
        "GWAS-SSF harmonised layout (hm_code beside harmonised main columns, or hm_code with both "
        "hm_other_allele and hm_effect_allele), so the reference of its indels "
        "cannot be told (ADR 0079)"
    )


def _iter_variants(path: SourcePath) -> Iterator[SourceVariant]:
    yield from stream_projected_variants(
        path, replace(VARIANT_COLUMNS, reference=_declaration(path))
    )


def _header(path: SourcePath) -> list[str]:
    """A GWAS-SSF file's header names, reading no data row.

    Shares `_iter_rows`' opener rule so the effect source `GwasSsfReader`
    reports is the one its row parser would resolve for the same file.
    """
    with _open_text(path) as fh:
        return next(csv.reader(fh, delimiter="\t"), [])


def require_z_score_usable(
    path: SourcePath,
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
    `SourceRow` *is* a `TabularMetricsRow`; it adds only the identifier this
    seam never reads. It reads an infinite effect as the projections do, absent
    or recovered from the odds ratio; only the association stream keeps it (D4).
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

    `path` may be a `TabularSource` rather than a file: a Column Map's
    translation of a provider file into these columns (issue #281), which is
    how the generic tabular reader reads every mapped collection by this
    reader's rules.
    """

    path: SourcePath
    stored_effect_scale: StoredEffectScale = StoredEffectScale.SD
    chunk_rows: int = DEFAULT_CHUNK_ROWS
    imputation_score_declaration: ImputationScoreDeclaration | None = None

    @property
    def reference(self) -> ReferenceAllele:
        """Which allele the rows declare the genome's reference.

        A GWAS-SSF file's is `_REFERENCE`. A Column Map's translation declares
        its own: one whose `other_allele` is a `ref` column declares it (`REF`).
        """
        if isinstance(self.path, TabularSource):
            return self.path.reference
        return _REFERENCE

    @property
    def _metrics_columns(self) -> MetricsProjectionColumns:
        return replace(_METRICS_COLUMNS, reference=_declaration(self.path))

    @property
    def imputation_score_column(self) -> str | None:
        """Exact provider-backed column, or None when no score is declared."""
        declaration = self.imputation_score_declaration
        if declaration is None:
            return None
        resolve_imputation_score_column(
            [name.encode("utf-8") for name in _header(self.path)], declaration.column_name
        )
        return declaration.column_name

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

    def _association_rows(self, tally: StreamTally | None) -> Iterator[SourceRow]:
        """The rows both association projections read, an infinite effect kept."""
        source = self.effect_source
        # A z-score source is derived, not read: refuse it on a case-control
        # scale and refuse an unsigned column before yielding any row (#215).
        require_z_score_usable(self.path, source, self.stored_effect_scale)
        if source is None:
            # A file naming no effect column holds no associations at all, and
            # yields none, rather than one row with no effect per source row.
            return
        yield from _iter_rows(
            self.path,
            effect_source=source,
            score_column=self.imputation_score_column,
            tally=tally,
            infinite_effects=True,
        )

    def stream_associations(self, tally: StreamTally | None = None) -> Iterator[ReaderAssociation]:
        yield from stream_associations(
            self._association_rows(tally), self.stored_effect_scale, self.reference
        )

    def stream_batches(self, tally: StreamTally | None = None) -> Iterator[ReaderBatch]:
        yield from stream_batches(
            self._association_rows(tally), self.reference, chunk_rows=self.chunk_rows
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
        yield from stream_projected_metrics(
            self.path, self._metrics_columns, score_column=self.imputation_score_column
        )

    def stream_metric_chunks(self) -> Iterator[MetricsChunk]:
        """The same projection `stream_metrics` yields, a block at a time.

        What the one-pass resolver actually reads (issue #209): the projection
        is identical row for row, and normalising a block's distinct alleles
        once rather than every row's separately roughly halves the time a
        genome-wide source takes.
        """
        require_z_score_usable(self.path, self.effect_source, self.stored_effect_scale)
        yield from stream_projected_metric_chunks(
            self.path,
            self._metrics_columns,
            chunk_rows=self.chunk_rows,
            score_column=self.imputation_score_column,
        )

    def extract_at_sites(
        self, alids: Iterable[str], *, region: str | None = None, regions_file: RegionsFile = None
    ) -> dict[str, SiteMetrics]:
        return extract_at_sites(_iter_rows(self.path), alids, region=region)
