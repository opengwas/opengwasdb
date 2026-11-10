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

`stream_association_batches` (#319) is `stream_associations` as columns, a block
of the file at a time and held to it bit for bit (see "Association batches"
below); Pass 2 reads it.

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
import logging
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Protocol, cast

import numpy as np
import pandas as pd

import opengwasdb.readers.columnar as columnar
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
    AssociationBatch,
    ImputationScore,
    ImputationScoreDeclaration,
    ImputationScoreStatus,
    ReaderAssociation,
    RegionsFile,
    SiteMetrics,
    SourceVariant,
    StreamTally,
    batched,
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
    normalise_allele,
    normalise_chromosome,
    orient_or_drop,
)
from opengwasdb.variants.rsid import normalise_rsid

GWAS_SSF_CAPABILITY = "opengwasdb.gwas-ssf"
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


class _Cells(Protocol):
    """A row's cells as `_read_row_cells` reads them, by the header's own names."""

    def number(self, column: str) -> float | None:
        """The cell as `parse_finite_float` reads it."""
        ...

    def frequency(self) -> float | None:
        """The `effect_allele_frequency` cell as `parse_af` reads it, untallied."""
        ...


class _ParsedCells:
    """A row read as a mapping of header names to cell text."""

    __slots__ = ("row",)

    def __init__(self, row: Mapping[str, str | None]) -> None:
        self.row = row

    def number(self, column: str) -> float | None:
        return parse_finite_float(self.row.get(column))

    def frequency(self) -> float | None:
        return parse_af(self.row.get("effect_allele_frequency"))


def _read_row_cells(
    cells: _Cells,
    source: EffectSource | None,
    sample_size_column: str | None,
    p_value: float | None,
) -> RowCells:
    """One row's candidate effect and precision cells, parsed by the shared rules.

    The column names are GWAS-SSF's own: `effect_allele_frequency` is the
    frequency of the effect allele, and `ci_lower`/`ci_upper` are the 95%
    interval's bounds. The second effect column is looked up by the spelling
    `resolve_effect_source` matched, padding included.

    A cell is read only where `row_statistics` reads it: the second effect
    column for a beta the row does not give, the interval and p-value for an
    SE it does not give, and the frequency and sample size for a signed z.
    The rest are left unread, which is what `row_statistics` makes of them.
    ``p_value`` is the row's `p_value` cell, which the caller reads for every
    row because the row carries it. Both the per-row reader and the batch path
    form a row's cells here, so the two cannot read a column differently.
    """
    standard_error = cells.number("standard_error")
    if source is None:
        return RowCells(standard_error=standard_error)
    effect = cells.number(source.column_name)
    if source.kind is EffectSourceKind.Z_SCORE:
        return _z_score_cells(cells, effect, standard_error, sample_size_column)
    fallback_column = source.fallback_column_name
    odds_ratio = (
        cells.number(fallback_column)
        if fallback_column and effect is None and source.kind is EffectSourceKind.BETA
        else None
    )
    if standard_error is not None and standard_error > 0.0:
        return RowCells(effect=effect, odds_ratio=odds_ratio, standard_error=standard_error)
    return RowCells(
        effect=effect,
        odds_ratio=odds_ratio,
        standard_error=standard_error,
        ci_lower=cells.number("ci_lower"),
        ci_upper=cells.number("ci_upper"),
        p_value=p_value,
    )


def _row_cells(
    row: Mapping[str, str | None],
    source: EffectSource | None,
    sample_size_column: str | None,
    p_value: float | None,
) -> RowCells:
    """`_read_row_cells` of a row read as a mapping of header names to cell text."""
    return _read_row_cells(_ParsedCells(row), source, sample_size_column, p_value)


def _z_score_cells(
    cells: _Cells,
    z: float | None,
    standard_error: float | None,
    sample_size_column: str | None,
) -> RowCells:
    """A signed z's row: its z, frequency and sample size, from which `row_statistics`
    derives the effect and SE (#215)."""
    return RowCells(
        effect=z,
        standard_error=standard_error,
        frequency=cells.frequency(),
        sample_size=cells.number(sample_size_column) if sample_size_column else None,
    )


def _tabular_row(
    row: Mapping[str, str | None],
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
        declaration=declaration,
        row=row,
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
    row: Mapping[str, str | None],
    source: EffectSource | None,
    beta: float | None,
    from_fallback: bool,
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
    row: Mapping[str, str | None], source: EffectSource | None, *, recovered: bool
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
    columns = _infinite_columns(source, recovered=recovered)
    return _first_infinite(parse_effect(row.get(column)) for column in columns)


def _infinite_columns(source: EffectSource, *, recovered: bool) -> list[str]:
    """The columns whose infinite value is a row's effect: the effect column, and
    the odds-ratio fallback unless the effect was ``recovered`` from something."""
    columns = [source.column_name]
    if not recovered and source.fallback_column_name:
        columns.append(source.fallback_column_name)
    return columns


def _first_infinite(effects: Iterable[float | None]) -> float | None:
    """The first of ``effects`` (each as `parse_effect` reads a cell) that is infinite."""
    for effect in effects:
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


def _gunzipped(path: str | Path) -> bool:
    """Whether a file is read through gzip: only `.gz` is (`.bgz` is not), for
    every way this reader opens one."""
    return str(path).endswith(".gz")


def _open_text(path: SourcePath) -> IO[str]:
    """One source as text: a file by its own opener rule, a translated source as UTF-8.

    The file rule is unchanged by issue #281 -- only `.gz` is gunzipped here --
    so a file read by name gives exactly what it always gave.
    """
    if isinstance(path, TabularSource):
        return io.TextIOWrapper(path.open_binary(), encoding="utf-8", newline="")
    opener = gzip.open if _gunzipped(path) else open
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
        yield from _parsed_rows(
            reader, source, sample_size_column, score_column, tally or StreamTally(),
            infinite_effects, _declaration(path, reader.fieldnames),
        )


def _parsed_rows(
    rows: Iterable[Mapping[str, str | None]],
    source: EffectSource | None,
    sample_size_column: str | None,
    score_column: str | None,
    tally: StreamTally,
    infinite_effects: bool,
    declaration: DeclaredReference,
) -> Iterator[SourceRow]:
    """`_tabular_row` of each of ``rows`` (a file read as mappings of header names
    to cells) that names a variant, counting the indels a harmoniser code placed."""
    faults, frequencies = tally.faults, tally.frequencies
    for row in rows:
        parsed = _tabular_row(
            row, source, sample_size_column, score_column, frequencies, faults,
            infinite_effects, declaration=declaration,
        )
        if parsed is None:
            continue
        tally.indel_rows_kept_by_hm_code += _kept_by_code(declaration, parsed)
        yield parsed


def _kept_by_code(declaration: DeclaredReference, row: SourceRow) -> bool:
    """Whether ``row`` is an indel its harmoniser code placed."""
    return declaration.column is not None and len(row.ref.strip()) != len(row.alt.strip())


def _declaration(path: SourcePath, header: Sequence[str] | None = None) -> DeclaredReference:
    """Which of ``path``'s rows declare their `other_allele` the reference.

    A Column Map's translation declares by its allele convention, of every row
    or none. A GWAS-SSF file declares by its layout (`_ssf_layout_reference`),
    read from ``header`` or, when not given, from the file.
    """
    if isinstance(path, TabularSource):
        return declared_reference(path.reference)
    return _ssf_layout_reference(_header(path) if header is None else header, path)


def _ssf_layout_reference(header: Sequence[str], source: object) -> DeclaredReference:
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
    if declaration is not None:
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


# ── Association batches (#319) ───────────────────────────────────────────────
#
# Pass 2 reads a GWAS-SSF source's associations as columns, a block of the
# decompressed file at a time, by the engine FinnGen's batches share
# (`opengwasdb.readers.columnar`). The per-row reader (`_tabular_row`) is the
# reference: a block holds exactly its associations, in order, bit for bit.
#
# A row is decided by its columns when its identity passes every rule and the
# indel it may be is one its layout declares the reference of (ADR 0079); its
# statistics are then formed as `row_statistics` forms them, from the cells
# already parsed. Any other row -- one the rules refuse, whose refusal the tally
# must name -- is read by `_tabular_row` from its own cells. A block with a quote,
# a carriage return, a NUL or a byte outside ASCII, and a source with a declared
# imputation score, go to the per-row reader whole.

#: Decompressed bytes one block reads, 32 MiB (see `finngen._BLOCK_BYTES`).
_BLOCK_BYTES = 1 << 25

#: The cells an association reads whatever the file's effect column is.
_IDENTITY_COLUMNS = ("chromosome", "base_pair_location", "effect_allele", "other_allele")
#: The statistic columns each association reads, by what the block calls them.
_STATISTIC_NAMES = {
    "se": "standard_error", "lower": "ci_lower", "upper": "ci_upper", "p_value": "p_value",
    "af": "effect_allele_frequency",
}

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Plan:
    """Where a source's header puts each cell an association reads.

    ``names`` are the header's names as `csv` reads a header with no quote. A
    repeated name is its last column, as a `csv.DictReader` row holds it.
    """

    names: list[str]
    reads: dict[str, int]

    @classmethod
    def of(
        cls,
        names: list[str],
        source: EffectSource,
        sample_size_column: str | None,
        score_column: str | None,
        declaration: DeclaredReference,
    ) -> _Plan | str:
        """The plan for a header whose names are ``names``, or why the columnar
        parse leaves the source to the per-row reader: a column that names a
        variant is missing."""
        at = {name: index for index, name in enumerate(names)}
        missing = [name for name in _IDENTITY_COLUMNS if name not in at]
        if missing:
            return f"its header names no {', '.join(missing)} column"
        read = (source.column_name, source.fallback_column_name, sample_size_column, score_column)
        wanted = (
            *_IDENTITY_COLUMNS, *_STATISTIC_NAMES.values(), *read, declaration.column_name,
        )
        return cls(names, {name: at[name] for name in dict.fromkeys(wanted) if name in at})


@dataclass(frozen=True)
class _Format:
    """GWAS-SSF's columns and rules, to the engine; ``plan`` is ``None`` for a
    source the per-row reader reads throughout."""

    plan: _Plan | None
    source: EffectSource
    sample_size_column: str | None
    score_column: str | None
    declaration: DeclaredReference
    log: logging.Logger = log
    label: str = "GWAS-SSF source"

    def block_bytes(self) -> int:
        return _BLOCK_BYTES

    def per_row(
        self, text: io.TextIOBase, names: list[str] | None, source: columnar.Source
    ) -> Iterator[AssociationBatch]:
        rows = csv.DictReader(text, fieldnames=names, delimiter="\t")
        parsed = _parsed_rows(
            rows, self.source, self.sample_size_column, self.score_column, source.tally,
            True, self.declaration,
        )
        associations = stream_associations(parsed, source.scale)
        yield from batched(
            associations, rows=columnar.ROW_BATCH, scores=self.score_column is not None
        )

    def block_batches(
        self, lines: bytes, offset: int, source: columnar.Source
    ) -> Iterator[AssociationBatch]:
        plan = self.plan
        assert plan is not None
        yield from columnar.block_batches(
            lines,
            offset,
            source,
            lambda text: _Block.read(text, plan, self),
            lambda block, text: block.associations(text, self, source),
            plan.names,
        )


def _open_binary(path: SourcePath) -> IO[bytes]:
    """One source's bytes, opened as `_open_text` opens it: only `.gz` is gunzipped."""
    if isinstance(path, TabularSource):
        return path.open_binary()
    return cast(IO[bytes], gzip.open(path, "rb") if _gunzipped(path) else open(path, "rb"))


def _association_batches(
    path: SourcePath,
    source: EffectSource,
    score_column: str | None,
    tally: StreamTally,
    scale: StoredEffectScale,
) -> Iterator[AssociationBatch]:
    """``path``'s associations as batches, with the rules `_iter_rows` applies."""
    names = _header(path)
    sample_size_column = _sample_size_column(names, source)
    score_column = _validated_score_column(names, score_column)
    declaration = _declaration(path, names)

    def start(header: list[str] | str) -> tuple[columnar.Format, str | None]:
        plan = None if isinstance(header, str) else _Plan.of(
            header, source, sample_size_column, score_column, declaration
        )
        refusal = plan if isinstance(plan, str) else None
        fmt = _Format(
            None if isinstance(plan, str) else plan, source, sample_size_column, score_column,
            declaration,
        )
        return fmt, refusal

    with _open_binary(path) as fh:
        # GWAS Catalog files end every line in CR LF, which `csv` reads as plain.
        yield from columnar.association_batches(fh, path, tally, scale, start, crlf=True)


@dataclass(frozen=True)
class _Block:
    """One block's identity cells and numbers, before any row is decided.

    A number is an array of the block's rows, NaN where the cell is missing,
    ``None`` where pandas read some cell otherwise than as `float()` reads it
    and so no row of the block is decided. An absent column is all missing.
    """

    identity: columnar.Identity
    #: Per row: whether the row declares its other allele the reference, and
    #: whether the raw alleles differ in length once stripped (`_kept_by_code`).
    declared: np.ndarray
    raw_indel: np.ndarray
    effect: np.ndarray | None
    fallback: np.ndarray | None
    se: np.ndarray | None
    lower: np.ndarray | None
    upper: np.ndarray | None
    p_value: np.ndarray | None
    af: np.ndarray | None
    sample_size: np.ndarray | None
    #: The declared imputation score column, or ``None`` where none is declared.
    score: np.ndarray | None
    #: Each statistic column by the header's name for it, as `_Cells` reads it.
    by_name: dict[str, np.ndarray | None]

    @classmethod
    def read(cls, lines: bytes, plan: _Plan, fmt: _Format) -> _Block:
        reads = plan.reads
        names = _STATISTIC_NAMES | {
            "effect": fmt.source.column_name,
            "fallback": fmt.source.fallback_column_name,
            "sample_size": fmt.sample_size_column,
            "score": fmt.score_column,
        }
        index = {key: reads.get(name) if name else None for key, name in names.items()}
        hm = fmt.declaration.index({name.encode(): at for name, at in reads.items()})
        frame = _frame(lines, plan, index, hm)
        # `ref` is the other allele and `alt` the effect allele, as `orient_or_drop` is given them.
        identity = columnar.Identity.read(
            frame, reads["chromosome"], reads["other_allele"], reads["effect_allele"],
            reads["base_pair_location"], normalise_chromosome, normalise_allele,
        )
        arrays = {key: columnar.numbers(frame, index[key], "f") for key in names}
        by_name = {name: array for key, array in arrays.items() if (name := names[key])}
        arrays["score"] = arrays["score"] if fmt.score_column else None
        return cls(
            identity,
            _declared(frame, hm, fmt.declaration),
            _raw_indel(identity.ref, identity.alt),
            **arrays,
            by_name=by_name,
        )

    def decided(self) -> np.ndarray:
        """The rows the columns decide: every identity rule passed, an indel only
        where the row declares its reference, and every number read as the
        per-row reader reads it."""
        if any(values is None for values in self._numbers()):
            return np.zeros(len(self.identity), dtype=bool)
        indel = columnar.is_indel(self.identity.ref, self.identity.alt)
        decided: np.ndarray = self.identity.valid() & (~indel | self.declared)
        return decided

    def _numbers(self) -> list[np.ndarray | None]:
        """Every column read as numbers, an absent one as all missing."""
        return list(self.by_name.values())

    def associations(
        self, lines: bytes, fmt: _Format, source: columnar.Source
    ) -> AssociationBatch:
        """The block's associations: the decided rows from the columns, the rest
        from `_tabular_row`, in the block's order."""
        plan = fmt.plan
        assert plan is not None
        tally = source.tally
        decided = self.decided()
        # A frequency pandas reads as NaN was one of the missing tokens, which
        # `parse_af` reads as absent and nothing counts; any other spelling of
        # NaN makes the whole column text, and the block is left to the rows.
        af = np.full(len(decided), np.nan) if self.af is None else self.af
        columnar.made_absent(af, decided, tally.frequencies)
        if fmt.declaration.column is not None:
            tally.indel_rows_kept_by_hm_code += int(np.count_nonzero(decided & self.raw_indel))
        beta, se = _statistics(self, decided, fmt.source, fmt.sample_size_column)
        reported = np.divide(
            beta, se, out=np.full(len(decided), np.nan),
            where=decided & ~np.isnan(beta) & ~np.isnan(se),
        )
        columns = self.identity.columns(reported, se, af, decided)
        if fmt.score_column:
            # A block the columns do not decide holds no score array; each of its
            # rows then takes its score from `_tabular_row`.
            columns.imputation_score, columns.imputation_status = _scores(
                np.full(len(decided), np.nan) if self.score is None else self.score
            )
        undecided = np.flatnonzero(~decided)
        if len(undecided):
            source.undecided += len(undecided)
            columns.place(_undecided(lines, plan, undecided, fmt, source))
        return columns.batch()


def _finite(values: np.ndarray | None, rows: int) -> np.ndarray:
    """``values`` where finite, else NaN: `parse_finite_float`'s answer, None as NaN."""
    if values is None:
        return np.full(rows, np.nan)
    return np.where(np.isfinite(values), values, np.nan)


def _statistics(
    block: _Block, decided: np.ndarray, source: EffectSource, sample_size_column: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """The decided rows' ``(beta, se)``, NaN where `row_statistics` gives none.

    The rows every column reports plainly -- a usable effect and a positive
    SE; a z with the frequency and sample size it derives from -- are formed as
    arrays. The rest go through `row_statistics` itself, from the cells
    already parsed: the second effect column, an interval or p-value for the
    SE, and an effect the row reports as infinite.
    """
    rows = len(decided)
    beta, se = np.full(rows, np.nan), np.full(rows, np.nan)
    effect = _finite(block.effect, rows)
    reported = _finite(block.se, rows)
    with np.errstate(all="ignore"):
        reported_ok = reported > 0
        if source.kind is EffectSourceKind.Z_SCORE:
            fast = _derive_z(block, effect, decided, beta, se)
        elif source.kind is EffectSourceKind.ODDS_RATIO:
            fast = decided & (effect > 0) & reported_ok
            beta[fast] = [math.log(odds) for odds in effect[fast].tolist()]
            se[fast] = reported[fast]
        else:
            fast = decided & ~np.isnan(effect) & reported_ok
            beta[fast] = effect[fast]
            se[fast] = reported[fast]
    for row in np.flatnonzero(decided & ~fast).tolist():
        beta[row], se[row] = _row_statistics(block, row, source, sample_size_column)
    return beta, se


def _derive_z(
    block: _Block, z: np.ndarray, decided: np.ndarray, beta: np.ndarray, se: np.ndarray
) -> np.ndarray:
    """A signed z's ``(beta, se)`` where `derive_z_score_effect` derives them
    without dividing by zero, which the per-row rule would raise on: the rows
    returned are those formed. ``z`` is the z where finite."""
    rows = len(decided)
    frequency = np.full(rows, np.nan) if block.af is None else block.af
    sample_size = _finite(block.sample_size, rows)
    usable: np.ndarray = (
        decided & ~np.isnan(z) & (frequency > 0) & (frequency < 1) & (sample_size > 0)
    )
    variance = 2.0 * frequency * (1.0 - frequency)
    precision = variance * (sample_size + z * z)
    usable &= np.isfinite(precision) & (precision > 0)
    derived = 1.0 / np.sqrt(precision[usable])
    se[usable] = derived
    beta[usable] = z[usable] * derived
    return usable


class _BlockCells:
    """One row of a block as `_read_row_cells` reads it: the columns already parsed,
    each cell as `parse_finite_float` reads it, a missing one as ``None``."""

    __slots__ = ("block", "row")

    def __init__(self, block: _Block, row: int) -> None:
        self.block, self.row = block, row

    def number(self, column: str) -> float | None:
        values = self.block.by_name.get(column)
        if values is None:
            return None
        value = float(values[self.row])
        return value if math.isfinite(value) else None

    def frequency(self) -> float | None:
        value = self.number("effect_allele_frequency")
        return value if value is not None and 0.0 < value < 1.0 else None


def _scores(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A declared score column's value and status per row, as
    `parse_imputation_score` reads each cell: a missing token (NaN, the only
    spelling pandas reads as NaN) is MISSING, an infinity NONFINITE, any other
    number USABLE inside [0, 1] and OUT_OF_RANGE outside it."""
    finite = np.isfinite(values)
    status = np.full(len(values), ImputationScoreStatus.USABLE, dtype=object)
    status[np.isnan(values)] = ImputationScoreStatus.MISSING
    status[np.isinf(values)] = ImputationScoreStatus.NONFINITE
    status[finite & ((values < 0.0) | (values > 1.0))] = ImputationScoreStatus.OUT_OF_RANGE
    return np.where(finite, values, np.nan), status


def _row_statistics(
    block: _Block, row: int, source: EffectSource, sample_size_column: str | None
) -> tuple[float, float]:
    """One row's ``(beta, se)`` by `row_statistics` and `_reported_effect`, NaN for none."""
    cells = _BlockCells(block, row)
    statistics = row_statistics(
        source, _read_row_cells(cells, source, sample_size_column, cells.number("p_value"))
    )
    beta, from_fallback = statistics.beta, statistics.effect_from_odds_ratio_fallback
    if beta is None or from_fallback:
        infinite = _first_infinite(
            None if (values := block.by_name.get(column)) is None else float(values[row])
            for column in _infinite_columns(source, recovered=beta is not None)
        )
        beta = beta if infinite is None else infinite
    return (
        math.nan if beta is None else beta,
        math.nan if statistics.se is None else statistics.se,
    )


def _undecided(
    lines: bytes, plan: _Plan, rows: np.ndarray, fmt: _Format, source: columnar.Source
) -> Iterator[tuple[int, ReaderAssociation | None]]:
    """Each of ``rows``, read by `_tabular_row` from its own line."""
    tally = source.tally
    faults, frequencies = tally.faults, tally.frequencies
    for row, cells in columnar.line_cells(lines, plan.reads, rows):
        parsed = _tabular_row(
            cells, fmt.source, fmt.sample_size_column, fmt.score_column, frequencies, faults,
            True, declaration=fmt.declaration,
        )
        if parsed is None:
            yield row, None
            continue
        tally.indel_rows_kept_by_hm_code += _kept_by_code(fmt.declaration, parsed)
        yield row, next(stream_associations([parsed], source.scale))


def _frame(
    lines: bytes, plan: _Plan, index: dict[str, int | None], hm: int | None
) -> pd.DataFrame:
    """A block's cells ``plan`` reads: the identity columns and the harmoniser
    code as distinct text, the statistic columns in ``index`` as `float()` reads
    them, and a declared score with the tokens `parse_imputation_score` reads as
    missing."""
    reads = plan.reads
    categories = [reads[name] for name in ("chromosome", "effect_allele", "other_allele")]
    missing = {at: list(columnar.MISSING_STATISTICS) for at in index.values() if at is not None}
    if index["score"] is not None:
        missing[index["score"]] = list(columnar.MISSING_SCORES)
    return pd.read_csv(
        io.BytesIO(lines),
        names=range(len(plan.names)),
        usecols=sorted({*reads.values()}),
        dtype={at: "category" for at in (*categories, *([hm] if hm is not None else []))},
        na_values=missing,
        # `float()` is correctly rounded and pandas' default converter is not.
        float_precision="round_trip",
        **columnar.PANDAS,
    )


def _declared(frame: pd.DataFrame, hm: int | None, declaration: DeclaredReference) -> np.ndarray:
    """Per row: whether it declares its other allele the reference (ADR 0079)."""
    rows = len(frame)
    if declaration.every_row:
        return np.ones(rows, dtype=bool)
    if hm is None:
        return np.zeros(rows, dtype=bool)
    column = frame[hm]
    categories = column.cat.categories.to_numpy(dtype=object)
    declares = np.array([declaration.of_cell(cell) for cell in categories.tolist()], dtype=bool)
    codes = column.cat.codes.to_numpy()
    return np.zeros(rows, dtype=bool) if not len(declares) else declares[codes]


def _raw_indel(ref: columnar.Cells, alt: columnar.Cells) -> np.ndarray:
    lengths = np.vectorize(lambda cell: len(cell.strip()), otypes=[np.int64])
    different: np.ndarray = ref.rows(lengths(ref.verbatim)) != alt.rows(lengths(alt.verbatim))
    return different



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
        yield from stream_associations(self._association_rows(tally), self.stored_effect_scale)

    def stream_association_batches(
        self, tally: StreamTally | None = None
    ) -> Iterator[AssociationBatch]:
        """`stream_associations`, a block of the file at a time, as columns
        (#319): what Pass 2 routes. The same associations bit for bit, and the
        same ``tally``; see `_association_batches`."""
        source = self.effect_source
        require_z_score_usable(self.path, source, self.stored_effect_scale)
        if source is None:
            return
        yield from _association_batches(
            self.path, source, self.imputation_score_column, tally or StreamTally(),
            self.stored_effect_scale,
        )

    def stream_batches(self, tally: StreamTally | None = None) -> Iterator[ReaderBatch]:
        yield from stream_batches(self._association_rows(tally), chunk_rows=self.chunk_rows)

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
