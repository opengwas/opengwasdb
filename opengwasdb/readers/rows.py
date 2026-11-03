"""The rows every Source Reader reads, and the projections built from them.

Each Source Format's reader parses its source into rows -- a `SourceRow` per
association, oriented by `opengwasdb.variants.normalise` and carrying the
source's own effect allele's statistics -- and this module presents them:

- `stream_associations`, one `ReaderAssociation` per row, which Variant
  Routing's Pass 2 and phenotype-SD estimation read;
- `stream_variants`, each row's variant and rsid;
- `metrics_chunks_from_rows`, the resolver's block form (`MetricsChunk`);
- `extract_at_sites`, A1 frequency and SE at requested sites.

Every one orients through `opengwasdb.readers.orientation`, so a flipped row's
z is negated and its frequency complemented in one place. A reader with a
faster path for one projection (a column-projected scan, a byte-level variant
loop) keeps it, held to these by parity tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, KeysView
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import DTypeLike

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.interface import (
    ImputationScore,
    ImputationScoreStatus,
    ReaderAssociation,
    ReferenceAllele,
    SiteMetrics,
    SourceVariant,
)
from opengwasdb.readers.orientation import a1_frequencies, is_palindromic, swap_one

__all__ = [
    "DEFAULT_CHUNK_ROWS",
    "MetricsChunk",
    "SourceRow",
    "TabularMetricsRow",
    "extract_at_sites",
    "metrics_chunks_from_rows",
    "stream_associations",
    "stream_variants",
]


@dataclass(frozen=True)
class TabularMetricsRow:
    """One source row's variant identity and the statistics both pre-build
    annotation stages read (issue #207).

    The parent of :class:`SourceRow`, which adds the source's own identifier for
    the row and nothing else. The split is not cosmetic: the one-pass resolver
    reads a genome-wide file per Analysis and never looks at an rsid, so the
    metrics projection does not pay for the identifier columns or for decoding
    them -- and a row shape that could not name an rsid is not a row shape
    anything else should be handed.

    ``af_alt`` is the source's own frequency for its own effect allele -- not
    A1-oriented, which is why ``flipped`` travels beside it (ADR 0036). ``ref``
    and ``alt`` are the source's own labels, verbatim.
    """

    chromosome: str
    position: int
    ref: str
    alt: str
    alid: str
    flipped: bool
    af_alt: float | None
    beta: float | None
    se: float | None
    imputation_score: ImputationScore = ImputationScore()
    #: Which columns the row's own values came from (stores #176), carried here
    #: so the blocked projection built from `metrics_chunks_from_rows` reports
    #: the same provenance the projections themselves do.
    effect_from_odds_ratio_fallback: bool = False
    se_from_ci: bool = False
    se_from_p_value: bool = False


#: Rows per block. Peak memory is a block's ALID strings rather than the whole
#: source, so this is the knob that bounds a resolver worker's footprint: 50,000
#: costs about 23 MB against a genome-wide source where 1,000,000 costs 359 MB,
#: and measures fractionally *faster* for it (issue #209).
DEFAULT_CHUNK_ROWS = 50_000


@dataclass(frozen=True)
class MetricsChunk:
    """A block of projected source rows held as columns (issue #209).

    The column-oriented counterpart of :class:`TabularMetricsRow`: the same
    projection over many rows at once, so the resolver's per-row Python cost --
    a frozen dataclass, an f-string ALID and six parse calls for every row of a
    genome-wide file -- becomes a handful of array operations per block. Only
    rows naming a usable canonical variant are present, which is exactly the set
    `stream_projected_metrics` yields, in the same order.

    `palindromic` travels instead of the source's own `ref`/`alt` labels because
    deciding strand ambiguity is the only thing those labels are read for. It is
    computed from the *verbatim* labels, as `is_palindromic` is given them
    row-wise -- not from the normalised alleles, which differ for a
    whitespace-padded cell. Matching the row-wise answer matters more than
    matching the tidier one.

    Absent statistics are `NaN`, never `0.0`. Each of `af_alt`, `beta` and `se`
    carries the usability rule its row-wise `parse_*` counterpart applies, so a
    value present here is one the row-wise path would also have reported.
    """

    alid: np.ndarray
    flipped: np.ndarray
    palindromic: np.ndarray
    af_alt: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    imputation_score: np.ndarray = field(default_factory=lambda: np.array([], dtype="float64"))
    imputation_score_status: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    #: Which column each row's effect and standard error came from (stores
    #: #176), one flag per row: `effect_from_odds_ratio_fallback` is a row whose
    #: `beta` was unusable and whose `odds_ratio` supplied the effect, and
    #: `se_from_ci` / `se_from_p_value` are the two derived standard errors.
    #: They are what the resolver's build-eligible counts report, so they cover
    #: exactly the rows `beta`/`se` carry values for.
    effect_from_odds_ratio_fallback: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    se_from_ci: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    se_from_p_value: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    def __post_init__(self) -> None:
        _fill_default_arrays(self)

    def __len__(self) -> int:
        return int(self.alid.shape[0])

    def a1_frequency(self) -> np.ndarray:
        """Each row's frequency of its canonical A1, NaN where it reports none."""
        return a1_frequencies(self.flipped, self.af_alt)


#: Every `MetricsChunk` array a reader may leave out entirely, with the value it
#: is filled with when absent. One list, so the default fill and the alignment
#: check cannot cover different sets of arrays.
_CHUNK_DEFAULTS: tuple[tuple[str, object, object], ...] = (
    ("imputation_score", math.nan, None),
    ("imputation_score_status", ImputationScoreStatus.UNDECLARED, object),
    ("effect_from_odds_ratio_fallback", False, bool),
    ("se_from_ci", False, bool),
    ("se_from_p_value", False, bool),
)


def _fill_default_arrays(chunk: MetricsChunk) -> None:
    """Fill each absent `MetricsChunk` array with its own "nothing to report" value.

    A blocked reader built before a field existed leaves its array empty: the
    score arrays (#175) and the effect/precision provenance (stores #176). What
    fills them is an absence -- `NaN`, `UNDECLARED`, `False` -- never a
    plausible measurement, so a reader that had nothing to say cannot be read as
    having said something. Every array's length is checked here, because the
    accumulation downstream indexes them against `alid`.
    """
    size = len(chunk.alid)
    for name, fill, dtype in _CHUNK_DEFAULTS:
        array = getattr(chunk, name)
        if not array.size and size:
            array = np.full(size, fill, dtype=dtype)
            object.__setattr__(chunk, name, array)
        if len(array) != size:
            raise ValueError(f"{name} must align with MetricsChunk.alid")


def metrics_chunks_from_rows(
    rows: Iterable[TabularMetricsRow], *, chunk_rows: int = DEFAULT_CHUNK_ROWS
) -> Iterator[MetricsChunk]:
    """Block a row-wise projection, for a reader that has no blocked one.

    The resolver accumulates from :class:`MetricsChunk` and only that, so there
    is one accumulation to get right rather than a fast one and a slow one that
    can disagree. A reader that can only yield rows pays for the blocking and
    gets the same answer (issue #209).
    """
    batch: list[TabularMetricsRow] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= chunk_rows:
            yield _chunk_of_rows(batch)
            batch = []
    if batch:
        yield _chunk_of_rows(batch)


def _score_chunk_arrays(rows: list[TabularMetricsRow]) -> tuple[np.ndarray, np.ndarray]:
    return (
        _row_array(rows, lambda row: _or_nan(row.imputation_score.value), "float64"),
        _row_array(rows, lambda row: row.imputation_score.status, object),
    )


def _row_array(
    rows: list[TabularMetricsRow],
    value: Callable[[TabularMetricsRow], object],
    dtype: DTypeLike,
) -> np.ndarray:
    """One chunk array, from a row-wise projection's rows, in row order."""
    return np.array([value(row) for row in rows], dtype=dtype)


def _chunk_of_rows(rows: list[TabularMetricsRow]) -> MetricsChunk:
    scores, statuses = _score_chunk_arrays(rows)
    return MetricsChunk(
        alid=_row_array(rows, lambda row: row.alid, object),
        flipped=_row_array(rows, lambda row: row.flipped, bool),
        palindromic=_row_array(rows, lambda row: is_palindromic(row.ref, row.alt), bool),
        af_alt=_row_array(rows, lambda row: _or_nan(row.af_alt), "float64"),
        beta=_row_array(rows, lambda row: _or_nan(row.beta), "float64"),
        se=_row_array(rows, lambda row: _or_nan(row.se), "float64"),
        imputation_score=scores,
        imputation_score_status=statuses,
        # The row's own provenance travels with it, so a reader that can only
        # yield rows still reports the resolver's counts from the same values
        # the blocked projection would (stores #176).
        effect_from_odds_ratio_fallback=_row_array(
            rows, lambda row: row.effect_from_odds_ratio_fallback, bool
        ),
        se_from_ci=_row_array(rows, lambda row: row.se_from_ci, bool),
        se_from_p_value=_row_array(rows, lambda row: row.se_from_p_value, bool),
    )


def _or_nan(value: float | None) -> float:
    return math.nan if value is None else value


@dataclass(frozen=True)
class SourceRow(TabularMetricsRow):
    """One source row, as every Source Format's reader hands it to the presenters.

    :class:`TabularMetricsRow` plus what only some projections read:

    - ``rsid``, the source's own rs identifier for the row, blank when it records
      none, never fabricated;
    - ``z``, a z the source reports itself (GWAS-VCF's ``EZ``, or ``ES/SE`` where it
      reports no ``EZ``), so a stored z is the reported one rather than rebuilt as
      ``beta / se``. ``None`` where the source reports none, and the association's z
      is then ``beta / se``;
    - ``source_id``, the row's identifier as the source spells it, blank when the
      format names none.

    Its ``beta`` is the effect the row reports, an infinite one included, which
    gives the association an infinite z for Association Admission to drop (D4);
    the metrics projections read an infinite beta as absent.
    """

    rsid: str = ""
    z: float | None = None
    source_id: str = ""


def _association_z(row: SourceRow) -> float:
    """The row's z in its own orientation: the reported one, else ``beta / se``.

    NaN where it has neither an effect nor a usable SE: such a row is not an
    association, and Association Admission drops and logs it.
    """
    if row.z is not None:
        return row.z
    if row.beta is None or row.se is None:
        return math.nan
    return row.beta / row.se


def stream_associations(
    rows: Iterable[SourceRow],
    stored_effect_scale: StoredEffectScale,
    reference: ReferenceAllele = ReferenceAllele.UNDECLARED,
) -> Iterator[ReaderAssociation]:
    """Each row as a `ReaderAssociation`, oriented to A1 by `swap_one`.

    ``reference`` is what the format declares of every row's alleles, and is
    carried onto each association unchanged. A row with no usable SE has a NaN
    SE, and one with no effect a NaN z, for Association Admission to drop and
    log; a reader recovers an SE it can (a CI, a p) before this.
    """
    # Each association is built without its frozen ``__init__``: a Dense or
    # Hybrid build routes one per source row, and the ``__init__`` with its
    # frequency check is half of what the stream costs. The check cannot fail
    # here, since ``eaf`` is `parse_af`'s value, inside (0, 1), or its
    # complement, so one dictionary update gives the same value.
    new, set_dict = object.__new__, object.__setattr__
    for row in rows:
        reported = row.z
        if reported is None:
            beta, se = row.beta, row.se
            reported = math.nan if beta is None or se is None else beta / se
        z, eaf = swap_one(row.flipped, reported, row.af_alt)
        association = new(ReaderAssociation)
        set_dict(
            association,
            "__dict__",
            {
                "chromosome": row.chromosome,
                "position": row.position,
                "ref": row.ref,
                "alt": row.alt,
                "z": z,
                "se": math.nan if row.se is None else row.se,
                "stored_effect_scale": stored_effect_scale,
                "eaf": eaf,
                "imputation_score": row.imputation_score,
                "reference": reference,
            },
        )
        yield association


def stream_variants(rows: Iterable[SourceRow]) -> Iterator[SourceVariant]:
    """Each row's variant, in the source's own labelling, with its rsid."""
    for row in rows:
        yield SourceVariant(
            chromosome=row.chromosome,
            position=row.position,
            ref=row.ref,
            alt=row.alt,
            rsid=row.rsid,
        )


def extract_at_sites(
    rows: Iterable[SourceRow], alids: Iterable[str]
) -> dict[str, SiteMetrics]:
    """Each requested site's A1 frequency and SE, from the rows that report both.

    A palindromic pair is left out, since nothing puts its frequency on a
    strand; a later row wins a repeated ALID.
    """
    wanted = alids if isinstance(alids, (set, frozenset, dict, KeysView)) else set(alids)
    if not wanted:
        return {}
    result: dict[str, SiteMetrics] = {}
    for row in rows:
        if row.alid not in wanted or row.af_alt is None or row.se is None:
            continue
        if is_palindromic(row.ref, row.alt):
            continue
        _z, af = swap_one(row.flipped, math.nan, row.af_alt)
        result[row.alid] = SiteMetrics(af=cast(float, af), se=row.se)
    return result
