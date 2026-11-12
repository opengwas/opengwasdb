"""In-memory SourceReader fake (issue #19; extended for #20 and #56).

Lets annotator/consumer tests exercise the `SourceReader` interface with no
bcftools subprocess or fixture VCF file required.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator

import numpy as np
from numpy.typing import DTypeLike

from opengwasdb.readers.interface import (
    ReaderAssociation,
    RegionsFile,
    SiteMetrics,
    SourceVariant,
    StreamTally,
)
from opengwasdb.readers.rows import MetricsChunk, ReaderBatch
from opengwasdb.variants.normalise import canonical_alid, identity_allele, normalise_chromosome


class FakeReader:
    """A SourceReader backed by in-memory data supplied at construction.

    `variants` defaults to a `SourceVariant` (rsid blank) per each
    supplied association when omitted -- the common case where nothing needs
    to exercise `stream_variants`'s superset relationship to
    `stream_associations` (issue #20). Pass it explicitly to simulate a
    variant present in the source but dropped from the association stream
    (e.g. invalid SE). `metric_chunks` is what `stream_metric_chunks` yields.

    The associations are already on A1, as a reader yields them, so their one
    batch carries them as they are.
    """

    def __init__(
        self,
        associations: Iterable[ReaderAssociation] = (),
        sites: dict[str, SiteMetrics] | None = None,
        variants: Iterable[SourceVariant] | None = None,
        metric_chunks: Iterable[MetricsChunk] = (),
    ) -> None:
        self._associations = list(associations)
        self._sites = dict(sites) if sites is not None else {}
        self._variants = (
            list(variants)
            if variants is not None
            else [
                SourceVariant(a.chromosome, a.position, a.ref, a.alt)
                for a in self._associations
            ]
        )
        self._metric_chunks = list(metric_chunks)

    def stream_associations(self, tally: StreamTally | None = None) -> Iterator[ReaderAssociation]:
        yield from self._associations

    def stream_batches(self, tally: StreamTally | None = None) -> Iterator[ReaderBatch]:
        if self._associations:
            yield _batch(self._associations)

    def stream_variants(self) -> Iterator[SourceVariant]:
        yield from self._variants

    def stream_metric_chunks(self) -> Iterator[MetricsChunk]:
        yield from self._metric_chunks

    def extract_at_sites(
        self, alids: Iterable[str], *, region: str | None = None, regions_file: RegionsFile = None
    ) -> dict[str, SiteMetrics]:
        wanted = set(alids)
        chromosome = None if region is None else normalise_chromosome(region)
        return {
            alid: metrics
            for alid, metrics in self._sites.items()
            if alid in wanted and (chromosome is None or alid.split(":", 1)[0] == chromosome)
        }


def _batch(rows: list[ReaderAssociation]) -> ReaderBatch:
    """Associations already on A1, as the one batch of their columns."""

    def column(value: Callable[[ReaderAssociation], object], dtype: DTypeLike) -> np.ndarray:
        return np.array([value(a) for a in rows], dtype=dtype)

    return ReaderBatch(
        alid=column(lambda a: canonical_alid(a.chromosome, a.position, a.ref, a.alt), object),
        source_row=np.arange(len(rows), dtype=np.int64),
        flipped=column(lambda a: identity_allele(a.alt) > identity_allele(a.ref), bool),
        z=column(lambda a: a.z, np.float64),
        se=column(lambda a: a.se, np.float64),
        eaf=column(lambda a: math.nan if a.eaf is None else a.eaf, np.float64),
        imputation_score=column(
            lambda a: math.nan if a.imputation_score.value is None else a.imputation_score.value,
            np.float64,
        ),
        imputation_status=column(lambda a: a.imputation_score.status, object),
        rsid=np.full(len(rows), "", dtype=object),
        ref=column(lambda a: a.ref, object),
        alt=column(lambda a: a.alt, object),
    )
