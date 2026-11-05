"""The BESD Source Reader (`opengwasdb.besd`; issues #282, #56; ADR 0062).

A BESD source is one file set -- `<prefix>.esi`, `<prefix>.epi`,
`<prefix>.besd` -- holding many Analyses: one per `.epi` probe. Everything per
variant lives in the `.esi` and is shared by every probe: its alleles, its SNP
ID, and the frequency of its effect allele `a1` (eQTLGen's `.esi` carries one
on every row). Everything per association lives in the `.besd`: each probe's
beta and SE, with no per-variant sample size.

So the `.esi` is read once per file set into a :class:`BesdSource` -- each
row's identity, orientation and frequency, as arrays -- and shared by every
probe's :class:`BesdProbeReader`, which joins its own `.besd` slice onto them.
A probe never re-reads or re-orients the `.esi`. A row is oriented in the
file set's declared assembly, or, for a reader opened lifted, on the reference
build by the GWAS-VCF rule (`oriented_variant`).
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np

from opengwasdb.layouts.ragged.besd_reader import iter_esi, read_besd_probe, read_epi
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers import interface
from opengwasdb.readers.gwas_vcf import oriented_variant
from opengwasdb.readers.orientation import a1_frequencies, is_palindromic, swap
from opengwasdb.readers.rows import MetricsChunk, ReaderBatch
from opengwasdb.stats import parse_af
from opengwasdb.variants.normalise import (
    MalformedIdentity,
    allele_fault,
    identity_fault,
    normalise_chromosome,
    oriented_alid,
)

__all__ = [
    "BESD_CAPABILITY",
    "BESD_SUFFIXES",
    "BesdProbeReader",
    "BesdSource",
    "besd_member_files",
    "load_besd_source",
]

#: A manifest row's `source_reader_capability` for a BESD probe. Its
#: `source_file` is the file set's prefix, as `build-ragged-besd` takes it.
BESD_CAPABILITY = "opengwasdb.besd"
BESD_SUFFIXES = (".esi", ".epi", ".besd")
_UNDECLARED_SCORE = interface.ImputationScoreStatus.UNDECLARED


def besd_member_files(prefix: str | Path) -> tuple[Path, Path, Path]:
    """The `.esi`, `.epi` and `.besd` files beside `prefix` (appended, not replaced)."""
    esi, epi, besd = (Path(f"{prefix}{suffix}") for suffix in BESD_SUFFIXES)
    return esi, epi, besd


@dataclass(frozen=True)
class BesdSource:
    """One BESD file set's per-variant facts, indexed by `.esi` row.

    `alid` is `""` on a row that names no canonical variant: a malformed
    identity (`malformed` names it and why, #365), alleles that do not orient,
    or a lift that did not land. No association at such a row is ever read.
    `af` is the reported frequency of `a1`, the effect allele, or `NaN` when the
    row reports none or an unusable one (`opengwasdb.stats.parse_af`'s rule) --
    never a substitute -- and `made_absent` marks the rows that reported one
    the rule makes absent (D7). `chromosome`, `position`, `a1`, `a2` and
    `snp_id` are the row as the `.esi` gives it, its chromosome normalised.
    """

    prefix: Path
    probes: Mapping[str, int]
    alid: np.ndarray
    flipped: np.ndarray
    palindromic: np.ndarray
    af: np.ndarray
    made_absent: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    malformed: Mapping[int, str] = field(default_factory=dict)
    chromosome: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    position: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    a1: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    a2: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    snp_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))

    @property
    def n_rows(self) -> int:
        return int(self.alid.shape[0])

    @property
    def n_frequencies(self) -> int:
        """Rows with a canonical variant and a usable frequency."""
        return int(np.count_nonzero(self.has_canonical_variant & np.isfinite(self.af)))

    @cached_property
    def has_canonical_variant(self) -> np.ndarray:
        """Each row's having a canonical variant, computed once per file set:
        every probe's read indexes it, and comparing nine million ALIDs per
        probe cost OGS-00027's build longer than reading every probe."""
        present: np.ndarray = self.alid != ""
        return present

    def a1_frequencies(self) -> np.ndarray:
        """Each row's frequency of its canonical A1, the orientation ancestry reads."""
        return a1_frequencies(self.flipped, self.af)

    def probe_rows(self, probe_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One probe's ``(.esi rows, betas, ses)``, read from its `.besd` slice."""
        probe_idx = self.probes.get(probe_id)
        if probe_idx is None:
            raise ValueError(f"probe {probe_id!r} is not in {self.prefix}.epi")
        _esi, _epi, besd = besd_member_files(self.prefix)
        rows, betas, ses = read_besd_probe(besd, len(self.probes), probe_idx)
        if rows.size and int(rows.max()) >= self.n_rows:
            raise ValueError(
                f"{besd}: probe {probe_id!r} names .esi row {int(rows.max())}, "
                f"past the {self.n_rows} rows of {self.prefix}.esi"
            )
        return rows.astype(np.int64), betas, ses

    def read_probe_metrics(self, probe_id: str) -> MetricsChunk:
        """Read one probe's slice and join it to this file set's variant facts."""
        return self.probe_metrics(probe_id)[1]

    def probe_metrics(self, probe_id: str) -> tuple[np.ndarray, MetricsChunk]:
        """One probe's metrics block, and the `.esi` row each of its rows is at."""
        rows, betas, ses = self.probe_rows(probe_id)
        keep = self.has_canonical_variant[rows]
        rows = rows[keep]
        beta = betas[keep].astype(np.float64)
        se = ses[keep].astype(np.float64)
        return rows, MetricsChunk(
            alid=self.alid[rows],
            flipped=self.flipped[rows],
            palindromic=self.palindromic[rows],
            af_alt=self.af[rows],
            beta=np.where(np.isfinite(beta), beta, np.nan),
            se=np.where(np.isfinite(se) & (se > 0), se, np.nan),
        )


def load_besd_source(prefix: str | Path, *, liftover: object | None = None) -> BesdSource:
    """Read one BESD file set's `.epi` probes and `.esi` variants, once.

    `liftover`, when set, lifts each `.esi` row to the reference build before
    orientation by the GWAS-VCF rule (`oriented_variant`): one unique
    mapping, alleles complemented on a reverse-strand one, and an indel or an
    ambiguous mapping refused rather than placed by position alone.

    A file set is read once per process for each lift, and shared by every
    probe's reader; one rewritten since is read again.
    """
    esi, epi, _besd = besd_member_files(prefix)
    stamps = tuple((member.stat().st_size, member.stat().st_mtime_ns) for member in (esi, epi))
    return _load(Path(prefix).resolve(), liftover, stamps)


@lru_cache(maxsize=4)
def _load(
    prefix: Path, liftover: object | None, _stamps: tuple[tuple[int, int], ...]
) -> BesdSource:
    esi, epi, _besd = besd_member_files(prefix)
    probes = _probe_rows(epi)
    columns = _EsiColumns()
    for variant in iter_esi(esi):
        columns.add(variant.row_idx, variant.chromosome, variant.bp, variant.a1 or "",
                    variant.a2 or "", variant.snp_id, variant.freq, liftover)
    return columns.source(prefix, probes)


@dataclass
class _EsiColumns:
    """A `.esi`'s rows as they are read, before they become `BesdSource`'s arrays."""

    alid: list[str] = field(default_factory=list)
    flipped: bytearray = field(default_factory=bytearray)
    palindromic: bytearray = field(default_factory=bytearray)
    af: array[float] = field(default_factory=lambda: array("d"))
    made_absent: bytearray = field(default_factory=bytearray)
    malformed: dict[int, str] = field(default_factory=dict)
    chromosome: list[str] = field(default_factory=list)
    position: array[int] = field(default_factory=lambda: array("q"))
    a1: list[str] = field(default_factory=list)
    a2: list[str] = field(default_factory=list)
    snp_id: list[str] = field(default_factory=list)
    _chromosomes: dict[str, MalformedIdentity | str] = field(default_factory=dict)

    def add(self, row: int, chrom: str, bp: int, a1: str, a2: str, snp_id: str,
            freq: float | None, liftover: object | None) -> None:
        chromosome = self._chromosome(chrom)
        fault = chromosome if isinstance(chromosome, MalformedIdentity) else allele_fault(a1, a2)
        if fault is not None:
            self.malformed[row] = fault.reason
            alid, flipped = "", False
        else:
            alid, flipped = _oriented(liftover, str(chromosome), chrom, bp, a1, a2)
        self.alid.append(alid)
        self.flipped.append(flipped)
        self.palindromic.append(is_palindromic(a1, a2))
        af = parse_af(freq)
        self.af.append(math.nan if af is None else af)
        self.made_absent.append(af is None and freq is not None)
        self.chromosome.append("" if fault is not None else str(chromosome))
        self.position.append(bp)
        self.a1.append(a1)
        self.a2.append(a2)
        self.snp_id.append(snp_id)

    def _chromosome(self, chrom: str) -> MalformedIdentity | str:
        known = self._chromosomes.get(chrom)
        if known is None:
            fault = identity_fault(chrom)
            known = fault if fault is not None else normalise_chromosome(chrom)
            self._chromosomes[chrom] = known
        return known

    def source(self, prefix: Path, probes: Mapping[str, int]) -> BesdSource:
        return BesdSource(
            prefix=prefix,
            probes=probes,
            alid=np.asarray(self.alid, dtype=object),
            flipped=np.frombuffer(bytes(self.flipped), dtype=np.bool_),
            palindromic=np.frombuffer(bytes(self.palindromic), dtype=np.bool_),
            af=np.frombuffer(self.af, dtype=np.float64),
            made_absent=np.frombuffer(bytes(self.made_absent), dtype=np.bool_),
            malformed=self.malformed,
            chromosome=np.asarray(self.chromosome, dtype=object),
            position=np.frombuffer(self.position, dtype=np.int64),
            a1=np.asarray(self.a1, dtype=object),
            a2=np.asarray(self.a2, dtype=object),
            snp_id=np.asarray(self.snp_id, dtype=object),
        )


def _oriented(
    liftover: object | None, chromosome: str, chrom: str, bp: int, a1: str, a2: str
) -> tuple[str, bool]:
    """One well-formed row's ``(alid, flipped)``, ``("", False)`` where it names none.

    `a1` is the `.esi`'s effect allele. Unlifted, the row is oriented where it
    is, by `oriented_alid`; lifted, by `oriented_variant` on the reference build.
    """
    if liftover is not None:
        orientation = oriented_variant(liftover, chrom, str(bp), a1, a2)
        return ("", False) if orientation is None else (
            orientation.variant.alid, orientation.flipped
        )
    oriented = oriented_alid(chromosome, bp, a1, a2) if bp > 0 else None
    return ("", False) if oriented is None else oriented


def _probe_rows(epi: Path) -> dict[str, int]:
    """`{probe_id: .epi row}`; a repeated probe ID would make an Analysis ambiguous."""
    probes: dict[str, int] = {}
    for probe in read_epi(epi):
        if probe.probe_id in probes:
            raise ValueError(f"{epi}: probe {probe.probe_id!r} appears more than once")
        probes[probe.probe_id] = probe.row_idx
    return probes


#: A BESD names an effect allele `a1` and another `a2`, and declares neither
#: the genome's reference: every projection of every probe carries this.
_REFERENCE = interface.ReferenceAllele.UNDECLARED


@dataclass(frozen=True)
class BesdProbeReader:
    """The Source Reader for one probe of a BESD file set: one Analysis.

    Every projection reads the probe's own `.besd` slice and joins it to the
    file set's shared `BesdSource`. A row at a `.esi` row whose identity is
    malformed is left out of every projection, and named in a stream's
    ``tally`` (#365); one whose variant does not orient is left out. An
    association's `alt` is the `.esi`'s `a1` and its `ref` `a2`, as a
    GWAS-VCF's ALT is its effect allele; neither is declared the reference.

    The variants, associations and batches are the file set's own, oriented in
    its declared assembly (`source`, read once per process from ``prefix``).
    ``aligned``, given, is the same file set lifted onto the reference build:
    the metric blocks and sites are read from it, as every reader opened with
    a lift presents its annotation projections. Each projection's z and
    frequency are therefore on the A1 of the alleles it reports.

    The metrics projection follows the tabular usability rule -- a finite beta;
    a finite, positive SE -- and is `NaN` otherwise; the association
    projections keep the SE the `.besd` reports, for Association Admission to
    judge. A probe absent from the `.epi`, or an association pointing past the
    `.esi`, raises: either is a mismatched or corrupt file set.

    `stored_effect_scale` is the probe's declared scale, or `None` where it
    declares none: the batches carry no scale, and `stream_associations`
    refuses to stream without one rather than assume it.
    """

    prefix: str | Path
    probe_id: str
    stored_effect_scale: StoredEffectScale | None = None
    aligned: BesdSource | None = None

    def __post_init__(self) -> None:
        if self.aligned is not None and self.aligned.prefix != Path(self.prefix).resolve():
            raise ValueError(
                f"BESD probe {self.probe_id!r}: the aligned file set {self.aligned.prefix} "
                f"is not {self.prefix}"
            )

    @cached_property
    def source(self) -> BesdSource:
        """The file set in its declared assembly (`load_besd_source`, read once)."""
        return load_besd_source(self.prefix)

    @property
    def annotated(self) -> BesdSource:
        """The file set the metric blocks and sites are read from."""
        return self.source if self.aligned is None else self.aligned

    def stream_metric_chunks(self) -> Iterator[MetricsChunk]:
        """The probe's associations as one block: a cis probe has a few thousand."""
        yield self.annotated.read_probe_metrics(self.probe_id)

    def stream_batches(
        self, tally: interface.StreamTally | None = None
    ) -> Iterator[ReaderBatch]:
        """The probe's associations as one batch, on A1, in `.besd` order.

        ``tally`` names each row at a malformed `.esi` row and counts the rows
        read at a `.esi` frequency made absent (D7). Each row's
        ``source_row`` is its `.esi` row, which also names it in Admission's
        log (``esi:<row>``).
        """
        source = self.source
        rows, betas, ses = source.probe_rows(self.probe_id)
        rows, betas, ses = self._well_formed(rows, betas, ses, tally or interface.StreamTally())
        keep = source.has_canonical_variant[rows]
        rows, beta, se = rows[keep], betas[keep].astype(np.float64), ses[keep].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where((se > 0) & np.isfinite(se), beta / se, np.nan)
        flipped = source.flipped[rows]
        z, eaf = swap(flipped, z, source.af[rows])
        snp_ids = source.snp_id[rows].tolist()
        yield ReaderBatch(
            alid=source.alid[rows],
            source_row=rows,
            flipped=flipped,
            z=z,
            se=se,
            eaf=eaf,
            imputation_score=np.full(len(rows), np.nan),
            imputation_status=np.full(len(rows), _UNDECLARED_SCORE, dtype=object),
            rsid=np.array([s if s.startswith("rs") else "" for s in snp_ids], dtype=object),
            ref=source.a2[rows],
            alt=source.a1[rows],
            reference=_REFERENCE,
            key_prefix="esi:",
        )

    def _well_formed(
        self, rows: np.ndarray, betas: np.ndarray, ses: np.ndarray, tally: interface.StreamTally
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The probe's rows less those at a malformed `.esi` row, which ``tally``
        names; and the frequencies made absent at the rest, counted."""
        malformed = self.source.malformed
        if malformed:
            at = [i for i, row in enumerate(rows.tolist()) if row in malformed]
            tally.faults.rows.extend((f"esi:{int(rows[i])}", malformed[int(rows[i])]) for i in at)
            if at:
                keep = np.ones(len(rows), dtype=bool)
                keep[at] = False
                rows, betas, ses = rows[keep], betas[keep], ses[keep]
        tally.frequencies.made_absent += int(self.source.made_absent[rows].sum())
        return rows, betas, ses

    def stream_associations(
        self, tally: interface.StreamTally | None = None
    ) -> Iterator[interface.ReaderAssociation]:
        if self.stored_effect_scale is None:
            raise ValueError(
                f"BESD probe {self.probe_id!r} declares no stored_effect_scale; its "
                "associations are not streamed under an assumed one"
            )
        source = self.source
        for batch in self.stream_batches(tally):
            rows = batch.source_row
            for row, z, se, eaf in zip(
                rows.tolist(), batch.z.tolist(), batch.se.tolist(), batch.eaf.tolist(),
                strict=True,
            ):
                yield interface.ReaderAssociation(
                    chromosome=source.chromosome[row],
                    position=int(source.position[row]),
                    ref=source.a2[row],
                    alt=source.a1[row],
                    z=z,
                    se=se,
                    stored_effect_scale=self.stored_effect_scale,
                    eaf=None if math.isnan(eaf) else eaf,
                    reference=batch.reference,
                )

    def stream_variants(self) -> Iterator[interface.SourceVariant]:
        """Every `.esi` row naming a variant, in `.esi` order: the file set's, not the probe's."""
        source = self.source
        for row in np.flatnonzero(source.has_canonical_variant).tolist():
            snp_id = source.snp_id[row]
            yield interface.SourceVariant(
                chromosome=source.chromosome[row],
                position=int(source.position[row]),
                ref=source.a2[row],
                alt=source.a1[row],
                rsid=snp_id if snp_id.startswith("rs") else "",
            )

    def extract_at_sites(
        self,
        alids: Iterable[str],
        *,
        region: str | None = None,
        regions_file: interface.RegionsFile = None,
    ) -> dict[str, interface.SiteMetrics]:
        """The probe's A1 frequency and SE at each requested site it reports.

        A `.besd` cannot be read by position, so ``regions_file`` reads nothing less.
        """
        wanted = set(alids)
        rows, chunk = self.annotated.probe_metrics(self.probe_id)
        usable = np.isfinite(chunk.se) & np.isfinite(chunk.af_alt) & ~chunk.palindromic
        if region is not None:
            # The chromosome the `.esi` names the row on, as for every reader.
            usable &= self.source.chromosome[rows] == normalise_chromosome(region)
        return {
            alid: interface.SiteMetrics(af=af, se=se)
            for alid, af, se in zip(
                chunk.alid[usable].tolist(), chunk.a1_frequency()[usable].tolist(),
                chunk.se[usable].tolist(), strict=True,
            )
            if alid in wanted
        }
