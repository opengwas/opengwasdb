"""BESD sources for the one-pass Analysis resolver (issue #282, ADR 0062).

A BESD source is one file set -- `<prefix>.esi`, `<prefix>.epi`,
`<prefix>.besd` -- holding many Analyses: one per `.epi` probe. Everything per
variant lives in the `.esi` and is shared by every probe: its alleles, and the
frequency of its effect allele `a1` (eQTLGen's `.esi` carries one on every
row). Everything per association lives in the `.besd`: each probe's beta and
SE, with no per-variant sample size.

So the `.esi` is read once into a :class:`BesdSource` -- canonical identity,
orientation and frequency per row, as arrays -- and each probe's
:class:`BesdProbeReader` joins its own `.besd` slice onto those arrays. A probe
never re-reads or re-orients the `.esi`.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opengwasdb.layouts.ragged.besd_reader import iter_esi, read_besd_probe, read_epi
from opengwasdb.readers.gwas_vcf import is_palindromic, oriented_variant
from opengwasdb.readers.tabular import MetricsChunk
from opengwasdb.stats import parse_af

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


def besd_member_files(prefix: str | Path) -> tuple[Path, Path, Path]:
    """The `.esi`, `.epi` and `.besd` files beside `prefix` (appended, not replaced)."""
    esi, epi, besd = (Path(f"{prefix}{suffix}") for suffix in BESD_SUFFIXES)
    return esi, epi, besd


@dataclass(frozen=True)
class BesdSource:
    """One BESD file set's per-variant facts, indexed by `.esi` row.

    `alid` is `""` on a row that names no usable canonical variant (alleles
    that do not normalise, or a lift that did not land); no association at
    such a row is ever read. `af` is the reported frequency of `a1`, the
    effect allele, or `NaN` when the row reports none or an unusable one
    (`opengwasdb.stats.parse_af`'s rule) -- never a substitute.
    """

    prefix: Path
    probes: Mapping[str, int]
    alid: np.ndarray
    flipped: np.ndarray
    palindromic: np.ndarray
    af: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.alid.shape[0])

    @property
    def n_frequencies(self) -> int:
        """Rows with a usable variant and a usable frequency."""
        return int(np.count_nonzero(self.usable & np.isfinite(self.af)))

    @property
    def usable(self) -> np.ndarray:
        usable: np.ndarray = self.alid != ""
        return usable

    def a1_frequencies(self) -> np.ndarray:
        """Each row's frequency of its canonical A1, the orientation ancestry reads."""
        oriented: np.ndarray = np.where(self.flipped, 1.0 - self.af, self.af)
        return oriented


def load_besd_source(prefix: str | Path, *, liftover: object | None = None) -> BesdSource:
    """Read one BESD file set's `.epi` probes and `.esi` variants, once.

    `liftover`, when set, lifts each `.esi` row to the reference build before
    orientation by the GWAS-VCF rule (`oriented_variant`): one unique
    mapping, alleles complemented on a reverse-strand one, and an indel or an
    ambiguous mapping refused rather than placed by position alone.
    """
    esi, epi, _besd = besd_member_files(prefix)
    probes = _probe_rows(epi)
    alids: list[str] = []
    flipped, palindromic = bytearray(), bytearray()
    frequencies = array("d")
    for snp in iter_esi(esi):
        a1, a2 = snp.a1 or "", snp.a2 or ""
        orientation = oriented_variant(liftover, snp.chromosome, str(snp.bp), a1, a2)
        alids.append("" if orientation is None else orientation.variant.alid)
        flipped.append(orientation is not None and orientation.flipped)
        palindromic.append(is_palindromic(a1, a2))
        af = parse_af(snp.freq)
        frequencies.append(math.nan if af is None else af)
    return BesdSource(
        prefix=Path(prefix),
        probes=probes,
        alid=np.asarray(alids, dtype=object),
        flipped=np.frombuffer(bytes(flipped), dtype=np.bool_),
        palindromic=np.frombuffer(bytes(palindromic), dtype=np.bool_),
        af=np.frombuffer(frequencies, dtype=np.float64),
    )


def _probe_rows(epi: Path) -> dict[str, int]:
    """`{probe_id: .epi row}`; a repeated probe ID would make an Analysis ambiguous."""
    probes: dict[str, int] = {}
    for probe in read_epi(epi):
        if probe.probe_id in probes:
            raise ValueError(f"{epi}: probe {probe.probe_id!r} appears more than once")
        probes[probe.probe_id] = probe.row_idx
    return probes


@dataclass(frozen=True)
class BesdProbeReader:
    """One probe's associations as the resolver's metrics projection (ADR 0044).

    Yields a single block: a cis probe's associations are a few thousand rows.
    A row whose `.esi` variant is unusable is left out, as every reader leaves
    out a row naming no canonical variant. `beta` and `se` follow the tabular
    usability rule -- a finite beta; a finite, positive SE -- and are `NaN`
    otherwise. A probe absent from the `.epi`, or an association pointing past
    the `.esi`, raises: either is a mismatched or corrupt file set.
    """

    source: BesdSource
    probe_id: str

    def stream_metric_chunks(self) -> Iterator[MetricsChunk]:
        probe_idx = self.source.probes.get(self.probe_id)
        if probe_idx is None:
            raise ValueError(f"probe {self.probe_id!r} is not in {self.source.prefix}.epi")
        _esi, _epi, besd = besd_member_files(self.source.prefix)
        rows, betas, ses = read_besd_probe(besd, len(self.source.probes), probe_idx)
        if rows.size and int(rows.max()) >= self.source.n_rows:
            raise ValueError(
                f"{besd}: probe {self.probe_id!r} names .esi row {int(rows.max())}, "
                f"past the {self.source.n_rows} rows of {self.source.prefix}.esi"
            )
        keep = self.source.usable[rows]
        rows = rows[keep]
        beta = betas[keep].astype(np.float64)
        se = ses[keep].astype(np.float64)
        yield MetricsChunk(
            alid=self.source.alid[rows],
            flipped=self.source.flipped[rows],
            palindromic=self.source.palindromic[rows],
            af_alt=self.source.af[rows],
            beta=np.where(np.isfinite(beta), beta, np.nan),
            se=np.where(np.isfinite(se) & (se > 0), se, np.nan),
        )
