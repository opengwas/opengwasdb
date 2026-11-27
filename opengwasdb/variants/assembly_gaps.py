"""Where a Reference Assembly has sequence: its contigs' lengths and assembly gaps (#410).

GRCh38 has no sequence in its assembly gaps, the runs of `N` that start every
chromosome (chr1 1-10,000), fill each centromere and cover the acrocentric
short arms, and none past a contig's end. A source row placed there names no
variant: positions like these were how OGS-00011 came to hold `1:5:G:T`, a
GWAS Catalog row lifted onto `chr4_GL000008v2_random:5` and written under its
original chromosome label. Admission drops such a row (`in_assembly_gap`,
`past_contig_end`), and `validate` fails a release that holds one
(`variant_table.assembly_gap`).

A gap is a run of two or more `N`. A single `N` is not one: UCSC's FASTA
writes each of GRCh38's IUPAC ambiguity codes as `N`, and real variants sit on
them (`17:83128871:G:T`, rs9747886, whose alleles are the `K` there).

The table is package data, `grch38-assembly-gaps.tsv` beside this module,
derived once from the FASTA by `derive` (`scripts/derive_grch38_assembly_gaps.py`
writes it, and its header names the FASTA and its checksum), so nothing that
reads it needs the 3 GB FASTA.
"""

from __future__ import annotations

import gzip
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

import numpy as np

from opengwasdb.store_reader.assembly import normalise_build
from opengwasdb.store_reader.identity import normalise_chromosome

__all__ = [
    "GRCH38_TABLE", "GRCH38_TABLE_SHA256", "AssemblySequence", "derive", "grch38",
    "sequence_of", "site_arrays",
]

#: The committed GRCh38 table: a contig row per primary contig, a gap row per run.
GRCH38_TABLE: Final = Path(__file__).with_name("grch38-assembly-gaps.tsv")
#: The committed table's sha256, as `scripts/derive_grch38_assembly_gaps.py` printed it
#: (786 gaps, 25 contigs): a table that lost a row is refused, never read as fewer gaps.
GRCH38_TABLE_SHA256: Final = "ebff5d6c0987cc3b9bad3f8d08ac7411d7fec5d08e063ff92ef06c455194380c"
#: The shortest run of `N` that is a gap; a single `N` is an ambiguity code.
MIN_GAP: Final = 2
#: The primary contigs, as UCSC's FASTA names them.
_PRIMARY: Final = frozenset({f"chr{n}" for n in (*range(1, 23), "X", "Y", "M")})


@dataclass(frozen=True)
class AssemblySequence:
    """One assembly's primary contigs: each one's length and its gaps.

    ``gaps`` maps a contig to its runs' first and last positions, 1-based and
    inclusive, each array sorted. A contig the table does not hold (a GRCh38
    source's alt contig, which spec §4 keeps as a chromosome) is not judged.
    """

    assembly: str
    lengths: Mapping[str, int]
    gaps: Mapping[str, tuple[np.ndarray, np.ndarray]]

    def in_gap(self, chromosomes: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Per row: whether its position is inside one of its contig's gaps."""
        found = np.zeros(len(positions), dtype=bool)
        for chromosome, rows in _by_chromosome(chromosomes):
            runs = self.gaps.get(chromosome)
            if runs is None:
                continue
            starts, ends = runs
            at = positions[rows]
            run = np.searchsorted(starts, at, side="right") - 1
            found[rows] = (run >= 0) & (at <= ends[np.maximum(run, 0)])
        return found

    def outside(self, chromosomes: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Per row: whether the assembly has no sequence at it, in a gap or past the end."""
        found: np.ndarray = self.in_gap(chromosomes, positions) | self.past_end(
            chromosomes, positions
        )
        return found

    def outside_at(self, chromosomes: Iterable[str], positions: Iterable[int]) -> np.ndarray:
        """`outside` of sites given one by one (`site_arrays`)."""
        return self.outside(*site_arrays(chromosomes, positions))

    def past_end(self, chromosomes: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Per row: whether its position is beyond its contig's last base."""
        found = np.zeros(len(positions), dtype=bool)
        for chromosome, rows in _by_chromosome(chromosomes):
            length = self.lengths.get(chromosome)
            if length is not None:
                found[rows] = positions[rows] > length
        return found


def site_arrays(
    chromosomes: Iterable[str], positions: Iterable[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Sites given one by one as the arrays `AssemblySequence` and Admission place rows by:
    the chromosomes as objects, the positions as int64."""
    return (
        np.array(list(chromosomes), dtype=object),
        np.fromiter(positions, dtype=np.int64),
    )


def _by_chromosome(chromosomes: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Each distinct chromosome and the rows naming it."""
    labels, inverse = np.unique(np.asarray(chromosomes, dtype=object).astype(str),
                                return_inverse=True)
    return [(str(label), np.flatnonzero(inverse == i)) for i, label in enumerate(labels)]


def _read(path: Path, assembly: str) -> AssemblySequence:
    """A table `derive`'s rows were written to: ``contig`` and ``gap`` rows."""
    lengths: dict[str, int] = {}
    runs: dict[str, list[tuple[int, int]]] = {}
    for kind, contig, start, end in _rows(path):
        if kind == "contig":
            lengths[contig] = int(end)
        elif kind == "gap":
            runs.setdefault(contig, []).append((int(start), int(end)))
        else:
            raise ValueError(f"{path}: a row of kind {kind!r}, neither contig nor gap")
    gaps = {
        contig: (np.array([s for s, _ in spans]), np.array([e for _, e in spans]))
        for contig, spans in runs.items()
    }
    return AssemblySequence(assembly, lengths, gaps)


def _rows(path: Path) -> list[list[str]]:
    """The table's rows under its header, its comment lines skipped."""
    with path.open(encoding="utf-8") as fh:
        header, *body = (line.rstrip("\n").split("\t") for line in fh if not line.startswith("#"))
    if header != ["kind", "contig", "start", "end"]:
        raise ValueError(f"{path}: header {header} is not kind, contig, start, end")
    return body


def grch38() -> AssemblySequence:
    """GRCh38's primary contigs and gaps, read once from the committed table, which must
    be the one its checksum records."""
    return _grch38_from(GRCH38_TABLE, GRCH38_TABLE_SHA256)


@cache
def _grch38_from(path: Path, sha256: str) -> AssemblySequence:
    found = hashlib.sha256(path.read_bytes()).hexdigest()
    if found != sha256:
        raise ValueError(
            f"{path}: sha256 {found}, but the GRCh38 gap table recorded is {sha256}: a table "
            "that lost or changed a row would place rows wrongly, so it is not read; "
            "re-derive it with scripts/derive_grch38_assembly_gaps.py and record its sha256"
        )
    return _read(path, "hg38")


def sequence_of(assembly: str) -> AssemblySequence | None:
    """The table of ``assembly`` (any spelling `normalise_build` reads), or None: only
    GRCh38, the assembly every Store is on, has one."""
    return grch38() if normalise_build(assembly) == "hg38" else None


def derive(fasta: Path) -> tuple[dict[str, int], list[tuple[str, int, int]]]:
    """A gzipped FASTA's primary contigs' lengths and their runs of `MIN_GAP` or more
    `N` (either case), 1-based and inclusive, in the FASTA's contig order.

    Contigs are named as a Store names them (`normalise_chromosome`: `chr1` is
    `1`, `chrM` is `MT`); every other contig is skipped.
    """
    lengths: dict[str, int] = {}
    runs: list[tuple[str, int, int]] = []

    def close(name: str | None, chunks: list[bytes]) -> None:
        if name is None or name not in _PRIMARY:
            return
        contig = normalise_chromosome(name)
        bases = np.frombuffer(b"".join(chunks), dtype=np.uint8)
        is_n = (bases == ord("N")) | (bases == ord("n"))
        edges = np.diff(np.concatenate(([0], is_n.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(edges == 1) + 1, np.flatnonzero(edges == -1)
        lengths[contig] = len(bases)
        runs.extend(
            (contig, int(s), int(e)) for s, e in zip(starts, ends, strict=True)
            if e - s + 1 >= MIN_GAP
        )

    name: str | None = None
    chunks: list[bytes] = []
    with gzip.open(fasta, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                close(name, chunks)
                name, chunks = line[1:].split()[0].decode("ascii"), []
            elif name in _PRIMARY:
                chunks.append(line.rstrip())
    close(name, chunks)
    return lengths, runs
