"""GRCh37 to GRCh38 with explicit allele and measurement losses (ADR 0015)."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import numpy.typing as npt

from ogld.build.importing.matrix import IntArray
from ogld.build.importing.parts import optional_module
from ogld.identity import AUTOSOMES, IdentityError, order_key, orient
from ogld.store import Variants


@dataclass(frozen=True)
class SourceVariant:
    index: int
    chrom: str
    position: int
    ref: str
    alt: str
    af: float


def text_input(path: Path) -> TextIO:
    """The export and FASTA may be plain text or gzip/BGZF."""
    return gzip.open(path, "rt") if path.suffix in (".gz", ".bgz") else path.open()


def variant_chromosomes(path: Path) -> Iterator[tuple[str, list[SourceVariant]]]:
    """Stream Hail's chromosome-contiguous export without retaining genome-wide row objects."""
    seen: set[str] = set()
    for name, rows in groupby(_variant_rows(path), key=lambda row: row.chrom):
        if name in seen:
            raise ValueError(f"{path}: chromosome {name} is not contiguous in the Hail export")
        seen.add(name)
        yield name, list(rows)


def _variant_rows(path: Path) -> Iterator[SourceVariant]:
    indices: set[int] = set()
    with text_input(path) as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            chrom, position = row["locus"].split(":")
            if chrom not in AUTOSOMES:
                raise ValueError(f"{path}: unsupported chromosome {chrom}")
            alleles = json.loads(row["alleles"])
            if len(alleles) != 2:
                raise ValueError(f"{path}: row {row['idx']} is not biallelic")
            index = int(row["idx"])
            if index < 0 or index in indices or int(position) < 1:
                raise ValueError(f"{path}: duplicate/invalid index or position at {row['idx']}")
            indices.add(index)
            yield SourceVariant(
                index, chrom, int(position), alleles[0], alleles[1], float(row["AF"])
            )


def _reference(path: Path, chrom: str) -> str:
    lines: list[str] = []
    selected = False
    found = False
    with text_input(path) as stream:
        for line in stream:
            if line.startswith(">"):
                selected = line[1:].split()[0].removeprefix("chr") == chrom
                found |= selected
            elif selected:
                lines.append(line.strip())
    if not found:
        raise ValueError(f"{path}: no FASTA sequence for chromosome {chrom}")
    return "".join(lines).upper()


@dataclass(frozen=True)
class Lifted:
    variants: Variants
    index: IntArray
    source_position: IntArray
    sign: IntArray
    negative_strand: npt.NDArray[np.bool_]

    def __len__(self) -> int:
        return len(self.index)

    @property
    def alid(self) -> npt.NDArray[np.str_]:
        return self.variants.alid

    def select(self, mask: npt.NDArray[np.bool_]) -> Lifted:
        v = self.variants
        return Lifted(
            Variants(
                v.chrom[mask],
                v.position[mask],
                v.alid[mask],
                v.a1[mask],
                v.a2[mask],
                v.a1_frequency[mask],
            ),
            self.index[mask],
            self.source_position[mask],
            self.sign[mask],
            self.negative_strand[mask],
        )


def _span(row: SourceVariant, a: Any, b: Any) -> tuple[int, str] | str:
    if a[0] != "chr" + row.chrom or b[0] != a[0] or b[2] != a[2]:
        return "liftover_chromosome_or_strand"
    if abs(b[1] - a[1]) != len(row.ref) - 1:
        return "liftover_allele_span"
    return min(a[1], b[1]), a[2]


def _mapped(row: SourceVariant, lift: Any) -> tuple[int, str] | str:
    matches = [
        lift.convert_coordinate("chr" + row.chrom, position) or []
        for position in (row.position - 1, row.position + len(row.ref) - 2)
    ]
    if any(len(match) != 1 for match in matches):
        return "liftover_missing_or_ambiguous"
    return _span(row, matches[0][0], matches[1][0])


@dataclass(frozen=True)
class LiftedVariant:
    chrom: str
    position: int
    alid: str
    a1: str
    a2: str
    a1_frequency: float
    source_row: int
    source_position: int
    sign: int
    negative_strand: bool


def _target_alleles(
    row: SourceVariant, strand: str, target: int, reference: str
) -> tuple[str, str] | str:
    ref, alt = row.ref, row.alt
    if strand == "-":
        ref, alt = (_reverse(allele) for allele in (ref, alt))
    if reference[target : target + len(ref)] != ref:
        return "allele_mismatch"
    return ref, alt


def _a1_frequency(af: float, a1_is_alt: bool) -> float | str:
    if not np.isfinite(af) or not 0 < af < 1:
        return "invalid_frequency"
    return af if a1_is_alt else 1 - af


def _target(row: SourceVariant, lift: Any, reference: str) -> LiftedVariant | str:
    mapped = _mapped(row, lift)
    if isinstance(mapped, str):
        return mapped
    target, strand = mapped
    alleles = _target_alleles(row, strand, target, reference)
    if isinstance(alleles, str):
        return alleles
    ref, alt = alleles
    try:
        identity = orient(row.chrom, target + 1, ref, alt)
    except IdentityError:
        return "allele_identity"
    frequency = _a1_frequency(row.af, identity.a1_is_alt)
    if isinstance(frequency, str):
        return frequency
    return LiftedVariant(
        chrom=row.chrom,
        position=target + 1,
        alid=identity.alid,
        a1=identity.a1,
        a2=identity.a2,
        a1_frequency=frequency,
        source_row=row.index,
        source_position=row.position,
        sign=1 if identity.a1_is_alt else -1,
        negative_strand=strand == "-",
    )


def _reverse(allele: str) -> str:
    return allele.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _targets(
    rows: list[SourceVariant], lift: Any, reference: str
) -> tuple[list[LiftedVariant], Counter[str]]:
    kept: list[LiftedVariant] = []
    losses: Counter[str] = Counter()
    for row in rows:
        got = _target(row, lift, reference)
        if isinstance(got, str):
            losses[got] += 1
        else:
            kept.append(got)
    return kept, losses


def _without_collisions(kept: list[LiftedVariant], losses: Counter[str]) -> list[LiftedVariant]:
    multiplicity = Counter(row.alid for row in kept)
    unique = [row for row in kept if multiplicity[row.alid] == 1]
    losses["liftover_identity_collision"] = len(kept) - len(unique)
    return unique


def lift_variants(
    rows: list[SourceVariant], chain: Path, fasta: Path
) -> tuple[Lifted, dict[str, int]]:
    """Drop every member of an identity collision, as in #89's measured prototype."""
    if not rows or len({row.chrom for row in rows}) != 1:
        raise ValueError("liftover needs one nonempty chromosome")
    lift = optional_module("pyliftover").LiftOver(str(chain))
    reference = _reference(fasta, rows[0].chrom)
    kept, losses = _targets(rows, lift, reference)
    unique = _without_collisions(kept, losses)
    unique.sort(key=lambda row: order_key(row.chrom, row.position, row.alid))
    return _table(unique), dict(losses)


def _variants(rows: list[LiftedVariant]) -> Variants:
    return Variants(
        np.array([row.chrom for row in rows], dtype=str),
        np.array([row.position for row in rows], dtype=np.int64),
        np.array([row.alid for row in rows], dtype=str),
        np.array([row.a1 for row in rows], dtype=str),
        np.array([row.a2 for row in rows], dtype=str),
        np.array([row.a1_frequency for row in rows], dtype=np.float64),
    )


def _table(rows: list[LiftedVariant]) -> Lifted:
    return Lifted(
        _variants(rows),
        np.array([row.source_row for row in rows], dtype=np.int64),
        np.array([row.source_position for row in rows], dtype=np.int64),
        np.array([row.sign for row in rows], dtype=np.int64),
        np.array([row.negative_strand for row in rows], dtype=bool),
    )


def _start_within(target: IntArray, i: int, distance: int, start: int) -> int:
    while start < i and target[start] < target[i] - distance:
        start += 1
    return start


def _add_extreme(queue: deque[int], source: IntArray, i: int, start: int, minimum: bool) -> None:
    while queue and queue[0] < start:
        queue.popleft()
    while queue and ((source[queue[-1]] >= source[i]) == minimum):
        queue.pop()
    queue.append(i)


def _complete(source: IntArray, target: IntArray, distance: int) -> bool:
    low: deque[int] = deque()
    high: deque[int] = deque()
    start = 0
    for i, _position in enumerate(target):
        start = _start_within(target, i, distance, start)
        _add_extreme(low, source, i, start, True)
        _add_extreme(high, source, i, start, False)
        if source[high[0]] - source[low[0]] > 10_000_000:
            return False
    return True


def far_distance(source: IntArray, target: IntArray) -> int:
    """Largest target radius containing no pair outside the published 10 Mb source band."""
    lo, hi = 0, 10_000_000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _complete(source, target, mid):
            lo = mid
        else:
            hi = mid - 1
    return lo
