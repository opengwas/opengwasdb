"""GRCh37 to GRCh38 with explicit allele and measurement losses (ADR 0015)."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, deque
from dataclasses import dataclass
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


def read_variants(path: Path) -> dict[str, list[SourceVariant]]:
    """Read the one-time Hail export, preserving its global matrix row indices."""
    result: dict[str, list[SourceVariant]] = {}
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
            result.setdefault(chrom, []).append(
                SourceVariant(index, chrom, int(position), alleles[0], alleles[1], float(row["AF"]))
            )
    return result


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


def _mapped(row: SourceVariant, lift: Any) -> tuple[int, str] | str:
    matches = [
        lift.convert_coordinate("chr" + row.chrom, position) or []
        for position in (row.position - 1, row.position + len(row.ref) - 2)
    ]
    if any(len(match) != 1 for match in matches):
        return "liftover_missing_or_ambiguous"
    a, b = matches[0][0], matches[1][0]
    if a[0] != "chr" + row.chrom or b[0] != a[0] or b[2] != a[2]:
        return "liftover_chromosome_or_strand"
    if abs(b[1] - a[1]) != len(row.ref) - 1:
        return "liftover_allele_span"
    return min(a[1], b[1]), a[2]


def _target(row: SourceVariant, lift: Any, reference: str) -> tuple[Any, ...] | str:
    mapped = _mapped(row, lift)
    if isinstance(mapped, str):
        return mapped
    target, strand = mapped
    ref, alt = row.ref, row.alt
    if strand == "-":
        ref, alt = (_reverse(allele) for allele in (ref, alt))
    if reference[target : target + len(ref)] != ref:
        return "allele_mismatch"
    try:
        identity = orient(row.chrom, target + 1, ref, alt)
    except IdentityError:
        return "allele_identity"
    if not np.isfinite(row.af) or not 0 < row.af < 1:
        return "invalid_frequency"
    return (
        row.chrom,
        target + 1,
        identity.alid,
        identity.a1,
        identity.a2,
        row.af if identity.a1_is_alt else 1 - row.af,
        row.index,
        row.position,
        1 if identity.a1_is_alt else -1,
        strand == "-",
    )


def _reverse(allele: str) -> str:
    return allele.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def lift_variants(
    rows: list[SourceVariant], chain: Path, fasta: Path
) -> tuple[Lifted, dict[str, int]]:
    """Drop every member of an identity collision, as in #89's measured prototype."""
    if not rows or len({row.chrom for row in rows}) != 1:
        raise ValueError("liftover needs one nonempty chromosome")
    lift = optional_module("pyliftover").LiftOver(str(chain))
    reference = _reference(fasta, rows[0].chrom)
    kept: list[tuple[Any, ...]] = []
    losses: Counter[str] = Counter()
    for row in rows:
        got = _target(row, lift, reference)
        if isinstance(got, str):
            losses[got] += 1
        else:
            kept.append(got)
    multiplicity = Counter(row[2] for row in kept)
    unique = [row for row in kept if multiplicity[row[2]] == 1]
    losses["liftover_identity_collision"] = len(kept) - len(unique)
    unique.sort(key=lambda row: order_key(row[0], row[1], row[2]))
    columns = list(zip(*unique, strict=True)) if unique else [() for _ in range(10)]
    arrays = [np.array(column) for column in columns]
    variants = Variants(
        arrays[0].astype(str),
        arrays[1].astype(np.int64),
        arrays[2].astype(str),
        arrays[3].astype(str),
        arrays[4].astype(str),
        arrays[5].astype(np.float64),
    )
    return Lifted(
        variants,
        arrays[6].astype(np.int64),
        arrays[7].astype(np.int64),
        arrays[8].astype(np.int64),
        arrays[9].astype(bool),
    ), dict(losses)


def _complete(source: IntArray, target: IntArray, distance: int) -> bool:
    low: deque[int] = deque()
    high: deque[int] = deque()
    start = 0
    for i, position in enumerate(target):
        while start < i and target[start] < position - distance:
            start += 1
        for queue, minimum in ((low, True), (high, False)):
            while queue and queue[0] < start:
                queue.popleft()
            while queue and ((source[queue[-1]] >= source[i]) == minimum):
                queue.pop()
            queue.append(i)
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
